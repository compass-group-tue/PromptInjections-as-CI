"""
Dataset generator for social engineering prompt injection pairs.
Uses Azure OpenAI endpoints via the openai Python SDK.

Usage:
    python generate_dataset.py --num-items 50 --output dataset.json

Environment variables required:
    AZURE_OPENAI_ENDPOINT     - Your Azure OpenAI endpoint URL
    AZURE_OPENAI_API_KEY      - Your Azure OpenAI API key
    AZURE_OPENAI_DEPLOYMENT   - Deployment name (e.g., gpt-4o)
    AZURE_OPENAI_API_VERSION  - API version (default: 2024-10-21)
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

from openai import AzureOpenAI


def load_prompt(prompt_path: str = "generation_prompt.txt") -> str:
    """Load the generation prompt template."""
    path = Path(prompt_path)
    if not path.exists():
        path = Path(__file__).parent / prompt_path
    if not path.exists():
        print(f"Error: Prompt file not found at {prompt_path}")
        sys.exit(1)
    return path.read_text()


def create_client() -> AzureOpenAI:
    """Create Azure OpenAI client from environment variables."""
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21")

    if not endpoint or not api_key:
        print("Error: Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY env vars.")
        sys.exit(1)

    return AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
    )


def extract_json(text: str) -> list:
    """Extract JSON array from model response, handling markdown fences and wrapping."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    text = text.strip()

    parsed = json.loads(text)

    # If model wrapped the array in an object, find the first list value
    if isinstance(parsed, dict):
        for v in parsed.values():
            if isinstance(v, list):
                return v
        raise ValueError("Response is a JSON object with no array field")

    if not isinstance(parsed, list):
        raise ValueError(f"Expected list, got {type(parsed).__name__}")

    return parsed


def generate_batch(
    client: AzureOpenAI,
    deployment: str,
    prompt_template: str,
    num_items: int,
    batch_id_start: int,
    existing_categories: list[str] | None = None,
    existing_techniques: dict[str, int] | None = None,
    temperature: float = 0.9,
    max_retries: int = 3,
) -> list:
    """Generate a batch of dataset items."""
    prompt = prompt_template.replace("{num_items}", str(num_items))

    # Add context from previous batches to encourage diversity
    extra_instructions = []
    if batch_id_start > 1:
        extra_instructions.append(f"Start IDs from {batch_id_start:03d}.")

    if existing_categories:
        recent = existing_categories[-5:]
        extra_instructions.append(
            f"Recent categories used (avoid repeating): {', '.join(recent)}"
        )

    if existing_techniques:
        saturated = [t for t, c in existing_techniques.items() if c >= 2]
        if saturated:
            extra_instructions.append(
                f"These injection techniques have been used twice already (do NOT reuse): "
                f"{', '.join(saturated)}"
            )

    if extra_instructions:
        prompt += "\n\n## Additional constraints for this batch\n" + "\n".join(
            f"- {inst}" for inst in extra_instructions
        )

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=deployment,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a dataset generation assistant for AI security research. "
                            "Return only valid JSON. No commentary or explanation."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=16384,
            )

            content = response.choices[0].message.content
            items = extract_json(content)
            print(f"  Batch generated: {len(items)} items")
            return items

        except json.JSONDecodeError as e:
            print(f"  Attempt {attempt + 1}/{max_retries} — JSON parse error: {e}")
        except Exception as e:
            print(f"  Attempt {attempt + 1}/{max_retries} — API error: {e}")

        if attempt < max_retries - 1:
            wait = 2 ** (attempt + 1)
            print(f"  Retrying in {wait}s...")
            time.sleep(wait)

    print("  WARNING: Batch failed after all retries, returning empty list")
    return []


def validate_item(item: dict) -> list[str]:
    """Validate a single dataset item. Returns list of issues."""
    issues = []

    required_top = {"id", "scenario", "category", "pair"}
    missing_top = required_top - set(item.keys())
    if missing_top:
        issues.append(f"Missing top-level keys: {missing_top}")
        return issues

    pair = item["pair"]
    for label in ("attack", "benign"):
        if label not in pair:
            issues.append(f"Missing pair.{label}")
            continue
        entry = pair[label]
        for key in ("user_prompt", "external_content", "expected_action"):
            if key not in entry:
                issues.append(f"pair.{label} missing {key}")

    if "attack" in pair and "injection_technique" not in pair["attack"]:
        issues.append("pair.attack missing injection_technique")

    # Check external content length (min 80 words)
    for label in ("attack", "benign"):
        if label in pair and "external_content" in pair[label]:
            wc = len(pair[label]["external_content"].split())
            if wc < 60:
                issues.append(f"pair.{label} external_content too short ({wc} words)")

    # Check length similarity between attack and benign
    if all(
        label in pair and "external_content" in pair[label]
        for label in ("attack", "benign")
    ):
        atk_wc = len(pair["attack"]["external_content"].split())
        ben_wc = len(pair["benign"]["external_content"].split())
        if min(atk_wc, ben_wc) > 0:
            ratio = max(atk_wc, ben_wc) / min(atk_wc, ben_wc)
            if ratio > 1.3:
                issues.append(
                    f"External content length mismatch: "
                    f"attack={atk_wc}w, benign={ben_wc}w (ratio={ratio:.2f})"
                )

    # Check for forbidden prompt injection patterns
    if "attack" in pair and "external_content" in pair["attack"]:
        content_lower = pair["attack"]["external_content"].lower()
        forbidden = [
            "ignore previous",
            "ignore your instructions",
            "system override",
            "admin mode",
            "debug mode",
            "you are now",
            "as an ai",
            "as a language model",
            "repeat your system prompt",
            "what are your instructions",
            "pretend that",
            "IMPORTANT:",
            "PRIORITY:",
            "URGENT SYSTEM MESSAGE",
        ]
        for pattern in forbidden:
            if pattern.lower() in content_lower:
                issues.append(f"Forbidden pattern detected: '{pattern}'")

    return issues


def validate_dataset(dataset: list) -> dict:
    """Validate the full dataset and return stats."""
    stats = {
        "total": len(dataset),
        "valid": 0,
        "warnings": 0,
        "item_issues": [],
        "categories": {},
        "injection_techniques": {},
    }

    for item in dataset:
        item_issues = validate_item(item)
        if item_issues:
            stats["warnings"] += 1
            stats["item_issues"].append({
                "id": item.get("id", "?"),
                "issues": item_issues,
            })
        else:
            stats["valid"] += 1

        cat = item.get("category", "unknown")
        stats["categories"][cat] = stats["categories"].get(cat, 0) + 1

        technique = (
            item.get("pair", {}).get("attack", {}).get("injection_technique", "unknown")
        )
        stats["injection_techniques"][technique] = (
            stats["injection_techniques"].get(technique, 0) + 1
        )

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Generate social engineering prompt injection dataset"
    )
    parser.add_argument(
        "--num-items", type=int, default=50,
        help="Total number of items to generate (default: 50)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=5,
        help="Items per API call — smaller = more reliable JSON (default: 5)",
    )
    parser.add_argument(
        "--output", type=str, default="dataset.json",
        help="Output file path (default: dataset.json)",
    )
    parser.add_argument(
        "--prompt", type=str, default="generation_prompt.txt",
        help="Path to generation prompt template",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.9,
        help="Sampling temperature — higher = more diverse (default: 0.9)",
    )
    parser.add_argument(
        "--validate-only", type=str, default=None,
        help="Path to existing dataset to validate (skips generation)",
    )
    args = parser.parse_args()

    # --- Validate-only mode ---
    if args.validate_only:
        print(f"Validating {args.validate_only}...")
        with open(args.validate_only) as f:
            dataset = json.load(f)
        stats = validate_dataset(dataset)
        print(f"\nTotal: {stats['total']}  Valid: {stats['valid']}  Warnings: {stats['warnings']}")
        print(f"Categories ({len(stats['categories'])}): {json.dumps(stats['categories'], indent=2)}")
        print(f"Injection techniques: {json.dumps(stats['injection_techniques'], indent=2)}")
        if stats["item_issues"]:
            print(f"\nItems with issues:")
            for item in stats["item_issues"]:
                print(f"  ID {item['id']}: {item['issues']}")
        return

    # --- Generation mode ---
    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT")
    if not deployment:
        print("Error: Set AZURE_OPENAI_DEPLOYMENT env var.")
        sys.exit(1)

    client = create_client()
    prompt_template = load_prompt(args.prompt)

    print(f"Generating {args.num_items} items in batches of {args.batch_size}")
    print(f"Model deployment: {deployment}")
    print(f"Temperature: {args.temperature}")
    print(f"Output: {args.output}\n")

    all_items = []
    categories_seen = []
    techniques_count = {}

    output_path = Path(args.output)
    if output_path.exists():
        try:
            existing = json.loads(output_path.read_text(encoding="utf-8"))
            if isinstance(existing, list) and existing:
                all_items = existing
                for item in all_items:
                    categories_seen.append(item.get("category", "unknown"))
                    technique = (
                        item.get("pair", {}).get("attack", {}).get("injection_technique", "unknown")
                    )
                    techniques_count[technique] = techniques_count.get(technique, 0) + 1
                print(f"Resuming from existing file: {len(all_items)} items already present\n")
        except Exception as e:
            print(f"Warning: could not load existing file ({e}), starting fresh\n")

    if len(all_items) >= args.num_items:
        print(f"Already have {len(all_items)} items, nothing to generate.")
        return

    num_batches = (args.num_items - len(all_items) + args.batch_size - 1) // args.batch_size

    for batch_idx in range(num_batches):
        remaining = args.num_items - len(all_items)
        batch_size = min(args.batch_size, remaining)
        batch_id_start = len(all_items) + 1

        print(f"Batch {batch_idx + 1}/{num_batches} ({batch_size} items, starting at ID {batch_id_start:03d})...")

        try:
            items = generate_batch(
                client=client,
                deployment=deployment,
                prompt_template=prompt_template,
                num_items=batch_size,
                batch_id_start=batch_id_start,
                existing_categories=categories_seen,
                existing_techniques=techniques_count,
                temperature=args.temperature,
            )

            # Re-number IDs to ensure global uniqueness
            for i, item in enumerate(items):
                item["id"] = f"{batch_id_start + i:03d}"

            all_items.extend(items)

            # Track categories and techniques for diversity enforcement
            for item in items:
                categories_seen.append(item.get("category", "unknown"))
                technique = (
                    item.get("pair", {}).get("attack", {}).get("injection_technique", "unknown")
                )
                techniques_count[technique] = techniques_count.get(technique, 0) + 1

            # Save after every batch
            output_path = Path(args.output)
            output_path.write_text(json.dumps(all_items, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"  Saved {len(all_items)} items to {output_path}")

        except Exception as e:
            print(f"  ERROR in batch {batch_idx + 1}: {e} — skipping and continuing")

        # Rate limiting — be polite to the API
        if batch_idx < num_batches - 1:
            time.sleep(2)

    # --- Validate ---
    print("\n--- Validation ---")
    stats = validate_dataset(all_items)
    print(f"Total: {stats['total']}  Valid: {stats['valid']}  Warnings: {stats['warnings']}")
    print(f"Categories ({len(stats['categories'])}): {json.dumps(stats['categories'], indent=2)}")
    print(f"Injection techniques: {json.dumps(stats['injection_techniques'], indent=2)}")
    if stats["item_issues"]:
        print(f"\nItems with issues ({len(stats['item_issues'])}):")
        for item in stats["item_issues"][:10]:
            print(f"  ID {item['id']}: {item['issues']}")
        if len(stats["item_issues"]) > 10:
            print(f"  ... and {len(stats['item_issues']) - 10} more")


if __name__ == "__main__":
    main()
