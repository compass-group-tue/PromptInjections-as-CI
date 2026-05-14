"""
Dataset generator for norm inference experiments.
Uses Azure OpenAI endpoints via the openai Python SDK.

Usage:
    python generate_dataset.py --num-items 64 --output dataset.json

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

VALID_DOMAINS = {"email", "code", "project_management", "finance"}
VALID_K = {5}
VALID_ESCALATIONS = {"adjacent", "cross_category", "privilege_escalation"}
VALID_GROUND_TRUTHS = {"execute", "ask_user", "ambiguous"}


def load_prompt(prompt_path: str = "prompt.txt") -> str:
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
    """Extract JSON array from model response, handling markdown fences and wrapping.

    Also handles the case where the model outputs multiple concatenated JSON
    objects/arrays instead of a single array (triggers 'Extra data' on json.loads).
    """
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    text = text.strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as first_err:
        # Model may have output several objects/arrays back-to-back instead of
        # wrapping them in a single array. Parse sequentially with raw_decode.
        items: list = []
        decoder = json.JSONDecoder()
        remaining = text
        try:
            while remaining.strip():
                remaining = remaining.lstrip()
                obj, end = decoder.raw_decode(remaining)
                remaining = remaining[end:]
                if isinstance(obj, list):
                    items.extend(obj)
                elif isinstance(obj, dict):
                    items.append(obj)
            if items:
                return items
        except json.JSONDecodeError:
            pass
        raise first_err  # re-raise the original error if sequential parse also fails

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
    batch_num: int,
    existing_domains: list[str] | None = None,
    existing_escalations: dict[str, int] | None = None,
    existing_k_values: dict[int, int] | None = None,
    temperature: float = 0.9,
    max_retries: int = 3,
) -> list:
    """Generate a batch of dataset items."""
    prompt = prompt_template.replace("{N}", str(num_items))

    # Add context from previous batches to encourage diversity
    extra_instructions = []

    if existing_domains:
        recent = existing_domains[-6:]
        extra_instructions.append(
            f"Recent domains used (vary from these): {', '.join(recent)}"
        )

    if existing_escalations:
        saturated = [e for e, c in existing_escalations.items() if c >= num_items // 2]
        if saturated:
            extra_instructions.append(
                f"These escalation types are over-represented (avoid): {', '.join(saturated)}"
            )

    if existing_k_values:
        saturated_k = [str(k) for k, c in existing_k_values.items() if c >= num_items // 2]
        if saturated_k:
            extra_instructions.append(
                f"k values already over-represented (avoid): {', '.join(saturated_k)}"
            )

    if batch_num > 1:
        extra_instructions.append(
            "Generate entirely new workplace scenarios — different companies, people, and projects than before."
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
                            "You are a dataset generation assistant for AI safety research. "
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

    required_top = {
        "id", "domain", "delegation", "k", "escalation",
        "system_prompt", "setup_message", "delegation_turn",
        "boundary_turn", "history", "test_message",
        "ground_truth", "correct_reasoning", "why_agent_might_overgeneralize",
    }
    missing_top = required_top - set(item.keys())
    if missing_top:
        issues.append(f"Missing top-level keys: {missing_top}")
        return issues

    # Domain
    if item["domain"] not in VALID_DOMAINS:
        issues.append(f"Invalid domain: {item['domain']!r}")

    # k
    if item["k"] not in VALID_K:
        issues.append(f"Invalid k value: {item['k']!r} (expected one of {VALID_K})")

    # Escalation
    if item["escalation"] not in VALID_ESCALATIONS:
        issues.append(f"Invalid escalation: {item['escalation']!r}")

    # Ground truth
    if item["ground_truth"] not in VALID_GROUND_TRUTHS:
        issues.append(f"Invalid ground_truth: {item['ground_truth']!r}")

    # No example should be a control condition — all must probe overgeneralization
    if item["ground_truth"] == "execute":
        issues.append("ground_truth='execute' is not allowed — every example must probe overgeneralization (ask_user or ambiguous)")

    # Delegation fields
    delegation = item.get("delegation", {})
    for key in ("sender", "scope", "autonomy_statement", "boundary_statement"):
        if key not in delegation:
            issues.append(f"delegation missing key: {key!r}")

    # delegation_turn fields
    dt = item.get("delegation_turn", {})
    for key in ("trigger_message", "agent_asks_user", "user_grants_autonomy"):
        if key not in dt:
            issues.append(f"delegation_turn missing key: {key!r}")

    # boundary_turn non-empty
    if not item.get("boundary_turn", "").strip():
        issues.append("boundary_turn is empty")

    # history length matches k
    history = item.get("history", [])
    if not isinstance(history, list):
        issues.append("history is not a list")
    else:
        if len(history) != item["k"]:
            issues.append(f"history length {len(history)} != k={item['k']}")
        for i, entry in enumerate(history):
            for key in ("incoming_message", "agent_response"):
                if key not in entry:
                    issues.append(f"history[{i}] missing key: {key!r}")

    # test_message non-empty
    if not item.get("test_message", "").strip():
        issues.append("test_message is empty")

    return issues


DOMAIN_ABBREV = {
    "email":              "email",
    "code":               "code",
    "project_management": "pm",
    "finance":            "fin",
}


def reassign_ids(items: list) -> list:
    """
    Re-assign IDs to all items so they are unique and consistent.

    Format: NI-{domain_abbrev}-{per_domain_counter:03d}
    Items are grouped by domain in the order they appear, so existing items
    keep their relative positions and new items are appended after them.
    """
    counters: dict[str, int] = {}
    for item in items:
        domain = item.get("domain", "unknown")
        abbrev = DOMAIN_ABBREV.get(domain, domain)
        counters[abbrev] = counters.get(abbrev, 0) + 1
        item["id"] = f"NI-{abbrev}-{counters[abbrev]:03d}"
    return items


def validate_dataset(dataset: list) -> dict:
    """Validate the full dataset and return stats."""
    stats = {
        "total": len(dataset),
        "valid": 0,
        "warnings": 0,
        "item_issues": [],
        "domains": {},
        "k_values": {},
        "escalations": {},
        "ground_truths": {},
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

        domain = item.get("domain", "unknown")
        stats["domains"][domain] = stats["domains"].get(domain, 0) + 1

        k = item.get("k", "unknown")
        stats["k_values"][str(k)] = stats["k_values"].get(str(k), 0) + 1

        escalation = item.get("escalation", "unknown")
        stats["escalations"][escalation] = stats["escalations"].get(escalation, 0) + 1

        gt = item.get("ground_truth", "unknown")
        stats["ground_truths"][gt] = stats["ground_truths"].get(gt, 0) + 1

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Generate norm inference dataset"
    )
    parser.add_argument(
        "--num-items", type=int, default=64,
        help="Total number of items to generate (default: 64)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=4,
        help="Items per API call — smaller = more reliable JSON (default: 4)",
    )
    parser.add_argument(
        "--output", type=str, default="dataset.json",
        help="Output file path (default: dataset.json)",
    )
    parser.add_argument(
        "--prompt", type=str, default="prompt.txt",
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
    parser.add_argument(
        "--append-from", type=str, default=None,
        help="Seed generation from this existing dataset file instead of --output. "
             "New items are appended after the existing ones and IDs are re-assigned.",
    )
    args = parser.parse_args()

    # --- Validate-only mode ---
    if args.validate_only:
        print(f"Validating {args.validate_only}...")
        with open(args.validate_only) as f:
            dataset = json.load(f)
        stats = validate_dataset(dataset)
        print(f"\nTotal: {stats['total']}  Valid: {stats['valid']}  Warnings: {stats['warnings']}")
        print(f"Domains:       {json.dumps(stats['domains'])}")
        print(f"k values:      {json.dumps(stats['k_values'])}")
        print(f"Escalations:   {json.dumps(stats['escalations'])}")
        print(f"Ground truths: {json.dumps(stats['ground_truths'])}")
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
    domains_seen = []
    escalations_count = {}
    k_values_count = {}

    output_path = Path(args.output)

    # Determine seed file: --append-from takes priority over --output
    seed_path = Path(args.append_from) if args.append_from else output_path
    if seed_path.exists():
        try:
            existing = json.loads(seed_path.read_text(encoding="utf-8"))
            if isinstance(existing, list) and existing:
                all_items = existing
                for item in all_items:
                    domains_seen.append(item.get("domain", "unknown"))
                    esc = item.get("escalation", "unknown")
                    escalations_count[esc] = escalations_count.get(esc, 0) + 1
                    k = item.get("k", "unknown")
                    k_values_count[k] = k_values_count.get(k, 0) + 1
                source_label = f"--append-from {seed_path}" if args.append_from else str(seed_path)
                print(f"Loaded {len(all_items)} existing items from {source_label}\n")
        except Exception as e:
            print(f"Warning: could not load seed file ({e}), starting fresh\n")

    if len(all_items) >= args.num_items:
        print(f"Already have {len(all_items)} items, nothing to generate.")
        return

    num_batches = (args.num_items - len(all_items) + args.batch_size - 1) // args.batch_size

    for batch_idx in range(num_batches):
        remaining = args.num_items - len(all_items)
        batch_size = min(args.batch_size, remaining)
        batch_num = batch_idx + 1

        print(f"Batch {batch_num}/{num_batches} ({batch_size} items)...")

        try:
            items = generate_batch(
                client=client,
                deployment=deployment,
                prompt_template=prompt_template,
                num_items=batch_size,
                batch_num=batch_num,
                existing_domains=domains_seen,
                existing_escalations=escalations_count,
                existing_k_values=k_values_count,
                temperature=args.temperature,
            )

            all_items.extend(items)

            # Track for diversity enforcement
            for item in items:
                domains_seen.append(item.get("domain", "unknown"))
                esc = item.get("escalation", "unknown")
                escalations_count[esc] = escalations_count.get(esc, 0) + 1
                k = item.get("k", "unknown")
                k_values_count[k] = k_values_count.get(k, 0) + 1

            # Re-assign IDs and save after every batch
            reassign_ids(all_items)
            output_path.write_text(
                json.dumps(all_items, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            print(f"  Saved {len(all_items)} items to {output_path}")

        except Exception as e:
            print(f"  ERROR in batch {batch_num}: {e} — skipping and continuing")

        # Rate limiting
        if batch_idx < num_batches - 1:
            time.sleep(2)

    # --- Validate ---
    print("\n--- Validation ---")
    stats = validate_dataset(all_items)
    print(f"Total: {stats['total']}  Valid: {stats['valid']}  Warnings: {stats['warnings']}")
    print(f"Domains:       {json.dumps(stats['domains'])}")
    print(f"k values:      {json.dumps(stats['k_values'])}")
    print(f"Escalations:   {json.dumps(stats['escalations'])}")
    print(f"Ground truths: {json.dumps(stats['ground_truths'])}")
    if stats["item_issues"]:
        print(f"\nItems with issues ({len(stats['item_issues'])}):")
        for item in stats["item_issues"][:10]:
            print(f"  ID {item['id']}: {item['issues']}")
        if len(stats["item_issues"]) > 10:
            print(f"  ... and {len(stats['item_issues']) - 10} more")


if __name__ == "__main__":
    main()
