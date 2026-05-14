"""
Compute overgeneralization rates from a checkpoint file.

Usage:
    python compute_rates.py <checkpoint.json> [--output rates.json]

The checkpoint file is the per-sample list produced by eval_agent.py.
Results are saved as a focused rates JSON (no per_sample blob).
"""

import sys
import json
import argparse
from pathlib import Path

# Reuse compute_summary from eval_agent in the same directory
sys.path.insert(0, str(Path(__file__).parent))
from eval_agent import compute_summary, CONDITIONS


def compute_rates(checkpoint_path: Path) -> dict:
    data = json.loads(checkpoint_path.read_text())
    return compute_summary(data)


def print_rates(summary: dict) -> None:
    W = 74
    print("\n" + "=" * W)
    print("OVERGENERALIZATION RATES")
    print(f"Items: {summary['n_items']}")
    print("-" * W)
    for ck, clabel in CONDITIONS:
        cs = summary["conditions"][ck]["overgeneralization"]
        print(f"  {clabel:<48}  {cs['count']:>3}/{cs['total']:<3}  {cs['rate']:>5.1%}")
    print("-" * W)

    print("\nBy domain:")
    for domain, conds in summary["by_domain"].items():
        row = "  |  ".join(
            f"{ck}: {conds[ck]['count']}/{conds[ck]['total']} ({conds[ck]['rate']:.1%})"
            for ck, _ in CONDITIONS
        )
        print(f"  {domain:<22}  {row}")

    print("\nBy escalation type:")
    for esc, conds in summary["by_escalation"].items():
        row = "  |  ".join(
            f"{ck}: {conds[ck]['count']}/{conds[ck]['total']} ({conds[ck]['rate']:.1%})"
            for ck, _ in CONDITIONS
        )
        print(f"  {esc:<22}  {row}")

    print("\nBy actual k:")
    print(f"  {'k':<8}  {'n':>6}  {'zero_shot':>12}  {'full_history':>14}")
    for k_str, conds in summary["by_k_actual"].items():
        n_k = conds["zero_shot"]["total"]
        zs  = conds["zero_shot"]["rate"]
        fh  = conds["full_history"]["rate"]
        flag = "  ← unexpected" if k_str != "5" else ""
        print(f"  k={k_str:<6}  {n_k:>6}  {zs:>11.1%}  {fh:>13.1%}{flag}")
    print("=" * W)


def main():
    parser = argparse.ArgumentParser(
        description="Compute overgeneralization rates from a checkpoint file"
    )
    parser.add_argument("checkpoint", help="Path to checkpoint.json")
    parser.add_argument("--output", default=None,
                        help="Output path for rates JSON (default: <checkpoint_dir>/rates.json)")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: {checkpoint_path} not found", file=sys.stderr)
        sys.exit(1)

    summary = compute_rates(checkpoint_path)
    print_rates(summary)

    out_path = Path(args.output) if args.output else checkpoint_path.parent / "rates.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nRates saved to {out_path}")


if __name__ == "__main__":
    main()
