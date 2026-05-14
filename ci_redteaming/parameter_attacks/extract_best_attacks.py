"""
Extract the best attack email per scenario from a PAIR checkpoint.

"Best" is defined as:
  - attack_succeeded=True  → the first stream where success=True, last iter of that stream
    (which is the iter where send_email_called=True, since the stream stops there)
  - attack_succeeded=False → last iteration of stream 0 (most refined attempt)

Output dataset is a flat JSON list suitable for use in transferability_eval.py.

Usage:
    python extract_best_attacks.py \
        --checkpoint results/pair_.../checkpoint.json \
        --output     results/best_attacks_<name>.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


def extract_best(scenario: dict) -> dict:
    streams = scenario["streams"]

    if scenario["attack_succeeded"]:
        # Find first stream (by stream_id order) that succeeded
        winning_stream = next(s for s in sorted(streams, key=lambda s: s["stream_id"]) if s["success"])
        best_iter = winning_stream["history"][-1]
        return {
            "email": best_iter["email_content"],
            "stream_id": winning_stream["stream_id"],
            "iteration": best_iter["iteration"],
            "technique": best_iter.get("technique", "unknown"),
        }
    else:
        # No stream succeeded — take last iter of stream 0
        stream0 = next(s for s in streams if s["stream_id"] == 0)
        best_iter = stream0["history"][-1]
        return {
            "email": best_iter["email_content"],
            "stream_id": 0,
            "iteration": best_iter["iteration"],
            "technique": best_iter.get("technique", "unknown"),
        }


def main():
    parser = argparse.ArgumentParser(description="Extract best-attack emails from a PAIR checkpoint")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint.json from a PAIR run")
    parser.add_argument("--output",     required=True, help="Path to write best-attacks dataset JSON")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    checkpoint = json.loads(checkpoint_path.read_text())
    log.info(f"Loaded {len(checkpoint)} scenarios from {checkpoint_path}")

    records = []
    n_succeeded = 0

    for scenario in checkpoint:
        best = extract_best(scenario)
        records.append({
            "id":                scenario["id"],
            "scenario":          scenario["scenario"],
            "category":          scenario.get("category"),
            "user_prompt":       scenario["user_prompt"],
            "best_attack_email": best["email"],
            "pair_succeeded":    scenario["attack_succeeded"],
            "best_stream":       best["stream_id"],
            "best_iteration":    best["iteration"],
            "technique":         best["technique"],
        })
        if scenario["attack_succeeded"]:
            n_succeeded += 1

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(records, indent=2))

    log.info(f"Scenarios: {len(records)} total, {n_succeeded} with PAIR success")
    log.info(f"Best-attacks dataset written to {out_path}")


if __name__ == "__main__":
    main()
