"""
Evaluate prompt injection classifiers on the social engineering dataset.

Dataset: ../paired_emails_dataset.json
  JSON array of scenario pairs, each with:
    pair.attack.external_content  → malicious input (social-engineering attack embedded in email)
    pair.benign.external_content  → benign input    (same email context, no injection)

Classifiers evaluated:
  - protectai/deberta-v3-base-prompt-injection-v2  (labels: INJECTION / SAFE)
  - meta-llama/Prompt-Guard-86M                    (labels: JAILBREAK / INJECTION / BENIGN)
  - meta-llama/Llama-Prompt-Guard-2-86M            (labels: JAILBREAK / INJECTION / BENIGN)
"""

# ── HuggingFace soft-lock fix for /fast (no hard file-locking) ──────────────
from filelock import SoftFileLock
import filelock
filelock.FileLock = SoftFileLock

import os
import sys
import json
import argparse
import logging
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ── Model download helper ─────────────────────────────────────────────────────

def _ensure_model_local(model_id: str) -> str:
    """
    Download all model files via direct HTTP requests, bypassing HuggingFace
    Hub / XET transfer which hangs on /fast. Uses the HF API to list files,
    then fetches each one with requests (same method used for datasets).
    Returns the local directory path; subsequent calls reuse the cache.
    """
    import requests

    hf_home = os.environ.get("HF_HOME", "${HF_HOME:-$HOME/.cache/huggingface}")
    local_dir = os.path.join(hf_home, "models", model_id.replace("/", "--"))

    # Resolve token (needed for gated models like meta-llama)
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token is None:
        token_file = "$HOME/.hf_token"
        if os.path.exists(token_file):
            token = open(token_file).read().strip() or None
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    # Already downloaded — check for weights file
    if os.path.isdir(local_dir) and any(
        f.endswith((".safetensors", ".bin"))
        for f in os.listdir(local_dir)
    ):
        log.info(f"  Model already cached at {local_dir}")
        return local_dir

    os.makedirs(local_dir, exist_ok=True)

    # Fetch file list from HF API
    api_url = f"https://huggingface.co/api/models/{model_id}"
    log.info(f"  Fetching file list for {model_id} ...")
    resp = requests.get(api_url, headers=headers, timeout=30)
    resp.raise_for_status()
    siblings = resp.json().get("siblings", [])
    filenames = [s["rfilename"] for s in siblings]
    log.info(f"  Files to download: {filenames}")

    base_url = f"https://huggingface.co/{model_id}/resolve/main"
    for fname in filenames:
        dest = os.path.join(local_dir, fname)
        if os.path.exists(dest):
            continue
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        url = f"{base_url}/{fname}"
        log.info(f"  Downloading {fname} ...")
        r = requests.get(url, headers=headers, stream=True, timeout=600)
        if r.status_code == 403:
            raise PermissionError(
                f"HTTP 403 for {model_id}/{fname}. "
                "The model may be gated — accept the license at huggingface.co "
                "and ensure HF_TOKEN is set with access."
            )
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
        log.info(f"    → {os.path.getsize(dest) / 1e6:.1f} MB")

    log.info(f"  Model ready at {local_dir}")
    return local_dir


# ── Classifier wrappers ───────────────────────────────────────────────────────

class ProtectAIClassifier:
    """
    protectai/deberta-v3-base-prompt-injection-v2
    Labels: INJECTION (positive) / SAFE (negative)
    """
    MODEL_ID = "protectai/deberta-v3-base-prompt-injection-v2"
    INJECTION_LABEL = "INJECTION"

    def __init__(self, device):
        log.info(f"Loading {self.MODEL_ID} ...")
        local_dir = _ensure_model_local(self.MODEL_ID)
        self.tokenizer = AutoTokenizer.from_pretrained(local_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(local_dir)
        self.model.to(device).eval()
        self.device = device
        self.id2label = self.model.config.id2label
        log.info(f"  id2label: {self.id2label}")

    @torch.no_grad()
    def predict_batch(self, texts, batch_size=32):
        """Return list of (predicted_label, injection_score) for each text."""
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)
            logits = self.model(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = logits.argmax(-1).cpu().numpy()
            for pred_id, prob_row in zip(preds, probs):
                label = self.id2label[pred_id]
                inj_id = next(
                    k for k, v in self.id2label.items() if v == self.INJECTION_LABEL
                )
                results.append((label, float(prob_row[inj_id])))
        return results


class PromptGuardClassifier:
    """
    meta-llama/Prompt-Guard-86M  (v1)
    Labels: JAILBREAK / INJECTION / BENIGN
    Positive = INJECTION or JAILBREAK
    """
    MODEL_ID = "meta-llama/Prompt-Guard-86M"
    POSITIVE_LABELS = {"INJECTION", "JAILBREAK"}

    def __init__(self, device):
        log.info(f"Loading {self.MODEL_ID} ...")
        local_dir = _ensure_model_local(self.MODEL_ID)
        self.tokenizer = AutoTokenizer.from_pretrained(local_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(local_dir)
        self.model.to(device).eval()
        self.device = device
        self.id2label = self.model.config.id2label
        log.info(f"  id2label: {self.id2label}")

    @torch.no_grad()
    def predict_batch(self, texts, batch_size=32):
        """Return list of (predicted_label, injection_score) for each text."""
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)
            logits = self.model(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = logits.argmax(-1).cpu().numpy()
            for pred_id, prob_row in zip(preds, probs):
                label = self.id2label[pred_id]
                inj_score = sum(
                    float(prob_row[k])
                    for k, v in self.id2label.items()
                    if v in self.POSITIVE_LABELS
                )
                results.append((label, inj_score))
        return results


class PromptGuard2Classifier(PromptGuardClassifier):
    """
    meta-llama/Llama-Prompt-Guard-2-86M  (v2)
    Binary classification: BENIGN / MALICIOUS.
    The config.json lacks id2label, so the model reports LABEL_0 / LABEL_1.
    LABEL_1 = MALICIOUS (positive class).
    """
    MODEL_ID = "meta-llama/Llama-Prompt-Guard-2-86M"
    POSITIVE_LABELS = {"MALICIOUS", "LABEL_1"}


# ── Dataset loading ───────────────────────────────────────────────────────────

def load_dataset(dataset_path: str):
    """
    Load the social engineering dataset and extract:
      - benign_texts:    list of external_content strings from pair.benign
      - attack_texts:    list of external_content strings from pair.attack
      - metadata:        list of dicts with id, scenario, category,
                         injection_technique for each pair

    Both lists are parallel (index i corresponds to the same scenario pair).
    """
    log.info(f"Loading dataset from {dataset_path} ...")
    with open(dataset_path) as f:
        data = json.load(f)

    benign_texts = []
    attack_texts = []
    metadata = []

    for item in data:
        pair = item["pair"]
        benign_texts.append(pair["benign"]["external_content"])
        attack_texts.append(pair["attack"]["external_content"])
        metadata.append({
            "id": item["id"],
            "scenario": item["scenario"],
            "category": item["category"],
            "injection_technique": pair["attack"].get("injection_technique", ""),
        })

    log.info(f"  Loaded {len(benign_texts)} benign / {len(attack_texts)} attack pairs.")
    if benign_texts:
        log.info(f"  Benign example: {benign_texts[0][:120]!r}")
        log.info(f"  Attack example: {attack_texts[0][:120]!r}")
    return benign_texts, attack_texts, metadata


# ── Metrics ───────────────────────────────────────────────────────────────────

from sklearn.metrics import roc_auc_score, average_precision_score


def compute_metrics(y_true, scores, threshold=0.5):
    """
    Binary classification metrics given true labels and predicted scores.
    Primary metrics: AUROC and Average Precision (AUPRC) — threshold-free.
    Also reports threshold-dependent stats at threshold=0.5 for reference.
    """
    total = len(y_true)
    P = sum(y_true)
    N = total - P

    if P > 0 and N > 0:
        auroc         = roc_auc_score(y_true, scores)
        avg_precision = average_precision_score(y_true, scores)
    else:
        auroc = avg_precision = float("nan")

    preds = [1 if s >= threshold else 0 for s in scores]
    tp = sum(p == 1 and t == 1 for p, t in zip(preds, y_true))
    tn = sum(p == 0 and t == 0 for p, t in zip(preds, y_true))
    fp = sum(p == 1 and t == 0 for p, t in zip(preds, y_true))
    fn = sum(p == 0 and t == 1 for p, t in zip(preds, y_true))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr_50    = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return dict(
        total=total, n_positive=P, n_negative=N,
        auroc=auroc, avg_precision=avg_precision,
        tp=tp, tn=tn, fp=fp, fn=fn,
        precision=precision, recall=recall, fpr=fpr_50, f1=f1,
    )


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_classifier(clf, benign_texts, attack_texts, metadata, batch_size=32):
    """
    Evaluate a classifier on benign and attack emails from the social engineering dataset.

    Returns a dict with:
      - combined: AUROC/AUPRC on the full set (benign=0, attack=1)
      - benign_fpr: FPR stats on the benign-only subset
      - attack_tpr: TPR (detection rate) on the attack-only subset
      - per_sample: per-example scores with metadata
    """
    results = {}

    log.info("  Scoring benign emails ...")
    benign_preds  = clf.predict_batch(benign_texts, batch_size=batch_size)
    benign_scores = [s for _, s in benign_preds]

    log.info("  Scoring attack emails ...")
    attack_preds  = clf.predict_batch(attack_texts, batch_size=batch_size)
    attack_scores = [s for _, s in attack_preds]

    # ── Combined AUROC / AUPRC (benign=0, attack=1) ──────────────────────────
    combined_labels = [0] * len(benign_texts) + [1] * len(attack_texts)
    combined_scores = benign_scores + attack_scores
    combined = compute_metrics(combined_labels, combined_scores)
    results["combined"] = combined
    log.info(
        f"    Combined AUROC={combined['auroc']:.3f}  AUPRC={combined['avg_precision']:.3f}  "
        f"n_benign={len(benign_texts)}  n_attack={len(attack_texts)}"
    )

    # ── FPR on benign only ────────────────────────────────────────────────────
    fp_metrics = compute_metrics([0] * len(benign_texts), benign_scores)
    results["benign_fpr"] = {
        "fpr": fp_metrics["fpr"],
        "fp": fp_metrics["fp"],
        "total": fp_metrics["total"],
    }
    log.info(f"    FPR@0.5={fp_metrics['fpr']:.3f}  ({fp_metrics['fp']}/{fp_metrics['total']} flagged)")

    # ── TPR on attacks only ───────────────────────────────────────────────────
    atk_metrics = compute_metrics([1] * len(attack_texts), attack_scores)
    results["attack_tpr"] = {
        "tpr": atk_metrics["recall"],
        "tp": atk_metrics["tp"],
        "total": atk_metrics["total"],
    }
    log.info(f"    TPR@0.5={atk_metrics['recall']:.3f}  ({atk_metrics['tp']}/{atk_metrics['total']} detected)")

    # ── Per-sample results ────────────────────────────────────────────────────
    per_sample = []
    for i, meta in enumerate(metadata):
        benign_label, b_score = benign_preds[i]
        attack_label, a_score = attack_preds[i]
        per_sample.append({
            **meta,
            "benign": {
                "score": b_score,
                "predicted_label": benign_label,
                "flagged": b_score >= 0.5,
            },
            "attack": {
                "score": a_score,
                "predicted_label": attack_label,
                "detected": a_score >= 0.5,
            },
        })
    results["per_sample"] = per_sample

    # ── Per injection-technique breakdown ─────────────────────────────────────
    technique_stats = {}
    for entry in per_sample:
        tech = entry["injection_technique"] or "unknown"
        if tech not in technique_stats:
            technique_stats[tech] = {"total": 0, "detected": 0}
        technique_stats[tech]["total"] += 1
        if entry["attack"]["detected"]:
            technique_stats[tech]["detected"] += 1
    for tech, stats in technique_stats.items():
        stats["tpr"] = stats["detected"] / stats["total"] if stats["total"] > 0 else 0.0
    results["per_technique"] = technique_stats
    per_tech_summary = {t: f"{v['tpr']:.2f}" for t, v in technique_stats.items()}
    log.info(f"    Per-technique TPR: {per_tech_summary}")

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--classifiers",
        nargs="+",
        default=["protectai", "promptguard", "promptguard2"],
        choices=["protectai", "promptguard", "promptguard2"],
        help="Which classifiers to evaluate",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Path to dataset.json (defaults to ../paired_emails_dataset.json relative to this script)",
    )
    parser.add_argument(
        "--output", default="results.json", help="Path to save JSON results"
    )
    parser.add_argument(
        "--hf-cache",
        default=os.environ.get("HF_HOME", "${HF_HOME:-$HOME/.cache/huggingface}"),
        help="HuggingFace cache directory",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-items", type=int, default=None,
                        help="Evaluate only the first N emails (default: all)")
    args = parser.parse_args()

    os.environ["HF_HOME"] = args.hf_cache
    os.environ["TRANSFORMERS_CACHE"] = os.path.join(args.hf_cache, "hub")
    os.environ["HF_DATASETS_CACHE"] = os.path.join(args.hf_cache, "datasets")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Using device: {device}")

    # Resolve dataset path
    dataset_path = args.dataset
    if dataset_path is None:
        dataset_path = os.path.join(os.path.dirname(__file__), "..", "paired_emails_dataset.json")
    dataset_path = os.path.abspath(dataset_path)

    benign_texts, attack_texts, metadata = load_dataset(dataset_path)
    if args.max_items:
        benign_texts = benign_texts[:args.max_items]
        attack_texts = attack_texts[:args.max_items]
        metadata     = metadata[:args.max_items]
        log.info(f"Limiting to first {args.max_items} pairs")

    classifier_map = {
        "protectai": ProtectAIClassifier,
        "promptguard": PromptGuardClassifier,
        "promptguard2": PromptGuard2Classifier,
    }

    all_results = {}
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for clf_name in args.classifiers:
        log.info(f"\n{'='*60}")
        log.info(f"Evaluating classifier: {clf_name}")
        log.info(f"{'='*60}")
        try:
            clf = classifier_map[clf_name](device=device)
            res = evaluate_classifier(clf, benign_texts, attack_texts, metadata, args.batch_size)
            all_results[clf_name] = res
            del clf
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except PermissionError as e:
            log.error(f"  SKIPPED {clf_name}: {e}")
            all_results[clf_name] = {"error": str(e)}
        except Exception as e:
            log.error(f"  FAILED {clf_name}: {e}", exc_info=True)
            all_results[clf_name] = {"error": str(e)}

        # Save after each classifier so partial results are never lost
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
        log.info(f"  Results checkpoint saved to {out_path}")

    log.info(f"\nFinal results saved to {out_path}")

    # Print summary table
    print("\n" + "=" * 75)
    print(f"{'CLASSIFIER':<20} {'AUROC':>6} {'AUPRC':>6} {'FPR@.5':>7} {'TPR@.5':>7}  N_ben  N_atk")
    print("-" * 75)
    for clf_name, res in all_results.items():
        if "error" in res:
            print(f"{clf_name:<20}  SKIPPED: {res['error'][:40]}")
            continue
        c   = res.get("combined", {})
        fpr = res.get("benign_fpr", {}).get("fpr", float("nan"))
        tpr = res.get("attack_tpr", {}).get("tpr", float("nan"))
        n_ben = res.get("benign_fpr", {}).get("total", "?")
        n_atk = res.get("attack_tpr", {}).get("total", "?")
        print(
            f"{clf_name:<20} {c.get('auroc', float('nan')):>6.3f} "
            f"{c.get('avg_precision', float('nan')):>6.3f} "
            f"{fpr:>7.3f} {tpr:>7.3f}  {n_ben:>5}  {n_atk:>5}"
        )
    print("=" * 75)
    print("AUROC/AUPRC on combined benign+attack set (benign=0, attack=1).")
    print("FPR@.5 / TPR@.5 at threshold 0.5. Attack set = social engineering external_content.")

    # Per-technique breakdown
    print("\nPer-injection-technique TPR@0.5:")
    print(f"  {'TECHNIQUE':<35} {'DETECTED':>9} {'TOTAL':>7} {'TPR':>6}")
    print(f"  {'-'*60}")
    for clf_name, res in all_results.items():
        if "error" in res or "per_technique" not in res:
            continue
        print(f"  [{clf_name}]")
        for tech, stats in res["per_technique"].items():
            print(f"  {'  ' + tech:<35} {stats['detected']:>9} {stats['total']:>7} {stats['tpr']:>6.2f}")


if __name__ == "__main__":
    main()
