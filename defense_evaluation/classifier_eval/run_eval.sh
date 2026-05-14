#!/bin/bash
# run_eval.sh — Evaluates ProtectAI DeBERTa, PromptGuard 1, and PromptGuard 2
# on the paired email dataset.
#
# Usage:
#   bash run_eval.sh
#
# Configurable via environment variables:
#   HF_HOME       HuggingFace cache directory  (default: ~/.cache/huggingface)
#   HF_TOKEN      HuggingFace token            (or set via key file below)
#   MAX_ITEMS     Evaluate first N pairs only  (default: all)
#
# Models required (download with huggingface-cli if not cached):
#   protectai/deberta-v3-base-prompt-injection-v2
#   meta-llama/Prompt-Guard-86M
#   meta-llama/Llama-Prompt-Guard-2-86M

set -e

# ── Activate environment ─────────────────────────────────────────────────────
# source /path/to/your/venv/bin/activate

# ── HuggingFace token (needed for gated models e.g. meta-llama) ──────────────
# Edit path to your token file, or export HF_TOKEN directly:
# if [ -f /path/to/hf_token ]; then
#     export HF_TOKEN=$(cat /path/to/hf_token)
# fi

# ── HuggingFace cache ───────────────────────────────────────────────────────
export HF_HOME=${HF_HOME:-${HOME}/.cache/huggingface}
export TRANSFORMERS_CACHE=${HF_HOME}/hub
export HF_DATASETS_CACHE=${HF_HOME}/datasets
export HUGGINGFACE_HUB_CACHE=${HF_HOME}/hub
export HF_HUB_DISABLE_LOCKING=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/results"
mkdir -p "${RESULTS_DIR}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULT_FILE="${RESULTS_DIR}/eval_${TIMESTAMP}.json"

echo "============================================================"
echo "Social Engineering Dataset — Classifier Evaluation"
echo "============================================================"
echo "Python:  $(which python)"
echo "Results: ${RESULT_FILE}"
echo "GPU:     $(nvidia-smi -L 2>/dev/null || echo 'none')"
echo ""

python "${SCRIPT_DIR}/eval_classifier.py" \
    --classifiers protectai promptguard promptguard2 \
    --dataset "${SCRIPT_DIR}/../paired_emails_dataset.json" \
    --output "${RESULT_FILE}" \
    --hf-cache "${HF_HOME}" \
    --batch-size 32 \
    ${MAX_ITEMS:+--max-items ${MAX_ITEMS}}

echo ""
echo "Done. Results written to ${RESULT_FILE}"
