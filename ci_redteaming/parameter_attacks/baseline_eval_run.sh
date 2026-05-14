#!/bin/bash
# baseline_eval_run.sh — executed on the HTCondor worker node

set -e

# Activate your virtual environment, e.g.: source .venv/bin/activate

source /etc/profile.d/modules.sh
export SOFT_FILELOCK=1

BACKEND="${BACKEND:-anthropic}"
MODEL="${MODEL:-claude-sonnet-4-6}"
REASONING_EFFORT="${REASONING_EFFORT:-}"
DELAY="${DELAY:-0.5}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/results"
mkdir -p "${RESULTS_DIR}"

# if [ -f /path/to/anthropic_token ]; then
#     export ANTHROPIC_API_KEY=$(cat /path/to/anthropic_token)
# fi
# if [ -f /path/to/azure_openai_token ]; then
#     export AZURE_OPENAI_API_KEY=$(cat /path/to/azure_openai_token)
# fi
# if [ -f /path/to/openai_token ]; then
#     export OPENAI_API_KEY=$(cat /path/to/openai_token)
# fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT="${RESULTS_DIR}/baseline_${BACKEND}_${MODEL//\//-}_${TIMESTAMP}.json"

echo "============================================================"
echo "Baseline Evaluation — send_email rate (iter-0 emails)"
echo "============================================================"
echo "Model:      ${BACKEND}/${MODEL}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Output:     ${OUTPUT}"
echo ""

EXTRA_ARGS=()
if [ -n "${REASONING_EFFORT}" ]; then
    EXTRA_ARGS+=(--reasoning-effort "${REASONING_EFFORT}")
fi

python "${SCRIPT_DIR}/baseline_eval.py" \
    --checkpoint "${CHECKPOINT}" \
    --dataset    "${SCRIPT_DIR}/../../defense_evaluation/paired_emails_dataset.json" \
    --output     "${OUTPUT}" \
    --backend    "${BACKEND}" \
    --model      "${MODEL}" \
    --delay      "${DELAY}" \
    "${EXTRA_ARGS[@]}"

echo ""
echo "Done. Results written to ${OUTPUT}"
