#!/bin/bash
# transferability_eval_run.sh — executed on the HTCondor worker node
#
# Tests attack transferability: baseline (original emails) vs PAIR-optimized emails
# on a new target model.
#
# Env vars:
#   BACKEND            anthropic | openai | azure | vllm | gemini  (default: gemini)
#   MODEL              model / deployment name             (default: claude-sonnet-4-6)
#   REASONING_EFFORT   low|medium|high (OpenAI o-series only, optional)
#   CHECKPOINT         path to PAIR checkpoint.json
#   BEST_ATTACKS       path to best_attacks dataset JSON
#   DELAY              seconds between API calls           (default: 0.5)

set -e

# Activate your virtual environment, e.g.: source .venv/bin/activate

source /etc/profile.d/modules.sh
export SOFT_FILELOCK=1

BACKEND="${BACKEND:-gemini}"
MODEL="${MODEL:-gemini-3-pro-preview}"
REASONING_EFFORT="${REASONING_EFFORT:-}"
DELAY="${DELAY:-0.5}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
ENABLE_THINKING="${ENABLE_THINKING:-0}"
THINKING_BUDGET="${THINKING_BUDGET:-1000}"
MAX_TOKENS="${MAX_TOKENS:-3000}"

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
# if [ -f /path/to/gemini_key ]; then
#     export GEMINI_API_KEY=$(cat /path/to/gemini_key)
# fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SAMPLE_SUFFIX=""
if [ -n "${MAX_SAMPLES}" ]; then
    SAMPLE_SUFFIX="_sample${MAX_SAMPLES}"
fi
OUTPUT="${RESULTS_DIR}/transfer_${BACKEND}_${MODEL//\//-}_${TIMESTAMP}${SAMPLE_SUFFIX}.json"

echo "============================================================"
echo "Transferability Evaluation — PAIR attack transfer rate"
echo "============================================================"
echo "Model:        ${BACKEND}/${MODEL}"
echo "Checkpoint:   ${CHECKPOINT}"
echo "Best attacks: ${BEST_ATTACKS}"
echo "Output:       ${OUTPUT}"
echo ""

EXTRA_ARGS=()
if [ -n "${REASONING_EFFORT}" ]; then
    EXTRA_ARGS+=(--reasoning-effort "${REASONING_EFFORT}")
fi
if [ -n "${MAX_SAMPLES}" ]; then
    EXTRA_ARGS+=(--max-samples "${MAX_SAMPLES}")
fi
if [ "${ENABLE_THINKING}" = "1" ]; then
    EXTRA_ARGS+=(--enable-thinking --thinking-budget "${THINKING_BUDGET}")
fi
EXTRA_ARGS+=(--max-tokens "${MAX_TOKENS}")

python3 "${SCRIPT_DIR}/transferability_eval.py" \
    --checkpoint   "${CHECKPOINT}" \
    --dataset      "${SCRIPT_DIR}/../../defense_evaluation/paired_emails_dataset.json" \
    --best-attacks "${BEST_ATTACKS}" \
    --output       "${OUTPUT}" \
    --backend      "${BACKEND}" \
    --model        "${MODEL}" \
    --delay        "${DELAY}" \
    "${EXTRA_ARGS[@]}"

echo ""
echo "Done. Results written to ${OUTPUT}"
