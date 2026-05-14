#!/bin/bash
# run_eval.sh — executed on the HTCondor worker node
#
# Configurable via environment variables (set in eval_agent.sub or exported):
#   BACKEND           anthropic | openai | azure | vllm  (default: azure)
#   MODEL             model / deployment name             (default: gpt-4o)
#   AZURE_ENDPOINT    Azure OpenAI endpoint URL           (required for azure backend)
#   AZURE_API_VERSION Azure OpenAI API version            (default: 2024-02-01)
#   DELAY             seconds between API calls           (default: 0.5)
#   MAX_ITEMS         evaluate only first N scenarios     (default: all)

set -e

# ── Activate environment ──────────────────────────────────────────────────────
# source /path/to/your/venv/bin/activate

# ── Cluster-specific env ──────────────────────────────────────────────────────
source /etc/profile.d/modules.sh
export SOFT_FILELOCK=1

# ── Defaults ──────────────────────────────────────────────────────────────────
BACKEND="${BACKEND:-azure}"
MODEL="${MODEL:-gpt-5-chat}"
AZURE_API_VERSION="${AZURE_API_VERSION:-2025-01-01-preview}"
DELAY="${DELAY:-0.5}"
MAX_ITEMS="${MAX_ITEMS:-}"

# ── API keys (file → env, model_client.py will also check the files directly) ────────────────
# Edit paths to your token files, or export API keys directly.
# if [ -f /path/to/anthropic_token ]; then
#     export ANTHROPIC_API_KEY=$(cat /path/to/anthropic_token)
# fi
# if [ -f /path/to/azure_openai_token ]; then
#     export AZURE_OPENAI_API_KEY=$(cat /path/to/azure_openai_token)
# fi
# if [ -f /path/to/openai_token ]; then
#     export OPENAI_API_KEY=$(cat /path/to/openai_token)
# fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/results"
mkdir -p "${RESULTS_DIR}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULT_FILE="${RESULTS_DIR}/eval_${BACKEND}_${MODEL//\//-}_${TIMESTAMP}.json"

echo "============================================================"
echo "Social Engineering Dataset — Agentic Email Assistant Eval"
echo "============================================================"
echo "Backend:  ${BACKEND}"
echo "Model:    ${MODEL}"
echo "Python:   $(which python)"
echo "Results:  ${RESULT_FILE}"
echo ""

# ── Build backend-specific flags ──────────────────────────────────────────────
EXTRA_ARGS=""
if [ "${BACKEND}" = "azure" ]; then
    if [ -z "${AZURE_ENDPOINT}" ]; then
        echo "ERROR: AZURE_ENDPOINT is not set." >&2
        exit 1
    fi
    EXTRA_ARGS="--azure-endpoint ${AZURE_ENDPOINT} --azure-api-version ${AZURE_API_VERSION}"
fi
if [ "${BACKEND}" = "vllm" ] && [ -n "${VLLM_BASE_URL}" ]; then
    EXTRA_ARGS="--vllm-base-url ${VLLM_BASE_URL}"
fi

python "${SCRIPT_DIR}/eval_agent.py" \
    --backend   "${BACKEND}" \
    --model     "${MODEL}" \
    --dataset   "${SCRIPT_DIR}/../paired_emails_dataset.json" \
    --output    "${RESULT_FILE}" \
    --delay     "${DELAY}" \
    ${MAX_ITEMS:+--max-items ${MAX_ITEMS}} \
    ${EXTRA_ARGS}

echo ""
echo "Done. Results written to ${RESULT_FILE}"
