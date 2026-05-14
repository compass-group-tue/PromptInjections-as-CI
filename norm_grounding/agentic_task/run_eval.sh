#!/bin/bash
# run_eval.sh — HTCondor worker script for norm inference agentic evaluation
#
# Configurable via environment variables (set in eval_agent.sub or exported):
#   BACKEND           anthropic | openai | azure | vllm  (default: anthropic)
#   MODEL             model / deployment name             (default: claude-sonnet-4-6)
#   AZURE_ENDPOINT    Azure OpenAI endpoint URL           (required for azure)
#   AZURE_API_VERSION Azure OpenAI API version            (default: 2025-01-01-preview)
#   DATASET           path to dataset JSON                (default: auto-detect latest)
#   BOUNDARY_TURN     1 to include boundary_turn          (default: 0 = excluded)
#   DELAY             seconds between API calls           (default: 0.5)
#   MAX_ITEMS         evaluate only first N items         (default: all)

# ── Activate environment ──────────────────────────────────────────────────────
# source /path/to/your/venv/bin/activate

# ── Cluster-specific env ──────────────────────────────────────────────────────
source /etc/profile.d/modules.sh
export SOFT_FILELOCK=1

# ── Defaults ──────────────────────────────────────────────────────────────────
BACKEND="${BACKEND:-openai}"
MODEL="${MODEL:-gpt-5.2}"
AZURE_API_VERSION="${AZURE_API_VERSION:-2025-01-01-preview}"
BOUNDARY_TURN="${BOUNDARY_TURN:-0}"
DELAY="${DELAY:-0.5}"
MAX_ITEMS="${MAX_ITEMS:-}"

# ── API keys (file → env) ────────────────────────────────────────────────────────────────────────────────
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
# if [ -f /path/to/gemini_key ]; then
#     export GEMINI_API_KEY=$(cat /path/to/gemini_key)
# fi

# ── Validate required vars for azure backend ──────────────────────────────────
if [ "${BACKEND}" = "azure" ] && [ -z "${AZURE_ENDPOINT}" ]; then
    echo "ERROR: AZURE_ENDPOINT is not set." >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/results"
mkdir -p "${RESULTS_DIR}"

BOUNDARY_TAG="no_boundary"
BOUNDARY_FLAG=""
if [ "${BOUNDARY_TURN}" = "1" ]; then
    BOUNDARY_TAG="with_boundary"
    BOUNDARY_FLAG="--boundary-turn"
fi

# Stable filename (no timestamp) so resubmitted jobs resume from the same checkpoint.
RESULT_FILE="${RESULTS_DIR}/eval_${BACKEND}_${MODEL//\//-}_${BOUNDARY_TAG}.json"

echo "============================================================"
echo "Norm Inference Dataset — Agentic Evaluation"
echo "============================================================"
echo "Backend:       ${BACKEND}"
echo "Model:         ${MODEL}"
echo "Boundary turn: ${BOUNDARY_TURN}"
echo "Delay:         ${DELAY}s"
echo "Python:        $(which python)"
echo "Results:       ${RESULT_FILE}"
echo ""

# ── Build backend-specific flags ──────────────────────────────────────────────
EXTRA_ARGS=""
if [ "${BACKEND}" = "azure" ]; then
    EXTRA_ARGS="--azure-endpoint ${AZURE_ENDPOINT} --azure-api-version ${AZURE_API_VERSION}"
fi
if [ "${BACKEND}" = "vllm" ] && [ -n "${VLLM_BASE_URL}" ]; then
    EXTRA_ARGS="--vllm-base-url ${VLLM_BASE_URL}"
fi

# ── Dataset flag ──────────────────────────────────────────────────────────────
DATASET="${DATASET:-${SCRIPT_DIR}/../dataset.json}"
DATASET_FLAG="--dataset ${DATASET}"

python "${SCRIPT_DIR}/eval_agent.py" \
    --backend   "${BACKEND}" \
    --model     "${MODEL}" \
    --output    "${RESULT_FILE}" \
    --delay     "${DELAY}" \
    ${BOUNDARY_FLAG} \
    ${MAX_ITEMS:+--max-items ${MAX_ITEMS}} \
    ${DATASET_FLAG} \
    ${EXTRA_ARGS}

echo ""
echo "Done. Results written to ${RESULT_FILE}"
