#!/bin/bash
# run_generate.sh — executed on the HTCondor worker node
#
# Generates the norm inference dataset via Azure OpenAI.
#
# Configurable via environment variables (set in generate.sub or exported):
#   NUM_ITEMS         Total examples to generate        (default: 64)
#   BATCH_SIZE        Items per API call                (default: 4)
#   TEMPERATURE       Sampling temperature              (default: 0.9)
#   AZURE_ENDPOINT    Azure OpenAI endpoint URL         (required)
#   AZURE_DEPLOYMENT  Deployment name (e.g., gpt-4o)   (required)
#   AZURE_API_VERSION Azure OpenAI API version          (default: 2024-10-21)

set -e

# ── Activate environment ──────────────────────────────────────────────────────
# Activate your virtual environment, e.g.: source .venv/bin/activate

# ── Cluster-specific env ──────────────────────────────────────────────────────

# ── Defaults ──────────────────────────────────────────────────────────────────
NUM_ITEMS="${NUM_ITEMS:-32}"
BATCH_SIZE="${BATCH_SIZE:-4}"
TEMPERATURE="${TEMPERATURE:-0.9}"
AZURE_API_VERSION="${AZURE_API_VERSION:-2024-10-21}"

# ── API keys (file → env) ─────────────────────────────────────────────────────
# if [ -f /path/to/azure_openai_token ]; then
#     export AZURE_OPENAI_API_KEY=$(cat /path/to/azure_openai_token)
# fi

# ── Validate required vars ────────────────────────────────────────────────────
if [ -z "${AZURE_ENDPOINT}" ]; then
    echo "ERROR: AZURE_ENDPOINT is not set." >&2
    exit 1
fi
if [ -z "${AZURE_DEPLOYMENT}" ]; then
    echo "ERROR: AZURE_DEPLOYMENT is not set." >&2
    exit 1
fi

export AZURE_OPENAI_ENDPOINT="${AZURE_ENDPOINT}"
export AZURE_OPENAI_DEPLOYMENT="${AZURE_DEPLOYMENT}"
export AZURE_OPENAI_API_VERSION="${AZURE_API_VERSION}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/results"
mkdir -p "${RESULTS_DIR}"
mkdir -p "${SCRIPT_DIR}/logs"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_FILE="${RESULTS_DIR}/dataset_${AZURE_DEPLOYMENT//\//-}_${TIMESTAMP}.json"

echo "============================================================"
echo "Norm Inference Dataset — Generation"
echo "============================================================"
echo "Azure endpoint:  ${AZURE_ENDPOINT}"
echo "Deployment:      ${AZURE_DEPLOYMENT}"
echo "API version:     ${AZURE_API_VERSION}"
echo "Num items:       ${NUM_ITEMS}"
echo "Batch size:      ${BATCH_SIZE}"
echo "Temperature:     ${TEMPERATURE}"
echo "Python:          $(which python)"
echo "Output:          ${OUTPUT_FILE}"
echo ""

# To resume/append to an existing dataset, pass:
#   --append-from <path/to/previous/dataset.json>
APPEND_FROM_ARG=""
if [ -n "${APPEND_FROM}" ]; then
    APPEND_FROM_ARG="--append-from ${APPEND_FROM}"
fi

python "${SCRIPT_DIR}/generate_dataset.py" \
    --num-items   "${NUM_ITEMS}" \
    --batch-size  "${BATCH_SIZE}" \
    --temperature "${TEMPERATURE}" \
    --prompt      "${SCRIPT_DIR}/prompt.txt" \
    --output      "${OUTPUT_FILE}" \
    ${APPEND_FROM_ARG}

echo ""
echo "Done. Dataset written to ${OUTPUT_FILE}"
