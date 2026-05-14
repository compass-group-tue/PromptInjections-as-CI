#!/bin/bash
# run_edit.sh — executed on the HTCondor worker node
#
# Edits the norm inference dataset via Azure OpenAI.
#
# Configurable via environment variables (set in edit.sub or exported):
#   INPUT_FILE        Input dataset JSON path           (default: latest in results/)
#   TEMPERATURE       LLM temperature                   (default: 0.0)
#   DELAY             Seconds between API calls         (default: 1.0)
#   AZURE_ENDPOINT    Azure OpenAI endpoint URL         (required)
#   AZURE_DEPLOYMENT  Deployment name (e.g., gpt-5-chat)(required)
#   AZURE_API_VERSION Azure OpenAI API version          (default: 2024-10-21)

set -e

# ── Activate environment ──────────────────────────────────────────────────────
# Activate your virtual environment, e.g.: source .venv/bin/activate

# ── Cluster-specific env ──────────────────────────────────────────────────────

# ── Defaults ──────────────────────────────────────────────────────────────────
TEMPERATURE="${TEMPERATURE:-0.0}"
DELAY="${DELAY:-1.0}"
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

# ── Resolve input file ────────────────────────────────────────────────────────
if [ -n "${INPUT_FILE}" ]; then
    INPUT_PATH="${INPUT_FILE}"
else
    # Default: latest dataset_*.json in results/
    INPUT_PATH=$(ls -t "${RESULTS_DIR}"/dataset_*.json 2>/dev/null | head -n1)
    if [ -z "${INPUT_PATH}" ]; then
        # Fall back to the top-level dataset.json
        INPUT_PATH="${SCRIPT_DIR}/../dataset.json"
    fi
fi

if [ ! -f "${INPUT_PATH}" ]; then
    echo "ERROR: input file not found: ${INPUT_PATH}" >&2
    exit 1
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_FILE="${RESULTS_DIR}/dataset_edited_${AZURE_DEPLOYMENT//\//-}_${TIMESTAMP}.json"

echo "============================================================"
echo "Norm Inference Dataset — Editing"
echo "============================================================"
echo "Azure endpoint:  ${AZURE_ENDPOINT}"
echo "Deployment:      ${AZURE_DEPLOYMENT}"
echo "API version:     ${AZURE_API_VERSION}"
echo "Temperature:     ${TEMPERATURE}"
echo "Delay:           ${DELAY}s"
echo "Input:           ${INPUT_PATH}"
echo "Output:          ${OUTPUT_FILE}"
echo ""

python "${SCRIPT_DIR}/edit_dataset.py" \
    --input       "${INPUT_PATH}" \
    --output      "${OUTPUT_FILE}" \
    --temperature "${TEMPERATURE}" \
    --delay       "${DELAY}" \
    --resume

echo ""
echo "Done. Edited dataset written to ${OUTPUT_FILE}"
