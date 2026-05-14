#!/bin/bash
# run_generate.sh — Generates the paired emails prompt injection dataset via Azure OpenAI.
#
# Usage:
#   bash run_generate.sh
#
# Configurable via environment variables:
#   NUM_ITEMS         Total examples to generate        (default: 4200)
#   BATCH_SIZE        Items per API call                (default: 5)
#   TEMPERATURE       Sampling temperature              (default: 0.9)
#   AZURE_ENDPOINT    Azure OpenAI endpoint URL         (required)
#   AZURE_DEPLOYMENT  Deployment name (e.g., gpt-4o)   (required)
#   AZURE_API_VERSION Azure OpenAI API version          (default: 2025-01-01-preview)
#   AZURE_OPENAI_API_KEY  API key                       (required, or set via key file below)
#
# Example:
#   AZURE_ENDPOINT=https://your-resource.openai.azure.com \
#   AZURE_DEPLOYMENT=gpt-4o \
#   AZURE_OPENAI_API_KEY=your-key \
#   bash run_generate.sh

set -e

# ── Activate environment ──────────────────────────────────────────────────────
# source /path/to/your/venv/bin/activate

# ── Defaults ──────────────────────────────────────────────────────────────────
NUM_ITEMS="${NUM_ITEMS:-5}"
BATCH_SIZE="${BATCH_SIZE:-2}"
TEMPERATURE="${TEMPERATURE:-0.9}"
AZURE_API_VERSION="${AZURE_API_VERSION:-2025-01-01-preview}"
AZURE_ENDPOINT="${AZURE_ENDPOINT:-}"

# ── API keys (file → env) ─────────────────────────────────────────────────────
if [ -f "${AZURE_OPENAI_API_KEY_FILE:-}" ]; then
    export AZURE_OPENAI_API_KEY=$(cat "${AZURE_OPENAI_API_KEY_FILE}")
fi

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
echo "Paired Emails Dataset — Generation"
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

python "${SCRIPT_DIR}/generate_dataset.py" \
    --num-items   "${NUM_ITEMS}" \
    --batch-size  "${BATCH_SIZE}" \
    --temperature "${TEMPERATURE}" \
    --prompt      "${SCRIPT_DIR}/generation_prompt.txt" \
    --output      "${OUTPUT_FILE}"

echo ""
echo "Done. Dataset written to ${OUTPUT_FILE}"
