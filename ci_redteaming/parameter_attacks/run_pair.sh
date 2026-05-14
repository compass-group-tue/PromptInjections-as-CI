#!/bin/bash
# run_pair.sh — executed on the HTCondor worker node
#
# Configurable via environment variables (set in pair_attack.sub or exported):
#   ATTACKER_BACKEND          anthropic | openai | azure | vllm  (default: azure)
#   ATTACKER_MODEL            model / deployment name             (default: gpt-5-chat)
#   TARGET_BACKEND            anthropic | openai | azure | vllm  (default: azure)
#   TARGET_MODEL              model / deployment name             (default: gpt-5-chat)
#
#   Per-agent API keys (optional — falls back to shared key files):
#   ATTACKER_API_KEY          API key for the attacker agent
#   TARGET_API_KEY            API key for the target agent
#
#   Azure-specific:
#   AZURE_ENDPOINT            Azure OpenAI endpoint URL
#   AZURE_API_VERSION         Azure OpenAI API version            (default: 2025-01-01-preview)
#
#   Judge agent (defaults to attacker backend/model if unset):
#   JUDGE_BACKEND             anthropic | openai | azure | vllm
#   JUDGE_MODEL               model / deployment name
#   JUDGE_API_KEY             API key for the judge agent (optional)
#
#   OpenAI-specific:
#   OPENAI_API_KEY            OpenAI API key (or set ATTACKER/TARGET_API_KEY above)
#                             Falls back to $HOME/.openai_token if unset.
#                             Example: ATTACKER_BACKEND=openai ATTACKER_MODEL=gpt-4o
#
#   N_STREAMS                 independent attack streams per scenario  (default: 3)
#   MAX_ITER                  max iterations per stream               (default: 10)
#   DELAY                     seconds between API calls               (default: 0.5)
#   LIMIT                     only attack first N scenarios (default: 3)

set -e

# ── Activate environment ──────────────────────────────────────────────────────
# Activate your virtual environment, e.g.: source .venv/bin/activate

# ── Cluster-specific env ──────────────────────────────────────────────────────
source /etc/profile.d/modules.sh
export SOFT_FILELOCK=1

# ── Defaults ──────────────────────────────────────────────────────────────────
ATTACKER_BACKEND="${ATTACKER_BACKEND:-anthropic}"
ATTACKER_MODEL="${ATTACKER_MODEL:-claude-sonnet-4-6}"
TARGET_BACKEND="${TARGET_BACKEND:-anthropic}"
TARGET_MODEL="${TARGET_MODEL:-claude-sonnet-4-6}"
AZURE_ENDPOINT="${AZURE_ENDPOINT:-https://<YOUR_AZURE_ENDPOINT>.openai.azure.com}"
AZURE_API_VERSION="${AZURE_API_VERSION:-2025-01-01-preview}"
N_STREAMS="${N_STREAMS:-3}"
MAX_ITER="${MAX_ITER:-15}"
DELAY="${DELAY:-0.5}"
LIMIT="${LIMIT:-150}"
SEED="${SEED:-42}"

# ── API keys (file → env) ─────────────────────────────────────────────────────
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

if [ -n "${RESUME_OUTPUT}" ]; then
    RESULT_FILE="${RESUME_OUTPUT}"
else
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    RESULT_FILE="${RESULTS_DIR}/pair_${ATTACKER_BACKEND}_${ATTACKER_MODEL//\//-}_vs_${TARGET_BACKEND}_${TARGET_MODEL//\//-}_${TIMESTAMP}.json"
fi

echo "============================================================"
echo "PAIR Social-Engineering Attack — Email Assistant"
echo "============================================================"
echo "Attacker: ${ATTACKER_BACKEND}/${ATTACKER_MODEL}"
echo "Target:   ${TARGET_BACKEND}/${TARGET_MODEL}"
echo "Streams:  ${N_STREAMS}   MaxIter: ${MAX_ITER}"
echo "Results:  ${RESULT_FILE}"
echo ""

# ── Build backend flags ───────────────────────────────────────────────────────
EXTRA_ARGS=""
if [ "${ATTACKER_BACKEND}" = "azure" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --attacker-azure-endpoint ${AZURE_ENDPOINT} --attacker-azure-api-version ${AZURE_API_VERSION}"
fi
if [ "${TARGET_BACKEND}" = "azure" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --target-azure-endpoint ${AZURE_ENDPOINT} --target-azure-api-version ${AZURE_API_VERSION}"
fi
if [ -n "${ATTACKER_API_KEY}" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --attacker-api-key ${ATTACKER_API_KEY}"
fi
if [ -n "${TARGET_API_KEY}" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --target-api-key ${TARGET_API_KEY}"
fi
if [ -n "${JUDGE_BACKEND}" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --judge-backend ${JUDGE_BACKEND}"
fi
if [ -n "${JUDGE_MODEL}" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --judge-model ${JUDGE_MODEL}"
fi
if [ -n "${JUDGE_API_KEY}" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --judge-api-key ${JUDGE_API_KEY}"
fi
if [ -n "${JUDGE_BACKEND}" ] && [ "${JUDGE_BACKEND}" = "azure" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --judge-azure-endpoint ${AZURE_ENDPOINT} --judge-azure-api-version ${AZURE_API_VERSION}"
fi
if [ -n "${LIMIT}" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --limit ${LIMIT}"
fi
EXTRA_ARGS="${EXTRA_ARGS} --seed ${SEED}"

python "${SCRIPT_DIR}/pair_attack.py" \
    --attacker-backend  "${ATTACKER_BACKEND}" \
    --attacker-model    "${ATTACKER_MODEL}" \
    --target-backend    "${TARGET_BACKEND}" \
    --target-model      "${TARGET_MODEL}" \
    --dataset           "${SCRIPT_DIR}/../../defense_evaluation/paired_emails_dataset.json" \
    --output            "${RESULT_FILE}" \
    --n-streams         "${N_STREAMS}" \
    --max-iter          "${MAX_ITER}" \
    --delay             "${DELAY}" \
    --batch-size        1 \
    ${EXTRA_ARGS}

echo ""
echo "Done. Results written to ${RESULT_FILE}"
