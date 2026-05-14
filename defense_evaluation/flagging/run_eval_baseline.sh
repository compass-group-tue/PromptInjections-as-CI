#!/bin/bash
# run_eval_baseline.sh — HTCondor worker script for Llama-3.3-70B baseline detection eval
#
# Loads meta-llama/Llama-3.3-70B-Instruct directly via vLLM (no LoRA adapter)
# and evaluates prompt-injection detection on the secalign_for_detection dataset.
#
# Configurable via environment variables (set in eval_baseline.sub):
#   MODEL              HF model ID           (default: meta-llama/Llama-3.3-70B-Instruct)
#   TENSOR_PARALLEL    Number of GPUs        (default: 4)
#   MAX_MODEL_LEN      vLLM max seq len      (default: 8192)
#   HF_HOME            HF cache root         (default: ${HF_HOME:-$HOME/.cache/huggingface})
#   MAX_ITEMS          Evaluate first N items only (default: all)
#   BATCH_SIZE         Checkpoint every N new items (default: 50)
#   MAX_TOKENS         Max generation tokens per item (default: 1024)

set -e

# ── Activate environment ──────────────────────────────────────────────────────
# source /path/to/your/venv/bin/activate

# ── Cluster-specific env ──────────────────────────────────────────────────────
source /etc/profile.d/modules.sh
module load cuda
export SOFTFILELOCK=1

# vLLM workers look for 'gcc-4.6' by exact name; create a versioned symlink.
if ! command -v gcc-4.6 &>/dev/null; then
    _GCC_BIN=$(command -v gcc)
    if [ -n "$_GCC_BIN" ]; then
        _GCC_SHIM=$(mktemp -d)
        ln -s "$_GCC_BIN" "${_GCC_SHIM}/gcc-4.6"
        export PATH="${_GCC_SHIM}:${PATH}"
    fi
fi

# ── HuggingFace cache ─────────────────────────────────────────────────────────
HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export HF_HOME
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"

# HF token for gated models (Llama 3.3)
# Edit path to your HuggingFace token file, or export HF_TOKEN directly.
# if [ -f /path/to/hf_token ]; then
#     export HUGGING_FACE_HUB_TOKEN=$(cat /path/to/hf_token)
#     export HF_TOKEN="${HUGGING_FACE_HUB_TOKEN}"
# fi

# ── Defaults ──────────────────────────────────────────────────────────────────
MODEL="${MODEL:-meta-llama/Llama-3.3-70B-Instruct}"
TENSOR_PARALLEL="${TENSOR_PARALLEL:-4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
BATCH_SIZE="${BATCH_SIZE:-50}"
MAX_TOKENS="${MAX_TOKENS:-1024}"

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/results"
mkdir -p "${RESULTS_DIR}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULT_FILE="${RESULTS_DIR}/eval_baseline_${MODEL//\//-}_${TIMESTAMP}.json"

echo "============================================================"
echo "SecAlign Detection Evaluation — Llama Baseline"
echo "============================================================"
echo "Model:           ${MODEL}"
echo "Tensor parallel: ${TENSOR_PARALLEL}"
echo "Max model len:   ${MAX_MODEL_LEN}"
echo "HF home:         ${HF_HOME}"
echo "Python:          $(which python)"
echo "Results:         ${RESULT_FILE}"
echo ""

# ── Verify model is downloaded ────────────────────────────────────────────────
python - <<'PYCHECK'
import os, sys
hf_home = os.environ["HF_HOME"]
model_id = os.environ.get("MODEL", "meta-llama/Llama-3.3-70B-Instruct")
slug = "models--" + model_id.replace("/", "--")
snap_dir = os.path.join(hf_home, "hub", slug, "snapshots")
if os.path.isdir(snap_dir) and os.listdir(snap_dir):
    print(f"  [OK] {model_id}")
else:
    print(f"  [MISSING] {model_id} — not found under {snap_dir}")
    print("\nDownload the model first with huggingface-cli or snapshot_download.")
    sys.exit(1)
PYCHECK

echo ""

# ── Build optional flags ──────────────────────────────────────────────────────
EXTRA_ARGS=""
if [ -n "${MAX_ITEMS}" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --max-items ${MAX_ITEMS}"
fi

# ── Run evaluation ────────────────────────────────────────────────────────────
python "${SCRIPT_DIR}/eval_detection.py" \
    --backend             llama_baseline \
    --model               "${MODEL}" \
    --dataset             "${SCRIPT_DIR}/../llmail_inject_dataset.json" \
    --output              "${RESULT_FILE}" \
    --delay               0.0 \
    --batch-size          "${BATCH_SIZE}" \
    --max-tokens          "${MAX_TOKENS}" \
    --tensor-parallel-size "${TENSOR_PARALLEL}" \
    --hf-home             "${HF_HOME}" \
    --max-model-len       "${MAX_MODEL_LEN}" \
    ${EXTRA_ARGS}

echo ""
echo "Done. Results written to ${RESULT_FILE}"
