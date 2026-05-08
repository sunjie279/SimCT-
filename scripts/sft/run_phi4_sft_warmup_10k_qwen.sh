#!/bin/bash
# ============================================================================
# SFT Phi-4-mini-instruct on sft_warmup_10k_qwen dataset (8723 samples)
# Dataset: Qwen2.5-7B teacher responses, quality-filtered
#   Math: 4526 (51.9%) | Code: 4197 (48.1%)
# Full fine-tuning, 8x GPUs, bf16
# ============================================================================

set -e
set -x

export SGLANG_DISABLE_CUDNN_CHECK=1
export FORCE_TORCHRUN=1
export NNODES=1
export NPROC_PER_NODE=8

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/phi4_sft_warmup_10k_qwen.yaml"
LOG_FILE="${SCRIPT_DIR}/phi4_sft_warmup_10k_qwen.log"

# Copy model to local SSD for faster I/O
MODEL_PATH=${MODEL_PATH:-"./models"}
LOCAL_MODEL_DIR="/root/workspace/models/Phi-4-mini-instruct"
REMOTE_MODEL_DIR="${MODEL_PATH}/Phi-4-mini-instruct"
if [ ! -d "${LOCAL_MODEL_DIR}" ]; then
    echo "=== Copying model to local SSD ==="
    mkdir -p "$(dirname "${LOCAL_MODEL_DIR}")"
    rsync --progress -a "${REMOTE_MODEL_DIR}/" "${LOCAL_MODEL_DIR}/"
    echo "=== Model copy complete ==="
fi

echo "=== Starting SFT Training ==="
echo "Config: ${CONFIG_FILE}"
echo "Log: ${LOG_FILE}"
echo "GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)"
echo "Start time: $(date)"

llamafactory-cli train "${CONFIG_FILE}" 2>&1 | tee "${LOG_FILE}"

echo "=== Training Complete ==="
echo "End time: $(date)"
