#!/bin/bash
# ============================================================================
# Build SFT warmup dataset: 10k + Qwen2.5-7B-Instruct
# ============================================================================
# Usage: bash scripts/sft/run_build_sft_10k_qwen.sh
#
# Prerequisites: Teacher responses must already be generated at:
#   $DATA_PATH/teacher_responses_10k_qwen2.5-7b
# ============================================================================

set -e
set -x

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DATASET_BASE=${DATA_PATH:-"./data"}

RESPONSE_DIR="${DATASET_BASE}/teacher_responses_10k_qwen2.5-7b"
SOURCE_DATASET="${DATASET_BASE}/mixed_math_code_10k_with_source"
OUTPUT_DIR="${DATASET_BASE}/sft_warmup_10k_qwen2.5-7b"
MODEL_NAME="Qwen2.5-7B-Instruct"

# Check if response directory exists
if [ ! -d "${RESPONSE_DIR}" ]; then
    echo "ERROR: Response directory not found: ${RESPONSE_DIR}"
    echo "Please run the response generation script first:"
    echo "  bash scripts/sft/run_generate_responses_10k_qwen.sh"
    exit 1
fi

echo "=== Building SFT Warmup Dataset ==="
echo "  Response dir: ${RESPONSE_DIR}"
echo "  Source dataset: ${SOURCE_DATASET}"
echo "  Output dir: ${OUTPUT_DIR}"
echo "  Model: ${MODEL_NAME}"
date

python ${SCRIPT_DIR}/build_sft_warmup_dataset.py \
    --response-dir ${RESPONSE_DIR} \
    --source-dataset ${SOURCE_DATASET} \
    --output-dir ${OUTPUT_DIR} \
    --model-name ${MODEL_NAME}

echo "=== Done ==="
date
