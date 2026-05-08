#!/bin/bash
# ============================================================================
# Build SFT warmup dataset: 10k + Phi-4-mini-instruct
# ============================================================================
# Usage: bash scripts/sft/run_build_sft_10k_phi4.sh
#
# Supports both:
#   - Single response dir: teacher_responses_10k_phi-4-mini/
#   - Sharded response dirs: teacher_responses_10k_phi-4-mini_shard*/ (auto-merged)
#
# Prerequisites: Teacher responses must already be generated.
# ============================================================================

set -e
set -x

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DATASET_BASE=${DATA_PATH:-"./data"}

RESPONSE_DIR="${DATASET_BASE}/teacher_responses_10k_phi-4-mini"
SOURCE_DATASET="${DATASET_BASE}/mixed_math_code_10k_with_source"
OUTPUT_DIR="${DATASET_BASE}/sft_warmup_10k_phi-4-mini"
MODEL_NAME="Phi-4-mini-instruct"

# Check if sharded directories exist
SHARD_DIRS=$(ls -d ${RESPONSE_DIR}_shard* 2>/dev/null || true)

if [ -n "${SHARD_DIRS}" ]; then
    # Sharded mode: auto-merge
    SHARD_COUNT=$(echo "${SHARD_DIRS}" | wc -l)
    echo "=== Found ${SHARD_COUNT} shard directories, will auto-merge ==="
    echo "${SHARD_DIRS}"

    echo "=== Building SFT Warmup Dataset (merging shards) ==="
    echo "  Response base dir: ${RESPONSE_DIR}"
    echo "  Source dataset: ${SOURCE_DATASET}"
    echo "  Output dir: ${OUTPUT_DIR}"
    echo "  Model: ${MODEL_NAME}"
    date

    python ${SCRIPT_DIR}/build_sft_warmup_dataset.py \
        --response-dir ${RESPONSE_DIR} \
        --auto-merge-shards \
        --source-dataset ${SOURCE_DATASET} \
        --output-dir ${OUTPUT_DIR} \
        --model-name ${MODEL_NAME}

elif [ -d "${RESPONSE_DIR}" ]; then
    # Single directory mode (original behavior)
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

else
    echo "ERROR: No response directory found!"
    echo "  Expected: ${RESPONSE_DIR} or ${RESPONSE_DIR}_shard*"
    echo "Please run the response generation script first:"
    echo "  bash scripts/sft/run_generate_responses_10k_phi4.sh"
    echo "  or (sharded):"
    echo "  bash scripts/sft/run_generate_responses_10k_phi4_sharded.sh 0 4"
    exit 1
fi

echo "=== Done ==="
date
