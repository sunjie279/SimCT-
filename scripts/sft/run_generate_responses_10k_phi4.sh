#!/bin/bash
# ============================================================================
# Generate teacher responses: Phi-4-mini-instruct on mixed_math_code_10k
# ============================================================================
# Usage: bash scripts/sft/run_generate_responses_10k_phi4.sh
#
# This script:
#   1. Copies model to local SSD (if not already present)
#   2. Starts SGLang server with DP=8
#   3. Waits for server to be ready
#   4. Runs the response generation script (8 trajectories per question)
#   5. Shuts down the server
# ============================================================================

set -e
set -x

export SGLANG_DISABLE_CUDNN_CHECK=1

# --- Configuration ---
MODEL_NAME="Phi-4-mini-instruct"
MODEL_PATH=${MODEL_PATH:-"./models"}
DATA_PATH=${DATA_PATH:-"./data"}
REMOTE_MODEL_PATH="${MODEL_PATH}/${MODEL_NAME}"
LOCAL_MODEL_PATH="/root/workspace/models/${MODEL_NAME}"
DATASET_PATH="${DATA_PATH}/mixed_math_code_10k_with_source"
DATASET_TAG="10k"
OUTPUT_BASE="${DATA_PATH}"
PORT=30000
DP_SIZE=8
SERVER_LOG="/tmp/sglang_server_responses_10k_phi4.log"
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# --- Step 0: Copy model to local SSD ---
if [ ! -d "${LOCAL_MODEL_PATH}" ]; then
    echo "=== Copying model to local SSD ==="
    mkdir -p "$(dirname ${LOCAL_MODEL_PATH})"
    rsync --progress -a "${REMOTE_MODEL_PATH}/" "${LOCAL_MODEL_PATH}/"
    echo "Model copied to ${LOCAL_MODEL_PATH}"
else
    echo "Model already exists at ${LOCAL_MODEL_PATH}"
fi

# --- Step 1: Clean up old processes ---
echo "=== Cleaning up old processes on port ${PORT} ==="
OLD_PIDS=$(lsof -ti :${PORT} 2>/dev/null || true)
if [ -n "${OLD_PIDS}" ]; then
    echo "Killing old processes on port ${PORT}: ${OLD_PIDS}"
    echo "${OLD_PIDS}" | xargs kill -9 2>/dev/null || true
    sleep 3
fi
pkill -9 -f "sglang.launch_server.*--port ${PORT}" 2>/dev/null || true
sleep 2

# --- Step 2: Start SGLang server ---
echo "=== Starting SGLang Server ==="
echo "Model: ${LOCAL_MODEL_PATH}"
echo "DP size: ${DP_SIZE}"
date

python -m sglang.launch_server \
    --model-path ${LOCAL_MODEL_PATH} \
    --dp-size ${DP_SIZE} \
    --tp-size 1 \
    --port ${PORT} \
    --mem-fraction-static 0.78 \
    --trust-remote-code \
    > ${SERVER_LOG} 2>&1 &

SERVER_PID=$!
echo "Server PID: ${SERVER_PID}"

cleanup() {
    echo "=== Cleaning up ==="
    kill ${SERVER_PID} 2>/dev/null || true
    wait ${SERVER_PID} 2>/dev/null || true
}
trap cleanup EXIT

# --- Step 3: Wait for server ---
echo "Waiting for server to be ready..."
SERVER_READY=0
for i in $(seq 1 180); do
    if ! kill -0 ${SERVER_PID} 2>/dev/null; then
        echo "Server process died! Check ${SERVER_LOG}"
        tail -50 ${SERVER_LOG}
        exit 1
    fi
    HTTP_CODE=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:${PORT}/health 2>/dev/null || echo "000")
    if [ "${HTTP_CODE}" = "200" ]; then
        echo "Server is ready! (waited ${i}*5 seconds = $((i*5))s)"
        SERVER_READY=1
        break
    fi
    if [ $((i % 12)) -eq 0 ]; then
        echo "  Still waiting... (${i}*5s elapsed, last HTTP code: ${HTTP_CODE})"
    fi
    sleep 5
done

if [ ${SERVER_READY} -ne 1 ]; then
    echo "Server failed to start after 900s. Check ${SERVER_LOG}"
    tail -50 ${SERVER_LOG}
    kill ${SERVER_PID} 2>/dev/null
    exit 1
fi

# --- Step 4: Run generation ---
echo "=== Running Generation ==="
date

python ${SCRIPT_DIR}/generate_teacher_responses.py \
    --model-name ${MODEL_NAME} \
    --dataset-path ${DATASET_PATH} \
    --dataset-tag ${DATASET_TAG} \
    --output-base ${OUTPUT_BASE} \
    --base-url http://127.0.0.1:${PORT} \
    --temperature 0.6 \
    --top-p 0.95 \
    --n-trajectories 8 \
    --max-concurrent 64 \
    --max-new-tokens 4096 \
    --batch-size 2000 \
    --resume

echo "=== Done ==="
date
