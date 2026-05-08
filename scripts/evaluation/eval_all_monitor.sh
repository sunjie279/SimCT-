#!/bin/bash
# =============================================================================
# Monitoring evaluation script 1/2 for KDFlow (span_ctkd + dskd)
# Supports: aime24, aime25, aime26, math500
#
# Architecture:
#   1. Group by model, launch one SGLang server per model
#   2. Model <= 9B: TP=1, DP=8 (8 replicas on 8*H20)
#   3. All datasets for the same model share one server
#   4. Uses sglang.launch_server built-in DP support
#   5. Model is copied to local SSD before serving for fast I/O
# =============================================================================

set -o pipefail
export SGLANG_DISABLE_CUDNN_CHECK=1

# ======================== Configuration (edit here) ========================

# Format: "model_path:tp_size:dp_size"
# model_path can be:
#   - absolute path: /path/to/model
#   - relative name: simple_ctkd-gsm8k-5e-6 (resolved under CKPTS_DIR)
MODELS=(
    # --- model_path ---
    # example "model_path:1:8"
)

# Format: "dataset_name:max_tokens"
# NOTE on max_tokens sizing:
#   - gemma-2-2b-it has max_position_embeddings=8192 (the smallest model)
#   - Qwen2.5-* models have max_position_embeddings=32768
#   - Phi-4-mini-instruct has max_position_embeddings=131072
#   max_tokens is the max NEW tokens to generate. Total context = prompt + max_tokens.

DATASETS=(
    "gsm8k:4096"
    "math500:4096"

    "mbpp:2048"
    "live-code-bench-v6:4096"
)

# ======================== Paths ========================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
EVAL_SCRIPT="${SCRIPT_DIR}/evaluation.py"
OUTPUT_DIR="${SCRIPT_DIR}/output/results_t0.6_topp0.95"
CKPTS_DIR="${PROJECT_DIR}/output/ckpts"
LOCAL_MODELS_DIR="/root/workspace/models"

# ======================== Server settings ========================

TOTAL_GPUS=8
PORT=30000
BASE_URL="http://127.0.0.1:${PORT}"
MAX_CONCURRENT=256
SERVER_WAIT=600
MEM_FRACTION=0.85

# ======================== Activate GCC 12 (for SGLang JIT) ========================
if [ -f /opt/rh/gcc-toolset-12/enable ]; then
    source /opt/rh/gcc-toolset-12/enable
    echo "GCC version: $(g++ --version 2>&1 | head -1)"
fi

# Disable CuDNN version check in SGLang (Conv3d bug irrelevant for LLM inference)
export SGLANG_DISABLE_CUDNN_CHECK=1

# Clear old JIT compilation cache
rm -rf /root/.cache/tvm-ffi/
echo "✓ JIT cache cleared"

# ======================== Python environment ========================
echo "Python path: $(which python)"
echo "Python version: $(python --version 2>&1)"

if ! python -c "import sglang" 2>/dev/null; then
    echo "✗ Error: sglang not installed"
    exit 1
fi
echo "✓ sglang $(python -c 'import sglang; print(sglang.__version__)') ready"

echo "GPU status:"
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader 2>&1 || echo "  (nvidia-smi query failed)"
fi
python -c "import torch; print(f'  CUDA available: {torch.cuda.is_available()}, GPU count: {torch.cuda.device_count()}')" 2>&1 || true

# ======================== State tracking ========================

SGLANG_PID=""
TASK_TOTAL=$(( ${#MODELS[@]} * ${#DATASETS[@]} ))
TASK_CURRENT=0
TASK_SUCCESS=0
TASK_FAILED=0
FAILED_TASKS=""

# ======================== Helper functions ========================

# Resolve model path: absolute path or relative name under CKPTS_DIR
resolve_model_path() {
    local model_ref="$1"
    if [[ "$model_ref" = /* ]]; then
        echo "$model_ref"
    else
        echo "${CKPTS_DIR}/${model_ref}"
    fi
}

# Get model display name
# For models under CKPTS_DIR, preserve relative path (e.g. simple_ctkd-gsm8k-5e-6/step10)
# For other models, use basename only (e.g. gemma-2-2b-it)
get_model_name() {
    local model_path="$1"
    # Normalize: remove trailing slash
    model_path="${model_path%/}"
    local ckpts_normalized="${CKPTS_DIR%/}"

    if [[ "${model_path}" == "${ckpts_normalized}/"* ]]; then
        # Strip CKPTS_DIR prefix to get relative path
        echo "${model_path#${ckpts_normalized}/}"
    else
        basename "${model_path}"
    fi
}

# Check if results already exist for a model+dataset
has_existing_result() {
    local model_name=$1
    local dataset=$2
    local eval_output="${OUTPUT_DIR}/${model_name}/${dataset}"
    local existing=$(find "${eval_output}" -maxdepth 1 -name "*.json" -not -name "*predictions*" 2>/dev/null | head -1)
    [ -n "${existing}" ]
}

# Ensure model is on local SSD
ensure_model_local() {
    local model_path="$1"
    local model_name=$(get_model_name "$model_path")
    local dst="${LOCAL_MODELS_DIR}/${model_name}"

    if [ -d "${dst}" ] && [ -f "${dst}/config.json" ] && \
       { [ -f "${dst}/tokenizer.json" ] || [ -f "${dst}/tokenizer_config.json" ] || [ -f "${dst}/tokenizer.model" ]; }; then
        echo "  ✓ Local cache hit: ${dst}"
        return 0
    fi

    # Incomplete cache detected — remove and re-copy
    if [ -d "${dst}" ] && [ -f "${dst}/config.json" ]; then
        echo "  ⚠ Incomplete local cache detected (missing tokenizer files), re-copying..."
        rm -rf "${dst}"
    fi

    if [ ! -d "${model_path}" ]; then
        echo "  ✗ Model not found: ${model_path}"
        return 1
    fi

    echo "  📦 Copying model to local SSD..."
    local copy_start=$(date +%s)
    mkdir -p "${dst}"

    if rsync -a --info=progress2 --exclude='epoch_*' --exclude='rollout_data' --exclude='global_step*' "${model_path}/" "${dst}/" 2>/dev/null; then
        local copy_end=$(date +%s)
        local copy_time=$((copy_end - copy_start))
        local cache_size=$(du -sh "${dst}" 2>/dev/null | cut -f1)
        echo "  ✓ Model cached locally (${cache_size}, ${copy_time}s)"
    else
        echo "  rsync unavailable, using cp..."
        if find "${model_path}" -maxdepth 1 -type f -exec cp {} "${dst}/" \; 2>/dev/null; then
            local copy_end=$(date +%s)
            local copy_time=$((copy_end - copy_start))
            echo "  ✓ Model cached locally (${copy_time}s)"
        else
            echo "  ⚠ Local cache failed, will read from network storage"
            rm -rf "${dst}" 2>/dev/null
            return 1
        fi
    fi
}

start_sglang_server() {
    local model_path=$1
    local model_name=$(get_model_name "$model_path")
    local local_path="${LOCAL_MODELS_DIR}/${model_name}"
    local tp=${2:-1}
    local dp=${3:-8}

    # Log file
    local log_dir="${OUTPUT_DIR}/logs"
    mkdir -p "${log_dir}"
    local log_name=$(echo "${model_name}" | tr '/' '_')
    local log_file="${log_dir}/sglang_server_${log_name}.log"

    echo ""
    echo "=========================================="
    echo "Starting SGLang server: ${model_name}"
    echo "  Model path: ${local_path}"
    echo "  TP=${tp}, DP=${dp}, Total GPU=$((tp * dp))"
    echo "  Port: ${PORT}, mem_fraction: ${MEM_FRACTION}"
    echo "  Log file: ${log_file}"
    echo "=========================================="

    # Determine actual model path (prefer local SSD)
    local serve_path="${local_path}"
    if [ ! -d "${serve_path}" ] || [ ! -f "${serve_path}/config.json" ]; then
        serve_path="${model_path}"
        if [ ! -d "${serve_path}" ]; then
            echo "✗ Model path not found: ${serve_path}"
            return 1
        fi
        echo "  ⚠ Using original path: ${serve_path}"
    fi

    # Ensure previous server is stopped
    stop_sglang_server 2>/dev/null || true
    sleep 2

    # Check GPU memory, clean residual processes
    echo "Checking GPU memory status..."
    if command -v nvidia-smi &>/dev/null; then
        local gpu_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr '\n' ' ')
        echo "  Current GPU mem(MiB): ${gpu_used}"

        local has_residual=false
        for mem in $gpu_used; do
            if [ "$mem" -gt 2000 ] 2>/dev/null; then
                has_residual=true
                break
            fi
        done

        if [ "$has_residual" = true ]; then
            echo "  ⚠ Detected large GPU memory usage, cleaning residual processes..."
            pkill -f "sglang" 2>/dev/null || true
            sleep 3
            gpu_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr '\n' ' ')
            echo "  After cleanup GPU mem(MiB): ${gpu_used}"
        else
            echo "  ✓ GPU memory clean"
        fi
    fi

    # Clear log file
    > "${log_file}"

    # Launch SGLang server
    local STDBUF_CMD=""
    if command -v stdbuf &>/dev/null; then
        STDBUF_CMD="stdbuf -oL"
    fi
    PYTHONUNBUFFERED=1 $STDBUF_CMD python -m sglang.launch_server \
        --model-path "${serve_path}" \
        --tp-size ${tp} \
        --dp-size ${dp} \
        --port ${PORT} \
        --mem-fraction-static ${MEM_FRACTION} \
        --trust-remote-code \
        --log-level info \
        > "${log_file}" 2>&1 &

    SGLANG_PID=$!
    echo "SGLang server PID: ${SGLANG_PID}"

    sleep 2

    if ! kill -0 $SGLANG_PID 2>/dev/null; then
        echo "✗ SGLang server crashed immediately! Last 50 lines of log:"
        tail -50 "${log_file}" 2>/dev/null || echo "(log file empty or missing)"
        return 1
    fi

    echo "Waiting for server to be ready..."
    echo "  (Waiting for ${dp} DP workers to load model + capture CUDA graph...)"
    echo "------------------------------------------"

    local last_line=0
    local elapsed=0
    local healthy_workers=0
    while [ $elapsed -lt $SERVER_WAIT ]; do
        if grep -q "The server is fired up and ready to roll" "${log_file}" 2>/dev/null && \
           curl -sf --max-time 3 "${BASE_URL}/v1/models" > /dev/null 2>&1; then
            echo ""
            echo "------------------------------------------"
            echo "✓ SGLang server ready! (${elapsed}s)"
            echo "  DP workers: ${dp} model replicas running in parallel"

            # Warm-up
            echo "  Running warm-up request..."
            curl -s -X POST "${BASE_URL}/v1/completions" \
                -H "Content-Type: application/json" \
                -d '{"model":"default","prompt":"Hello","max_tokens":1,"temperature":0}' \
                > /dev/null 2>&1 || true
            echo "  ✓ Warm-up done"
            return 0
        fi

        if ! kill -0 $SGLANG_PID 2>/dev/null; then
            echo ""
            echo "------------------------------------------"
            echo "✗ SGLang server process exited!"
            if grep -q "OutOfMemoryError\|CUDA out of memory" "${log_file}" 2>/dev/null; then
                echo "  Cause: CUDA OOM!"
                grep "OutOfMemoryError\|CUDA out of memory" "${log_file}" 2>/dev/null | head -5
            else
                echo "  Last 30 lines of log:"
                tail -30 "${log_file}" 2>/dev/null
            fi
            echo "  Full log: ${log_file}"
            return 1
        fi

        # Stream key log lines
        if [ -f "${log_file}" ]; then
            local total_lines=$(wc -l < "${log_file}" 2>/dev/null || echo 0)
            if [ "$total_lines" -gt "$last_line" ]; then
                sed -n "$((last_line+1)),${total_lines}p" "${log_file}" | \
                    grep -v "detect_connection_mode" | \
                    grep -v "Step retrying" | \
                    grep -v "Step started.*attempt=" | \
                    grep -v "Step failed.*will_retry=true" | \
                    grep -v "server_args=ServerArgs" | \
                    grep -v "FutureWarning" | \
                    grep -v "frozen importlib" | \
                    grep -v "cuda\.cudart module is deprecated" | \
                    grep -v "cuda\.nvrtc module is deprecated" | \
                    grep -v "cuda\.bindings" | \
                    grep -v "Ignore import error" | \
                    grep -v "^$" || true
                last_line=$total_lines
            fi

            local new_healthy
            new_healthy=$(grep -c "Capture cuda graph end" "${log_file}" 2>/dev/null) || new_healthy=0
            if [ "$new_healthy" -gt "$healthy_workers" ]; then
                healthy_workers=$new_healthy
                echo "  [Progress] CUDA graph captured: ${healthy_workers}/${dp} workers"
            fi
        fi

        if [ $((elapsed % 15)) -eq 0 ] && [ $elapsed -gt 0 ]; then
            local gpu_mem=""
            if command -v nvidia-smi &>/dev/null; then
                gpu_mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr '\n' ',' | sed 's/,$//')
            fi
            echo "  ⏳ ${elapsed}s/${SERVER_WAIT}s [GPU mem(MiB): ${gpu_mem:-N/A}]"
        fi

        sleep 3
        elapsed=$((elapsed + 3))
    done

    echo ""
    echo "------------------------------------------"
    echo "✗ SGLang server startup timeout (${SERVER_WAIT}s)!"
    echo "  Full log: ${log_file}"
    tail -30 "${log_file}" 2>/dev/null || echo "(no log)"
    kill $SGLANG_PID 2>/dev/null
    return 1
}

stop_sglang_server() {
    if [ ! -z "${SGLANG_PID:-}" ] && kill -0 $SGLANG_PID 2>/dev/null; then
        echo "Stopping SGLang server (PID: ${SGLANG_PID})..."
        kill $SGLANG_PID 2>/dev/null
        wait $SGLANG_PID 2>/dev/null || true
        echo "✓ Server stopped"
    fi

    pkill -f "sglang.launch_server.*--port ${PORT}" 2>/dev/null || true

    local wait_count=0
    while [ $wait_count -lt 10 ]; do
        if ! ss -tlnp 2>/dev/null | grep -q ":${PORT} " && \
           ! netstat -tlnp 2>/dev/null | grep -q ":${PORT} "; then
            break
        fi
        sleep 1
        wait_count=$((wait_count + 1))
    done

    SGLANG_PID=""
}

run_eval() {
    local model_path=$1
    local model_name=$2
    local dataset=$3
    local max_tokens=$4

    TASK_CURRENT=$((TASK_CURRENT + 1))
    local task_start=$(date +%s)
    local eval_output="${OUTPUT_DIR}/${model_name}/${dataset}"

    echo ""
    echo ">>> [${TASK_CURRENT}/${TASK_TOTAL}] Evaluating ${dataset} (model: ${model_name})"
    echo ">>> max_tokens: ${max_tokens}"
    echo ">>> output: ${eval_output}"
    echo ">>> concurrency: ${MAX_CONCURRENT}"

    # Check if results already exist
    local existing_result=$(find "${eval_output}" -maxdepth 1 -name "*.json" -not -name "*predictions*" 2>/dev/null | head -1)
    if [ -n "${existing_result}" ]; then
        echo ">>> ✓ Results already exist, skipping"
        echo ">>> Result file: ${existing_result}"
        TASK_SUCCESS=$((TASK_SUCCESS + 1))
        return 0
    fi

    if python "${EVAL_SCRIPT}" \
        --model_path "${model_path}" \
        --dataset "${dataset}" \
        --base_url "${BASE_URL}" \
        --max_concurrent ${MAX_CONCURRENT} \
        --max_new_tokens ${max_tokens} \
        --output_dir "${eval_output}" \
        --temperature 0.6 \
        --top_p 0.95; then

        local task_end=$(date +%s)
        local task_elapsed=$((task_end - task_start))
        echo ">>> ✓ [${TASK_CURRENT}/${TASK_TOTAL}] ${model_name}/${dataset} done (${task_elapsed}s)"
        TASK_SUCCESS=$((TASK_SUCCESS + 1))
        return 0
    else
        local task_end=$(date +%s)
        local task_elapsed=$((task_end - task_start))
        echo ">>> ✗ [${TASK_CURRENT}/${TASK_TOTAL}] ${model_name}/${dataset} failed (${task_elapsed}s)"
        TASK_FAILED=$((TASK_FAILED + 1))
        FAILED_TASKS="${FAILED_TASKS}  - ${model_name}/${dataset}"$'\n'
        return 1
    fi
}

# ======================== Cleanup hook ========================
cleanup() {
    echo ""
    echo "Cleaning up resources..."
    stop_sglang_server
    if [ -d "${LOCAL_MODELS_DIR}" ]; then
        local cache_size=$(du -sh "${LOCAL_MODELS_DIR}" 2>/dev/null | cut -f1)
        echo "Local model cache preserved: ${LOCAL_MODELS_DIR} (${cache_size})"
    fi
}
trap cleanup EXIT

# ======================== Monitoring helpers ========================

POLL_INTERVAL=60        # seconds between each scan
MIN_AGE_SECONDS=120     # model dir must exist for at least 5 minutes (300s)

# Check if a model directory is ready for evaluation:
#   1. Directory exists and contains config.json (or at least some model files)
#   2. The directory has not been modified in the last MIN_AGE_SECONDS seconds
#      (this prevents evaluating a model that is still being saved)
is_model_ready() {
    local model_path="$1"

    # Directory must exist
    if [ ! -d "${model_path}" ]; then
        return 1
    fi

    # Find the most recently modified file in the directory
    local newest_file_time
    newest_file_time=$(find "${model_path}" -type f -printf '%T@\n' 2>/dev/null | sort -rn | head -1)

    if [ -z "${newest_file_time}" ]; then
        # Directory exists but is empty — not ready
        return 1
    fi

    local now
    now=$(date +%s)
    # newest_file_time may have decimals, truncate to integer
    local newest_int=${newest_file_time%.*}
    local age=$((now - newest_int))

    if [ "${age}" -lt "${MIN_AGE_SECONDS}" ]; then
        return 1
    fi

    return 0
}

# Check if ALL datasets for a given model have been evaluated
is_model_fully_evaluated() {
    local model_name="$1"
    for ds_entry in "${DATASETS[@]}"; do
        local ds_name
        ds_name=$(echo "${ds_entry}" | cut -d: -f1)
        if ! has_existing_result "${model_name}" "${ds_name}"; then
            return 1
        fi
    done
    return 0
}

# ======================== Start monitoring ========================

mkdir -p "${OUTPUT_DIR}"

echo "=========================================="
echo "KDFlow Monitoring Evaluation (SGLang DP)"
echo "=========================================="
echo "Total GPUs: ${TOTAL_GPUS}"
echo "Models to watch: ${#MODELS[@]}, Datasets: ${#DATASETS[@]}"
echo "Concurrency: ${MAX_CONCURRENT}"
echo "Total eval tasks: ${TASK_TOTAL}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Poll interval: ${POLL_INTERVAL}s"
echo "Min model age: ${MIN_AGE_SECONDS}s ($(( MIN_AGE_SECONDS / 60 ))min)"
echo "=========================================="

TOTAL_START=$(date +%s)

# Associative array to track which models have been fully evaluated
declare -A EVALUATED_MODELS

# ----------------------------------------------------------
# Monitoring loop: poll every POLL_INTERVAL seconds
# ----------------------------------------------------------
while true; do
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

    # Count how many models are done
    DONE_COUNT=0
    PENDING_COUNT=0
    NOT_READY_COUNT=0

    for model_entry in "${MODELS[@]}"; do
        MODEL_REF=$(echo "${model_entry}" | cut -d: -f1)
        MODEL_PATH=$(resolve_model_path "${MODEL_REF}")
        MODEL_NAME=$(get_model_name "${MODEL_PATH}")

        # Already evaluated in a previous iteration
        if [ "${EVALUATED_MODELS[${MODEL_NAME}]:-}" = "done" ]; then
            DONE_COUNT=$((DONE_COUNT + 1))
            continue
        fi

        # Check if results already exist from a prior run
        if is_model_fully_evaluated "${MODEL_NAME}"; then
            echo "[${TIMESTAMP}] ✓ ${MODEL_NAME} — already evaluated (results exist)"
            EVALUATED_MODELS["${MODEL_NAME}"]="done"
            # Count the tasks as success
            for ds_entry in "${DATASETS[@]}"; do
                ds_name=$(echo "${ds_entry}" | cut -d: -f1)
                ds_tokens=$(echo "${ds_entry}" | cut -d: -f2)
                TASK_CURRENT=$((TASK_CURRENT + 1))
                TASK_SUCCESS=$((TASK_SUCCESS + 1))
            done
            DONE_COUNT=$((DONE_COUNT + 1))
            continue
        fi

        # Check if model directory is ready (exists + stable for 5 min)
        if ! is_model_ready "${MODEL_PATH}"; then
            NOT_READY_COUNT=$((NOT_READY_COUNT + 1))
            continue
        fi

        # Model is ready and not yet evaluated — run evaluation now
        PENDING_COUNT=$((PENDING_COUNT + 1))

        echo ""
        echo "[${TIMESTAMP}] 🔍 Detected ready model: ${MODEL_NAME}"
        echo "  Path: ${MODEL_PATH}"

        MODEL_TP=$(echo "${model_entry}" | cut -d: -f2)
        MODEL_DP=$(echo "${model_entry}" | cut -d: -f3)

        echo "=========================================="
        echo "Model: ${MODEL_NAME} (TP=${MODEL_TP}, DP=${MODEL_DP})"
        echo "=========================================="

        ensure_model_local "${MODEL_PATH}"

        # Check if all datasets already done (partial re-check)
        all_done=true
        for ds_entry in "${DATASETS[@]}"; do
            ds_name=$(echo "${ds_entry}" | cut -d: -f1)
            if ! has_existing_result "${MODEL_NAME}" "${ds_name}"; then
                all_done=false
                break
            fi
        done

        if [ "${all_done}" = true ]; then
            echo ">>> All tasks for ${MODEL_NAME} already have results, skipping server launch"
            for ds_entry in "${DATASETS[@]}"; do
                ds_name=$(echo "${ds_entry}" | cut -d: -f1)
                ds_tokens=$(echo "${ds_entry}" | cut -d: -f2)
                run_eval "${MODEL_PATH}" "${MODEL_NAME}" "${ds_name}" "${ds_tokens}"
            done
        else
            if start_sglang_server "${MODEL_PATH}" "${MODEL_TP}" "${MODEL_DP}"; then
                for ds_entry in "${DATASETS[@]}"; do
                    ds_name=$(echo "${ds_entry}" | cut -d: -f1)
                    ds_tokens=$(echo "${ds_entry}" | cut -d: -f2)
                    run_eval "${MODEL_PATH}" "${MODEL_NAME}" "${ds_name}" "${ds_tokens}"
                done
                stop_sglang_server
            else
                echo "✗ Skipping ${MODEL_NAME} evaluation (server startup failed)"
                num_datasets=${#DATASETS[@]}
                TASK_CURRENT=$((TASK_CURRENT + num_datasets))
                TASK_FAILED=$((TASK_FAILED + num_datasets))
                for ds_entry in "${DATASETS[@]}"; do
                    ds_name=$(echo "${ds_entry}" | cut -d: -f1)
                    FAILED_TASKS="${FAILED_TASKS}  - ${MODEL_NAME}/${ds_name} (server failed)"$'\n'
                done
            fi
        fi

        # Mark as evaluated (even if failed, to avoid infinite retry)
        EVALUATED_MODELS["${MODEL_NAME}"]="done"
        DONE_COUNT=$((DONE_COUNT + 1))
    done

    # Check if all models are done
    TOTAL_MODELS=${#MODELS[@]}
    echo ""
    echo "[${TIMESTAMP}] === Status: ${DONE_COUNT}/${TOTAL_MODELS} models done, ${NOT_READY_COUNT} not ready ==="

    if [ "${DONE_COUNT}" -ge "${TOTAL_MODELS}" ]; then
        echo ""
        echo "[${TIMESTAMP}] 🎉 All ${TOTAL_MODELS} models have been evaluated. Exiting monitor."
        break
    fi

    echo "[${TIMESTAMP}] ⏳ Waiting ${POLL_INTERVAL}s before next scan..."
    sleep ${POLL_INTERVAL}
done

# ======================== Done ========================

TOTAL_END=$(date +%s)
TOTAL_ELAPSED=$((TOTAL_END - TOTAL_START))
TOTAL_MIN=$((TOTAL_ELAPSED / 60))

echo ""
echo "=========================================="
echo "All evaluations complete!"
echo "Total time: ${TOTAL_ELAPSED}s (${TOTAL_MIN}min)"
echo "Success: ${TASK_SUCCESS}/${TASK_TOTAL}"
echo "Failed: ${TASK_FAILED}/${TASK_TOTAL}"
if [ -n "${FAILED_TASKS}" ]; then
    echo ""
    echo "Failed tasks:"
    printf '%s' "${FAILED_TASKS}"
fi
echo "Results saved to: ${OUTPUT_DIR}/"
echo "=========================================="

# ======================== Results Summary Table ========================

echo ""
echo "=========================================="
echo "Results Summary Table"
echo "=========================================="

# Collect all dataset names that have results
ALL_DATASETS=()
for ds_entry in "${DATASETS[@]}"; do
    ds_name=$(echo "${ds_entry}" | cut -d: -f1)
    ALL_DATASETS+=("${ds_name}")
done

# Build header
HEADER="| Model"
SEPARATOR="| :---"
for ds in "${ALL_DATASETS[@]}"; do
    HEADER="${HEADER} | ${ds}"
    SEPARATOR="${SEPARATOR} | :---:"
done
HEADER="${HEADER} | average |"
SEPARATOR="${SEPARATOR} | :---: |"

echo ""
echo "${HEADER}"
echo "${SEPARATOR}"

# Build rows: one per model
for model_entry in "${MODELS[@]}"; do
    MODEL_REF=$(echo "${model_entry}" | cut -d: -f1)
    MODEL_PATH=$(resolve_model_path "${MODEL_REF}")
    MODEL_NAME=$(get_model_name "${MODEL_PATH}")

    ROW="| ${MODEL_NAME}"
    scores=()
    valid_scores=0
    total_score=0
    for ds in "${ALL_DATASETS[@]}"; do
        metrics_file="${OUTPUT_DIR}/${MODEL_NAME}/${ds}/${METRICS_FILE:-metrics.json}"
        if [ -f "${metrics_file}" ]; then
            # Extract score from metrics.json using python for reliability
            score=$(python -c "
import json, sys
try:
    with open('${metrics_file}') as f:
        d = json.load(f)
    score_val = d['score']*100
    print(f'{score_val:.2f}')
    sys.exit(0)
except Exception as e:
    sys.exit(1)
" 2>/dev/null)
            if [ $? -eq 0 ]; then
                ROW="${ROW} | ${score}%"
                scores+=("${score}")
                total_score=$(awk "BEGIN {printf \"%.4f\", $total_score + $score}")
                valid_scores=$((valid_scores + 1))
            else
                ROW="${ROW} | ERR"
            fi
        else
            ROW="${ROW} | -"
        fi
    done
    
    # Calculate average if we have valid scores
    if [ $valid_scores -gt 0 ]; then
        average=$(awk "BEGIN {printf \"%.2f\", $total_score / $valid_scores}")
        ROW="${ROW} | ${average}% |"
    else
        ROW="${ROW} | - |"
    fi
    echo "${ROW}"
done

echo ""
echo "=========================================="

exit ${TASK_FAILED}
