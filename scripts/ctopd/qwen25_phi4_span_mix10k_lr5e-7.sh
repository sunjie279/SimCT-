#!/bin/bash
# Cross-Tokenizer On-Policy Distillation on Mixed Math+Code 10K
# Teacher: Qwen2.5-7B-Instruct  ->  Student: phi-4-mini (SFT warmup)
# Algorithm: SpanCTKD (Span-based Cross-Tokenizer KD)
# Learning Rate: 5e-7, Epochs: 1

set -e
set -x

# ============ Path Configuration ============
MODEL_PATH=${MODEL_PATH:-"./models"}
DATA_PATH=${DATA_PATH:-"./data"}
OUTPUT_PATH=${OUTPUT_PATH:-"./output/ckpts"}

# ============ Sync student checkpoint to local SSD ============
LOCAL_STUDENT_DIR=~/workspace/models/phi4-mini-sft-warmup-10k-qwen-lr2e-6
REMOTE_STUDENT_DIR=${OUTPUT_PATH}/phi4-mini-sft-warmup-10k-qwen-lr2e-6
if [ ! -d "${LOCAL_STUDENT_DIR}/checkpoint-40" ]; then
    echo "Syncing student checkpoint to local SSD..."
    mkdir -p "${LOCAL_STUDENT_DIR}"
    rsync -ah --progress "${REMOTE_STUDENT_DIR}/checkpoint-40/" "${LOCAL_STUDENT_DIR}/checkpoint-40/"
fi

# ============ TrainingArguments ============
OPTS=""
OPTS+=" --num_nodes 1"
OPTS+=" --num_gpus_per_node 8"
OPTS+=" --backend fsdp2"
OPTS+=" --train_batch_size 64"
OPTS+=" --micro_train_batch_size 1"
OPTS+=" --learning_rate 5e-7"
OPTS+=" --lr_warmup_ratio 0.05"
OPTS+=" --num_epochs 1"
OPTS+=" --save_path ${OUTPUT_PATH}/qwen25-phi4-span-mix10k-5e-7"
OPTS+=" --bf16 True"
OPTS+=" --gradient_checkpointing True"
OPTS+=" --enable_sleep True"

# ============ ModelArguments ============
OPTS+=" --student_name_or_path ${LOCAL_STUDENT_DIR}/checkpoint-40"
OPTS+=" --teacher_name_or_path ${MODEL_PATH}/Qwen2.5-7B-Instruct"
OPTS+=" --enable_thinking False"

# ============ RolloutArguments (On-Policy) ============
OPTS+=" --rollout_batch_size 64"
OPTS+=" --rollout_num_engines 8"
OPTS+=" --rollout_tp_size 1"
OPTS+=" --rollout_mem_fraction_static 0.6"
OPTS+=" --n_samples_per_prompt 1"
OPTS+=" --generate_max_len 4096"
OPTS+=" --temperature 0.6"

# ============ DataArguments ============
OPTS+=" --train_dataset_path ${DATA_PATH}/mixed_math_code_10k"
OPTS+=" --max_len 8192"
OPTS+=" --input_key messages"
OPTS+=" --apply_chat_template True"
OPTS+=" --preprocess_num_workers 8"
OPTS+=" --packing_samples True"

# ============ DistillationArguments ============
OPTS+=" --kd_ratio 1.0"
OPTS+=" --kd_loss_fn rkl"
OPTS+=" --kd_algorithm span_ctkd"
OPTS+=" --teacher_dp_size 8"
OPTS+=" --teacher_tp_size 1"
OPTS+=" --teacher_mem_fraction_static 0.5"
OPTS+=" --teacher_context_length 32768"

# ============ LoggingArguments ============
OPTS+=" --logging_steps 5"
OPTS+=" --save_steps 20"
OPTS+=" --use_wandb False"

export SGLANG_DISABLE_CUDNN_CHECK=1 
python -m kdflow.cli.train_kd_on_policy $OPTS
