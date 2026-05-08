# SimCT: Recovering Lost Supervision for Cross-Tokenizer On-Policy Distillation

This repository contains the code for reproducing the experiments in our paper *"SimCT: Recovering Lost Supervision for Cross-Tokenizer On-Policy Distillation"*.

---

## 📑 Table of Contents

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [Reproducing Experiments](#reproducing-experiments)
  - [Step 1: Data Preparation](#step-1-data-preparation)
  - [Step 2: Generate Teacher Responses](#step-2-generate-teacher-responses)
  - [Step 3: SFT Warmup Training](#step-3-sft-warmup-training)
  - [Step 4: Cross-Tokenizer On-Policy Distillation](#step-4-cross-tokenizer-on-policy-distillation)
  - [Step 5: Evaluation](#step-5-evaluation)
- [Acknowledgement](#acknowledgement)
- [Citation](#citation)

---

## Overview

**Experimental Setup:**

| Role | Model |
|------|-------|
| Teacher | Qwen2.5-7B-Instruct |
| Teacher | Phi-4-mini-instruct |
| Student | Gemma-2-2B-IT |
| Student | Phi-4-mini-instruct |

**Evaluation Benchmarks:** GSM8K, MATH-500, MBPP, LiveCodeBench-v6

---

## Environment Setup

Our experiments are built on top of [KDFlow](https://github.com/songmzhang/KDFlow). Please install the dependencies:

```bash
git clone https://github.com/sunjie279/SimCT-.git
cd SimCT_
pip install -e ./
pip install flash_attn==2.8.3 --no-build-isolation
```

Then set the following environment variables:

```bash
export MODEL_PATH="./models"         # Directory containing model weights
export DATA_PATH="./data"            # Directory for datasets
export OUTPUT_PATH="./output/ckpts"  # Directory for checkpoints
```

Required model weights (download from HuggingFace):
- `$MODEL_PATH/Qwen2.5-7B-Instruct`
- `$MODEL_PATH/Phi-4-mini-instruct`
- `$MODEL_PATH/gemma-2-2b-it`

---

## Reproducing Experiments

The full pipeline consists of 5 steps:

```
Data Preparation → Generate Teacher Responses → SFT Warmup → Distillation Training → Evaluation
```

### Step 1: Data Preparation

We construct a 10K mixed math+code training dataset from multiple sources:

```bash
# Download raw datasets from HuggingFace
python scripts/data/download_datasets.py

# Prepare individual datasets
python scripts/data/prepare_gsm8k.py
python scripts/data/prepare_orca_math.py

# Build the 10K mixed dataset
python scripts/data/prepare_mixed_math_code.py
```

This produces:
- `$DATA_PATH/mixed_math_code_10k/` — Training prompts
- `$DATA_PATH/mixed_math_code_10k_with_source/` — Training prompts with source labels

### Step 2: Generate Teacher Responses

Generate 8 trajectories per question for each teacher model using SGLang:

```bash
# Qwen2.5-7B-Instruct
bash scripts/sft/run_generate_responses_10k_qwen.sh

# Phi-4-mini-instruct
bash scripts/sft/run_generate_responses_10k_phi4.sh
```

Each script starts an SGLang server (DP=8), generates responses (temperature=0.6, top_p=0.95), and saves to `$DATA_PATH/teacher_responses_10k_<model_tag>/`.

### Step 3: SFT Warmup Training

Before distillation, the student needs an SFT warmup to establish basic instruction-following capability.

#### 3.1 Build SFT Dataset

Select the shortest correct response per question from teacher trajectories:

```bash
# From Qwen responses
bash scripts/sft/run_build_sft_10k_qwen.sh

# From Phi-4 responses
bash scripts/sft/run_build_sft_10k_phi4.sh
```

#### 3.2 Run SFT Training

We use [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) for SFT:

```bash
# Gemma-2-2B-IT with Qwen teacher data
bash scripts/sft/run_gemma2_sft_warmup_10k_qwen.sh

# Gemma-2-2B-IT with Phi-4 teacher data
bash scripts/sft/run_gemma2_sft_warmup_10k_phi4.sh

# Phi-4-mini with Qwen teacher data
bash scripts/sft/run_phi4_sft_warmup_10k_qwen.sh
```

### Step 4: Cross-Tokenizer On-Policy Distillation

Run distillation training for each teacher→student pair:

```bash
# Qwen2.5-7B → Gemma-2-2B
bash scripts/ctopd/qwen25_gemma2_span_mix10k_lr5e-7.sh

# Qwen2.5-7B → Phi-4-mini
bash scripts/ctopd/qwen25_phi4_span_mix10k_lr5e-7.sh

# Phi-4-mini → Gemma-2-2B
bash scripts/ctopd/phi4_gemma2_span_mix10k_lr5e-7.sh
```


### Step 5: Evaluation

Evaluate on GSM8K, MATH-500, MBPP, and LiveCodeBench-v6:

```bash
# Prepare LiveCodeBench data (one-time)
python scripts/evaluation/prepare_lcb_data.py

# Run all evaluations
bash scripts/evaluation/eval_all_monitor.sh
```

Or evaluate a single checkpoint:

```bash
python scripts/evaluation/evaluation.py \
    --model_path $MODEL_PATH/your-checkpoint \
    --dataset gsm8k \
    --base_url http://127.0.0.1:30000 \
    --temperature 0.6 \
    --top_p 0.95 \
    --n 1 \
    --max_tokens 4096
```

Supported datasets: `gsm8k`, `math500`, `mbpp`, `live-code-bench-v6`

---

## Acknowledgement

This codebase is built on top of [KDFlow](https://github.com/songmzhang/KDFlow), a user-friendly and efficient framework for LLM knowledge distillation. We sincerely thank the KDFlow team for their excellent work.

---

## Citation

```bibtex
@article{simct2026,
      title={SimCT: Recovering Lost Supervision for Cross-Tokenizer On-Policy Distillation},
      author={TODO},
      year={2026},
}
```
