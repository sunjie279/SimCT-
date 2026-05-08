#!/usr/bin/env python3
"""
Prepare LiveCodeBench-v6 dataset for evaluation.

LiveCodeBench (https://livecodebench.github.io/) is a contamination-free
benchmark for code generation. Version 6 covers problems released up to
2024-07-01.

This script downloads the dataset from HuggingFace and converts it to the
format expected by our evaluation pipeline.

Usage:
    # With internet access (or proxy):
    python evaluation/prepare_lcb_data.py

    # The dataset will be saved to:
    #   $DATA_PATH/ (set via DATA_PATH environment variable)
"""

import json
import os
import sys

DATASETS_DIR = os.environ.get("DATA_PATH", "./data")
OUTPUT_DIR = os.path.join(DATASETS_DIR, "live-code-bench-v6")


def main():
    print("=" * 60)
    print("Preparing LiveCodeBench-v6 dataset")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    problems_file = os.path.join(OUTPUT_DIR, "problems.jsonl")
    if os.path.exists(problems_file):
        with open(problems_file) as f:
            count = sum(1 for _ in f)
        print(f"✓ Dataset already exists: {problems_file} ({count} problems)")
        return

    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: 'datasets' package not installed.")
        sys.exit(1)

    print("Downloading LiveCodeBench dataset from HuggingFace...")
    print("  (This requires internet access)")

    try:
        # LiveCodeBench v6: problems up to 2024-07-01
        ds = load_dataset("livecodebench/code_generation_lite", version_tag="release_v6")
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print()
        print("If you are in an offline environment, you can:")
        print("  1. Download the dataset on a machine with internet access:")
        print("     python -c \"from datasets import load_dataset; "
              "ds = load_dataset('livecodebench/code_generation_lite', version_tag='release_v6'); "
              "ds.save_to_disk('/path/to/live-code-bench-v6')\"")
        print("  2. Copy the saved dataset to this server:")
        print(f"     rsync -a /path/to/live-code-bench-v6/ {OUTPUT_DIR}/")
        print("  3. Or manually create problems.jsonl with the following format:")
        print("     {\"question_id\": \"...\", \"question_content\": \"...\", \"test_cases\": [...]}")
        sys.exit(1)

    # Convert to JSONL format
    split_name = list(ds.keys())[0] if hasattr(ds, 'keys') else "test"
    data = ds[split_name] if hasattr(ds, 'keys') else ds

    print(f"Dataset loaded: {len(data)} problems")
    print(f"Columns: {data.column_names}")

    count = 0
    with open(problems_file, 'w', encoding='utf-8') as f:
        for i in range(len(data)):
            sample = data[i]
            # Normalize field names
            problem = {
                "question_id": sample.get("question_id", sample.get("task_id", f"lcb_{i}")),
                "question_content": sample.get("question_content", sample.get("description", "")),
                "test_cases": sample.get("test_cases", sample.get("public_test_cases", "[]")),
            }
            # Preserve all original fields
            for k, v in sample.items():
                if k not in problem:
                    problem[k] = v
            f.write(json.dumps(problem, ensure_ascii=False) + "\n")
            count += 1

    print(f"✓ Saved {count} problems to {problems_file}")

    # Also save as HF dataset for compatibility
    try:
        data.save_to_disk(os.path.join(OUTPUT_DIR, "dataset"))
        print(f"✓ Also saved HF dataset format to {OUTPUT_DIR}/dataset/")
    except Exception as e:
        print(f"  (HF dataset save skipped: {e})")

    print("Done!")


if __name__ == "__main__":
    main()
