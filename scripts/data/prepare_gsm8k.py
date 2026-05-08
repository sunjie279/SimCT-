"""
Prepare GSM8K dataset for KDFlow On-Policy KD training.

Converts GSM8K's {question, answer} format into OpenAI messages format
that KDFlow's PromptDataset can consume.

Usage:
    python scripts/prepare_gsm8k.py

Output:
    $DATA_PATH/gsm8k_for_kdflow/ (set via environment variable, default: ./data)
"""

import os
from datasets import load_from_disk, Dataset

DATA_PATH = os.environ.get("DATA_PATH", "./data")
INPUT_PATH = os.path.join(DATA_PATH, "gsm8k/train")
OUTPUT_PATH = os.path.join(DATA_PATH, "gsm8k_for_kdflow")


def convert_sample(sample):
    """Convert a GSM8K sample to OpenAI messages format."""
    messages = [{"role": "user", "content": sample["question"]}]
    # Extract the final numeric answer after ####
    answer = sample["answer"]
    final_answer = answer.split("####")[-1].strip() if "####" in answer else answer
    return {
        "messages": messages,
        "label": final_answer,
    }


def main():
    print(f"Loading GSM8K from: {INPUT_PATH}")
    dataset = load_from_disk(INPUT_PATH)
    print(f"Original dataset size: {len(dataset)}")
    print(f"Original columns: {dataset.column_names}")
    print(f"Sample: {dataset[0]}")

    print("\nConverting to OpenAI messages format...")
    converted = dataset.map(
        convert_sample,
        remove_columns=dataset.column_names,
        num_proc=4,
        desc="Converting GSM8K",
    )

    print(f"\nConverted columns: {converted.column_names}")
    print(f"Sample: {converted[0]}")

    print(f"\nSaving to: {OUTPUT_PATH}")
    converted.save_to_disk(OUTPUT_PATH)
    print("Done!")


if __name__ == "__main__":
    main()
