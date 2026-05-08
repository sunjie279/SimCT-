#!/usr/bin/env python3
"""
Prepare Orca-Math 200K dataset for KDFlow training.

Downloads microsoft/orca-math-word-problems-200k and converts it to the
same format as gsm8k_for_kdflow:
  - messages: list of dicts with 'role' and 'content' (only user message, since on-policy KD generates its own responses)
  - label: the ground truth answer (kept for reference, not used in on-policy training)

Output: $DATA_PATH/orca_math_for_kdflow (set via environment variable, default: ./data)
"""

import os
from datasets import load_dataset, Dataset

# Output path
DATA_PATH = os.environ.get("DATA_PATH", "./data")
OUTPUT_DIR = os.path.join(DATA_PATH, "orca_math_for_kdflow")


def convert_to_messages_format(example):
    """Convert Orca-Math format to KDFlow messages format.
    
    Orca-Math has: question, answer
    KDFlow needs: messages (list of {role, content}), label
    
    For on-policy distillation, we only need the user prompt.
    The teacher model will generate the response during training.
    """
    messages = [
        {"role": "user", "content": example["question"]}
    ]
    return {
        "messages": messages,
        "label": example["answer"]
    }


def main():
    print("=" * 60)
    print("Preparing Orca-Math 200K dataset for KDFlow")
    print("=" * 60)

    # Download dataset
    print("\n[1/3] Downloading microsoft/orca-math-word-problems-200k ...")
    ds = load_dataset("microsoft/orca-math-word-problems-200k", split="train")
    print(f"  Downloaded {len(ds)} examples")
    print(f"  Features: {ds.features}")
    print(f"  Sample: {ds[0]}")

    # Convert format
    print("\n[2/3] Converting to KDFlow messages format ...")
    converted = ds.map(
        convert_to_messages_format,
        remove_columns=ds.column_names,
        num_proc=8,
        desc="Converting to messages format"
    )
    print(f"  Converted {len(converted)} examples")
    print(f"  New features: {converted.features}")
    print(f"  Sample: {converted[0]}")

    # Save
    print(f"\n[3/3] Saving to {OUTPUT_DIR} ...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    converted.save_to_disk(OUTPUT_DIR)
    print(f"  ✓ Saved {len(converted)} examples to {OUTPUT_DIR}")

    # Verify
    print("\n[Verify] Loading saved dataset ...")
    from datasets import load_from_disk
    verify_ds = load_from_disk(OUTPUT_DIR)
    print(f"  Loaded {len(verify_ds)} examples")
    print(f"  Sample[0]: {verify_ds[0]}")
    print(f"  Sample[-1]: {verify_ds[-1]}")
    print("\n✓ Done!")


if __name__ == "__main__":
    main()
