#!/usr/bin/env python3
"""
"""Download all HuggingFace datasets needed for the 10k mixed dataset.
Run this script once in an environment with internet access.

Saves to: $DATA_PATH/ (set via environment variable, default: ./data)
"""
import os
from datasets import load_dataset

OUTPUT_BASE = os.environ.get("DATA_PATH", "./data")
HF_CACHE = os.path.join(OUTPUT_BASE, ".hf_cache")

DATASETS = [
    {
        "name": "lighteval/MATH-Hard",
        "local_name": "math_hard_raw",
        "split": "train",
        "config": None,
        "trust_remote_code": True,
    },
    {
        "name": "lighteval/MATH-Hard",
        "local_name": "math500_test",
        "split": "test",
        "config": None,
        "trust_remote_code": True,
    },
    {
        "name": "nvidia/OpenMathInstruct-1",
        "local_name": "open_math_instruct_raw",
        "split": "train",
        "config": None,
        "trust_remote_code": True,
    },
    {
        "name": "nvidia/OpenCodeInstruct",
        "local_name": "open_code_instruct_raw",
        "split": "train",
        "config": None,
        "trust_remote_code": True,
    },
    {
        "name": "KodCode/KodCode",
        "local_name": "kodcode_raw",
        "split": "train",
        "config": None,
        "trust_remote_code": True,
    },
    {
        "name": "deepmind/code_contests",
        "local_name": "code_contests_raw",
        "split": "train",
        "config": None,
        "trust_remote_code": True,
    },
]


def main():
    os.makedirs(HF_CACHE, exist_ok=True)

    for info in DATASETS:
        hf_name = info["name"]
        local_dir = os.path.join(OUTPUT_BASE, info["local_name"])

        if os.path.exists(local_dir) and os.listdir(local_dir):
            print(f"[SKIP] {hf_name} -> {local_dir} (already exists)")
            continue

        print(f"\n{'='*60}")
        print(f"[DOWNLOAD] {hf_name} (split={info['split']})")
        print(f"  -> {local_dir}")
        print(f"{'='*60}")

        try:
            kwargs = {
                "split": info["split"],
                "cache_dir": HF_CACHE,
                "trust_remote_code": info.get("trust_remote_code", False),
            }
            if info["config"]:
                ds = load_dataset(hf_name, info["config"], **kwargs)
            else:
                ds = load_dataset(hf_name, **kwargs)

            print(f"  Loaded: {len(ds)} samples")
            print(f"  Features: {list(ds.features.keys())}")

            os.makedirs(local_dir, exist_ok=True)
            ds.save_to_disk(local_dir)
            print(f"  ✓ Saved to {local_dir}")

        except Exception as e:
            print(f"  ✗ FAILED: {e}")

    # Also save OpenMathInstruct-1 from cache if already downloaded
    # (This is now handled in the main DATASETS list above)

    print("\n" + "="*60)
    print("All downloads complete! Summary:")
    print("="*60)
    for info in DATASETS:
        local_dir = os.path.join(OUTPUT_BASE, info["local_name"])
        exists = os.path.exists(local_dir) and os.listdir(local_dir)
        status = "✓" if exists else "✗"
        print(f"  {status} {info['name']} ({info['split']}) -> {info['local_name']}")
    print("="*60)


if __name__ == "__main__":
    main()
