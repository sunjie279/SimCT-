#!/usr/bin/env python3
"""
Build SFT warmup dataset from teacher responses.

This script:
1. Loads teacher response dataset (with multiple trajectories per question)
2. Loads the original dataset with ground truth labels and source info
3. Verifies response correctness using source-specific logic
4. Filters low-quality responses (too short, too long, repetitive)
5. Selects the shortest correct response per question
6. Outputs both ShareGPT format (for LLaMA-Factory) and HuggingFace Dataset format

Usage:
    # Build SFT warmup from Qwen2.5-7B-Instruct responses on 10k dataset
    python scripts/build_sft_warmup_dataset.py \\
        --response-dir /path/to/teacher_responses_10k_qwen2.5-7b \\
        --source-dataset /path/to/mixed_math_code_10k_with_source \\
        --output-dir /path/to/sft_warmup_10k_qwen2.5-7b \\
        --model-name Qwen2.5-7B-Instruct

    # Use existing teacher_answers (n=1, Qwen only) for quick test
    python scripts/build_sft_warmup_dataset.py \\
        --response-dir /path/to/teacher_answers_mix10k \\
        --source-dataset /path/to/mixed_math_code_10k_with_source \\
        --output-dir /path/to/sft_warmup_test \\
        --model-name Qwen2.5-7B-Instruct
"""

import argparse
import json
import os
import sys
import logging
import re
from collections import defaultdict
from typing import Dict, List, Any, Optional

from datasets import Dataset, load_from_disk
from tqdm import tqdm

# Add parent directory to path for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from answer_verifier import verify_response, get_source_type, is_math_source, is_code_source

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================
DATASET_BASE = os.environ.get("DATA_PATH", "./data")

# Quality filter thresholds
MIN_RESPONSE_LENGTH = 20          # Minimum response length in characters
MAX_MATH_RESPONSE_LENGTH = 4000   # Maximum math response length
MAX_CODE_RESPONSE_LENGTH = 4000   # Maximum code response length
MAX_REPETITION_RATIO = 0.30       # Maximum repetition ratio


# ============================================================================
# Quality filters
# ============================================================================

def compute_repetition_ratio(text: str, ngram_size: int = 10) -> float:
    """Compute the ratio of repeated n-grams in text.

    Returns a value between 0 and 1, where higher means more repetitive.
    """
    if not text or len(text) < ngram_size * 2:
        return 0.0

    words = text.split()
    if len(words) < ngram_size * 2:
        return 0.0

    ngrams = []
    for i in range(len(words) - ngram_size + 1):
        ngrams.append(tuple(words[i:i + ngram_size]))

    if not ngrams:
        return 0.0

    unique_ngrams = set(ngrams)
    return 1.0 - len(unique_ngrams) / len(ngrams)


def is_quality_response(response: str, source: str) -> tuple:
    """Check if a response passes quality filters.

    Returns:
        (passes: bool, reason: str)
    """
    if not response or not response.strip():
        return False, "empty"

    stripped = response.strip()
    length = len(stripped)

    # Too short
    if length < MIN_RESPONSE_LENGTH:
        return False, "too_short"

    # Too long (source-dependent)
    if is_math_source(source) and length > MAX_MATH_RESPONSE_LENGTH:
        return False, "too_long_math"
    if is_code_source(source) and length > MAX_CODE_RESPONSE_LENGTH:
        return False, "too_long_code"

    # Repetition check
    rep_ratio = compute_repetition_ratio(stripped)
    if rep_ratio > MAX_REPETITION_RATIO:
        return False, f"repetitive({rep_ratio:.2f})"

    return True, "ok"


# ============================================================================
# Main logic
# ============================================================================

def build_question_index(source_ds) -> Dict[int, Dict]:
    """Build an index from question_idx to source dataset sample.

    If the source dataset doesn't have question_idx, use row index.
    """
    index = {}
    for i in range(len(source_ds)):
        sample = source_ds[i]
        index[i] = {
            "label": sample.get("label", ""),
            "source": sample.get("source", "unknown"),
            "messages": sample["messages"],
        }
    return index


def build_sft_dataset(args):
    """Main function to build the SFT warmup dataset."""

    response_dirs = args.resolved_response_dirs

    # --- Validate inputs ---
    for rdir in response_dirs:
        if not os.path.exists(rdir):
            logger.error(f"Response directory not found: {rdir}")
            logger.error("Available directories in dataset base:")
            if os.path.exists(DATASET_BASE):
                for d in sorted(os.listdir(DATASET_BASE)):
                    if "teacher" in d or "response" in d:
                        logger.error(f"  {os.path.join(DATASET_BASE, d)}")
            sys.exit(1)

    if not os.path.exists(args.source_dataset):
        logger.error(f"Source dataset not found: {args.source_dataset}")
        logger.error("Available datasets:")
        if os.path.exists(DATASET_BASE):
            for d in sorted(os.listdir(DATASET_BASE)):
                if "with_source" in d:
                    logger.error(f"  {os.path.join(DATASET_BASE, d)}")
        sys.exit(1)

    # --- Load datasets (merge multiple response dirs if needed) ---
    from datasets import concatenate_datasets
    all_response_datasets = []
    for rdir in response_dirs:
        logger.info(f"Loading teacher responses from {rdir}")
        ds = load_from_disk(rdir)
        logger.info(f"  Loaded {len(ds)} records from {os.path.basename(rdir)}")
        all_response_datasets.append(ds)

    if len(all_response_datasets) == 1:
        response_ds = all_response_datasets[0]
    else:
        response_ds = concatenate_datasets(all_response_datasets)
        logger.info(f"  Merged {len(response_dirs)} directories -> {len(response_ds)} total records")
    logger.info(f"  Response dataset features: {list(response_ds.features.keys())}")

    logger.info(f"Loading source dataset from {args.source_dataset}")
    source_ds = load_from_disk(args.source_dataset)
    logger.info(f"  Source dataset: {len(source_ds)} records, features: {list(source_ds.features.keys())}")

    # Build question index from source dataset
    question_index = build_question_index(source_ds)
    logger.info(f"  Question index built: {len(question_index)} questions")

    # --- Group responses by question_idx ---
    logger.info("\nGrouping responses by question...")
    question_responses = defaultdict(list)

    has_question_idx = "question_idx" in response_ds.features
    has_source_field = "source" in response_ds.features

    for i in tqdm(range(len(response_ds)), desc="Loading responses"):
        sample = response_ds[i]
        if has_question_idx:
            q_idx = sample["question_idx"]
        else:
            # If no question_idx, infer from trajectory_id or use sequential index
            if "trajectory_id" in response_ds.features:
                # Assume responses are ordered: q0_t0, q0_t1, ..., q1_t0, q1_t1, ...
                traj_id = sample.get("trajectory_id", 0)
                n_traj = max(1, max(r.get("trajectory_id", 0) for r in [response_ds[j] for j in range(min(100, len(response_ds)))]) + 1)
                q_idx = i // n_traj
            else:
                q_idx = i

        response_text = sample.get("label", "")
        # Get source from response dataset or fall back to source dataset
        if has_source_field:
            source = sample.get("source", "unknown")
        else:
            source = question_index.get(q_idx, {}).get("source", "unknown")

        question_responses[q_idx].append({
            "response": response_text,
            "source": source,
            "messages": sample.get("messages", question_index.get(q_idx, {}).get("messages", [])),
        })

    logger.info(f"  Unique questions with responses: {len(question_responses)}")
    traj_counts = [len(v) for v in question_responses.values()]
    logger.info(f"  Trajectories per question: min={min(traj_counts)}, max={max(traj_counts)}, "
                f"avg={sum(traj_counts)/len(traj_counts):.1f}")

    # --- Verify and filter responses ---
    logger.info("\nVerifying and filtering responses...")

    # Statistics
    stats = {
        "total_responses": 0,
        "correct_responses": 0,
        "quality_passed": 0,
        "selected": 0,
        "by_source": defaultdict(lambda: {
            "total": 0, "correct": 0, "quality_passed": 0, "selected": 0,
            "total_length": 0, "selected_length": 0,
        }),
        "filter_reasons": defaultdict(int),
    }

    # For each question, find the best (shortest correct) response
    selected_records = {}  # q_idx -> best record

    for q_idx in tqdm(sorted(question_responses.keys()), desc="Filtering"):
        responses = question_responses[q_idx]
        source_info = question_index.get(q_idx, {})
        gold_label = source_info.get("label", "")
        source = source_info.get("source", responses[0].get("source", "unknown"))
        messages = source_info.get("messages", responses[0].get("messages", []))
        source_type = get_source_type(source)

        best_response = None
        best_length = float('inf')

        for resp_data in responses:
            response_text = resp_data["response"]
            stats["total_responses"] += 1
            stats["by_source"][source]["total"] += 1
            stats["by_source"][source]["total_length"] += len(response_text) if response_text else 0

            # Step 1: Verify correctness
            is_correct = verify_response(response_text, gold_label, source)
            if not is_correct:
                stats["filter_reasons"]["incorrect"] += 1
                continue

            stats["correct_responses"] += 1
            stats["by_source"][source]["correct"] += 1

            # Step 2: Quality filter
            passes, reason = is_quality_response(response_text, source)
            if not passes:
                stats["filter_reasons"][reason] += 1
                continue

            stats["quality_passed"] += 1
            stats["by_source"][source]["quality_passed"] += 1

            # Step 3: Select shortest
            resp_len = len(response_text.strip())
            if resp_len < best_length:
                best_length = resp_len
                best_response = {
                    "messages": messages,
                    "response": response_text.strip(),
                    "source": source,
                    "question_idx": q_idx,
                }

        if best_response is not None:
            selected_records[q_idx] = best_response
            stats["selected"] += 1
            stats["by_source"][source]["selected"] += 1
            stats["by_source"][source]["selected_length"] += len(best_response["response"])

    logger.info(f"\n  Total responses processed: {stats['total_responses']}")
    logger.info(f"  Correct responses: {stats['correct_responses']}")
    logger.info(f"  Quality passed: {stats['quality_passed']}")
    logger.info(f"  Selected (best per question): {stats['selected']}")

    # --- Balance math/code ratio ---
    # Count current distribution
    math_count = sum(1 for r in selected_records.values() if is_math_source(r["source"]))
    code_count = sum(1 for r in selected_records.values() if is_code_source(r["source"]))
    other_count = len(selected_records) - math_count - code_count

    logger.info(f"\n  Current distribution: math={math_count}, code={code_count}, other={other_count}")

    # We don't forcefully rebalance - just report the ratio
    # The original dataset is ~58:42 math:code, so we expect similar ratio
    total_mc = math_count + code_count
    if total_mc > 0:
        math_ratio = math_count / total_mc
        logger.info(f"  Math:Code ratio = {math_ratio:.1%}:{1-math_ratio:.1%} "
                     f"(target ~58:42)")

    # --- Build output datasets ---
    logger.info(f"\nBuilding output datasets...")
    os.makedirs(args.output_dir, exist_ok=True)

    # Sort by question_idx for deterministic output
    sorted_records = [selected_records[k] for k in sorted(selected_records.keys())]

    # 1. ShareGPT format (for LLaMA-Factory)
    sharegpt_data = []
    for record in sorted_records:
        # Extract user content from messages
        user_content = ""
        for msg in record["messages"]:
            if msg["role"] == "user":
                user_content = msg["content"]
                break

        sharegpt_data.append({
            "conversations": [
                {"from": "human", "value": user_content},
                {"from": "gpt", "value": record["response"]},
            ],
        })

    sharegpt_path = os.path.join(args.output_dir, "sft_warmup_sharegpt.json")
    with open(sharegpt_path, "w", encoding="utf-8") as f:
        json.dump(sharegpt_data, f, indent=2, ensure_ascii=False)
    logger.info(f"  ShareGPT format saved: {sharegpt_path} ({len(sharegpt_data)} records)")

    # 2. HuggingFace Dataset format
    hf_records = []
    for record in sorted_records:
        hf_records.append({
            "messages": record["messages"],
            "label": record["response"],
            "source": record["source"],
            "question_idx": record["question_idx"],
        })

    hf_ds = Dataset.from_list(hf_records)
    hf_ds.save_to_disk(args.output_dir)
    logger.info(f"  HuggingFace Dataset saved: {args.output_dir} ({len(hf_ds)} records)")

    # --- Print detailed statistics report ---
    logger.info(f"\n{'='*70}")
    logger.info(f"SFT Warmup Dataset Construction Report")
    logger.info(f"{'='*70}")
    logger.info(f"  Model: {args.model_name}")
    logger.info(f"  Response source(s): {', '.join(response_dirs)}")
    logger.info(f"  Source dataset: {args.source_dataset}")
    logger.info(f"  Output: {args.output_dir}")
    logger.info(f"{'='*70}")

    logger.info(f"\n  Overall Statistics:")
    logger.info(f"    Total responses: {stats['total_responses']}")
    logger.info(f"    Correct: {stats['correct_responses']} "
                f"({stats['correct_responses']/max(stats['total_responses'],1)*100:.1f}%)")
    logger.info(f"    Quality passed: {stats['quality_passed']} "
                f"({stats['quality_passed']/max(stats['total_responses'],1)*100:.1f}%)")
    logger.info(f"    Final selected: {stats['selected']} "
                f"({stats['selected']/max(len(question_responses),1)*100:.1f}% of questions)")

    logger.info(f"\n  Filter reasons:")
    for reason, count in sorted(stats["filter_reasons"].items(), key=lambda x: -x[1]):
        logger.info(f"    {reason}: {count}")

    logger.info(f"\n  Per-source breakdown:")
    logger.info(f"    {'Source':<25} {'Total':>8} {'Correct':>8} {'Quality':>8} {'Selected':>8} {'Avg Len (sel)':>14}")
    logger.info(f"    {'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*14}")
    for source in sorted(stats["by_source"].keys()):
        s = stats["by_source"][source]
        avg_sel_len = s["selected_length"] / max(s["selected"], 1)
        logger.info(f"    {source:<25} {s['total']:>8} {s['correct']:>8} {s['quality_passed']:>8} "
                     f"{s['selected']:>8} {avg_sel_len:>14.0f}")

    # Average response length comparison
    total_orig_len = sum(s["total_length"] for s in stats["by_source"].values())
    total_sel_len = sum(s["selected_length"] for s in stats["by_source"].values())
    avg_orig = total_orig_len / max(stats["total_responses"], 1)
    avg_sel = total_sel_len / max(stats["selected"], 1)
    logger.info(f"\n  Average response length:")
    logger.info(f"    Before filtering: {avg_orig:.0f} chars")
    logger.info(f"    After filtering:  {avg_sel:.0f} chars")
    logger.info(f"    Reduction: {(1 - avg_sel/max(avg_orig, 1))*100:.1f}%")

    logger.info(f"\n  Final dataset:")
    logger.info(f"    Size: {len(sorted_records)} records")
    logger.info(f"    Math: {math_count} ({math_count/max(len(sorted_records),1)*100:.1f}%)")
    logger.info(f"    Code: {code_count} ({code_count/max(len(sorted_records),1)*100:.1f}%)")
    if other_count > 0:
        logger.info(f"    Other: {other_count} ({other_count/max(len(sorted_records),1)*100:.1f}%)")
    logger.info(f"\n  Output files:")
    logger.info(f"    ShareGPT: {sharegpt_path}")
    logger.info(f"    HF Dataset: {args.output_dir}")
    logger.info(f"{'='*70}")

    # Save statistics as JSON
    stats_output = {
        "model": args.model_name,
        "response_dirs": response_dirs,
        "source_dataset": args.source_dataset,
        "total_responses": stats["total_responses"],
        "correct_responses": stats["correct_responses"],
        "quality_passed": stats["quality_passed"],
        "selected": stats["selected"],
        "math_count": math_count,
        "code_count": code_count,
        "avg_response_length_before": round(avg_orig, 1),
        "avg_response_length_after": round(avg_sel, 1),
        "filter_reasons": dict(stats["filter_reasons"]),
        "per_source": {
            source: dict(s) for source, s in stats["by_source"].items()
        },
    }
    stats_path = os.path.join(args.output_dir, "build_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats_output, f, indent=2, ensure_ascii=False)
    logger.info(f"\n  Statistics saved: {stats_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Build SFT warmup dataset from teacher responses",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # From multi-trajectory responses (8 per question)
    python scripts/build_sft_warmup_dataset.py \\
        --response-dir /path/to/teacher_responses_10k_qwen2.5-7b \\
        --source-dataset /path/to/mixed_math_code_10k_with_source \\
        --output-dir /path/to/sft_warmup_10k_qwen2.5-7b \\
        --model-name Qwen2.5-7B-Instruct

    # From existing single-trajectory teacher answers
    python scripts/build_sft_warmup_dataset.py \\
        --response-dir /path/to/teacher_answers_mix10k \\
        --source-dataset /path/to/mixed_math_code_10k_with_source \\
        --output-dir /path/to/sft_warmup_test \\
        --model-name Qwen2.5-7B-Instruct
        """,
    )
    parser.add_argument("--response-dir", type=str, default=None,
                        help="Path to teacher response dataset (HuggingFace Dataset format). "
                             "Can also be the base dir when using --auto-merge-shards.")
    parser.add_argument("--response-dirs", type=str, nargs="+", default=None,
                        help="Multiple response directories to merge (e.g. shard outputs). "
                             "Alternative to --response-dir.")
    parser.add_argument("--auto-merge-shards", action="store_true",
                        help="Auto-discover shard directories matching "
                             "'{response-dir}_shard*' pattern and merge them.")
    parser.add_argument("--source-dataset", type=str, required=True,
                        help="Path to original dataset with ground truth labels and source field")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for the SFT warmup dataset")
    parser.add_argument("--model-name", type=str, default="unknown",
                        help="Teacher model name (for reporting)")
    args = parser.parse_args()

    # Resolve response directories
    if args.response_dirs:
        # Explicit list of directories
        args.resolved_response_dirs = args.response_dirs
    elif args.response_dir and args.auto_merge_shards:
        # Auto-discover shard directories
        import glob
        base_dir = args.response_dir
        pattern = base_dir + "_shard*"
        shard_dirs = sorted(glob.glob(pattern))
        if not shard_dirs:
            logger.error(f"No shard directories found matching pattern: {pattern}")
            sys.exit(1)
        logger.info(f"Auto-discovered {len(shard_dirs)} shard directories:")
        for d in shard_dirs:
            logger.info(f"  {d}")
        args.resolved_response_dirs = shard_dirs
    elif args.response_dir:
        # Single directory (original behavior)
        args.resolved_response_dirs = [args.response_dir]
    else:
        parser.error("Must specify either --response-dir or --response-dirs")

    build_sft_dataset(args)


if __name__ == "__main__":
    main()
