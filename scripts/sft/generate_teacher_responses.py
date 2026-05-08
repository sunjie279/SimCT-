#!/usr/bin/env python3
"""
Generate teacher responses for SFT warmup dataset construction.

Generalized script that supports multiple teacher models and datasets.
For each question, generates N trajectories with configurable sampling params.

Output: HuggingFace Dataset saved to disk.
Format: {"messages": [{"role": "user", "content": "..."}],
         "label": "<teacher_generated_answer>",
         "source": "<dataset_source>",
         "question_idx": <original_index>,
         "trajectory_id": <0..N-1>}

Usage:
    # Step 1: Start SGLang server separately
    SGLANG_DISABLE_CUDNN_CHECK=1 python -m sglang.launch_server \
        --model-path $MODEL_PATH/Qwen2.5-7B-Instruct \        --dp-size 8 --tp-size 1 --port 30000 --mem-fraction-static 0.85 \\
        --trust-remote-code

    # Step 2: Run this script
    # Example: Qwen2.5-7B-Instruct on 10k dataset
    python scripts/generate_teacher_responses.py \\
        --model-name Qwen2.5-7B-Instruct \\
        --dataset-path /path/to/mixed_math_code_10k_with_source \\
        --dataset-tag 10k \\
        --output-base /path/to/dataset \\
        --n-trajectories 8

    # Example: Phi-4-mini-instruct on 20k dataset
    python scripts/generate_teacher_responses.py \\
        --model-name Phi-4-mini-instruct \\
        --dataset-path /path/to/mixed_math_code_20k_with_source \\
        --dataset-tag 20k \\
        --output-base /path/to/dataset \\
        --n-trajectories 8 \\
        --max-prompt-chars 45000
"""

import argparse
import asyncio
import aiohttp
import json
import os
import sys
import time
import logging
from typing import List, Dict, Any

from datasets import Dataset, load_from_disk
from tqdm.asyncio import tqdm_asyncio

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# Default Configuration
# ============================================================================
DEFAULT_BASE_URL = "http://127.0.0.1:30000"
DEFAULT_TEMPERATURE = 0.6
DEFAULT_TOP_P = 0.95
DEFAULT_N_TRAJECTORIES = 8
DEFAULT_MAX_CONCURRENT = 256
DEFAULT_MAX_NEW_TOKENS = 4096
DEFAULT_BATCH_SIZE = 5000

# Model-specific max prompt chars (approximate token-to-char ratio ~3.2)
MODEL_MAX_PROMPT_CHARS = {
    "Qwen2.5-7B-Instruct": 90000,      # ~32k ctx
    "Phi-4-mini-instruct": 45000,       # ~16k ctx
}

OUTPUT_BASE = os.environ.get("DATA_PATH", "./data")


# ============================================================================
# SGLang Async Client (reused from existing codebase)
# ============================================================================
class SGLangClient:
    """Async SGLang client for high-throughput inference."""

    def __init__(self, base_url: str = DEFAULT_BASE_URL, max_concurrent: int = 256, timeout: int = 300):
        self.base_url = base_url.rstrip("/")
        self.chat_url = f"{self.base_url}/v1/chat/completions"
        self.max_concurrent = max_concurrent
        self.timeout = timeout

    async def _post(self, session, url, payload, semaphore):
        max_retries = 3
        for attempt in range(max_retries):
            async with semaphore:
                try:
                    async with session.post(
                        url, json=payload,
                        timeout=aiohttp.ClientTimeout(total=self.timeout),
                    ) as resp:
                        if resp.status != 200:
                            text = await resp.text()
                            logger.warning(f"Request failed (HTTP {resp.status}): {text[:200]}")
                            if attempt < max_retries - 1:
                                await asyncio.sleep(2 ** attempt)
                                continue
                            return {"error": text, "choices": []}
                        return await resp.json()
                except Exception as e:
                    logger.warning(f"Request error (attempt {attempt+1}): {e}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    return {"error": str(e), "choices": []}

    def chat_batch(
        self,
        messages_list: List[List[Dict]],
        model: str = "default",
        max_new_tokens: int = 2048,
        temperature: float = 0.6,
        top_p: float = 0.95,
        n: int = 1,
        show_progress: bool = True,
        desc: str = "Generating",
    ) -> List[List[str]]:
        """Send batch chat requests. Returns list of list of generated texts (n per request)."""
        return asyncio.run(self._chat_batch_async(
            messages_list, model, max_new_tokens, temperature, top_p, n, show_progress, desc
        ))

    async def _chat_batch_async(self, messages_list, model, max_new_tokens, temperature, top_p, n, show_progress, desc):
        semaphore = asyncio.Semaphore(self.max_concurrent)
        connector = aiohttp.TCPConnector(limit=self.max_concurrent + 50)
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = []
            for messages in messages_list:
                payload = {
                    "model": model,
                    "messages": messages,
                    "max_tokens": max_new_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "n": n,
                }
                tasks.append(self._post(session, self.chat_url, payload, semaphore))

            if show_progress:
                results = await tqdm_asyncio.gather(*tasks, desc=desc)
            else:
                results = await asyncio.gather(*tasks)

        outputs = []
        for r in results:
            choices = r.get("choices", [])
            texts = []
            for c in choices:
                msg = c.get("message", {})
                texts.append(msg.get("content", ""))
            # Pad if fewer than n
            while len(texts) < n:
                texts.append("")
            outputs.append(texts)
        return outputs


def wait_for_server(base_url: str, timeout: int = 60):
    """Wait for SGLang server to be ready."""
    import urllib.request
    health_url = f"{base_url}/health"
    start = time.time()
    while time.time() - start < timeout:
        try:
            req = urllib.request.Request(health_url)
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    logger.info("SGLang server is ready!")
                    return True
        except Exception:
            pass
        time.sleep(3)
    raise TimeoutError(f"Server not ready after {timeout}s")


def make_model_tag(model_name: str) -> str:
    """Convert model name to a short tag for directory naming.

    Examples:
        Qwen2.5-7B-Instruct -> qwen2.5-7b
        Phi-4-mini-instruct -> phi-4-mini
    """
    name = model_name.lower()
    # Remove common suffixes
    for suffix in ["-instruct", "-it", "-chat"]:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


def main():
    parser = argparse.ArgumentParser(
        description="Generate teacher responses for SFT warmup dataset construction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Qwen2.5-7B-Instruct on 10k dataset (8 trajectories)
    python scripts/generate_teacher_responses.py \\
        --model-name Qwen2.5-7B-Instruct \\
        --dataset-path /path/to/mixed_math_code_10k_with_source \\
        --dataset-tag 10k

    # Phi-4-mini-instruct on 20k dataset
    python scripts/generate_teacher_responses.py \\
        --model-name Phi-4-mini-instruct \\
        --dataset-path /path/to/mixed_math_code_20k_with_source \\
        --dataset-tag 20k \\
        --max-prompt-chars 45000
        """,
    )
    parser.add_argument("--model-name", type=str, required=True,
                        help="Teacher model name (e.g. Qwen2.5-7B-Instruct, Phi-4-mini-instruct)")
    parser.add_argument("--dataset-path", type=str, required=True,
                        help="Path to the source dataset (with_source version preferred)")
    parser.add_argument("--dataset-tag", type=str, required=True,
                        help="Short tag for the dataset (e.g. 10k, 20k), used in output dir naming")
    parser.add_argument("--output-base", type=str, default=OUTPUT_BASE,
                        help=f"Base output directory (default: {OUTPUT_BASE})")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override full output directory path (ignores --output-base and auto-naming)")
    parser.add_argument("--base-url", type=str, default=DEFAULT_BASE_URL,
                        help="SGLang server base URL")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument("--n-trajectories", type=int, default=DEFAULT_N_TRAJECTORIES,
                        help="Number of trajectories per question (default: 8)")
    parser.add_argument("--max-concurrent", type=int, default=DEFAULT_MAX_CONCURRENT)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--max-prompt-chars", type=int, default=None,
                        help="Skip prompts longer than this (auto-detected from model name if not set)")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help="Process questions in batches of this size")
    parser.add_argument("--shard-id", type=int, default=None,
                        help="Shard index for distributed generation (0-based). "
                             "When set, only processes a subset of the dataset.")
    parser.add_argument("--num-shards", type=int, default=None,
                        help="Total number of shards for distributed generation. "
                             "Must be set together with --shard-id.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last saved batch")
    args = parser.parse_args()

    # Validate shard arguments
    if (args.shard_id is None) != (args.num_shards is None):
        parser.error("--shard-id and --num-shards must be set together")
    if args.shard_id is not None:
        if args.shard_id < 0 or args.shard_id >= args.num_shards:
            parser.error(f"--shard-id must be in [0, {args.num_shards - 1}], got {args.shard_id}")

    # Auto-detect max-prompt-chars from model name
    if args.max_prompt_chars is None:
        args.max_prompt_chars = MODEL_MAX_PROMPT_CHARS.get(args.model_name, 90000)
        logger.info(f"Auto-detected max-prompt-chars={args.max_prompt_chars} for model {args.model_name}")

    # Build output directory
    model_tag = make_model_tag(args.model_name)
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(args.output_base, f"teacher_responses_{args.dataset_tag}_{model_tag}")

    # Append shard suffix if running in sharded mode
    if args.shard_id is not None:
        output_dir = output_dir + f"_shard{args.shard_id}of{args.num_shards}"
    logger.info(f"Output directory: {output_dir}")

    # Check server
    logger.info(f"Connecting to SGLang server at {args.base_url}")
    try:
        wait_for_server(args.base_url, timeout=60)
    except TimeoutError:
        logger.error(f"No SGLang server found at {args.base_url}.")
        logger.error("Please start the server first. Example:")
        logger.error(f"  SGLANG_DISABLE_CUDNN_CHECK=1 python -m sglang.launch_server \\")
        logger.error(f"    --model-path $MODEL_PATH/{args.model_name} \\")
        logger.error(f"    --dp-size 8 --tp-size 1 --port 30000 --mem-fraction-static 0.85")
        sys.exit(1)

    client = SGLangClient(
        base_url=args.base_url,
        max_concurrent=args.max_concurrent,
        timeout=300,
    )

    # Load dataset
    logger.info(f"Loading dataset from {args.dataset_path}")
    if not os.path.exists(args.dataset_path):
        logger.error(f"Dataset path not found: {args.dataset_path}")
        logger.error("Available datasets in output base:")
        base = os.path.dirname(args.dataset_path)
        if os.path.exists(base):
            for d in sorted(os.listdir(base)):
                logger.error(f"  {os.path.join(base, d)}")
        sys.exit(1)

    ds = load_from_disk(args.dataset_path)
    total_questions = len(ds)
    logger.info(f"  Total questions: {total_questions}")
    logger.info(f"  Features: {ds.features}")

    # Check if dataset has 'source' field
    has_source = "source" in ds.features
    if not has_source:
        logger.warning("Dataset does not have 'source' field. Will use 'unknown' as source.")

    # Filter out prompts that exceed model context length
    skipped_indices = set()
    for i in range(total_questions):
        prompt_len = sum(len(m["content"]) for m in ds[i]["messages"])
        if prompt_len > args.max_prompt_chars:
            skipped_indices.add(i)
    valid_indices = [i for i in range(total_questions) if i not in skipped_indices]

    logger.info(f"\n{'='*60}")
    logger.info(f"Teacher Response Generation")
    logger.info(f"{'='*60}")
    logger.info(f"  Model: {args.model_name} (tag: {model_tag})")
    logger.info(f"  Dataset: {args.dataset_path}")
    logger.info(f"  Dataset tag: {args.dataset_tag}")
    logger.info(f"  Skipped {len(skipped_indices)} questions with prompt > {args.max_prompt_chars} chars")
    logger.info(f"  Valid questions: {len(valid_indices)}")

    # Apply sharding if requested
    if args.shard_id is not None:
        shard_size = (len(valid_indices) + args.num_shards - 1) // args.num_shards
        shard_start = args.shard_id * shard_size
        shard_end = min(shard_start + shard_size, len(valid_indices))
        valid_indices = valid_indices[shard_start:shard_end]
        logger.info(f"  Shard {args.shard_id}/{args.num_shards}: processing indices [{shard_start}, {shard_end}) "
                    f"= {len(valid_indices)} questions")

    # Build index mapping: valid_idx -> original_idx
    original_indices = valid_indices[:]

    # Select only valid samples
    ds = ds.select(valid_indices)
    total_questions = len(ds)

    logger.info(f"  Questions to process: {total_questions}")
    logger.info(f"  Trajectories per question: {args.n_trajectories}")
    logger.info(f"  Temperature: {args.temperature}, Top-p: {args.top_p}")
    logger.info(f"  Max new tokens: {args.max_new_tokens}")
    logger.info(f"  Max prompt chars: {args.max_prompt_chars}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Total records to generate: {total_questions * args.n_trajectories}")
    logger.info(f"  Output: {output_dir}")
    logger.info(f"{'='*60}\n")

    os.makedirs(output_dir, exist_ok=True)
    temp_dir = os.path.join(output_dir, "temp_batches")
    os.makedirs(temp_dir, exist_ok=True)

    # Check for resume
    completed_batches = set()
    if args.resume:
        for f in os.listdir(temp_dir):
            if f.startswith("batch_") and f.endswith(".jsonl"):
                batch_idx = int(f.split("_")[1].split(".")[0])
                completed_batches.add(batch_idx)
        if completed_batches:
            logger.info(f"  Resuming: found {len(completed_batches)} completed batches")

    # Process in batches
    num_batches = (total_questions + args.batch_size - 1) // args.batch_size
    start_time = time.time()

    for batch_idx in range(num_batches):
        if batch_idx in completed_batches:
            logger.info(f"  Skipping batch {batch_idx+1}/{num_batches} (already completed)")
            continue

        batch_start = batch_idx * args.batch_size
        batch_end = min(batch_start + args.batch_size, total_questions)
        batch_data = ds.select(range(batch_start, batch_end))

        logger.info(f"\n--- Batch {batch_idx+1}/{num_batches} (questions {batch_start}-{batch_end-1}) ---")

        # Build messages list
        messages_list = []
        for i in range(len(batch_data)):
            sample = batch_data[i]
            messages_list.append(sample["messages"])

        # Generate
        all_outputs = client.chat_batch(
            messages_list=messages_list,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            n=args.n_trajectories,
            show_progress=True,
            desc=f"Batch {batch_idx+1}/{num_batches}",
        )

        # Save batch results to temp file
        batch_file = os.path.join(temp_dir, f"batch_{batch_idx:04d}.jsonl")
        with open(batch_file, "w") as f:
            for i in range(len(batch_data)):
                global_idx = batch_start + i
                original_idx = original_indices[global_idx]
                sample = batch_data[i]
                trajectories = all_outputs[i]
                source = sample.get("source", "unknown") if has_source else "unknown"
                for traj_id, output_text in enumerate(trajectories):
                    record = {
                        "messages": sample["messages"],
                        "label": output_text,
                        "source": source,
                        "question_idx": original_idx,
                        "trajectory_id": traj_id,
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

        elapsed = time.time() - start_time
        questions_done = batch_end
        questions_per_sec = questions_done / elapsed if elapsed > 0 else 0
        eta = (total_questions - questions_done) / questions_per_sec if questions_per_sec > 0 else 0
        logger.info(f"  Batch {batch_idx+1} done. {questions_done}/{total_questions} questions. "
                     f"Speed: {questions_per_sec:.1f} q/s. ETA: {eta/60:.1f} min")

    # Merge all batches into final dataset
    logger.info(f"\n{'='*60}")
    logger.info(f"Merging all batches into final dataset...")
    logger.info(f"{'='*60}")

    all_records = []
    for batch_idx in range(num_batches):
        batch_file = os.path.join(temp_dir, f"batch_{batch_idx:04d}.jsonl")
        with open(batch_file, "r") as f:
            for line in f:
                if line.strip():
                    all_records.append(json.loads(line))

    logger.info(f"  Total records: {len(all_records)}")

    # Save as HuggingFace Dataset
    final_ds = Dataset.from_list(all_records)
    final_ds.save_to_disk(output_dir)
    logger.info(f"  Saved to {output_dir}")
    logger.info(f"  Features: {final_ds.features}")

    # Save generation config
    config = {
        "source_dataset": args.dataset_path,
        "dataset_tag": args.dataset_tag,
        "model": args.model_name,
        "model_tag": model_tag,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "n_trajectories": args.n_trajectories,
        "max_new_tokens": args.max_new_tokens,
        "max_prompt_chars": args.max_prompt_chars,
        "total_questions": total_questions,
        "skipped_questions": len(skipped_indices),
        "total_records": len(all_records),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    config_path = os.path.join(output_dir, "generation_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    # Print summary
    total_time = time.time() - start_time
    logger.info(f"\n{'='*60}")
    logger.info(f"Generation Complete!")
    logger.info(f"  Model: {args.model_name}")
    logger.info(f"  Dataset: {args.dataset_tag}")
    logger.info(f"  Total questions: {total_questions}")
    logger.info(f"  Total records: {len(all_records)}")
    logger.info(f"  Total time: {total_time/60:.1f} min")
    logger.info(f"  Speed: {total_questions/total_time:.1f} questions/sec")
    logger.info(f"  Output: {output_dir}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
