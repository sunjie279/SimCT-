#!/usr/bin/env python3
"""
Evaluation script for KDFlow - powered by SGLang DP inference engine.
Supports: gsm8k, math500, aime24, aime25, aime26, humaneval, mbpp, live-code-bench-v6
Prerequisites:
    # Start SGLang server (recommend DP=8 for high throughput)
    python -m sglang.launch_server --model-path <model_path> --dp-size 8 --tp-size 1 --port 30000

Usage:
    python evaluation/evaluation.py --model_path /path/to/model --dataset gsm8k
    python evaluation/evaluation.py --model_path /path/to/model --dataset math500
"""

import argparse
import os
import sys
import re
import json
import time
import asyncio
import aiohttp
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

from datasets import load_from_disk
from tqdm.asyncio import tqdm_asyncio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# Paths
# ============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATASETS_DIR = ""
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
DEFAULT_BASE_URL = "http://127.0.0.1:30000"

# Dataset metrics
DATASET_METRICS = {
    "gsm8k": "Accuracy",
    "math500": "Accuracy",
    "mbpp": "pass@1",
    "live-code-bench-v6": "pass@1",
}

# Few-shot dataset names map to the same underlying data directory
# e.g. "gsm8k-5shot" uses the "gsm8k" data, "humaneval-3shot" uses "humaneval" data
DATASET_PATH_ALIASES = {
    "gsm8k-1shot": "gsm8k",
    "gsm8k-5shot": "gsm8k",
    "humaneval-1shot": "humaneval",
    "humaneval-3shot": "humaneval",
}

# Standard file names under results/{model}/{dataset}/
PREDICTIONS_FILE = "predictions.jsonl"
METRICS_FILE = "metrics.json"


# ============================================================================
# SGLang Client - async high-concurrency inference via OpenAI-compatible API
# ============================================================================
class SGLangClient:
    """SGLang inference client optimized for DP=8 data parallelism."""

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        max_concurrent: int = 256,
        timeout: int = 300,
    ):
        self.base_url = base_url.rstrip("/")
        self.completions_url = f"{self.base_url}/v1/completions"
        self.chat_url = f"{self.base_url}/v1/chat/completions"
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        self._supports_system_role = None
        self._logged_chat_template_kwargs_warning = False

    async def _post(
        self,
        session: aiohttp.ClientSession,
        url: str,
        payload: dict,
        semaphore: asyncio.Semaphore,
    ) -> dict:
        """Send a single request with retry.

        If the payload contains ``chat_template_kwargs`` and the server
        rejects it (HTTP 400/422), we automatically retry without that key
        so that non-thinking models are not affected.
        """
        max_retries = 3
        for attempt in range(max_retries):
            async with semaphore:
                try:
                    async with session.post(
                        url,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=self.timeout),
                    ) as resp:
                        if resp.status != 200:
                            text = await resp.text()
                            # If server rejects chat_template_kwargs, retry without it
                            if (
                                resp.status in (400, 422)
                                and "chat_template_kwargs" in payload
                                and ("chat_template_kwargs" in text or "Unexpected" in text)
                            ):
                                if not self._logged_chat_template_kwargs_warning:
                                    logger.warning(
                                        "Server does not recognise chat_template_kwargs; "
                                        "falling back to requests without it. "
                                        "Thinking mode may remain enabled for thinking models."
                                    )
                                    self._logged_chat_template_kwargs_warning = True
                                payload_without = {k: v for k, v in payload.items() if k != "chat_template_kwargs"}
                                # Retry immediately with cleaned payload
                                async with session.post(
                                    url,
                                    json=payload_without,
                                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                                ) as resp2:
                                    if resp2.status == 200:
                                        return await resp2.json()
                                    text2 = await resp2.text()
                                    logger.warning(f"Retry without chat_template_kwargs also failed (HTTP {resp2.status}): {text2[:200]}")
                                    return {"error": text2, "choices": []}
                            logger.warning(f"Request failed (HTTP {resp.status}): {text[:200]}")
                            if attempt < max_retries - 1:
                                await asyncio.sleep(1)
                                continue
                            return {"error": text, "choices": []}
                        return await resp.json()
                except asyncio.TimeoutError:
                    logger.warning(f"Request timeout (attempt {attempt + 1}/{max_retries})")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(1)
                        continue
                    return {"error": "timeout", "choices": []}
                except Exception as e:
                    logger.warning(f"Request error: {e}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(1)
                        continue
                    return {"error": str(e), "choices": []}
        return {"error": "max retries exceeded", "choices": []}

    async def _check_system_role_support(self, model: str = "default") -> bool:
        """Check if the served model supports system role in chat completions."""
        if self._supports_system_role is not None:
            return self._supports_system_role

        test_payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hi"},
            ],
            "max_tokens": 1,
            "temperature": 0,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.chat_url,
                    json=test_payload,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    if resp.status == 200:
                        self._supports_system_role = True
                        logger.info("System role support: ✓ supported")
                    else:
                        text = await resp.text()
                        if "System role not supported" in text:
                            self._supports_system_role = False
                            logger.info("System role support: ✗ not supported (will merge into user message)")
                        else:
                            self._supports_system_role = True
                            logger.warning(f"System role check got unexpected error: {text[:200]}")
        except Exception as e:
            logger.warning(f"System role check failed: {e}, assuming supported")
            self._supports_system_role = True

        return self._supports_system_role

    def _merge_system_into_user(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Merge system role content into the first user message."""
        system_content = ""
        new_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_content += msg["content"] + "\n"
            else:
                new_messages.append(msg)

        if system_content and new_messages:
            first_user = new_messages[0]
            if first_user["role"] == "user":
                new_messages[0] = {
                    "role": "user",
                    "content": system_content.strip() + "\n\n" + first_user["content"],
                }

        return new_messages if new_messages else messages

    async def chat_batch_async(
        self,
        messages_list: List[List[Dict[str, str]]],
        model: str = "default",
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        stop: Optional[List[str]] = None,
        show_progress: bool = True,
    ) -> List[str]:
        """Batch async chat generation via /v1/chat/completions."""
        await self._check_system_role_support(model)

        if not self._supports_system_role:
            messages_list = [self._merge_system_into_user(msgs) for msgs in messages_list]

        semaphore = asyncio.Semaphore(self.max_concurrent)
        tasks = []
        indices = []

        for idx, messages in enumerate(messages_list):
            payload = {
                "model": model,
                "messages": messages,
                "max_tokens": max_new_tokens,
                "temperature": temperature,
                # Disable thinking mode for all models (no-op for non-thinking models)
                "chat_template_kwargs": {"enable_thinking": False},
            }
            if temperature > 0 and top_p < 1.0:
                payload["top_p"] = top_p
            if stop:
                payload["stop"] = stop
            tasks.append(payload)
            indices.append(idx)

        results = [None] * len(messages_list)

        async with aiohttp.ClientSession() as session:
            async_tasks = [
                self._post(session, self.chat_url, payload, semaphore)
                for payload in tasks
            ]

            if show_progress:
                responses = await tqdm_asyncio.gather(
                    *async_tasks, desc="Inference(chat)", total=len(async_tasks)
                )
            else:
                responses = await asyncio.gather(*async_tasks)

            empty_count = 0
            for idx, resp in zip(indices, responses):
                if resp.get("choices"):
                    msg = resp["choices"][0].get("message", {})
                    content = msg.get("content", "")
                    results[idx] = content
                    if not content or not content.strip():
                        empty_count += 1
                        if resp.get("error"):
                            logger.debug(f"Sample {idx}: empty content with error: {resp['error']}")
                else:
                    results[idx] = ""
                    empty_count += 1
                    error_info = resp.get("error", "no choices in response")
                    logger.debug(f"Sample {idx}: empty output - {error_info}")

            # Warn if too many empty outputs
            total_count = len(responses)
            if total_count > 0 and empty_count / total_count > 0.5:
                logger.warning(
                    f"🚨 CRITICAL: {empty_count}/{total_count} ({empty_count/total_count*100:.1f}%) "
                    f"responses returned empty output! This likely indicates a chat template "
                    f"incompatibility or server configuration issue."
                )

        return results

    def chat_batch(self, messages_list, **kwargs) -> List[str]:
        """Sync wrapper for chat completions."""
        return asyncio.run(self.chat_batch_async(messages_list=messages_list, **kwargs))

    def health_check(self) -> bool:
        """Check if SGLang server is available."""
        import urllib.request
        try:
            url = f"{self.base_url}/health"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status == 200
        except Exception:
            try:
                url = f"{self.base_url}/v1/models"
                req = urllib.request.Request(url, method="GET")
                with urllib.request.urlopen(req, timeout=5) as resp:
                    return resp.status == 200
            except Exception:
                return False

    def get_model_name(self) -> str:
        """Get the model name served by SGLang."""
        import urllib.request
        try:
            url = f"{self.base_url}/v1/models"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode())
                if data.get("data"):
                    return data["data"][0]["id"]
        except Exception:
            pass
        return "unknown"


def save_predictions(predictions: list, output_path: str):
    """Save predictions in JSONL format."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for pred in predictions:
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")
    logger.info(f"Predictions saved to {output_path}")


def check_existing_results(output_dir: str) -> Optional[str]:
    """Check if prediction results already exist."""
    pred_path = os.path.join(output_dir, PREDICTIONS_FILE)
    if os.path.exists(pred_path) and os.path.getsize(pred_path) > 0:
        return pred_path
    return None


# ============================================================================
# Answer extraction utilities
# ============================================================================


def strip_thinking_content(text: str) -> Tuple[str, bool, bool]:
    """Strip <think>...</think> content from model output (defensive fallback).

    Handles three cases:
      (a) Complete <think>...</think> tags → strip and return content after </think>
      (b) Only <think> without </think> (truncated) → mark as truncated,
          try to extract answer from the tail as fallback
      (c) No <think> tag → return as-is

    Returns:
        (stripped_text, had_thinking, was_truncated)
    """
    if not text:
        return text, False, False

    # Check if text contains <think> tag
    think_start = text.find("<think>")
    if think_start == -1:
        return text, False, False

    # Found <think>, now look for </think>
    think_end = text.find("</think>")
    if think_end != -1:
        # Case (a): Complete <think>...</think> — strip it
        # Keep content before <think> and after </think>
        before = text[:think_start].strip()
        after = text[think_end + len("</think>"):].strip()
        stripped = (before + "\n" + after).strip() if before else after
        return stripped, True, False
    else:
        # Case (b): Only <think> without </think> — truncated
        # Content before <think> (if any)
        before = text[:think_start].strip()
        if before:
            # There was content before <think>, use that
            return before, True, True

        # Everything is inside <think>... (truncated)
        # Try to extract from the tail (last 500 chars) as fallback
        thinking_content = text[think_start + len("<think>"):]
        tail = thinking_content[-500:] if len(thinking_content) > 500 else thinking_content
        return tail, True, True


QWEN_MATH_SYSTEM_PROMPT = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)

# GSM8K official system prompt: use native #### format (matching official dataset)
GSM8K_SYSTEM_PROMPT = (
    "Please reason step by step, and put your final answer after ####."
)


def extract_number_from_answer(answer_text: str) -> Optional[float]:
    """Extract number from answer text, supports #### and \\boxed{} formats."""
    if not answer_text:
        return None
    answer_text = answer_text.replace(",", "")

    if "####" in answer_text:
        match = re.search(r'####\s*(-?\d+\.?\d*)', answer_text)
        if match:
            return float(match.group(1))

    boxed_match = re.search(r'\\boxed\{([^}]+)\}', answer_text)
    if boxed_match:
        boxed_content = boxed_match.group(1).replace(",", "").strip()
        numbers = re.findall(r'-?\d+\.?\d*', boxed_content)
        if numbers:
            return float(numbers[-1])

    numbers = re.findall(r'-?\d+\.?\d*', answer_text)
    if numbers:
        return float(numbers[-1])

    return None


def normalize_answer_string(answer: str) -> str:
    """Normalize a LaTeX answer string for comparison."""
    if not answer:
        return ""
    s = answer.strip()
    s = re.sub(r'\\text\s*\{([^}]*)\}', r'\1', s)
    s = s.replace('\\dfrac', '\\frac').replace('\\tfrac', '\\frac')
    s = s.replace('\\left', '').replace('\\right', '')
    s = re.sub(r'\\[,;:!]', '', s)
    s = s.replace('\\%', '%')
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def extract_boxed_answer(text: str) -> str:
    """Extract the last \\boxed{...} content from text, handling nested braces."""
    idx = text.rfind('\\boxed')
    if idx == -1:
        return ""

    brace_start = text.find('{', idx)
    if brace_start == -1:
        return ""

    depth = 0
    for i in range(brace_start, len(text)):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                return text[brace_start + 1:i]

    return text[brace_start + 1:]


def try_parse_number(s: str) -> Optional[float]:
    """Try to parse a string as a number."""
    s = s.strip().replace(',', '')
    if s.endswith('.'):
        s = s[:-1]
    try:
        return float(s)
    except ValueError:
        return None


def is_math_equivalent(pred: str, gold: str) -> bool:
    """Check if predicted and gold answers are mathematically equivalent.

    Multi-level comparison:
    1. Exact string match (after normalization)
    2. Numeric comparison (with tolerance)
    3. Sympy symbolic comparison (if available)
    """
    if not pred or not gold:
        return False

    # Level 1: Normalized string match
    norm_pred = normalize_answer_string(pred)
    norm_gold = normalize_answer_string(gold)
    if norm_pred == norm_gold:
        return True

    # Level 2: Numeric comparison
    num_pred = try_parse_number(norm_pred)
    num_gold = try_parse_number(norm_gold)
    if num_pred is not None and num_gold is not None:
        return abs(num_pred - num_gold) < 1e-6

    # Level 3: Sympy symbolic comparison
    try:
        from sympy.parsing.latex import parse_latex
        from sympy import simplify, N

        sym_pred = parse_latex(pred)
        sym_gold = parse_latex(gold)

        diff = simplify(sym_pred - sym_gold)
        if diff == 0:
            return True

        try:
            val_pred = complex(N(sym_pred))
            val_gold = complex(N(sym_gold))
            if abs(val_pred - val_gold) < 1e-6:
                return True
        except (TypeError, ValueError):
            pass
    except Exception:
        pass

    return False


# ============================================================================
# GSM8K Evaluator
# ============================================================================

def evaluate_gsm8k(
    client: SGLangClient,
    model_name: str,
    dataset_path: str,
    split: str = "test",
    max_new_tokens: int = 2048,
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_concurrent: int = 256,
    output_dir: str = "./results",
    debug: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Evaluate GSM8K dataset using 0-shot chat mode with native #### format."""
    logger.info(f"Loading GSM8K dataset ({split} split)...")
    logger.info(f"Eval mode: 0-shot chat (####)")
    logger.info(f"Params: temperature={temperature}, top_p={top_p}, max_new_tokens={max_new_tokens}")

    dataset = load_from_disk(dataset_path)
    data = dataset[split]

    if debug:
        data = data.select(range(min(10, len(data))))
        logger.info(f"Debug mode: processing {len(data)} samples only")

    logger.info(f"Dataset size: {len(data)}")

    questions = []
    gold_answers = []
    for i in range(len(data)):
        sample = data[i]
        questions.append(sample['question'])
        gold_answers.append(sample['answer'])

    logger.info(f"Total {len(questions)} requests, starting concurrent inference...")
    client.max_concurrent = max_concurrent

    # 0-shot chat mode — use GSM8K native #### format
    no_system_prompt = kwargs.get('no_system_prompt', False)
    messages_list = []
    for question in questions:
        if no_system_prompt:
            messages_list.append([
                {"role": "user", "content": GSM8K_SYSTEM_PROMPT + "\n\n" + question},
            ])
        else:
            messages_list.append([
                {"role": "system", "content": GSM8K_SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ])
    generated_texts = client.chat_batch(
        messages_list=messages_list,
        model=model_name,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        show_progress=True,
    )

    predictions = []
    correct = 0
    total = 0
    truncated_count = 0
    extraction_failed_count = 0
    empty_output_count = 0

    for i in range(len(generated_texts)):
        generated = generated_texts[i]
        gold_answer = gold_answers[i]
        question = questions[i]

        # Strip thinking content before answer extraction (defensive fallback)
        stripped, had_thinking, was_truncated = strip_thinking_content(generated)

        pred_number = extract_number_from_answer(stripped)
        gold_number = extract_number_from_answer(gold_answer)

        is_correct = False
        if pred_number is not None and gold_number is not None:
            is_correct = abs(pred_number - gold_number) < 1e-6

        if is_correct:
            correct += 1
        total += 1

        # Detect truncation and extraction failures
        is_empty_output = not generated or not generated.strip()
        is_extraction_failed = pred_number is None and not is_empty_output
        if was_truncated:
            truncated_count += 1
        if is_extraction_failed:
            extraction_failed_count += 1
        if is_empty_output:
            empty_output_count += 1

        pred_dict = {
            'question': question,
            'gold_answer': gold_answer,
            'gold_number': gold_number,
            'generated': generated,
            'pred_number': pred_number,
            'is_correct': is_correct,
            'eval_mode': '0-shot chat (####)',
        }
        if was_truncated:
            pred_dict['truncated'] = True
        if is_extraction_failed:
            pred_dict['extraction_failed'] = True
        if is_empty_output:
            pred_dict['empty_output'] = True
        if had_thinking:
            pred_dict['had_thinking'] = True
        predictions.append(pred_dict)

    accuracy = correct / total if total > 0 else 0.0

    logger.info(f"\nResults (0-shot chat):")
    logger.info(f"  Correct: {correct}/{total}")
    logger.info(f"  Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

    pred_file = os.path.join(output_dir, PREDICTIONS_FILE)
    save_predictions(predictions, pred_file)

    return {
        'score': accuracy,
        'correct': correct,
        'total_samples': total,
        'eval_mode': '0-shot chat (####)',
        'truncated_count': truncated_count,
        'extraction_failed_count': extraction_failed_count,
        'empty_output_count': empty_output_count,
        'predictions': predictions,
    }


# ============================================================================
# MATH-500 Evaluator# ============================================================================

def evaluate_math500(
    client: SGLangClient,
    model_name: str,
    dataset_path: str,
    split: str = "test",
    max_new_tokens: int = 2048,
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_concurrent: int = 256,
    output_dir: str = "./results",
    debug: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Evaluate MATH-500 dataset using 0-shot chat mode with boxed format.

    MATH-500 answers are LaTeX strings; uses multi-level equivalence checking.
    """
    logger.info(f"Loading MATH-500 dataset...")
    logger.info(f"Eval mode: 0-shot chat (boxed)")
    logger.info(f"Params: temperature={temperature}, top_p={top_p}, max_new_tokens={max_new_tokens}")

    # MATH-500 is stored as a single split (from HuggingFaceH4/MATH-500)
    dataset_disk_path = os.path.join(dataset_path, "dataset")
    if os.path.exists(dataset_disk_path):
        data = load_from_disk(dataset_disk_path)
    else:
        ds = load_from_disk(dataset_path)
        if isinstance(ds, dict) and split in ds:
            data = ds[split]
        else:
            data = ds

    if debug:
        data = data.select(range(min(10, len(data))))
        logger.info(f"Debug mode: processing {len(data)} samples only")

    logger.info(f"Dataset size: {len(data)}")

    problems = []
    gold_answers = []
    subjects = []
    levels = []
    unique_ids = []

    for i in range(len(data)):
        sample = data[i]
        problems.append(sample['problem'])
        gold_answers.append(sample['answer'])
        subjects.append(sample.get('subject', ''))
        levels.append(sample.get('level', 0))
        unique_ids.append(sample.get('unique_id', f'math500_{i}'))

    logger.info(f"Total {len(problems)} requests, starting concurrent inference...")
    client.max_concurrent = max_concurrent

    # 0-shot chat mode
    no_system_prompt = kwargs.get('no_system_prompt', False)
    messages_list = []
    for problem in problems:
        if no_system_prompt:
            messages_list.append([
                {"role": "user", "content": QWEN_MATH_SYSTEM_PROMPT + "\n\n" + problem},
            ])
        else:
            messages_list.append([
                {"role": "system", "content": QWEN_MATH_SYSTEM_PROMPT},
                {"role": "user", "content": problem},
            ])
    generated_texts = client.chat_batch(
        messages_list=messages_list,
        model=model_name,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        show_progress=True,
    )

    predictions = []
    correct = 0
    total = 0
    truncated_count = 0
    extraction_failed_count = 0
    empty_output_count = 0
    subject_stats = {}
    level_stats = {}

    for i in range(len(generated_texts)):
        generated = generated_texts[i]
        gold_answer = gold_answers[i]
        problem = problems[i]
        subject = subjects[i]
        level = levels[i]
        uid = unique_ids[i]

        # Strip thinking content before answer extraction (defensive fallback)
        stripped, had_thinking, was_truncated = strip_thinking_content(generated)

        pred_answer = extract_boxed_answer(stripped)
        is_correct = is_math_equivalent(pred_answer, gold_answer)

        if is_correct:
            correct += 1
        total += 1

        # Detect truncation and extraction failures
        is_empty_output = not generated or not generated.strip()
        is_extraction_failed = not pred_answer and not is_empty_output
        if was_truncated:
            truncated_count += 1
        if is_extraction_failed:
            extraction_failed_count += 1
        if is_empty_output:
            empty_output_count += 1

        # Track per-subject stats
        if subject:
            if subject not in subject_stats:
                subject_stats[subject] = {'correct': 0, 'total': 0}
            subject_stats[subject]['total'] += 1
            if is_correct:
                subject_stats[subject]['correct'] += 1

        # Track per-level stats
        if level:
            level_key = f"Level {level}"
            if level_key not in level_stats:
                level_stats[level_key] = {'correct': 0, 'total': 0}
            level_stats[level_key]['total'] += 1
            if is_correct:
                level_stats[level_key]['correct'] += 1

        pred_dict = {
            'unique_id': uid,
            'problem': problem,
            'gold_answer': gold_answer,
            'generated': generated,
            'pred_answer': pred_answer,
            'is_correct': is_correct,
            'subject': subject,
            'level': level,
            'eval_mode': '0-shot chat (boxed)',
        }
        if was_truncated:
            pred_dict['truncated'] = True
        if is_extraction_failed:
            pred_dict['extraction_failed'] = True
        if is_empty_output:
            pred_dict['empty_output'] = True
        if had_thinking:
            pred_dict['had_thinking'] = True
        predictions.append(pred_dict)

    accuracy = correct / total if total > 0 else 0.0

    logger.info(f"\nResults (0-shot chat):")
    logger.info(f"  Correct: {correct}/{total}")
    logger.info(f"  Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

    if subject_stats:
        logger.info(f"\n  Per-subject breakdown:")
        for subj, stats in sorted(subject_stats.items()):
            subj_acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0
            logger.info(f"    {subj}: {stats['correct']}/{stats['total']} ({subj_acc * 100:.1f}%)")

    if level_stats:
        logger.info(f"\n  Per-level breakdown:")
        for lvl, stats in sorted(level_stats.items()):
            lvl_acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0
            logger.info(f"    {lvl}: {stats['correct']}/{stats['total']} ({lvl_acc * 100:.1f}%)")

    pred_file = os.path.join(output_dir, PREDICTIONS_FILE)
    save_predictions(predictions, pred_file)

    return {
        'score': accuracy,
        'correct': correct,
        'total_samples': total,
        'eval_mode': '0-shot chat (boxed)',
        'truncated_count': truncated_count,
        'extraction_failed_count': extraction_failed_count,
        'empty_output_count': empty_output_count,
        'subject_stats': {k: {**v, 'accuracy': v['correct'] / v['total'] if v['total'] > 0 else 0.0}
                          for k, v in subject_stats.items()},
        'level_stats': {k: {**v, 'accuracy': v['correct'] / v['total'] if v['total'] > 0 else 0.0}
                        for k, v in level_stats.items()},
        'predictions': predictions,
    }


# ============================================================================
# AIME Evaluator (AIME 2024, 2025, 2026)
# ============================================================================

def evaluate_aime(
    client: SGLangClient,
    model_name: str,
    dataset_path: str,
    split: str = "test",
    max_new_tokens: int = 2048,
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_concurrent: int = 256,
    output_dir: str = "./results",
    debug: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Evaluate AIME dataset using 0-shot chat mode with boxed format.

    AIME answers are always integers (0-999). We extract the boxed answer
    and compare numerically.
    """
    logger.info(f"Loading AIME dataset from {dataset_path}...")
    logger.info(f"Eval mode: 0-shot chat (boxed)")
    logger.info(f"Params: temperature={temperature}, top_p={top_p}, max_new_tokens={max_new_tokens}")

    # Load dataset - AIME is stored as a single split
    data = load_from_disk(dataset_path)
    if isinstance(data, dict):
        # DatasetDict - pick first available split
        split_name = list(data.keys())[0]
        data = data[split_name]

    if debug:
        data = data.select(range(min(5, len(data))))
        logger.info(f"Debug mode: processing {len(data)} samples only")

    logger.info(f"Dataset size: {len(data)}")

    problems = []
    gold_answers = []

    for i in range(len(data)):
        sample = data[i]
        # Support both 'problem' and 'Problem' field names
        problem = sample.get('problem') or sample.get('Problem', '')
        answer = str(sample.get('answer') or sample.get('Answer', ''))
        problems.append(problem)
        gold_answers.append(answer)

    logger.info(f"Total {len(problems)} requests, starting concurrent inference...")
    client.max_concurrent = max_concurrent

    # 0-shot chat mode
    no_system_prompt = kwargs.get('no_system_prompt', False)
    messages_list = []
    for problem in problems:
        if no_system_prompt:
            messages_list.append([
                {"role": "user", "content": QWEN_MATH_SYSTEM_PROMPT + "\n\n" + problem},
            ])
        else:
            messages_list.append([
                {"role": "system", "content": QWEN_MATH_SYSTEM_PROMPT},
                {"role": "user", "content": problem},
            ])
    generated_texts = client.chat_batch(
        messages_list=messages_list,
        model=model_name,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        show_progress=True,
    )

    predictions = []
    correct = 0
    total = 0
    truncated_count = 0
    extraction_failed_count = 0
    empty_output_count = 0

    for i in range(len(generated_texts)):
        generated = generated_texts[i]
        gold_answer = gold_answers[i]
        problem = problems[i]

        # Strip thinking content before answer extraction (defensive fallback)
        stripped, had_thinking, was_truncated = strip_thinking_content(generated)

        # Extract answer from \boxed{} first, then fallback to number extraction
        pred_answer = extract_boxed_answer(stripped)
        is_correct = False

        # AIME answers are integers, so compare numerically
        pred_num = try_parse_number(pred_answer) if pred_answer else None
        gold_num = try_parse_number(gold_answer)

        if pred_num is not None and gold_num is not None:
            is_correct = abs(pred_num - gold_num) < 1e-6

        # Fallback: try extracting number from full generated text
        if not is_correct and not pred_answer:
            pred_num_fallback = extract_number_from_answer(stripped)
            if pred_num_fallback is not None and gold_num is not None:
                is_correct = abs(pred_num_fallback - gold_num) < 1e-6
                pred_answer = str(pred_num_fallback)

        if is_correct:
            correct += 1
        total += 1

        # Detect truncation and extraction failures
        is_empty_output = not generated or not generated.strip()
        is_extraction_failed = not pred_answer and not is_empty_output
        if was_truncated:
            truncated_count += 1
        if is_extraction_failed:
            extraction_failed_count += 1
        if is_empty_output:
            empty_output_count += 1

        pred_dict = {
            'problem': problem,
            'gold_answer': gold_answer,
            'generated': generated,
            'pred_answer': pred_answer,
            'is_correct': is_correct,
            'eval_mode': '0-shot chat (boxed)',
        }
        if was_truncated:
            pred_dict['truncated'] = True
        if is_extraction_failed:
            pred_dict['extraction_failed'] = True
        if is_empty_output:
            pred_dict['empty_output'] = True
        if had_thinking:
            pred_dict['had_thinking'] = True
        predictions.append(pred_dict)

    accuracy = correct / total if total > 0 else 0.0

    logger.info(f"\nResults (0-shot chat):")
    logger.info(f"  Correct: {correct}/{total}")
    logger.info(f"  Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

    pred_file = os.path.join(output_dir, PREDICTIONS_FILE)
    save_predictions(predictions, pred_file)

    return {
        'score': accuracy,
        'correct': correct,
        'total_samples': total,
        'eval_mode': '0-shot chat (boxed)',
        'truncated_count': truncated_count,
        'extraction_failed_count': extraction_failed_count,
        'empty_output_count': empty_output_count,
        'predictions': predictions,
    }


# ============================================================================
# HumanEval Evaluator
# ============================================================================

HUMANEVAL_SYSTEM_PROMPT = (
    "Complete the following Python function. Only output the function body "
    "(the code that goes inside the function), without repeating the function "
    "signature or adding any explanation."
)


def _extract_code_block(text: str) -> str:
    """Extract code from markdown code block if present, otherwise return as-is."""
    if not text:
        return ""
    # Try to extract from ```python ... ``` or ``` ... ```
    import re as _re
    pattern = _re.compile(r'```(?:python)?\s*\n(.*?)```', _re.DOTALL)
    match = pattern.search(text)
    if match:
        return match.group(1).rstrip()
    return text.rstrip()


def _extract_function_body(generated: str, prompt: str) -> str:
    """Extract the function body completion from model output.

    The model may repeat the prompt or include extra text. We try to
    extract only the completion part that should be appended to the prompt.
    """
    code = _extract_code_block(generated)

    # If the model repeated the prompt, strip it
    if code.startswith(prompt.rstrip()):
        code = code[len(prompt.rstrip()):]

    # Remove leading blank lines but preserve indentation
    lines = code.split('\n')
    # Find first non-empty line
    start = 0
    for i, line in enumerate(lines):
        if line.strip():
            start = i
            break
    code = '\n'.join(lines[start:])

    return code


def evaluate_humaneval(
    client: SGLangClient,
    model_name: str,
    dataset_path: str,
    split: str = "test",
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_concurrent: int = 256,
    output_dir: str = "./results",
    debug: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Evaluate HumanEval dataset using code generation + execution.

    Uses the official human_eval package for functional correctness evaluation.
    """
    logger.info(f"Loading HumanEval dataset...")
    logger.info(f"Eval mode: 0-shot chat (code completion)")
    logger.info(f"Params: temperature={temperature}, top_p={top_p}, max_new_tokens={max_new_tokens}")

    # Load dataset
    dataset = load_from_disk(dataset_path)
    if isinstance(dataset, dict):
        data = dataset.get(split, dataset[list(dataset.keys())[0]])
    else:
        data = dataset

    if debug:
        data = data.select(range(min(10, len(data))))
        logger.info(f"Debug mode: processing {len(data)} samples only")

    logger.info(f"Dataset size: {len(data)}")

    task_ids = []
    prompts = []
    for i in range(len(data)):
        sample = data[i]
        task_ids.append(sample['task_id'])
        prompts.append(sample['prompt'])

    logger.info(f"Total {len(prompts)} requests, starting concurrent inference...")
    client.max_concurrent = max_concurrent

    # Build chat messages: ask model to complete the function
    no_system_prompt = kwargs.get('no_system_prompt', False)
    messages_list = []
    for prompt in prompts:
        user_msg = f"Complete the following Python function:\n\n{prompt}"
        if no_system_prompt:
            messages_list.append([
                {"role": "user", "content": HUMANEVAL_SYSTEM_PROMPT + "\n\n" + user_msg},
            ])
        else:
            messages_list.append([
                {"role": "system", "content": HUMANEVAL_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ])

    generated_texts = client.chat_batch(
        messages_list=messages_list,
        model=model_name,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        show_progress=True,
    )

    # Build samples JSONL for human_eval evaluation
    samples_file = os.path.join(output_dir, "humaneval_samples.jsonl")
    predictions = []
    samples_for_eval = []
    truncated_count = 0
    extraction_failed_count = 0
    empty_output_count = 0

    for i in range(len(generated_texts)):
        generated = generated_texts[i]
        task_id = task_ids[i]
        prompt = prompts[i]

        # Strip thinking content before code extraction (defensive fallback)
        stripped, had_thinking, was_truncated = strip_thinking_content(generated)

        # Extract completion (function body)
        completion = _extract_function_body(stripped, prompt)

        samples_for_eval.append({
            "task_id": task_id,
            "completion": completion,
        })

        # Detect truncation and extraction failures
        is_empty_output = not generated or not generated.strip()
        is_extraction_failed = not completion.strip() and not is_empty_output
        if was_truncated:
            truncated_count += 1
        if is_extraction_failed:
            extraction_failed_count += 1
        if is_empty_output:
            empty_output_count += 1

        pred_dict = {
            "task_id": task_id,
            "prompt": prompt,
            "generated": generated,
            "completion": completion,
            "eval_mode": "0-shot chat (code completion)",
        }
        if was_truncated:
            pred_dict['truncated'] = True
        if is_extraction_failed:
            pred_dict['extraction_failed'] = True
        if is_empty_output:
            pred_dict['empty_output'] = True
        if had_thinking:
            pred_dict['had_thinking'] = True
        predictions.append(pred_dict)

    # Write samples JSONL
    with open(samples_file, 'w') as f:
        for s in samples_for_eval:
            f.write(json.dumps(s) + '\n')

    # Save predictions
    pred_file = os.path.join(output_dir, PREDICTIONS_FILE)
    save_predictions(predictions, pred_file)

    # Run functional correctness evaluation using official human_eval
    logger.info("Running functional correctness evaluation (code execution)...")
    try:
        from human_eval.evaluation import evaluate_functional_correctness
        # Use the original HumanEval.jsonl as problem file
        problem_file = os.path.join(dataset_path, "HumanEval.jsonl")
        if not os.path.exists(problem_file):
            # Fallback: use the default from human_eval package
            from human_eval.data import HUMAN_EVAL
            problem_file = HUMAN_EVAL

        pass_at_k = evaluate_functional_correctness(
            sample_file=samples_file,
            k=[1],
            n_workers=8,
            timeout=10.0,
            problem_file=problem_file,
            ignore_incomplete=debug,
        )
        score = pass_at_k.get("pass@1", 0.0)
        logger.info(f"HumanEval pass@1: {score:.4f} ({score * 100:.2f}%)")
    except Exception as e:
        logger.error(f"Functional correctness evaluation failed: {e}")
        # Fallback: cannot compute score without execution
        score = 0.0
        pass_at_k = {"pass@1": 0.0, "error": str(e)}

    return {
        'score': score,
        'pass_at_k': pass_at_k,
        'total_samples': len(predictions),
        'eval_mode': '0-shot chat (code completion)',
        'truncated_count': truncated_count,
        'extraction_failed_count': extraction_failed_count,
        'empty_output_count': empty_output_count,
        'predictions': predictions,
    }


# ============================================================================
# MBPP Evaluator
# ============================================================================

MBPP_SYSTEM_PROMPT = (
    "You are a Python programming assistant. Write a Python function to solve "
    "the given task. Your code must pass the provided test cases. "
    "Only output the Python code, without any explanation."
)

# NOTE: Switched from 3-shot to 0-shot with test cases in prompt.
# The 3-shot approach caused 87%+ failures because the model generated
# its own function names that didn't match the test case expectations.
# Including test cases in the prompt (0-shot) is the standard MBPP eval approach.


def _run_mbpp_tests(code: str, test_list: List[str], test_setup_code: str = "",
                     timeout: float = 10.0) -> bool:
    """Execute MBPP test cases against generated code."""
    import multiprocessing

    full_code = ""
    if test_setup_code:
        full_code += test_setup_code + "\n"
    full_code += code + "\n"
    for test in test_list:
        full_code += test + "\n"

    def _exec_code(code_str, result_queue):
        try:
            exec_globals = {}
            exec(code_str, exec_globals)
            result_queue.put(True)
        except Exception as e:
            result_queue.put(False)

    result_queue = multiprocessing.Queue()
    proc = multiprocessing.Process(target=_exec_code, args=(full_code, result_queue))
    proc.start()
    proc.join(timeout=timeout)

    if proc.is_alive():
        proc.kill()
        proc.join()
        return False

    if result_queue.empty():
        return False

    return result_queue.get()


def evaluate_mbpp(
    client: SGLangClient,
    model_name: str,
    dataset_path: str,
    split: str = "test",
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_concurrent: int = 256,
    output_dir: str = "./results",
    debug: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Evaluate MBPP dataset using 0-shot code generation + execution with test cases in prompt."""
    logger.info(f"Loading MBPP dataset ({split} split)...")
    logger.info(f"Eval mode: 0-shot chat with test cases (code generation)")
    logger.info(f"Params: temperature={temperature}, top_p={top_p}, max_new_tokens={max_new_tokens}")

    dataset = load_from_disk(dataset_path)
    if isinstance(dataset, dict):
        data = dataset.get(split, dataset[list(dataset.keys())[0]])
    else:
        data = dataset

    if debug:
        data = data.select(range(min(10, len(data))))
        logger.info(f"Debug mode: processing {len(data)} samples only")

    logger.info(f"Dataset size: {len(data)}")

    task_ids = []
    texts = []
    codes = []
    test_lists = []
    test_setup_codes = []

    for i in range(len(data)):
        sample = data[i]
        task_ids.append(sample['task_id'])
        texts.append(sample['text'])
        codes.append(sample['code'])
        test_lists.append(sample['test_list'])
        test_setup_codes.append(sample.get('test_setup_code', ''))

    logger.info(f"Total {len(texts)} requests, starting concurrent inference...")
    client.max_concurrent = max_concurrent

    # Build chat messages: 0-shot with test cases in prompt
    # This ensures the model sees the expected function name from test cases
    # Build chat messages: 0-shot with test cases in prompt
    no_system_prompt = kwargs.get('no_system_prompt', False)
    messages_list = []
    for idx, text in enumerate(texts):
        test_cases_str = "\n".join(test_lists[idx])
        user_content = (
            f"{text}\n\n"
            f"Your code should pass these tests:\n"
            f"{test_cases_str}"
        )
        if no_system_prompt:
            messages_list.append([
                {"role": "user", "content": MBPP_SYSTEM_PROMPT + "\n\n" + user_content},
            ])
        else:
            messages_list.append([
                {"role": "system", "content": MBPP_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ])

    generated_texts = client.chat_batch(
        messages_list=messages_list,
        model=model_name,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        show_progress=True,
    )

    # Evaluate each sample by executing test cases
    predictions = []
    correct = 0
    total = 0
    truncated_count = 0
    extraction_failed_count = 0
    empty_output_count = 0

    logger.info("Running test case execution...")
    for i in range(len(generated_texts)):
        generated = generated_texts[i]
        task_id = task_ids[i]
        text = texts[i]
        test_list = test_lists[i]
        test_setup_code = test_setup_codes[i]

        # Strip thinking content before code extraction (defensive fallback)
        stripped, had_thinking, was_truncated = strip_thinking_content(generated)

        # Extract code from model output
        code = _extract_code_block(stripped)

        # Run tests
        passed = _run_mbpp_tests(code, test_list, test_setup_code)

        if passed:
            correct += 1
        total += 1

        # Detect truncation and extraction failures
        is_empty_output = not generated or not generated.strip()
        is_extraction_failed = not code.strip() and not is_empty_output
        if was_truncated:
            truncated_count += 1
        if is_extraction_failed:
            extraction_failed_count += 1
        if is_empty_output:
            empty_output_count += 1

        pred_dict = {
            'task_id': task_id,
            'text': text,
            'gold_code': codes[i],
            'generated': generated,
            'extracted_code': code,
            'test_list': test_list,
            'passed': passed,
            'eval_mode': '0-shot chat with test cases (code generation)',
        }
        if was_truncated:
            pred_dict['truncated'] = True
        if is_extraction_failed:
            pred_dict['extraction_failed'] = True
        if is_empty_output:
            pred_dict['empty_output'] = True
        if had_thinking:
            pred_dict['had_thinking'] = True
        predictions.append(pred_dict)

        if (i + 1) % 50 == 0:
            logger.info(f"  Progress: {i + 1}/{len(generated_texts)}, pass rate so far: {correct}/{total}")

    accuracy = correct / total if total > 0 else 0.0

    logger.info(f"\nResults (0-shot chat with test cases):")
    logger.info(f"  Passed: {correct}/{total}")
    logger.info(f"  pass@1: {accuracy:.4f} ({accuracy * 100:.2f}%)")

    pred_file = os.path.join(output_dir, PREDICTIONS_FILE)
    save_predictions(predictions, pred_file)

    return {
        'score': accuracy,
        'correct': correct,
        'total_samples': total,
        'eval_mode': '0-shot chat with test cases (code generation)',
        'truncated_count': truncated_count,
        'extraction_failed_count': extraction_failed_count,
        'empty_output_count': empty_output_count,
        'predictions': predictions,
    }


# ============================================================================
# LiveCodeBench-v6 Evaluator
# ============================================================================

LCB_SYSTEM_PROMPT = (
    "You are a Python programming assistant. Solve the given competitive "
    "programming problem. Read from standard input and write to standard output. "
    "Only output the Python code, without any explanation."
)


def _run_lcb_tests(code: str, test_cases: List[Dict[str, str]],
                    timeout: float = 10.0) -> Tuple[int, int]:
    """Execute LiveCodeBench test cases (stdin/stdout) against generated code.

    Batches all test cases into a single subprocess call for efficiency.
    The wrapper script runs the user code against each test case's stdin,
    compares output, and prints a JSON summary.
    Each individual test case is guarded by signal.alarm to prevent TLE.

    Returns (num_passed, num_total).
    """
    import subprocess
    import tempfile

    total = len(test_cases)
    if total == 0:
        return 0, 0

    # Build a wrapper script that runs the user code against all test cases
    # in a single process, avoiding repeated subprocess startup overhead.
    test_inputs = [tc.get("input", "") for tc in test_cases]
    test_outputs = [tc.get("output", "").strip() for tc in test_cases]

    # Encode test data as JSON embedded in the wrapper
    import json as _json
    test_data_json = _json.dumps({"inputs": test_inputs, "outputs": test_outputs})

    # Per-test-case timeout in seconds (embedded in wrapper)
    per_test_timeout = int(timeout)

    wrapper_code = '''import sys, io, json, signal

def _timeout_handler(signum, frame):
    raise TimeoutError()

test_data = json.loads("""__TEST_DATA__""")
inputs = test_data["inputs"]
expected = test_data["outputs"]

# The user code as a string
user_code = """__USER_CODE__"""
compiled_code = compile(user_code, "<solution>", "exec")

passed = 0
for i in range(len(inputs)):
    try:
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(__PER_TEST_TIMEOUT__)
        old_stdin = sys.stdin
        old_stdout = sys.stdout
        sys.stdin = io.StringIO(inputs[i])
        capture = io.StringIO()
        sys.stdout = capture
        exec_globals = {"__name__": "__main__"}
        exec(compiled_code, exec_globals)
        sys.stdin = old_stdin
        sys.stdout = old_stdout
        signal.alarm(0)
        actual = capture.getvalue().strip()
        if actual == expected[i]:
            passed += 1
    except (TimeoutError, SystemExit):
        sys.stdin = old_stdin
        sys.stdout = old_stdout
        signal.alarm(0)
    except Exception:
        sys.stdin = old_stdin
        sys.stdout = old_stdout
        signal.alarm(0)

print(passed, file=sys.stderr)
'''

    # Safely embed the JSON and user code into the wrapper
    # Use repr-based escaping to avoid triple-quote conflicts
    safe_test_data = test_data_json.replace('\\', '\\\\').replace('"""', '\\"\\"\\"')
    safe_user_code = code.replace('\\', '\\\\').replace('"""', '\\"\\"\\"')
    wrapper_code = wrapper_code.replace('__TEST_DATA__', safe_test_data)
    wrapper_code = wrapper_code.replace('__USER_CODE__', safe_user_code)
    wrapper_code = wrapper_code.replace('__PER_TEST_TIMEOUT__', str(per_test_timeout))

    # Per-problem timeout: tighter cap to avoid long tail blocking
    total_timeout = min(timeout * total, 120.0)  # cap at 2 minutes

    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(wrapper_code)
            f.flush()
            tmp_path = f.name

        result = subprocess.run(
            ["python3", tmp_path],
            capture_output=True,
            text=True,
            timeout=total_timeout,
        )
        # The wrapper prints passed count to stderr
        try:
            passed = int(result.stderr.strip().split('\n')[-1])
        except (ValueError, IndexError):
            passed = 0
    except subprocess.TimeoutExpired:
        passed = 0
    except Exception:
        passed = 0
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    return passed, total


def _run_lcb_tests_for_problem(args: Tuple) -> Tuple[int, int, int, bool]:
    """Wrapper for parallel execution. Takes (index, code, test_cases) tuple.
    Returns (index, num_passed, num_total, all_passed).
    """
    idx, code, test_cases = args
    if test_cases:
        num_passed, num_total = _run_lcb_tests(code, test_cases)
        all_passed = (num_passed == num_total) and num_total > 0
    else:
        num_passed, num_total = 0, 0
        all_passed = False
    return idx, num_passed, num_total, all_passed


def evaluate_lcb(
    client: SGLangClient,
    model_name: str,
    dataset_path: str,
    split: str = "test",
    max_new_tokens: int = 2048,
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_concurrent: int = 256,
    output_dir: str = "./results",
    debug: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Evaluate LiveCodeBench-v6 dataset using code generation + execution.

    LiveCodeBench problems are competitive programming style: read from stdin,
    write to stdout. Each problem has multiple test cases.
    """
    logger.info(f"Loading LiveCodeBench-v6 dataset...")
    logger.info(f"Eval mode: 0-shot chat (competitive programming)")
    logger.info(f"Params: temperature={temperature}, top_p={top_p}, max_new_tokens={max_new_tokens}")

    # Load dataset - support both HF dataset and JSONL format
    data_file = os.path.join(dataset_path, "problems.jsonl")
    if os.path.exists(data_file):
        # JSONL format
        problems = []
        with open(data_file, 'r') as f:
            for line in f:
                if line.strip():
                    problems.append(json.loads(line))
    else:
        # Try HF dataset format
        dataset = load_from_disk(dataset_path)
        if isinstance(dataset, dict):
            data = dataset.get(split, dataset[list(dataset.keys())[0]])
        else:
            data = dataset
        problems = [data[i] for i in range(len(data))]

    if debug:
        problems = problems[:min(5, len(problems))]
        logger.info(f"Debug mode: processing {len(problems)} samples only")

    logger.info(f"Dataset size: {len(problems)}")

    # Extract problem descriptions and test cases
    problem_ids = []
    descriptions = []
    test_cases_list = []

    for p in problems:
        pid = p.get('question_id', p.get('task_id', p.get('id', '')))
        problem_ids.append(str(pid))
        desc = p.get('question_content', p.get('description', p.get('prompt', '')))
        descriptions.append(desc)

        # Collect all test cases: public + private
        all_tc = []

        # Parse public test cases (JSON string or list)
        pub_tc = p.get('public_test_cases', p.get('test_cases', p.get('tests', [])))
        if isinstance(pub_tc, str):
            try:
                pub_tc = json.loads(pub_tc)
            except json.JSONDecodeError:
                pub_tc = []
        if isinstance(pub_tc, list):
            all_tc.extend(pub_tc)

        # Decode private test cases (base64 -> zlib -> pickle -> JSON)
        priv_raw = p.get('private_test_cases', '')
        if priv_raw and isinstance(priv_raw, str):
            try:
                import base64, zlib, pickle
                decoded = base64.b64decode(priv_raw)
                decompressed = zlib.decompress(decoded)
                tc_str = pickle.loads(decompressed)
                priv_tc = json.loads(tc_str) if isinstance(tc_str, str) else tc_str
                if isinstance(priv_tc, list):
                    all_tc.extend(priv_tc)
                    logger.debug(f"  Problem {pid}: {len(pub_tc)} public + {len(priv_tc)} private test cases")
            except Exception as e:
                logger.warning(f"  Failed to decode private_test_cases for {pid}: {e}")

        test_cases_list.append(all_tc)

    logger.info(f"Test cases loaded: avg {sum(len(tc) for tc in test_cases_list) / max(len(test_cases_list), 1):.1f} per problem")

    logger.info(f"Total {len(descriptions)} requests, starting concurrent inference...")
    client.max_concurrent = max_concurrent

    # Build chat messages
    no_system_prompt = kwargs.get('no_system_prompt', False)
    messages_list = []
    for desc in descriptions:
        if no_system_prompt:
            messages_list.append([
                {"role": "user", "content": LCB_SYSTEM_PROMPT + "\n\n" + desc},
            ])
        else:
            messages_list.append([
                {"role": "system", "content": LCB_SYSTEM_PROMPT},
                {"role": "user", "content": desc},
            ])

    generated_texts = client.chat_batch(
        messages_list=messages_list,
        model=model_name,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        show_progress=True,
    )

    # Evaluate each solution — parallel test execution
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing

    # Strip thinking content and extract code for all problems
    thinking_flags = []  # (had_thinking, was_truncated) per problem
    codes = []
    for i in range(len(generated_texts)):
        stripped, had_thinking, was_truncated = strip_thinking_content(generated_texts[i])
        thinking_flags.append((had_thinking, was_truncated))
        codes.append(_extract_code_block(stripped))

    # Prepare parallel tasks: (index, code, test_cases)
    tasks = [(i, codes[i], test_cases_list[i]) for i in range(len(generated_texts))]

    # Use up to N workers (bounded by CPU count and problem count)
    num_workers = min(multiprocessing.cpu_count(), len(tasks), 64)
    logger.info(f"Running test case execution... (parallel, {num_workers} workers)")

    # Results indexed by problem index
    results_map = {}
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(_run_lcb_tests_for_problem, t): t[0] for t in tasks}
        done_count = 0
        for future in as_completed(futures, timeout=600):
            try:
                idx, num_passed, num_total, all_passed = future.result(timeout=180)
            except Exception as e:
                idx = futures[future]
                logger.warning(f"  Problem {idx} test execution failed: {e}")
                num_passed, num_total, all_passed = 0, 0, False
            results_map[idx] = (num_passed, num_total, all_passed)
            done_count += 1
            if done_count % 20 == 0:
                logger.info(f"  Test execution progress: {done_count}/{len(tasks)}")
        # Handle any remaining futures that timed out
        for future, idx in futures.items():
            if idx not in results_map:
                logger.warning(f"  Problem {idx} timed out, marking as failed")
                future.cancel()
                results_map[idx] = (0, 0, False)

    # Assemble predictions in original order
    predictions = []
    correct = 0
    total = 0
    truncated_count = 0
    extraction_failed_count = 0
    empty_output_count = 0

    for i in range(len(generated_texts)):
        generated = generated_texts[i]
        pid = problem_ids[i]
        desc = descriptions[i]
        code = codes[i]
        had_thinking, was_truncated = thinking_flags[i]
        num_passed, num_total, all_passed = results_map[i]

        if all_passed:
            correct += 1
        total += 1

        # Detect truncation and extraction failures
        is_empty_output = not generated or not generated.strip()
        is_extraction_failed = not code.strip() and not is_empty_output
        if was_truncated:
            truncated_count += 1
        if is_extraction_failed:
            extraction_failed_count += 1
        if is_empty_output:
            empty_output_count += 1

        pred_dict = {
            'problem_id': pid,
            'description': desc[:500],  # Truncate for storage
            'generated': generated,
            'extracted_code': code,
            'num_tests_passed': num_passed,
            'num_tests_total': num_total,
            'all_passed': all_passed,
            'eval_mode': '0-shot chat (competitive programming)',
        }
        if was_truncated:
            pred_dict['truncated'] = True
        if is_extraction_failed:
            pred_dict['extraction_failed'] = True
        if is_empty_output:
            pred_dict['empty_output'] = True
        if had_thinking:
            pred_dict['had_thinking'] = True
        predictions.append(pred_dict)

    accuracy = correct / total if total > 0 else 0.0

    logger.info(f"\nResults (0-shot chat):")
    logger.info(f"  Passed: {correct}/{total}")
    logger.info(f"  pass@1: {accuracy:.4f} ({accuracy * 100:.2f}%)")

    pred_file = os.path.join(output_dir, PREDICTIONS_FILE)
    save_predictions(predictions, pred_file)

    return {
        'score': accuracy,
        'correct': correct,
        'total_samples': total,
        'eval_mode': '0-shot chat (competitive programming)',
        'truncated_count': truncated_count,
        'extraction_failed_count': extraction_failed_count,
        'empty_output_count': empty_output_count,
        'predictions': predictions,
    }


# ============================================================================
# GSM8K 5-shot Evaluator
# ============================================================================

# ============================================================================


def _build_gsm8k_fewshot_examples(dataset_path: str, n_shots: int = 5) -> str:
    """Build few-shot examples string from GSM8K training set.

    Answers are kept in GSM8K native #### format to match the official
    evaluation protocol.
    """
    dataset = load_from_disk(dataset_path)
    train_data = dataset['train']

    examples = []
    for i in range(n_shots):
        q = train_data[i]['question']
        a = train_data[i]['answer']
        examples.append(f"Question: {q}\nAnswer: {a}")

    return "\n\n".join(examples)


def evaluate_gsm8k_5shot(
    client: SGLangClient,
    model_name: str,
    dataset_path: str,
    split: str = "test",
    max_new_tokens: int = 2048,
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_concurrent: int = 256,
    output_dir: str = "./results",
    debug: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Evaluate GSM8K dataset using 5-shot chat mode with native #### format.

    Uses the first 5 training samples as few-shot examples prepended to each
    test question. Both few-shot examples and expected output use the official
    GSM8K #### answer format.
    """
    logger.info(f"Loading GSM8K dataset ({split} split)...")
    logger.info(f"Eval mode: 5-shot chat (####)")
    logger.info(f"Params: temperature={temperature}, top_p={top_p}, max_new_tokens={max_new_tokens}")

    dataset = load_from_disk(dataset_path)
    data = dataset[split]

    if debug:
        data = data.select(range(min(10, len(data))))
        logger.info(f"Debug mode: processing {len(data)} samples only")

    logger.info(f"Dataset size: {len(data)}")

    # Build few-shot prefix from training set
    fewshot_prefix = _build_gsm8k_fewshot_examples(dataset_path, n_shots=5)
    logger.info(f"Built 5-shot prefix from training set ({len(fewshot_prefix)} chars)")

    questions = []
    gold_answers = []
    for i in range(len(data)):
        sample = data[i]
        questions.append(sample['question'])
        gold_answers.append(sample['answer'])

    logger.info(f"Total {len(questions)} requests, starting concurrent inference...")
    client.max_concurrent = max_concurrent

    # 5-shot chat mode: prepend few-shot examples to each question
    # Use GSM8K native #### format (not \boxed{}) to match official protocol
    no_system_prompt = kwargs.get('no_system_prompt', False)
    messages_list = []
    for question in questions:
        user_content = fewshot_prefix + "\n\nQuestion: " + question + "\nAnswer:"
        if no_system_prompt:
            messages_list.append([
                {"role": "user", "content": GSM8K_SYSTEM_PROMPT + "\n\n" + user_content},
            ])
        else:
            messages_list.append([
                {"role": "system", "content": GSM8K_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ])
    generated_texts = client.chat_batch(
        messages_list=messages_list,
        model=model_name,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        show_progress=True,
    )

    predictions = []
    correct = 0
    total = 0
    truncated_count = 0
    extraction_failed_count = 0
    empty_output_count = 0

    for i in range(len(generated_texts)):
        generated = generated_texts[i]
        gold_answer = gold_answers[i]
        question = questions[i]

        stripped, had_thinking, was_truncated = strip_thinking_content(generated)

        pred_number = extract_number_from_answer(stripped)
        gold_number = extract_number_from_answer(gold_answer)

        is_correct = False
        if pred_number is not None and gold_number is not None:
            is_correct = abs(pred_number - gold_number) < 1e-6

        if is_correct:
            correct += 1
        total += 1

        is_empty_output = not generated or not generated.strip()
        is_extraction_failed = pred_number is None and not is_empty_output
        if was_truncated:
            truncated_count += 1
        if is_extraction_failed:
            extraction_failed_count += 1
        if is_empty_output:
            empty_output_count += 1

        pred_dict = {
            'question': question,
            'gold_answer': gold_answer,
            'gold_number': gold_number,
            'generated': generated,
            'pred_number': pred_number,
            'is_correct': is_correct,
            'eval_mode': '5-shot chat (####)',
        }
        if was_truncated:
            pred_dict['truncated'] = True
        if is_extraction_failed:
            pred_dict['extraction_failed'] = True
        if is_empty_output:
            pred_dict['empty_output'] = True
        if had_thinking:
            pred_dict['had_thinking'] = True
        predictions.append(pred_dict)

    accuracy = correct / total if total > 0 else 0.0

    logger.info(f"\nResults (5-shot chat):")
    logger.info(f"  Correct: {correct}/{total}")
    logger.info(f"  Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

    pred_file = os.path.join(output_dir, PREDICTIONS_FILE)
    save_predictions(predictions, pred_file)

    return {
        'score': accuracy,
        'correct': correct,
        'total_samples': total,
        'eval_mode': '5-shot chat (####)',
        'truncated_count': truncated_count,
        'extraction_failed_count': extraction_failed_count,
        'empty_output_count': empty_output_count,
        'predictions': predictions,
    }


# ============================================================================
# GSM8K 1-shot Evaluator
# ============================================================================

def evaluate_gsm8k_1shot(
    client: SGLangClient,
    model_name: str,
    dataset_path: str,
    split: str = "test",
    max_new_tokens: int = 2048,
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_concurrent: int = 256,
    output_dir: str = "./results",
    debug: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Evaluate GSM8K dataset using 1-shot chat mode with native #### format.

    Uses the first training sample as a single few-shot example prepended to
    each test question. Both the example and expected output use the official
    GSM8K #### answer format.
    """
    logger.info(f"Loading GSM8K dataset ({split} split)...")
    logger.info(f"Eval mode: 1-shot chat (####)")
    logger.info(f"Params: temperature={temperature}, top_p={top_p}, max_new_tokens={max_new_tokens}")

    dataset = load_from_disk(dataset_path)
    data = dataset[split]

    if debug:
        data = data.select(range(min(10, len(data))))
        logger.info(f"Debug mode: processing {len(data)} samples only")

    logger.info(f"Dataset size: {len(data)}")

    # Build 1-shot prefix from training set (reuse the helper with n_shots=1)
    fewshot_prefix = _build_gsm8k_fewshot_examples(dataset_path, n_shots=1)
    logger.info(f"Built 1-shot prefix from training set ({len(fewshot_prefix)} chars)")

    questions = []
    gold_answers = []
    for i in range(len(data)):
        sample = data[i]
        questions.append(sample['question'])
        gold_answers.append(sample['answer'])

    logger.info(f"Total {len(questions)} requests, starting concurrent inference...")
    client.max_concurrent = max_concurrent

    # 1-shot chat mode: prepend single example to each question
    # Use GSM8K native #### format (not \boxed{}) to match official protocol
    no_system_prompt = kwargs.get('no_system_prompt', False)
    messages_list = []
    for question in questions:
        user_content = fewshot_prefix + "\n\nQuestion: " + question + "\nAnswer:"
        if no_system_prompt:
            messages_list.append([
                {"role": "user", "content": GSM8K_SYSTEM_PROMPT + "\n\n" + user_content},
            ])
        else:
            messages_list.append([
                {"role": "system", "content": GSM8K_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ])
    generated_texts = client.chat_batch(
        messages_list=messages_list,
        model=model_name,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        show_progress=True,
    )

    predictions = []
    correct = 0
    total = 0
    truncated_count = 0
    extraction_failed_count = 0
    empty_output_count = 0

    for i in range(len(generated_texts)):
        generated = generated_texts[i]
        gold_answer = gold_answers[i]
        question = questions[i]

        stripped, had_thinking, was_truncated = strip_thinking_content(generated)

        pred_number = extract_number_from_answer(stripped)
        gold_number = extract_number_from_answer(gold_answer)

        is_correct = False
        if pred_number is not None and gold_number is not None:
            is_correct = abs(pred_number - gold_number) < 1e-6

        if is_correct:
            correct += 1
        total += 1

        is_empty_output = not generated or not generated.strip()
        is_extraction_failed = pred_number is None and not is_empty_output
        if was_truncated:
            truncated_count += 1
        if is_extraction_failed:
            extraction_failed_count += 1
        if is_empty_output:
            empty_output_count += 1

        pred_dict = {
            'question': question,
            'gold_answer': gold_answer,
            'gold_number': gold_number,
            'generated': generated,
            'pred_number': pred_number,
            'is_correct': is_correct,
            'eval_mode': '1-shot chat (####)',
        }
        if was_truncated:
            pred_dict['truncated'] = True
        if is_extraction_failed:
            pred_dict['extraction_failed'] = True
        if is_empty_output:
            pred_dict['empty_output'] = True
        if had_thinking:
            pred_dict['had_thinking'] = True
        predictions.append(pred_dict)

    accuracy = correct / total if total > 0 else 0.0

    logger.info(f"\nResults (1-shot chat):")
    logger.info(f"  Correct: {correct}/{total}")
    logger.info(f"  Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

    pred_file = os.path.join(output_dir, PREDICTIONS_FILE)
    save_predictions(predictions, pred_file)

    return {
        'score': accuracy,
        'correct': correct,
        'total_samples': total,
        'eval_mode': '1-shot chat (####)',
        'truncated_count': truncated_count,
        'extraction_failed_count': extraction_failed_count,
        'empty_output_count': empty_output_count,
        'predictions': predictions,
    }


# ============================================================================
# HumanEval 3-shot Function-Only Evaluator
# ============================================================================

# Hand-crafted 3-shot examples for HumanEval.
# Each example mirrors the HumanEval style: function signature + docstring with
# >>> test cases, followed by a canonical function body completion.
# These examples do NOT overlap with any HumanEval test problem.

HUMANEVAL_3SHOT_EXAMPLES = [
    {
        "prompt": '''from typing import List\n\n\ndef cumulative_sum(numbers: List[int]) -> List[int]:\n    """ Given a list of integers, return a new list where each element\n    is the cumulative sum up to that index.\n    >>> cumulative_sum([1, 2, 3, 4])\n    [1, 3, 6, 10]\n    >>> cumulative_sum([5])\n    [5]\n    >>> cumulative_sum([])\n    []\n    """\n''',
        "completion": '''    result = []\n    running = 0\n    for n in numbers:\n        running += n\n        result.append(running)\n    return result''',
    },
    {
        "prompt": '''def count_vowels(s: str) -> int:\n    """ Count the number of vowels (a, e, i, o, u) in the given string.\n    The function should be case-insensitive.\n    >>> count_vowels("hello")\n    2\n    >>> count_vowels("AEIOU")\n    5\n    >>> count_vowels("xyz")\n    0\n    """\n''',
        "completion": '''    return sum(1 for c in s.lower() if c in "aeiou")''',
    },
    {
        "prompt": '''from typing import List\n\n\ndef second_largest(numbers: List[int]) -> int:\n    """ Return the second largest unique number in the list.\n    The list is guaranteed to have at least two distinct values.\n    >>> second_largest([1, 3, 2, 5, 4])\n    4\n    >>> second_largest([10, 10, 5])\n    5\n    >>> second_largest([-1, -2, -3])\n    -2\n    """\n''',
        "completion": '''    unique = sorted(set(numbers), reverse=True)\n    return unique[1]''',
    },
]


def _build_humaneval_fewshot_prefix(examples: list = None) -> str:
    """Build the few-shot prefix string for HumanEval evaluation.

    Args:
        examples: List of example dicts with 'prompt' and 'completion' keys.
                  Defaults to HUMANEVAL_3SHOT_EXAMPLES if not provided.
    """
    if examples is None:
        examples = HUMANEVAL_3SHOT_EXAMPLES
    parts = []
    for ex in examples:
        parts.append(
            f"Complete the following Python function:\n\n{ex['prompt']}\n"
            f"Completion:\n{ex['completion']}"
        )
    return "\n\n---\n\n".join(parts)


def evaluate_humaneval_3shot(
    client: SGLangClient,
    model_name: str,
    dataset_path: str,
    split: str = "test",
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_concurrent: int = 256,
    output_dir: str = "./results",
    debug: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Evaluate HumanEval dataset using 3-shot function-only code completion.

    Uses 3 hand-crafted examples that closely match HumanEval style.
    Scoring logic is identical to the 0-shot version (functional correctness).
    """
    logger.info(f"Loading HumanEval dataset...")
    logger.info(f"Eval mode: 3-shot chat (function-only code completion)")
    logger.info(f"Params: temperature={temperature}, top_p={top_p}, max_new_tokens={max_new_tokens}")

    dataset = load_from_disk(dataset_path)
    if isinstance(dataset, dict):
        data = dataset.get(split, dataset[list(dataset.keys())[0]])
    else:
        data = dataset

    if debug:
        data = data.select(range(min(10, len(data))))
        logger.info(f"Debug mode: processing {len(data)} samples only")

    logger.info(f"Dataset size: {len(data)}")

    task_ids = []
    prompts = []
    for i in range(len(data)):
        sample = data[i]
        task_ids.append(sample['task_id'])
        prompts.append(sample['prompt'])

    # Build few-shot prefix
    fewshot_prefix = _build_humaneval_fewshot_prefix()
    logger.info(f"Built 3-shot prefix ({len(fewshot_prefix)} chars)")

    logger.info(f"Total {len(prompts)} requests, starting concurrent inference...")
    client.max_concurrent = max_concurrent

    # Build chat messages with few-shot prefix
    no_system_prompt = kwargs.get('no_system_prompt', False)
    messages_list = []
    for prompt in prompts:
        user_msg = (
            fewshot_prefix
            + "\n\n---\n\n"
            + f"Complete the following Python function:\n\n{prompt}\n"
            + "Completion:\n"
        )
        if no_system_prompt:
            messages_list.append([
                {"role": "user", "content": HUMANEVAL_SYSTEM_PROMPT + "\n\n" + user_msg},
            ])
        else:
            messages_list.append([
                {"role": "system", "content": HUMANEVAL_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ])

    generated_texts = client.chat_batch(
        messages_list=messages_list,
        model=model_name,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        show_progress=True,
    )

    # Build samples JSONL for human_eval evaluation
    samples_file = os.path.join(output_dir, "humaneval_samples.jsonl")
    predictions = []
    samples_for_eval = []
    truncated_count = 0
    extraction_failed_count = 0
    empty_output_count = 0

    for i in range(len(generated_texts)):
        generated = generated_texts[i]
        task_id = task_ids[i]
        prompt = prompts[i]

        stripped, had_thinking, was_truncated = strip_thinking_content(generated)
        completion = _extract_function_body(stripped, prompt)

        samples_for_eval.append({
            "task_id": task_id,
            "completion": completion,
        })

        is_empty_output = not generated or not generated.strip()
        is_extraction_failed = not completion.strip() and not is_empty_output
        if was_truncated:
            truncated_count += 1
        if is_extraction_failed:
            extraction_failed_count += 1
        if is_empty_output:
            empty_output_count += 1

        pred_dict = {
            "task_id": task_id,
            "prompt": prompt,
            "generated": generated,
            "completion": completion,
            "eval_mode": "3-shot chat (function-only code completion)",
        }
        if was_truncated:
            pred_dict['truncated'] = True
        if is_extraction_failed:
            pred_dict['extraction_failed'] = True
        if is_empty_output:
            pred_dict['empty_output'] = True
        if had_thinking:
            pred_dict['had_thinking'] = True
        predictions.append(pred_dict)

    # Write samples JSONL
    with open(samples_file, 'w') as f:
        for s in samples_for_eval:
            f.write(json.dumps(s) + '\n')

    # Save predictions
    pred_file = os.path.join(output_dir, PREDICTIONS_FILE)
    save_predictions(predictions, pred_file)

    # Run functional correctness evaluation using official human_eval
    logger.info("Running functional correctness evaluation (code execution)...")
    try:
        from human_eval.evaluation import evaluate_functional_correctness
        problem_file = os.path.join(dataset_path, "HumanEval.jsonl")
        if not os.path.exists(problem_file):
            from human_eval.data import HUMAN_EVAL
            problem_file = HUMAN_EVAL

        pass_at_k = evaluate_functional_correctness(
            sample_file=samples_file,
            k=[1],
            n_workers=8,
            timeout=10.0,
            problem_file=problem_file,
            ignore_incomplete=debug,
        )
        score = pass_at_k.get("pass@1", 0.0)
        logger.info(f"HumanEval pass@1: {score:.4f} ({score * 100:.2f}%)")
    except Exception as e:
        logger.error(f"Functional correctness evaluation failed: {e}")
        score = 0.0
        pass_at_k = {"pass@1": 0.0, "error": str(e)}

    return {
        'score': score,
        'pass_at_k': pass_at_k,
        'total_samples': len(predictions),
        'eval_mode': '3-shot chat (function-only code completion)',
        'truncated_count': truncated_count,
        'extraction_failed_count': extraction_failed_count,
        'empty_output_count': empty_output_count,
        'predictions': predictions,
    }


# ============================================================================
# HumanEval 1-shot Function-Only Evaluator
# ============================================================================

def evaluate_humaneval_1shot(
    client: SGLangClient,
    model_name: str,
    dataset_path: str,
    split: str = "test",
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_concurrent: int = 256,
    output_dir: str = "./results",
    debug: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Evaluate HumanEval dataset using 1-shot function-only code completion.

    Uses the first hand-crafted example from HUMANEVAL_3SHOT_EXAMPLES.
    Scoring logic is identical to the 0-shot version (functional correctness).
    """
    logger.info(f"Loading HumanEval dataset...")
    logger.info(f"Eval mode: 1-shot chat (function-only code completion)")
    logger.info(f"Params: temperature={temperature}, top_p={top_p}, max_new_tokens={max_new_tokens}")

    dataset = load_from_disk(dataset_path)
    if isinstance(dataset, dict):
        data = dataset.get(split, dataset[list(dataset.keys())[0]])
    else:
        data = dataset

    if debug:
        data = data.select(range(min(10, len(data))))
        logger.info(f"Debug mode: processing {len(data)} samples only")

    logger.info(f"Dataset size: {len(data)}")

    task_ids = []
    prompts = []
    for i in range(len(data)):
        sample = data[i]
        task_ids.append(sample['task_id'])
        prompts.append(sample['prompt'])

    # Build 1-shot prefix (use only the first example)
    fewshot_prefix = _build_humaneval_fewshot_prefix(examples=HUMANEVAL_3SHOT_EXAMPLES[:1])
    logger.info(f"Built 1-shot prefix ({len(fewshot_prefix)} chars)")

    logger.info(f"Total {len(prompts)} requests, starting concurrent inference...")
    client.max_concurrent = max_concurrent

    # Build chat messages with 1-shot prefix
    no_system_prompt = kwargs.get('no_system_prompt', False)
    messages_list = []
    for prompt in prompts:
        user_msg = (
            fewshot_prefix
            + "\n\n---\n\n"
            + f"Complete the following Python function:\n\n{prompt}\n"
            + "Completion:\n"
        )
        if no_system_prompt:
            messages_list.append([
                {"role": "user", "content": HUMANEVAL_SYSTEM_PROMPT + "\n\n" + user_msg},
            ])
        else:
            messages_list.append([
                {"role": "system", "content": HUMANEVAL_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ])

    generated_texts = client.chat_batch(
        messages_list=messages_list,
        model=model_name,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        show_progress=True,
    )

    # Build samples JSONL for human_eval evaluation
    samples_file = os.path.join(output_dir, "humaneval_samples.jsonl")
    predictions = []
    samples_for_eval = []
    truncated_count = 0
    extraction_failed_count = 0
    empty_output_count = 0

    for i in range(len(generated_texts)):
        generated = generated_texts[i]
        task_id = task_ids[i]
        prompt = prompts[i]

        stripped, had_thinking, was_truncated = strip_thinking_content(generated)
        completion = _extract_function_body(stripped, prompt)

        samples_for_eval.append({
            "task_id": task_id,
            "completion": completion,
        })

        is_empty_output = not generated or not generated.strip()
        is_extraction_failed = not completion.strip() and not is_empty_output
        if was_truncated:
            truncated_count += 1
        if is_extraction_failed:
            extraction_failed_count += 1
        if is_empty_output:
            empty_output_count += 1

        pred_dict = {
            "task_id": task_id,
            "prompt": prompt,
            "generated": generated,
            "completion": completion,
            "eval_mode": "1-shot chat (function-only code completion)",
        }
        if was_truncated:
            pred_dict['truncated'] = True
        if is_extraction_failed:
            pred_dict['extraction_failed'] = True
        if is_empty_output:
            pred_dict['empty_output'] = True
        if had_thinking:
            pred_dict['had_thinking'] = True
        predictions.append(pred_dict)

    # Write samples JSONL
    with open(samples_file, 'w') as f:
        for s in samples_for_eval:
            f.write(json.dumps(s) + '\n')

    # Save predictions
    pred_file = os.path.join(output_dir, PREDICTIONS_FILE)
    save_predictions(predictions, pred_file)

    # Run functional correctness evaluation using official human_eval
    logger.info("Running functional correctness evaluation (code execution)...")
    try:
        from human_eval.evaluation import evaluate_functional_correctness
        problem_file = os.path.join(dataset_path, "HumanEval.jsonl")
        if not os.path.exists(problem_file):
            from human_eval.data import HUMAN_EVAL
            problem_file = HUMAN_EVAL

        pass_at_k = evaluate_functional_correctness(
            sample_file=samples_file,
            k=[1],
            n_workers=8,
            timeout=10.0,
            problem_file=problem_file,
            ignore_incomplete=debug,
        )
        score = pass_at_k.get("pass@1", 0.0)
        logger.info(f"HumanEval pass@1: {score:.4f} ({score * 100:.2f}%)")
    except Exception as e:
        logger.error(f"Functional correctness evaluation failed: {e}")
        score = 0.0
        pass_at_k = {"pass@1": 0.0, "error": str(e)}

    return {
        'score': score,
        'pass_at_k': pass_at_k,
        'total_samples': len(predictions),
        'eval_mode': '1-shot chat (function-only code completion)',
        'truncated_count': truncated_count,
        'extraction_failed_count': extraction_failed_count,
        'empty_output_count': empty_output_count,
        'predictions': predictions,
    }


# ============================================================================
# Dataset evaluator registry
# ============================================================================
DATASET_EVALUATORS = {
    "gsm8k": evaluate_gsm8k,
    "gsm8k-1shot": evaluate_gsm8k_1shot,
    "gsm8k-5shot": evaluate_gsm8k_5shot,
    "math500": evaluate_math500,
    "aime24": evaluate_aime,
    "aime25": evaluate_aime,
    "aime26": evaluate_aime,
    "humaneval": evaluate_humaneval,
    "humaneval-1shot": evaluate_humaneval_1shot,
    "humaneval-3shot": evaluate_humaneval_3shot,
    "mbpp": evaluate_mbpp,
    "live-code-bench-v6": evaluate_lcb,
}


# ============================================================================
# CLI
# ============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="KDFlow Evaluation Script (SGLang DP inference engine)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Start SGLang server first (in another terminal, DP=8)
    python -m sglang.launch_server \\
        --model-path /path/to/model --dp-size 8 --tp-size 1 --port 30000

    # GSM8K evaluation
    python evaluation/evaluation.py --model_path /path/to/model --dataset gsm8k

    # MATH-500 evaluation
    python evaluation/evaluation.py --model_path /path/to/model --dataset math500

    # AIME evaluation
    python evaluation/evaluation.py --model_path /path/to/model --dataset aime24
    python evaluation/evaluation.py --model_path /path/to/model --dataset aime25
    python evaluation/evaluation.py --model_path /path/to/model --dataset aime26

    # Code evaluation
    python evaluation/evaluation.py --model_path /path/to/model --dataset humaneval
    python evaluation/evaluation.py --model_path /path/to/model --dataset mbpp
    python evaluation/evaluation.py --model_path /path/to/model --dataset live-code-bench-v6
        """,
    )

    parser.add_argument(
        "--model_path", type=str, required=True,
        help="Path to the model directory (used as model identifier)",
    )
    parser.add_argument(
        "--dataset", type=str, required=True,
        choices=list(DATASET_EVALUATORS.keys()),
        help=f"Dataset name, choices: {list(DATASET_EVALUATORS.keys())}",
    )
    parser.add_argument(
        "--base_url", type=str, default=DEFAULT_BASE_URL,
        help=f"SGLang server URL (default: {DEFAULT_BASE_URL})",
    )
    parser.add_argument(
        "--max_concurrent", type=int, default=256,
        help="Max concurrent requests (default: 256, recommended for DP=8)",
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=2048,
        help="Max new tokens to generate (default: 2048)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0,
        help="Generation temperature, 0 = greedy decoding (default: 0.0)",
    )
    parser.add_argument(
        "--top_p", type=float, default=1.0,
        help="Nucleus sampling top_p (default: 1.0)",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory (default: PROJECT_DIR/results)",
    )
    parser.add_argument(
        "--split", type=str, default="test",
        help="Dataset split (default: test)",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Debug mode, process only a few samples",
    )
    parser.add_argument(
        "--timeout", type=int, default=300,
        help="Per-request timeout in seconds (default: 300)",
    )
    parser.add_argument(
        "--no-system-prompt", action="store_true",
        help="Disable system prompt (merge into user message). "
             "Useful for models with chat template incompatibility.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve output directory
    # If --output_dir is specified, use it directly (caller handles path structure)
    # Otherwise, auto-build: RESULTS_DIR / model_basename / dataset
    model_name = os.path.basename(args.model_path.rstrip('/'))
    if args.output_dir:
        eval_output_dir = args.output_dir
    else:
        eval_output_dir = os.path.join(RESULTS_DIR, model_name, args.dataset)
    os.makedirs(eval_output_dir, exist_ok=True)

    print("=" * 80)
    print("KDFlow Evaluation (SGLang DP inference engine)")
    print("=" * 80)
    print(f"Model path: {args.model_path}")
    print(f"Model name: {model_name}")
    print(f"Dataset: {args.dataset}")
    print(f"Metric: {DATASET_METRICS.get(args.dataset, 'N/A')}")
    print(f"SGLang server: {args.base_url}")
    print(f"Max concurrent: {args.max_concurrent}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Temperature: {args.temperature}")
    print(f"Top_p: {args.top_p}")
    print(f"Split: {args.split}")
    print(f"Output dir: {eval_output_dir}")
    print("=" * 80)

    # Step 1: Check if prediction results already exist
    existing_pred = check_existing_results(eval_output_dir)
    if existing_pred:
        print(f"\n✓ Prediction results already exist: {existing_pred}")
        print("  Skipping evaluation. Delete the file to re-run.")
        return

    # Step 2: Check dataset path (resolve aliases for few-shot variants)
    dataset_dir_name = DATASET_PATH_ALIASES.get(args.dataset, args.dataset)
    dataset_path = os.path.join(DATASETS_DIR, dataset_dir_name)
    if not os.path.exists(dataset_path):
        print(f"Error: dataset path not found: {dataset_path}")
        print(f"Available datasets: {os.listdir(DATASETS_DIR)}")
        sys.exit(1)

    # Step 3: Create SGLang client and check server
    print("\nChecking SGLang server...")
    client = SGLangClient(
        base_url=args.base_url,
        max_concurrent=args.max_concurrent,
        timeout=args.timeout,
    )

    if not client.health_check():
        print(f"Error: SGLang server unavailable: {args.base_url}")
        print("Please start SGLang server first:")
        print(f"  python -m sglang.launch_server \\")
        print(f"      --model-path {args.model_path} \\")
        print(f"      --dp-size 8 --tp-size 1 --port 30000")
        sys.exit(1)

    served_model = client.get_model_name()
    print(f"✓ SGLang server ready, model: {served_model}")

    # Step 4: Run evaluation
    print(f"\nStarting {args.dataset} evaluation...")
    start_time = time.time()

    evaluator = DATASET_EVALUATORS[args.dataset]
    results = evaluator(
        client=client,
        model_name=served_model,
        dataset_path=dataset_path,
        split=args.split,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        max_concurrent=args.max_concurrent,
        output_dir=eval_output_dir,
        debug=args.debug,
        no_system_prompt=args.no_system_prompt,
    )

    elapsed = time.time() - start_time

    # Print results
    print("\n" + "=" * 80)
    print("Evaluation Results")
    print("=" * 80)
    print(f"Model: {model_name}")
    print(f"Dataset: {args.dataset}")
    print(f"Metric: {DATASET_METRICS[args.dataset]}")
    print(f"Score: {results['score']:.4f}")
    print(f"Total samples: {results['total_samples']}")
    print(f"Elapsed: {elapsed:.1f}s ({elapsed / 60:.1f}min)")
    print(f"Throughput: {results['total_samples'] / elapsed:.1f} samples/s")
    if results.get('eval_mode'):
        print(f"Eval mode: {results['eval_mode']}")

    # Print truncation and extraction statistics
    total_samples = results['total_samples']
    truncated_count = results.get('truncated_count', 0)
    extraction_failed_count = results.get('extraction_failed_count', 0)
    empty_output_count = results.get('empty_output_count', 0)
    truncated_ratio = truncated_count / total_samples if total_samples > 0 else 0.0
    empty_output_ratio = empty_output_count / total_samples if total_samples > 0 else 0.0

    if truncated_count or extraction_failed_count or empty_output_count:
        print(f"\nDiagnostics:")
        print(f"  Truncated: {truncated_count}/{total_samples} ({truncated_ratio * 100:.1f}%)")
        print(f"  Extraction failed: {extraction_failed_count}/{total_samples}")
        print(f"  Empty output: {empty_output_count}/{total_samples} ({empty_output_ratio * 100:.1f}%)")

    if truncated_ratio > 0.1:
        print(f"\n⚠️  WARNING: Truncation ratio ({truncated_ratio * 100:.1f}%) exceeds 10%!")
        print(f"   Consider increasing --max_new_tokens (current: {args.max_new_tokens})")
    if empty_output_ratio > 0.5:
        print(f"\n🚨 CRITICAL WARNING: Empty output ratio ({empty_output_ratio * 100:.1f}%) exceeds 50%!")
        print(f"   This likely indicates a chat template incompatibility or server issue.")
        print(f"   Try: --no-system-prompt or check SGLang server logs.")

    print("=" * 80)

    # Save aggregated metrics
    result_file = os.path.join(eval_output_dir, METRICS_FILE)
    metrics_data = {
        'model': model_name,
        'model_path': args.model_path,
        'served_model': served_model,
        'dataset': args.dataset,
        'metric': DATASET_METRICS.get(args.dataset, 'N/A'),
        'score': results['score'],
        'total_samples': total_samples,
        'elapsed_seconds': elapsed,
        'throughput_sps': total_samples / elapsed if elapsed > 0 else 0.0,
        'eval_mode': results.get('eval_mode', 'default'),
        'system_role_supported': client._supports_system_role,
        'truncated_count': truncated_count,
        'truncated_ratio': round(truncated_ratio, 4),
        'extraction_failed_count': extraction_failed_count,
        'empty_output_count': empty_output_count,
        'timestamp': datetime.now().isoformat(),
        'config': {
            'base_url': args.base_url,
            'max_concurrent': args.max_concurrent,
            'max_new_tokens': args.max_new_tokens,
            'temperature': args.temperature,
            'top_p': args.top_p,
            'split': args.split,
        },
    }
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(metrics_data, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {result_file}")

    # ---- Anomaly detection ----
    score = results['score']
    extraction_failed_ratio = extraction_failed_count / total_samples if total_samples > 0 else 0.0

    anomaly_detected = False

    if score == 0 and total_samples > 0:
        anomaly_detected = True
        print(f"\n⚠️  WARNING: Score is 0! This likely indicates a configuration issue, not model capability.")
        if empty_output_ratio > 0.5:
            print(f"   → High empty output ratio ({empty_output_ratio * 100:.1f}%): check chat template compatibility.")
            print(f"   → Try: --no-system-prompt")
        elif truncated_ratio > 0.1:
            print(f"   → High truncation ratio ({truncated_ratio * 100:.1f}%): increase --max_new_tokens (current: {args.max_new_tokens})")
        elif extraction_failed_ratio > 0.5:
            print(f"   → High extraction failure ratio ({extraction_failed_ratio * 100:.1f}%): check answer format.")
    elif score < 0.05 and total_samples > 10:
        anomaly_detected = True
        print(f"\n⚠️  WARNING: Score ({score * 100:.2f}%) is suspiciously low for {total_samples} samples.")
        if truncated_ratio > 0.1:
            print(f"   → Truncation ratio is {truncated_ratio * 100:.1f}%: consider increasing --max_new_tokens (current: {args.max_new_tokens})")
        if extraction_failed_ratio > 0.3:
            print(f"   → Extraction failure ratio is {extraction_failed_ratio * 100:.1f}%: answer format may not match extraction logic.")
        if empty_output_ratio > 0.3:
            print(f"   → Empty output ratio is {empty_output_ratio * 100:.1f}%: check server logs and chat template.")

    if truncated_ratio > 0.2 and not anomaly_detected:
        print(f"\n⚠️  WARNING: High truncation ratio ({truncated_ratio * 100:.1f}%).")
        print(f"   Suggestion: increase --max_new_tokens (current: {args.max_new_tokens})")

    if extraction_failed_ratio > 0.3 and not anomaly_detected:
        print(f"\n⚠️  WARNING: High extraction failure ratio ({extraction_failed_ratio * 100:.1f}%).")
        print(f"   The model may be outputting answers in an unexpected format.")


if __name__ == "__main__":
    main()
