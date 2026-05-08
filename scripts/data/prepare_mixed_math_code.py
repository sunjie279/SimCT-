#!/usr/bin/env python3
"""
Prepare mixed math+code 10k dataset from 8 sources.

=== Math (5,800) ===
  1) GSM8K train:          1,800
  2) Orca-Math 200K:       2,200
  3) OpenMathInstruct-1:   1,000
  4) MATH minus MATH-500:    800

=== Code (4,200) ===
  5) OpenCodeInstruct:     1,800
  6) KodCode:                900
  7) TACO:                   900
  8) CodeContests:           600

Total: 10,000

Output format (KDFlow):
  - messages: [{"role": "user", "content": "<question>"}]
  - label: the ground truth answer/solution

Output: $DATA_PATH/mixed_math_code_10k (set via environment variable, default: ./data)
"""

import os
import re
import json
import random
import logging
from collections import Counter
from datasets import Dataset, load_from_disk, load_dataset
from tqdm import tqdm

# ============================================================================
# Configuration
# ============================================================================
OUTPUT_BASE = os.environ.get("DATA_PATH", "./data")
OUTPUT_DIR = os.path.join(OUTPUT_BASE, "mixed_math_code_10k")
OUTPUT_DIR_WITH_SOURCE = os.path.join(OUTPUT_BASE, "mixed_math_code_10k_with_source")

# All data paths (local). Run download_datasets.py first to download HF datasets.
GSM8K_LOCAL = os.path.join(OUTPUT_BASE, "gsm8k")
ORCA_MATH_LOCAL = os.path.join(OUTPUT_BASE, "orca_math_for_kdflow")
TACO_LOCAL = os.path.join(OUTPUT_BASE, "taco_raw")
OPEN_MATH_INSTRUCT_LOCAL = os.path.join(OUTPUT_BASE, "open_math_instruct_raw")
MATH_FULL_LOCAL = os.path.join(OUTPUT_BASE, "math_full_raw")
MATH_TEST_LOCAL = os.path.join(OUTPUT_BASE, "math_full_test")
OPEN_CODE_INSTRUCT_LOCAL = os.path.join(OUTPUT_BASE, "open_code_instruct_raw")
KODCODE_LOCAL = os.path.join(OUTPUT_BASE, "kodcode_raw")
CODE_CONTESTS_LOCAL = os.path.join(OUTPUT_BASE, "code_contests_raw")

SEED = 42
HF_CACHE = os.path.join(OUTPUT_BASE, ".hf_cache")

# Length thresholds (in characters)
MAX_ANSWER_LEN_GSM8K = 2000       # filter out abnormally long GSM8K answers
MAX_QUESTION_LEN_ORCA = 500       # prefer shorter Orca-Math questions
MAX_ANSWER_LEN_ORCA = 2000        # filter out very long Orca-Math answers
MAX_ANSWER_LEN_OMI = 3000         # OpenMathInstruct answer length cap
MAX_QUESTION_LEN_CODE = 3000      # code question length cap
MAX_SOLUTION_LEN_CODE = 5000      # code solution length cap
MAX_QUESTION_LEN_CONTEST = 4000   # contest question length cap
MAX_SOLUTION_LEN_CONTEST = 5000   # contest solution length cap

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ============================================================================
# Helper functions
# ============================================================================
def make_record(question: str, answer: str, source: str):
    """Create a KDFlow-format record."""
    return {
        "messages": [{"role": "user", "content": question.strip()}],
        "label": answer.strip(),
        "source": source,
    }


def has_asymptote(text: str) -> bool:
    """Check if text contains Asymptote code (graphics-dependent)."""
    return "[asy]" in text.lower() or "\\begin{asy}" in text.lower()


def code_ratio(text: str) -> float:
    """Estimate the ratio of code-like content in text."""
    lines = text.strip().split("\n")
    if not lines:
        return 0.0
    code_lines = sum(1 for l in lines if l.strip().startswith((">>>", "```", "def ", "import ", "from ", "print(", "    ")))
    return code_lines / len(lines)


def extract_python_solution(solutions_str: str) -> str:
    """Extract a single Python solution from TACO/CodeContests solutions field."""
    if not solutions_str:
        return ""
    try:
        sols = json.loads(solutions_str)
        if isinstance(sols, list):
            # Pick the shortest reasonable Python solution
            valid = [s for s in sols if isinstance(s, str) and len(s) < MAX_SOLUTION_LEN_CONTEST and len(s) > 20]
            if valid:
                return min(valid, key=len)
        elif isinstance(sols, str):
            return sols
    except (json.JSONDecodeError, TypeError):
        if isinstance(solutions_str, str) and len(solutions_str) < MAX_SOLUTION_LEN_CONTEST:
            return solutions_str
    return ""


def is_interactive_problem(text: str) -> bool:
    """Heuristic: check if a competitive programming problem is interactive."""
    lower = text.lower()
    return any(kw in lower for kw in ["interactive", "interactor", "flush", "fflush"])


# ============================================================================
# Source 1: GSM8K train (1,800)
# ============================================================================
def load_gsm8k(target: int = 1800) -> list:
    log.info(f"[1/8] Loading GSM8K train -> target {target}")

    # GSM8K local: DatasetDict with train/test splits
    # Original format: question + answer (not messages+label)
    ds = load_from_disk(os.path.join(GSM8K_LOCAL, "train"))
    log.info(f"  Raw train size: {len(ds)}")

    candidates = []
    for i in range(len(ds)):
        sample = ds[i]
        q = sample.get("question", "")
        a = sample.get("answer", "")

        if not q or not a:
            continue
        # Filter abnormally long answers
        if len(a) > MAX_ANSWER_LEN_GSM8K:
            continue
        candidates.append(make_record(q, a, "gsm8k"))

    log.info(f"  After filtering: {len(candidates)}")
    random.shuffle(candidates)
    result = candidates[:target]
    log.info(f"  Sampled: {len(result)}")
    return result


# ============================================================================
# Source 2: Orca-Math 200K (2,200)
# ============================================================================
def load_orca_math(target: int = 2200) -> list:
    log.info(f"[2/8] Loading Orca-Math -> target {target}")

    ds = load_from_disk(ORCA_MATH_LOCAL)
    log.info(f"  Raw size: {len(ds)}")

    candidates = []
    for i in range(len(ds)):
        sample = ds[i]
        msgs = sample.get("messages", [])
        label = sample.get("label", "")

        if not msgs or not label:
            continue

        q = msgs[0]["content"] if msgs else ""
        a = label

        if not q or not a:
            continue
        # Prefer shorter questions (grade-school style)
        if len(q) > MAX_QUESTION_LEN_ORCA:
            continue
        # Filter overly long answers
        if len(a) > MAX_ANSWER_LEN_ORCA:
            continue

        candidates.append(make_record(q, a, "orca_math"))

    log.info(f"  After filtering: {len(candidates)}")

    # Sort by answer length (prefer concise) and sample
    candidates.sort(key=lambda x: len(x["label"]))
    # Take from the shorter half preferentially, but still randomize
    shorter_half = candidates[:len(candidates) * 2 // 3]
    longer_half = candidates[len(candidates) * 2 // 3:]
    random.shuffle(shorter_half)
    random.shuffle(longer_half)

    # 80% from shorter, 20% from longer
    n_short = min(int(target * 0.8), len(shorter_half))
    n_long = min(target - n_short, len(longer_half))
    result = shorter_half[:n_short] + longer_half[:n_long]

    # If still not enough, fill from remaining
    if len(result) < target:
        remaining = shorter_half[n_short:] + longer_half[n_long:]
        result += remaining[:target - len(result)]

    random.shuffle(result)
    log.info(f"  Sampled: {len(result)}")
    return result


# ============================================================================
# Source 3: OpenMathInstruct-1 (1,000)
# ============================================================================
def load_open_math_instruct(target: int = 1000) -> list:
    log.info(f"[3/8] Loading OpenMathInstruct-1 -> target {target}")

    ds = load_from_disk(OPEN_MATH_INSTRUCT_LOCAL)
    log.info(f"  Raw size: {len(ds)}")

    candidates = []
    for i in tqdm(range(len(ds)), desc="  Filtering OpenMathInstruct"):
        sample = ds[i]
        # Fields: question, generated_solution, expected_answer, ...
        q = sample.get("question", "") or sample.get("problem", "")
        a = sample.get("generated_solution", "") or sample.get("solution", "")

        if not q or not a:
            continue
        # Skip samples with too much code
        if code_ratio(a) > 0.4:
            continue
        # Skip overly long reasoning chains
        if len(a) > MAX_ANSWER_LEN_OMI:
            continue
        # Skip very short answers (likely broken)
        if len(a) < 50:
            continue

        candidates.append(make_record(q, a, "open_math_instruct"))

    log.info(f"  After filtering: {len(candidates)}")
    random.shuffle(candidates)
    result = candidates[:target]
    log.info(f"  Sampled: {len(result)}")
    return result


# ============================================================================
# Source 4: MATH minus MATH-500 (800)
# ============================================================================
def load_math_test_problems() -> set:
    """Load MATH test problems for deduplication (avoid test leakage)."""
    problems = set()
    if os.path.exists(MATH_TEST_LOCAL):
        try:
            ds_test = load_from_disk(MATH_TEST_LOCAL)
            for i in range(len(ds_test)):
                p = ds_test[i].get("problem", "").strip()
                if p:
                    problems.add(p)
            log.info(f"  Loaded {len(problems)} MATH test problems for dedup")
        except Exception as e:
            log.warning(f"  Could not load MATH test for dedup: {e}")
    else:
        log.warning(f"  MATH test not found at {MATH_TEST_LOCAL}, skipping dedup")
    return problems


def load_math_minus_500(target: int = 800) -> list:
    log.info(f"[4/8] Loading MATH (EleutherAI/hendrycks_math) minus test -> target {target}")

    ds = load_from_disk(MATH_FULL_LOCAL)
    log.info(f"  Raw size: {len(ds)}")

    # Load MATH test for deduplication
    math_test_problems = load_math_test_problems()

    # Organize by level and subject
    by_level = {1: [], 2: [], 3: [], 4: [], 5: []}
    skipped_math500 = 0
    for i in tqdm(range(len(ds)), desc="  Filtering MATH"):
        sample = ds[i]
        q = sample.get("problem", "")
        a = sample.get("solution", "")
        level_str = sample.get("level", "")
        subject = sample.get("type", "") or sample.get("subject", "")

        if not q or not a:
            continue
        # Deduplicate against MATH test set
        if q.strip() in math_test_problems:
            skipped_math500 += 1
            continue
        # Skip Asymptote/graphics-heavy problems
        if has_asymptote(q) or has_asymptote(a):
            continue

        # Parse level
        level = 0
        if isinstance(level_str, (int, float)):
            level = int(level_str)
        elif isinstance(level_str, str):
            m = re.search(r"(\d)", level_str)
            if m:
                level = int(m.group(1))

        if level < 1 or level > 5:
            continue

        by_level[level].append({
            "q": q, "a": a, "level": level, "subject": subject
        })

    log.info(f"  Skipped {skipped_math500} MATH test duplicates")
    for lv in range(1, 6):
        log.info(f"  Level {lv}: {len(by_level[lv])} candidates")

    # Target distribution: level 1-2: 300, level 3: 350, level 4: 150, level 5: 0
    level_targets = {1: 150, 2: 150, 3: 350, 4: 150, 5: 0}

    result = []
    for lv, tgt in level_targets.items():
        if tgt == 0:
            continue
        pool = by_level[lv]
        # Try to balance subjects within each level
        by_subj = {}
        for item in pool:
            subj = item["subject"] or "unknown"
            by_subj.setdefault(subj, []).append(item)

        subjects = list(by_subj.keys())
        per_subj = max(1, tgt // len(subjects)) if subjects else tgt
        level_samples = []

        for subj in subjects:
            random.shuffle(by_subj[subj])
            level_samples.extend(by_subj[subj][:per_subj])

        random.shuffle(level_samples)
        level_samples = level_samples[:tgt]

        # If not enough from balanced sampling, fill randomly
        if len(level_samples) < tgt:
            all_remaining = [x for x in pool if x not in level_samples]
            random.shuffle(all_remaining)
            level_samples.extend(all_remaining[:tgt - len(level_samples)])

        for item in level_samples:
            result.append(make_record(item["q"], item["a"], "math_minus_500"))

    random.shuffle(result)
    log.info(f"  Sampled: {len(result)} (target: {target})")
    return result[:target]


# ============================================================================
# Source 5: OpenCodeInstruct (1,800)
# ============================================================================
def load_open_code_instruct(target: int = 1800) -> list:
    log.info(f"[5/8] Loading OpenCodeInstruct -> target {target}")

    ds = load_from_disk(OPEN_CODE_INSTRUCT_LOCAL)
    log.info(f"  Raw size: {len(ds)}")

    candidates = []
    seen = 0
    max_scan = min(200000, len(ds))  # scan up to 200k samples

    for i in tqdm(range(max_scan), desc="  Scanning OpenCodeInstruct"):
        sample = ds[i]
        seen += 1

        # OpenCodeInstruct fields: input (question), output (solution), domain
        q = sample.get("input", "")
        a = sample.get("output", "")

        if not q or not a:
            continue

        # Filter: skip too long
        if len(q) > MAX_QUESTION_LEN_CODE or len(a) > MAX_SOLUTION_LEN_CODE:
            continue
        # Filter: skip too short
        if len(q) < 30 or len(a) < 30:
            continue

        # Skip multi-file / engineering context
        q_lower = q.lower()
        if any(kw in q_lower for kw in ["sql", "html", "css", "javascript", "react",
                                          "django", "flask", "docker", "kubernetes",
                                          "debug", "fix the", "fix this", "refactor"]):
            continue

        # Skip if solution doesn't look like Python code
        # OpenCodeInstruct output often starts with ```python
        if "def " not in a and "class " not in a and "import " not in a and "python" not in a.lower()[:50]:
            continue

        # Strip markdown code fences if present
        a_clean = a.strip()
        if a_clean.startswith("```python"):
            a_clean = a_clean[len("```python"):].strip()
        if a_clean.startswith("```"):
            a_clean = a_clean[3:].strip()
        if a_clean.endswith("```"):
            a_clean = a_clean[:-3].strip()
        a = a_clean

        candidates.append(make_record(q, a, "open_code_instruct"))

        # Early stop if we have enough
        if len(candidates) >= target * 3:
            break

    log.info(f"  Scanned {seen}, after filtering: {len(candidates)}")
    random.shuffle(candidates)
    result = candidates[:target]
    log.info(f"  Sampled: {len(result)}")
    return result


# ============================================================================
# Source 6: KodCode (900)
# ============================================================================
def load_kodcode(target: int = 900) -> list:
    log.info(f"[6/8] Loading KodCode -> target {target}")

    ds = load_from_disk(KODCODE_LOCAL)
    log.info(f"  Raw size: {len(ds)}")

    candidates = []
    for i in tqdm(range(len(ds)), desc="  Filtering KodCode"):
        sample = ds[i]

        # KodCode has question, solution, test fields
        q = sample.get("question", "") or sample.get("problem", "") or sample.get("instruction", "")
        a = sample.get("solution", "") or sample.get("output", "") or sample.get("response", "")

        if not q or not a:
            continue

        # Only Python
        lang = (sample.get("language", "") or sample.get("lang", "") or "").lower()
        # If language field exists, check it; otherwise check solution content
        if lang and "python" not in lang:
            continue

        # Filter by length
        if len(q) > MAX_QUESTION_LEN_CODE or len(a) > MAX_SOLUTION_LEN_CODE:
            continue
        if len(q) < 20 or len(a) < 20:
            continue

        # Prefer function-level solutions
        if "def " not in a:
            continue

        candidates.append(make_record(q, a, "kodcode"))

    log.info(f"  After filtering: {len(candidates)}")

    # Sort by length (prefer medium-length, not too short or too long)
    candidates.sort(key=lambda x: len(x["label"]))
    # Take from the middle range
    n = len(candidates)
    if n > target * 2:
        start = n // 6
        end = n * 5 // 6
        mid_candidates = candidates[start:end]
    else:
        mid_candidates = candidates

    random.shuffle(mid_candidates)
    result = mid_candidates[:target]
    log.info(f"  Sampled: {len(result)}")
    return result


# ============================================================================
# Source 7: TACO (900)
# ============================================================================
def load_taco(target: int = 900) -> list:
    log.info(f"[7/8] Loading TACO -> target {target}")

    # TACO local data is a single Dataset (not DatasetDict), files directly in taco_raw/
    ds = load_from_disk(TACO_LOCAL)
    log.info(f"  Raw size: {len(ds)}")

    # Organize by difficulty
    easy_medium = []
    hard = []

    for i in tqdm(range(len(ds)), desc="  Filtering TACO"):
        sample = ds[i]
        q = sample.get("question", "")
        solutions_str = sample.get("solutions", "")
        difficulty = (sample.get("difficulty", "") or "").lower().strip()

        if not q:
            continue
        # Skip interactive problems
        if is_interactive_problem(q):
            continue
        # Skip very long questions
        if len(q) > MAX_QUESTION_LEN_CONTEST:
            continue

        # Extract a Python solution
        sol = extract_python_solution(solutions_str)
        if not sol or len(sol) < 20:
            continue
        if len(sol) > MAX_SOLUTION_LEN_CONTEST:
            continue

        record = make_record(q, sol, "taco")

        # Classify difficulty
        if any(d in difficulty for d in ["easy", "medium", "medium_hard",
                                          "introductory", "interview",
                                          "a", "b", "1", "2"]):
            easy_medium.append(record)
        elif any(d in difficulty for d in ["hard", "very_hard", "competition",
                                            "c", "d", "e", "3", "4", "5"]):
            hard.append(record)
        else:
            # Unknown difficulty -> treat as medium
            easy_medium.append(record)

    log.info(f"  Easy/Medium: {len(easy_medium)}, Hard: {len(hard)}")

    # Target: easy/medium 650, hard 250
    random.shuffle(easy_medium)
    random.shuffle(hard)
    n_easy = min(650, len(easy_medium))
    n_hard = min(250, len(hard))

    result = easy_medium[:n_easy] + hard[:n_hard]

    # Fill if not enough
    if len(result) < target:
        remaining = easy_medium[n_easy:] + hard[n_hard:]
        random.shuffle(remaining)
        result += remaining[:target - len(result)]

    random.shuffle(result)
    result = result[:target]
    log.info(f"  Sampled: {len(result)}")
    return result


# ============================================================================
# Source 8: CodeContests (600)
# ============================================================================
def load_code_contests(target: int = 600) -> list:
    log.info(f"[8/8] Loading CodeContests -> target {target}")

    ds = load_from_disk(CODE_CONTESTS_LOCAL)
    log.info(f"  Raw size: {len(ds)}")

    easy_medium = []
    hard = []

    for i in tqdm(range(len(ds)), desc="  Filtering CodeContests"):
        sample = ds[i]
        q = sample.get("description", "") or sample.get("problem", "")
        difficulty = sample.get("difficulty", 0)
        solutions = sample.get("solutions", {})

        if not q:
            continue
        # Skip interactive
        if is_interactive_problem(q):
            continue
        # Skip very long questions
        if len(q) > MAX_QUESTION_LEN_CONTEST:
            continue

        # Extract Python solution
        sol = ""
        # CodeContests stores solutions as {"language": [...], "solution": [...]}
        if isinstance(solutions, dict):
            langs = solutions.get("language", [])
            sols = solutions.get("solution", [])
            for lang_id, s in zip(langs, sols):
                # Python language ID is typically 3
                if lang_id == 3 or (isinstance(lang_id, str) and "python" in str(lang_id).lower()):
                    if isinstance(s, str) and 20 < len(s) < MAX_SOLUTION_LEN_CONTEST:
                        sol = s
                        break
        elif isinstance(solutions, list):
            for s in solutions:
                if isinstance(s, str) and "def " in s and 20 < len(s) < MAX_SOLUTION_LEN_CONTEST:
                    sol = s
                    break

        if not sol:
            continue

        record = make_record(q, sol, "code_contests")

        # Classify difficulty (CodeContests uses numeric difficulty)
        if isinstance(difficulty, (int, float)):
            if difficulty <= 1500:
                easy_medium.append(record)
            else:
                hard.append(record)
        else:
            easy_medium.append(record)

    log.info(f"  Easy/Medium: {len(easy_medium)}, Hard: {len(hard)}")

    # Target: easy/medium 400, hard 200
    random.shuffle(easy_medium)
    random.shuffle(hard)
    n_easy = min(400, len(easy_medium))
    n_hard = min(200, len(hard))

    result = easy_medium[:n_easy] + hard[:n_hard]

    # Fill if not enough
    if len(result) < target:
        remaining = easy_medium[n_easy:] + hard[n_hard:]
        random.shuffle(remaining)
        result += remaining[:target - len(result)]

    random.shuffle(result)
    result = result[:target]
    log.info(f"  Sampled: {len(result)}")
    return result


# ============================================================================
# Main
# ============================================================================
def main():
    random.seed(SEED)

    print("=" * 70)
    print("Preparing Mixed Math+Code 10k Dataset")
    print("=" * 70)
    print("  Math (5,800):")
    print("    GSM8K train:          1,800")
    print("    Orca-Math 200K:       2,200")
    print("    OpenMathInstruct-1:   1,000")
    print("    MATH minus MATH-500:    800")
    print("  Code (4,200):")
    print("    OpenCodeInstruct:     1,800")
    print("    KodCode:                900")
    print("    TACO:                   900")
    print("    CodeContests:           600")
    print(f"  Total target: 10,000")
    print(f"  Output: {OUTPUT_DIR}")
    print("=" * 70)

    # Verify all local datasets exist
    required = {
        "GSM8K": os.path.join(GSM8K_LOCAL, "train"),
        "Orca-Math": ORCA_MATH_LOCAL,
        "OpenMathInstruct-1": OPEN_MATH_INSTRUCT_LOCAL,
        "MATH-Full": MATH_FULL_LOCAL,
        "OpenCodeInstruct": OPEN_CODE_INSTRUCT_LOCAL,
        "KodCode": KODCODE_LOCAL,
        "TACO": TACO_LOCAL,
        "CodeContests": CODE_CONTESTS_LOCAL,
    }
    missing = [name for name, path in required.items() if not os.path.exists(path)]
    if missing:
        print(f"\n  ✗ Missing local datasets: {missing}")
        print(f"    Run download_datasets.py first to download them.")
        return
    print("  ✓ All local datasets found.")

    # ---- Load all sources ----
    all_records = []

    # Math sources
    gsm8k_data = load_gsm8k(1800)
    all_records.extend(gsm8k_data)

    orca_data = load_orca_math(2200)
    all_records.extend(orca_data)

    omi_data = load_open_math_instruct(1000)
    all_records.extend(omi_data)

    math_data = load_math_minus_500(800)
    all_records.extend(math_data)

    # Code sources
    oci_data = load_open_code_instruct(1800)
    all_records.extend(oci_data)

    kodcode_data = load_kodcode(900)
    all_records.extend(kodcode_data)

    taco_data = load_taco(900)
    all_records.extend(taco_data)

    cc_data = load_code_contests(600)
    all_records.extend(cc_data)

    # ---- Summary before save ----
    print("\n" + "=" * 70)
    print("Collection Summary:")
    source_counts = Counter(r["source"] for r in all_records)
    for src, cnt in sorted(source_counts.items()):
        print(f"  {src}: {cnt}")
    print(f"  TOTAL: {len(all_records)}")
    print("=" * 70)

    # ---- Shuffle ----
    random.shuffle(all_records)

    # ---- Build and save ----
    print("\n[Save] Building datasets...")

    # KDFlow format (without source)
    kdflow_records = [{"messages": r["messages"], "label": r["label"]} for r in all_records]
    kdflow_ds = Dataset.from_list(kdflow_records)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    kdflow_ds.save_to_disk(OUTPUT_DIR)
    print(f"  ✓ Saved {len(kdflow_ds)} samples to {OUTPUT_DIR}")
    print(f"    Features: {kdflow_ds.features}")

    # Full version with source (for analysis)
    full_ds = Dataset.from_list(all_records)
    os.makedirs(OUTPUT_DIR_WITH_SOURCE, exist_ok=True)
    full_ds.save_to_disk(OUTPUT_DIR_WITH_SOURCE)
    print(f"  ✓ Saved full version (with source) to {OUTPUT_DIR_WITH_SOURCE}")

    # ---- Verify ----
    print("\n[Verify] Loading saved dataset...")
    verify_ds = load_from_disk(OUTPUT_DIR)
    print(f"  Loaded {len(verify_ds)} samples")
    print(f"  Features: {verify_ds.features}")

    verify_full = load_from_disk(OUTPUT_DIR_WITH_SOURCE)
    verify_dist = Counter(verify_full[i]["source"] for i in range(len(verify_full)))
    print(f"  Source distribution:")
    for src, cnt in sorted(verify_dist.items()):
        print(f"    {src}: {cnt}")

    # Show sample from each source
    print("\n  Sample previews:")
    shown_sources = set()
    for i in range(len(verify_full)):
        src = verify_full[i]["source"]
        if src not in shown_sources:
            shown_sources.add(src)
            content = verify_full[i]["messages"][0]["content"][:100]
            label = verify_full[i]["label"][:80]
            print(f"  [{src}] Q: {content}...")
            print(f"         A: {label}...")
        if len(shown_sources) == len(verify_dist):
            break

    print("\n" + "=" * 70)
    print(f"✓ Done! Mixed 10k dataset ready for KDFlow training.")
    print(f"  Path: {OUTPUT_DIR}")
    print(f"  Size: {len(verify_ds)} samples")
    math_total = sum(v for k, v in verify_dist.items()
                     if k in ("gsm8k", "orca_math", "open_math_instruct", "math_minus_500"))
    code_total = sum(v for k, v in verify_dist.items()
                     if k in ("open_code_instruct", "kodcode", "taco", "code_contests"))
    print(f"  Math: {math_total} | Code: {code_total}")
    print("=" * 70)


if __name__ == "__main__":
    main()
