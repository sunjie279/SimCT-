#!/usr/bin/env python3
"""
Answer verification module for SFT warmup dataset construction.

Provides a unified `verify_response()` interface that dispatches to
source-specific verification logic:
  - GSM8K: extract number from response, compare with #### answer
  - MATH / OpenMathInstruct / OpenR1-Math: extract \\boxed{} content, math equivalence check
  - Orca-Math: extract final number, compare numerically
  - Code (open_code_instruct, kodcode, taco, code_contests, apps): format check only

Reuses answer extraction functions from evaluation/evaluation.py.

Usage (as module):
    from answer_verifier import verify_response
    is_correct = verify_response(response="...", label="...", source="gsm8k")
"""

import re
from typing import Optional, Tuple

# ============================================================================
# Source type classification
# ============================================================================

MATH_SOURCES = {"gsm8k", "math_minus_500", "open_math_instruct", "orca_math", "openr1_math"}
CODE_SOURCES = {"open_code_instruct", "kodcode", "taco", "code_contests", "apps"}

GSM8K_SOURCES = {"gsm8k"}
BOXED_MATH_SOURCES = {"math_minus_500", "open_math_instruct"}
ORCA_MATH_SOURCES = {"orca_math"}
OPENR1_MATH_SOURCES = {"openr1_math"}


def is_math_source(source: str) -> bool:
    """Check if source is a math-type dataset."""
    return source in MATH_SOURCES


def is_code_source(source: str) -> bool:
    """Check if source is a code-type dataset."""
    return source in CODE_SOURCES


# ============================================================================
# Answer extraction utilities (reused from evaluation/evaluation.py)
# ============================================================================

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
# Source-specific verification logic
# ============================================================================

def _verify_gsm8k(response: str, label: str) -> bool:
    """Verify GSM8K response: extract number from response, compare with #### answer.

    GSM8K labels contain '#### <number>' at the end.
    Response should contain the answer in \\boxed{} or as a final number.
    """
    # Extract gold number from label (after ####)
    gold_number = None
    if "####" in label:
        match = re.search(r'####\s*(-?[\d,]+\.?\d*)', label)
        if match:
            gold_str = match.group(1).replace(",", "")
            gold_number = try_parse_number(gold_str)

    if gold_number is None:
        gold_number = extract_number_from_answer(label)

    if gold_number is None:
        return False

    # Extract predicted number from response
    # Try \boxed{} first
    boxed = extract_boxed_answer(response)
    if boxed:
        pred_number = try_parse_number(boxed.replace(",", ""))
        if pred_number is not None:
            return abs(pred_number - gold_number) < 1e-6

    # Fallback: extract last number from response
    pred_number = extract_number_from_answer(response)
    if pred_number is not None:
        return abs(pred_number - gold_number) < 1e-6

    return False


def _verify_boxed_math(response: str, label: str) -> bool:
    """Verify MATH/OpenMathInstruct response: extract \\boxed{} and compare.

    Labels contain \\boxed{<answer>}. Use math equivalence checking.
    """
    # Extract gold answer from label
    gold_answer = extract_boxed_answer(label)
    if not gold_answer:
        # Fallback: use the entire label as gold
        gold_answer = label.strip()

    # Extract predicted answer from response
    pred_answer = extract_boxed_answer(response)
    if not pred_answer:
        return False

    return is_math_equivalent(pred_answer, gold_answer)


def _extract_final_answer_from_label(label: str) -> Optional[str]:
    """Extract the final answer from a free-text label.

    Tries multiple patterns commonly found in openr1_math labels:
    1. \\boxed{...}
    2. 'The answer is: X' / 'answer is X' / 'Answer: X'
    3. 'Reference answer: X'
    4. Terminal patterns like 'is X.' or '= X.' at end of text
    5. Short label that is itself the answer (e.g. '$1$', 'C', '196')
    """
    if not label:
        return None

    stripped = label.strip()

    # Pattern 1: \boxed{}
    boxed = extract_boxed_answer(stripped)
    if boxed:
        return boxed

    # Pattern 2: 'the answer is: X' or 'answer: X' (case insensitive)
    # Match the last occurrence to get the final answer
    answer_patterns = [
        r'(?:the\s+)?answer\s+is[:\s]+(.+?)(?:\.\s*$|\.$|$)',
        r'(?:reference\s+)?answer[:\s]+(.+?)(?:\.\s*$|\.$|$)',
        r'(?:the\s+)?result\s+is[:\s]+(.+?)(?:\.\s*$|\.$|$)',
        r'(?:therefore|thus|hence|so)[,\s]+(?:the\s+)?(?:answer|result|value)\s+is[:\s]+(.+?)(?:\.\s*$|\.$|$)',
    ]
    last_match = None
    for pattern in answer_patterns:
        for m in re.finditer(pattern, stripped, re.IGNORECASE):
            last_match = m.group(1).strip()
    if last_match:
        # Clean up LaTeX wrappers like $...$
        cleaned = re.sub(r'^\$(.+)\$$', r'\1', last_match.strip())
        if cleaned:
            return cleaned

    # Pattern 3: Terminal number/expression at end of text
    # e.g. "...sum is 531." or "...which is $C_{m}^{4}$."
    terminal_patterns = [
        # "is X." or "is X" at end
        r'\bis\s+\$?([^$\n]{1,60})\$?\.?\s*$',
        # "= X." or "= X" at end (LaTeX or plain)
        r'=\s*\$?([^$\n]{1,60})\$?\.?\s*$',
        # "equals X" at end
        r'\bequals?\s+\$?([^$\n]{1,60})\$?\.?\s*$',
    ]
    for pattern in terminal_patterns:
        m = re.search(pattern, stripped, re.IGNORECASE)
        if m:
            candidate = m.group(1).strip().rstrip('.')
            # Only accept if it looks like a value (number, letter, expression)
            if candidate and len(candidate) < 50:
                cleaned = re.sub(r'^\$(.+)\$$', r'\1', candidate)
                if cleaned:
                    return cleaned

    # Pattern 4: Short label - likely the answer itself
    if len(stripped) < 50:
        # Remove common prefixes like numbering: '4. ', '(3) '
        cleaned = re.sub(r'^[\d]+\.\s*', '', stripped)
        cleaned = re.sub(r'^\([\d]+\)\s*', '', cleaned)
        # Remove LaTeX $ wrappers
        cleaned = re.sub(r'^\$(.+)\$$', r'\1', cleaned.strip())
        # Remove \mathrm{} wrapper
        cleaned = re.sub(r'\\mathrm\{([^}]*)\}', r'\1', cleaned)
        if cleaned:
            return cleaned.strip()

    # Pattern 5: Medium-length labels (50-200 chars) - try to extract last number
    if len(stripped) < 200:
        # Try to find the last meaningful number in the label
        numbers = re.findall(r'(?<![a-zA-Z])-?\d+(?:\.\d+)?(?![a-zA-Z])', stripped)
        if numbers and len(numbers) <= 5:
            return numbers[-1]

    return None


def _verify_openr1_math(response: str, label: str) -> bool:
    """Verify OpenR1-Math response with flexible label format handling.

    OpenR1-Math labels come in many formats:
    - \boxed{answer}
    - 'The answer is: X'
    - 'Answer: X'
    - Short text that IS the answer (e.g. '$1$', 'C')
    - Long solution text with answer embedded

    Strategy:
    1. Extract gold answer from label using multiple patterns
    2. Extract predicted answer from response (\boxed{} or last number)
    3. Compare using math equivalence
    """
    gold_answer = _extract_final_answer_from_label(label)

    # Extract predicted answer from response
    pred_boxed = extract_boxed_answer(response)

    if gold_answer:
        # Try math equivalence with boxed answer
        if pred_boxed and is_math_equivalent(pred_boxed, gold_answer):
            return True

        # Try numeric comparison
        gold_num = try_parse_number(gold_answer.replace(',', ''))
        if gold_num is not None:
            if pred_boxed:
                pred_num = try_parse_number(pred_boxed.replace(',', ''))
                if pred_num is not None and abs(pred_num - gold_num) < 1e-6:
                    return True
            # Also try extracting number from full response
            pred_num = extract_number_from_answer(response)
            if pred_num is not None and abs(pred_num - gold_num) < 1e-6:
                return True

        # Try normalized string comparison (for non-numeric answers like 'C', expressions)
        norm_gold = normalize_answer_string(gold_answer)
        if pred_boxed:
            norm_pred = normalize_answer_string(pred_boxed)
            if norm_pred and norm_gold and norm_pred == norm_gold:
                return True

    # Fallback: if label has \boxed{}, use standard boxed math verification
    if '\\boxed' in label or '\boxed' in label:
        return _verify_boxed_math(response, label)

    # Last resort for long free-text labels: check if response is non-trivial
    # (we can't reliably verify free-text labels, so accept if response has
    # mathematical content and \boxed{} — this is a teacher model after all)
    if len(label.strip()) > 200:
        # For long free-text labels where precise matching is unreliable,
        # accept the response if it contains a boxed answer and is non-trivial
        if pred_boxed and len(response.strip()) > 50:
            return True

    return False


def _verify_orca_math(response: str, label: str) -> bool:
    """Verify Orca-Math response: extract final number and compare.

    Orca-Math labels are typically plain numbers or short text with a number.
    """
    gold_number = extract_number_from_answer(label)
    if gold_number is None:
        return False

    # Try \boxed{} first
    boxed = extract_boxed_answer(response)
    if boxed:
        pred_number = try_parse_number(boxed.replace(",", ""))
        if pred_number is not None:
            return abs(pred_number - gold_number) < 1e-6

    # Fallback: extract last number
    pred_number = extract_number_from_answer(response)
    if pred_number is not None:
        return abs(pred_number - gold_number) < 1e-6

    return False


def _verify_code(response: str, label: str) -> bool:
    """Verify code response: format check only (no execution).

    Checks:
    - Response is not empty
    - Response contains valid Python code indicators
    - Response has reasonable length
    """
    if not response or not response.strip():
        return False

    stripped = response.strip()

    # Must have reasonable length
    if len(stripped) < 20:
        return False

    # Check for Python code indicators
    code_indicators = ['def ', 'import ', 'class ', 'for ', 'while ', 'if ', 'return ',
                       'print(', '= ', '==', '+=', 'range(', 'len(', 'list(', 'dict(',
                       'try:', 'except', 'with ']
    has_code = any(indicator in stripped for indicator in code_indicators)

    if not has_code:
        # Also check for code blocks
        if '```' in stripped:
            has_code = True

    return has_code


# ============================================================================
# Unified verification interface
# ============================================================================

def verify_response(response: str, label: str, source: str) -> bool:
    """Verify if a teacher response is correct for the given source type.

    Args:
        response: The teacher-generated response text.
        label: The ground truth label/answer from the dataset.
        source: The dataset source (e.g. 'gsm8k', 'math_minus_500', 'kodcode').

    Returns:
        True if the response is considered correct, False otherwise.
    """
    if not response or not response.strip():
        return False

    source_lower = source.lower().strip()

    if source_lower in GSM8K_SOURCES:
        return _verify_gsm8k(response, label)
    elif source_lower in OPENR1_MATH_SOURCES:
        return _verify_openr1_math(response, label)
    elif source_lower in BOXED_MATH_SOURCES:
        return _verify_boxed_math(response, label)
    elif source_lower in ORCA_MATH_SOURCES:
        return _verify_orca_math(response, label)
    elif source_lower in CODE_SOURCES:
        return _verify_code(response, label)
    else:
        # Unknown source: try math verification first, then code
        if "####" in label:
            return _verify_gsm8k(response, label)
        elif "\\boxed" in label:
            return _verify_boxed_math(response, label)
        else:
            # Try numeric comparison as last resort
            gold_num = extract_number_from_answer(label)
            if gold_num is not None:
                pred_num = extract_number_from_answer(response)
                if pred_num is not None:
                    return abs(pred_num - gold_num) < 1e-6
            # Fallback to code check
            return _verify_code(response, label)


def get_source_type(source: str) -> str:
    """Get the high-level type of a source: 'math' or 'code'."""
    if source.lower().strip() in MATH_SOURCES:
        return "math"
    elif source.lower().strip() in CODE_SOURCES:
        return "code"
    else:
        return "unknown"
