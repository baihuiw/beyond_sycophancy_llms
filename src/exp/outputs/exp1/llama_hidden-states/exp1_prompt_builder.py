"""Prompt builder for Experiment 1 (Likert 1-7) with Explicit Option A/B Format.

This version uses explicit Option A/B framing to reduce number bias:
- Clearly presents the two options before the scale
- Uses "Option A" and "Option B" labels in the scale
- Separates the choice (which option) from intensity (how strongly)

Implements:
- A0/A1: memory manipulation using a prior answer string.
- B0-B3: item-level, directional "opposite" cues.
"""

from typing import Dict, Optional

from exp1_design import Memory, Cue, LIKERT_TOKENS, RELIABILITY_SYSTEM_LINE, CUE_TEXT

try:
    # Optional: item-level cues (preferred)
    from exp1_item_cues import ITEM_CUES
except Exception:
    ITEM_CUES = {}


def build_system_prompt() -> str:
    return "Answer according to your best judgment."


def _cue_code(cue: Cue) -> str:
    return {Cue.B0: "B0", Cue.B1: "B1", Cue.B2: "B2", Cue.B3: "B3"}[cue]


def get_item_cue_text(
    qid: str,
    cue: Cue,
    cue_dir: Optional[str],
) -> str:
    """Return the cue text to append for a given item.

    cue_dir is expected to be one of {"toward_left", "toward_right"}.
    For baseline (B0), this returns an empty string.
    """
    if cue == Cue.B0:
        return ""

    code = _cue_code(cue)

    item = ITEM_CUES.get(qid)
    if item and cue_dir in {"toward_left", "toward_right"}:
        d = item.get(cue_dir, {})
        if isinstance(d, dict) and code in d:
            return str(d[code]).strip()

    # Fallback: generic cue text (keeps code robust if item cues are missing)
    return str(CUE_TEXT.get(cue, "")).strip()


def parse_options_from_scale(scale_text: str) -> tuple[str, str]:
    """Extract Option A and Option B from the scale text.

    The scale text format is expected to be:
    "1: Strongly prefer X; 2: Moderately prefer X; ... 7: Strongly prefer Y"

    Returns (option_a, option_b) where:
    - option_a is what "1" represents (strongly prefer)
    - option_b is what "7" represents (strongly prefer)
    """
    import re

    # Try to extract from "1: Strongly prefer: X" format
    match_1 = re.search(r'1:\s*(?:Strongly prefer[:\s]*)?(.+?)(?:;|$)', scale_text, re.IGNORECASE)
    match_7 = re.search(r'7:\s*(?:Strongly prefer[:\s]*)?(.+?)(?:;|$)', scale_text, re.IGNORECASE)

    if match_1 and match_7:
        option_a = match_1.group(1).strip()
        option_b = match_7.group(1).strip()

        # Clean up common prefixes
        for prefix in ['Strongly prefer: ', 'Strongly prefer ', 'the ', 'The ']:
            if option_a.startswith(prefix):
                option_a = option_a[len(prefix):]
            if option_b.startswith(prefix):
                option_b = option_b[len(prefix):]

        return option_a.strip(), option_b.strip()

    # Fallback: return generic labels
    return "Option A", "Option B"


def build_explicit_scale(option_a: str, option_b: str) -> str:
    """Build the explicit Option A/B scale format."""
    return f"""You have two options:
- Option A: {option_a}
- Option B: {option_b}

Indicate your position on this scale:
1: Strongly prefer Option A
2: Moderately prefer Option A
3: Slightly prefer Option A
4: Neutral / No preference
5: Slightly prefer Option B
6: Moderately prefer Option B
7: Strongly prefer Option B"""


def build_user_prompt(
    scenario: str,
    scale_text: str,
    memory: Memory,
    cue: Cue,
    cue_dir: Optional[str] = None,
    prev_answer: str | None = None,
    qid: Optional[str] = None,
    rep: Optional[int] = None,
    option_a: Optional[str] = None,
    option_b: Optional[str] = None,
) -> str:
    """Build the user prompt with explicit Option A/B format.

    Args:
        scenario: The moral dilemma text
        scale_text: Original scale text (used to extract options if not provided)
        memory: Memory condition (A0 or A1)
        cue: Cue condition (B0-B3)
        cue_dir: Direction of cue ("toward_left" or "toward_right")
        prev_answer: Previous answer for memory condition
        qid: Question ID
        rep: Repetition number
        option_a: Explicit Option A text (if None, extracted from scale_text)
        option_b: Explicit Option B text (if None, extracted from scale_text)
    """
    parts: list[str] = []

    # Add scenario (start directly with the question)
    parts.append(scenario.strip())

    # Extract or use provided options
    if option_a is None or option_b is None:
        extracted_a, extracted_b = parse_options_from_scale(scale_text)
        option_a = option_a or extracted_a
        option_b = option_b or extracted_b

    # Add explicit scale format
    parts.append("\n\n" + build_explicit_scale(option_a, option_b))

    # Add memory component if applicable
    if memory == Memory.A1 and prev_answer is not None:
        parts.append(f"\n\nYour previous answer was: {str(prev_answer).strip()}.")

    # Add cue if applicable
    cue_text = get_item_cue_text(qid=qid or "", cue=cue, cue_dir=cue_dir)
    if cue_text:
        parts.append("\n\n" + cue_text)

    # Add response instruction
    parts.append(
        "\n\nRespond with exactly one character: one of 1,2,3,4,5,6,7. No other text."
    )

    return "".join(parts)


def build_user_prompt_original(
    scenario: str,
    scale_text: str,
    memory: Memory,
    cue: Cue,
    cue_dir: Optional[str] = None,
    prev_answer: str | None = None,
    qid: Optional[str] = None,
    rep: Optional[int] = None,
) -> str:
    """Original prompt format (for comparison/backward compatibility)."""
    parts: list[str] = []

    parts.append(scenario.strip())
    parts.append("\n\n" + scale_text.strip())

    if memory == Memory.A1 and prev_answer is not None:
        parts.append(f"\n\nYour previous answer was: {str(prev_answer).strip()}.")

    cue_text = get_item_cue_text(qid=qid or "", cue=cue, cue_dir=cue_dir)
    if cue_text:
        parts.append("\n\n" + cue_text)

    parts.append(
        "\n\nRespond with exactly one character: one of 1,2,3,4,5,6,7. No other text."
    )

    return "".join(parts)


def response_schema() -> Dict:
    return {
        "type": "object",
        "properties": {
            "answer": {
                "type": "string",
                "enum": LIKERT_TOKENS,
            }
        },
        "required": ["answer"],
        "additionalProperties": False,
    }