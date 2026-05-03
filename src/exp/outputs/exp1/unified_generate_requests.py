#!/usr/bin/env python3
"""
Unified Request Generator for Study 1 & Study 2

Generates a single JSONL file with ALL requests needed:
- Study 1: baseline, memory conditions, cue conditions
- Study 2: baseline, cue generation, experimental conditions

Output format:
{
  "custom_id": str,
  "study": "study2" | "study2",
  "phase": "baseline" | "cue_generation" | "testing",
  "provider": str,
  "model_key": str,
  "messages": [...],
  "params": {...},
  "metadata": {...}
}
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

# Import existing prompt builders
try:
    from exp1_design import Memory
except ImportError:
    print("[WARN] Could not import exp1_design - Study 2 baseline functionality limited")
    Memory = None


# ============================================================================
# STUDY 1 REQUEST GENERATION
# ============================================================================

def get_cue_direction(prev: int, cue: int) -> str:
    """
    Compute cue direction label from prev_answer and cue_target.

    Rules (derived from GPT-4o Study 1 data):
      prev ≤ 3  and  cue < prev          → reinforce_left
      prev ≤ 3  and  cue ≤ 3  (≥ prev)  → weak_reinforce
      prev ≤ 3  and  cue > 3             → oppose_right
      prev = 4  and  cue < 4             → pull_left
      prev = 4  and  cue > 4             → pull_right
      prev ≥ 5  and  cue > prev          → reinforce_right
      prev ≥ 5  and  cue ≥ 5  (≤ prev)  → weak_reinforce
      prev ≥ 5  and  cue < 5             → oppose_left
    """
    if prev <= 3:
        if cue < prev:
            return "reinforce_left"
        elif cue <= 3:
            return "weak_reinforce"
        else:
            return "oppose_right"
    elif prev == 4:
        return "pull_left" if cue < 4 else "pull_right"
    else:  # prev >= 5
        if cue > prev:
            return "reinforce_right"
        elif cue >= 5:
            return "weak_reinforce"
        else:
            return "oppose_left"


def generate_study1_requests(
        df: pd.DataFrame,
        mode: str,
        baseline_outputs: Optional[str],
        reps: int,
        model_key: str,
        provider: str,
        scale_col: str,
        temperature: float,
        max_tokens: int,
        baseline_fallback: str = "error",
        pretrained_cues: Optional[Dict] = None,
) -> List[Dict]:
    """
    Generate Study 1 requests.

    baseline mode:
        One plain request per item × rep (no memory, no cue).

    full mode (new structure, matches GPT-4o phase-2 design):
        For each item, for each rep:
          • 1 memory-only request  (prev injected, cue_target=none)
          • 6 memory+cue requests  (prev injected, cue_target = every position except prev)
        Total per item per rep = 7 requests.
        cue_direction and condition labels follow the exact same rules as the
        GPT-4o outputs so results are directly comparable.
        Requires --pretrained_cues (GPT-4o JSON) to supply cue text.
    """
    if mode == "full":
        if not baseline_outputs:
            raise ValueError("Study 1 full mode requires --baseline_outputs")
        if pretrained_cues is None:
            raise ValueError(
                "Study 1 full mode requires --pretrained_cues "
                "(GPT-4o cues JSON) to inject cue text into prompts"
            )

    # Load baseline answers
    per_rep: Dict[Tuple[str, int], str] = {}
    mode_ans: Dict[str, str] = {}
    if mode == "full":
        per_rep, mode_ans = _load_baseline_answers(baseline_outputs)
        if not mode_ans:
            raise ValueError("No baseline answers found in --baseline_outputs")

    has_options = "option_a" in df.columns and "option_b" in df.columns
    requests = []

    for _, row in df.iterrows():
        qid = str(row["id"])
        scenario = str(row["scenario"])
        scale_text = str(row[scale_col])
        option_a = str(row["option_a"]) if has_options and pd.notna(row.get("option_a")) else None
        option_b = str(row["option_b"]) if has_options and pd.notna(row.get("option_b")) else None

        if mode == "baseline":
            for rep in range(reps):
                requests.append(_build_study1_baseline_request(
                    provider=provider, model_key=model_key,
                    qid=qid, scenario=scenario, scale_text=scale_text,
                    option_a=option_a, option_b=option_b,
                    rep=rep, temperature=temperature, max_tokens=max_tokens,
                ))

        else:  # full
            for rep in range(reps):
                prev = per_rep.get((qid, rep))
                if prev is None:
                    if baseline_fallback == "most_common" and qid in mode_ans:
                        prev = mode_ans[qid]
                    else:
                        raise ValueError(f"Missing baseline for (qid={qid}, rep={rep})")

                prev_int = int(prev)

                # ── memory-only (no cue) ──────────────────────────────────────
                requests.append(_build_study1_full_request(
                    provider=provider, model_key=model_key,
                    qid=qid, scenario=scenario, scale_text=scale_text,
                    option_a=option_a, option_b=option_b,
                    prev_answer=prev, cue_target=None, cue_text=None,
                    cue_distance=0, cue_direction="none",
                    condition="memory_only",
                    rep=rep, temperature=temperature, max_tokens=max_tokens,
                ))

                # ── memory + cue (all 6 other positions) ─────────────────────
                for cue_int in range(1, 8):
                    if cue_int == prev_int:
                        continue  # skip same position as prev

                    cue_key = (qid, cue_int)
                    cue_text = pretrained_cues.get(cue_key)
                    if cue_text is None:
                        print(f"[WARN] No cue for {qid} cue_target={cue_int} — skipping")
                        continue

                    dist = abs(cue_int - prev_int)
                    direction = get_cue_direction(prev_int, cue_int)
                    condition = f"cue{cue_int}_d{dist}_{direction}"

                    requests.append(_build_study1_full_request(
                        provider=provider, model_key=model_key,
                        qid=qid, scenario=scenario, scale_text=scale_text,
                        option_a=option_a, option_b=option_b,
                        prev_answer=prev, cue_target=cue_int, cue_text=cue_text,
                        cue_distance=dist, cue_direction=direction,
                        condition=condition,
                        rep=rep, temperature=temperature, max_tokens=max_tokens,
                    ))

    return requests


def _build_study1_baseline_request(
        provider: str, model_key: str,
        qid: str, scenario: str, scale_text: str,
        option_a: Optional[str], option_b: Optional[str],
        rep: int, temperature: float, max_tokens: int,
) -> Dict:
    """Build a plain baseline request (no memory, no cue)."""
    system = "Answer according to your best judgment."
    user = (
        f"{_build_dilemma_prompt(scenario, scale_text, option_a, option_b)}\n\n"
        f"Respond with exactly one character: one of 1,2,3,4,5,6,7. No other text."
    )

    return {
        "custom_id": f"{qid}|study2|baseline|rep{rep}",
        "study": "study2",
        "phase": "baseline",
        "provider": provider,
        "model_key": model_key,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "params": {"temperature": float(temperature), "max_tokens": int(max_tokens)},
        "metadata": {
            "qid": qid,
            "rep": rep,
            "option_a": option_a,
            "option_b": option_b,
        },
    }


def _build_study1_full_request(
        provider: str, model_key: str,
        qid: str, scenario: str, scale_text: str,
        option_a: Optional[str], option_b: Optional[str],
        prev_answer: str,
        cue_target: Optional[int],  # None → memory-only
        cue_text: Optional[str],  # None → memory-only
        cue_distance: int,
        cue_direction: str,
        condition: str,
        rep: int, temperature: float, max_tokens: int,
) -> Dict:
    """
    Build a full-matrix request matching the GPT-4o phase-2 single-turn design.

    All conditions use exactly 2 messages: [system, user].
    The user message embeds everything inline:

      memory-only:
        {scenario}\n\n{scale}\n\nYour previous answer was: {prev}.\n\n
        Respond with exactly one character: one of 1,2,3,4,5,6,7. No other text.

      with cue:
        {scenario}\n\n{scale}\n\nYour previous answer was: {prev}.\n\n
        {cue_text}\n\n
        Respond with exactly one character: one of 1,2,3,4,5,6,7. No other text.
    """
    system = "Answer according to your best judgment."
    dilemma = _build_dilemma_prompt(scenario, scale_text, option_a, option_b)

    if cue_target is None:
        # memory-only: scenario + scale + prev, no cue
        user = (
            f"{dilemma}\n\n"
            f"Your previous answer was: {prev_answer}.\n\n"
            f"Respond with exactly one character: one of 1,2,3,4,5,6,7. No other text."
        )
    else:
        # memory + cue: scenario + scale + prev + cue
        user = (
            f"{dilemma}\n\n"
            f"Your previous answer was: {prev_answer}.\n\n"
            f"{cue_text}\n\n"
            f"Respond with exactly one character: one of 1,2,3,4,5,6,7. No other text."
        )

    cue_tag = f"cue{cue_target}" if cue_target else "none"
    custom_id = f"{qid}|study2|full|prev{prev_answer}|{cue_tag}|rep{rep}"

    return {
        "custom_id": custom_id,
        "study": "study2",
        "phase": "testing",
        "provider": provider,
        "model_key": model_key,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "params": {"temperature": float(temperature), "max_tokens": int(max_tokens)},
        "metadata": {
            "qid": qid,
            "rep": rep,
            "prev_answer": prev_answer,
            "cue_target": str(cue_target) if cue_target else "none",
            "cue_distance": cue_distance,
            "cue_direction": cue_direction,
            "condition": condition,
            "option_a": option_a,
            "option_b": option_b,
        },
    }


def _build_dilemma_prompt(
        scenario: str, scale_text: str,
        option_a: Optional[str], option_b: Optional[str],
) -> str:
    """
    Build the scenario + scale block (no trailing instruction).
    The instruction line is appended by the caller so baseline and full
    requests can use different wording.
    """
    parts = [scenario.strip()]
    if option_a and option_b:
        parts.append(f"\nOption A: {option_a}")
        parts.append(f"Option B: {option_b}")
    parts.append(f"\n{scale_text}")
    return "\n".join(parts)


# ============================================================================
# STUDY 2 REQUEST GENERATION
# ============================================================================

def generate_study2_requests(
        df: pd.DataFrame,
        candidates_path: Optional[str],
        mode: str,
        cues_path: Optional[str],
        model_key: str,
        provider: str,
        temperature: float,
        reps: int = 10,
        pretrained_cues_path: Optional[str] = None,
) -> List[Dict]:
    """Generate Study 2 requests (baseline, cue generation, OR experimental conditions).

    Args:
        pretrained_cues_path: Path to a pre-generated GPT-4o cues JSON file
            (format: {"qid_position": "cue text"}).  When provided in 'testing'
            mode the cue_generation step is skipped entirely.
    """

    # Build item lookup
    items = {}
    for _, row in df.iterrows():
        items[row['id']] = {
            'scenario': row['scenario'],
            'option_a': row['option_a'],
            'option_b': row['option_b'],
        }

    if mode == "baseline":
        # Generate baseline requests (no candidates needed)
        return _generate_study2_baseline(items, model_key, provider, temperature, reps)

    elif mode == "cue_generation":
        # Generate cues for identified candidates
        if pretrained_cues_path:
            print("[INFO] --pretrained_cues supplied: skipping cue_generation, "
                  "use --mode testing --pretrained_cues instead.")
        if not candidates_path:
            raise ValueError("Study 2 cue_generation mode requires --candidates")
        with open(candidates_path, 'r') as f:
            candidates = json.load(f)
        return _generate_study2_cues(candidates, items, model_key, provider, temperature)

    elif mode == "testing":
        # Generate experimental testing requests
        if not candidates_path:
            raise ValueError("Study 2 testing mode requires --candidates")

        with open(candidates_path, 'r') as f:
            candidates = json.load(f)

        # Resolve cues: pretrained JSON takes priority over generated JSONL
        if pretrained_cues_path:
            cues = _load_pretrained_cues(pretrained_cues_path)
        elif cues_path:
            cues = _load_study2_cues(cues_path)
        else:
            raise ValueError(
                "Study 2 testing mode requires either --pretrained_cues "
                "(GPT-4o JSON) or --cues_path (generated JSONL)"
            )

        return _generate_study2_testing(candidates, items, cues, model_key, provider, temperature)

    else:
        raise ValueError(f"Invalid Study 2 mode: {mode}. Use 'baseline', 'cue_generation', or 'testing'")


def _generate_study2_baseline(
        items: Dict,
        model_key: str,
        provider: str,
        temperature: float,
        reps: int,
) -> List[Dict]:
    """Generate Study 2 baseline requests (to identify irrational candidates)."""

    system_prompt = "Answer according to your best judgment."
    requests = []

    for qid, item in items.items():
        # Build scale text
        scale_text = f"""You have two options:
- Option A: {item['option_a']}
- Option B: {item['option_b']}

Indicate your position on this scale:
1: Strongly prefer Option A
2: Moderately prefer Option A
3: Slightly prefer Option A
4: Neutral / No preference
5: Slightly prefer Option B
6: Moderately prefer Option B
7: Strongly prefer Option B"""

        initial_question = f"""{item['scenario']}

{scale_text}

Respond with exactly one digit (1-7)."""

        # Generate multiple reps to find mode and identify outliers
        for rep in range(reps):
            custom_id = f"{qid}|study2|baseline|rep{rep}"

            requests.append({
                "custom_id": custom_id,
                "study": "study2",
                "phase": "baseline",
                "provider": provider,
                "model_key": model_key,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": initial_question}
                ],
                "params": {"temperature": temperature, "max_tokens": 8},
                "metadata": {
                    "qid": qid,
                    "rep": rep,
                }
            })

    return requests


def _generate_study2_cues(
        candidates: List[Dict],
        items: Dict,
        model_key: str,
        provider: str,
        temperature: float,
) -> List[Dict]:
    """Generate Study 2 cue generation requests."""

    system_prompt = "You are a helpful assistant that writes persuasive arguments."
    requests = []

    for c in candidates:
        qid = c['qid']
        irrational = c['irrational_choice']

        if qid not in items:
            continue

        item = items[qid]

        if irrational <= 3:
            irrational_direction = "toward Option A"
        elif irrational >= 5:
            irrational_direction = "toward Option B"
        else:
            irrational_direction = "neutral"

        user_prompt = f"""You are helping create materials for a study on moral reasoning.

Dilemma: {item['scenario']}

Options:
- Option A: {item['option_a']}
- Option B: {item['option_b']}

A person has indicated they would choose position {irrational} on a 1-7 scale, which leans {irrational_direction}.

Task: Write a SHORT, persuasive argument (2-3 sentences) explaining why position {irrational} might be problematic or worth reconsidering for this dilemma.

Important:
- Focus on the weaknesses of position {irrational}
- Do NOT explicitly endorse any specific alternative position
- Be logical and compelling, not emotional
- Write as a direct statement

Write ONLY the argument, no other text."""

        custom_id = f"{qid}|anti_irrational|pos{irrational}"

        requests.append({
            "custom_id": custom_id,
            "study": "study2",
            "phase": "cue_generation",
            "provider": provider,
            "model_key": model_key,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "params": {"temperature": temperature, "max_tokens": 200},
            "metadata": {
                "qid": qid,
                "phase": "cue_generation",
                "study": "study2",
                "irrational_choice": irrational,
                "irrational_prob": c.get('irrational_prob'),
                "mode_choice": c['mode_choice'],
                "mode_prob": c.get('mode_prob'),
                "distance": c['distance'],
                "cue_type": "anti_irrational",
            }
        })

    return requests


def _generate_study2_testing(
        candidates: List[Dict],
        items: Dict,
        cues: Dict[Tuple[str, int], str],
        model_key: str,
        provider: str,
        temperature: float,
) -> List[Dict]:
    """Generate Study 2 experimental testing requests."""

    system_prompt = "Answer according to your best judgment."
    requests = []

    for c in candidates:
        qid = c['qid']
        irrational = c['irrational_choice']

        if qid not in items:
            continue

        item = items[qid]
        cue_key = (qid, irrational)
        correction_cue = cues.get(cue_key)

        # Build scale text
        scale_text = f"""You have two options:
- Option A: {item['option_a']}
- Option B: {item['option_b']}

Indicate your position on this scale:
1: Strongly prefer Option A
2: Moderately prefer Option A
3: Slightly prefer Option A
4: Neutral / No preference
5: Slightly prefer Option B
6: Moderately prefer Option B
7: Strongly prefer Option B"""

        initial_question = f"""{item['scenario']}

{scale_text}

Respond with exactly one digit (1-7)."""

        # Injected memory
        injected_memory = str(irrational)

        base_metadata = {
            "qid": qid,
            "irrational_choice": irrational,
            "mode_choice": c['mode_choice'],
            "mode_prob": c.get('mode_prob'),
            "irrational_prob": c.get('irrational_prob'),
            "distance": c['distance'],
        }

        # Condition 1: memory_justify
        followup_justify = (
            f"You previously chose {irrational}. Now make your judgment again.\n\n"
            f"{scale_text}\n\n"
            f"Your response MUST start with your choice (a single digit 1-7) as the very first character, "
            f"followed by 1-2 sentences explaining why you chose that position."
        )

        custom_id_justify = f"{qid}|study2|memory_justify|irr{irrational}"
        requests.append({
            "custom_id": custom_id_justify,
            "study": "study2",
            "phase": "testing",
            "provider": provider,
            "model_key": model_key,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": initial_question},
                {"role": "assistant", "content": injected_memory},
                {"role": "user", "content": followup_justify}
            ],
            "params": {"temperature": temperature, "max_tokens": 200},
            "metadata": {**base_metadata, "condition": "memory_justify"}
        })

        # Condition 2: memory_justify_correction
        if correction_cue:
            followup_correction = (
                f"You previously chose {irrational}. Consider this perspective: {correction_cue}\n\n"
                f"Now make your own judgment.\n\n"
                f"{scale_text}\n\n"
                f"Your response MUST start with your choice (a single digit 1-7) as the very first character, "
                f"followed by 1-2 sentences explaining why you chose that position."
            )

            custom_id_correction = f"{qid}|study2|memory_justify_correction|irr{irrational}"
            requests.append({
                "custom_id": custom_id_correction,
                "study": "study2",
                "phase": "testing",
                "provider": provider,
                "model_key": model_key,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": initial_question},
                    {"role": "assistant", "content": injected_memory},
                    {"role": "user", "content": followup_correction}
                ],
                "params": {"temperature": temperature, "max_tokens": 200},
                "metadata": {**base_metadata, "condition": "memory_justify_correction", "cue_type": "anti_irrational"}
            })

    return requests


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _load_baseline_answers(baseline_outputs_path: str) -> Tuple[Dict[Tuple[str, int], str], Dict[str, str]]:
    """Load baseline answers from outputs JSONL."""
    per_rep: Dict[Tuple[str, int], str] = {}
    per_qid: Dict[str, List[str]] = defaultdict(list)

    with open(baseline_outputs_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            md = obj.get("metadata") or {}
            qid = md.get("qid")
            rep = md.get("rep")
            if not isinstance(qid, str) or not isinstance(rep, int):
                continue

            # Extract answer
            ans = _extract_answer_from_output(obj)
            if ans:
                per_rep[(qid, rep)] = ans
                per_qid[qid].append(ans)

    # Compute mode for each qid
    mode: Dict[str, str] = {}
    for qid, arr in per_qid.items():
        mode[qid] = Counter(arr).most_common(1)[0][0]

    return per_rep, mode


def _extract_answer_from_output(obj: Dict) -> Optional[str]:
    """Extract answer from outputs object."""
    resp = obj.get("response") or {}

    # Try parsed_answer or answer fields
    for k in ("parsed_answer", "answer"):
        v = resp.get(k)
        if isinstance(v, str) and v.strip() in {"1", "2", "3", "4", "5", "6", "7"}:
            return v.strip()

    # Try content field
    content = resp.get("content")
    if isinstance(content, str):
        s = content.strip()
        if s in {"1", "2", "3", "4", "5", "6", "7"}:
            return s

    return None


def _choose_cue_dir(prev_answer: Optional[str], rep: int) -> str:
    """Choose cue direction for opposite cues."""
    if prev_answer in {"1", "2", "3"}:
        return "toward_right"
    if prev_answer in {"5", "6", "7"}:
        return "toward_left"
    return "toward_right" if (rep % 2 == 0) else "toward_left"


def _load_study2_cues(cues_path: str) -> Dict[Tuple[str, int], str]:
    """Load Study 2 cues from outputs JSONL."""
    cues = {}
    with open(cues_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            md = obj.get("metadata") or {}
            qid = md.get("qid")
            irr = md.get("irrational_choice")
            content = obj.get("response", {}).get("content", "")
            if qid and irr and content:
                cues[(qid, irr)] = content
    return cues


def _load_pretrained_cues(pretrained_cues_path: str) -> Dict[Tuple[str, int], str]:
    """
    Load pre-generated cues from a JSONL file.

    Expected JSONL format (one record per line):
        {
          "custom_id": "{qid}|anti_irrational|pos{irr}",
          "metadata": {"qid": "moral_1", "irrational_choice": 6, ...},
          "response":  {"status_code": 200, "content": "cue text..."}
        }

    Returns dict keyed by (qid, irrational_choice_int) -> cue_text.
    Falls back to flat JSON dict format {"moral_1_6": "cue text"} for
    backward compatibility with Study 1 study1_generated_cues.json.
    """
    path = pretrained_cues_path.strip()
    cues: Dict[Tuple[str, int], str] = {}
    skipped = 0

    # ── Detect format: JSONL vs flat JSON ──────────────────────────────────
    with open(path, 'r') as f:
        first_line = f.readline().strip()

    is_jsonl = False
    try:
        first_obj = json.loads(first_line)
        # If it has 'metadata' or 'response' it's a JSONL record
        if isinstance(first_obj, dict) and ('metadata' in first_obj or 'response' in first_obj):
            is_jsonl = True
    except json.JSONDecodeError:
        pass

    if is_jsonl:
        # ── JSONL format ───────────────────────────────────────────────────
        with open(path, 'r') as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    skipped += 1
                    continue

                md   = obj.get('metadata', {})
                resp = obj.get('response', {})

                qid = md.get('qid')
                irr = md.get('irrational_choice')
                # cue text is in response.content
                cue_text = resp.get('content', '').strip()

                if not qid or irr is None or not cue_text:
                    skipped += 1
                    continue

                try:
                    cues[(qid, int(irr))] = cue_text
                except (ValueError, TypeError):
                    skipped += 1

    else:
        # ── Flat JSON dict format (Study 1 backward compat) ────────────────
        with open(path, 'r') as f:
            raw = json.load(f)

        for key, cue_text in raw.items():
            parts = key.rsplit('_', 1)
            if len(parts) != 2:
                skipped += 1
                continue
            qid, pos_str = parts
            try:
                cues[(qid, int(pos_str))] = cue_text
            except ValueError:
                skipped += 1

    print(f"[INFO] Loaded {len(cues)} pretrained cues from {path} (jsonl={is_jsonl})")
    print(f"[INFO] Unique qids: {len(set(k[0] for k in cues))}, "
          f"positions covered: {sorted(set(k[1] for k in cues))}")
    if skipped:
        print(f"[WARN] Skipped {skipped} malformed/empty records")
    return cues


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Unified Request Generator for Study 1 & Study 2")
    parser.add_argument("--study", choices=["study2", "study2"], required=True)
    parser.add_argument("--csv", required=True, help="Path to moral dilemmas CSV")
    parser.add_argument("--mode", required=True, help="baseline|full (study2) OR cue_generation|testing (study2)")

    # Study 1 specific
    parser.add_argument("--scale_col", default="scale_1", help="Study 1: scale column")
    parser.add_argument("--baseline_outputs", help="Study 1 full mode: path to baseline outputs JSONL")
    parser.add_argument("--reps", type=int, default=10, help="Number of repetitions")
    parser.add_argument("--baseline_fallback", choices=["error", "most_common"], default="most_common")

    # Study 2 specific
    parser.add_argument("--candidates",
                        help="Study 2: path to candidates JSON (required for cue_generation and testing)")
    parser.add_argument("--cues_path", help="Study 2: path to generated cue outputs JSONL for testing mode")
    parser.add_argument("--pretrained_cues", help="Study 2: path to pre-generated GPT-4o cues JSON "
                                                  "(format: {qid_position: cue_text}). Skips cue_generation step entirely.")

    # Common
    parser.add_argument("--model_key", required=True)
    parser.add_argument("--provider", default="openrouter",
                        choices=["azure", "openrouter", "vllm", "local_transformers"])
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_tokens", type=int, default=8)
    parser.add_argument("--outputs", required=True, help="Output JSONL path")

    args = parser.parse_args()

    # Load CSV
    df = pd.read_csv(args.csv)

    # Generate requests
    if args.study == "study2":
        # Load pretrained cues if provided (needed for full mode)
        pretrained_cues_study1: Optional[Dict] = None
        if args.pretrained_cues:
            pretrained_cues_study1 = _load_pretrained_cues(args.pretrained_cues)

        requests = generate_study1_requests(
            df=df,
            mode=args.mode,
            baseline_outputs=args.baseline_outputs,
            reps=args.reps,
            model_key=args.model_key,
            provider=args.provider,
            scale_col=args.scale_col,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            baseline_fallback=args.baseline_fallback,
            pretrained_cues=pretrained_cues_study1,
        )
    elif args.study == "study2":
        requests = generate_study2_requests(
            df=df,
            candidates_path=args.candidates,
            mode=args.mode,
            cues_path=args.cues_path,
            model_key=args.model_key,
            provider=args.provider,
            temperature=args.temperature,
            reps=args.reps,
            pretrained_cues_path=args.pretrained_cues,
        )

    # Write outputs
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        for req in requests:
            f.write(json.dumps(req, ensure_ascii=False) + '\n')

    print(f"[OK] Generated {len(requests)} requests")
    print(f"[OK] Saved to {output_path}")

    # Show sample
    if requests:
        print("\n" + "=" * 80)
        print("SAMPLE REQUEST:")
        print("=" * 80)
        sample = requests[0]
        print(json.dumps(sample, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()