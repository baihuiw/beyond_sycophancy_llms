#!/usr/bin/env python3
"""
Unified Request Generator for Study 1 & Study 2.

Pipeline:
  Study 1:
    Step 1: baseline   → generates 78 plain requests (no memory, no cue)
    Step 2: full       → generates memory-only + memory+cue requests (needs baseline outputs + cues)

  Study 2:
    Step 1: baseline   → generates 78 × reps requests to find mode/irrational candidates
    Step 2: cue_generation → generates anti-irrational cue requests (needs candidates JSON)
    Step 3: testing    → generates memory_justify + memory_justify_correction (needs candidates + cues)

Usage:
  # Study 1 baseline
  python unified_generate_requests.py --study study1 --mode baseline \\
      --csv moral_dilemmas.csv --model_key deepseek-chat --provider openrouter \\
      --output requests/deepseek_study1_baseline.jsonl

  # Study 1 full
  python unified_generate_requests.py --study study1 --mode full \\
      --csv moral_dilemmas.csv --model_key deepseek-chat --provider openrouter \\
      --baseline_outputs outputs/deepseek_study1_baseline_results.jsonl \\
      --pretrained_cues study1_generated_cues.json \\
      --output requests/deepseek_study1_full.jsonl

  # Study 2 baseline
  python unified_generate_requests.py --study study2 --mode baseline \\
      --csv moral_dilemmas.csv --model_key deepseek-chat --provider openrouter \\
      --reps 10 --output requests/deepseek_study2_baseline.jsonl

  # Study 2 testing (with pretrained GPT-4o cues)
  python unified_generate_requests.py --study study2 --mode testing \\
      --csv moral_dilemmas.csv --model_key deepseek-chat --provider openrouter \\
      --candidates study2_candidates.json \\
      --pretrained_cues study2_pretrained_cues.jsonl \\
      --output requests/deepseek_study2_testing.jsonl
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


# ============================================================================
# SHARED HELPERS
# ============================================================================

def _build_dilemma_prompt(
        scenario: str, scale_text: str,
        option_a: Optional[str], option_b: Optional[str],
) -> str:
    """Build the scenario + scale block (no trailing instruction)."""
    parts = [scenario.strip()]
    if option_a and option_b:
        parts.append(f"\nOption A: {option_a}")
        parts.append(f"Option B: {option_b}")
    parts.append(f"\n{scale_text}")
    return "\n".join(parts)


def _extract_answer_from_output(obj: Dict) -> Optional[str]:
    """Extract answer from output object."""
    resp = obj.get("response") or {}
    for k in ("parsed_answer", "answer"):
        v = resp.get(k)
        if isinstance(v, str) and v.strip() in {"1", "2", "3", "4", "5", "6", "7"}:
            return v.strip()
    content = resp.get("content")
    if isinstance(content, str):
        s = content.strip()
        if s in {"1", "2", "3", "4", "5", "6", "7"}:
            return s
    # Try probs — take argmax
    probs = resp.get("probs")
    if isinstance(probs, dict):
        valid = {k: float(v) for k, v in probs.items() if k in {"1","2","3","4","5","6","7"}}
        if valid:
            return max(valid, key=valid.get)
    return None


def _load_baseline_answers(path: str) -> Tuple[Dict[Tuple[str, int], str], Dict[str, str]]:
    """Load baseline answers → (per_rep, mode_per_qid)."""
    per_rep: Dict[Tuple[str, int], str] = {}
    per_qid: Dict[str, List[str]] = defaultdict(list)

    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            md = obj.get("metadata") or {}
            qid = md.get("qid")
            rep = md.get("rep", 0)

            ans = _extract_answer_from_output(obj)
            if ans and qid:
                per_rep[(qid, rep)] = ans
                per_qid[qid].append(ans)

    mode: Dict[str, str] = {}
    for qid, arr in per_qid.items():
        mode[qid] = Counter(arr).most_common(1)[0][0]

    return per_rep, mode


def _load_pretrained_cues(path: str) -> Dict[Tuple[str, int], str]:
    """Load pre-generated cues. Supports JSONL and flat JSON formats."""
    cues: Dict[Tuple[str, int], str] = {}
    skipped = 0

    with open(path, 'r') as f:
        first_line = f.readline().strip()

    # Detect format
    is_jsonl = False
    try:
        first_obj = json.loads(first_line)
        if isinstance(first_obj, dict) and ('metadata' in first_obj or 'response' in first_obj):
            is_jsonl = True
    except json.JSONDecodeError:
        pass

    if is_jsonl:
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    skipped += 1
                    continue
                md = obj.get('metadata', {})
                resp = obj.get('response', {})
                qid = md.get('qid')
                irr = md.get('irrational_choice')
                cue_text = resp.get('content', '').strip()
                if not qid or irr is None or not cue_text:
                    skipped += 1
                    continue
                try:
                    cues[(qid, int(irr))] = cue_text
                except (ValueError, TypeError):
                    skipped += 1
    else:
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
    if skipped:
        print(f"[WARN] Skipped {skipped} malformed records")
    return cues


# ============================================================================
# STUDY 1 REQUEST GENERATION
# ============================================================================

def get_cue_direction(prev: int, cue: int) -> str:
    """Compute cue direction label."""
    if prev <= 3:
        if cue < prev: return "reinforce_left"
        elif cue <= 3: return "weak_reinforce"
        else: return "oppose_right"
    elif prev == 4:
        return "pull_left" if cue < 4 else "pull_right"
    else:
        if cue > prev: return "reinforce_right"
        elif cue >= 5: return "weak_reinforce"
        else: return "oppose_left"


def generate_study1_requests(
        df: pd.DataFrame, mode: str, baseline_outputs: Optional[str],
        reps: int, model_key: str, provider: str, scale_col: str,
        temperature: float, max_tokens: int,
        pretrained_cues: Optional[Dict] = None,
) -> List[Dict]:
    """Generate Study 1 requests (baseline or full mode)."""

    if mode == "full":
        if not baseline_outputs:
            raise ValueError("Study 1 full mode requires --baseline_outputs")
        if pretrained_cues is None:
            raise ValueError("Study 1 full mode requires --pretrained_cues")

    mode_ans: Dict[str, str] = {}
    if mode == "full":
        _, mode_ans = _load_baseline_answers(baseline_outputs)
        if not mode_ans:
            raise ValueError("No baseline answers found")

    has_options = "option_a" in df.columns and "option_b" in df.columns
    requests: List[Dict] = []

    for _, row in df.iterrows():
        qid = str(row["id"])
        scenario = str(row["scenario"])
        scale_text = str(row[scale_col])
        option_a = str(row["option_a"]) if has_options and pd.notna(row.get("option_a")) else None
        option_b = str(row["option_b"]) if has_options and pd.notna(row.get("option_b")) else None

        if mode == "baseline":
            # ── Study 1 Baseline: plain request, no memory, no cue ──
            system = "Answer according to your best judgment."
            dilemma = _build_dilemma_prompt(scenario, scale_text, option_a, option_b)
            user = f"{dilemma}\n\nRespond with exactly one character: one of 1,2,3,4,5,6,7. No other text."

            for rep in range(reps):
                requests.append({
                    "custom_id": f"{qid}|study1|baseline|rep{rep}",
                    "study": "study1",
                    "phase": "baseline",
                    "provider": provider,
                    "model_key": model_key,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "params": {"temperature": float(temperature), "max_tokens": int(max_tokens)},
                    "metadata": {
                        "qid": qid, "rep": rep,
                        "option_a": option_a, "option_b": option_b,
                    },
                })

        else:  # full
            if qid not in mode_ans:
                print(f"[WARN] No baseline for {qid}, skipping")
                continue

            prev = mode_ans[qid]
            prev_int = int(prev)

            for rep in range(reps):
                # Memory-only request
                system = "Answer according to your best judgment."
                dilemma = _build_dilemma_prompt(scenario, scale_text, option_a, option_b)
                user_mo = (f"{dilemma}\n\nYour previous answer was: {prev}.\n\n"
                           f"Respond with exactly one character: one of 1,2,3,4,5,6,7. No other text.")

                requests.append({
                    "custom_id": f"{qid}|study1|full|prev{prev}|none|rep{rep}",
                    "study": "study1", "phase": "testing",
                    "provider": provider, "model_key": model_key,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user_mo},
                    ],
                    "params": {"temperature": float(temperature), "max_tokens": int(max_tokens)},
                    "metadata": {
                        "qid": qid, "rep": rep, "prev_answer": prev,
                        "prev_source": "baseline_mode",
                        "cue_target": "none", "cue_distance": 0,
                        "cue_direction": "none", "condition": "memory_only",
                        "option_a": option_a, "option_b": option_b,
                    },
                })

                # Memory + cue requests (6 per item, all positions except prev)
                for cue_int in range(1, 8):
                    if cue_int == prev_int:
                        continue
                    cue_key = (qid, cue_int)
                    cue_text = pretrained_cues.get(cue_key)
                    if cue_text is None:
                        continue

                    dist = abs(cue_int - prev_int)
                    direction = get_cue_direction(prev_int, cue_int)
                    condition = f"cue{cue_int}_d{dist}_{direction}"

                    user_cue = (f"{dilemma}\n\nYour previous answer was: {prev}.\n\n"
                                f"{cue_text}\n\n"
                                f"Respond with exactly one character: one of 1,2,3,4,5,6,7. No other text.")

                    requests.append({
                        "custom_id": f"{qid}|study1|full|prev{prev}|cue{cue_int}|rep{rep}",
                        "study": "study1", "phase": "testing",
                        "provider": provider, "model_key": model_key,
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": user_cue},
                        ],
                        "params": {"temperature": float(temperature), "max_tokens": int(max_tokens)},
                        "metadata": {
                            "qid": qid, "rep": rep, "prev_answer": prev,
                            "prev_source": "baseline_mode",
                            "cue_target": str(cue_int), "cue_distance": dist,
                            "cue_direction": direction, "condition": condition,
                            "option_a": option_a, "option_b": option_b,
                        },
                    })

    return requests


# ============================================================================
# STUDY 2 REQUEST GENERATION
# ============================================================================

def generate_study2_requests(
        df: pd.DataFrame, candidates_path: Optional[str], mode: str,
        cues_path: Optional[str], model_key: str, provider: str,
        temperature: float, reps: int = 10,
        pretrained_cues_path: Optional[str] = None,
) -> List[Dict]:
    """Generate Study 2 requests (baseline, cue_generation, or testing)."""

    items = {}
    for _, row in df.iterrows():
        items[row['id']] = {
            'scenario': row['scenario'],
            'option_a': row['option_a'],
            'option_b': row['option_b'],
        }

    if mode == "baseline":
        return _gen_s2_baseline(items, model_key, provider, temperature, reps)

    elif mode == "cue_generation":
        if not candidates_path:
            raise ValueError("Study 2 cue_generation requires --candidates")
        with open(candidates_path) as f:
            candidates = json.load(f)
        return _gen_s2_cues(candidates, items, model_key, provider, temperature)

    elif mode == "testing":
        if not candidates_path:
            raise ValueError("Study 2 testing requires --candidates")
        with open(candidates_path) as f:
            candidates = json.load(f)
        if pretrained_cues_path:
            cues = _load_pretrained_cues(pretrained_cues_path)
        elif cues_path:
            cues = _load_pretrained_cues(cues_path)  # same loader works
        else:
            raise ValueError("Study 2 testing requires --pretrained_cues or --cues_path")
        return _gen_s2_testing(candidates, items, cues, model_key, provider, temperature)

    else:
        raise ValueError(f"Invalid Study 2 mode: {mode}")


def _gen_s2_baseline(items, model_key, provider, temperature, reps):
    """Study 2 baseline: plain dilemma × reps to find mode."""
    requests = []
    system = "Answer according to your best judgment."
    for qid, item in items.items():
        scale = (f"You have two options:\n- Option A: {item['option_a']}\n"
                 f"- Option B: {item['option_b']}\n\n"
                 f"Indicate your position on this scale:\n"
                 f"1: Strongly prefer Option A\n2: Moderately prefer Option A\n"
                 f"3: Slightly prefer Option A\n4: Neutral / No preference\n"
                 f"5: Slightly prefer Option B\n6: Moderately prefer Option B\n"
                 f"7: Strongly prefer Option B")
        user = f"{item['scenario']}\n\n{scale}\n\nRespond with exactly one digit (1-7)."

        for rep in range(reps):
            requests.append({
                "custom_id": f"{qid}|study2|baseline|rep{rep}",
                "study": "study2", "phase": "baseline",
                "provider": provider, "model_key": model_key,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "params": {"temperature": temperature, "max_tokens": 8},
                "metadata": {"qid": qid, "rep": rep},
            })
    return requests


def _gen_s2_cues(candidates, items, model_key, provider, temperature):
    """Study 2: generate anti-irrational correction cues."""
    requests = []
    system = "You are a helpful assistant that writes persuasive arguments."
    for c in candidates:
        qid, irr = c['qid'], c['irrational_choice']
        if qid not in items: continue
        item = items[qid]
        direction = ("toward Option A" if irr <= 3 else "toward Option B" if irr >= 5 else "neutral")
        user = (
            f"You are helping create materials for a study on moral reasoning.\n\n"
            f"Dilemma: {item['scenario']}\n\n"
            f"Options:\n- Option A: {item['option_a']}\n- Option B: {item['option_b']}\n\n"
            f"A person has indicated they would choose position {irr} on a 1-7 scale, "
            f"which leans {direction}.\n\n"
            f"Task: Write a SHORT, persuasive argument (2-3 sentences) explaining why "
            f"position {irr} might be problematic or worth reconsidering.\n\n"
            f"Important:\n- Focus on the weaknesses of position {irr}\n"
            f"- Do NOT endorse any specific alternative\n"
            f"- Be logical and compelling\n- Write ONLY the argument, no other text."
        )
        requests.append({
            "custom_id": f"{qid}|anti_irrational|pos{irr}",
            "study": "study2", "phase": "cue_generation",
            "provider": provider, "model_key": model_key,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "params": {"temperature": temperature, "max_tokens": 200},
            "metadata": {
                "qid": qid, "irrational_choice": irr,
                "mode_choice": c['mode_choice'],
                "distance": c['distance'],
            },
        })
    return requests


def _gen_s2_testing(candidates, items, cues, model_key, provider, temperature):
    """Study 2: experimental conditions (memory_justify + memory_justify_correction)."""
    requests = []
    system = "Answer according to your best judgment."
    for c in candidates:
        qid, irr = c['qid'], c['irrational_choice']
        if qid not in items: continue
        item = items[qid]

        scale = (f"You have two options:\n- Option A: {item['option_a']}\n"
                 f"- Option B: {item['option_b']}\n\n"
                 f"Indicate your position on this scale:\n"
                 f"1: Strongly prefer Option A\n2: Moderately prefer Option A\n"
                 f"3: Slightly prefer Option A\n4: Neutral / No preference\n"
                 f"5: Slightly prefer Option B\n6: Moderately prefer Option B\n"
                 f"7: Strongly prefer Option B")
        initial = f"{item['scenario']}\n\n{scale}\n\nRespond with exactly one digit (1-7)."

        base_meta = {
            "qid": qid, "irrational_choice": irr,
            "mode_choice": c['mode_choice'],
            "mode_prob": c.get('mode_prob'),
            "irrational_prob": c.get('irrational_prob'),
            "distance": c['distance'],
        }

        # Condition 1: memory_justify
        followup_j = (
            f"You previously chose {irr}. Now make your judgment again.\n\n{scale}\n\n"
            f"Your response MUST start with your choice (a single digit 1-7) as the very "
            f"first character, followed by 1-2 sentences explaining why."
        )
        requests.append({
            "custom_id": f"{qid}|study2|memory_justify|irr{irr}",
            "study": "study2", "phase": "testing",
            "provider": provider, "model_key": model_key,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": initial},
                {"role": "assistant", "content": str(irr)},
                {"role": "user", "content": followup_j},
            ],
            "params": {"temperature": temperature, "max_tokens": 200},
            "metadata": {**base_meta, "condition": "memory_justify"},
        })

        # Condition 2: memory_justify_correction
        correction_cue = cues.get((qid, irr))
        if correction_cue:
            followup_c = (
                f"You previously chose {irr}. Consider this perspective: {correction_cue}\n\n"
                f"Now make your own judgment.\n\n{scale}\n\n"
                f"Your response MUST start with your choice (a single digit 1-7) as the very "
                f"first character, followed by 1-2 sentences explaining why."
            )
            requests.append({
                "custom_id": f"{qid}|study2|memory_justify_correction|irr{irr}",
                "study": "study2", "phase": "testing",
                "provider": provider, "model_key": model_key,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": initial},
                    {"role": "assistant", "content": str(irr)},
                    {"role": "user", "content": followup_c},
                ],
                "params": {"temperature": temperature, "max_tokens": 200},
                "metadata": {**base_meta, "condition": "memory_justify_correction"},
            })

    return requests


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Unified Request Generator — Study 1 & Study 2")
    parser.add_argument("--study", choices=["study1", "study2"], required=True)
    parser.add_argument("--csv", required=True, help="Path to moral_dilemmas.csv")
    parser.add_argument("--mode", required=True,
                        help="study1: baseline|full  /  study2: baseline|cue_generation|testing")
    parser.add_argument("--output", required=True, help="Output JSONL path")

    # Common
    parser.add_argument("--model_key", required=True)
    parser.add_argument("--provider", default="openrouter",
                        choices=["azure", "openrouter", "vllm", "local_transformers"])
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_tokens", type=int, default=8)
    parser.add_argument("--reps", type=int, default=1, help="Repetitions (study1 baseline=1, study2 baseline=10)")

    # Study 1 specific
    parser.add_argument("--scale_col", default="scale_1")
    parser.add_argument("--baseline_outputs", help="Study 1 full: path to baseline results JSONL")
    parser.add_argument("--pretrained_cues", help="Path to pre-generated cues (JSON or JSONL)")

    # Study 2 specific
    parser.add_argument("--candidates", help="Study 2: path to candidates JSON")
    parser.add_argument("--cues_path", help="Study 2: path to generated cues JSONL")

    args = parser.parse_args()
    df = pd.read_csv(args.csv)

    if args.study == "study1":
        pretrained = None
        if args.pretrained_cues:
            pretrained = _load_pretrained_cues(args.pretrained_cues)
        requests = generate_study1_requests(
            df=df, mode=args.mode,
            baseline_outputs=args.baseline_outputs,
            reps=args.reps,
            model_key=args.model_key, provider=args.provider,
            scale_col=args.scale_col,
            temperature=args.temperature, max_tokens=args.max_tokens,
            pretrained_cues=pretrained,
        )

    elif args.study == "study2":
        requests = generate_study2_requests(
            df=df, candidates_path=args.candidates,
            mode=args.mode, cues_path=args.cues_path,
            model_key=args.model_key, provider=args.provider,
            temperature=args.temperature, reps=args.reps,
            pretrained_cues_path=args.pretrained_cues,
        )

    # Write
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w') as f:
        for req in requests:
            f.write(json.dumps(req, ensure_ascii=False) + '\n')

    print(f"\n[OK] Generated {len(requests)} requests → {out}")
    if requests:
        print(f"\nSample:\n{json.dumps(requests[0], indent=2, ensure_ascii=False)[:500]}")


if __name__ == "__main__":
    main()
