from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

from exp1_design import Cue, Memory
from exp1_prompt_builder import build_system_prompt, build_user_prompt


def _read_jsonl(path: str) -> Iterable[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _extract_answer_from_output_row(obj: Dict) -> Optional[str]:
    """Best-effort answer extraction from an exp1 outputs.jsonl line.

    Supports:
      - obj["response"]["parsed_answer"] or ["answer"]
      - obj["response"]["content"] containing a single digit 1-7
    """
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
    return None


def load_baseline_answers(baseline_outputs_path: str) -> Tuple[Dict[Tuple[str, int], str], Dict[str, str]]:
    """Return (qid,rep)->answer mapping plus per-qid mode answer.

    The per-qid mode is used ONLY as a fallback when a specific rep is missing.
    """
    per_rep: Dict[Tuple[str, int], str] = {}
    per_qid: Dict[str, List[str]] = defaultdict(list)

    for obj in _read_jsonl(baseline_outputs_path):
        md = obj.get("metadata") or {}
        qid = md.get("qid")
        rep = md.get("rep")
        if not isinstance(qid, str) or not isinstance(rep, int):
            continue
        ans = _extract_answer_from_output_row(obj)
        if ans is None:
            continue
        per_rep[(qid, rep)] = ans
        per_qid[qid].append(ans)

    mode: Dict[str, str] = {}
    for qid, arr in per_qid.items():
        mode[qid] = Counter(arr).most_common(1)[0][0]
    return per_rep, mode


def choose_cue_dir(prev_answer: Optional[str], rep: int) -> str:
    """Directional rule for "opposite" cues.

    - If prev_answer in {1,2,3}: opposite is toward_right.
    - If prev_answer in {5,6,7}: opposite is toward_left.
    - If prev_answer == 4 or missing: alternate by rep parity for balance.
    """
    if prev_answer in {"1", "2", "3"}:
        return "toward_right"
    if prev_answer in {"5", "6", "7"}:
        return "toward_left"
    return "toward_right" if (rep % 2 == 0) else "toward_left"


def build_request_row(
    *,
    provider: str,
    model_key: str,
    qid: str,
    scenario: str,
    scale_text: str,
    option_a: Optional[str],
    option_b: Optional[str],
    memory: Memory,
    cue: Cue,
    cue_dir: str,
    rep: int,
    prev_answer: Optional[str],
    temperature: float,
    max_tokens: int,
) -> Dict:
    system = build_system_prompt()
    user = build_user_prompt(
        scenario=scenario,
        scale_text=scale_text,
        memory=memory,
        cue=cue,
        cue_dir=cue_dir,
        prev_answer=prev_answer,
        qid=qid,
        rep=rep,
        option_a=option_a,
        option_b=option_b,
    )

    return {
        "provider": provider,
        "model_key": model_key,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "params": {
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
        },
        "metadata": {
            "qid": qid,
            "memory": memory.value,
            "cue": cue.value,
            "cue_dir": cue_dir,
            "rep": rep,
            "option_a": option_a,
            "option_b": option_b,
            **({"prev_answer": prev_answer} if (memory == Memory.A1 and prev_answer is not None) else {}),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to moral dilemmas CSV (with option_a, option_b columns)")
    ap.add_argument("--scale_col", default="scale_1", help="Which scale column to use (default: scale_1)")
    ap.add_argument("--mode", choices=["baseline", "full"], required=True)
    ap.add_argument("--baseline_outputs", default=None, help="baseline_outputs.jsonl (required for mode=full)")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--model_key", required=True)
    ap.add_argument("--provider", default="azure", choices=["azure", "openrouter"], help="Request provider label")
    ap.add_argument("--outdir", required=True)

    # sampling knobs
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max_tokens", type=int, default=8)
    ap.add_argument(
        "--baseline_fallback",
        choices=["error", "most_common"],
        default="error",
        help="What to do if (qid,rep) answer is missing when building WITH_MEMORY rows",
    )
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    # Required columns
    needed = {"id", "scenario", args.scale_col}
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}. Found: {list(df.columns)}")

    # Check for option_a and option_b columns
    has_options = "option_a" in df.columns and "option_b" in df.columns
    if has_options:
        print("[INFO] Found option_a and option_b columns - using explicit Option A/B format")
    else:
        print("[WARN] option_a/option_b columns not found - will extract from scale_text")

    per_rep: Dict[Tuple[str, int], str] = {}
    mode_ans: Dict[str, str] = {}
    if args.mode == "full":
        if not args.baseline_outputs:
            raise ValueError("--baseline_outputs is required when --mode full")
        per_rep, mode_ans = load_baseline_answers(args.baseline_outputs)
        if not mode_ans:
            raise ValueError(
                "No baseline answers could be extracted from --baseline_outputs. "
                "Ensure baseline ran successfully and outputs contain parsed answers."
            )

    out_path = Path(args.outdir)
    out_path.mkdir(parents=True, exist_ok=True)
    out_file = out_path / f"exp1_{args.model_key}_{args.mode}_requests.jsonl"
    rows_out: List[Dict] = []

    for _, row in df.iterrows():
        qid = str(row["id"])
        scenario = str(row["scenario"])
        scale_text = str(row[args.scale_col])

        # Get option_a and option_b if available
        option_a = str(row["option_a"]) if has_options and pd.notna(row.get("option_a")) else None
        option_b = str(row["option_b"]) if has_options and pd.notna(row.get("option_b")) else None

        if args.mode == "baseline":
            # baseline: A0 x B0 only
            for rep in range(args.reps):
                cue_dir = choose_cue_dir(prev_answer=None, rep=rep)
                rows_out.append(
                    build_request_row(
                        provider=args.provider,
                        model_key=args.model_key,
                        qid=qid,
                        scenario=scenario,
                        scale_text=scale_text,
                        option_a=option_a,
                        option_b=option_b,
                        memory=Memory.A0,
                        cue=Cue.B0,
                        cue_dir=cue_dir,
                        rep=rep,
                        prev_answer=None,
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                    )
                )

        else:
            # full: (A0/A1) x (B0..B3)
            for rep in range(args.reps):
                # Choose prev_answer for A1
                prev = per_rep.get((qid, rep))
                if prev is None:
                    if args.baseline_fallback == "most_common" and qid in mode_ans:
                        prev = mode_ans[qid]
                    else:
                        raise ValueError(
                            f"Missing baseline answer for (qid={qid}, rep={rep}). "
                            "Re-run baseline with the same reps and ensure outputs are parseable, "
                            "or set --baseline_fallback most_common."
                        )

                for memory in Memory:
                    for cue in Cue:
                        use_prev = prev if memory == Memory.A1 else None
                        cue_dir = choose_cue_dir(prev_answer=prev, rep=rep)
                        rows_out.append(
                            build_request_row(
                                provider=args.provider,
                                model_key=args.model_key,
                                qid=qid,
                                scenario=scenario,
                                scale_text=scale_text,
                                option_a=option_a,
                                option_b=option_b,
                                memory=memory,
                                cue=cue,
                                cue_dir=cue_dir,
                                rep=rep,
                                prev_answer=use_prev,
                                temperature=args.temperature,
                                max_tokens=args.max_tokens,
                            )
                        )

    with open(out_file, "w", encoding="utf-8") as f:
        for r in rows_out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[OK] wrote {len(rows_out)} requests -> {out_file}")

    # Show a sample prompt
    if rows_out:
        print("\n" + "="*80)
        print("SAMPLE PROMPT (first request):")
        print("="*80)
        sample = rows_out[0]
        print(f"\n[System]: {sample['messages'][0]['content']}")
        print(f"\n[User]:\n{sample['messages'][1]['content']}")


if __name__ == "__main__":
    main()
