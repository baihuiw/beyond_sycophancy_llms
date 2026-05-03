import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent))
from exp1_design import Memory, LIKERT_TOKENS


CUE_GENERATION_SYSTEM = """You are helping design a study on moral reasoning. Your task is to generate persuasive arguments for different positions on moral dilemmas.

Generate arguments that are:
1. Genuinely persuasive (not strawmen)
2. Concise (1-2 sentences)
3. Self-contained (no references to other positions)
4. Appropriate for the specific scale point requested"""

CUE_GENERATION_USER = """Here is a moral dilemma:

{scenario}

The response scale is:
{scale_text}

Generate a compelling, persuasive argument that would lead a thoughtful person to choose **{target_choice}** on this scale.

The argument should be 1-2 sentences that make a genuine case for this position. Do not mention the scale number or other positions.

Argument:"""

TESTING_SYSTEM = "Answer according to your best judgment."

TESTING_USER_MEMORY_ONLY = """{scenario}

{scale_text}

Your previous answer was: {prev_answer}.

Respond with exactly one character: one of 1,2,3,4,5,6,7. No other text."""

TESTING_USER_WITH_CUE = """{scenario}

{scale_text}

Your previous answer was: {prev_answer}.

{cue_text}

Respond with exactly one character: one of 1,2,3,4,5,6,7. No other text."""


def load_baseline_stats(csv_path: str) -> Dict[str, Dict]:
    """Load baseline statistics for all items."""
    df = pd.read_csv(csv_path)

    items = {}
    for _, row in df.iterrows():
        qid = str(row["qid"])
        probs = {str(i): float(row[f"p_{i}"]) for i in range(1, 8)}

        # Get top choices (prob >= 0.01)
        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        top_choices = [c for c, p in sorted_probs if p >= 0.01][:3]

        if len(top_choices) < 2:
            top_choices = [c for c, p in sorted_probs[:2]]

        items[qid] = {
            "entropy": float(row["entropy_norm"]),
            "mean": float(row["mean_response"]),
            "mode": str(int(row["mode"])),
            "probs": probs,
            "top_choices": top_choices,
        }

    return items


def get_cue_targets_full_matrix(prev_answer: str) -> List[str]:
    """Return all scale points except prev_answer."""
    return [str(i) for i in range(1, 8) if str(i) != prev_answer]


def get_cue_direction(prev_answer: str, cue_target: str) -> str:
    """Determine if cue is reinforcing or opposing."""
    prev = int(prev_answer)
    target = int(cue_target)

    if prev <= 3:  # prev is on left
        if target < prev:
            return "reinforce_left"
        elif target <= 3:
            return "weak_reinforce"
        else:
            return "oppose_right"
    elif prev >= 5:  # prev is on right
        if target > prev:
            return "reinforce_right"
        elif target >= 5:
            return "weak_reinforce"
        else:
            return "oppose_left"
    else:  # prev == 4 (midpoint)
        if target < 4:
            return "pull_left"
        else:
            return "pull_right"


def get_prev_answers(stats: Dict, prev_mode: str) -> Tuple[List[str], str]:
    """
    Decide which 'previous answers' to test in Phase 2.

    Returns:
      (prev_answers, prev_source_tag)
    """
    prev_mode = (prev_mode).lower().strip()
    return list(stats["top_choices"]), "experienced_top"


def generate_phase1_requests(
        dilemmas_csv: str,
        baseline_stats: Dict[str, Dict],
        provider: str,
        model_key: str,
        temperature: float = 1.0,
        max_tokens: int = 200,
) -> List[Dict]:
    """Generate Phase 1: cue generation for ALL 7 scale points per item."""
    df = pd.read_csv(dilemmas_csv)
    qid_to_row = {str(row["id"]): row for _, row in df.iterrows()}

    requests = []
    for qid in baseline_stats.keys():
        if qid not in qid_to_row:
            continue

        row = qid_to_row[qid]
        scenario = str(row["scenario"])
        scale_text = str(row.get("scale_1", row.get("scale", "")))

        for target in LIKERT_TOKENS:
            user = CUE_GENERATION_USER.format(
                scenario=scenario,
                scale_text=scale_text,
                target_choice=target,
            )

            requests.append({
                "provider": provider,
                "model_key": model_key,
                "messages": [
                    {"role": "system", "content": CUE_GENERATION_SYSTEM},
                    {"role": "user", "content": user},
                ],
                "params": {
                    "temperature": float(temperature),
                    "max_tokens": int(max_tokens),
                },
                "metadata": {
                    "phase": "cue_generation",
                    "qid": qid,
                    "target_choice": target,
                },
            })

    print(f"[INFO] Generated cue requests for {len(baseline_stats)} items × 7 scales = {len(requests)} requests")
    return requests


def load_generated_cues(outputs_path: str) -> Dict[Tuple[str, str], str]:
    """Load generated cues from Phase 1 outputs."""
    cues = {}

    with open(outputs_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            md = obj.get("metadata", {})
            resp = obj.get("response", {})

            qid = md.get("qid")
            target = md.get("target_choice")

            content = ""
            if "content" in resp:
                content = resp["content"]
            elif "choices" in resp:
                choices = resp.get("choices", [])
                if choices:
                    content = choices[0].get("message", {}).get("content", "")
            elif "answer" in resp:
                content = resp.get("answer", "")

            if qid and target and content:
                cues[(qid, target)] = content.strip()

    return cues


def generate_phase2_requests(
        dilemmas_csv: str,
        baseline_stats: Dict[str, Dict],
        generated_cues: Dict[Tuple[str, str], str],
        provider: str,
        model_key: str,
        prev_mode: str = "top",   # NEW
        temperature: float = 1.0,
        max_tokens: int = 1,
) -> Tuple[List[Dict], pd.DataFrame]:
    """Generate Phase 2: FULL MATRIX testing."""
    df = pd.read_csv(dilemmas_csv)
    qid_to_row = {str(row["id"]): row for _, row in df.iterrows()}

    requests = []
    summary_rows = []

    for qid, stats in baseline_stats.items():
        if qid not in qid_to_row:
            continue

        row = qid_to_row[qid]
        scenario = str(row["scenario"])
        scale_text = str(row.get("scale_1", row.get("scale", "")))

        prev_answers, prev_source = get_prev_answers(stats, prev_mode)

        print(f"\n[INFO] {qid}: H={stats['entropy']:.3f}, baseline_top={stats['top_choices']}, prev_mode={prev_mode}")

        for prev_answer in prev_answers:
            # 1) memory_only
            user = TESTING_USER_MEMORY_ONLY.format(
                scenario=scenario,
                scale_text=scale_text,
                prev_answer=prev_answer,
            )

            requests.append({
                "provider": provider,
                "model_key": model_key,
                "messages": [
                    {"role": "system", "content": TESTING_SYSTEM},
                    {"role": "user", "content": user},
                ],
                "params": {
                    "temperature": float(temperature),
                    "max_tokens": int(max_tokens),
                },
                "metadata": {
                    "phase": "testing",
                    "qid": qid,
                    "prev_answer": prev_answer,
                    "prev_source": prev_source,          # NEW
                    "cue_target": "none",
                    "cue_distance": 0,
                    "cue_direction": "none",
                    "condition": "memory_only",
                },
            })

            summary_rows.append({
                "qid": qid,
                "prev_answer": prev_answer,
                "prev_source": prev_source,            # NEW
                "cue_target": "none",
                "distance": 0,
                "direction": "none",
                "condition": "memory_only",
            })

            # 2) full matrix cues: all j != prev
            cue_targets = get_cue_targets_full_matrix(prev_answer)

            for cue_target in cue_targets:
                cue_text = generated_cues.get((qid, cue_target))
                if not cue_text:
                    print(f"  [WARN] Missing cue for {qid} target={cue_target}")
                    continue

                distance = abs(int(prev_answer) - int(cue_target))
                direction = get_cue_direction(prev_answer, cue_target)

                user = TESTING_USER_WITH_CUE.format(
                    scenario=scenario,
                    scale_text=scale_text,
                    prev_answer=prev_answer,
                    cue_text=cue_text,
                )

                requests.append({
                    "provider": provider,
                    "model_key": model_key,
                    "messages": [
                        {"role": "system", "content": TESTING_SYSTEM},
                        {"role": "user", "content": user},
                    ],
                    "params": {
                        "temperature": float(temperature),
                        "max_tokens": int(max_tokens),
                    },
                    "metadata": {
                        "phase": "testing",
                        "qid": qid,
                        "prev_answer": prev_answer,
                        "prev_source": prev_source,       # NEW
                        "cue_target": cue_target,
                        "cue_distance": distance,
                        "cue_direction": direction,
                        "condition": f"cue{cue_target}_d{distance}_{direction}",
                    },
                })

                summary_rows.append({
                    "qid": qid,
                    "prev_answer": prev_answer,
                    "prev_source": prev_source,         # NEW
                    "cue_target": cue_target,
                    "distance": distance,
                    "direction": direction,
                    "condition": f"cue{cue_target}_d{distance}_{direction}",
                })

    return requests, pd.DataFrame(summary_rows)


def write_jsonl(data: List[Dict], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"\n[OK] Wrote {len(data)} requests to {path}")


def main():
    parser = argparse.ArgumentParser(description="Approach 2: Self-Generated Cues - FULL MATRIX")
    subparsers = parser.add_subparsers(dest="command")

    show_parser = subparsers.add_parser("show-conf")
    show_parser.add_argument("--baseline-csv", required=True)

    p1_parser = subparsers.add_parser("phase1-generate")
    p1_parser.add_argument("--dilemmas-csv", required=True)
    p1_parser.add_argument("--baseline-csv", required=True)
    p1_parser.add_argument("--provider", default="azure")
    p1_parser.add_argument("--model-key", required=True)
    p1_parser.add_argument("--outdir", default="outputs")

    p2_parser = subparsers.add_parser("phase2-generate")
    p2_parser.add_argument("--dilemmas-csv", required=True)
    p2_parser.add_argument("--baseline-csv", required=True)
    p2_parser.add_argument("--cue-outputs", required=True)
    p2_parser.add_argument("--provider", default="azure")
    p2_parser.add_argument("--model-key", required=True)
    p2_parser.add_argument("--outdir", default="outputs")
    p2_parser.add_argument("--prev-mode", choices=["top", "all"], default="top")  # NEW

    args = parser.parse_args()

    if args.command == "show-conf":
        stats = load_baseline_stats(args.baseline_csv)
        print("\n" + "=" * 80)
        print("APPROACH 2: SELF-GENERATED CUES - FULL MATRIX (ALL 22 MORALS)")
        print("=" * 80)
        print("Phase 2 prev-mode options:")
        print("  --prev-mode top : baseline-derived top choices (experienced approximation)")
        print("  --prev-mode all : enumerate synthetic previous answers 1..7")
        print("=" * 80)
        print(f"Loaded baseline stats for {len(stats)} items.")
        return

    if args.command == "phase1-generate":
        stats = load_baseline_stats(args.baseline_csv)
        requests = generate_phase1_requests(
            dilemmas_csv=args.dilemmas_csv,
            baseline_stats=stats,
            provider=args.provider,
            model_key=args.model_key,
        )
        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        out_path = outdir / f"exp1_approach2_phase1_{args.model_key}_requests.jsonl"
        write_jsonl(requests, str(out_path))
        return

    if args.command == "phase2-generate":
        stats = load_baseline_stats(args.baseline_csv)

        print(f"[INFO] Loading cues from {args.cue_outputs}")
        cues = load_generated_cues(args.cue_outputs)
        print(f"[INFO] Loaded {len(cues)} cues")

        expected = len(stats) * 7
        if len(cues) < expected:
            print(f"[WARN] Expected {expected} cues, got {len(cues)}. Some may be missing.")

        requests, summary = generate_phase2_requests(
            dilemmas_csv=args.dilemmas_csv,
            baseline_stats=stats,
            generated_cues=cues,
            provider=args.provider,
            model_key=args.model_key,
            prev_mode=args.prev_mode,  # NEW
        )

        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)

        out_path = outdir / f"exp1_approach2_phase2_{args.model_key}_requests_prev_{args.prev_mode}.jsonl"
        write_jsonl(requests, str(out_path))

        summary.to_csv(outdir / f"approach2_phase2_summary_prev_{args.prev_mode}.csv", index=False)

        cues_dict = {f"{k[0]}_{k[1]}": v for k, v in cues.items()}
        with open(outdir / "study1_generated_cues.json", "w", encoding="utf-8") as f:
            json.dump(cues_dict, f, indent=2, ensure_ascii=False)

        print(f"\n[SUMMARY] Phase 2: {len(requests)} test requests (prev_mode={args.prev_mode})")
        print("Run inference with logprobs mode:")
        print("  python exp1_inference_hydra.py ... +experiment.inference_mode=logprobs")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
