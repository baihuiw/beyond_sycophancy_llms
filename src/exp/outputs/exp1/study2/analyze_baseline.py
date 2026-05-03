import argparse
import json
import numpy as np
from pathlib import Path
from collections import defaultdict


def get_dist7(probs_dict):
    """Extract normalized 7-position distribution."""
    d = np.array([float(probs_dict.get(str(i), 0) or 0) for i in range(1, 8)])
    return d / d.sum() if d.sum() > 0 else d


def main():
    parser = argparse.ArgumentParser(description="Analyze baseline → extract mode + Study 2 candidates")
    parser.add_argument("--baseline", required=True, help="Path to baseline results JSONL")
    parser.add_argument("--model", default="model", help="Model name prefix for output files")
    parser.add_argument("--output_dir", default="outputs/", help="Output directory")
    parser.add_argument("--min_mode_prob", type=float, default=0.50,
                        help="Minimum P(mode) to consider a dilemma for Study 2")
    parser.add_argument("--min_irr_prob", type=float, default=0.0001,
                        help="Minimum P(irrational) to include as candidate")
    parser.add_argument("--max_candidates_per_qid", type=int, default=3,
                        help="Max irrational candidates per dilemma (closest to mode first)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load baseline results ──
    results = []
    with open(args.baseline) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            resp = obj.get("response", {})
            if resp.get("status_code") != 200:
                continue
            results.append(obj)

    print(f"Loaded {len(results)} baseline results")

    # ── If multiple reps, aggregate by qid (take mean probs) ──
    per_qid = defaultdict(list)
    for obj in results:
        qid = obj.get("metadata", {}).get("qid")
        if not qid:
            continue
        probs = obj.get("response", {}).get("probs", {})
        if probs:
            per_qid[qid].append(probs)

    print(f"Unique dilemmas: {len(per_qid)}")

    # ── Compute per-qid summary ──
    summaries = {}
    for qid, probs_list in sorted(per_qid.items()):
        # Average across reps if multiple
        avg_probs = {}
        for pos in ["1", "2", "3", "4", "5", "6", "7"]:
            vals = [float(p.get(pos, 0) or 0) for p in probs_list]
            avg_probs[pos] = np.mean(vals)

        # Normalize
        total = sum(avg_probs.values())
        if total > 0:
            avg_probs = {k: v / total for k, v in avg_probs.items()}

        # Mode
        mode_pos = max(avg_probs, key=avg_probs.get)
        mode_prob = avg_probs[mode_pos]

        # Entropy
        p_arr = np.array([avg_probs[str(i)] for i in range(1, 8)])
        entropy = -np.sum(p_arr * np.log(p_arr + 1e-12))
        entropy_norm = entropy / np.log(7)

        summaries[qid] = {
            "qid": qid,
            "mode_choice": int(mode_pos),
            "mode_prob": float(mode_prob),
            "probs": {k: float(v) for k, v in avg_probs.items()},
            "entropy_norm": float(entropy_norm),
            "n_reps": len(probs_list),
        }

    # Save summary
    summary_path = out_dir / f"{args.model}_baseline_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"Saved baseline summary: {summary_path}")

    # ── Extract Study 2 candidates ──
    # Rule: for every dilemma, include ALL positions where P > min_irr_prob and pos != mode
    # No minimum mode_prob filter — every dilemma participates
    # No max_candidates_per_qid — include all qualifying positions
    candidates = []
    qids_with_candidates = set()
    qids_without = []

    for qid, s in sorted(summaries.items()):
        mode = s["mode_choice"]
        mode_prob = s["mode_prob"]
        probs = s["probs"]

        irr_candidates = []
        for pos_str, prob in probs.items():
            pos = int(pos_str)
            if pos == mode:
                continue
            if prob < args.min_irr_prob:
                continue

            irr_candidates.append({
                "qid": qid,
                "irrational_choice": pos,
                "irrational_prob": float(prob),
                "mode_choice": mode,
                "mode_prob": float(mode_prob),
                "distance": abs(pos - mode),
            })

        # Sort by distance (closest first)
        irr_candidates.sort(key=lambda x: x["distance"])

        if irr_candidates:
            candidates.extend(irr_candidates)
            qids_with_candidates.add(qid)
        else:
            # Fallback: if no position meets threshold, take the highest non-mode prob
            qids_without.append(qid)
            non_mode = [(int(k), v) for k, v in probs.items() if int(k) != mode and v > 0]
            if non_mode:
                non_mode.sort(key=lambda x: -x[1])
                best_pos, best_prob = non_mode[0]
                candidates.append({
                    "qid": qid,
                    "irrational_choice": best_pos,
                    "irrational_prob": float(best_prob),
                    "mode_choice": mode,
                    "mode_prob": float(mode_prob),
                    "distance": abs(best_pos - mode),
                })
                qids_with_candidates.add(qid)

    if qids_without:
        print(f"[INFO] {len(qids_without)} dilemmas below min_irr_prob threshold — used fallback")

    # Save candidates
    candidates_path = out_dir / f"{args.model}_study2_candidates.json"
    with open(candidates_path, "w") as f:
        json.dump(candidates, f, indent=2)

    print(f"\nStudy 2 Candidates: {len(candidates)} from {len(set(c['qid'] for c in candidates))} dilemmas")
    print(f"Saved: {candidates_path}")

    # ── Print summary ──
    print(f"\n{'='*60}")
    print(f"BASELINE ANALYSIS SUMMARY — {args.model}")
    print(f"{'='*60}")

    modes = [s["mode_choice"] for s in summaries.values()]
    mode_probs = [s["mode_prob"] for s in summaries.values()]
    entropies = [s["entropy_norm"] for s in summaries.values()]

    from collections import Counter
    mode_dist = Counter(modes)
    print(f"\nMode distribution: {dict(sorted(mode_dist.items()))}")
    print(f"Mean P(mode): {np.mean(mode_probs):.3f} (SD={np.std(mode_probs):.3f})")
    print(f"Mean entropy (norm): {np.mean(entropies):.3f}")

    # Position breakdown
    extreme = sum(1 for m in modes if m in [1, 2, 6, 7])
    moderate = sum(1 for m in modes if m in [3, 5])
    neutral = sum(1 for m in modes if m == 4)
    print(f"\nExtreme (1-2, 6-7): {extreme} ({extreme/len(modes)*100:.1f}%)")
    print(f"Moderate (3, 5):    {moderate} ({moderate/len(modes)*100:.1f}%)")
    print(f"Neutral (4):        {neutral} ({neutral/len(modes)*100:.1f}%)")

    # Candidate breakdown
    dists = [c["distance"] for c in candidates]
    dist_counts = Counter(dists)
    print(f"\nCandidates by distance from mode:")
    for d in sorted(dist_counts):
        print(f"  distance {d}: {dist_counts[d]} candidates")

    print(f"\nNext steps:")
    print(f"  1. Review candidates: {candidates_path}")
    print(f"  2. Generate Study 2 requests:")
    print(f"     python unified_generate_requests.py \\")
    print(f"         --study study2 --mode testing \\")
    print(f"         --csv moral_dilemmas.csv \\")
    print(f"         --model_key {args.model} --provider openrouter \\")
    print(f"         --candidates {candidates_path} \\")
    print(f"         --pretrained_cues study2_pretrained_cues.jsonl \\")
    print(f"         --output requests/{args.model}_study2_testing.jsonl")


if __name__ == "__main__":
    main()
