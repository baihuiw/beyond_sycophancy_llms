#!/usr/bin/env python3
"""
Identify Study 2 Candidates from Baseline Outputs

Analyzes baseline responses to identify candidates where the model showed
irrational/inconsistent responses (outliers from the modal response).

This follows the Study 2 design: find instances where the model's response
differs from its typical response pattern.

Usage:
    python identify_study2_candidates.py \
        --baseline outputs/study2_baseline_llama3_70b.jsonl \
        --outputs study2_candidates_llama3_70b.json \
        --min_distance 1 \
        --max_candidates_per_item 3
"""

import argparse
import json
from collections import Counter, defaultdict
from typing import Dict, List, Tuple


def load_jsonl(filepath: str) -> List[Dict]:
    """Load JSONL file."""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def extract_answer(obj: Dict) -> str:
    """Extract answer from response object."""
    resp = obj.get('response', {})

    # Try different possible fields
    for field in ['answer', 'parsed_answer']:
        if field in resp and resp[field] in ['1', '2', '3', '4', '5', '6', '7']:
            return resp[field]

    # Try content field
    content = resp.get('content', '')
    if isinstance(content, str) and content.strip() in ['1', '2', '3', '4', '5', '6', '7']:
        return content.strip()

    return None


def identify_candidates(
        baseline_outputs: List[Dict],
        min_distance: int = 1,
        max_candidates_per_item: int = 2,
        min_irrational_prob: float = 0.0001,
) -> List[Dict]:
    candidates = []

    for obj in baseline_outputs:
        md = obj.get('metadata', {})
        qid = md.get('qid')
        if not qid:
            continue

        resp = obj.get('response', {})
        probs = resp.get('probs')
        if not probs:
            continue

        mode_pos = max(probs, key=lambda k: float(probs[k]))
        mode_prob = float(probs[mode_pos])

        # All non-mode positions, sorted by probability descending
        non_mode = sorted(
            [
                {
                    'qid': qid,
                    'irrational_choice': int(pos_str),
                    'mode_choice': int(mode_pos),
                    'mode_prob': mode_prob,
                    'irrational_prob': float(prob_val),
                    'distance': abs(int(pos_str) - int(mode_pos)),
                }
                for pos_str, prob_val in probs.items()
                if pos_str != mode_pos
            ],
            key=lambda x: -x['irrational_prob'],
        )

        # Apply distance and prob filters
        filtered = [
            c for c in non_mode
            if c['distance'] >= min_distance and c['irrational_prob'] >= min_irrational_prob
        ]

        # Guarantee at least one candidate per item:
        # if nothing passes the filters, take the single best non-mode position
        if not filtered:
            # relax distance filter first, then prob filter
            filtered = [c for c in non_mode if c['distance'] >= min_distance]
            if not filtered:
                filtered = non_mode  # last resort: take closest non-mode position

        # Sort final set: highest prob first (most meaningful alternative)
        filtered.sort(key=lambda x: -x['irrational_prob'])
        candidates.extend(filtered[:max_candidates_per_item])

    return candidates


def main():
    parser = argparse.ArgumentParser(description="Identify Study 2 candidates from baseline outputs")
    parser.add_argument('--baseline', required=True, help='Path to baseline outputs JSONL')
    parser.add_argument('--outputs', required=True, help='Path to outputs candidates JSON')
    parser.add_argument('--min_distance', type=int, default=1,
                        help='Minimum distance from mode to be considered outlier (default: 1)')
    parser.add_argument('--max_candidates_per_item', type=int, default=2,
                        help='Maximum candidates to select per item (default: 3)')
    parser.add_argument('--min_mode_prob', type=float, default=0.0,
                        help='Minimum mode probability to consider item (default: 0.0, disabled)')
    parser.add_argument('--min_irrational_prob', type=float, default=0.10,
                        help='Minimum probability of irrational position to be a candidate (default: 0.10)')

    args = parser.parse_args()

    print("=" * 70)
    print("STUDY 2 CANDIDATE IDENTIFICATION")
    print("=" * 70)
    print(f"\nLoading baseline outputs from: {args.baseline}")

    baseline_outputs = load_jsonl(args.baseline)
    print(f"Loaded {len(baseline_outputs)} baseline responses")

    # Identify candidates
    print(f"\nIdentifying candidates with:")
    print(f"  - min_distance: {args.min_distance}")
    print(f"  - max_candidates_per_item: {args.max_candidates_per_item}")
    print(f"  - min_mode_prob: {args.min_mode_prob}")

    candidates = identify_candidates(
        baseline_outputs,
        min_distance=args.min_distance,
        max_candidates_per_item=args.max_candidates_per_item,
        min_irrational_prob=args.min_irrational_prob,
    )

    # Filter by mode probability
    candidates = [c for c in candidates if c['mode_prob'] >= args.min_mode_prob]

    # Statistics
    print(f"\n{'=' * 70}")
    print(f"RESULTS:")
    print(f"{'=' * 70}")
    print(f"Total candidates identified: {len(candidates)}")

    if candidates:
        # Group by qid to count items
        qids = set(c['qid'] for c in candidates)
        print(f"Number of items with candidates: {len(qids)}")
        print(f"Average candidates per item: {len(candidates) / len(qids):.2f}")

        # Distance distribution
        distance_counts = Counter([c['distance'] for c in candidates])
        print(f"\nDistance distribution:")
        for dist in sorted(distance_counts.keys()):
            print(f"  Distance {dist}: {distance_counts[dist]} candidates")

        # Show sample candidates
        print(f"\nSample candidates:")
        for i, c in enumerate(candidates[:5]):
            print(f"\n  Candidate {i + 1}:")
            print(f"    QID: {c['qid']}")
            print(f"    Irrational choice: {c['irrational_choice']}")
            print(f"    Mode choice: {c['mode_choice']}")
            print(f"    Distance: {c['distance']}")
            print(f"    Mode probability: {c['mode_prob']:.2%}")
            print(f"    Irrational probability: {c['irrational_prob']:.2%}")

    # Save candidates
    with open(args.output, 'w') as f:
        json.dump(candidates, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"Saved {len(candidates)} candidates to: {args.output}")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
