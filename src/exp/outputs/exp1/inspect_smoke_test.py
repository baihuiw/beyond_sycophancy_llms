import json
import sys
from collections import defaultdict
from pathlib import Path


DIGITS = ["1", "2", "3", "4", "5", "6", "7"]


def load_jsonl(path: Path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def render_distribution(probs: dict, width: int = 30) -> str:
    """ASCII bar chart of the 7-way distribution."""
    if not probs:
        return "  (no distribution)"
    lines = []
    for d in DIGITS:
        p = probs.get(d) if probs else None
        if p is None:
            bar = "(no data)"
            pct = "   ?"
        else:
            bar_len = int(round(p * width))
            bar = "█" * bar_len + "·" * (width - bar_len)
            pct = f"{p*100:5.1f}%"
        lines.append(f"    {d}: {bar} {pct}")
    return "\n".join(lines)


def render_episode(transcript: list, summary: dict):
    """Print a full debate episode transcript + summary."""
    qid = summary["qid"]
    rep = summary["rep"]
    n_rounds = summary["n_rounds"]

    print()
    print("═" * 78)
    print(f"  EPISODE  qid={qid}  rep={rep}  n_rounds={n_rounds}")
    print("═" * 78)

    # Group transcript by (round, agent)
    by_round_agent = defaultdict(dict)
    for t in transcript:
        key = (t["round"], t["agent"])
        by_round_agent[key][t["kind"]] = t

    # ── Baselines (round 0) ──
    print("\n───────────────  ROUND 0: Independent baselines  ───────────────\n")
    for agent in ("A1", "A2"):
        entry = by_round_agent[(0, agent)]
        commit = entry.get("commitment", {})
        arg = entry.get("argument", {})

        print(f"  ◆ {agent} — baseline commitment digit: {commit.get('position')}")
        print(render_distribution(commit.get("probs")))
        print(f"\n  {agent} — opening argument:")
        print(indent(arg.get("text", "(missing)"), "      "))
        print()

    # ── Debate rounds ──
    for r in range(1, n_rounds + 1):
        print(f"───────────────  ROUND {r}  ───────────────\n")
        for agent in ("A1", "A2"):
            entry = by_round_agent[(r, agent)]
            arg = entry.get("argument", {})
            commit = entry.get("commitment", {})

            print(f"  ◆ {agent} — argument:")
            print(indent(arg.get("text", "(missing)"), "      "))
            print()
            print(f"  ◆ {agent} — commitment after this argument: position = {commit.get('position')}")
            print(render_distribution(commit.get("probs")))
            if commit.get("text"):
                print(f"\n  {agent} — justification:")
                print(indent(commit.get("text", ""), "      "))
            print()

    # ── Summary ──
    print("───────────────  SUMMARY  ───────────────\n")
    print(f"  A1 trajectory: {summary['a1_position_trajectory']}")
    print(f"  A2 trajectory: {summary['a2_position_trajectory']}")
    print(f"  Initial distance: {summary['initial_distance']}  →  Final distance: {summary['final_distance']}")
    print(f"  Pattern: {summary['pattern']}")
    print(f"  A1 position changes: {summary['a1_n_changes']}  |  switched side: {summary['a1_switched_side']}  |  moved toward peer: {summary['a1_moved_toward_peer']}")
    print(f"  A2 position changes: {summary['a2_n_changes']}  |  switched side: {summary['a2_switched_side']}  |  moved toward peer: {summary['a2_moved_toward_peer']}")
    print()


def indent(text: str, prefix: str = "    ") -> str:
    if not text:
        return prefix + "(empty)"
    return "\n".join(prefix + line for line in text.splitlines())


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 inspect_smoke_test.py <output-dir>")
        print("  (e.g. outputs/study4_smoke)")
        sys.exit(1)

    out_dir = Path(sys.argv[1])
    transcripts_path = out_dir / "transcripts.jsonl"
    summaries_path = out_dir / "summaries.jsonl"

    if not transcripts_path.exists() or not summaries_path.exists():
        print(f"ERROR: Missing {transcripts_path} or {summaries_path}")
        sys.exit(1)

    transcripts = load_jsonl(transcripts_path)
    summaries = load_jsonl(summaries_path)
    print(f"Loaded {len(transcripts)} transcript events across {len(summaries)} episodes.\n")

    # Index transcripts by (qid, rep)
    by_ep = defaultdict(list)
    for t in transcripts:
        by_ep[(t["qid"], t["rep"])].append(t)

    for s in summaries:
        ep_transcript = by_ep[(s["qid"], s["rep"])]
        render_episode(ep_transcript, s)

    # Cross-episode digest
    print("═" * 78)
    print("  CROSS-EPISODE DIGEST")
    print("═" * 78)
    patterns = defaultdict(int)
    for s in summaries:
        patterns[s["pattern"]] += 1
    print("  Patterns:")
    for p, n in patterns.items():
        print(f"    {p}: {n}")

    # Mean initial/final distance
    inits = [s["initial_distance"] for s in summaries if s["initial_distance"] is not None]
    finals = [s["final_distance"] for s in summaries if s["final_distance"] is not None]
    if inits:
        print(f"  Mean initial distance: {sum(inits)/len(inits):.2f}")
    if finals:
        print(f"  Mean final distance:   {sum(finals)/len(finals):.2f}")


if __name__ == "__main__":
    main()
