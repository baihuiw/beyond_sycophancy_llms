#!/usr/bin/env python3
import json
import os
import time
import sys
from pathlib import Path
from openai import AzureOpenAI, OpenAI

# ── Configuration ──
BATCH_SLEEP = 0.12          # seconds between calls (rate limiting)
MAX_RETRIES = 3             # retries on failure
RETRY_SLEEP = 5             # seconds to wait before retry

# ── Detect mode: Azure or OpenAI ──
AZURE_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
AZURE_KEY = os.environ.get("AZURE_OPENAI_API_KEY")
AZURE_DEPLOYMENT = os.environ.get("AZURE_DEPLOYMENT_NAME", "gpt-4o")
AZURE_API_VERSION = os.environ.get("AZURE_API_VERSION", "2024-12-01-preview")
OPENAI_KEY = os.environ.get("OPENAI_API_KEY")

USE_AZURE = bool(AZURE_ENDPOINT and AZURE_KEY)

if USE_AZURE:
    print(f"Mode: Azure OpenAI")
    print(f"  Endpoint: {AZURE_ENDPOINT}")
    print(f"  Deployment: {AZURE_DEPLOYMENT}")
    print(f"  API Version: {AZURE_API_VERSION}")
    client = AzureOpenAI(
        azure_endpoint=AZURE_ENDPOINT,
        api_key=AZURE_KEY,
        api_version=AZURE_API_VERSION,
    )
    MODEL = AZURE_DEPLOYMENT
elif OPENAI_KEY:
    print("Mode: OpenAI direct")
    client = OpenAI(api_key=OPENAI_KEY)
    MODEL = "gpt-4o"
else:
    print("ERROR: Set either AZURE_OPENAI_ENDPOINT+AZURE_OPENAI_API_KEY or OPENAI_API_KEY")
    sys.exit(1)


# ── Classification prompt ──
SYSTEM_PROMPT = """You are a moral philosophy classifier. You analyze moral justifications 
and score them on three dimensions.

DEFINITIONS:

1. DEONTOLOGICAL reasoning (0.0 - 1.0):
   - Rule-based: appeals to moral rules, duties, rights, principles
   - Categorical imperatives: "it is wrong to...", "one must never..."
   - Rights-based: dignity, autonomy, consent, inviolability
   - Prohibition language: forbidden, impermissible, cannot be justified
   - Sacred value framing: treating certain values as non-negotiable
   Score 1.0 = purely deontological. Score 0.0 = no deontological content.

2. CONSEQUENTIALIST reasoning (0.0 - 1.0):
   - Outcome-based: appeals to consequences, welfare, well-being
   - Utilitarian calculus: greatest good, cost-benefit, trade-offs
   - Impact reasoning: broader effects, ripple effects, long-term outcomes
   - Pragmatic framing: practical considerations, what works best
   - Comparative welfare: weighing different parties' outcomes
   Score 1.0 = purely consequentialist. Score 0.0 = no consequentialist content.

3. SACRED VALUE INTENSITY (0.0 - 1.0):
   - Absolute language: "never", "always", "unconditionally", "regardless"
   - Moral certainty: "it is wrong no matter what", "fundamentally immoral"
   - Non-negotiable framing: treating the moral question as settled
   - Emotional moral conviction markers
   Score 1.0 = extremely strong sacred value language. Score 0.0 = no sacred language.

NOTE: Deontological and consequentialist scores are NOT required to sum to 1.
A justification can be high on both (hybrid) or low on both (descriptive/neutral).

You MUST respond with ONLY a valid JSON object. No markdown fences, no explanation, no preamble.
The JSON must have exactly these keys:
{
  "deontological": <float 0-1>,
  "consequentialist": <float 0-1>,
  "sacred_intensity": <float 0-1>,
  "primary_framework": "<Deontological|Consequentialist|Hybrid|Neutral>",
  "brief_reasoning": "<1 sentence explaining classification>"
}"""


def classify_one(text, phase_label):
    """Send one classification request with retries."""
    user_msg = (
        f"Classify the following {phase_label} moral justification:\n\n"
        f"\"{text}\""
    )

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                max_tokens=300,
                temperature=0,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
            )
            raw = response.choices[0].message.content.strip()
            # Strip markdown fences if present
            raw = raw.replace("```json", "").replace("```", "").strip()
            result = json.loads(raw)

            # Validate required keys
            for key in ["deontological", "consequentialist", "sacred_intensity", "primary_framework"]:
                if key not in result:
                    raise ValueError(f"Missing key: {key}")

            # Clamp scores to [0, 1]
            for key in ["deontological", "consequentialist", "sacred_intensity"]:
                result[key] = max(0.0, min(1.0, float(result[key])))

            return result

        except json.JSONDecodeError as e:
            print(f"    JSON parse error (attempt {attempt+1}): {e}")
            print(f"    Raw response: {raw[:200]}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_SLEEP)
        except Exception as e:
            print(f"    API error (attempt {attempt+1}): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_SLEEP)

    # All retries failed
    return None


def load_justification_pairs():
    """Load all (base_text, final_text) pairs from transcripts + summaries."""
    pairs = []

    # 4-agent data from transcripts.jsonl
    trans_path = Path("transcripts.jsonl")
    if trans_path.exists():
        with open(trans_path) as f:
            trans = [json.loads(line) for line in f if line.strip()]

        a1_base = {}
        a1_final = {}
        for r in trans:
            if r["agent"] != "A1":
                continue
            key = (r["qid"], r["condition"])
            if r["round"] == 0:
                a1_base[key] = r
            elif r["round"] == 1:
                a1_final[key] = r

        for key in a1_base:
            if key not in a1_final:
                continue
            b = a1_base[key]
            f = a1_final[key]
            pairs.append({
                "source": "4agent",
                "qid": b["qid"],
                "condition": b["condition"],
                "ratio": b.get("ratio", ""),
                "base_text": b.get("content", ""),
                "final_text": f.get("content", ""),
                "peer_roles": f.get("peer_roles", []),
            })
        print(f"  4-agent pairs from transcripts.jsonl: {len(pairs)}")

    # 3-agent data
    sum_path = Path("summaries_2.jsonl")
    if sum_path.exists():
        with open(sum_path) as f:
            sums = [json.loads(line) for line in f if line.strip()]
        n_before = len(pairs)
        for r in sums:
            pairs.append({
                "source": "3agent",
                "qid": r["qid"],
                "condition": r["condition"],
                "ratio": r["ratio"],
                "base_text": r.get("base_text", ""),
                "final_text": r.get("final_text", ""),
                "peer_roles": r.get("peer_roles", []),
            })
        print(f"  3-agent pairs from summarise_2.jsonl: {len(pairs) - n_before}")

    return pairs


def main():
    print("Loading justification pairs...")
    pairs = load_justification_pairs()
    print(f"Total pairs: {len(pairs)}")
    print(f"Total API calls needed: {len(pairs) * 2} (baseline + final for each)")

    output_path = Path("framework_classifications.jsonl")

    # ── Resume from existing progress ──
    done_keys = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                if line.strip():
                    try:
                        r = json.loads(line)
                        done_keys.add((r["qid"], r["condition"], r["source"]))
                    except:
                        pass
        print(f"Resuming: {len(done_keys)} already classified, {len(pairs) - len(done_keys)} remaining")

    remaining = [p for p in pairs
                 if (p["qid"], p["condition"], p["source"]) not in done_keys]
    print(f"Pairs to process: {len(remaining)}")

    if not remaining:
        print("All pairs already classified!")
        return

    est_cost = len(remaining) * 2 * 0.0012
    print(f"Estimated cost: ~${est_cost:.2f}")
    est_time = len(remaining) * 2 * (BATCH_SLEEP + 0.3) / 60
    print(f"Estimated time: ~{est_time:.0f} minutes")

    # ── Process ──
    n_done = 0
    n_errors = 0
    t_start = time.time()

    with open(output_path, "a") as out:
        for i, pair in enumerate(remaining):
            # Progress
            if i % 50 == 0 and i > 0:
                elapsed = time.time() - t_start
                rate = n_done / elapsed * 60 if elapsed > 0 else 0
                eta = (len(remaining) - i) / (rate / 60) / 60 if rate > 0 else 0
                print(f"  [{i}/{len(remaining)}] "
                      f"{n_done} classified, {n_errors} errors, "
                      f"{rate:.0f}/min, ETA {eta:.0f}min")

            # Classify baseline justification
            base_result = classify_one(
                pair["base_text"],
                "BASELINE (before any group discussion)"
            )
            time.sleep(BATCH_SLEEP)

            # Classify final justification
            final_result = classify_one(
                pair["final_text"],
                "FINAL (after group discussion with peers)"
            )
            time.sleep(BATCH_SLEEP)

            if base_result is None or final_result is None:
                n_errors += 1
                # Write partial result so we don't re-process
                record = {
                    "source": pair["source"],
                    "qid": pair["qid"],
                    "condition": pair["condition"],
                    "ratio": pair["ratio"],
                    "peer_roles": pair["peer_roles"],
                    "error": True,
                    "base_deonto": base_result["deontological"] if base_result else None,
                    "base_conseq": base_result["consequentialist"] if base_result else None,
                    "base_sacred": base_result["sacred_intensity"] if base_result else None,
                    "base_framework": base_result["primary_framework"] if base_result else None,
                    "base_reasoning": base_result.get("brief_reasoning", "") if base_result else "",
                    "final_deonto": final_result["deontological"] if final_result else None,
                    "final_conseq": final_result["consequentialist"] if final_result else None,
                    "final_sacred": final_result["sacred_intensity"] if final_result else None,
                    "final_framework": final_result["primary_framework"] if final_result else None,
                    "final_reasoning": final_result.get("brief_reasoning", "") if final_result else "",
                }
            else:
                n_done += 1
                record = {
                    "source": pair["source"],
                    "qid": pair["qid"],
                    "condition": pair["condition"],
                    "ratio": pair["ratio"],
                    "peer_roles": pair["peer_roles"],
                    "error": False,
                    # Baseline scores
                    "base_deonto": base_result["deontological"],
                    "base_conseq": base_result["consequentialist"],
                    "base_sacred": base_result["sacred_intensity"],
                    "base_framework": base_result["primary_framework"],
                    "base_reasoning": base_result.get("brief_reasoning", ""),
                    # Final scores
                    "final_deonto": final_result["deontological"],
                    "final_conseq": final_result["consequentialist"],
                    "final_sacred": final_result["sacred_intensity"],
                    "final_framework": final_result["primary_framework"],
                    "final_reasoning": final_result.get("brief_reasoning", ""),
                }

            out.write(json.dumps(record) + "\n")
            out.flush()

    elapsed = time.time() - t_start
    print(f"\n{'='*55}")
    print(f"Done! Processed {n_done + n_errors} pairs in {elapsed/60:.1f} minutes")
    print(f"  Successful: {n_done}")
    print(f"  Errors: {n_errors}")
    print(f"  Output: {output_path}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
