#!/usr/bin/env python3
"""
Study 4 :Two-Agent Dynamic Debate on Debatable Questions

Core research question:
  In a dynamic-game debate, do LLM agents defend initial positions
  or get persuaded toward their opponent? Is belief movement sensitive to
  initial distance between them?

Pipeline structure per dilemma-rep (aligned with the design you described):

  ROUND 0 — INITIAL COMMITMENT (no debate yet)
    For each agent A1, A2 independently:
      (a) Show: question + Option A + Option B + 7-point scale
      (b) COMMITMENT PROBE: "State your position (1-7)." → capture logprobs
                             over {1,...,7} as the initial belief distribution.
      (c) ARGUMENT: "Now explain your position in 2-3 sentences."
    → gives (p_init, choice_init, argument_init) for each agent.

  ROUNDS 1..N — DEBATE WITH INTERMEDIATE COMMITMENTS
    For round r in 1..N:
      For each agent (alternating A1, A2):
        1. ARGUMENT: agent sees full shared history (both agents' past
           turns), generates its response (2-3 sentences) pushing back /
           building on the exchange.
        2. INTERMEDIATE COMMITMENT PROBE: after writing its argument,
           agent is asked "Given what you and the other have said, what
           is your position now? (1-7)". We capture the digit AND the
           logprob distribution over {1..7}. Then a 2-3 sentence
           justification follows.
    → gives (p_r, choice_r, justification_r) for each agent at each round.

Outputs (JSONL, one row = one probe event):
  - transcripts.jsonl : every turn (arguments + commitments), with full text
  - summaries.jsonl   : per-episode trajectories, distributions, movement metrics

Key schema differences vs. the original study4_free_deliberation.py:
  * Each round emits TWO records per agent: "argument" and "commitment"
  * The commitment probe always runs AFTER the agent has spoken in that round,
    so the distribution reflects post-argument belief (cleaner causal read)
  * Initial distribution is captured BEFORE the agent writes its baseline
    argument, so r=0 cleanly represents the agent's untouched prior
  * Scale text and option A/B are shown explicitly in every prompt

Resume is supported at the (qid, rep) granularity.

Usage:
  python3 study4_dynamic_debate.py \\
      --dilemmas /mnt/user-data/outputs/study4_dilemmas.csv \\
      --output-dir outputs/study4 \\
      --provider azure --model gpt-4o \\
      --n-rounds 4 --n-reps 10
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests


# ═════════════════════════════════════════════════════════════════════════════
# Constants
# ═════════════════════════════════════════════════════════════════════════════

DIGITS = ["1", "2", "3", "4", "5", "6", "7"]
RETRYABLE_STATUS = {429, 500, 502, 503, 504}


# ═════════════════════════════════════════════════════════════════════════════
# Data
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class Dilemma:
    id: str
    scenario: str
    option_a: str
    option_b: str
    scale_text: str
    variation: int


def load_dilemmas(csv_path: str) -> List[Dilemma]:
    out: List[Dilemma] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out.append(Dilemma(
                id=row["id"],
                scenario=row["scenario"],
                option_a=row.get("option_a", "") or "",
                option_b=row.get("option_b", "") or "",
                scale_text=row.get("scale_1", "") or "",
                variation=int(row.get("variation", 0) or 0),
            ))
    return out


# ═════════════════════════════════════════════════════════════════════════════
# JSONL I/O + resume support
# ═════════════════════════════════════════════════════════════════════════════

def append_jsonl(row: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def load_completed_keys(summary_path: Path) -> set:
    done: set = set()
    if not summary_path.exists():
        return done
    with open(summary_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                done.add((obj.get("qid", ""), obj.get("rep", 0)))
            except json.JSONDecodeError:
                continue
    return done


def log(msg: str, path: Path) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ═════════════════════════════════════════════════════════════════════════════
# Chat client (Azure / OpenAI / OpenRouter compatible)
# ═════════════════════════════════════════════════════════════════════════════

class ChatClient:
    def __init__(
        self,
        provider: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 250,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: int = 60,
        api_version: Optional[str] = None,
    ):
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

        if provider == "azure":
            self.endpoint = (endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT", "")).rstrip("/")
            self.api_key = api_key or os.environ.get("AZURE_OPENAI_API_KEY")
            self.api_version = api_version or os.environ.get(
                "AZURE_OPENAI_API_VERSION", "2024-02-15-preview"
            )
        elif provider == "openrouter":
            self.endpoint = (endpoint or "https://openrouter.ai/api/v1").rstrip("/")
            self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
        else:  # openai
            self.endpoint = (endpoint or "https://api.openai.com/v1").rstrip("/")
            self.api_key = api_key or os.environ.get("OPENAI_API_KEY")

    def chat(
        self,
        messages: List[Dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        logprobs: Optional[bool] = None,
        top_logprobs: Optional[int] = None,
        retries: int = 8,
        backoff_base: float = 1.6,
        backoff_max: float = 60.0,
    ) -> Dict:
        if self.provider == "azure":
            url = (
                f"{self.endpoint}/openai/deployments/{self.model}/chat/completions"
                f"?api-version={self.api_version}"
            )
            headers = {"Content-Type": "application/json", "api-key": self.api_key}
        else:
            url = f"{self.endpoint}/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            if self.provider == "openrouter":
                headers["HTTP-Referer"] = os.environ.get("OPENROUTER_REFERRER", "http://localhost")
                headers["X-Title"] = "study4_dynamic_debate"

        payload = {
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
        }
        if self.provider != "azure":
            payload["model"] = self.model
        if logprobs is not None:
            payload["logprobs"] = bool(logprobs)
        if top_logprobs is not None:
            payload["top_logprobs"] = int(top_logprobs)

        last_err: Optional[Exception] = None
        for attempt in range(retries + 1):
            try:
                r = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
                if r.status_code == 200:
                    return r.json()
                if r.status_code in RETRYABLE_STATUS:
                    last_err = RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
                    if attempt < retries:
                        sleep_s = min(backoff_max, backoff_base ** attempt) * (0.7 + 0.6 * random.random())
                        print(f"[retry] status={r.status_code} attempt={attempt+1}/{retries} sleep={sleep_s:.1f}s",
                              flush=True)
                        time.sleep(sleep_s)
                        continue
                raise RuntimeError(f"Chat failed ({r.status_code}): {r.text[:500]}")
            except (requests.exceptions.Timeout,
                    requests.exceptions.ConnectionError,
                    requests.exceptions.ChunkedEncodingError) as e:
                last_err = e
                if attempt < retries:
                    sleep_s = min(backoff_max, backoff_base ** attempt) * (0.7 + 0.6 * random.random())
                    time.sleep(sleep_s)
                    continue
                raise
        raise RuntimeError(f"Chat failed after retries: {last_err}")


# ═════════════════════════════════════════════════════════════════════════════
# Logprob extraction + commitment probe
# ═════════════════════════════════════════════════════════════════════════════

def _safe_exp(x: float) -> float:
    try:
        return math.exp(x)
    except OverflowError:
        return 0.0


def extract_digit_distribution_from_logprobs(raw: Dict) -> Dict:
    """Extract a normalized 7-way distribution over {1..7} from the first
    generated token's top_logprobs. Returns
        { logprobs: {d: lp|None}, probs: {d: p|None}, sampled_digit: '1'..'7'|None }
    """
    out_lp = {d: None for d in DIGITS}
    out_p = {d: None for d in DIGITS}

    try:
        lp_obj = raw["choices"][0].get("logprobs", None)
        if not lp_obj:
            return {"logprobs": out_lp, "probs": out_p, "sampled_digit": None}

        content_list = lp_obj.get("content", None)
        if not content_list:
            return {"logprobs": out_lp, "probs": out_p, "sampled_digit": None}

        top = content_list[0].get("top_logprobs", []) or []
        best: Dict[str, float] = {}
        for item in top:
            tok = item.get("token", "")
            lp = item.get("logprob", None)
            if lp is None or not tok:
                continue
            ch = tok.strip()[:1]
            if ch in DIGITS:
                if (ch not in best) or (lp > best[ch]):
                    best[ch] = lp
        for d in DIGITS:
            out_lp[d] = best.get(d)

        avail = [(d, out_lp[d]) for d in DIGITS if out_lp[d] is not None]
        if not avail:
            return {"logprobs": out_lp, "probs": out_p, "sampled_digit": None}

        max_lp = max(lp for _, lp in avail)
        unnorm = {d: (_safe_exp(out_lp[d] - max_lp) if out_lp[d] is not None else 0.0) for d in DIGITS}
        Z = sum(unnorm.values())
        if Z > 0:
            out_p = {d: unnorm[d] / Z for d in DIGITS}

        sampled = None
        try:
            sampled = raw["choices"][0]["message"]["content"].strip()[:1]
        except Exception:
            pass

        return {"logprobs": out_lp, "probs": out_p, "sampled_digit": sampled}
    except Exception:
        return {"logprobs": out_lp, "probs": out_p, "sampled_digit": None}


def extract_first_digit(text: str) -> Optional[int]:
    if not text:
        return None
    # Robust: first 1-7 digit anywhere in first ~10 chars (agents may add formatting)
    for ch in text.strip()[:10]:
        if ch in "1234567":
            return int(ch)
    # Fallback: first digit anywhere
    m = re.search(r"[1-7]", text)
    return int(m.group(0)) if m else None


def commitment_probe(
    client: ChatClient,
    system_prompt: str,
    context_history: List[Dict],
    probe_question: str,
) -> Dict:
    """
    Ask the agent for its current Likert position as a SINGLE digit with
    logprobs over {1..7}. Temperature=0, max_tokens=1 → the returned token
    IS the commitment; top_logprobs gives the full distribution.

    Returns: {logprobs, probs, sampled_digit} dict (from
             extract_digit_distribution_from_logprobs).
    """
    messages = [{"role": "system", "content": system_prompt}] + context_history + [
        {"role": "user", "content": probe_question}
    ]
    try:
        raw = client.chat(
            messages=messages,
            temperature=0.0,
            max_tokens=1,
            logprobs=True,
            top_logprobs=20,
        )
        return extract_digit_distribution_from_logprobs(raw)
    except Exception as e:
        print(f"[probe_failed] {e}", flush=True)
        return {"logprobs": None, "probs": None, "sampled_digit": None}


# ═════════════════════════════════════════════════════════════════════════════
# Prompts
# ═════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = (
    "You are {agent_name}, one of two participants in a structured debate about a "
    "contested question. There is no single correct answer — reasonable people "
    "disagree. Your task is to state your honest position on a 1-7 Likert scale "
    "and engage genuinely with the other participant's arguments. You may change "
    "your position if you are persuaded, or hold firm if you are not. Do not agree "
    "just to be polite, and do not disagree just to be contrary. Stay on topic and "
    "avoid meta-commentary about the debate itself."
)


def format_dilemma_block(d: Dilemma) -> str:
    """Format the debate topic as seen by both agents at every turn."""
    return (
        f"QUESTION:\n{d.scenario}\n\n"
        f"OPTION A:\n{d.option_a}\n\n"
        f"OPTION B:\n{d.option_b}\n\n"
        f"RESPONSE SCALE:\n{d.scale_text}"
    )


def format_history(turns: List[Dict]) -> str:
    """Render accumulated debate history as a readable transcript for the prompt."""
    if not turns:
        return "(No prior exchanges yet.)"
    lines = []
    for t in turns:
        if t["kind"] == "argument":
            lines.append(
                f"{t['agent']} — Round {t['round']} (position: {t['position']}):\n{t['text']}"
            )
        elif t["kind"] == "commitment":
            # Commitment turns produce a digit + short justification; surface the justification
            lines.append(
                f"{t['agent']} — Round {t['round']} commitment (position: {t['position']}):\n{t['text']}"
            )
    return "\n\n".join(lines)


# User-turn prompts
PROMPT_BASELINE_ARG = (
    "{dilemma}\n\n"
    "This is the opening round. You have not heard the other participant yet. "
    "In 2-3 sentences, state your own reasoning on this question. Do NOT begin "
    "with a digit — just write your argument."
)

PROMPT_BASELINE_COMMIT = (
    "Based on the question above and the option descriptions, what is your "
    "position on the 1-7 scale? Respond with EXACTLY ONE CHARACTER — one of "
    "1, 2, 3, 4, 5, 6, 7. Nothing else."
)

PROMPT_DEBATE_ARG = (
    "{dilemma}\n\n"
    "DEBATE SO FAR:\n{history}\n\n"
    "As {agent_name}, respond to the other participant's most recent argument. "
    "In 2-3 sentences, address their points directly, then give your own reasoning. "
    "Do NOT begin with a digit — just write your response."
)

PROMPT_INTERMEDIATE_COMMIT = (
    "Given everything you and the other participant have said, what is your "
    "position on the 1-7 scale RIGHT NOW? Respond with EXACTLY ONE CHARACTER "
    "— one of 1, 2, 3, 4, 5, 6, 7. Nothing else."
)

PROMPT_COMMIT_JUSTIFY = (
    "In 2-3 sentences, explain why you chose {digit}. Did the other participant's "
    "argument move you at all? Be honest."
)


# ═════════════════════════════════════════════════════════════════════════════
# Core simulation
# ═════════════════════════════════════════════════════════════════════════════

def simulate_debate(
    client: ChatClient,
    d: Dilemma,
    n_rounds: int,
    rep: int,
    max_tokens_arg: int = 220,
    max_tokens_just: int = 150,
) -> Tuple[List[Dict], Dict]:
    """
    Run a single two-agent debate episode.

    Returns:
      transcript: flat list of event dicts (arguments + commitments), each
                  tagged with qid/rep/round/agent/kind.
      summary   : episode-level summary with trajectories + distributions.
    """
    dilemma_block = format_dilemma_block(d)

    transcript: List[Dict] = []
    # shared_history is what BOTH agents see at every turn
    shared_history: List[Dict] = []
    # per-agent running records
    positions: Dict[str, List[Optional[int]]] = {"A1": [], "A2": []}
    distributions: Dict[str, List[Optional[Dict[str, float]]]] = {"A1": [], "A2": []}

    # ─────────────────────────────────────────────────────────────────────────
    # ROUND 0 — INDEPENDENT BASELINES
    # Each agent answers the question without seeing the other.
    # Order: (1) commitment probe → capture distribution,
    #        (2) argument generation → record.
    # Commitment is measured BEFORE the agent writes any baseline argument,
    # so it's a clean prior.
    # ─────────────────────────────────────────────────────────────────────────
    for agent in ("A1", "A2"):
        sys_prompt = SYSTEM_PROMPT.format(agent_name=agent)

        # --- Baseline commitment probe ---
        probe_ctx = [
            {"role": "user", "content": dilemma_block
                + "\n\nRead the question and the two options carefully."}
        ]
        probe_result = commitment_probe(
            client=client,
            system_prompt=sys_prompt,
            context_history=probe_ctx,
            probe_question=PROMPT_BASELINE_COMMIT,
        )
        pos = int(probe_result["sampled_digit"]) if probe_result.get("sampled_digit") in DIGITS else None
        positions[agent].append(pos)
        distributions[agent].append(probe_result.get("probs"))

        transcript.append({
            "qid": d.id, "rep": rep, "round": 0, "agent": agent, "kind": "commitment",
            "text": "", "position": pos,
            "probs": probe_result.get("probs"),
            "logprobs": probe_result.get("logprobs"),
            "sampled_digit": probe_result.get("sampled_digit"),
        })

        # --- Baseline argument ---
        arg_messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": PROMPT_BASELINE_ARG.format(dilemma=dilemma_block)},
        ]
        raw = client.chat(
            messages=arg_messages,
            temperature=client.temperature,
            max_tokens=max_tokens_arg,
        )
        arg_text = raw["choices"][0]["message"]["content"].strip()

        transcript.append({
            "qid": d.id, "rep": rep, "round": 0, "agent": agent, "kind": "argument",
            "text": arg_text, "position": pos,
        })
        shared_history.append({
            "agent": agent, "round": 0, "kind": "argument",
            "text": arg_text, "position": pos,
        })

    # ─────────────────────────────────────────────────────────────────────────
    # ROUNDS 1..N — ALTERNATING DEBATE WITH POST-ARGUMENT COMMITMENT PROBES
    # Each agent in each round:
    #   (1) writes an argument responding to the current shared history
    #   (2) commits a Likert digit with logprob distribution
    #   (3) writes a 2-3 sentence justification of that digit
    # The commitment comes AFTER the argument so the distribution reflects
    # the agent's stated belief having just heard-and-responded-to the other.
    # ─────────────────────────────────────────────────────────────────────────
    for round_num in range(1, n_rounds + 1):
        for agent in ("A1", "A2"):
            sys_prompt = SYSTEM_PROMPT.format(agent_name=agent)
            hist_text = format_history(shared_history)

            # --- (1) Argument turn ---
            arg_prompt = PROMPT_DEBATE_ARG.format(
                dilemma=dilemma_block,
                history=hist_text,
                agent_name=agent,
            )
            arg_messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": arg_prompt},
            ]
            raw = client.chat(
                messages=arg_messages,
                temperature=client.temperature,
                max_tokens=max_tokens_arg,
            )
            arg_text = raw["choices"][0]["message"]["content"].strip()

            # --- (2) Commitment probe (post-argument) ---
            # Context fed to the probe: dilemma + history + the agent's just-written argument
            probe_ctx = [
                {"role": "user", "content": arg_prompt},
                {"role": "assistant", "content": arg_text},
            ]
            probe_result = commitment_probe(
                client=client,
                system_prompt=sys_prompt,
                context_history=probe_ctx,
                probe_question=PROMPT_INTERMEDIATE_COMMIT,
            )
            digit_str = probe_result.get("sampled_digit")
            pos = int(digit_str) if digit_str in DIGITS else None
            positions[agent].append(pos)
            distributions[agent].append(probe_result.get("probs"))

            # --- (3) Justification ---
            # Feed the same probe context + the digit so the agent justifies THAT number
            just_text = ""
            if pos is not None:
                just_messages = [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": arg_prompt},
                    {"role": "assistant", "content": arg_text},
                    {"role": "user", "content": PROMPT_INTERMEDIATE_COMMIT},
                    {"role": "assistant", "content": str(pos)},
                    {"role": "user", "content": PROMPT_COMMIT_JUSTIFY.format(digit=pos)},
                ]
                try:
                    raw_j = client.chat(
                        messages=just_messages,
                        temperature=client.temperature,
                        max_tokens=max_tokens_just,
                    )
                    just_text = raw_j["choices"][0]["message"]["content"].strip()
                except Exception as e:
                    just_text = f"[justification_failed: {e}]"

            # --- Record transcript entries ---
            transcript.append({
                "qid": d.id, "rep": rep, "round": round_num, "agent": agent,
                "kind": "argument",
                "text": arg_text, "position": pos,
            })
            transcript.append({
                "qid": d.id, "rep": rep, "round": round_num, "agent": agent,
                "kind": "commitment",
                "text": just_text, "position": pos,
                "probs": probe_result.get("probs"),
                "logprobs": probe_result.get("logprobs"),
                "sampled_digit": probe_result.get("sampled_digit"),
            })

            # Update shared history (the OTHER agent will see this next)
            shared_history.append({
                "agent": agent, "round": round_num, "kind": "argument",
                "text": arg_text, "position": pos,
            })
            shared_history.append({
                "agent": agent, "round": round_num, "kind": "commitment",
                "text": just_text, "position": pos,
            })

    # ─────────────────────────────────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────────────────────────────────
    a1_base, a2_base = positions["A1"][0], positions["A2"][0]
    a1_final, a2_final = positions["A1"][-1], positions["A2"][-1]

    def safe_abs_diff(x: Optional[int], y: Optional[int]) -> Optional[int]:
        return abs(x - y) if (x is not None and y is not None) else None

    initial_distance = safe_abs_diff(a1_base, a2_base)
    final_distance = safe_abs_diff(a1_final, a2_final)

    if initial_distance is not None and final_distance is not None:
        if final_distance < initial_distance:
            pattern = "convergence"
        elif final_distance > initial_distance:
            pattern = "polarization"
        else:
            pattern = "stable"
    else:
        pattern = "unknown"

    def count_changes(traj: List[Optional[int]]) -> int:
        return sum(
            1 for i in range(1, len(traj))
            if traj[i] is not None and traj[i - 1] is not None and traj[i] != traj[i - 1]
        )

    summary = {
        "qid": d.id,
        "rep": rep,
        "n_rounds": n_rounds,
        "a1_baseline": a1_base,
        "a2_baseline": a2_base,
        "a1_final": a1_final,
        "a2_final": a2_final,
        "initial_distance": initial_distance,
        "final_distance": final_distance,
        "pattern": pattern,
        "a1_position_trajectory": positions["A1"],  # length n_rounds + 1
        "a2_position_trajectory": positions["A2"],
        "a1_distributions": distributions["A1"],
        "a2_distributions": distributions["A2"],
        "a1_n_changes": count_changes(positions["A1"]),
        "a2_n_changes": count_changes(positions["A2"]),
        "a1_moved_toward_peer": (
            None if initial_distance is None or final_distance is None
            else safe_abs_diff(a1_final, a2_base) < safe_abs_diff(a1_base, a2_base)
        ),
        "a2_moved_toward_peer": (
            None if initial_distance is None or final_distance is None
            else safe_abs_diff(a2_final, a1_base) < safe_abs_diff(a2_base, a1_base)
        ),
        "a1_switched_side": (
            a1_base is not None and a1_final is not None
            and (a1_base - 4) * (a1_final - 4) < 0
        ),
        "a2_switched_side": (
            a2_base is not None and a2_final is not None
            and (a2_base - 4) * (a2_final - 4) < 0
        ),
    }
    return transcript, summary


# ═════════════════════════════════════════════════════════════════════════════
# Main runner
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Study 4: Two-Agent Dynamic Debate")
    parser.add_argument("--dilemmas", type=str, required=True,
                        help="Path to study4_dilemmas.csv")
    parser.add_argument("--output-dir", type=str, default="outputs/study4")
    parser.add_argument("--provider", type=str, default="azure",
                        choices=["azure", "openai", "openrouter"])
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--endpoint", type=str, default=None)
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--api-version", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--n-rounds", type=int, default=4,
                        help="Debate rounds after the baseline (3-5 recommended)")
    parser.add_argument("--n-reps", type=int, default=10,
                        help="Repetitions per dilemma")
    parser.add_argument("--max-tokens-arg", type=int, default=220,
                        help="Max tokens for each argument turn")
    parser.add_argument("--max-tokens-just", type=int, default=150,
                        help="Max tokens for each commitment justification")
    parser.add_argument("--max-dilemmas", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None,
                        help="Optional seed for reproducibility of ordering")
    args = parser.parse_args()

    # --- Setup paths ---
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    transcript_path = out_dir / "transcripts.jsonl"
    summary_path = out_dir / "summaries.jsonl"
    log_path = out_dir / "progress.log"
    error_path = out_dir / "errors.jsonl"

    # --- Resume support ---
    completed = load_completed_keys(summary_path)
    log(f"Resume: {len(completed)} episodes already done.", log_path)

    # --- Load dilemmas ---
    dilemmas = load_dilemmas(args.dilemmas)
    if args.max_dilemmas:
        dilemmas = dilemmas[: args.max_dilemmas]
    log(f"Loaded {len(dilemmas)} dilemmas from {args.dilemmas}.", log_path)

    # --- Client ---
    client = ChatClient(
        provider=args.provider,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens_arg,
        endpoint=args.endpoint,
        api_key=args.api_key,
        api_version=args.api_version,
    )

    if args.seed is not None:
        random.seed(args.seed)

    total = len(dilemmas) * args.n_reps
    done = 0
    for d in dilemmas:
        for rep in range(args.n_reps):
            key = (d.id, rep)
            if key in completed:
                done += 1
                continue

            try:
                transcript, summary = simulate_debate(
                    client=client,
                    d=d,
                    n_rounds=args.n_rounds,
                    rep=rep,
                    max_tokens_arg=args.max_tokens_arg,
                    max_tokens_just=args.max_tokens_just,
                )
                for t in transcript:
                    append_jsonl(t, transcript_path)
                append_jsonl(summary, summary_path)

                done += 1
                if done % 5 == 0 or done == total:
                    log(
                        f"[{done}/{total}] qid={d.id} rep={rep} "
                        f"A1:{summary['a1_baseline']}→{summary['a1_final']} | "
                        f"A2:{summary['a2_baseline']}→{summary['a2_final']} | "
                        f"d0={summary['initial_distance']} dN={summary['final_distance']} "
                        f"pattern={summary['pattern']}",
                        log_path,
                    )
            except Exception as e:
                append_jsonl(
                    {
                        "qid": d.id, "rep": rep, "error": str(e),
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    },
                    error_path,
                )
                log(f"[ERROR] qid={d.id} rep={rep}: {e}", log_path)
                continue

    log(f"DONE. {done}/{total} episodes completed.", log_path)


if __name__ == "__main__":
    main()
