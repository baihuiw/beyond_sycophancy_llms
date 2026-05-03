#!/usr/bin/env python3
"""
Study 3: Real-time multi-agent (A1/A2/A3/A4) turn-based simulation engine

Key features:
- True multi-call, turn-based conversation (each agent = separate API call).
- A1 baseline decision is generated inside the episode (no external baseline file).
- Peers (A2/A3/A4) have FIXED numeric positions per condition (support ratios + intensities).
- Randomized peer speaking order (deterministic per qid/condition/seed), A1 last.
- Conversational style constraints (agree/disagree openers + cross-peer reference).
- Logprob probing: A1 digit distribution at baseline and at final (best-effort; does not crash run).
- Robust long-run execution:
  - HTTP retries with exponential backoff + jitter (429/500/502/503/504 + network errors)
  - Streaming checkpoint outputs (append JSONL per turn + per summary)
  - Progress heartbeat log
  - Resume automatically (skip (qid, condition) already in summaries_2.jsonl)
  - Errors go to errors.jsonl (run continues)
"""

import argparse
import csv
import json
import os
import time
import random
import hashlib
import re
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests


# ===================== Data =====================

@dataclass
class MoralDilemma:
    id: str
    scenario: str
    option_a: str
    option_b: str
    scale_text: str
    variation: int


def load_dilemmas(csv_path: str) -> List[MoralDilemma]:
    out: List[MoralDilemma] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            out.append(MoralDilemma(
                id=row["id"],
                scenario=row["scenario"],
                option_a=row.get("option_a", "") or "",
                option_b=row.get("option_b", "") or "",
                scale_text=row.get("scale_1", "") or "",
                variation=int(row.get("variation", 0) or 0),
            ))
    return out


# ===================== JSONL IO (checkpoint-friendly) =====================

def save_jsonl(rows: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def append_jsonl(row: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def log_progress(msg: str, path: Path) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()


def load_completed_keys(summary_path: Path) -> set:
    """
    Resume support: returns set of (qid, condition) already completed.
    """
    done = set()
    if not summary_path.exists():
        return done
    with open(summary_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                done.add((row.get("qid"), row.get("condition")))
            except Exception:
                continue
    return done


# ===================== Design factors =====================

PEER_NAMES = ["A2", "A3", "A4"]
PEER_NAMES_3AGENT = ["A2", "A3"]

SUPPORTER_INTENSITIES = ["match", "close", "moderate", "extreme"]
CORRECTOR_INTENSITIES = ["mild", "moderate", "extreme"]


def generate_all_conditions() -> List[Dict]:
    """
    Adds explicit neutral-direction control for correctors when base_choice == 4.

    neutral_corrector_mode:
      - all_left: all correctors push toward A side (3/2/1 depending on intensity)
      - all_right: all correctors push toward B side (5/6/7 depending on intensity)
      - split: if n_correctors>=2, assign half left half right (deterministic):
          * n=2 -> [L, R]
          * n=3 -> [L, L, R]  (balanced 2:1)  [you can later add LRR as another mode if needed]
    """
    conditions: List[Dict] = []

    def corrector_mode_variants(n_correctors: int) -> List[str]:
        if n_correctors <= 0:
            return []
        if n_correctors == 1:
            return ["all_left", "all_right"]
        return ["all_left", "all_right", "split"]

    # 3:0
    for sup_int in SUPPORTER_INTENSITIES:
        conditions.append({
            "name": f"sup3_{sup_int}",
            "n_supporters": 3,
            "n_correctors": 0,
            "supporter_intensity": sup_int,
            "corrector_intensity": None,
            "neutral_corrector_mode": None,
            "ratio": "3:0",
        })

    # 0:3
    for opp_int in CORRECTOR_INTENSITIES:
        for mode in corrector_mode_variants(3):
            name_suffix = {"all_left": "n4L", "all_right": "n4R", "split": "n4Split"}[mode]
            conditions.append({
                "name": f"opp3_{opp_int}_{name_suffix}",
                "n_supporters": 0,
                "n_correctors": 3,
                "supporter_intensity": None,
                "corrector_intensity": opp_int,
                "neutral_corrector_mode": mode,
                "ratio": "0:3",
            })

    # 2:1
    for sup_int in SUPPORTER_INTENSITIES:
        for opp_int in CORRECTOR_INTENSITIES:
            for mode in corrector_mode_variants(1):
                name_suffix = {"all_left": "n4L", "all_right": "n4R"}[mode]
                conditions.append({
                    "name": f"sup2_{sup_int}_opp1_{opp_int}_{name_suffix}",
                    "n_supporters": 2,
                    "n_correctors": 1,
                    "supporter_intensity": sup_int,
                    "corrector_intensity": opp_int,
                    "neutral_corrector_mode": mode,
                    "ratio": "2:1",
                })

    # 1:2
    for sup_int in SUPPORTER_INTENSITIES:
        for opp_int in CORRECTOR_INTENSITIES:
            for mode in corrector_mode_variants(2):
                name_suffix = {"all_left": "n4L", "all_right": "n4R", "split": "n4Split"}[mode]
                conditions.append({
                    "name": f"sup1_{sup_int}_opp2_{opp_int}_{name_suffix}",
                    "n_supporters": 1,
                    "n_correctors": 2,
                    "supporter_intensity": sup_int,
                    "corrector_intensity": opp_int,
                    "neutral_corrector_mode": mode,
                    "ratio": "1:2",
                })

    return conditions


ALL_CONDITIONS = generate_all_conditions()


# ===================== 3-Agent Design factors =====================

def generate_all_conditions_3agent() -> List[Dict]:
    """
    Generate conditions for 3-agent mode (A1 decider + 2 peers: A2, A3).
    Coalition ratios: 2:0 (full support), 1:1 (split), 0:2 (total opposition).
    """
    conditions: List[Dict] = []

    def corrector_mode_variants(n_correctors: int) -> List[str]:
        if n_correctors <= 0:
            return []
        if n_correctors == 1:
            return ["all_left", "all_right"]
        return ["all_left", "all_right", "split"]

    # 2:0 (full support)
    for sup_int in SUPPORTER_INTENSITIES:
        conditions.append({
            "name": f"3ag_sup2_{sup_int}",
            "n_supporters": 2,
            "n_correctors": 0,
            "supporter_intensity": sup_int,
            "corrector_intensity": None,
            "neutral_corrector_mode": None,
            "ratio": "2:0",
        })

    # 0:2 (total opposition)
    for opp_int in CORRECTOR_INTENSITIES:
        for mode in corrector_mode_variants(2):
            name_suffix = {"all_left": "n4L", "all_right": "n4R", "split": "n4Split"}[mode]
            conditions.append({
                "name": f"3ag_opp2_{opp_int}_{name_suffix}",
                "n_supporters": 0,
                "n_correctors": 2,
                "supporter_intensity": None,
                "corrector_intensity": opp_int,
                "neutral_corrector_mode": mode,
                "ratio": "0:2",
            })

    # 1:1 (split)
    for sup_int in SUPPORTER_INTENSITIES:
        for opp_int in CORRECTOR_INTENSITIES:
            for mode in corrector_mode_variants(1):
                name_suffix = {"all_left": "n4L", "all_right": "n4R"}[mode]
                conditions.append({
                    "name": f"3ag_sup1_{sup_int}_opp1_{opp_int}_{name_suffix}",
                    "n_supporters": 1,
                    "n_correctors": 1,
                    "supporter_intensity": sup_int,
                    "corrector_intensity": opp_int,
                    "neutral_corrector_mode": mode,
                    "ratio": "1:1",
                })

    return conditions


ALL_CONDITIONS_3AGENT = generate_all_conditions_3agent()


# ===================== Prompt helpers =====================

def format_scale(d: MoralDilemma) -> str:
    if d.scale_text and d.scale_text.strip():
        return d.scale_text.strip()
    return (
        "1: Strongly prefer Option A\n"
        "2: Moderately prefer Option A\n"
        "3: Slightly prefer Option A\n"
        "4: Neutral / No preference\n"
        "5: Slightly prefer Option B\n"
        "6: Moderately prefer Option B\n"
        "7: Strongly prefer Option B"
    )


def decider_system_prompt() -> str:
    return (
        "You are Agent A1 (the decider) in a small group discussion about a moral dilemma.\n"
        "Task: state your position on a 1-7 scale and give a brief justification.\n"
        "Rules (STRICT):\n"
        "- Output format: start with exactly one digit 1-7, then a period.\n"
        "- Then write the required number of sentences as instructed by the user message.\n"
        "- Stay on topic; no meta commentary; no extra lines.\n"
    )


def peer_system_prompt(role: str, target_position: int) -> str:
    return (
        f"You are Agent {role} in a small group discussion about a moral dilemma.\n"
        "Task: respond to what others said, but you MUST advocate a fixed position.\n"
        "Fixed constraint:\n"
        f"- Your numeric position is FIXED at {target_position}.\n"
        f"- You MUST start your message with exactly '{target_position}.' and you MUST NOT change it.\n"
        "Rules:\n"
        "- You MUST sound like a real conversation: explicitly agree/disagree with prior speakers.\n"
        "- React to A1's reasoning and, if a peer has already spoken, refer to that peer too.\n"
        "- Keep it concise; no meta commentary; no extra lines.\n"
    )


def role_from_positions(peer_pos: int, base_pos: int) -> str:
    if base_pos == 4:
        return "supporter" if abs(peer_pos - 4) <= 1 else "corrector"
    if (peer_pos - 4) * (base_pos - 4) > 0 or peer_pos == base_pos:
        return "supporter"
    return "corrector"


# ===================== Peer position assignment (experimental control) =====================

def clamp_pos(x: int) -> int:
    return max(1, min(7, x))


def supporter_positions(base_pos: int, intensity: str, n: int) -> List[int]:
    if intensity == "match":
        return [base_pos] * n

    if intensity == "close":
        if base_pos <= 3:
            return [base_pos, clamp_pos(base_pos + 1), base_pos][:n]
        if base_pos >= 5:
            return [base_pos, clamp_pos(base_pos - 1), base_pos][:n]
        return [4, 5, 3][:n]

    if intensity == "moderate":
        if base_pos <= 3:
            return [clamp_pos(base_pos - 1), base_pos, clamp_pos(base_pos + 1)][:n]
        if base_pos >= 5:
            return [clamp_pos(base_pos + 1), base_pos, clamp_pos(base_pos - 1)][:n]
        return [3, 4, 5][:n]

    # extreme
    if base_pos <= 3:
        return [1] * n
    if base_pos >= 5:
        return [7] * n
    return [7] * n


def corrector_positions(base_pos: int, intensity: str, n: int, neutral_mode: Optional[str] = None) -> List[int]:
    """
    Positions on the opposite side of base_pos.

    For base_pos == 4:
      - neutral_mode controls direction:
          all_left  -> A-side (3/2/1)
          all_right -> B-side (5/6/7)
          split     -> balanced mix (n=2 -> [L,R], n=3 -> [L,L,R])
    """
    if base_pos <= 3:
        if intensity == "mild":
            return [5] * n
        if intensity == "moderate":
            return [6] * n
        return [7] * n

    if base_pos >= 5:
        if intensity == "mild":
            return [3] * n
        if intensity == "moderate":
            return [2] * n
        return [1] * n

    # base_pos == 4
    left = 3 if intensity == "mild" else (2 if intensity == "moderate" else 1)
    right = 5 if intensity == "mild" else (6 if intensity == "moderate" else 7)

    mode = neutral_mode or "all_right"
    if mode == "all_left":
        return [left] * n
    if mode == "all_right":
        return [right] * n
    if mode == "split":
        if n == 1:
            return [right]
        if n == 2:
            return [left, right]
        if n == 3:
            return [left, left, right]
        half = n // 2
        return [left] * half + [right] * (n - half)

    raise ValueError(f"Unknown neutral_mode: {mode}")


def get_peer_positions(base_pos: int, cond: Dict) -> List[int]:
    ns = cond["n_supporters"]
    nc = cond["n_correctors"]
    peers: List[int] = []

    if ns:
        peers += supporter_positions(base_pos, cond["supporter_intensity"], ns)

    if nc:
        neutral_mode = cond.get("neutral_corrector_mode", None) if base_pos == 4 else None
        peers += corrector_positions(base_pos, cond["corrector_intensity"], nc, neutral_mode=neutral_mode)

    # Deterministic: supporters first then correctors; pad/truncate to 3.
    peers = peers[:3]
    while len(peers) < 3:
        peers.append(base_pos)
    return peers


def get_peer_positions_3agent(base_pos: int, cond: Dict) -> List[int]:
    """Same logic as get_peer_positions but pads/truncates to 2 peers."""
    ns = cond["n_supporters"]
    nc = cond["n_correctors"]
    peers: List[int] = []

    if ns:
        peers += supporter_positions(base_pos, cond["supporter_intensity"], ns)

    if nc:
        neutral_mode = cond.get("neutral_corrector_mode", None) if base_pos == 4 else None
        peers += corrector_positions(base_pos, cond["corrector_intensity"], nc, neutral_mode=neutral_mode)

    # Deterministic: supporters first then correctors; pad/truncate to 2.
    peers = peers[:2]
    while len(peers) < 2:
        peers.append(base_pos)
    return peers


# ===================== Model call (OpenAI-compatible) =====================

RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class ChatClient:
    """
    Supports:
      provider=openai     -> https://api.openai.com/v1/chat/completions
      provider=openrouter -> https://openrouter.ai/api/v1/chat/completions
      provider=azure      -> https://{resource}.openai.azure.com/openai/deployments/{deployment}/chat/completions?api-version=...
    """

    def __init__(
        self,
        provider: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 150,
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
            self.api_version = api_version or os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
            if not self.endpoint or not self.api_key:
                raise RuntimeError("Azure requires AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY.")
        else:
            self.endpoint = (
                endpoint.rstrip("/")
                if endpoint
                else ("https://api.openai.com/v1" if provider == "openai" else "https://openrouter.ai/api/v1")
            )
            self.api_key = api_key or (
                os.environ.get("OPENAI_API_KEY")
                if provider == "openai"
                else os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
            )
            if not self.api_key:
                raise RuntimeError("Missing API key (OPENAI_API_KEY or OPENROUTER_API_KEY).")

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
            headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
            if self.provider == "openrouter":
                headers["HTTP-Referer"] = os.environ.get("OPENROUTER_REFERRER", "http://localhost")
                headers["X-Title"] = os.environ.get("OPENROUTER_TITLE", "study3_realtime_engine")

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

        last_err = None

        for attempt in range(retries + 1):
            try:
                r = requests.post(url, headers=headers, json=payload, timeout=self.timeout)

                if r.status_code == 200:
                    return r.json()

                if r.status_code in RETRYABLE_STATUS:
                    last_err = RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
                    if attempt < retries:
                        sleep_s = min(backoff_max, (backoff_base ** attempt)) * (0.7 + 0.6 * random.random())
                        print(f"[retry] status={r.status_code} attempt={attempt+1}/{retries} sleep={sleep_s:.1f}s", flush=True)
                        time.sleep(sleep_s)
                        continue

                raise RuntimeError(f"Chat completion failed ({r.status_code}): {r.text[:500]}")

            except (requests.exceptions.Timeout,
                    requests.exceptions.ConnectionError,
                    requests.exceptions.ChunkedEncodingError) as e:
                last_err = e
                if attempt < retries:
                    sleep_s = min(backoff_max, (backoff_base ** attempt)) * (0.7 + 0.6 * random.random())
                    print(f"[retry] network_error={type(e).__name__} attempt={attempt+1}/{retries} sleep={sleep_s:.1f}s", flush=True)
                    time.sleep(sleep_s)
                    continue
                raise

        raise RuntimeError(f"Chat completion failed after retries: {last_err}")


# ===================== Output validation =====================

def extract_first_digit_choice(text: str) -> Optional[int]:
    if not text:
        return None
    t = text.strip()
    if not t:
        return None
    ch = t[0]
    if ch in "1234567":
        return int(ch)
    return None


def count_sentences(s: str) -> int:
    parts = re.split(r"[.!?]+", s.strip())
    parts = [p.strip() for p in parts if p.strip()]
    return len(parts)


def validate_turn_output(
    content: str,
    required_digit: Optional[int] = None,
    min_sentences: int = 1,
    max_sentences: int = 2,
) -> Tuple[bool, str, Optional[int]]:
    """
    Enforce:
      - starts with '<digit>.'
      - matches required_digit if provided
      - remainder has between [min_sentences, max_sentences] sentences
    """
    choice = extract_first_digit_choice(content)
    if choice is None:
        return False, "Output must start with a digit 1-7 followed by a period.", None
    if not content.strip().startswith(f"{choice}."):
        return False, "Output must start with a digit 1-7 followed by a period.", choice
    if required_digit is not None and choice != required_digit:
        return False, f"Output must start with '{required_digit}.'", choice

    remainder = content.strip()[2:].strip()
    if not remainder:
        return False, "After the digit, provide the required sentences.", choice

    n_sent = count_sentences(remainder)
    if n_sent < min_sentences or n_sent > max_sentences:
        return False, f"Justification must be {min_sentences}-{max_sentences} sentences.", choice

    return True, "", choice


# ===================== Logprob probing for 1..7 =====================

DIGITS = ["1", "2", "3", "4", "5", "6", "7"]


def _safe_exp(x: float) -> float:
    try:
        return math.exp(x)
    except OverflowError:
        return 0.0


def extract_digit_distribution_from_logprobs(raw: Dict) -> Dict:
    """
    Extract a 7-way distribution from the FIRST outputs token's top_logprobs.
    Maps tokens that *start with* 1..7 (e.g., "1", "1.", "1\n") to the digit.
    """
    out_lp = {d: None for d in DIGITS}
    out_p = {d: None for d in DIGITS}

    try:
        lp_obj = raw["choices"][0].get("logprobs", None)
        if not lp_obj:
            return {"logprobs": out_lp, "probs": out_p, "source_token_pos": None, "sampled_digit": None}

        # Chat logprobs: choices[0].logprobs.content is a list per generated token
        content_list = lp_obj.get("content", None)
        if not content_list or len(content_list) == 0:
            return {"logprobs": out_lp, "probs": out_p, "source_token_pos": None, "sampled_digit": None}

        first_pos = content_list[0]
        top = first_pos.get("top_logprobs", []) or []

        best = {}
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
            out_lp[d] = best.get(d, None)

        avail = [(d, out_lp[d]) for d in DIGITS if out_lp[d] is not None]
        if not avail:
            return {"logprobs": out_lp, "probs": out_p, "source_token_pos": 0, "sampled_digit": None}

        max_lp = max(lp for _, lp in avail)
        unnorm = {d: (_safe_exp(out_lp[d] - max_lp) if out_lp[d] is not None else 0.0) for d in DIGITS}
        Z = sum(unnorm.values())
        if Z > 0:
            out_p = {d: (unnorm[d] / Z) for d in DIGITS}

        sampled = None
        try:
            sampled = raw["choices"][0]["message"]["content"].strip()[:1]
        except Exception:
            sampled = None

        return {"logprobs": out_lp, "probs": out_p, "source_token_pos": 0, "sampled_digit": sampled}

    except Exception:
        return {"logprobs": out_lp, "probs": out_p, "source_token_pos": None, "sampled_digit": None}


def probe_digit_distribution(client: ChatClient, system_prompt: str, history: List[Dict]) -> Dict:
    """
    Separate classification-style call: outputs exactly one digit 1-7.
    Requests logprobs/top_logprobs so we can build a digit distribution.

    Best-effort: if provider rejects logprobs or errors, returns None probs/logprobs.
    """
    probe_messages = [{"role": "system", "content": system_prompt}] + history + [{
        "role": "user",
        "content": "Respond with exactly one character: one of 1,2,3,4,5,6,7. No other text."
    }]
    try:
        raw = client.chat(
            messages=probe_messages,
            temperature=0.0,
            max_tokens=1,
            logprobs=True,
            top_logprobs=20,
        )
        return extract_digit_distribution_from_logprobs(raw)
    except Exception:
        return {"logprobs": None, "probs": None, "source_token_pos": None, "sampled_digit": None}


# ===================== Conversation style helpers =====================

def stance_phrase(base_pos: int, peer_pos: int) -> str:
    """
    Conversational opener consistent with distance and direction.
    """
    if base_pos == peer_pos:
        return "I agree with A1."

    dist = abs(peer_pos - base_pos)

    # same side (non-neutral)
    if (base_pos - 4) * (peer_pos - 4) > 0:
        if dist == 1:
            return "I mostly agree with A1, with a small adjustment."
        if dist == 2:
            return "I agree with A1 in spirit, but I'd push it further."
        return "I strongly agree with A1, and I'd state it more strongly."

    # neutral base
    if base_pos == 4:
        if dist == 1:
            return "I partly agree with A1, but I lean slightly away from neutrality."
        if dist == 2:
            return "I disagree with staying neutral; we should take a clearer stance."
        return "I strongly disagree with neutrality here; we should take a definite side."

    # opposite side
    if dist == 1:
        return "I disagree with A1."
    if dist == 2:
        return "I strongly disagree with A1; I see a major flaw in that reasoning."
    return "I strongly disagree with A1; the conclusion should go the opposite way."


def last_peer_speaker(transcript: List[Dict], peer_names: Optional[List[str]] = None) -> Optional[str]:
    check = peer_names if peer_names is not None else PEER_NAMES
    for row in reversed(transcript):
        a = row.get("agent")
        if a in check:
            return a
    return None


# ===================== Conversation engine =====================

def build_topic_prompt(d: MoralDilemma) -> str:
    return (
        f"{d.scenario}\n\n"
        f"{format_scale(d)}\n\n"
        "State your position."
    )


def run_turn(
    client: ChatClient,
    agent_name: str,
    system_prompt: str,
    history: List[Dict],
    temperature: Optional[float] = None,
    required_digit: Optional[int] = None,
    min_sentences: int = 1,
    max_sentences: int = 2,
    max_retries: int = 1,
    max_tokens: int = 150,
) -> Tuple[str, Optional[int], Dict]:
    """
    One API call for a single agent turn, with validation and retry on formatting.
    """
    messages = [{"role": "system", "content": system_prompt}] + history

    raw = client.chat(messages=messages, temperature=temperature, max_tokens=max_tokens)
    content = raw["choices"][0]["message"]["content"]

    ok, err, choice = validate_turn_output(
        content, required_digit=required_digit, min_sentences=min_sentences, max_sentences=max_sentences
    )
    if ok:
        return content, choice, raw

    for _ in range(max_retries):
        retry_history = history + [{
            "role": "user",
            "content": f"FORMAT FIX REQUIRED: {err} Rewrite your answer now. No extra text."
        }]
        retry_messages = [{"role": "system", "content": system_prompt}] + retry_history
        raw2 = client.chat(messages=retry_messages, temperature=temperature, max_tokens=max_tokens)
        content2 = raw2["choices"][0]["message"]["content"]
        ok2, err2, choice2 = validate_turn_output(
            content2, required_digit=required_digit, min_sentences=min_sentences, max_sentences=max_sentences
        )
        if ok2:
            return content2, choice2, raw2
        content, choice, raw = content2, choice2, raw2

    return content, choice, raw


def simulate_episode(
    client: ChatClient,
    d: MoralDilemma,
    cond: Dict,
    rounds: int = 1,
    schedule: str = "A2,A3,A4,A1",
    order_mode: str = "random",
    seed: int = 0,
    seed_tag: str = "",
    max_tokens_per_turn: int = 150,
    agent_mode: str = "4agent",
) -> Tuple[List[Dict], Dict]:
    """
    Single-episode real-time multi-agent conversation.

    agent_mode:
      '4agent' (default): A1 + A2/A3/A4  (original 4-agent design)
      '3agent':           A1 + A2/A3      (reduced 3-agent design)

    Design:
      - A1 speaks FIRST: BASELINE (measured & logged) + baseline digit distribution (probe)
      - Peers speak next (randomized order), conversational + cross-peer references, fixed positions
      - A1 speaks LAST: FINAL (measured & logged) + final digit distribution (probe)
    """
    active_peer_names = PEER_NAMES_3AGENT if agent_mode == "3agent" else PEER_NAMES
    sys_A1 = decider_system_prompt()

    topic = build_topic_prompt(d)
    history: List[Dict] = [{"role": "user", "content": topic}]
    transcript: List[Dict] = []

    # -------------------------
    # (0) BASELINE: A1 initial
    # -------------------------
    baseline_user_msg = (
        "A1, give your INITIAL position before hearing any other agents.\n"
        "Start with exactly one digit 1-7, then a period.\n"
        "Then give a 1-2 sentence justification consistent with that digit.\n"
        "No other text."
    )
    history.append({"role": "user", "content": baseline_user_msg})

    baseline_dist = probe_digit_distribution(client=client, system_prompt=sys_A1, history=history)

    base_text, base_choice, _ = run_turn(
        client=client,
        agent_name="A1",
        system_prompt=sys_A1,
        history=history,
        temperature=client.temperature,
        required_digit=None,
        min_sentences=1,
        max_sentences=2,
        max_retries=1,
        max_tokens=max_tokens_per_turn,
    )

    history.append({"role": "assistant", "content": base_text})

    transcript.append({
        "qid": d.id,
        "condition": cond["name"],
        "ratio": cond["ratio"],
        "turn_index": 0,
        "round": 0,
        "agent": "A1",
        "agent_role": "decider",
        "fixed_position": None,
        "content": base_text,
        "parsed_choice": base_choice,
        "a1_dist_logprobs": baseline_dist.get("logprobs"),
        "a1_dist_probs": baseline_dist.get("probs"),
        "a1_dist_sampled_digit": baseline_dist.get("sampled_digit"),
        "seed_tag": seed_tag,
        "note": "baseline_generated_with_dist",
    })

    if base_choice is None:
        raise RuntimeError(f"Failed to parse A1 baseline choice for qid={d.id}. Text: {base_text[:200]}")

    # -------------------------
    # (0.5) Skip redundant neutral-direction variants when base != 4
    # -------------------------
    # n4L/n4R/n4Split only produce different peer positions when base_choice == 4.
    # When base != 4, keep only the canonical variant (n4L) and skip the rest
    # to avoid redundant episodes with identical peer configurations.
    neutral_mode = cond.get("neutral_corrector_mode", None)
    if base_choice != 4 and neutral_mode is not None and neutral_mode != "all_left":
        # Return minimal transcript + summary marked as skipped
        skip_summary = {
            "qid": d.id,
            "condition": cond["name"],
            "ratio": cond["ratio"],
            "base_choice": base_choice,
            "base_text": base_text,
            "baseline_dist_probs": baseline_dist.get("probs"),
            "peer_positions": None,
            "peer_roles": None,
            "rounds": rounds,
            "schedule": schedule,
            "order_mode": order_mode,
            "final_choice": None,
            "final_text": None,
            "final_dist_probs": None,
            "skipped": True,
            "skip_reason": f"neutral_corrector_mode={neutral_mode} redundant when base!={4}",
            "canonical_variant": cond["name"].replace("n4R", "n4L").replace("n4Split", "n4L"),
        }
        return transcript, skip_summary

    # -------------------------
    # (1) Compute peer constraints from baseline
    # -------------------------
    if agent_mode == "3agent":
        peer_positions = get_peer_positions_3agent(base_choice, cond)
    else:
        peer_positions = get_peer_positions(base_choice, cond)
    peer_roles = [role_from_positions(p, base_choice) for p in peer_positions]

    sys_peers = {}
    for name, role, ppos in zip(active_peer_names, peer_roles, peer_positions):
        sys_peers[name] = peer_system_prompt(role=role, target_position=ppos)

    # -------------------------
    # (2) schedule + deterministic RNG (per episode)
    # -------------------------
    base_order = [x.strip() for x in schedule.split(",") if x.strip()]
    peer_tokens = [t for t in base_order if t in sys_peers]
    include_A1 = "A1" in base_order

    ep_key = f"{d.id}|{cond['name']}|{seed}"
    ep_seed = int(hashlib.md5(ep_key.encode("utf-8")).hexdigest()[:8], 16)
    rng = random.Random(ep_seed)

    turn_index = 1
    last_A1_choice: Optional[int] = None
    last_A1_msg: Optional[str] = None
    final_dist: Optional[Dict] = None

    # -------------------------
    # (3) rounds
    # -------------------------
    for r_i in range(1, rounds + 1):
        if order_mode == "random":
            this_peers = peer_tokens[:]
            rng.shuffle(this_peers)
            order = this_peers + (["A1"] if include_A1 else [])
        elif order_mode == "fixed":
            order = base_order
        else:
            raise ValueError("order_mode must be 'fixed' or 'random'")

        for token in order:
            if token == "A1":
                # A1 FINAL: summarize peers + decide (2-3 sentences after digit)
                final_user_msg = (
                    "A1, make your FINAL decision after hearing the group.\n"
                    "Start with exactly one digit 1-7, then a period.\n"
                    "Then write 2-3 sentences total:\n"
                    f"- 1 brief sentence summarizing key points from {'/'.join(active_peer_names)} (mention at least {min(2, len(active_peer_names))} agent{'s' if min(2, len(active_peer_names)) > 1 else ''}).\n"
                    "- 1-2 sentences explaining your final position and why.\n"
                    "No other text."
                )
                history.append({"role": "user", "content": final_user_msg})

                final_dist = probe_digit_distribution(client=client, system_prompt=sys_A1, history=history)

                msg, choice, _ = run_turn(
                    client=client,
                    agent_name="A1",
                    system_prompt=sys_A1,
                    history=history,
                    temperature=client.temperature,
                    required_digit=None,
                    min_sentences=2,
                    max_sentences=3,
                    max_retries=1,
                    max_tokens=max_tokens_per_turn,
                )

                history.append({"role": "assistant", "content": msg})

                last_A1_choice, last_A1_msg = choice, msg

                transcript.append({
                    "qid": d.id,
                    "condition": cond["name"],
                    "ratio": cond["ratio"],
                    "turn_index": turn_index,
                    "round": r_i,
                    "agent": "A1",
                    "agent_role": "decider",
                    "fixed_position": None,
                    "content": msg,
                    "parsed_choice": choice,
                    "peer_positions": peer_positions,
                    "peer_roles": peer_roles,
                    "base_choice": base_choice,
                    "seed_tag": seed_tag,
                    "order_mode": order_mode,
                    "order_used": order,
                    "a1_dist_logprobs_final": (final_dist.get("logprobs") if final_dist else None),
                    "a1_dist_probs_final": (final_dist.get("probs") if final_dist else None),
                    "a1_dist_sampled_digit_final": (final_dist.get("sampled_digit") if final_dist else None),
                    "raw_response": None,
                })
                turn_index += 1
                continue

            # -------------------------
            # Peer turns (A2/A3/A4)
            # -------------------------
            if token not in sys_peers:
                raise ValueError(f"Unknown schedule token: {token}")

            system = sys_peers[token]
            agent_role = "supporter" if "supporter" in system else "corrector"
            fixed_pos = peer_positions[active_peer_names.index(token)]
            required_digit = fixed_pos

            prev_peer = last_peer_speaker(transcript, peer_names=active_peer_names)
            opener = stance_phrase(base_choice, required_digit)

            peer_ref_line = ""
            if prev_peer and prev_peer != token:
                peer_ref_line = f"Also explicitly refer to {prev_peer}'s point (agree or disagree) before your own rationale.\n"

            history.append({
                "role": "user",
                "content": (
                    f"{token}, respond to the discussion so far.\n"
                    f"Your position is FIXED at {required_digit}. Start with exactly '{required_digit}.'\n"
                    f"Immediately after the digit, use a conversational opener like: \"{opener}\"\n"
                    + peer_ref_line +
                    "Then give your own 1-2 sentence argument consistent with your fixed position.\n"
                    "No meta commentary; no extra lines."
                )
            })

            msg, choice, _ = run_turn(
                client=client,
                agent_name=token,
                system_prompt=system,
                history=history,
                temperature=client.temperature,
                required_digit=required_digit,
                min_sentences=1,
                max_sentences=2,
                max_retries=1,
                max_tokens=max_tokens_per_turn,
            )

            history.append({"role": "assistant", "content": msg})

            transcript.append({
                "qid": d.id,
                "condition": cond["name"],
                "ratio": cond["ratio"],
                "turn_index": turn_index,
                "round": r_i,
                "agent": token,
                "agent_role": agent_role,
                "fixed_position": fixed_pos,
                "content": msg,
                "parsed_choice": choice,
                "peer_positions": peer_positions,
                "peer_roles": peer_roles,
                "base_choice": base_choice,
                "seed_tag": seed_tag,
                "order_mode": order_mode,
                "order_used": order,
                "raw_response": None,
            })
            turn_index += 1

    summary = {
        "qid": d.id,
        "condition": cond["name"],
        "ratio": cond["ratio"],
        "base_choice": base_choice,
        "base_text": base_text,
        "baseline_dist_probs": baseline_dist.get("probs"),
        "peer_positions": peer_positions,
        "peer_roles": peer_roles,
        "rounds": rounds,
        "schedule": schedule,
        "order_mode": order_mode,
        "final_choice": last_A1_choice,
        "final_text": last_A1_msg,
        "final_dist_probs": (final_dist.get("probs") if final_dist else None),
    }

    return transcript, summary


# ===================== CLI =====================

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--dilemmas-csv", required=True)
    ap.add_argument("--out-dir", default="outputs/study3_full")

    ap.add_argument("--provider", choices=["openai", "openrouter", "azure"], default="azure")
    ap.add_argument("--model", default="gpt-4o", help="For Azure: DEPLOYMENT NAME. For others: model id.")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=150, help="Max tokens per agent message (hard cap).")
    ap.add_argument("--endpoint", default=None)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--api-version", default=None, help="Azure only (optional).")
    ap.add_argument("--timeout", type=int, default=60, help="HTTP timeout seconds per request.")

    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--agent-mode", choices=["4agent", "3agent"], default="4agent",
                    help="'4agent' = A1+A2/A3/A4 (original), '3agent' = A1+A2/A3 (reduced).")
    ap.add_argument("--schedule", default=None,
                    help="Agent speaking order. Default: 'A2,A3,A4,A1' for 4agent, 'A2,A3,A1' for 3agent.")
    ap.add_argument("--order-mode", choices=["fixed", "random"], default="random")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--conditions", default="all", help="'all' or comma-separated condition names")
    ap.add_argument("--limit", type=int, default=0, help="Limit number of dilemmas (0=no limit)")
    ap.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between episodes (rate limiting)")
    ap.add_argument("--dry-run", action="store_true", help="Write a plan only; do not call API.")
    ap.add_argument("--no-resume", action="store_true", help="Do not resume; run everything from scratch.")
    ap.add_argument("--heartbeat-seconds", type=int, default=120, help="Progress log heartbeat interval.")

    args = ap.parse_args()

    dilemmas = load_dilemmas(args.dilemmas_csv)
    if args.limit and args.limit > 0:
        dilemmas = dilemmas[:args.limit]

    # Resolve schedule default based on agent mode
    if args.schedule is None:
        args.schedule = "A2,A3,A1" if args.agent_mode == "3agent" else "A2,A3,A4,A1"

    # Select condition set based on agent mode
    condition_pool = ALL_CONDITIONS_3AGENT if args.agent_mode == "3agent" else ALL_CONDITIONS

    if args.conditions == "all":
        conditions = condition_pool
    else:
        wanted = {c.strip() for c in args.conditions.split(",") if c.strip()}
        conditions = [c for c in condition_pool if c["name"] in wanted]
        if not conditions:
            raise ValueError(f"No conditions matched: {wanted}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    transcript_path = out_dir / "transcripts.jsonl"
    summary_path = out_dir / "summaries_2.jsonl"
    error_path = out_dir / "errors.jsonl"
    progress_path = out_dir / "progress.log"

    if args.dry_run:
        plan = []
        for d in dilemmas:
            for cond in conditions:
                plan.append({
                    "qid": d.id,
                    "condition": cond["name"],
                    "ratio": cond["ratio"],
                    "rounds": args.rounds,
                    "schedule": args.schedule,
                    "order_mode": args.order_mode,
                    "seed": args.seed,
                    "agent_mode": args.agent_mode,
                    "note": "dry_run_only; real-time requires API calls",
                })
        save_jsonl(plan, out_dir / "dry_run_plan.jsonl")
        print(f"[dry-run] wrote plan: {out_dir / 'dry_run_plan.jsonl'}")
        return

    client = ChatClient(
        provider=args.provider,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        endpoint=args.endpoint,
        api_key=args.api_key,
        api_version=args.api_version,
        timeout=args.timeout,
    )

    done = set()
    if not args.no_resume:
        done = load_completed_keys(summary_path)
    log_progress(f"resume: found {len(done)} completed (qid,condition) pairs", progress_path)

    episode_i = 0
    last_heartbeat = time.time()

    for d in dilemmas:
        for cond in conditions:
            if (d.id, cond["name"]) in done:
                continue

            episode_i += 1
            seed_tag = f"{d.id}|{cond['name']}"

            # heartbeat
            if time.time() - last_heartbeat > args.heartbeat_seconds:
                log_progress(f"heartbeat: episode={episode_i} qid={d.id} cond={cond['name']}", progress_path)
                last_heartbeat = time.time()

            try:
                tr, summ = simulate_episode(
                    client=client,
                    d=d,
                    cond=cond,
                    rounds=args.rounds,
                    schedule=args.schedule,
                    order_mode=args.order_mode,
                    seed=args.seed,
                    seed_tag=seed_tag,
                    max_tokens_per_turn=args.max_tokens,
                    agent_mode=args.agent_mode,
                )

                # stream to disk immediately
                for row in tr:
                    append_jsonl(row, transcript_path)
                append_jsonl(summ, summary_path)

                done.add((d.id, cond["name"]))

                if summ.get("skipped"):
                    log_progress(f"skipped episode={episode_i} {seed_tag} ({summ.get('skip_reason', '')})", progress_path)
                elif episode_i % 25 == 0:
                    log_progress(f"completed episode={episode_i} last={seed_tag}", progress_path)

            except Exception as e:
                err_row = {
                    "qid": d.id,
                    "condition": cond["name"],
                    "ratio": cond["ratio"],
                    "seed_tag": seed_tag,
                    "error_type": type(e).__name__,
                    "error_message": str(e)[:2000],
                }
                append_jsonl(err_row, error_path)
                log_progress(f"ERROR episode={episode_i} {seed_tag} {type(e).__name__}", progress_path)

            if args.sleep and args.sleep > 0:
                time.sleep(args.sleep)

    log_progress("DONE", progress_path)
    print(f"Wrote transcripts: {transcript_path}")
    print(f"Wrote summaries:   {summary_path}")
    print(f"Wrote errors:      {error_path}")
    print(f"Wrote progress:    {progress_path}")


if __name__ == "__main__":
    main()

