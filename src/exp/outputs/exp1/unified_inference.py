#!/usr/bin/env python3
"""
Unified Inference Script: Logit Probabilities + Justification
Hydra-configured version.

Combines logprob extraction and free-text justification in a single API call:
  1. Forces model to outputs choice (1-7) as FIRST token
  2. Collects logprobs for that token → probability distribution over choices
  3. Continues generation → justification text

Supported providers: openrouter, azure, openai

Usage:
    python unified_inference.py --conf-name=study2_gpt4o_openrouter
    python unified_inference.py --conf-name=study2_gpt4o_openrouter \\
        experiment.data_input_path=requests/my_requests.jsonl \\
        experiment.output_full_path=outputs/my_outputs.jsonl

Output format per line:
{
    "custom_id": "...",
    "metadata": {...},
    "response": {
        "status_code": 200,
        "choice": 5,
        "justification": "Because...",
        "raw_content": "5 Because...",
        "logprobs": {"1": -2.3, "2": -1.5, ...},
        "probs":    {"1": 0.10, "2": 0.22, ...},
        "mean": 4.2,
        "entropy_norm": 0.71
    }
}
"""

import asyncio
import json
import math
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import hydra
from omegaconf import DictConfig
from openai import AsyncOpenAI


# ============================================================================
# IO HELPERS
# ============================================================================

def load_jsonl(path: str) -> List[Dict]:
    results = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def load_completed_ids(path: str) -> Set[str]:
    """Return set of custom_ids already written to outputs (for resume)."""
    done: Set[str] = set()
    if not Path(path).exists():
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                cid = obj.get("custom_id")
                if cid:
                    done.add(cid)
    return done


# ============================================================================
# RESPONSE PARSING
# ============================================================================

def extract_choice_and_justification(content: str) -> Tuple[Optional[int], str]:
    """
    Extract choice (1-7) and justification from generated text.
    Tries multiple patterns in order of reliability.
    """
    content = content.strip()

    # Pattern 1: starts with digit 1-7
    m = re.match(r'^([1-7])', content)
    if m:
        choice = int(m.group(1))
        rest = content[1:].strip()
        rest = re.sub(r'^[\s.\:\-\n]+', '', rest).strip()
        return choice, rest

    # Pattern 2: "CHOICE: X" format
    m = re.search(r'CHOICE:\s*([1-7])', content, re.IGNORECASE)
    if m:
        choice = int(m.group(1))
        exp = re.search(r'EXPLANATION:\s*(.+)', content, re.IGNORECASE | re.DOTALL)
        return choice, exp.group(1).strip() if exp else ""

    # Pattern 3: first digit 1-7 within first 20 characters
    for ch in content[:20]:
        if ch.isdigit() and 1 <= int(ch) <= 7:
            choice = int(ch)
            idx = content.index(ch)
            rest = re.sub(r'^[\s.\:\-\n]+', '', content[idx + 1:]).strip()
            return choice, rest

    return None, content


def compute_probs(logprobs_data) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Extract and normalize probabilities for positions 1-7 from API logprobs.

    Returns:
        logprobs: raw log-probability per position (−inf if not in top-k)
        probs:    softmax-normalized probability per position
    """
    choices = ["1", "2", "3", "4", "5", "6", "7"]
    logprobs: Dict[str, float] = {c: float("-inf") for c in choices}

    if (logprobs_data
            and hasattr(logprobs_data, "content")
            and logprobs_data.content):
        first_tok = logprobs_data.content[0]
        if hasattr(first_tok, "top_logprobs"):
            for item in first_tok.top_logprobs:
                token = item.token.strip()
                if token in choices:
                    logprobs[token] = item.logprob

    # Softmax-normalize (stable: subtract max before exp)
    max_lp = max(logprobs.values())
    if max_lp == float("-inf"):
        probs = {c: 1.0 / len(choices) for c in choices}
    else:
        exp_vals = {c: math.exp(lp - max_lp) for c, lp in logprobs.items()}
        total = sum(exp_vals.values())
        probs = {c: exp_vals[c] / total for c in choices}

    return logprobs, probs


# ============================================================================
# INFERENCE
# ============================================================================

async def infer_one(
    client: AsyncOpenAI,
    request: Dict,
    model: str,
    temperature: float,
    max_tokens: int,
    top_logprobs: int,
) -> Dict:
    """Run a single request and return structured outputs."""
    messages   = request["messages"]
    metadata   = request.get("metadata", {})
    custom_id  = request.get("custom_id", "")

    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            logprobs=True,
            top_logprobs=top_logprobs,
        )

        raw      = resp.choices[0].message.content.strip()
        lp_data  = resp.choices[0].logprobs

        choice, justification   = extract_choice_and_justification(raw)
        logprobs, probs         = compute_probs(lp_data)

        mean          = sum(int(c) * p for c, p in probs.items())
        entropy       = -sum(p * math.log(p + 1e-10) for p in probs.values())
        entropy_norm  = entropy / math.log(7)

        return {
            "custom_id": custom_id,
            "metadata":  metadata,
            "response": {
                "status_code":  200,
                "choice":       choice,
                "justification": justification,
                "raw_content":  raw,
                "logprobs":     logprobs,
                "probs":        probs,
                "mean":         mean,
                "entropy_norm": entropy_norm,
            },
        }

    except Exception as exc:
        return {
            "custom_id": custom_id,
            "metadata":  metadata,
            "response":  {"status_code": 500, "error": str(exc)},
        }


async def run_all(
    requests: List[Dict],
    client: AsyncOpenAI,
    model: str,
    output_path: str,
    concurrency: int,
    temperature: float,
    max_tokens: int,
    top_logprobs: int,
) -> None:
    sem   = asyncio.Semaphore(concurrency)
    total = len(requests)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    async def bounded(req: Dict) -> Dict:
        async with sem:
            return await infer_one(client, req, model,
                                   temperature, max_tokens, top_logprobs)

    tasks = [asyncio.create_task(bounded(r)) for r in requests]

    completed = 0
    with open(output_path, "a", encoding="utf-8") as f:
        for coro in asyncio.as_completed(tasks):
            result = await coro
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()
            completed += 1
            if completed % 10 == 0 or completed == total:
                print(f"[INFO] {completed}/{total} completed")

    print(f"[OK] Saved {completed} results → {output_path}")


# ============================================================================
# HYDRA ENTRY POINT
# ============================================================================

@hydra.main(version_base=None, config_path="/Users/wangbaihui/modeling_distributions_thesis/src/exp1_v2/outputs/exp1/configs", config_name="study2_gpt4o_openrouter")
def main(cfg: DictConfig) -> None:

    provider    = str(cfg.provider).lower()
    in_path     = str(cfg.experiment.data_input_path)
    out_path    = str(cfg.experiment.output_full_path)
    concurrency = int(cfg.experiment.get("concurrency",  10))
    temperature = float(cfg.experiment.get("temperature", 0.7))
    max_tokens  = int(cfg.experiment.get("max_tokens",   200))
    top_logprobs = int(cfg.experiment.get("top_logprobs", 10))
    resume      = bool(cfg.experiment.get("resume",      True))

    print(f"[INFO] Provider:     {provider}")
    print(f"[INFO] Input:        {in_path}")
    print(f"[INFO] Output:       {out_path}")
    print(f"[INFO] Concurrency:  {concurrency}")
    print(f"[INFO] Temperature:  {temperature}")
    print(f"[INFO] Max tokens:   {max_tokens}")
    print(f"[INFO] Top logprobs: {top_logprobs}")

    # ── Load requests ────────────────────────────────────────────────────────
    requests = load_jsonl(in_path)
    print(f"[INFO] Loaded {len(requests)} requests")

    if resume:
        done = load_completed_ids(out_path)
        if done:
            print(f"[INFO] Resume: skipping {len(done)} already completed")
        requests = [r for r in requests if r.get("custom_id") not in done]
        print(f"[INFO] Remaining: {len(requests)}")

    if not requests:
        print("[OK] Nothing to do.")
        return

    # ── Build client ─────────────────────────────────────────────────────────
    if provider == "openrouter":
        api_key  = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("Set OPENROUTER_API_KEY environment variable")
        base_url = str(cfg.openrouter.get("base_url", "https://openrouter.ai/api/v1"))
        model    = str(cfg.openrouter.model)
        client   = AsyncOpenAI(api_key=api_key, base_url=base_url)

    elif provider == "azure":
        from openai import AsyncAzureOpenAI
        api_key    = os.getenv("AZURE_OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Set AZURE_OPENAI_API_KEY environment variable")
        client = AsyncAzureOpenAI(
            api_key=api_key,
            azure_endpoint=str(cfg.azure.endpoint),
            api_version=str(cfg.azure.api_version),
        )
        model = str(cfg.azure.deployment)

    elif provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Set OPENAI_API_KEY environment variable")
        model  = str(cfg.openai.model)
        client = AsyncOpenAI(api_key=api_key)

    else:
        raise ValueError(f"Unknown provider: {provider}. Use openrouter, azure, or openai.")

    print(f"[INFO] Model: {model}")

    # ── Run ──────────────────────────────────────────────────────────────────
    asyncio.run(run_all(
        requests=requests,
        client=client,
        model=model,
        output_path=out_path,
        concurrency=concurrency,
        temperature=temperature,
        max_tokens=max_tokens,
        top_logprobs=top_logprobs,
    ))


if __name__ == "__main__":
    main()