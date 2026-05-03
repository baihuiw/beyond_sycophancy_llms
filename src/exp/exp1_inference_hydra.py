from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import hydra
import numpy as np
from omegaconf import DictConfig
from openai import AsyncAzureOpenAI, AsyncOpenAI


LIKERT_TOKENS = ["1", "2", "3", "4", "5", "6", "7"]
DIGIT_RE = re.compile(r"[1-7]")

AZURE_API_KEY = ""
OPENROUTER_API_KEY = ""
# ============================================================
# JSONL IO
# ============================================================
def load_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_no} of {path}: {e}") from e


def _make_custom_id(req: Dict[str, Any]) -> str:
    """Deterministic ID for resume."""
    cid = req.get("custom_id")
    if cid is not None and str(cid).strip():
        return str(cid)

    md = req.get("metadata") or {}
    if isinstance(md, dict):
        qid = str(md.get("qid", "")).strip()
        mem = str(md.get("memory", "")).strip()
        cue = str(md.get("cue", "")).strip()
        rep = str(md.get("rep", "")).strip()
        if qid or mem or cue or rep:
            return f"{qid}|{mem}|{cue}|{rep}"

    blob = json.dumps(req, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()


def _read_existing_custom_ids(out_path: str) -> Set[str]:
    p = Path(out_path)
    if not p.exists():
        return set()
    seen: Set[str] = set()
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                cid = str(obj.get("custom_id") or "")
                if cid:
                    seen.add(cid)
            except Exception:
                continue
    return seen


def _append_jsonl_line(out_path: str, obj: Dict[str, Any]) -> None:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# ============================================================
# Helpers
# ============================================================
def _require_env(env_name: str) -> str:
    v = os.getenv(env_name, "").strip()
    if not v:
        raise RuntimeError(f"Missing {env_name}. Please set it as an environment variable.")
    return v


def _prepare_messages(req: Dict[str, Any]) -> List[Dict[str, str]]:
    if "messages" in req:
        return req["messages"]
    if "body" in req and isinstance(req["body"], dict) and "messages" in req["body"]:
        return req["body"]["messages"]
    if "request" in req and isinstance(req["request"], dict) and "messages" in req["request"]:
        return req["request"]["messages"]
    raise ValueError("Request missing 'messages' field")


def _extract_first_1to7(text: str) -> Optional[str]:
    if not text:
        return None
    m = DIGIT_RE.search(text)
    return m.group(0) if m else None


# ============================================================
# Distribution / stats
# ============================================================
def logits_to_probs(logits: Dict[str, float]) -> Dict[str, float]:
    """Convert logits (logprobs are fine) to probabilities over 1..7 via softmax."""
    if not logits:
        return {tok: 1.0 / len(LIKERT_TOKENS) for tok in LIKERT_TOKENS}

    vals = np.array([logits.get(tok, -100.0) for tok in LIKERT_TOKENS], dtype=np.float64)
    exp_vals = np.exp(vals - np.max(vals))
    probs = exp_vals / np.sum(exp_vals)
    return {tok: float(p) for tok, p in zip(LIKERT_TOKENS, probs)}


def probs_to_stats(probs: Dict[str, float]) -> Dict[str, float]:
    mean = sum(int(tok) * probs[tok] for tok in LIKERT_TOKENS)
    p_vals = [probs[tok] for tok in LIKERT_TOKENS]
    entropy_raw = -sum(p * math.log(p + 1e-12) for p in p_vals)
    entropy_norm = entropy_raw / math.log(len(LIKERT_TOKENS))
    return {"mean": float(mean), "entropy_norm": float(entropy_norm)}


def counts_to_probs(counts: Counter) -> Dict[str, float]:
    total = sum(counts.values())
    if total <= 0:
        return {tok: 1.0 / len(LIKERT_TOKENS) for tok in LIKERT_TOKENS}
    return {tok: float(counts.get(tok, 0) / total) for tok in LIKERT_TOKENS}


# ============================================================
# Exceptions
# ============================================================
class LogprobsUnavailable(RuntimeError):
    pass


# ============================================================
# Logprobs extraction
# ============================================================
async def extract_logits_azure(
    client: AsyncAzureOpenAI,
    messages: List[Dict[str, str]],
    deployment: str,
    top_logprobs: int = 20,
) -> Dict[str, float]:
    response = await client.chat.completions.create(
        model=deployment,
        messages=messages,
        max_tokens=1,
        logprobs=True,
        top_logprobs=top_logprobs,
        temperature=0.0,
    )

    if not (response.choices and response.choices[0].logprobs and response.choices[0].logprobs.content):
        raise LogprobsUnavailable("Azure response did not include logprobs.content")

    content_logprobs = response.choices[0].logprobs.content
    if not content_logprobs or len(content_logprobs) == 0:
        raise LogprobsUnavailable("Azure logprobs.content empty")

    top_logprobs_list = content_logprobs[0].top_logprobs or []
    result: Dict[str, float] = {}
    for item in top_logprobs_list:
        if item.token in LIKERT_TOKENS:
            result[item.token] = float(item.logprob)

    if not result:
        raise LogprobsUnavailable("Azure top_logprobs contained none of the Likert tokens")

    # Fill missing tokens with low logprob for consistent softmax
    for tok in LIKERT_TOKENS:
        result.setdefault(tok, -100.0)
    return result


async def extract_logits_openrouter(
    client: AsyncOpenAI,
    messages: List[Dict[str, str]],
    model: str,
    top_logprobs: int = 20,
    require_parameters: bool = False,
) -> Dict[str, float]:
    # OpenRouter supports provider routing hints in-body; with OpenAI SDK, pass via extra_body.
    extra_body = {"provider": {"require_parameters": True}} if require_parameters else None

    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=1,
        logprobs=True,
        top_logprobs=top_logprobs,
        temperature=0.0,
        extra_body=extra_body,
    )

    if not (response.choices and response.choices[0].logprobs and response.choices[0].logprobs.content):
        raise LogprobsUnavailable("OpenRouter response did not include logprobs.content")

    content_logprobs = response.choices[0].logprobs.content
    if not content_logprobs or len(content_logprobs) == 0:
        raise LogprobsUnavailable("OpenRouter logprobs.content empty")

    top_logprobs_list = content_logprobs[0].top_logprobs or []
    result: Dict[str, float] = {}
    for item in top_logprobs_list:
        if item.token in LIKERT_TOKENS:
            result[item.token] = float(item.logprob)

    if not result:
        raise LogprobsUnavailable("OpenRouter top_logprobs contained none of the Likert tokens")

    for tok in LIKERT_TOKENS:
        result.setdefault(tok, -100.0)
    return result


# ============================================================
# Sampling-based estimation
# ============================================================
async def sample_distribution_chat(
    client: Any,
    provider: str,
    messages: List[Dict[str, str]],
    model_or_deployment: str,
    *,
    n: int = 200,
    temperature: float = 0.7,
    max_tokens: int = 2,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    Estimate P(1..7) via repeated short generations. Works even without logprobs.
    Returns: (probs, debug_meta)
    """
    counts: Counter = Counter()
    nulls = 0

    for _ in range(n):
        if provider == "azure":
            resp = await client.chat.completions.create(
                model=model_or_deployment,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=["\n"],
            )
        elif provider == "openrouter":
            resp = await client.chat.completions.create(
                model=model_or_deployment,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=["\n"],
            )
        else:
            raise ValueError(f"Unknown provider for sampling: {provider}")

        content = ""
        if resp.choices and resp.choices[0].message and resp.choices[0].message.content:
            content = resp.choices[0].message.content

        choice = _extract_first_1to7(content)
        if choice is None:
            nulls += 1
        else:
            counts[choice] += 1

    probs = counts_to_probs(counts)
    meta = {
        "n": n,
        "nulls": nulls,
        "counts": {k: int(v) for k, v in counts.items()},
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
    }
    return probs, meta


# ============================================================
# Request processing with retry logic
# ============================================================
async def run_one_request_with_retries(
    client: Any,
    req: Dict[str, Any],
    provider: str,
    model_or_deployment: str,
    *,
    inference_mode: str = "auto",        # "auto" | "logprobs" | "sampling"
    max_retries: int = 5,
    top_logprobs: int = 20,
    sampling_n: int = 200,
    sampling_temperature: float = 0.7,
    sampling_max_tokens: int = 2,
    openrouter_require_parameters: bool = False,
) -> Dict[str, Any]:
    custom_id = _make_custom_id(req)
    metadata = req.get("metadata", {}) or {}
    messages = _prepare_messages(req)

    last_error = None
    for attempt in range(max_retries + 1):
        try:
            mode = inference_mode.lower().strip()

            # 1) LOGPROBS PATH
            if mode in {"auto", "logprobs"}:
                try:
                    if provider == "azure":
                        logits = await extract_logits_azure(client, messages, model_or_deployment, top_logprobs)
                    elif provider == "openrouter":
                        logits = await extract_logits_openrouter(
                            client,
                            messages,
                            model_or_deployment,
                            top_logprobs=top_logprobs,
                            require_parameters=openrouter_require_parameters,
                        )
                    else:
                        raise ValueError(f"Unknown provider: {provider}")

                    probs = logits_to_probs(logits)
                    stats = probs_to_stats(probs)
                    answer = max(probs.items(), key=lambda x: x[1])[0]

                    return {
                        "custom_id": custom_id,
                        "metadata": metadata,
                        "response": {
                            "status_code": 200,
                            "provider": provider,
                            "mode": "logprobs",
                            "logits": logits,
                            "probs": probs,
                            "mean": stats["mean"],
                            "entropy_norm": stats["entropy_norm"],
                            "answer": answer,
                        },
                    }

                except LogprobsUnavailable as e:
                    if mode == "logprobs":
                        raise  # strict mode: fail
                    # auto mode: fall through to sampling
                except Exception:
                    # other errors should retry in outer loop
                    raise

            # 2) SAMPLING PATH
            if mode in {"auto", "sampling"}:
                probs, samp_meta = await sample_distribution_chat(
                    client,
                    provider,
                    messages,
                    model_or_deployment,
                    n=sampling_n,
                    temperature=sampling_temperature,
                    max_tokens=sampling_max_tokens,
                )
                stats = probs_to_stats(probs)
                answer = max(probs.items(), key=lambda x: x[1])[0]

                return {
                    "custom_id": custom_id,
                    "metadata": metadata,
                    "response": {
                        "status_code": 200,
                        "provider": provider,
                        "mode": "sampling",
                        "sampling": samp_meta,
                        "probs": probs,
                        "mean": stats["mean"],
                        "entropy_norm": stats["entropy_norm"],
                        "answer": answer,
                    },
                }

            raise ValueError(f"Invalid inference_mode: {inference_mode}")

        except Exception as e:
            last_error = f"{type(e).__name__}: {str(e)}"
            if attempt < max_retries:
                wait_time = min(60.0, 2 ** attempt)
                print(f"[RETRY] {custom_id} attempt {attempt + 1}/{max_retries}: {last_error}")
                await asyncio.sleep(wait_time)
            else:
                return {
                    "custom_id": custom_id,
                    "metadata": metadata,
                    "response": {
                        "status_code": 500,
                        "error": last_error,
                        "provider": provider,
                        "mode": inference_mode,
                    },
                }

    return {
        "custom_id": custom_id,
        "metadata": metadata,
        "response": {
            "status_code": 500,
            "error": last_error or "Unknown error",
            "provider": provider,
            "mode": inference_mode,
        },
    }


# ============================================================
# Runner
# ============================================================
async def run_all_requests(
    client: Any,
    reqs: List[Dict[str, Any]],
    *,
    provider: str,
    model_or_deployment: str,
    out_path: str,
    concurrency: int = 10,
    max_retries: int = 5,
    top_logprobs: int = 20,
    rate_limit_sleep: float = 0.0,
    inference_mode: str = "auto",
    sampling_n: int = 200,
    sampling_temperature: float = 0.7,
    sampling_max_tokens: int = 2,
    openrouter_require_parameters: bool = False,
) -> None:
    sem = asyncio.Semaphore(concurrency)
    completed = 0
    total = len(reqs)
    start = time.time()
    status_counts = Counter()

    async def run_with_sem(req: Dict[str, Any]) -> Dict[str, Any]:
        async with sem:
            result = await run_one_request_with_retries(
                client,
                req,
                provider,
                model_or_deployment,
                inference_mode=inference_mode,
                max_retries=max_retries,
                top_logprobs=top_logprobs,
                sampling_n=sampling_n,
                sampling_temperature=sampling_temperature,
                sampling_max_tokens=sampling_max_tokens,
                openrouter_require_parameters=openrouter_require_parameters,
            )
            if rate_limit_sleep > 0:
                await asyncio.sleep(rate_limit_sleep)
            return result

    tasks = [asyncio.create_task(run_with_sem(r)) for r in reqs]

    for coro in asyncio.as_completed(tasks):
        out = await coro
        _append_jsonl_line(out_path, out)

        sc = out.get("response", {}).get("status_code", 500)
        status_counts[int(sc) if isinstance(sc, int) else 500] += 1
        completed += 1

        if completed % 10 == 0 or completed == total:
            elapsed = time.time() - start
            rps = completed / elapsed if elapsed > 0 else 0.0
            ok = status_counts.get(200, 0)
            err = completed - ok
            print(
                f"[INFO] Progress: {completed}/{total} ({rps:.2f} req/s) "
                f"OK={ok} ERR={err} status_counts={dict(status_counts)}"
            )


# ============================================================
# Hydra entrypoint
# ============================================================
@hydra.main(version_base=None, config_path="configs", config_name="exp1_logits_openrouter_gpt4o")
def main(cfg: DictConfig) -> None:
    provider = str(cfg.provider).lower()
    in_path = str(cfg.experiment.data_input_path)
    out_path = str(cfg.experiment.output_full_path)

    concurrency = int(cfg.experiment.get("concurrency", 10))
    max_retries = int(cfg.experiment.get("max_retries", 5))
    top_logprobs = int(cfg.experiment.get("top_logprobs", 20))
    rate_limit_sleep = float(cfg.experiment.get("rate_limit_sleep", 0.0))

    inference_mode = str(cfg.experiment.get("inference_mode", "auto"))
    sampling_n = int(cfg.experiment.get("sampling_n", 1))
    sampling_temperature = float(cfg.experiment.get("sampling_temperature", 0.7))
    sampling_max_tokens = int(cfg.experiment.get("sampling_max_tokens", 2))

    openrouter_require_parameters = bool(cfg.experiment.get("openrouter_require_parameters", False))

    print(f"[INFO] Provider: {provider}")
    print(f"[INFO] Reading requests: {in_path}")
    print(f"[INFO] inference_mode={inference_mode} sampling_n={sampling_n} sampling_temperature={sampling_temperature}")

    reqs_all = list(load_jsonl(in_path))
    print(f"[INFO] Requests loaded: {len(reqs_all)}")

    done_ids = _read_existing_custom_ids(out_path)
    if done_ids:
        print(f"[INFO] Resume enabled: found {len(done_ids)} completed items in {out_path}")

    reqs = [r for r in reqs_all if _make_custom_id(r) not in done_ids]
    print(f"[INFO] Remaining to run: {len(reqs)}")
    if not reqs:
        print("[OK] Nothing to do.")
        return

    # Initialize client
    if provider == "azure":
        api_key = _require_env("AZURE_OPENAI_API_KEY")
        endpoint = str(cfg.azure.endpoint)
        api_version = str(cfg.azure.api_version)
        deployment = str(cfg.azure.deployment)

        print(f"[INFO] Azure endpoint: {endpoint}")
        print(f"[INFO] Azure deployment: {deployment}")

        client = AsyncAzureOpenAI(api_key=api_key, azure_endpoint=endpoint, api_version=api_version)
        model_or_deployment = deployment

    elif provider == "openrouter":
        api_key = _require_env("OPENROUTER_API_KEY")
        base_url = str(cfg.openrouter.get("base_url", "https://openrouter.ai/api/v1"))
        model = str(cfg.openrouter.model)

        print(f"[INFO] OpenRouter base URL: {base_url}")
        print(f"[INFO] OpenRouter model: {model}")
        if openrouter_require_parameters:
            print("[INFO] OpenRouter: require_parameters=True (will error if logprobs unsupported)")

        client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        model_or_deployment = model

    else:
        raise ValueError(f"Unknown provider: {provider}. Use 'azure' or 'openrouter'")

    asyncio.run(
        run_all_requests(
            client,
            reqs,
            provider=provider,
            model_or_deployment=model_or_deployment,
            out_path=out_path,
            concurrency=concurrency,
            max_retries=max_retries,
            top_logprobs=top_logprobs,
            rate_limit_sleep=rate_limit_sleep,
            inference_mode=inference_mode,
            sampling_n=sampling_n,
            sampling_temperature=sampling_temperature,
            sampling_max_tokens=sampling_max_tokens,
            openrouter_require_parameters=openrouter_require_parameters,
        )
    )

    print(f"[OK] Saved results: {out_path}")


if __name__ == "__main__":
    main()

