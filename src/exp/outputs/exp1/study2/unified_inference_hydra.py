#!/usr/bin/env python3
import asyncio
import hashlib
import json
import math
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

import hydra
import numpy as np
from omegaconf import DictConfig
from openai import AsyncAzureOpenAI, AsyncOpenAI

LIKERT_TOKENS = ["1", "2", "3", "4", "5", "6", "7"]
DIGIT_RE = re.compile(r"[1-7]")


# ============================================================================
# JSONL IO
# ============================================================================

def load_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    """Load JSONL file."""
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
    """Create deterministic custom ID for resume support."""
    cid = req.get("custom_id")
    if cid and str(cid).strip():
        return str(cid)

    # Fallback: hash the entire request
    blob = json.dumps(req, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()


def _read_existing_custom_ids(out_path: str) -> Set[str]:
    """Read already completed custom IDs from outputs file."""
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
    """Append a line to JSONL outputs."""
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# ============================================================================
# HELPERS
# ============================================================================

def _prepare_messages(req: Dict[str, Any]) -> List[Dict[str, str]]:
    """Extract messages from request object."""
    if "messages" in req:
        return req["messages"]
    if "body" in req and isinstance(req["body"], dict) and "messages" in req["body"]:
        return req["body"]["messages"]
    if "request" in req and isinstance(req["request"], dict) and "messages" in req["request"]:
        return req["request"]["messages"]
    raise ValueError("Request missing 'messages' field")


def _extract_first_1to7(text: str) -> Optional[str]:
    """Extract first digit 1-7 from text."""
    if not text:
        return None
    m = DIGIT_RE.search(text)
    return m.group(0) if m else None


# ============================================================================
# DISTRIBUTION STATS
# ============================================================================

def logits_to_probs(logits: Dict[str, float]) -> Dict[str, float]:
    """Convert logits to probabilities via softmax."""
    if not logits:
        return {tok: 1.0 / len(LIKERT_TOKENS) for tok in LIKERT_TOKENS}

    vals = np.array([logits.get(tok, -100.0) for tok in LIKERT_TOKENS], dtype=np.float64)
    exp_vals = np.exp(vals - np.max(vals))
    probs = exp_vals / np.sum(exp_vals)
    return {tok: float(p) for tok, p in zip(LIKERT_TOKENS, probs)}


def probs_to_stats(probs: Dict[str, float]) -> Dict[str, float]:
    """Compute mean and normalized entropy from probabilities."""
    mean = sum(int(tok) * probs[tok] for tok in LIKERT_TOKENS)
    p_vals = [probs[tok] for tok in LIKERT_TOKENS]
    entropy_raw = -sum(p * math.log(p + 1e-12) for p in p_vals)
    entropy_norm = entropy_raw / math.log(len(LIKERT_TOKENS))
    return {"mean": float(mean), "entropy_norm": float(entropy_norm)}


# ============================================================================
# LOGPROBS EXTRACTION
# ============================================================================

class LogprobsUnavailable(RuntimeError):
    pass


async def extract_logits_azure(
        client: AsyncAzureOpenAI,
        messages: List[Dict[str, str]],
        deployment: str,
        top_logprobs: int = 20,
) -> Dict[str, float]:
    """Extract logits from Azure OpenAI."""
    response = await client.chat.completions.create(
        model=deployment,
        messages=messages,
        max_tokens=1,
        logprobs=True,
        top_logprobs=top_logprobs,
        temperature=0.0,
    )

    if not (response.choices and response.choices[0].logprobs and response.choices[0].logprobs.content):
        raise LogprobsUnavailable("Azure response missing logprobs.content")

    content_logprobs = response.choices[0].logprobs.content
    if not content_logprobs:
        raise LogprobsUnavailable("Azure logprobs.content empty")

    top_logprobs_list = content_logprobs[0].top_logprobs or []
    result: Dict[str, float] = {}
    for item in top_logprobs_list:
        if item.token in LIKERT_TOKENS:
            result[item.token] = float(item.logprob)

    if not result:
        raise LogprobsUnavailable("No Likert tokens in Azure top_logprobs")

    # Fill missing tokens with low logprob
    for tok in LIKERT_TOKENS:
        result.setdefault(tok, -100.0)

    return result


async def extract_logits_openrouter(
        client: AsyncOpenAI,
        messages: List[Dict[str, str]],
        model: str,
        top_logprobs: int = 20,
) -> Dict[str, float]:
    """Extract logits from OpenRouter."""
    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=1,
        logprobs=True,
        top_logprobs=top_logprobs,
        temperature=0.0,
    )

    if not (response.choices and response.choices[0].logprobs and response.choices[0].logprobs.content):
        raise LogprobsUnavailable("OpenRouter response missing logprobs.content")

    content_logprobs = response.choices[0].logprobs.content
    if not content_logprobs:
        raise LogprobsUnavailable("OpenRouter logprobs.content empty")

    top_logprobs_list = content_logprobs[0].top_logprobs or []
    result: Dict[str, float] = {}
    for item in top_logprobs_list:
        if item.token in LIKERT_TOKENS:
            result[item.token] = float(item.logprob)

    if not result:
        raise LogprobsUnavailable("No Likert tokens in OpenRouter top_logprobs")

    for tok in LIKERT_TOKENS:
        result.setdefault(tok, -100.0)

    return result


async def extract_logits_vllm(
        client: AsyncOpenAI,
        messages: List[Dict[str, str]],
        model: str,
        top_logprobs: int = 20,
) -> Dict[str, float]:
    """
    Extract logits from vLLM server (OpenAI-compatible API).

    vLLM supports logprobs via the same API as OpenAI.
    Start vLLM server with:
      vllm serve meta-llama/Meta-Llama-3-70B-Instruct \\
        --port 8000 \\
        --dtype float16 \\
        --max-model-len 4096
    """
    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=1,
        logprobs=True,
        top_logprobs=top_logprobs,
        temperature=0.0,
    )

    if not (response.choices and response.choices[0].logprobs and response.choices[0].logprobs.content):
        raise LogprobsUnavailable("vLLM response missing logprobs.content")

    content_logprobs = response.choices[0].logprobs.content
    if not content_logprobs:
        raise LogprobsUnavailable("vLLM logprobs.content empty")

    top_logprobs_list = content_logprobs[0].top_logprobs or []
    result: Dict[str, float] = {}
    for item in top_logprobs_list:
        if item.token in LIKERT_TOKENS:
            result[item.token] = float(item.logprob)

    if not result:
        raise LogprobsUnavailable("No Likert tokens in vLLM top_logprobs")

    for tok in LIKERT_TOKENS:
        result.setdefault(tok, -100.0)

    return result


# ============================================================================
# JUSTIFICATION GENERATION (Study 2)
# ============================================================================

async def generate_justification(
        client: Any,
        provider: str,
        messages: List[Dict[str, str]],
        model_or_deployment: str,
        temperature: float = 0.7,
        max_tokens: int = 200,
) -> tuple:
    """
    Generate free-text justification (for Study 2 memory_justify conditions).
    Returns (choice: int|None, justification: str, raw_content: str).
    """
    if provider in ("openrouter", "vllm"):
        response = await client.chat.completions.create(
            model=model_or_deployment,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        raw = response.choices[0].message.content.strip()
    elif provider == "azure":
        from openai import AsyncAzureOpenAI
        response = await client.chat.completions.create(
            model=model_or_deployment,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        raw = response.choices[0].message.content.strip()
    else:
        raise ValueError(f"generate_justification: unsupported provider {provider}")

    # Parse: first character should be digit 1-7
    import re
    choice = None
    justification = raw
    m = re.match(r'^([1-7])', raw)
    if m:
        choice = int(m.group(1))
        justification = re.sub(r'^[1-7][\.:\-\s]+', '', raw).strip()
    else:
        # Scan first 20 chars for any digit 1-7
        for ch in raw[:20]:
            if ch.isdigit() and 1 <= int(ch) <= 7:
                choice = int(ch)
                idx = raw.index(ch)
                justification = re.sub(r'^[\s\.:\-]+', '', raw[idx + 1:]).strip()
                break

    return choice, justification, raw


# ============================================================================
# SAMPLING FALLBACK
# ============================================================================

async def sample_distribution_chat(
        client: Any,
        provider: str,
        messages: List[Dict[str, str]],
        model_or_deployment: str,
        n: int = 200,
        temperature: float = 1.0,
        max_tokens: int = 2,
) -> tuple[Dict[str, float], Dict[str, Any]]:
    """
    Fallback sampling-based distribution estimation.

    Generates n samples and estimates probabilities by frequency.
    """
    if provider == "azure":
        response = await client.chat.completions.create(
            model=model_or_deployment,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            n=n,
        )
    elif provider in {"openrouter", "vllm"}:
        response = await client.chat.completions.create(
            model=model_or_deployment,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            n=n,
        )
    else:
        raise ValueError(f"Unknown provider for sampling: {provider}")

    counts = Counter()
    for choice in response.choices:
        content = choice.message.content or ""
        digit = _extract_first_1to7(content)
        if digit:
            counts[digit] += 1

    total = sum(counts.values())
    if total == 0:
        probs = {tok: 1.0 / len(LIKERT_TOKENS) for tok in LIKERT_TOKENS}
    else:
        probs = {tok: float(counts.get(tok, 0) / total) for tok in LIKERT_TOKENS}

    metadata = {
        "n": n,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "valid_samples": total,
        "counts": dict(counts),
    }

    return probs, metadata


# ============================================================================
# REQUEST PROCESSING
# ============================================================================

async def run_one_request_with_retries(
        client: Any,
        req: Dict[str, Any],
        provider: str,
        model_or_deployment: str,
        inference_mode: str = "logprobs",
        max_retries: int = 5,
        top_logprobs: int = 20,
        sampling_n: int = 200,
        sampling_temperature: float = 1.0,
        sampling_max_tokens: int = 2,
) -> Dict[str, Any]:
    """
    Process a single request with retry logic.

    Tries logprobs first, falls back to sampling if unavailable.
    """
    custom_id = _make_custom_id(req)
    messages = _prepare_messages(req)
    metadata = req.get("metadata", {})

    last_error = None

    for attempt in range(max_retries):
        try:
            # 1) TRY LOGPROBS PATH
            if inference_mode in {"logprobs", "auto"}:
                try:
                    if provider == "azure":
                        logits = await extract_logits_azure(
                            client, messages, model_or_deployment, top_logprobs
                        )
                    elif provider == "openrouter":
                        logits = await extract_logits_openrouter(
                            client, messages, model_or_deployment, top_logprobs
                        )
                    elif provider == "vllm":
                        logits = await extract_logits_vllm(
                            client, messages, model_or_deployment, top_logprobs
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
                    if inference_mode == "logprobs":
                        # User explicitly requested logprobs - fail
                        raise
                    # Auto mode: fall through to sampling
                    pass

            # 2) STUDY 2 MODE: logprobs (max_tokens=1) + justification (max_tokens=200)
            if inference_mode == "study2":
                # Step A: extract clean logprobs with max_tokens=1
                if provider == "azure":
                    logits = await extract_logits_azure(client, messages, model_or_deployment, top_logprobs)
                elif provider == "openrouter":
                    logits = await extract_logits_openrouter(client, messages, model_or_deployment, top_logprobs)
                elif provider == "vllm":
                    logits = await extract_logits_vllm(client, messages, model_or_deployment, top_logprobs)
                else:
                    raise ValueError(f"Unknown provider: {provider}")

                probs = logits_to_probs(logits)
                stats = probs_to_stats(probs)
                answer = max(probs.items(), key=lambda x: x[1])[0]

                # Step B: generate full response with justification
                choice, justification, raw = await generate_justification(
                    client, provider, messages, model_or_deployment,
                    temperature=0.7, max_tokens=sampling_max_tokens,
                )
                # Use logprob-derived answer as primary; choice from text as secondary check
                return {
                    "custom_id": custom_id,
                    "metadata": metadata,
                    "response": {
                        "status_code": 200,
                        "provider": provider,
                        "mode": "study2",
                        "logits": logits,
                        "probs": probs,
                        "mean": stats["mean"],
                        "entropy_norm": stats["entropy_norm"],
                        "answer": answer,
                        "choice_from_text": choice,
                        "justification": justification,
                        "raw_content": raw,
                    },
                }

            # 3) TEXT GENERATION MODE (for cue generation — returns raw text, no logprobs)
            if inference_mode == "text":
                if provider in ("openrouter", "vllm"):
                    response = await client.chat.completions.create(
                        model=model_or_deployment,
                        messages=messages,
                        temperature=sampling_temperature,
                        max_tokens=sampling_max_tokens,
                    )
                elif provider == "azure":
                    response = await client.chat.completions.create(
                        model=model_or_deployment,
                        messages=messages,
                        temperature=sampling_temperature,
                        max_tokens=sampling_max_tokens,
                    )
                else:
                    raise ValueError(f"Unknown provider: {provider}")

                content = response.choices[0].message.content or ""

                return {
                    "custom_id": custom_id,
                    "metadata": metadata,
                    "response": {
                        "status_code": 200,
                        "provider": provider,
                        "mode": "text",
                        "content": content.strip(),
                    },
                }

            # 4) SAMPLING FALLBACK
            if inference_mode in {"auto", "sampling"}:
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
            if attempt < max_retries - 1:
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


# ============================================================================
# RUNNER
# ============================================================================

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
        sampling_temperature: float = 1.0,
        sampling_max_tokens: int = 2,
) -> None:
    """Run all requests with concurrency control."""
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


# ============================================================================
# HYDRA ENTRYPOINT
# ============================================================================

@hydra.main(version_base=None, config_path="configs",
            config_name="unified_inference")
def main(cfg: DictConfig) -> None:
    """Main entry point with Hydra configuration."""
    provider = str(cfg.provider).lower()
    in_path = str(cfg.experiment.data_input_path)
    out_path = str(cfg.experiment.output_full_path)

    concurrency = int(cfg.experiment.get("concurrency", 10))
    max_retries = int(cfg.experiment.get("max_retries", 5))
    top_logprobs = int(cfg.experiment.get("top_logprobs", 20))
    rate_limit_sleep = float(cfg.experiment.get("rate_limit_sleep", 0.0))

    inference_mode = str(cfg.experiment.get("inference_mode", "auto"))
    sampling_n = int(cfg.experiment.get("sampling_n", 200))
    sampling_temperature = float(cfg.experiment.get("sampling_temperature", 1.0))
    sampling_max_tokens = int(cfg.experiment.get("sampling_max_tokens", 2))

    print(f"[INFO] Provider: {provider}")
    print(f"[INFO] Reading requests: {in_path}")
    print(f"[INFO] inference_mode={inference_mode}")

    reqs_all = list(load_jsonl(in_path))
    print(f"[INFO] Requests loaded: {len(reqs_all)}")

    # Resume support
    done_ids = _read_existing_custom_ids(out_path)
    if done_ids:
        print(f"[INFO] Resume: found {len(done_ids)} completed items")

    reqs = [r for r in reqs_all if _make_custom_id(r) not in done_ids]
    print(f"[INFO] Remaining to run: {len(reqs)}")

    if not reqs:
        print("[OK] Nothing to do.")
        return

    # Initialize client
    if provider == "azure":
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Missing AZURE_OPENAI_API_KEY environment variable")

        endpoint = str(cfg.azure.endpoint)
        api_version = str(cfg.azure.api_version)
        deployment = str(cfg.azure.deployment)

        print(f"[INFO] Azure endpoint: {endpoint}")
        print(f"[INFO] Azure deployment: {deployment}")

        client = AsyncAzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version
        )
        model_or_deployment = deployment

    elif provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("Missing OPENROUTER_API_KEY environment variable")

        base_url = str(cfg.openrouter.get("base_url", "https://openrouter.ai/api/v1"))
        model = str(cfg.openrouter.model)

        print(f"[INFO] OpenRouter base URL: {base_url}")
        print(f"[INFO] OpenRouter model: {model}")

        client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        model_or_deployment = model

    elif provider == "vllm":
        base_url = str(cfg.vllm.base_url)
        model = str(cfg.vllm.model)

        print(f"[INFO] vLLM base URL: {base_url}")
        print(f"[INFO] vLLM model: {model}")

        # vLLM doesn't require API key for local deployment
        client = AsyncOpenAI(
            api_key="EMPTY",  # vLLM doesn't check this
            base_url=base_url
        )
        model_or_deployment = model

    elif provider == "local_transformers":
        # ── Local HuggingFace Transformers (Llama on MPS/CUDA/CPU) ──────────
        # Runs synchronously — no async client needed.
        # Imports LocalLlamaInference from local_llama_inference.py (same dir).
        try:
            from local_llama_inference import LocalLlamaInference
        except ImportError:
            raise RuntimeError(
                "local_llama_inference.py not found in the same directory. "
                "Make sure it is alongside unified_inference_hydra.py."
            )

        lt_cfg = cfg.local_transformers
        model_path = str(lt_cfg.get("model_path", "meta-llama/Meta-Llama-3-8B-Instruct"))
        device = str(lt_cfg.get("device", "mps"))
        torch_dtype = str(lt_cfg.get("torch_dtype", "float16"))
        quantization = str(lt_cfg.get("quantization", "none"))
        extract_hs = bool(lt_cfg.get("extract_hidden_states", True))
        hs_layers = list(lt_cfg.get("hidden_state_layers", [-1, -5, -10, -15, -20, -25, -30]))

        print(f"[INFO] Model path:   {model_path}")
        print(f"[INFO] Device:       {device}")
        print(f"[INFO] dtype:        {torch_dtype}")
        print(f"[INFO] Quantization: {quantization}")
        print(f"[INFO] Hidden states: {extract_hs}, layers: {hs_layers}")

        local_model = LocalLlamaInference(
            model_path=model_path,
            device=device,
            dtype=torch_dtype,
            quantization=quantization,
            extract_hidden_states=extract_hs,
            hidden_state_layers=hs_layers if extract_hs else None,
        )

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        completed = 0
        with open(out_path, "a", encoding="utf-8") as f_out:
            for req in reqs:
                custom_id = _make_custom_id(req)
                try:
                    result = local_model.infer(req)
                    result["custom_id"] = custom_id
                except Exception as exc:
                    result = {
                        "custom_id": custom_id,
                        "metadata": req.get("metadata", {}),
                        "response": {"status_code": 500, "error": str(exc)},
                    }
                f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                f_out.flush()
                completed += 1
                if completed % 10 == 0 or completed == len(reqs):
                    print(f"[INFO] {completed}/{len(reqs)} completed")

        print(f"[OK] Saved results: {out_path}")
        return

    else:
        raise ValueError(f"Unknown provider: {provider}. Use 'azure', 'openrouter', 'vllm', or 'local_transformers'")

    # Run inference (API providers only — local_transformers returns above)
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
        )
    )

    print(f"[OK] Saved results: {out_path}")


if __name__ == "__main__":
    main()