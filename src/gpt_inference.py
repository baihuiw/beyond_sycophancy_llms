from __future__ import annotations

import asyncio
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Set

import hydra
from omegaconf import DictConfig
from openai import AsyncAzureOpenAI, AsyncOpenAI


AZURE_API_KEY = ""
OPENAI_COMPAT_API_KEY = ""


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


def _safe_float(x, default=None) -> Optional[float]:
    if x is None:
        return default
    try:
        return float(x)
    except Exception:
        return default


def _safe_int(x, default=None) -> Optional[int]:
    if x is None:
        return default
    try:
        return int(x)
    except Exception:
        return default


def _remove_json_schema_response_format(body: Dict[str, Any]) -> None:
    rf = body.get("response_format")
    if isinstance(rf, dict) and rf.get("type") == "json_schema":
        body.pop("response_format", None)


def _extract_status_code(e: Exception) -> int:
    sc = getattr(e, "status_code", None)
    if isinstance(sc, int):
        return sc
    resp = getattr(e, "response", None)
    if resp is not None:
        sc2 = getattr(resp, "status_code", None)
        if isinstance(sc2, int):
            return sc2
    return 500


def _should_retry(status_code: int) -> bool:
    return status_code in (408, 429, 500, 502, 503, 504)


def _backoff_seconds(attempt: int, base: float = 1.0, cap: float = 60.0) -> float:
    raw = min(cap, base * (2 ** (attempt - 1)))
    return raw * (0.6 + 0.8 * random.random())

def make_azure_client(cfg: DictConfig) -> AsyncAzureOpenAI:
    if not AZURE_API_KEY.strip():
        raise RuntimeError("AZURE_API_KEY is empty. Fill it at top of gpt_inference.py")
    return AsyncAzureOpenAI(
        api_version=cfg.azure.api_version,
        azure_endpoint=cfg.azure.endpoint,
        api_key=AZURE_API_KEY.strip(),
    )


def make_openai_compat_client(cfg: DictConfig) -> AsyncOpenAI:
    if not OPENAI_COMPAT_API_KEY.strip():
        raise RuntimeError("OPENAI_COMPAT_API_KEY is empty. Fill it at top of gpt_inference.py")
    base_url = str(cfg.openai_compat.base_url).rstrip("/")
    return AsyncOpenAI(api_key=OPENAI_COMPAT_API_KEY.strip(), base_url=base_url)


def _prepare_body(
    req: Dict[str, Any],
    *,
    forced_model: str,
    default_temperature: Optional[float],
    default_top_p: Optional[float],
    default_max_tokens: Optional[int],
) -> Dict[str, Any]:
    body = dict(req.get("body", {}) or {})
    body["model"] = forced_model

    _remove_json_schema_response_format(body)

    if default_temperature is not None and "temperature" not in body:
        body["temperature"] = float(default_temperature)
    if default_top_p is not None and "top_p" not in body:
        body["top_p"] = float(default_top_p)

    if (
        default_max_tokens is not None
        and "max_tokens" not in body
        and "max_completion_tokens" not in body
        and "max_new_tokens" not in body
    ):
        body["max_tokens"] = int(default_max_tokens)

    body.pop("stream", None)
    return body


async def _call_chat_completions(client: Any, body: Dict[str, Any]) -> Dict[str, Any]:
    resp = await client.chat.completions.create(**body)
    return resp.model_dump()


async def run_one_request_with_retries(
    *,
    provider: str,
    client: Any,
    req: Dict[str, Any],
    forced_model: str,
    default_temperature: Optional[float],
    default_top_p: Optional[float],
    default_max_tokens: Optional[int],
    max_retries: int,
    per_worker_sleep: float,
) -> Dict[str, Any]:
    custom_id = req.get("custom_id")
    metadata = req.get("metadata", {}) or {}

    body = _prepare_body(
        req,
        forced_model=forced_model,
        default_temperature=default_temperature,
        default_top_p=default_top_p,
        default_max_tokens=default_max_tokens,
    )

    last_err: Optional[str] = None
    last_code: int = 500

    for attempt in range(1, max_retries + 2):
        try:
            out = await _call_chat_completions(client, body)
            if per_worker_sleep > 0:
                await asyncio.sleep(per_worker_sleep)
            return {
                "custom_id": custom_id,
                "metadata": metadata,
                "response": {"status_code": 200, "body": out},
            }
        except Exception as e:
            code = _extract_status_code(e)
            last_code = code
            last_err = f"{type(e).__name__}: {str(e)}"

            if attempt >= (max_retries + 1) or not _should_retry(code):
                if per_worker_sleep > 0:
                    await asyncio.sleep(per_worker_sleep)
                return {
                    "custom_id": custom_id,
                    "metadata": metadata,
                    "response": {"status_code": code, "error": last_err, "provider": provider},
                }

            await asyncio.sleep(_backoff_seconds(attempt))

    return {
        "custom_id": custom_id,
        "metadata": metadata,
        "response": {"status_code": last_code, "error": last_err or "unknown error", "provider": provider},
    }


# -----------------------------
# Hydra entrypoint
# -----------------------------
@hydra.main(version_base=None, config_path="../configs", config_name="study1_azure")
def main(cfg: DictConfig) -> None:
    provider = str(getattr(cfg, "provider", "azure")).lower()

    if not getattr(cfg, "experiment", None):
        raise ValueError("Missing cfg.experiment in conf (use +experiment.data_input_path=... etc).")

    in_path = str(cfg.experiment.data_input_path)
    out_path = str(cfg.experiment.output_full_path)

    concurrency = _safe_int(getattr(cfg.experiment, "concurrency", None), 8) or 8
    max_retries = _safe_int(getattr(cfg.experiment, "max_retries", None), 5) or 5
    per_worker_sleep = _safe_float(getattr(cfg.experiment, "rate_limit", 0.0), 0.0) or 0.0

    default_temperature = None
    default_top_p = None
    default_max_tokens = None
    if getattr(cfg, "generation", None):
        default_temperature = _safe_float(getattr(cfg.generation, "temperature", None), None)
        default_top_p = _safe_float(getattr(cfg.generation, "top_p", None), None)
        default_max_tokens = _safe_int(getattr(cfg.generation, "max_tokens", None), None)

    print(f"[INFO] Provider: {provider}")
    print(f"[INFO] Reading requests: {in_path}")

    reqs_all = list(load_jsonl(in_path))
    print(f"[INFO] Requests loaded: {len(reqs_all)}")

    done_ids = _read_existing_custom_ids(out_path)
    if done_ids:
        print(f"[INFO] Resume enabled: found {len(done_ids)} completed items in {out_path}")

    reqs = [r for r in reqs_all if str(r.get("custom_id") or "") not in done_ids]
    print(f"[INFO] Remaining to run: {len(reqs)} (concurrency={concurrency}, max_retries={max_retries})")

    if not reqs:
        print("[OK] Nothing to do.")
        return

    if provider == "azure":
        for k in ["endpoint", "api_version", "deployment"]:
            if not getattr(cfg.azure, k, None):
                raise ValueError(f"Missing cfg.azure.{k} in conf.")
        client = make_azure_client(cfg)
        forced_model = str(cfg.azure.deployment)

    elif provider == "openai_compat":
        for k in ["base_url", "model"]:
            if not getattr(cfg.openai_compat, k, None):
                raise ValueError(f"Missing cfg.openai_compat.{k} in conf.")
        client = make_openai_compat_client(cfg)
        forced_model = str(cfg.openai_compat.model)

    else:
        raise ValueError("Unsupported provider. Use provider=azure or provider=openai_compat.")

    async def runner() -> None:
        sem = asyncio.Semaphore(concurrency)

        async def run_wrapped(r: Dict[str, Any]) -> Dict[str, Any]:
            async with sem:
                return await run_one_request_with_retries(
                    provider=provider,
                    client=client,
                    req=r,
                    forced_model=forced_model,
                    default_temperature=default_temperature,
                    default_top_p=default_top_p,
                    default_max_tokens=default_max_tokens,
                    max_retries=max_retries,
                    per_worker_sleep=per_worker_sleep,
                )

        tasks = [asyncio.create_task(run_wrapped(r)) for r in reqs]

        completed = 0
        total = len(tasks)
        start = time.time()

        for coro in asyncio.as_completed(tasks):
            out = await coro
            _append_jsonl_line(out_path, out)
            completed += 1

            if completed % 25 == 0 or completed == total:
                elapsed = time.time() - start
                rps = completed / elapsed if elapsed > 0 else 0.0
                print(f"[INFO] Progress: {completed}/{total}  ({rps:.2f} req/s)")

        try:
            await client.close()
        except Exception:
            pass

    asyncio.run(runner())
    print(f"[OK] Saved results (append): {out_path}")


if __name__ == "__main__":
    main()


