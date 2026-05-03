from __future__ import annotations

import argparse
import asyncio
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

from exp1_models import get_model_spec
from exp1_client import send_one


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


async def _worker(
    name: str,
    queue: asyncio.Queue,
    client: httpx.AsyncClient,
    spec,
    out_f,
    lock: asyncio.Lock,
    *,
    max_retries: int,
    base_backoff_s: float,
    jitter_s: float,
) -> None:
    while True:
        item = await queue.get()
        if item is None:
            queue.task_done()
            return

        req = item
        attempt = 0
        while True:
            try:
                resp_json = await send_one(client, spec, req)
                row = {
                    "metadata": req.get("metadata"),
                    "provider": spec.provider,
                    "model_key": spec.key,
                    "model": spec.model,
                    "request": {
                        "params": req.get("params"),
                        "messages": req.get("messages"),
                    },
                    "response": resp_json,
                }
                async with lock:
                    out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    out_f.flush()
                break

            except Exception as e:
                attempt += 1
                if attempt > max_retries:
                    # write error row and move on
                    row = {
                        "metadata": req.get("metadata"),
                        "provider": spec.provider,
                        "model_key": spec.key,
                        "model": spec.model,
                        "error": str(e),
                        "request": {
                            "params": req.get("params"),
                            "messages": req.get("messages"),
                        },
                    }
                    async with lock:
                        out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                        out_f.flush()
                    break

                # exponential backoff with jitter
                sleep_s = base_backoff_s * (2 ** (attempt - 1)) + random.random() * jitter_s
                await asyncio.sleep(sleep_s)

        queue.task_done()


async def main_async(args: argparse.Namespace) -> None:
    in_path = Path(args.infile)
    out_path = Path(args.outfile)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = _load_jsonl(in_path)
    if args.max_requests is not None:
        rows = rows[: args.max_requests]

    spec = get_model_spec(
        args.model_key,
        openrouter_model=args.openrouter_model,
        supports_json_schema=None,
    )

    queue: asyncio.Queue = asyncio.Queue()
    for r in rows:
        queue.put_nowait(r)
    for _ in range(args.concurrency):
        queue.put_nowait(None)

    limits = httpx.Limits(max_connections=args.concurrency, max_keepalive_connections=args.concurrency)
    async with httpx.AsyncClient(limits=limits) as client:
        lock = asyncio.Lock()
        with out_path.open("w", encoding="utf-8") as out_f:
            workers = [
                asyncio.create_task(
                    _worker(
                        f"w{i}",
                        queue,
                        client,
                        spec,
                        out_f,
                        lock,
                        max_retries=args.max_retries,
                        base_backoff_s=args.base_backoff_s,
                        jitter_s=args.jitter_s,
                    )
                )
                for i in range(args.concurrency)
            ]
            await queue.join()
            await asyncio.gather(*workers)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--model_key", required=True, help="Key from exp1_models.DEFAULT_MODELS")
    p.add_argument("--infile", required=True, help="Input JSONL requests")
    p.add_argument("--outfile", required=True, help="Output JSONL responses")
    p.add_argument("--openrouter_model", default=None, help="Override OpenRouter model slug")

    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--max_requests", type=int, default=None)

    p.add_argument("--max_retries", type=int, default=5)
    p.add_argument("--base_backoff_s", type=float, default=1.0)
    p.add_argument("--jitter_s", type=float, default=0.25)
    return p


def main() -> None:
    args = build_parser().parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
