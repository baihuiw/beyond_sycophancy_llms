import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Set

from openai import AsyncOpenAI


def load_jsonl(path: str) -> List[Dict]:
    results = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def load_existing_outputs(path: str) -> Set[str]:
    """Load already completed custom_ids for resume."""
    done = set()
    if Path(path).exists():
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    obj = json.loads(line)
                    # Use custom_id as unique identifier
                    custom_id = obj.get("custom_id")
                    if custom_id:
                        done.add(custom_id)
    return done


async def generate_one(
        client: AsyncOpenAI,
        request: Dict,
        model: str,
        temperature: float = 1.0,
        max_tokens: int = 200,
) -> Dict:
    """Generate text for one request."""
    messages = request["messages"]
    metadata = request.get("metadata", {})
    custom_id = request.get("custom_id", "")

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        content = response.choices[0].message.content.strip()

        return {
            "custom_id": custom_id,
            "metadata": metadata,
            "response": {
                "status_code": 200,
                "content": content,
            }
        }

    except Exception as e:
        return {
            "custom_id": custom_id,
            "metadata": metadata,
            "response": {
                "status_code": 500,
                "error": str(e),
            }
        }


async def run_all(
        requests: List[Dict],
        model: str,
        output_path: str,
        concurrency: int = 10,
        temperature: float = 1.0,
        max_tokens: int = 200,
):
    """Run all requests with concurrency control."""

    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Set OPENROUTER_API_KEY or OPENAI_API_KEY environment variable")

    base_url = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")

    client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    sem = asyncio.Semaphore(concurrency)

    # Open outputs file in append mode
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    async def run_with_sem(req: Dict) -> Dict:
        async with sem:
            return await generate_one(client, req, model, temperature, max_tokens)

    tasks = [asyncio.create_task(run_with_sem(r)) for r in requests]

    completed = 0
    total = len(requests)

    with open(output_path, "a", encoding="utf-8") as f:
        for coro in asyncio.as_completed(tasks):
            result = await coro
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()

            completed += 1
            if completed % 10 == 0 or completed == total:
                print(f"[INFO] Progress: {completed}/{total}")

    print(f"[OK] Saved {completed} results to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate text (cues) using LLM")
    parser.add_argument("--input", required=True, help="Input JSONL with requests")
    parser.add_argument("--outputs", required=True, help="Output JSONL path")
    parser.add_argument("--model", default="openai/gpt-4o", help="Model name")
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--resume", action="store_true", help="Skip already completed items")

    args = parser.parse_args()

    print(f"[INFO] Loading requests from {args.input}")
    requests = load_jsonl(args.input)
    print(f"[INFO] Loaded {len(requests)} requests")

    if args.resume:
        done = load_existing_outputs(args.output)
        print(f"[INFO] Found {len(done)} already completed")

        # Filter by custom_id
        requests = [r for r in requests if r.get("custom_id") not in done]
        print(f"[INFO] Remaining: {len(requests)}")

    if not requests:
        print("[OK] Nothing to do")
        return

    print(f"[INFO] Model: {args.model}")
    print(f"[INFO] Concurrency: {args.concurrency}")
    print(f"[INFO] Temperature: {args.temperature}")
    print(f"[INFO] Max tokens: {args.max_tokens}")

    asyncio.run(run_all(
        requests=requests,
        model=args.model,
        output_path=args.output,
        concurrency=args.concurrency,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    ))


if __name__ == "__main__":
    main()