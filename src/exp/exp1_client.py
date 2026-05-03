from __future__ import annotations

import os
from dataclasses import asdict
from typing import Any, Dict, Optional, Tuple

import httpx

from exp1_models import ModelSpec


def _env(name: str, default: Optional[str] = None) -> str:
    v = os.getenv(name)
    if v is None or v == "":
        if default is None:
            raise RuntimeError(f"Missing required environment variable: {name}")
        return default
    return v


def build_endpoint_and_headers(spec: ModelSpec) -> Tuple[str, Dict[str, str]]:
    """Return (url, headers) for the provider."""
    if spec.provider == "openrouter":
        base = os.getenv("OPENAI_COMPAT_BASE_URL", "https://openrouter.ai/api/v1")
        key = os.getenv("OPENAI_COMPAT_API_KEY") or os.getenv("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("Missing OPENAI_COMPAT_API_KEY (or OPENROUTER_API_KEY) for OpenRouter")
        url = base.rstrip("/") + "/chat/completions"
        headers: Dict[str, str] = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }
        # Optional attribution headers (harmless if omitted)
        app_url = os.getenv("OPENROUTER_HTTP_REFERER")
        app_title = os.getenv("OPENROUTER_X_TITLE")
        if app_url:
            headers["HTTP-Referer"] = app_url
        if app_title:
            headers["X-Title"] = app_title
        return url, headers

    if spec.provider == "azure":
        endpoint = _env("AZURE_OPENAI_ENDPOINT")
        api_key = _env("AZURE_OPENAI_API_KEY")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")

        if not spec.deployment_env:
            raise RuntimeError(f"Azure model spec missing deployment_env: {asdict(spec)}")
        deployment = _env(spec.deployment_env)

        # Azure chat completions endpoint pattern:
        # {endpoint}/openai/deployments/{deployment}/chat/completions?api-version={api_version}
        url = endpoint.rstrip("/") + f"/openai/deployments/{deployment}/chat/completions?api-version={api_version}"
        headers = {
            "api-key": api_key,
            "Content-Type": "application/json",
        }
        return url, headers

    raise RuntimeError(f"Unknown provider: {spec.provider}")


def build_body(spec: ModelSpec, request: Dict[str, Any]) -> Dict[str, Any]:
    """Build provider-specific request body.

    `request` is the per-line JSONL request produced by exp1_generate_requests.py.
    """
    params = dict(request.get("params") or {})
    messages = request["messages"]

    body: Dict[str, Any] = {
        "messages": messages,
        **params,
    }

    if spec.provider == "openrouter":
        if not spec.model:
            raise RuntimeError(f"OpenRouter model spec missing model slug: {asdict(spec)}")
        body["model"] = spec.model

    # Azure uses deployment in URL; do NOT set 'model' field.
    return body


async def send_one(
    client: httpx.AsyncClient,
    spec: ModelSpec,
    request: Dict[str, Any],
    *,
    timeout_s: float = 60.0,
) -> Dict[str, Any]:
    url, headers = build_endpoint_and_headers(spec)
    body = build_body(spec, request)

    r = await client.post(url, headers=headers, json=body, timeout=timeout_s)
    # For troubleshooting, keep both status and raw text on errors.
    if r.status_code >= 400:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:500]}")
    return r.json()
