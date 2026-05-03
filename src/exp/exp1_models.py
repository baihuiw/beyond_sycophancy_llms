from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple



AZURE_API_KEY = ""
OPENAI_COMPAT_API_KEY = ""


AZURE_ENDPOINT_DEFAULT: str = ""
AZURE_API_VERSION_DEFAULT: str = "2024-12-01-preview"
AZURE_GPT5_DEPLOYMENT_DEFAULT: str = "gpt-5-chat"


OPENAI_COMPAT_BASE_URL_DEFAULT: str = "https://openrouter.ai/api/v1"


def _first_nonempty(*vals: Optional[str]) -> Optional[str]:
    for v in vals:
        if v is None:
            continue
        s = str(v).strip()
        if s:
            return s
    return None


def get_azure_connection() -> Tuple[str, str, str]:
    """Return (endpoint, api_version, api_key) for Azure OpenAI.

    Env vars override module defaults / placeholders.
    """
    endpoint = _first_nonempty(os.getenv("AZURE_OPENAI_ENDPOINT"), AZURE_ENDPOINT_DEFAULT)
    api_version = _first_nonempty(os.getenv("AZURE_OPENAI_API_VERSION"), AZURE_API_VERSION_DEFAULT)
    api_key = _first_nonempty(os.getenv("AZURE_OPENAI_API_KEY"), AZURE_API_KEY)

    if not endpoint or not api_version:
        raise RuntimeError("Azure endpoint/api_version missing. Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_VERSION.")
    if not api_key:
        raise RuntimeError(
            "Azure API key missing. Set env AZURE_OPENAI_API_KEY or paste it into AZURE_API_KEY in exp1_models.py."
        )
    return endpoint, api_version, api_key


def get_openai_compat_connection() -> Tuple[str, str]:
    """Return (base_url, api_key) for OpenAI-compatible providers (OpenRouter)."""
    base_url = _first_nonempty(os.getenv("OPENAI_COMPAT_BASE_URL"), OPENAI_COMPAT_BASE_URL_DEFAULT)
    api_key = _first_nonempty(os.getenv("OPENAI_COMPAT_API_KEY"), OPENAI_COMPAT_API_KEY)

    if not base_url:
        raise RuntimeError("OPENAI_COMPAT_BASE_URL missing. Set env OPENAI_COMPAT_BASE_URL.")
    if not api_key:
        raise RuntimeError(
            "OpenAI-compatible API key missing. Set env OPENAI_COMPAT_API_KEY or paste it into OPENAI_COMPAT_API_KEY in exp1_models.py."
        )
    return base_url.rstrip("/"), api_key


def resolve_azure_deployment(default: str = AZURE_GPT5_DEPLOYMENT_DEFAULT) -> str:
    """Resolve Azure deployment name for GPT-5.

    If AZURE_OPENAI_DEPLOYMENT_GPT5 is set, it wins; otherwise the default.
    """
    return _first_nonempty(os.getenv("AZURE_OPENAI_DEPLOYMENT_GPT5"), default) or default


@dataclass(frozen=True)
class ModelSpec:
    key: str
    provider: str  # 'azure' or 'openrouter'
    model: Optional[str] = None          # OpenRouter model slug (OpenAI-compat)
    deployment_env: Optional[str] = None # Azure deployment NAME (kept for backward-compat)
    supports_json_schema: bool = False   # keep False for OpenRouter (often 400)


# Default model set (edit freely)
DEFAULT_MODELS: Dict[str, ModelSpec] = {
    # Azure GPT-5 (deployment-based)
    "azure_gpt5": ModelSpec(
        key="azure_gpt5",
        provider="azure",
        deployment_env=AZURE_GPT5_DEPLOYMENT_DEFAULT,
        supports_json_schema=False,
    ),

    # OpenRouter models (OpenAI-compatible)
    "openrouter_qwen25_14b": ModelSpec(
        key="openrouter_qwen25_14b",
        provider="openrouter",
        model="qwen/qwen2.5-14b-instruct",
        supports_json_schema=False,
    ),
    "openrouter_llama33_70b_free": ModelSpec(
        key="openrouter_llama33_70b_free",
        provider="openrouter",
        model="meta-llama/llama-3.3-70b-instruct:free",
        supports_json_schema=False,
    ),
    "openrouter_ministral14b_2512": ModelSpec(
        key="openrouter_ministral14b_2512",
        provider="openrouter",
        model="mistralai/ministral-14b-2512",
        supports_json_schema=False,
    ),
    # Lightweight DeepSeek option
    "openrouter_deepseek_r1_distill_llama_8b": ModelSpec(
        key="openrouter_deepseek_r1_distill_llama_8b",
        provider="openrouter",
        model="deepseek/deepseek-r1-distill-llama-8b",
        supports_json_schema=False,
    ),
}


def list_model_keys() -> List[str]:
    return sorted(DEFAULT_MODELS.keys())


def get_model_spec(
    model_key: str,
    *,
    openrouter_model: Optional[str] = None,
    supports_json_schema: Optional[bool] = None,
) -> ModelSpec:
    """Get a model spec.

    - For OpenRouter keys, you can override the model slug via openrouter_model.
    - You can force supports_json_schema on/off (default: False everywhere here).
    """
    if model_key not in DEFAULT_MODELS:
        keys = ", ".join(list_model_keys())
        raise KeyError(f"Unknown model_key={model_key}. Available: {keys}")

    spec = DEFAULT_MODELS[model_key]

    if spec.provider == "openrouter" and openrouter_model:
        spec = ModelSpec(
            key=spec.key,
            provider=spec.provider,
            model=openrouter_model,
            deployment_env=spec.deployment_env,
            supports_json_schema=spec.supports_json_schema,
        )

    if supports_json_schema is not None:
        spec = ModelSpec(
            key=spec.key,
            provider=spec.provider,
            model=spec.model,
            deployment_env=spec.deployment_env,
            supports_json_schema=supports_json_schema,
        )

    return spec

