# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
LiteLLM Router — named-model routing for bulk and reasoning tasks.

Reads LITELLM_ROUTER_CONFIG from the environment (set in .env) and builds
a litellm.Router that maps logical model names ("reasoning", "bulk") to
actual provider/model strings.

Most call sites should use get_model_for_role() from llm.provider directly.
Use this module when you need litellm.Router semantics: load balancing across
multiple replicas of the same model, per-model rate limits, or retry logic
that differs by model alias.

Usage:
    from quantstack.llm_router import router_completion

    response = router_completion(
        model="bulk",
        messages=[{"role": "user", "content": "summarize this"}],
        max_tokens=256,
    )

Environment:
    LITELLM_ROUTER_CONFIG  JSON string or path to YAML file defining the
                           model_list and routing_strategy. If unset, falls
                           back to direct litellm.completion via get_model_for_role.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any

import litellm
from loguru import logger

from quantstack.llm.provider import get_model_for_role


def _load_router_config() -> list[dict] | None:
    """Parse LITELLM_ROUTER_CONFIG from env. Returns model_list or None."""
    raw = os.getenv("LITELLM_ROUTER_CONFIG", "").strip()
    if not raw:
        return None

    # JSON string inline
    if raw.startswith("{"):
        try:
            cfg = json.loads(raw)
            return cfg.get("model_list")
        except json.JSONDecodeError as exc:
            logger.warning(f"[llm_router] LITELLM_ROUTER_CONFIG JSON parse error: {exc}")
            return None

    # Path to YAML file
    if os.path.isfile(raw):
        try:
            import yaml  # pyyaml; already in requirements via pyyaml dep
            with open(raw) as fh:
                cfg = yaml.safe_load(fh)
            return cfg.get("model_list") if isinstance(cfg, dict) else None
        except Exception as exc:
            logger.warning(f"[llm_router] LITELLM_ROUTER_CONFIG YAML load error: {exc}")
            return None

    logger.debug(
        f"[llm_router] LITELLM_ROUTER_CONFIG is neither JSON nor a valid file path — "
        "falling back to direct litellm.completion"
    )
    return None


@lru_cache(maxsize=1)
def _get_router() -> "litellm.Router | None":
    """Build and cache the LiteLLM Router. Returns None if config is missing."""
    model_list = _load_router_config()
    if not model_list:
        return None

    try:
        # Expand ${VAR} references in api_key fields
        _expand_env_refs(model_list)
        router = litellm.Router(model_list=model_list)
        logger.info(
            f"[llm_router] Router initialized with {len(model_list)} model(s): "
            f"{[m.get('model_name') for m in model_list]}"
        )
        return router
    except Exception as exc:
        logger.warning(f"[llm_router] Router init failed: {exc} — falling back to direct calls")
        return None


def _expand_env_refs(model_list: list[dict]) -> None:
    """Expand ${VAR} references in litellm_params.api_key fields in-place."""
    import re
    _pat = re.compile(r"\$\{([^}]+)\}")

    def _expand(val: str) -> str:
        return _pat.sub(lambda m: os.getenv(m.group(1), ""), val)

    for entry in model_list:
        params = entry.get("litellm_params", {})
        if "api_key" in params and isinstance(params["api_key"], str):
            params["api_key"] = _expand(params["api_key"])


def router_completion(model: str, messages: list[dict], **kwargs: Any) -> Any:
    """
    Call litellm via the configured Router if available, else via get_model_for_role.

    Args:
        model: Logical model name ("reasoning", "bulk") or a full LiteLLM model
               string ("groq/llama-3.3-70b-versatile"). If the Router is active,
               named aliases are resolved by it; full strings are passed through.
        messages: Chat message list.
        **kwargs: Forwarded to litellm.completion / router.completion.

    Returns:
        litellm ModelResponse.
    """
    router = _get_router()

    if router is not None:
        # Router is live — use it for load balancing + retry logic
        return router.completion(model=model, messages=messages, **kwargs)

    # Fallback: resolve via get_model_for_role if model is a logical alias,
    # otherwise pass straight to litellm.completion.
    resolved = model
    if model in ("reasoning", "research"):
        try:
            resolved = get_model_for_role("research")
        except Exception:
            resolved = get_model_for_role("bulk")
    elif model == "bulk":
        resolved = get_model_for_role("bulk")

    return litellm.completion(model=resolved, messages=messages, **kwargs)


__all__ = ["router_completion", "_get_router"]
