"""LLM provider selection with fallback chain.

Usage:
    from quantstack.llm.provider import get_model, get_model_with_fallback, get_model_for_role

    model = get_model("heavy")  # returns e.g. "bedrock/anthropic.claude-sonnet-4-..."
    model = get_model_with_fallback("heavy")  # tries fallback chain on failure
    model_str = get_model_for_role("bulk")  # returns model string for litellm callers
"""

import asyncio
import logging
import os
import time
from datetime import datetime, timezone

from quantstack.llm.config import (
    BEDROCK_PROMPT_CACHING_BETA,
    CHAT_TIERS,
    PROMPT_CACHING_ENABLED_DEFAULT,
    PROVIDER_CONFIGS,
    REQUIRED_ENV_VARS,
    VALID_TIERS,
    ModelConfig,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Legacy tier aliases — maps old IC/Pod/Workshop names to standard tiers
# ---------------------------------------------------------------------------

TIER_ALIASES: dict[str, str] = {
    "ic": "light",
    "pod": "heavy",
    "assistant": "heavy",
    "decoder": "light",
    "workshop": "heavy",
    "autonomous_pm": "heavy",
}

# Legacy env var overrides — checked before standard tier resolution
_LEGACY_ENV_OVERRIDES: dict[str, str] = {
    "ic": "LLM_MODEL_IC",
    "pod": "LLM_MODEL_POD",
    "assistant": "LLM_MODEL_ASSISTANT",
    "decoder": "LLM_MODEL_DECODER",
    "workshop": "LLM_MODEL_WORKSHOP",
    "autonomous_pm": "LLM_MODEL_AUTONOMOUS_PM",
    "bulk": "LLM_MODEL_BULK",
    "research": "LLM_MODEL_RESEARCH",
}


def _normalize_tier(tier: str) -> str:
    """Normalize tier name: lowercase, resolve aliases.

    Logs a warning when a legacy alias is used.
    """
    lower = tier.lower()
    if lower in TIER_ALIASES:
        canonical = TIER_ALIASES[lower]
        logger.debug(
            "Legacy tier '%s' resolved to '%s' via alias", lower, canonical,
        )
        return canonical
    return lower


class ConfigurationError(Exception):
    """Required environment variables missing for a provider."""


class AllProvidersFailedError(Exception):
    """Every provider in the fallback chain failed validation."""


FALLBACK_ORDER: list[str] = ["bedrock", "anthropic", "openai", "groq", "ollama", "bedrock_groq"]


def _validate_provider(provider: str) -> None:
    """Check that required env vars are set for the given provider.

    Raises ConfigurationError with details on missing vars.
    """
    required = REQUIRED_ENV_VARS.get(provider, [])
    missing = [v for v in required if not os.environ.get(v)]
    if missing:
        raise ConfigurationError(
            f"Provider '{provider}' requires env vars: {', '.join(missing)}"
        )


def get_model(tier: str) -> str:
    """Return the model string for a given tier.

    Reads LLM_PROVIDER from environment (default: "bedrock").
    Validates required env vars. Returns the model identifier for the tier.

    Accepts both standard tiers (heavy/medium/light/bulk/embedding) and legacy
    aliases (ic/pod/assistant/decoder/workshop). Legacy names are resolved via
    TIER_ALIASES.

    Raises:
        ValueError: If tier is not recognized.
        ConfigurationError: If required env vars are missing.
        KeyError: If LLM_PROVIDER names an unknown provider.
    """
    resolved = _normalize_tier(tier)
    if resolved not in VALID_TIERS:
        logger.warning("Rejected unrecognized tier '%s' — no naming-convention fallback", tier)
        raise ValueError(
            f"Unknown tier '{tier}'. Valid tiers: {sorted(VALID_TIERS)}"
        )

    provider = os.environ.get("LLM_PROVIDER", "bedrock")
    if provider not in PROVIDER_CONFIGS:
        raise KeyError(f"Unknown LLM provider '{provider}'")

    _validate_provider(provider)
    return getattr(PROVIDER_CONFIGS[provider], resolved)


def get_model_with_fallback(tier: str) -> str:
    """Return model string, falling back through providers on validation failure.

    Tries the primary provider first, then remaining providers from FALLBACK_ORDER.
    Only active when LLM_FALLBACK_ENABLED=true (default).

    Raises:
        AllProvidersFailedError: When every provider fails validation.
    """
    tier = _normalize_tier(tier)
    if tier not in VALID_TIERS:
        logger.warning("Rejected unrecognized tier '%s' — no naming-convention fallback", tier)
        raise ValueError(
            f"Unknown tier '{tier}'. Valid tiers: {sorted(VALID_TIERS)}"
        )

    fallback_enabled = os.environ.get("LLM_FALLBACK_ENABLED", "true").lower() == "true"
    if not fallback_enabled:
        return get_model(tier)

    primary = os.environ.get("LLM_PROVIDER", "bedrock")
    attempt_order = [primary] + [p for p in FALLBACK_ORDER if p != primary]

    errors: list[str] = []
    for provider in attempt_order:
        if provider not in PROVIDER_CONFIGS:
            errors.append(f"{provider}: unknown provider")
            continue
        try:
            _validate_provider(provider)
            model = getattr(PROVIDER_CONFIGS[provider], tier)
            if provider != primary:
                logger.warning(
                    "Primary provider '%s' unavailable, using fallback '%s'",
                    primary, provider,
                )
            return model
        except ConfigurationError as exc:
            errors.append(f"{provider}: {exc}")
            continue

    raise AllProvidersFailedError(
        f"All providers failed for tier '{tier}': " + "; ".join(errors)
    )


def get_model_for_role(role: str) -> str:
    """Return a model string for a named role, for use with litellm.completion().

    Handles both standard tiers (heavy/medium/light/bulk) and legacy aliases
    (ic/pod/assistant/decoder/workshop). Also supports special routing for
    "bulk" (Groq preferred) and "research" (heavy tier, fall back to bulk).

    Args:
        role: One of the standard tiers, legacy aliases, or special roles
              ("bulk", "research").

    Returns:
        LiteLLM model string (e.g. "groq/qwen/qwen3-32b").

    Raises:
        ValueError: If role is not recognized.
    """
    lower = role.lower()

    # Check for env var override first (legacy or new)
    env_var = _LEGACY_ENV_OVERRIDES.get(lower)
    if env_var:
        override = os.environ.get(env_var)
        if override:
            logger.debug("Role '%s': env override %s → %s", role, env_var, override)
            return override

    # Special routing: bulk prefers Groq for fast/cheap inference
    if lower == "bulk":
        if os.environ.get("GROQ_API_KEY"):
            try:
                _validate_provider("groq")
                return getattr(PROVIDER_CONFIGS["groq"], "bulk")
            except (ConfigurationError, KeyError):
                pass
        # Fall back to light tier on current provider
        logger.debug("bulk: Groq unavailable, falling back to light tier")
        return get_model_with_fallback("light")

    # Special routing: research uses heavy tier, falls back to bulk
    if lower == "research":
        try:
            return get_model_with_fallback("heavy")
        except AllProvidersFailedError:
            logger.debug("research: no heavy provider, falling back to bulk")
            return get_model_for_role("bulk")

    # Standard tier or alias
    resolved = _normalize_tier(lower)
    if resolved in VALID_TIERS:
        return get_model_with_fallback(resolved)

    raise ValueError(
        f"Unknown role '{role}'. Valid roles: "
        f"{sorted(set(list(VALID_TIERS) + list(TIER_ALIASES) + ['bulk', 'research']))}"
    )


def _instantiate_chat_model(config: ModelConfig):
    """Instantiate the appropriate LangChain ChatModel for the given config.

    Provider-specific imports are deferred because not all provider packages
    are installed in every environment.
    """
    provider = config.provider

    if config.thinking and provider not in ("anthropic",):
        logger.debug(
            "Provider '%s' does not support extended thinking; ignoring thinking config",
            provider,
        )

    if provider == "bedrock":
        try:
            from langchain_aws import ChatBedrock
        except ImportError:
            raise ImportError(
                "Provider 'bedrock' requires langchain-aws. "
                "Install it with: pip install langchain-aws"
            )
        model_kwargs = {"temperature": config.temperature, "max_tokens": config.max_tokens}
        kwargs: dict = {
            "model_id": config.model_id,
            "region_name": os.environ.get("AWS_DEFAULT_REGION", os.environ.get("BEDROCK_REGION", "us-east-1")),
            "model_kwargs": model_kwargs,
        }
        if config.prompt_caching:
            model_kwargs["anthropic_beta"] = [BEDROCK_PROMPT_CACHING_BETA]
        return ChatBedrock(**kwargs)

    if provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError:
            raise ImportError(
                "Provider 'anthropic' requires langchain-anthropic. "
                "Install it with: pip install langchain-anthropic"
            )
        anth_kwargs: dict = {
            "model": config.model_id,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
        }
        if config.prompt_caching:
            anth_kwargs["model_kwargs"] = anth_kwargs.get("model_kwargs", {})
            anth_kwargs["model_kwargs"]["extra_headers"] = {
                "anthropic-beta": BEDROCK_PROMPT_CACHING_BETA
            }
        if config.thinking:
            thinking_config = {**config.thinking}
            if "budget_tokens" not in thinking_config:
                thinking_config["budget_tokens"] = 5000
            anth_kwargs["thinking"] = thinking_config
        return ChatAnthropic(**anth_kwargs)

    if provider == "openai":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError(
                "Provider 'openai' requires langchain-openai. "
                "Install it with: pip install langchain-openai"
            )
        return ChatOpenAI(
            model=config.model_id,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )

    if provider == "gemini":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError:
            raise ImportError(
                "Provider 'gemini' requires langchain-google-genai. "
                "Install it with: pip install langchain-google-genai"
            )
        return ChatGoogleGenerativeAI(
            model=config.model_id,
            temperature=config.temperature,
            max_output_tokens=config.max_tokens,
        )

    if provider == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except ImportError:
            raise ImportError(
                "Provider 'ollama' requires langchain-ollama. "
                "Install it with: pip install langchain-ollama"
            )
        return ChatOllama(
            model=config.model_id,
            temperature=config.temperature,
        )

    if provider == "groq":
        try:
            from langchain_groq import ChatGroq
        except ImportError:
            raise ImportError(
                "Provider 'groq' requires langchain-groq. "
                "Install it with: pip install langchain-groq"
            )
        return ChatGroq(
            model=config.model_id,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )

    raise ValueError(f"Unsupported provider '{provider}' for chat model instantiation")


def get_chat_model(
    tier: str,
    thinking: dict | None = None,
    temperature: float | None = None,
    prompt_caching: bool | None = None,
):
    """Return a configured LangChain ChatModel for the given tier.

    Uses the same provider resolution and fallback logic as get_model_with_fallback().
    Parses the provider prefix from the model string to select the ChatModel class.

    Accepts legacy aliases (ic/pod/workshop/etc.) — they are resolved via TIER_ALIASES.

    Args:
        tier: LLM tier ("heavy", "medium", "light", "bulk") or legacy alias.
        thinking: Optional thinking config, e.g. {"type": "adaptive"}.
            Only supported for Anthropic provider. Silently ignored for others.
        temperature: Optional temperature override. If None, uses default (0.0).

    Raises:
        ValueError: If tier is 'embedding' or unrecognized.
    """
    tier = _normalize_tier(tier)
    if tier == "embedding":
        raise ValueError(
            "The 'embedding' tier is not a chat model. "
            "Use the embedding interface in rag/embeddings.py instead."
        )
    if tier not in CHAT_TIERS:
        raise ValueError(
            f"Unknown tier '{tier}'. Valid chat tiers: {sorted(CHAT_TIERS)}"
        )

    # LiteLLM proxy path: if LITELLM_PROXY_URL is set, return a ChatOpenAI
    # pointed at the proxy. The tier name is used directly as the model name
    # because litellm_config.yaml defines model groups matching tier names.
    proxy_url = os.environ.get("LITELLM_PROXY_URL")
    if proxy_url:
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError(
                "LiteLLM proxy mode requires langchain-openai. "
                "Install it with: pip install langchain-openai"
            )
        kwargs: dict = {
            "model": tier,
            "openai_api_base": f"{proxy_url.rstrip('/')}/v1",
            "openai_api_key": "not-needed",
            "temperature": temperature if temperature is not None else 0.0,
            "max_tokens": 8192 if thinking else 4096,
        }
        if thinking:
            kwargs["model_kwargs"] = {"thinking": thinking}
        return ChatOpenAI(**kwargs)

    # Resolve prompt caching: explicit arg > env var > default
    if prompt_caching is None:
        prompt_caching = (
            os.environ.get("PROMPT_CACHING_ENABLED", str(PROMPT_CACHING_ENABLED_DEFAULT)).lower()
            != "false"
        )

    model_string = get_model_with_fallback(tier)
    if "/" in model_string:
        provider, model_id = model_string.split("/", 1)
    else:
        provider = os.environ.get("LLM_PROVIDER", "bedrock")
        model_id = model_string

    # Only Anthropic-based providers support prompt caching
    caching_enabled = prompt_caching and provider in ("bedrock", "anthropic")
    if prompt_caching and provider not in ("bedrock", "anthropic"):
        logger.debug("Provider '%s' does not support prompt caching; ignoring", provider)

    max_tokens = 8192 if thinking else 4096
    config = ModelConfig(
        provider=provider,
        model_id=model_id,
        tier=tier,
        max_tokens=max_tokens,
        temperature=temperature if temperature is not None else 0.0,
        thinking=thinking,
        prompt_caching=caching_enabled,
    )
    return _instantiate_chat_model(config)


# ---------------------------------------------------------------------------
# Database-backed LLM config (Section 09 — LLM Unification)
# ---------------------------------------------------------------------------

# TTL cache: stores (timestamp, result) per tier
_llm_config_cache: dict[str, tuple[float, dict | None]] = {}
_LLM_CONFIG_TTL = 60  # seconds


def _read_llm_config_from_db(tier: str) -> dict | None:
    """Read LLM config for a tier from the llm_config table.

    Returns dict with keys provider, model, fallback_order or None if no row.
    Catches DB errors gracefully (table may not exist yet).
    """
    try:
        from quantstack.db import db_conn

        with db_conn() as conn:
            row = conn.execute(
                "SELECT provider, model, fallback_order FROM llm_config WHERE tier = %s",
                (tier,),
            ).fetchone()
            if row is None:
                return None
            return {
                "provider": row[0],
                "model": row[1],
                "fallback_order": row[2],
            }
    except Exception as exc:
        logger.debug("llm_config DB read failed (table may not exist): %s", exc)
        return None


def get_llm_config(tier: str) -> dict:
    """Resolve LLM config for a tier using three-level precedence.

    Precedence (highest to lowest):
        1. Environment variable: LLM_TIER_{TIER} (format: "provider/model")
        2. Database row in llm_config table
        3. Code default from PROVIDER_CONFIGS

    Returns:
        dict with keys: provider, model, source ("env", "db", or "code")
    """
    resolved = _normalize_tier(tier)
    if resolved not in VALID_TIERS:
        raise ValueError(f"Unknown tier '{tier}'. Valid tiers: {sorted(VALID_TIERS)}")

    # 1. Env var override: LLM_TIER_HEAVY, LLM_TIER_MEDIUM, etc.
    env_key = f"LLM_TIER_{resolved.upper()}"
    env_val = os.environ.get(env_key)
    if env_val:
        if "/" in env_val:
            prov, model = env_val.split("/", 1)
        else:
            prov = os.environ.get("LLM_PROVIDER", "bedrock")
            model = env_val
        return {"provider": prov, "model": model, "source": "env"}

    # 2. DB override (with TTL cache)
    now = time.monotonic()
    cached = _llm_config_cache.get(resolved)
    if cached and (now - cached[0]) < _LLM_CONFIG_TTL:
        db_row = cached[1]
    else:
        db_row = _read_llm_config_from_db(resolved)
        _llm_config_cache[resolved] = (now, db_row)

    if db_row is not None:
        return {
            "provider": db_row["provider"],
            "model": db_row["model"],
            "source": "db",
        }

    # 3. Code default
    provider = os.environ.get("LLM_PROVIDER", "bedrock")
    if provider not in PROVIDER_CONFIGS:
        provider = "bedrock"
    model_string = getattr(PROVIDER_CONFIGS[provider], resolved)
    if "/" in model_string:
        prov, model = model_string.split("/", 1)
    else:
        prov = provider
        model = model_string
    return {"provider": prov, "model": model, "source": "code"}


# ---------------------------------------------------------------------------
# Provider health check (Section 09 — LLM Unification)
# ---------------------------------------------------------------------------


async def check_provider_health() -> dict[str, dict]:
    """Ping each configured provider with a minimal completion request.

    Returns dict keyed by provider name:
        {"bedrock": {"status": "ok", "latency_ms": 340, "checked_at": "..."}, ...}

    Only checks providers with valid credentials. Uses a tiny prompt to minimize cost.
    Timeout: 10 seconds per provider.
    """
    results: dict[str, dict] = {}
    checked_at = datetime.now(timezone.utc).isoformat()

    for provider_name in PROVIDER_CONFIGS:
        try:
            _validate_provider(provider_name)
        except (ConfigurationError, Exception):
            # Provider not configured — skip
            results[provider_name] = {
                "status": "skipped",
                "reason": "not configured",
                "checked_at": checked_at,
            }
            continue

        try:
            model_string = PROVIDER_CONFIGS[provider_name].light
            if "/" in model_string:
                _, model_id = model_string.split("/", 1)
            else:
                model_id = model_string

            config = ModelConfig(
                provider=provider_name,
                model_id=model_id,
                tier="light",
                max_tokens=5,
                temperature=0.0,
            )
            chat_model = _instantiate_chat_model(config)

            start = time.monotonic()
            from langchain_core.messages import HumanMessage

            response = await asyncio.wait_for(
                chat_model.ainvoke([HumanMessage(content="Say 'ok'")]),
                timeout=10.0,
            )
            latency_ms = int((time.monotonic() - start) * 1000)

            results[provider_name] = {
                "status": "ok",
                "latency_ms": latency_ms,
                "checked_at": checked_at,
            }
        except asyncio.TimeoutError:
            results[provider_name] = {
                "status": "error",
                "error": "timeout",
                "checked_at": checked_at,
            }
        except Exception as exc:
            results[provider_name] = {
                "status": "error",
                "error": str(exc),
                "checked_at": checked_at,
            }

    return results
