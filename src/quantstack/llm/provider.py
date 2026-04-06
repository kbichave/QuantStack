"""LLM provider selection with fallback chain.

Usage:
    from quantstack.llm.provider import get_model, get_model_with_fallback

    model = get_model("heavy")  # returns e.g. "bedrock/anthropic.claude-sonnet-4-..."
    model = get_model_with_fallback("heavy")  # tries fallback chain on failure
"""

import logging
import os

from quantstack.llm.config import (
    CHAT_TIERS,
    PROVIDER_CONFIGS,
    REQUIRED_ENV_VARS,
    VALID_TIERS,
    ModelConfig,
)

logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Required environment variables missing for a provider."""


class AllProvidersFailedError(Exception):
    """Every provider in the fallback chain failed validation."""


FALLBACK_ORDER: list[str] = ["bedrock", "anthropic", "openai", "ollama"]


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

    Raises:
        ValueError: If tier is not recognized.
        ConfigurationError: If required env vars are missing.
        KeyError: If LLM_PROVIDER names an unknown provider.
    """
    if tier not in VALID_TIERS:
        raise ValueError(
            f"Unknown tier '{tier}'. Valid tiers: {sorted(VALID_TIERS)}"
        )

    provider = os.environ.get("LLM_PROVIDER", "bedrock")
    if provider not in PROVIDER_CONFIGS:
        raise KeyError(f"Unknown LLM provider '{provider}'")

    _validate_provider(provider)
    return getattr(PROVIDER_CONFIGS[provider], tier)


def get_model_with_fallback(tier: str) -> str:
    """Return model string, falling back through providers on validation failure.

    Tries the primary provider first, then remaining providers from FALLBACK_ORDER.
    Only active when LLM_FALLBACK_ENABLED=true (default).

    Raises:
        AllProvidersFailedError: When every provider fails validation.
    """
    if tier not in VALID_TIERS:
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
        return ChatBedrock(
            model_id=config.model_id,
            region_name=os.environ.get("AWS_DEFAULT_REGION", os.environ.get("BEDROCK_REGION", "us-east-1")),
            model_kwargs={"temperature": config.temperature, "max_tokens": config.max_tokens},
        )

    if provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError:
            raise ImportError(
                "Provider 'anthropic' requires langchain-anthropic. "
                "Install it with: pip install langchain-anthropic"
            )
        kwargs = {
            "model": config.model_id,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
        }
        if config.thinking:
            thinking_config = {**config.thinking}
            if "budget_tokens" not in thinking_config:
                thinking_config["budget_tokens"] = 5000
            kwargs["thinking"] = thinking_config
        return ChatAnthropic(**kwargs)

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

    raise ValueError(f"Unsupported provider '{provider}' for chat model instantiation")


def get_chat_model(tier: str, thinking: dict | None = None):
    """Return a configured LangChain ChatModel for the given tier.

    Uses the same provider resolution and fallback logic as get_model_with_fallback().
    Parses the provider prefix from the model string to select the ChatModel class.

    Args:
        tier: LLM tier ("heavy", "medium", "light").
        thinking: Optional thinking config, e.g. {"type": "adaptive"}.
            Only supported for Anthropic provider. Silently ignored for others.

    Raises:
        ValueError: If tier is 'embedding' or unrecognized.
    """
    if tier == "embedding":
        raise ValueError(
            "The 'embedding' tier is not a chat model. "
            "Use the embedding interface in rag/embeddings.py instead."
        )
    if tier not in CHAT_TIERS:
        raise ValueError(
            f"Unknown tier '{tier}'. Valid chat tiers: {sorted(CHAT_TIERS)}"
        )

    model_string = get_model_with_fallback(tier)
    if "/" in model_string:
        provider, model_id = model_string.split("/", 1)
    else:
        provider = os.environ.get("LLM_PROVIDER", "bedrock")
        model_id = model_string

    max_tokens = 8192 if thinking else 4096
    config = ModelConfig(
        provider=provider,
        model_id=model_id,
        tier=tier,
        max_tokens=max_tokens,
        thinking=thinking,
    )
    return _instantiate_chat_model(config)
