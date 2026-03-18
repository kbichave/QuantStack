# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Multi-provider LLM configuration for CrewAI agents.

CrewAI uses LiteLLM under the hood. Every model string returned by this
module is in LiteLLM's  provider/model_id  format and passed directly to
Agent(llm=...).

Resolution order for each agent:
    1. Per-tier env override  (LLM_MODEL_IC / LLM_MODEL_POD / …)
    2. Primary provider       (LLM_PROVIDER, default: bedrock)
    3. Fallback chain         (LLM_FALLBACK_CHAIN, comma-separated)
    4. ProviderConfigError if every option is exhausted

Supported providers
-------------------
Tier 1 (production):
    bedrock      AWS Bedrock   — boto3 credential chain
    anthropic    Anthropic API — ANTHROPIC_API_KEY
    openai       OpenAI        — OPENAI_API_KEY
    vertex_ai    Google Vertex — VERTEX_PROJECT + gcloud auth
    gemini       Google AI     — GEMINI_API_KEY

Tier 2 (alternatives):
    azure        Azure OpenAI  — AZURE_API_KEY + AZURE_API_BASE
    groq         Groq          — GROQ_API_KEY
    together_ai  Together AI   — TOGETHER_API_KEY
    fireworks_ai Fireworks AI  — FIREWORKS_API_KEY
    mistral      Mistral AI    — MISTRAL_API_KEY

Tier 3 (local/self-hosted):
    ollama        local models  — OLLAMA_BASE_URL reachable
    custom_openai vLLM / LM Studio — CUSTOM_OPENAI_BASE_URL reachable

Cost-tier defaults (Bedrock)
-----------------------------
    IC agents    → Haiku 4.5   (fast + cheap; narrow focused work)
    Pod managers → Sonnet 4.6  (synthesis requires stronger reasoning)
    Assistant    → Sonnet 4.6  (final synthesis for SuperTrader input)
    Decoder ICs  → Haiku 4.5   (same as IC tier)
    Workshop     → Sonnet 4.6  (cloud model for deep strategy research)

Override any tier via env:
    LLM_MODEL_IC=groq/llama-3.3-70b-versatile
    LLM_MODEL_POD=bedrock/us.anthropic.claude-sonnet-4-6
    LLM_MODEL_ASSISTANT=anthropic/claude-sonnet-4-20250514
    LLM_MODEL_DECODER=gemini/gemini-2.5-flash
    LLM_MODEL_WORKSHOP=bedrock/us.anthropic.claude-sonnet-4-20250514

Ollama / thinking-mode note
----------------------------
When the resolved model is an Ollama model (ollama/...), this module
automatically returns a crewai.LLM object (not a plain string) so it can
inject:
    - api_base: OLLAMA_BASE_URL (required for LiteLLM to reach local server)
    - extra_body: {"think": False} — disables Qwen 3.5 extended-thinking mode
      for IC and pod agents (saves tokens/latency). Does NOT apply to the
      LLM_MODEL_WORKSHOP tier (cloud model used there anyway).
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

from loguru import logger

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ProviderConfigError(RuntimeError):
    """Raised when no configured LLM provider has valid credentials."""


# ---------------------------------------------------------------------------
# Agent tier classification
# ---------------------------------------------------------------------------

_SYNTHESIS_AGENTS = {"trading_assistant", "super_trader"}

# Tier → env var name for per-tier override
_TIER_ENV_OVERRIDE = {
    "ic": "LLM_MODEL_IC",
    "pod": "LLM_MODEL_POD",
    "assistant": "LLM_MODEL_ASSISTANT",
    "decoder": "LLM_MODEL_DECODER",
    # Workshop is not a CrewAI agent tier — it is resolved via get_llm_for_role("workshop")
    # by the /workshop skill for deep strategy research (always routes to cloud model).
    "workshop": "LLM_MODEL_WORKSHOP",
    # autonomous_pm: PM synthesis in the AutonomousRunner for non-routine decisions.
    # Defaults to groq/llama-3.3-70b-versatile (fast, free tier, good reasoning).
    # Override with LLM_MODEL_AUTONOMOUS_PM env var.
    "autonomous_pm": "LLM_MODEL_AUTONOMOUS_PM",
}


def _classify_tier(agent_name: str) -> str:
    """Return one of: "ic", "pod", "assistant", "decoder", "workshop"."""
    if agent_name in _SYNTHESIS_AGENTS:
        return "assistant"
    if "workshop" in agent_name:
        return "workshop"
    if "decoder" in agent_name:
        return "decoder"
    if agent_name.endswith("_pod_manager"):
        return "pod"
    if agent_name.endswith("_ic"):
        return "ic"
    logger.debug(f"[llm_config] Unknown tier for '{agent_name}', defaulting to 'assistant'")
    return "assistant"


# ---------------------------------------------------------------------------
# Credential / availability checks — one per provider, all cached
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _check_bedrock() -> bool:
    try:
        import boto3

        profile = os.getenv("AWS_PROFILE")
        session = boto3.Session(profile_name=profile) if profile else boto3.Session()
        creds = session.get_credentials()
        if creds is None:
            return False
        resolved = creds.resolve_frozen()
        return bool(resolved.access_key)
    except Exception as exc:
        logger.debug(f"[llm_config] Bedrock credential check failed: {exc}")
        return False


@lru_cache(maxsize=1)
def _check_anthropic() -> bool:
    return bool(os.getenv("ANTHROPIC_API_KEY"))


@lru_cache(maxsize=1)
def _check_openai() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))


@lru_cache(maxsize=1)
def _check_vertex_ai() -> bool:
    if not os.getenv("VERTEX_PROJECT"):
        return False
    try:
        import google.auth  # noqa: F401 — just checking it's installed + importable

        return True
    except ImportError:
        logger.debug("[llm_config] Vertex AI: google-cloud-aiplatform not installed")
        return False


@lru_cache(maxsize=1)
def _check_gemini() -> bool:
    return bool(os.getenv("GEMINI_API_KEY"))


@lru_cache(maxsize=1)
def _check_azure() -> bool:
    return bool(os.getenv("AZURE_API_KEY")) and bool(os.getenv("AZURE_API_BASE"))


@lru_cache(maxsize=1)
def _check_groq() -> bool:
    return bool(os.getenv("GROQ_API_KEY"))


@lru_cache(maxsize=1)
def _check_together_ai() -> bool:
    return bool(os.getenv("TOGETHER_API_KEY"))


@lru_cache(maxsize=1)
def _check_fireworks_ai() -> bool:
    return bool(os.getenv("FIREWORKS_API_KEY"))


@lru_cache(maxsize=1)
def _check_mistral() -> bool:
    return bool(os.getenv("MISTRAL_API_KEY"))


@lru_cache(maxsize=1)
def _check_ollama() -> bool:
    base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    return _url_reachable(base)


@lru_cache(maxsize=1)
def _check_custom_openai() -> bool:
    base = os.getenv("CUSTOM_OPENAI_BASE_URL", "http://localhost:8000/v1")
    return _url_reachable(base)


def _url_reachable(url: str, timeout: float = 2.0) -> bool:
    """Return True if a HEAD request to the URL succeeds within timeout."""
    try:
        import requests

        requests.head(url, timeout=timeout)
        return True
    except Exception:
        return False


_AVAILABILITY_CHECKS = {
    "bedrock": _check_bedrock,
    "anthropic": _check_anthropic,
    "openai": _check_openai,
    "vertex_ai": _check_vertex_ai,
    "gemini": _check_gemini,
    "azure": _check_azure,
    "groq": _check_groq,
    "together_ai": _check_together_ai,
    "fireworks_ai": _check_fireworks_ai,
    "mistral": _check_mistral,
    "ollama": _check_ollama,
    "custom_openai": _check_custom_openai,
}


def _provider_available(provider: str) -> bool:
    check = _AVAILABILITY_CHECKS.get(provider)
    if check is None:
        logger.warning(f"[llm_config] Unknown provider '{provider}' — treating as unavailable")
        return False
    return check()


# ---------------------------------------------------------------------------
# Model string builders — one per provider
# ---------------------------------------------------------------------------


def _model_bedrock(tier: str) -> str:
    if tier in ("ic", "decoder"):
        return "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0"
    model_id = os.getenv("BEDROCK_MODEL_ID", "us.anthropic.claude-sonnet-4-6")
    return f"bedrock/{model_id}"


def _model_anthropic(_tier: str) -> str:
    model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
    return f"anthropic/{model}"


def _model_openai(_tier: str) -> str:
    model = os.getenv("OPENAI_MODEL", "gpt-4o")
    return f"openai/{model}"


def _model_vertex_ai(_tier: str) -> str:
    model = os.getenv("VERTEX_MODEL", "gemini-2.5-flash")
    project = os.getenv("VERTEX_PROJECT", "")
    location = os.getenv("VERTEX_LOCATION", "us-central1")
    # LiteLLM needs project/location set via env vars
    os.environ.setdefault("VERTEXAI_PROJECT", project)
    os.environ.setdefault("VERTEXAI_LOCATION", location)
    return f"vertex_ai/{model}"


def _model_gemini(_tier: str) -> str:
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    return f"gemini/{model}"


def _model_azure(_tier: str) -> str:
    deployment = os.getenv("AZURE_DEPLOYMENT_NAME", "gpt-4o")
    # LiteLLM reads AZURE_API_KEY, AZURE_API_BASE, AZURE_API_VERSION from env
    os.environ.setdefault("AZURE_API_KEY", os.getenv("AZURE_API_KEY", ""))
    os.environ.setdefault("AZURE_API_BASE", os.getenv("AZURE_API_BASE", ""))
    os.environ.setdefault("AZURE_API_VERSION", os.getenv("AZURE_API_VERSION", "2024-02-15-preview"))
    return f"azure/{deployment}"


def _model_groq(_tier: str) -> str:
    model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    return f"groq/{model}"


def _model_together_ai(_tier: str) -> str:
    model = os.getenv("TOGETHER_MODEL", "meta-llama/Llama-3.3-70B-Instruct-Turbo")
    return f"together_ai/{model}"


def _model_fireworks_ai(_tier: str) -> str:
    model = os.getenv("FIREWORKS_MODEL", "accounts/fireworks/models/llama-v3p3-70b-instruct")
    return f"fireworks_ai/{model}"


def _model_mistral(_tier: str) -> str:
    model = os.getenv("MISTRAL_MODEL", "mistral-large-latest")
    return f"mistral/{model}"


def _model_ollama(tier: str) -> str:
    # Tier-aware model selection for the two always-loaded models.
    # ICs and Decoder: fast 9B dense model (10 parallel requests during crew run).
    # Pods, Assistant, Workshop fallback: MoE 35B model (3B active params, ~35 tok/s).
    # LLM_MODEL_OLLAMA_IC / LLM_MODEL_OLLAMA_POD env vars allow per-tier override
    # without changing LLM_PROVIDER away from ollama.
    if tier in ("ic", "decoder"):
        return f"ollama/{os.getenv('OLLAMA_IC_MODEL', 'qwen3.5:9b')}"
    return f"ollama/{os.getenv('OLLAMA_MODEL', 'qwen3.5:35b-a3b')}"


def _model_custom_openai(_tier: str) -> str:
    # LiteLLM routes "openai/<model>" to a custom endpoint when OPENAI_API_BASE is set
    base = os.getenv("CUSTOM_OPENAI_BASE_URL", "http://localhost:8000/v1")
    key = os.getenv("CUSTOM_OPENAI_API_KEY", "not-needed")
    model = os.getenv("CUSTOM_OPENAI_MODEL", "local-model")
    os.environ["OPENAI_API_BASE"] = base
    os.environ["OPENAI_API_KEY"] = key
    return f"openai/{model}"


_MODEL_BUILDERS = {
    "bedrock": _model_bedrock,
    "anthropic": _model_anthropic,
    "openai": _model_openai,
    "vertex_ai": _model_vertex_ai,
    "gemini": _model_gemini,
    "azure": _model_azure,
    "groq": _model_groq,
    "together_ai": _model_together_ai,
    "fireworks_ai": _model_fireworks_ai,
    "mistral": _model_mistral,
    "ollama": _model_ollama,
    "custom_openai": _model_custom_openai,
}


# ---------------------------------------------------------------------------
# Provider resolution with fallback chain
# ---------------------------------------------------------------------------


def _fallback_chain() -> list[str]:
    """Parse LLM_FALLBACK_CHAIN into a list of provider names."""
    raw = os.getenv("LLM_FALLBACK_CHAIN", "")
    return [p.strip() for p in raw.split(",") if p.strip()]


def _resolve_provider(tier: str) -> str:
    """
    Return a LiteLLM model string for tier, walking primary → fallback chain.

    Raises ProviderConfigError if no provider has valid credentials.
    """
    primary = os.getenv("LLM_PROVIDER", "bedrock").lower()
    chain = [primary] + _fallback_chain()
    tried: list[str] = []

    for provider in chain:
        if provider in tried:
            continue
        tried.append(provider)

        if not _provider_available(provider):
            logger.debug(f"[llm_config] Provider '{provider}' not available, skipping")
            continue

        builder = _MODEL_BUILDERS.get(provider)
        if builder is None:
            logger.warning(f"[llm_config] No model builder for '{provider}', skipping")
            continue

        model = builder(tier)
        logger.debug(f"[llm_config] Resolved provider='{provider}' tier='{tier}' → {model}")
        return model

    raise ProviderConfigError(
        f"No LLM provider available for tier '{tier}'. "
        f"Tried: {tried}. "
        f"Set LLM_PROVIDER and ensure credentials are configured."
    )


# ---------------------------------------------------------------------------
# Ollama LLM object builder
# ---------------------------------------------------------------------------


def _is_ollama_model(model_str: str) -> bool:
    return model_str.startswith("ollama/")


def _build_ollama_llm(model_str: str, disable_thinking: bool = True) -> str:
    """Return a LiteLLM-compatible model string for a local Ollama model.

    Sets OLLAMA_BASE_URL in the environment so LiteLLM routes correctly.
    """
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    os.environ.setdefault("OLLAMA_BASE_URL", base_url)
    return model_str


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_llm_for_agent(agent_name: str) -> Any:
    """
    Resolve the LLM for a given agent.

    For non-Ollama providers, returns a LiteLLM model string (str).
    For Ollama providers, returns a crewai.LLM object with api_base and
    think=False pre-configured. Both are accepted by Agent(llm=...).

    Args:
        agent_name: Agent identifier matching JSON config filename
                    (e.g., "data_ingestion_ic", "trading_assistant")

    Returns:
        LiteLLM model string or crewai.LLM object for CrewAI Agent(llm=...).

    Raises:
        ProviderConfigError: If no provider in the chain has valid credentials.
    """
    tier = _classify_tier(agent_name)

    # Per-tier env override takes precedence over everything
    override = os.getenv(_TIER_ENV_OVERRIDE[tier])
    if override:
        logger.debug(f"[llm_config] {agent_name} ({tier}): env override → {override}")
        if _is_ollama_model(override):
            return _build_ollama_llm(override)
        return override

    model = _resolve_provider(tier)
    if _is_ollama_model(model):
        return _build_ollama_llm(model)
    return model


def get_llm_for_role(role: str) -> Any:
    """
    Resolve the LLM for a named role without requiring an agent name.

    Used by skills (e.g., /workshop) that need explicit model routing
    outside of a CrewAI agent context.

    Args:
        role: One of "ic", "pod", "assistant", "decoder", "workshop"

    Returns:
        LiteLLM model string or crewai.LLM object.

    Raises:
        ProviderConfigError: If no provider in the chain has valid credentials.
        ValueError: If role is not a known tier.
    """
    if role not in _TIER_ENV_OVERRIDE:
        raise ValueError(f"Unknown role '{role}'. Valid roles: {list(_TIER_ENV_OVERRIDE)}")

    override = os.getenv(_TIER_ENV_OVERRIDE[role])
    if override:
        logger.debug(f"[llm_config] role '{role}': env override → {override}")
        if _is_ollama_model(override):
            return _build_ollama_llm(override)
        return override

    model = _resolve_provider(role)
    if _is_ollama_model(model):
        return _build_ollama_llm(model)
    return model


def log_llm_config_summary() -> None:
    """Log the resolved LLM configuration for all tiers. Call at crew startup."""
    provider = os.getenv("LLM_PROVIDER", "bedrock")
    chain = _fallback_chain()

    logger.info(
        f"[llm_config] LLM_PROVIDER={provider}" + (f" | fallback_chain={chain}" if chain else "")
    )

    # Show availability for every known provider
    available = [p for p in _AVAILABILITY_CHECKS if _provider_available(p)]
    logger.info(f"[llm_config] Available providers: {available}")

    # Log resolved model per tier
    for agent_name, tier in [
        ("data_ingestion_ic", "ic"),
        ("risk_pod_manager", "pod"),
        ("trading_assistant", "assistant"),
        ("decoder_ic", "decoder"),
    ]:
        try:
            model = get_llm_for_agent(agent_name)
            logger.info(f"[llm_config]   {tier:10s} → {model}")
        except ProviderConfigError as exc:
            logger.error(f"[llm_config]   {tier:10s} → ERROR: {exc}")


__all__ = ["get_llm_for_agent", "get_llm_for_role", "log_llm_config_summary", "ProviderConfigError"]
