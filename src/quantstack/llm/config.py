"""LLM provider configuration — model tier map per provider."""

from dataclasses import dataclass

CHAT_TIERS = frozenset(("heavy", "medium", "light"))


@dataclass(frozen=True)
class ModelConfig:
    """Immutable config for instantiating a LangChain ChatModel."""

    provider: str
    model_id: str
    tier: str
    max_tokens: int = 4096
    temperature: float = 0.0
    thinking: dict | None = None


@dataclass(frozen=True)
class ProviderConfig:
    """Model identifiers for each reasoning tier within a single LLM provider.

    Tiers:
        heavy  — fund-manager, quant-researcher, trade-debater, risk,
                 ml-scientist, strategy-rd, options-analyst
        medium — earnings-analyst, position-monitor, daily-planner,
                 market-intel, trade-reflector
        light  — community-intel, execution-researcher, supervisor
        embedding — memory, RAG
    """
    heavy: str
    medium: str
    light: str
    embedding: str


PROVIDER_CONFIGS: dict[str, ProviderConfig] = {
    "bedrock": ProviderConfig(
        heavy="bedrock/us.anthropic.claude-sonnet-4-6",
        medium="bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
        light="bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
        embedding="ollama/mxbai-embed-large",
    ),
    "anthropic": ProviderConfig(
        heavy="anthropic/claude-sonnet-4-6",
        medium="anthropic/claude-sonnet-4-6",
        light="anthropic/claude-haiku-4-5",
        embedding="ollama/mxbai-embed-large",
    ),
    "openai": ProviderConfig(
        heavy="openai/gpt-4o",
        medium="openai/gpt-4o-mini",
        light="openai/gpt-4o-mini",
        embedding="openai/text-embedding-3-small",
    ),
    "gemini": ProviderConfig(
        heavy="gemini/gemini-2.5-pro",
        medium="gemini/gemini-2.0-flash",
        light="gemini/gemini-2.0-flash",
        embedding="ollama/mxbai-embed-large",
    ),
    "ollama": ProviderConfig(
        heavy="ollama/llama3:70b",
        medium="ollama/llama3.2",
        light="ollama/llama3.2",
        embedding="ollama/mxbai-embed-large",
    ),
}

VALID_TIERS = frozenset(("heavy", "medium", "light", "embedding"))

REQUIRED_ENV_VARS: dict[str, list[str]] = {
    "bedrock": ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_DEFAULT_REGION"],
    "anthropic": ["ANTHROPIC_API_KEY"],
    "openai": ["OPENAI_API_KEY"],
    "gemini": ["GEMINI_API_KEY"],
    "ollama": ["OLLAMA_BASE_URL"],
}
