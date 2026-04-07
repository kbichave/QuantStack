"""LLM provider configuration — model tier map per provider."""

from dataclasses import dataclass

CHAT_TIERS = frozenset(("heavy", "medium", "light", "bulk"))


@dataclass(frozen=True)
class ModelConfig:
    """Immutable config for instantiating a LangChain ChatModel."""

    provider: str
    model_id: str
    tier: str
    max_tokens: int = 4096
    temperature: float = 0.0
    thinking: dict | None = None
    prompt_caching: bool = False


# --- Prompt caching ---

# Feature flag: set PROMPT_CACHING_ENABLED=false to disable
PROMPT_CACHING_ENABLED_DEFAULT = True

# Bedrock beta header required to enable prompt caching
BEDROCK_PROMPT_CACHING_BETA = "prompt-caching-2024-07-31"

# Cache control marker applied to system prompts and tool definitions
CACHE_CONTROL_EPHEMERAL = {"type": "ephemeral"}

# --- Budget discipline (AR-9) ---
# Per-million-token cost rates (blended input/output average) for budget tracking.
# Costs vary by provider. These reflect the bedrock_groq hybrid defaults:
# heavy=Sonnet, medium=Groq 70B, light/bulk=Groq 8B.
# bedrock-only rates: heavy=9.00, medium=0.75, light=0.75, bulk=0.75.
MODEL_COST_PER_MTOK: dict[str, float] = {
    "light": 0.10,    # Groq 8B-instant ($0.05 in / $0.08 out)
    "medium": 0.64,   # Groq 70B ($0.59 in / $0.79 out)
    "heavy": 9.00,    # Sonnet (Bedrock)
    "bulk": 0.10,     # Groq 8B-instant
}

# Estimated token costs per node complexity for budget_check routing
BUDGET_ESTIMATE_SIMPLE = 5_000     # simple rule-based hypothesis
BUDGET_ESTIMATE_ML = 30_000        # ML model experiment


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
    bulk: str
    embedding: str


PROVIDER_CONFIGS: dict[str, ProviderConfig] = {
    "bedrock": ProviderConfig(
        heavy="bedrock/us.anthropic.claude-sonnet-4-6",
        medium="bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
        light="bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
        bulk="bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
        embedding="ollama/mxbai-embed-large",
    ),
    "anthropic": ProviderConfig(
        heavy="anthropic/claude-sonnet-4-6",
        medium="anthropic/claude-sonnet-4-6",
        light="anthropic/claude-haiku-4-5",
        bulk="anthropic/claude-haiku-4-5",
        embedding="ollama/mxbai-embed-large",
    ),
    "openai": ProviderConfig(
        heavy="openai/gpt-4o",
        medium="openai/gpt-4o-mini",
        light="openai/gpt-4o-mini",
        bulk="openai/gpt-4o-mini",
        embedding="openai/text-embedding-3-small",
    ),
    "gemini": ProviderConfig(
        heavy="gemini/gemini-2.5-pro",
        medium="gemini/gemini-2.0-flash",
        light="gemini/gemini-2.0-flash",
        bulk="gemini/gemini-2.0-flash",
        embedding="ollama/mxbai-embed-large",
    ),
    "ollama": ProviderConfig(
        heavy="ollama/llama3:70b",
        medium="ollama/llama3.2",
        light="ollama/llama3.2",
        bulk="ollama/llama3.2",
        embedding="ollama/mxbai-embed-large",
    ),
    "groq": ProviderConfig(
        heavy="groq/llama-3.3-70b-versatile",
        medium="groq/llama-3.3-70b-versatile",
        light="groq/llama-3.1-8b-instant",
        bulk="groq/llama-3.1-8b-instant",
        embedding="ollama/mxbai-embed-large",
    ),
    # Hybrid: Bedrock Sonnet for heavy reasoning, Groq Llama for operational agents.
    # Use LLM_PROVIDER=bedrock_groq to activate.
    "bedrock_groq": ProviderConfig(
        heavy="bedrock/us.anthropic.claude-sonnet-4-6",
        medium="groq/llama-3.3-70b-versatile",
        light="groq/llama-3.1-8b-instant",
        bulk="groq/llama-3.1-8b-instant",
        embedding="ollama/mxbai-embed-large",
    ),
}

VALID_TIERS = frozenset(("heavy", "medium", "light", "embedding", "bulk"))

REQUIRED_ENV_VARS: dict[str, list[str]] = {
    "bedrock": ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_DEFAULT_REGION"],
    "anthropic": ["ANTHROPIC_API_KEY"],
    "openai": ["OPENAI_API_KEY"],
    "gemini": ["GEMINI_API_KEY"],
    "ollama": ["OLLAMA_BASE_URL"],
    "groq": ["GROQ_API_KEY"],
    "bedrock_groq": ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_DEFAULT_REGION", "GROQ_API_KEY"],
}
