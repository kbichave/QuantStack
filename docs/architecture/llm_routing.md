# LLM Routing

Model selection, provider fallback, and tier assignment for all LangGraph agent nodes.

---

## Tier system

Every agent in `graphs/*/config/agents.yaml` declares an `llm_tier`. The tier resolves to a specific model via the active provider.

| Tier | Purpose | Agents |
|------|---------|--------|
| **heavy** | High-reasoning tasks | fund_manager, quant_researcher, trade_debater, risk_analyst, ml_scientist, strategy_rd, options_analyst |
| **medium** | Planning and monitoring | earnings_analyst, position_monitor, daily_planner, market_intel, trade_reflector, executor, strategy_promoter |
| **light** | Supervisory and scouting | community_intel, execution_researcher, health_monitor, self_healer, scheduled_tasks |
| **embedding** | RAG vector embeddings | Not a chat tier -- use `rag/embeddings.py` |

---

## Provider configs

Each provider maps tiers to specific model identifiers. Defined in `src/quantstack/llm/config.py`.

| Provider | Heavy | Medium | Light | Embedding | Required Env Vars |
|----------|-------|--------|-------|-----------|-------------------|
| **bedrock** (default) | claude-haiku-4-5 | claude-haiku-4-5 | claude-haiku-4-5 | mxbai-embed-large | `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION` |
| **anthropic** | claude-sonnet-4 | claude-sonnet-4 | claude-haiku-4-5 | mxbai-embed-large | `ANTHROPIC_API_KEY` |
| **openai** | gpt-4o | gpt-4o-mini | gpt-4o-mini | text-embedding-3-small | `OPENAI_API_KEY` |
| **gemini** | gemini-2.5-pro | gemini-2.0-flash | gemini-2.0-flash | mxbai-embed-large | `GEMINI_API_KEY` |
| **ollama** | llama3:70b | llama3.2 | llama3.2 | mxbai-embed-large | `OLLAMA_BASE_URL` |

Model strings use `provider/model_id` format (e.g., `bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0`).

---

## Fallback chain

When the primary provider is unavailable (missing env vars), the system tries alternatives.

```
FALLBACK_ORDER = ["bedrock", "anthropic", "openai", "ollama"]
```

**Flow:**
1. Read `LLM_PROVIDER` env var (default: `bedrock`)
2. Validate required env vars for that provider
3. If validation fails and `LLM_FALLBACK_ENABLED=true` (default): try next provider in FALLBACK_ORDER
4. If all providers fail: raise `AllProvidersFailedError`

Fallback logs a warning so degraded mode is visible in LangFuse traces.

---

## Model instantiation

`get_chat_model(tier)` is the main entry point. Called by graph builders.

```
get_chat_model("heavy")
    -> get_model_with_fallback("heavy")    # resolve model string via fallback
    -> parse "provider/model_id"           # split on first "/"
    -> ModelConfig(provider, model_id, tier)
    -> _instantiate_chat_model(config)     # provider-specific ChatModel class
    -> attach Langfuse callback handler    # per-call tracing (best-effort)
    -> return ChatModel
```

Provider-specific imports are deferred (not all packages installed in every env):
- bedrock: `langchain_aws.ChatBedrock`
- anthropic: `langchain_anthropic.ChatAnthropic`
- openai: `langchain_openai.ChatOpenAI`
- gemini: `langchain_google_genai.ChatGoogleGenerativeAI`
- ollama: `langchain_ollama.ChatOllama`

All models use `temperature=0.0` (deterministic) and `max_tokens=4096`.

---

## Adding a new provider

1. Add a `ProviderConfig` entry to `PROVIDER_CONFIGS` in `src/quantstack/llm/config.py`
2. Add required env vars to `REQUIRED_ENV_VARS`
3. Add provider name to `FALLBACK_ORDER` in `src/quantstack/llm/provider.py`
4. Add instantiation branch in `_instantiate_chat_model()` with deferred import

---

## Key files

| File | Purpose |
|------|---------|
| `src/quantstack/llm/config.py` | Provider configs, tier definitions, env var requirements |
| `src/quantstack/llm/provider.py` | Fallback chain, model instantiation, Langfuse attachment |
| `src/quantstack/rag/embeddings.py` | Embedding model config (separate from chat tiers) |
