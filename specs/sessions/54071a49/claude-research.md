# Research Findings — QuantStack 24/7 Readiness

**Date:** 2026-04-07
**Sources:** Codebase exploration + web research on implementation patterns

---

## Part 1: Codebase Research

### 1. Agent Executor (`src/quantstack/graphs/agent_executor.py`)

- **Three-mode tool loading**: Anthropic-native tool binding, LangGraph bigtool, and full-load fallback
- **`parse_json_response()`** with DLQ: On parse failure, silently returns `{}`. This is the LC2 critical finding — needs Pydantic schema validation + retry
- **Priority-based message pruning** (P0-P3): Already exists for context management, but no compaction at merge points
- **`build_system_message()`**: Has `cache_control` parameter support — infrastructure for prompt caching is partially in place, just not wired up
- **Modification hooks**: Schema validation goes at `parse_json_response()` (line ~474-521). Prompt caching wiring goes at `build_system_message()` (line ~152-178)

### 2. Risk Gate (`src/quantstack/execution/risk_gate.py`)

- **12+ sequential checks** in `check()`: daily loss, position caps, gross exposure, sector concentration, etc.
- **`RiskVerdict` / `RiskViolation`** data structures for structured rejection reasons
- **`monitor()`** runs every 60s for position-level health
- **Already has** `_check_pretrade_correlation()`, `_check_heat_budget()`, `_check_sector_concentration()` — but correlation is post-hoc only (H1), heat budget exists but daily notional cap doesn't (H3)
- **Greeks are NOT integrated** — options path only checks DTE + premium, not delta/gamma/vega/theta (QS-E3)
- **Modification hooks**: Pre-trade correlation gate goes at `check()`. Greeks integration goes in the options path. Intraday circuit breaker goes at `monitor()`

### 3. Graph Builders (`src/quantstack/graphs/{trading,research,supervisor}/graph.py`)

- **All accept** `ConfigWatcher` + `BaseCheckpointSaver` parameters
- **All compile** with `checkpointer=` parameter — MemorySaver currently passed
- **`RetryPolicy`** configured on all graphs (already handles transient LLM failures at graph level)
- **Modification hooks**: Swap `MemorySaver` → `AsyncPostgresSaver` at compilation point. Mode-aware routing goes in the graph builder's conditional edges

### 4. Tool Binding (`src/quantstack/tools/tool_binding.py`)

- **Three binding paths**: Anthropic native, bigtool (for large tool sets), full loading
- **Tool ordering is NOT deterministic** — uses dict iteration order from `TOOL_REGISTRY`. This is MC1 — breaks prompt cache on every request
- **Fix is a 1-line `sorted()` call** on the tool list before binding
- **92 of 122 tools are stubs** — agents waste LLM calls invoking tools that return errors (TC1)

### 5. EventBus (`src/quantstack/coordination/event_bus.py`)

- **Append-only PostgreSQL event log** with per-consumer cursors
- **Poll-based** consumption (not push) — explains the 5-10 min Supervisor→Trading latency (GC2)
- **7-day TTL** on events with automatic cleanup
- **`ACK_REQUIRED_EVENTS`** list + `check_missed_acks()` with 3 severity tiers
- **Trading Graph does NOT poll EventBus** — `safety_check` node exists but doesn't call `bus.poll()` (AC1)
- **Kill switch doesn't publish events** — triggers locally only (AC2)
- **Modification hooks**: Add `bus.poll()` call in Trading Graph's `safety_check` node (~5-10 lines). Add event publish in `kill_switch.trigger()` (~3 lines)

### 6. Kill Switch (`src/quantstack/execution/kill_switch.py`)

- **Two-layer**: Sentinel file + PostgreSQL flag (singleton pattern)
- **`AutoTriggerMonitor`** with 4 conditions: daily loss, consecutive failures, data staleness, broker disconnect
- **Thread-safe** with locks
- **No auto-recovery** — once triggered, requires manual `reset()` call (OH3)
- **No event publication** — triggers silently, other systems don't learn about it (AC2)

### 7. LLM Provider (`src/quantstack/llm/provider.py`)

- **`_instantiate_chat_model()`**: Handles bedrock, anthropic, openai, gemini, ollama, groq (just added)
- **`FALLBACK_ORDER`**: `["bedrock", "anthropic", "openai", "groq", "ollama", "bedrock_groq"]`
- **Prompt caching constants defined** (`BEDROCK_PROMPT_CACHING_BETA`, `CACHE_CONTROL_EPHEMERAL`) but not wired into model instantiation
- **No runtime failover** — provider checked at startup only. Mid-session 429/500 crashes the cycle (LH2)

### 8. Testing Setup

- **pytest configured** in `pyproject.toml` with `pythonpath = ["src"]`, `testpaths = ["tests", "src/quantstack/core/tests"]`
- **`asyncio_mode = "auto"`** — async tests supported
- **Markers**: `integration` (requires real DB), `regression` (behavioral contracts)
- **Coverage omissions**: Long list of external-dependency modules (brokers, streaming, RL, GPU, etc.)
- **Tests exist** in `tests/` and `src/quantstack/core/tests/` — need to verify coverage of safety-critical paths

### 9. Signal Cache (`src/quantstack/signal_engine/cache.py`)

- **TTL-based** cache with 3600s default (1 hour)
- **`get()`, `put()`, `invalidate()`, `clear()`, `stats()`** methods
- **Not hooked into scheduled_refresh** — intraday refresh every 5 min but cache holds stale data for 1 hour (DC1)
- **Fix**: Call `cache.invalidate(symbol)` at end of each `scheduled_refresh` cycle

### 10. Docker Compose (`docker-compose.yml`)

- **9+ services defined**: postgres, pgvector, langfuse, ollama, 3 graph services, scheduler (port 8422)
- **Observability stack defined but not fully wired**: loki, fluent-bit, grafana, cadvisor, prometheus
- **Scheduler** already has a service definition — just needs health check + restart policy (OC2)
- **No backup service** — needs pg_dump sidecar or cron (OC1)

---

## Part 2: Web Research

### Topic 1: LangGraph PostgresSaver Migration (Phase 1.8)

**Key Findings:**

- **`AsyncPostgresSaver`** is the recommended checkpoint saver for production LangGraph deployments
- Requires `langgraph-checkpoint-postgres` package + `psycopg[pool]` (already in pyproject.toml)
- **`setup()` is idempotent** — safe to call on every startup, creates tables if they don't exist
- **Crash recovery at super-step boundaries** — graph resumes from the last completed super-step after restart
- **Migration is drop-in** from MemorySaver: same `checkpointer=` parameter at graph compilation

**Implementation Pattern:**
```python
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool

pool = AsyncConnectionPool(conninfo=TRADER_PG_URL, min_size=2, max_size=5)
checkpointer = AsyncPostgresSaver(pool)
await checkpointer.setup()  # idempotent

graph = builder.compile(checkpointer=checkpointer)
```

**Migration Strategy:**
- Run MemorySaver + PostgresSaver in parallel for 1 week (log both, read from Memory)
- Cut over to PostgresSaver after verifying checkpoint consistency
- The `thread_id` config parameter is how graph runs are identified for recovery

**Caveats:**
- Each graph should get its own connection pool (don't share across graphs)
- Pool sizing: 2-5 connections per graph is sufficient
- Checkpoint data can grow — add periodic cleanup for old thread data

### Topic 2: Anthropic/Bedrock Prompt Caching (Phase 1.3, 1.4)

**Key Findings:**

- **Cache prefix order is: tools → system messages → user messages** (in that exact order)
- **Tool ordering changes invalidate the ENTIRE cache** — this is why MC1 (deterministic tool ordering) is a prerequisite for caching
- **Minimum token thresholds**: 1024 tokens for Haiku, 2048 for Sonnet, 4096 for Opus
- **5-minute TTL** that auto-refreshes on cache hit (rolling window)
- **90% cost reduction** on cached input tokens ($3.00/MTok → $0.30/MTok for Sonnet input)

**Implementation Pattern:**
```python
# System message with cache_control breakpoint
SystemMessage(content=system_prompt, additional_kwargs={
    "cache_control": {"type": "ephemeral"}
})

# For Bedrock: pass beta header
model = ChatBedrockConverse(
    model="us.anthropic.claude-sonnet-4-6",
    model_kwargs={"anthropic_beta": ["prompt-caching-2024-07-31"]}
)
```

**Critical Ordering:**
1. Fix tool ordering FIRST (MC1 — sorted alphabetically)
2. Then add cache_control to system messages
3. Then add cache_control to tool definitions
4. Measure cache hit rate in Langfuse before/after

**Caveats:**
- Dynamic content (user state, recent prices) should be AFTER the cache breakpoint
- Only the first cache_control breakpoint in each message type is honored
- Groq/Llama models do NOT support prompt caching — only applies to Anthropic/Bedrock models (heavy tier)

### Topic 3: LLM Circuit Breaker / Runtime Failover (Phase 2.1)

**Key Findings:**

- **LiteLLM Router** is the recommended approach for multi-provider failover:
  - Priority-based routing (primary → fallback)
  - Per-provider cooldown on errors
  - Automatic retry with exponential backoff
  - Already a dependency (`litellm>=1.82.2` in pyproject.toml)

- **Alternative**: LangChain's `with_fallbacks()` — simpler but less configurable:
  ```python
  model = primary_model.with_fallbacks([fallback_model_1, fallback_model_2])
  ```

**Implementation Pattern (LiteLLM Router):**
```python
from litellm import Router

router = Router(
    model_list=[
        {"model_name": "heavy", "litellm_params": {"model": "bedrock/claude-sonnet-4-6"}, "priority": 1},
        {"model_name": "heavy", "litellm_params": {"model": "anthropic/claude-sonnet-4-6"}, "priority": 2},
    ],
    retry_after=300,  # 5-min cooldown
    num_retries=2,
    fallbacks=[{"heavy": ["heavy"]}],  # failover to same tier, different provider
)
```

**Recommended Approach for QuantStack:**
- Use LangChain `with_fallbacks()` for simplicity (already in LangChain ecosystem)
- Wrap in a `CircuitBreaker` class that tracks per-provider health state
- On 429/500/timeout: retry same provider 2x → switch to next in FALLBACK_ORDER → cooldown failed provider 5 min
- Health state stored in-memory (no need for DB — provider health is transient)

**Error Classification:**
- Retryable: 429 (rate limit), 500 (server error), timeout
- Non-retryable: 400 (bad request), 401 (auth), 403 (forbidden)
- Provider-down: 3 consecutive retryable errors → cooldown

### Topic 4: Pydantic Structured Outputs for Agent Responses (Phase 1.2)

**Key Findings:**

- **`with_structured_output(method="json_schema")`** gives guaranteed schema compliance from Claude
- **`include_raw=True`** for debugging — returns both raw LLM output and parsed result
- **Field descriptions are critical for Claude** — Claude uses them to understand what to put in each field

**Implementation Pattern:**
```python
from pydantic import BaseModel, Field

class DailyPlanOutput(BaseModel):
    plan: str = Field(description="High-level trading plan for the day")
    candidates: list[str] = Field(description="Ticker symbols to evaluate for entry")
    risk_level: str = Field(description="One of: conservative, moderate, aggressive")

structured_model = model.with_structured_output(DailyPlanOutput, include_raw=True)
result = structured_model.invoke(messages)
```

**For QuantStack's Agent Executor:**
- Define Pydantic models for each critical agent output (daily_plan, entry_scan, position_review, fund_manager, risk_sizing)
- On parse failure: retry once with schema hint appended to prompt
- On second failure: log warning + DLQ event, return schema-compliant default (NOT `{}`)
- Structured outputs replace the fragile `parse_json_response()` regex parsing

**Caveats:**
- `with_structured_output()` adds ~100ms latency per call (schema enforcement overhead)
- Not all providers support it equally — Anthropic/Bedrock excellent, Groq/Llama may need `method="json_mode"` fallback
- For Groq models: test structured output compatibility; may need `method="json_mode"` instead of `"json_schema"`
- Keep schemas simple — deeply nested schemas increase failure rate

---

## Key Implementation Insights

### Dependency Graph (what must happen first)

1. **MC1 (tool ordering)** → prerequisite for **MC0c (prompt caching)**
2. **PostgresSaver (1.8)** → prerequisite for **crash recovery** in all subsequent phases
3. **Pydantic schemas (1.2)** → should be done early as it changes the agent executor interface
4. **EventBus wiring (1.9)** → prerequisite for **inter-graph urgency (3.8)**
5. **CI/CD (1.10)** → prerequisite for safe iteration on everything else

### Risk Areas Identified

- **Groq structured output compatibility**: Llama models may not support `json_schema` mode as reliably as Claude. Need fallback to `json_mode` or raw JSON parsing for medium/light tier agents.
- **Prompt caching + dynamic state**: System prompts contain agent persona (static, cacheable) + current state (dynamic, not cacheable). Need clear separation.
- **PostgresSaver pool sizing**: 3 graphs × 5 connections = 15 connections from checkpointing alone. Ensure PG `max_connections` accommodates this + existing connection usage.
- **Tool registry split (TC1)**: 92 stubbed tools being removed will change agent behavior. Some agents may have been "trained" (via prompt) to use tools that become unavailable. Need to update system prompts alongside registry cleanup.
