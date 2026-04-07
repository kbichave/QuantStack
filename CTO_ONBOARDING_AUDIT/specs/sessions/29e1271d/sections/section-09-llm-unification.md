# Section 9: LLM Provider Unification

## Goal

Ensure every LLM call in the codebase routes through the unified provider layer (`quantstack.llm.provider`), eliminate remaining hardcoded model strings, introduce a database-backed configuration table with a clear precedence chain (env var > DB > code default), and add provider health tracking to the supervisor graph.

## Dependencies

- **Section 01 (DB Schema):** The `llm_config` table must be created in `ensure_tables()` before this section's DB-read path works. If Section 01 is not yet implemented, the `get_llm_config()` function must gracefully fall back to code defaults when the table does not exist.

## Background

QuantStack has two LLM configuration modules with overlapping responsibilities:

1. **`src/quantstack/llm/config.py` + `src/quantstack/llm/provider.py`** — The canonical provider layer. Defines `ProviderConfig` per provider (bedrock, anthropic, openai, gemini, ollama, groq), tier resolution (`heavy/medium/light/bulk/embedding`), fallback chains, and `get_chat_model()` which returns LangChain ChatModel instances. All three LangGraph graphs (research, trading, supervisor) use this via agent YAML `llm_tier` fields resolved through `get_chat_model(cfg.llm_tier, ...)`.

2. **`src/quantstack/llm_config.py`** — A deprecated legacy module (emits `DeprecationWarning` on import). Contains its own provider availability checks, model string builders, and `get_llm_for_agent()` / `get_llm_for_role()` functions. Still imported by some call sites.

Additionally, `src/quantstack/llm_router.py` wraps `litellm.Router` for load-balanced bulk/reasoning calls, delegating to `get_model_for_role()` when no router config is set.

The problem: configuration is scattered across env vars, code defaults in two different modules, and agent YAML files. There is no runtime-changeable DB layer, no single source of truth, and at least one call site (`memory/mem0_client.py`) contains a hardcoded model string (`"gpt-4o-mini"`) outside the provider layer entirely.

## Tests

Write tests in `tests/unit/test_llm_unification.py` and `tests/integration/test_llm_unification.py`.

### Unit Tests

```python
# tests/unit/test_llm_unification.py

# Test: no hardcoded model strings remain outside llm_config.py and llm/provider.py
#
# Strategy: Use ast.parse + ast.walk to scan all .py files under src/quantstack/
# for string literals matching known model name patterns (claude, gpt-4, llama,
# qwen, gemini, mistral). Allowlist: llm/config.py, llm/provider.py,
# observability/cost_queries.py (price table), tools/models.py (created_by field
# containing "claude_code" is not a model string). Fail if any other file contains
# a match.

# Test: get_llm_config precedence — env var overrides DB, DB overrides code default
#
# Setup: patch DB to return {"provider": "openai", "model": "gpt-4o-mini"} for tier "heavy".
# Set env var LLM_TIER_HEAVY="anthropic/claude-sonnet-4-6".
# Assert get_llm_config("heavy") returns the env var value, not the DB value.
# Remove env var, assert get_llm_config("heavy") returns the DB value.
# Remove DB row, assert get_llm_config("heavy") returns the code default from PROVIDER_CONFIGS.

# Test: get_llm_config returns code default when no env var and no DB row
#
# Ensure no env var is set for the tier and no DB row exists.
# Assert the returned config matches the existing PROVIDER_CONFIGS[provider].heavy value.

# Test: get_llm_config returns DB row when present and no env var
#
# Insert a row into llm_config with tier="heavy", provider="openai", model="gpt-4o".
# Ensure no env var override is set.
# Assert get_llm_config("heavy") returns the DB values.

# Test: get_llm_config returns env var when present (ignores DB and default)
#
# Insert a DB row AND set an env var. Assert the env var wins.

# Test: llm_config table schema matches expected columns
#
# After ensure_tables(), inspect the llm_config table columns.
# Assert columns: tier (PK or unique), provider, model, fallback_order, updated_at.
```

### Integration Tests

```python
# tests/integration/test_llm_unification.py

# Test: changing llm_config DB row changes subsequent get_chat_model() output
#
# Insert a row for tier="light" pointing to a different provider/model.
# Call get_chat_model("light") and verify the returned ChatModel uses the DB-configured model.
# Update the row to a different model. Call again. Verify the change is reflected.
# (This confirms there is no stale caching preventing runtime changes.)

# Test: all agent YAML configs resolve to valid tiers via get_llm_for_agent()
#
# Load all agents.yaml files from research/config, trading/config, supervisor/config.
# For each agent entry with an llm_tier field, call get_chat_model(llm_tier) and
# assert it does not raise ValueError (i.e., the tier is recognized).
# Agent YAML paths:
#   src/quantstack/graphs/research/config/agents.yaml
#   src/quantstack/graphs/trading/config/agents.yaml
#   src/quantstack/graphs/supervisor/config/agents.yaml
```

## Implementation

### 9.1 Audit and Remove Remaining Hardcoded Model Strings

Phase 5.6 removed most hardcoded strings, but a targeted audit is still needed. The current state based on a grep of `src/quantstack/`:

**Known hardcoded model strings to fix:**

| File | Line | Current Value | Fix |
|------|------|---------------|-----|
| `src/quantstack/memory/mem0_client.py` | ~120 | `"model": "gpt-4o-mini"` | Replace with `get_model_for_role("bulk")` or make the Mem0 config read from the provider layer. Mem0 expects a provider/model dict, so build it from the resolved model string. |
| `src/quantstack/observability/cost_queries.py` | 17-27 | Price-per-token lookup table with model name keys | **Leave as-is.** This is a cost reference table, not a model selection call. The keys must match actual model names for cost calculation. Add a comment clarifying this is intentional. |

**Grep patterns to run for verification:**

Search all `.py` files under `src/quantstack/` (excluding `llm/config.py`, `llm/provider.py`, `llm_config.py`, `observability/cost_queries.py`) for these patterns:
- String literals containing: `claude-`, `gpt-4`, `gpt-3`, `llama`, `qwen`, `gemini-`, `mistral-`, `text-embedding`
- Direct imports of provider SDKs (`import anthropic`, `import openai`, `from groq import`) outside of `llm/provider.py` and `_instantiate_chat_model`

Any matches (other than the allowlisted cost table) should be replaced with calls through the provider layer.

**Deprecated module (`llm_config.py`) call sites:**

The legacy `quantstack.llm_config` module emits a deprecation warning on import. Grep for `from quantstack.llm_config import` and `from quantstack import llm_config` across the codebase. Each call site should be migrated to `quantstack.llm.provider` equivalents:
- `get_llm_for_agent(name)` -> `get_chat_model(agent_cfg.llm_tier)` (if in a graph context) or `get_model_for_role(tier)` (if getting a string)
- `get_llm_for_role(role)` -> `get_model_for_role(role)`
- `log_llm_config_summary()` -> Build equivalent using the new provider layer, or keep as thin wrapper

Once all call sites are migrated, the deprecated module can remain with its warning (removal is a separate cleanup task).

### 9.2 Database-Backed Configuration Table

**New table: `llm_config`**

Add to `src/quantstack/db.py` in the `ensure_tables()` function:

```sql
CREATE TABLE IF NOT EXISTS llm_config (
    tier         TEXT PRIMARY KEY,
    provider     TEXT NOT NULL,
    model        TEXT NOT NULL,
    fallback_order TEXT,          -- comma-separated provider names, nullable
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

No default rows are inserted. When the table is empty, the system falls back to the existing `PROVIDER_CONFIGS` code defaults. This is deliberate: the DB layer is an override mechanism, not the source of truth for defaults. Defaults live in code where they are version-controlled and reviewable.

**Configuration resolution function:**

Add to `src/quantstack/llm/provider.py`:

```python
async def get_llm_config(tier: str) -> dict:
    """Resolve LLM config for a tier using three-level precedence.

    Precedence (highest to lowest):
        1. Environment variable: LLM_TIER_{TIER} (e.g., LLM_TIER_HEAVY=anthropic/claude-sonnet-4-6)
        2. Database row in llm_config table for this tier
        3. Code default from PROVIDER_CONFIGS[current_provider]

    Returns:
        dict with keys: provider, model, fallback_order

    The env var format is "provider/model" (same as LiteLLM model strings).
    The DB row stores provider and model separately.
    The code default is derived from the current LLM_PROVIDER env var and PROVIDER_CONFIGS.
    """
```

Implementation notes:
- The env var naming convention is `LLM_TIER_{TIER}` (e.g., `LLM_TIER_HEAVY`, `LLM_TIER_MEDIUM`). This is distinct from the legacy `LLM_MODEL_IC` / `LLM_MODEL_POD` vars which map to roles, not tiers.
- DB reads should be lightweight. Use a short TTL cache (e.g., 60 seconds via `@lru_cache` with a timestamp check, or `cachetools.TTLCache`) so that runtime changes propagate within a minute without hitting the DB on every LLM call.
- If the `llm_config` table does not exist (e.g., `ensure_tables()` has not run yet), catch the `psycopg2.ProgrammingError` and fall through to code defaults. Log a warning once.
- The existing `get_model_with_fallback()` and `get_chat_model()` functions should be updated to call `get_llm_config()` internally, replacing the current direct `PROVIDER_CONFIGS` lookup. This keeps the public API unchanged while adding the DB layer underneath.

**Wiring into existing functions:**

Modify `get_model_with_fallback(tier)` to:
1. Call `get_llm_config(tier)` to get the resolved provider/model
2. If the result came from env or DB, use it directly (skip the fallback chain since the user explicitly chose it)
3. If the result came from code defaults, proceed with the existing fallback chain logic

This ensures that explicit overrides (env/DB) are respected without the fallback chain second-guessing them, while code defaults still get the safety net of fallback.

### 9.3 Provider Health Tracking

Add a provider health check to the supervisor graph's `health_check` node. This is a lightweight liveness probe, not a load test.

**New function in `src/quantstack/llm/provider.py`:**

```python
async def check_provider_health() -> dict[str, dict]:
    """Ping each configured provider with a minimal completion request.

    Returns a dict keyed by provider name:
        {
            "bedrock": {"status": "ok", "latency_ms": 340, "checked_at": "..."},
            "groq": {"status": "error", "error": "timeout", "checked_at": "..."},
        }

    Only checks providers that have valid credentials configured (skips unconfigured ones).
    Uses a tiny prompt ("Say 'ok'") with max_tokens=5 to minimize cost.
    Timeout: 10 seconds per provider.
    """
```

**Integration with supervisor:**

The supervisor graph's `health_check` node (in `src/quantstack/graphs/supervisor/nodes.py`) already runs periodic health checks. Add a call to `check_provider_health()` in that node. If the primary provider for any active tier is down and a fallback is being used, emit a system alert with:
- Category: `service_failure`
- Severity: `warning` (or `critical` if all providers for a tier are down)
- Title: `"LLM provider {name} unavailable — using fallback {fallback_name}"`

This depends on Section 02 (system alerts) being implemented. If system alerts are not yet available, log the warning via the standard logger and skip the alert emission.

**Health check frequency:** Once per supervisor cycle (300 seconds). Do not check more frequently -- LLM pings have cost and rate-limit implications.

### 9.4 Files to Create or Modify

| File | Action | What |
|------|--------|------|
| `src/quantstack/db.py` | Modify | Add `llm_config` table to `ensure_tables()` |
| `src/quantstack/llm/provider.py` | Modify | Add `get_llm_config()`, `check_provider_health()`, update `get_model_with_fallback()` to use DB layer |
| `src/quantstack/memory/mem0_client.py` | Modify | Replace hardcoded `"gpt-4o-mini"` with provider layer call |
| `src/quantstack/observability/cost_queries.py` | Modify | Add comment clarifying the price table is intentionally hardcoded |
| `src/quantstack/graphs/supervisor/nodes.py` | Modify | Add `check_provider_health()` call to health_check node |
| `tests/unit/test_llm_unification.py` | Create | Unit tests per spec above |
| `tests/integration/test_llm_unification.py` | Create | Integration tests per spec above |

### 9.5 What This Section Does NOT Cover

- **Removing the deprecated `llm_config.py` module entirely.** That is a separate cleanup task. This section ensures all new code paths use the canonical provider layer and that no new hardcoded strings are introduced.
- **LiteLLM Router configuration.** The `llm_router.py` module already delegates to `get_model_for_role()` and is not affected by the DB layer addition.
- **Agent YAML `llm_tier` field changes.** The existing tier assignments in agent YAML files are correct and route through `get_chat_model()`. No changes needed unless a tier needs rebalancing (operational decision, not a code change).
- **Embedding model unification.** The `embedding` tier in `PROVIDER_CONFIGS` already exists. Mem0's embedding config is a special case handled in 9.1.

## Verification Checklist

1. Run the hardcoded-string audit test. Zero violations outside allowlisted files.
2. With no `llm_config` DB rows and no tier env vars set, all existing behavior is unchanged (code defaults win).
3. Insert a `llm_config` row for tier `light` pointing to a different model. Verify `get_chat_model("light")` returns the new model.
4. Set `LLM_TIER_LIGHT=openai/gpt-4o-mini` env var. Verify it overrides both DB and code default.
5. Remove the env var. Verify DB row is now used.
6. Remove the DB row. Verify code default is restored.
7. Supervisor health check logs provider status without errors.
8. All agent YAML tiers resolve without `ValueError`.
