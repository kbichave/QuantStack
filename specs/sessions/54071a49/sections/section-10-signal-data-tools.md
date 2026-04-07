# Section 10: Signal & Data Quality Gates

## Overview

This section addresses four problems in signal reliability, data freshness, tool lifecycle management, and per-agent LLM temperature configuration:

1. **Signal cache holds stale data** (finding DC1) -- the signal cache has a 1-hour default TTL while intraday refresh runs every 5 minutes. Fresh data arrives but the cache continues serving stale briefs.
2. **Collectors compute on arbitrarily stale data** (finding DC3) -- no freshness gate exists before running collectors. If data refresh fails silently, collectors produce signals from hours-old data without warning.
3. **92 of 122 tools are stubbed** (finding TC1) -- agents bind to tools that are not yet implemented, wasting context window and confusing the LLM.
4. **All agents use the same temperature** -- hypothesis generation and creative research agents should use higher temperature, while execution agents (fund_manager, risk_sizing) must use temperature=0.0 for deterministic decisions.

## Dependencies

- **Section 03 (Tool Ordering & Caching):** The tool registry split relies on the deterministic tool ordering established in section 03. `get_tools_for_agent()` in `registry.py` already sorts tools alphabetically -- this section builds on that.
- No other section dependencies. Phase 1 (sections 01-06) must be complete before starting.

## Current State Assessment

Before implementing, note that significant progress has already been made on several of these items:

**Cache invalidation (partially done):** `scheduled_refresh.py` already calls `signal_cache.invalidate(sym)` after each successful data fetch in `run_intraday_refresh()` (lines 111, 132, 155) and calls `signal_cache.clear()` after EOD refresh (line 282). **This item may be complete.** Verify by checking that every code path that writes fresh data also invalidates the cache.

**Tool registry split (done):** `registry.py` already implements `ACTIVE_TOOLS`, `PLANNED_TOOLS`, and `DEGRADED_TOOLS` registries with a `classify_tools()` function that reads from `tool_manifest.yaml`. The `get_tools_for_agent()` function raises `KeyError` for planned/degraded tools and returns only active tools sorted alphabetically.

**Per-agent temperature (partially done):** `AgentConfig` already has a `temperature: float | None = None` field (line 29 of `config.py`). All three graph builders already pass `temperature=cfg.temperature` to `get_chat_model()`. The LLM provider layer already respects the temperature parameter. What remains is setting appropriate temperature values in the `agents.yaml` config files.

---

## Tests

### Test File: `tests/signal_engine/test_cache_invalidation.py`

```python
# tests/signal_engine/test_cache_invalidation.py

# Test: scheduled_refresh invalidates cache for refreshed symbol
#   - Mock AlphaVantageClient to return sample data for a symbol
#   - Pre-populate signal cache with a stale brief for that symbol
#   - Run run_intraday_refresh()
#   - Assert signal_cache.get(symbol) returns None (invalidated)

# Test: cache returns fresh data after invalidation + new put
#   - Invalidate a symbol's cache entry
#   - Put a new brief into the cache
#   - Assert get() returns the new brief, not the old one
```

### Test File: `tests/signal_engine/test_data_staleness.py`

```python
# tests/signal_engine/test_data_staleness.py

# Test: collector returns empty result when data staler than threshold
#   - Create a DataStore with data whose last_timestamp is >2x refresh interval old
#   - Run a collector (e.g., collect_technical)
#   - Assert the collector returns {} (empty dict) indicating staleness rejection
#   - Assert a staleness warning is logged

# Test: collector runs normally when data is fresh
#   - Create a DataStore with recent data (within threshold)
#   - Run the same collector
#   - Assert non-empty result is returned

# Test: staleness warning logged when data rejected
#   - Use caplog or loguru capture to verify a warning-level message
#     containing "stale" or "staleness" is emitted when data is rejected
```

### Test File: `tests/tools/test_tool_registry_split.py`

```python
# tests/tools/test_tool_registry_split.py

# Test: ACTIVE_TOOLS contains only working tools
#   - After classify_tools(), verify ACTIVE_TOOLS is non-empty
#   - Verify no tool in ACTIVE_TOOLS has status "planned" in the manifest

# Test: PLANNED_TOOLS contains only stubbed tools
#   - Verify every tool in PLANNED_TOOLS has status "planned" in the manifest

# Test: agent bindings only reference ACTIVE_TOOLS
#   - Load all three agents.yaml files
#   - For each agent, verify every tool in its tools: list is in ACTIVE_TOOLS
#   - If any tool is in PLANNED_TOOLS, the test fails with a clear message
```

### Test File: `tests/graphs/test_agent_temperature.py`

```python
# tests/graphs/test_agent_temperature.py

# Test: execution agents (fund_manager, risk_sizing) have temperature=0.0
#   - Load trading/config/agents.yaml
#   - Assert fund_manager config has temperature == 0.0
#   - Assert risk_assessor config has temperature == 0.0 (or None, which defaults to 0.0)

# Test: hypothesis_generation agent has temperature > 0 (e.g., 0.7)
#   - Load research/config/agents.yaml
#   - Assert quant_researcher config has temperature >= 0.5

# Test: agent_executor reads temperature from agent config
#   - Verify that graph builders pass config.temperature to get_chat_model()
#   - This is already true (see current state) -- test is a regression guard
```

---

## Implementation Details

### Item 1: Signal Cache Auto-Invalidation on Refresh

**Status: Likely already complete.**

The current `scheduled_refresh.py` already calls `signal_cache.invalidate(sym)` after each successful data fetch:
- Line 111: After bulk quotes refresh
- Line 132: After 5-min OHLCV refresh
- Line 155: After news sentiment refresh
- Line 282: `signal_cache.clear()` after full EOD refresh

**Verification task:** Audit all code paths that write fresh market data to the database. Ensure every such path also calls `signal_cache.invalidate(symbol)`. Specifically check:
- `run_eod_refresh()` -- currently calls `clear()` at the end (line 282), which is correct but could be more granular. This is acceptable since EOD is a bulk refresh.
- Any manual data ingestion paths (e.g., `acquisition_pipeline.py`) -- if these exist and write data, they should also invalidate the cache.

**File:** `src/quantstack/data/scheduled_refresh.py` -- likely no changes needed.

### Item 2: Data Staleness Rejection in Collectors

**What to build:** A freshness gate that runs before each collector in `SignalEngine._run_collectors()`. If the underlying data for a symbol is staler than a configurable threshold, the collector should return an empty dict and log a warning instead of computing signals from stale data.

**Approach:** Add a staleness check in `signal_engine/engine.py` within `_run_collectors()`. Before dispatching each collector coroutine, check the data freshness metadata for the symbol. The threshold should be configurable per collector type (technical data needs to be fresher than fundamentals).

**Design considerations:**

- The staleness threshold should be `2x the expected refresh interval` as default. For intraday collectors (technical, volume, flow), this means data older than ~10 minutes is suspect. For daily collectors (fundamentals, events), data older than ~2 days is suspect.
- The check should query `data_metadata.last_timestamp` (or equivalent) from the DataStore. If the DataStore does not expose this metadata, add a method like `get_last_update_timestamp(symbol: str, data_type: str) -> datetime | None`.
- Staleness rejection should be a soft gate: log a warning, return empty dict, increment a staleness counter in the brief's `collector_failures` list. It should NOT raise an exception or halt the engine.
- The staleness thresholds should be configurable via environment variables or a config dict, with sensible defaults.

**Staleness threshold defaults:**

| Collector Category | Expected Refresh | Staleness Threshold |
|-------------------|-----------------|---------------------|
| technical, volume, flow | 5 min (intraday) | 10 min |
| regime, risk, sentiment | 5 min (intraday) | 15 min |
| fundamentals, quality | daily | 48 hours |
| events, earnings_momentum | daily | 48 hours |
| macro, sector, cross_asset | daily | 72 hours |
| ml_signal | on-demand | 24 hours |
| ewf | on-demand | no gate (already graceful) |

**File to modify:** `src/quantstack/signal_engine/engine.py`

Add a helper function and integrate it into `_run_collectors()`:

```python
# Signature only -- implementation details left to implementer
_STALENESS_THRESHOLDS: dict[str, int] = {
    # collector_name -> max_age_seconds
    "technical": 600, "volume": 600, "flow": 600,
    "regime": 900, "risk": 900, "sentiment": 900,
    "fundamentals": 172800, "quality": 172800,
    # ... etc.
}

async def _check_data_freshness(
    symbol: str, collector_name: str, store: DataStore
) -> bool:
    """Return True if data is fresh enough for this collector, False otherwise."""
    ...
```

The `_run_collectors` method should wrap each collector coroutine with a freshness check that short-circuits to `{}` if data is stale.

### Item 3: Tool Registry Split (ACTIVE/PLANNED)

**Status: Already implemented.**

The tool registry in `src/quantstack/tools/registry.py` already has:
- `ACTIVE_TOOLS`, `PLANNED_TOOLS`, `DEGRADED_TOOLS` dictionaries (lines 306-308)
- `classify_tools()` function that reads `tool_manifest.yaml` and partitions tools (lines 331-358)
- `get_tools_for_agent()` that raises `KeyError` for planned/degraded tools (lines 437-469)
- `move_tool()` for runtime lifecycle transitions (lines 361-386)
- `classify_tools()` is called on import (line 390)

**Remaining work:**
1. **Audit `tool_manifest.yaml`** -- verify that all stubbed/non-functional tools are marked as `status: planned`. The manifest file exists at `src/quantstack/tools/tool_manifest.yaml`.
2. **Audit all `agents.yaml` files** -- verify that no agent's `tools:` list references a tool that is marked as `planned` in the manifest. If any do, remove them from the agent config.
3. **Add the top 10-15 most impactful stubs to a prioritized implementation list** -- analyze Langfuse traces for tool call frequency to identify which planned tools agents most often attempt to use.

**Files to audit:**
- `src/quantstack/tools/tool_manifest.yaml`
- `src/quantstack/graphs/trading/config/agents.yaml`
- `src/quantstack/graphs/research/config/agents.yaml`
- `src/quantstack/graphs/supervisor/config/agents.yaml`

### Item 4: Per-Agent Temperature Configuration

**Status: Infrastructure complete, config values needed.**

The full pipeline already works:
- `AgentConfig.temperature` field exists (`config.py` line 29)
- `load_agent_configs()` reads `temperature` from YAML (`config.py` line 131)
- All graph builders pass `temperature=cfg.temperature` to `get_chat_model()`
- `get_chat_model()` passes temperature to the provider (defaults to 0.0 when None)

**What remains:** Set appropriate `temperature` values in each `agents.yaml` file.

**Recommended temperature settings:**

| Agent | Graph | Temperature | Rationale |
|-------|-------|-------------|-----------|
| `fund_manager` | trading | 0.0 | Deterministic approval/rejection decisions |
| `risk_assessor` | trading | 0.0 | Risk sizing must be reproducible |
| `execution_manager` | trading | 0.0 | Order parameters must be deterministic |
| `exit_manager` | trading | 0.0 | Exit decisions must be consistent |
| `daily_planner` | trading | 0.2 | Slight variation for watchlist diversity |
| `entry_scanner` | trading | 0.1 | Low creativity, mostly pattern matching |
| `position_monitor` | trading | 0.0 | Monitoring must be factual |
| `trade_debater` | trading | 0.5 | Wants diverse counter-arguments |
| `trade_reflector` | trading | 0.3 | Some creativity for insight extraction |
| `options_analyst` | trading | 0.1 | Low temp for precise structure analysis |
| `market_intel` | trading | 0.3 | Some creativity for synthesis |
| `earnings_analyst` | trading | 0.2 | Moderate for narrative construction |
| `quant_researcher` | research | 0.7 | High creativity for hypothesis generation |
| `ml_scientist` | research | 0.3 | Moderate for experiment design |
| `strategy_rd` | research | 0.5 | Creative strategy ideation |
| `equity_swing_researcher` | research | 0.5 | Creative pattern discovery |
| `equity_investment_researcher` | research | 0.5 | Creative thesis development |
| `options_researcher` | research | 0.4 | Moderate for structure discovery |
| `execution_researcher` | research | 0.3 | Lower for execution optimization |
| `community_intel` | research | 0.3 | Moderate for synthesis |
| `health_monitor` | supervisor | 0.0 | Factual health reporting |
| `diagnostician` | supervisor | 0.2 | Some reasoning flexibility |
| `self_healer` | supervisor | 0.3 | Creative problem solving |
| `strategy_promoter` | supervisor | 0.1 | Mostly rule-based promotion |

**Files to modify:**
- `src/quantstack/graphs/trading/config/agents.yaml` -- Add `temperature:` to each agent block
- `src/quantstack/graphs/research/config/agents.yaml` -- Add `temperature:` to each agent block
- `src/quantstack/graphs/supervisor/config/agents.yaml` -- Add `temperature:` to each agent block

**Example YAML addition** (for an existing agent block):

```yaml
fund_manager:
  role: "..."
  goal: "..."
  # ... existing fields ...
  temperature: 0.0
```

---

## Implementation Checklist

1. **Verify cache invalidation completeness** -- Audit all data write paths in `scheduled_refresh.py`, `acquisition_pipeline.py`, and any other data ingestion modules. Confirm every write calls `signal_cache.invalidate()`. Document findings. If gaps exist, add the missing invalidation calls.

2. **Implement data staleness gate** -- Add `_check_data_freshness()` to `signal_engine/engine.py`. Integrate into `_run_collectors()` so stale collectors short-circuit to `{}`. Add configurable thresholds. Write tests in `tests/signal_engine/test_data_staleness.py`.

3. **Audit tool manifest** -- Review `tool_manifest.yaml` for accuracy. Verify all non-functional tools are marked `planned`. Cross-reference with `agents.yaml` files to ensure no agent binds to planned tools. Fix any mismatches.

4. **Set per-agent temperatures** -- Add `temperature:` field to every agent in all three `agents.yaml` files using the recommended values above. Write regression tests in `tests/graphs/test_agent_temperature.py`.

5. **Write and run all tests** -- Implement the four test files listed in the Tests section. Ensure all pass before marking this section complete.

## Rollback

- **Cache invalidation:** No rollback needed (additive, no behavior change if already working).
- **Data staleness gate:** Remove the freshness check from `_run_collectors()`. Collectors revert to processing any-age data.
- **Tool manifest audit:** Revert `agents.yaml` and `tool_manifest.yaml` changes.
- **Temperature settings:** Remove `temperature:` lines from `agents.yaml`. All agents revert to default 0.0.
