# Section 03: Readpoint Wiring

## Purpose

Six learning modules exist in `src/quantstack/learning/` but have zero consumers in the live system. Losses are recorded but never read downstream -- position sizing, entry decisions, and daily planning behave identically whether the system just had 10 consecutive wins or 10 consecutive losses. This section wires each ghost module into the specific production callsite where its output should influence decisions.

**Dependencies:** Section 01 (persistence migration) and Section 02 (ghost module audit) must be complete. StrategyBreaker and ICAttributionTracker must be persisting to PostgreSQL. OutcomeTracker and SkillTracker formulas must be audited and fixed.

**Blocks:** Sections 04 (failure taxonomy), 05 (loss aggregation), 07 (IC weight adjustment), 08 (signal correlation), and 11 (agent quality).

---

## Tests First

All tests go in `tests/unit/test_readpoint_wiring.py`. Use the existing `mock_settings` fixture for DB mocking. No real DB calls.

### Wire 1: `get_regime_strategies()` tool

```python
class TestGetRegimeStrategies:
    """Wire 1: Replace stub in meta_tools.py with real DB query."""

    async def test_returns_strategies_sorted_by_affinity(self, mock_settings):
        """Query strategies table, filter by regime_affinity JSONB column,
        return sorted descending by affinity score for the given regime."""

    async def test_includes_strategy_status_from_breaker(self, mock_settings):
        """Each returned strategy includes status: ACTIVE, SCALED, or TRIPPED
        sourced from StrategyBreaker.get_scale_factor()."""

    async def test_returns_empty_for_unknown_regime(self, mock_settings):
        """Regime with no matching strategies returns empty list, not error."""

    async def test_filters_out_retired_strategies(self, mock_settings):
        """Strategies with status='retired' in the strategies table are excluded."""
```

### Wire 2: StrategyBreaker in `risk_sizing`

```python
class TestBreakerInRiskSizing:
    """Wire 2: Multiply alpha_signal by breaker scale factor after Kelly computation."""

    async def test_active_strategy_unchanged(self, mock_settings):
        """Factor 1.0 (ACTIVE) leaves alpha_signal unchanged."""

    async def test_scaled_strategy_halved(self, mock_settings):
        """Factor 0.5 (SCALED) halves the alpha_signal amount."""

    async def test_tripped_strategy_zeroed(self, mock_settings):
        """Factor 0.0 (TRIPPED) zeros the alpha_signal -- no order generated."""

    async def test_exception_defaults_to_one(self, mock_settings):
        """If get_scale_factor() raises, default to 1.0 and log error.
        Fail-open because the risk gate is downstream."""

    async def test_factor_above_one_clamped(self, mock_settings):
        """Factor > 1.0 clamped to 1.0 to prevent amplification."""
```

### Wire 3: StrategyBreaker in `execute_entries`

```python
class TestBreakerInExecuteEntries:
    """Wire 3: Gate each order placement on breaker state."""

    async def test_tripped_order_skipped(self, mock_settings):
        """TRIPPED strategy (factor 0.0) skips order entirely."""

    async def test_active_order_placed_normally(self, mock_settings):
        """ACTIVE strategy proceeds to order placement."""

    async def test_skip_logged_with_reason(self, mock_settings):
        """Skipped orders log the breaker reason (drawdown, consecutive losses, etc.)."""
```

### Wire 4: SkillTracker in trade hooks

```python
class TestSkillTrackerInTradeHooks:
    """Wire 4: Call update_agent_skill() in on_trade_close()."""

    async def test_profitable_trade_records_correct(self, mock_settings):
        """Profitable close calls update_agent_skill(agent_name, prediction_correct=True, pnl)."""

    async def test_unprofitable_trade_records_incorrect(self, mock_settings):
        """Losing close calls update_agent_skill(agent_name, prediction_correct=False, pnl)."""

    async def test_agent_name_from_trade_context(self, mock_settings):
        """agent_name extracted from trade's debate_verdict or execution context."""
```

### Wire 5: ICAttribution in signal engine

```python
class TestICAttributionInSignalEngine:
    """Wire 5: Record per-collector signal values after synthesis."""

    async def test_record_called_per_collector(self, mock_settings):
        """After run() completes, ic_attribution.record() called once per collector
        with (symbol, collector_name, signal_value, forward_return=None)."""

    async def test_forward_return_initially_none(self, mock_settings):
        """forward_return is None at synthesis time -- backfilled on trade close."""

    async def test_trade_close_backfills_return(self, mock_settings):
        """on_trade_close() computes realized return and backfills matching
        ICAttribution records for that symbol."""
```

### Wire 6: Trade quality in daily plan

```python
class TestTradeQualityInDailyPlan:
    """Wire 6: Inject rolling trade quality context into daily plan prompt."""

    async def test_rolling_averages_per_dimension(self, mock_settings):
        """Computes 30-trade rolling averages for each of the 6 quality dimensions
        (execution_quality, thesis_accuracy, risk_management, timing_quality,
        sizing_quality, overall_score)."""

    async def test_weakest_dimension_identified(self, mock_settings):
        """The dimension with lowest average score is identified as the weakness."""

    async def test_insufficient_data_omits_context(self, mock_settings):
        """Fewer than 5 scored trades means quality context is omitted from the
        daily plan prompt rather than showing noisy statistics."""
```

---

## Implementation Details

### Wire 1: `get_regime_strategies()` in `src/quantstack/tools/langchain/meta_tools.py`

The current implementation is a stub returning `{"error": "Tool pending implementation"}`. Replace the body with:

1. Import `db_conn` from `quantstack.db` and `StrategyBreaker` from `quantstack.execution.strategy_breaker`.
2. Query the `strategies` table: `SELECT strategy_id, name, status, regime_affinity FROM strategies WHERE status != 'retired'`.
3. The `regime_affinity` column is JSONB containing per-regime affinity scores (e.g., `{"trending_up": 0.85, "ranging": 0.4}`). Filter rows where the requested regime key exists and its value is > 0.
4. For each matching strategy, call `strategy_breaker.get_scale_factor(strategy_id)` to get breaker status. Map factor to label: 1.0 = ACTIVE, 0.5 = SCALED, 0.0 = TRIPPED.
5. Sort by affinity descending. Return JSON list of `{strategy_id, name, affinity, breaker_status}`.

**Key file:** `src/quantstack/tools/langchain/meta_tools.py` -- the `get_regime_strategies` function starting at line 11.

**Error handling:** If the DB query fails, return `{"error": "Failed to query strategies", "strategies": []}` with a logged warning. If StrategyBreaker raises on any individual strategy, mark that strategy's breaker_status as "UNKNOWN" and continue.

### Wire 2: StrategyBreaker in `risk_sizing` in `src/quantstack/graphs/trading/nodes.py`

The `risk_sizing` node is an async function starting at line 517. After the Kelly fraction computation (which produces an `alpha_signal` amount for position sizing), add a StrategyBreaker check.

**Integration point:** After the Kelly/position-size computation completes but before the result is written to state. Approximately after the block that sets the final sizing values.

**Logic:**

1. Import `StrategyBreaker` at module level.
2. Call `breaker_factor = strategy_breaker.get_scale_factor(strategy_id)`.
3. Clamp: `breaker_factor = max(0.0, min(1.0, breaker_factor))`.
4. Multiply: `adjusted_size = computed_size * breaker_factor`.
5. Log: `logger.info(f"StrategyBreaker: {strategy_id} factor={breaker_factor}, size {computed_size} -> {adjusted_size}")`.

**Defensive bounds:** Wrap the `get_scale_factor()` call in a try/except. On any exception, default `breaker_factor = 1.0` and log `logger.error(f"StrategyBreaker.get_scale_factor failed for {strategy_id}, defaulting to 1.0: {e}")`. This is fail-open because the risk gate (`execution/risk_gate.py`) is the downstream safety boundary.

### Wire 3: StrategyBreaker in `execute_entries` in `src/quantstack/graphs/trading/nodes.py`

The `execute_entries` node starts at line 1036. Before placing each order in the order loop, add a breaker check.

**Logic:**

1. For each order about to be placed, call `scale_factor = strategy_breaker.get_scale_factor(strategy_id)`.
2. If `scale_factor == 0.0` (TRIPPED): skip the order entirely. Log `logger.warning(f"Order SKIPPED: strategy {strategy_id} is TRIPPED (breaker reason: {reason})")`.
3. If `scale_factor > 0.0 and scale_factor < 1.0` (SCALED): the sizing reduction already happened in Wire 2. Log `logger.info(f"Order proceeding at SCALED level for {strategy_id} (factor={scale_factor})")`.
4. If `scale_factor == 1.0` (ACTIVE): proceed normally, no extra logging needed.

**Why both Wire 2 and Wire 3?** Wire 2 adjusts sizing (a SCALED strategy gets half-sized positions). Wire 3 is a hard gate (a TRIPPED strategy gets zero orders, period). Both are needed because `risk_sizing` runs before `execute_entries` in the graph -- a TRIPPED strategy's size would already be 0 from Wire 2, but Wire 3 provides an explicit skip with logging rather than submitting a zero-size order.

### Wire 4: SkillTracker in `on_trade_close()` in `src/quantstack/hooks/trade_hooks.py`

The `on_trade_close` function starts at line 71. After the existing `ReflectionManager` call (which logs the trade for reflective analysis), add a SkillTracker call.

**Logic:**

1. Import `SkillTracker` at module level from `quantstack.learning.skill_tracker`.
2. Extract `agent_name` from the trade context. The trade record contains a `debate_verdict` field or an `executing_agent` field -- use whichever is populated. If neither exists, use `"unknown_agent"`.
3. Determine `prediction_correct = (realized_pnl > 0)`.
4. Call `skill_tracker.update_agent_skill(agent_name, prediction_correct, realized_pnl)`.
5. This is fire-and-forget -- do not await or block on the SkillTracker result. If it raises, log a warning and continue. The trade close hook must not fail because of a tracking side-effect.

**Data flow downstream:** Once this wire is active, SkillTracker accumulates per-agent win rates and signal P&L. Section 11 (agent quality) later reads these to generate AGENT_DEGRADATION events.

### Wire 5: ICAttribution in signal engine in `src/quantstack/signal_engine/engine.py`

The `SignalEngine.run()` method starts at line 90. After synthesis completes (after the `RuleBasedSynthesizer` produces the `SymbolBrief`), iterate over each collector's contribution.

**Logic:**

1. Import `ICAttributionTracker` at module level from `quantstack.learning.ic_attribution`.
2. After `run()` finishes synthesis for a symbol, iterate over the collector results (each collector returns its signal value as part of the synthesis input).
3. For each collector: call `ic_attribution.record(symbol=symbol, collector_name=collector.name, signal_value=collector_result.value, forward_return=None)`.
4. `forward_return` is None because the outcome is unknown at signal generation time.

**Backfill hook:** In `on_trade_close()` (same file as Wire 4), after recording the trade outcome, compute the realized return and update the matching ICAttribution records:

1. Query `ic_attribution_data` for rows matching the symbol and a timestamp window around the trade entry.
2. Update `forward_return` with the realized return for those rows.

**Survivorship bias note:** Per-trade backfill only covers traded symbols, not the full signal universe. The nightly `run_ic_computation()` (cross-sectional IC across all signaled symbols) remains the primary IC source for weight adjustments (Section 07). ICAttribution provides supplementary per-trade granularity.

### Wire 6: Trade quality in `daily_plan` in `src/quantstack/graphs/trading/nodes.py`

The `daily_plan` node starts at line 233. It constructs a prompt for the planning agent with market context, open positions, and strategy priorities.

**Logic:**

1. Query `trade_quality_scores` table: compute rolling 30-trade averages per dimension (`execution_quality`, `thesis_accuracy`, `risk_management`, `timing_quality`, `sizing_quality`, `overall_score`).
2. If fewer than 5 scored trades exist, skip the quality context entirely (cold-start guard).
3. Identify the weakest dimension (lowest average score).
4. Append to the daily plan prompt context:
   ```
   Recent trade quality analysis (last 30 trades):
   - Execution quality: {avg}/10
   - Thesis accuracy: {avg}/10
   - Risk management: {avg}/10
   - Timing quality: {avg}/10
   - Sizing quality: {avg}/10
   - Overall: {avg}/10
   Weakest area: {dimension} ({avg}/10). Focus on improving {specific_guidance}.
   ```
5. Map each dimension to specific guidance text (e.g., `timing_quality` -> "entry/exit timing relative to key levels", `sizing_quality` -> "position sizing relative to conviction and volatility").

**DB query pattern:** Use `db_conn()` context manager. Query:
```sql
SELECT
    AVG(execution_quality) as avg_exec,
    AVG(thesis_accuracy) as avg_thesis,
    AVG(risk_management) as avg_risk,
    AVG(timing_quality) as avg_timing,
    AVG(sizing_quality) as avg_sizing,
    AVG(overall_score) as avg_overall,
    COUNT(*) as trade_count
FROM trade_quality_scores
ORDER BY scored_at DESC
LIMIT 30
```

---

## Files Modified

| File | Change |
|------|--------|
| `src/quantstack/tools/langchain/meta_tools.py` | Replace `get_regime_strategies` stub with real DB query + StrategyBreaker status |
| `src/quantstack/graphs/trading/nodes.py` | Add StrategyBreaker check in `risk_sizing` (Wire 2) and `execute_entries` (Wire 3). Add trade quality context in `daily_plan` (Wire 6) |
| `src/quantstack/hooks/trade_hooks.py` | Add SkillTracker call in `on_trade_close` (Wire 4). Add ICAttribution forward_return backfill (Wire 5 backfill) |
| `src/quantstack/signal_engine/engine.py` | Add ICAttribution recording after synthesis (Wire 5) |
| `tests/unit/test_readpoint_wiring.py` | New file: all tests for this section |

---

## Rollback

Revert file changes via git. All six wires are additive code in existing functions. Reverting returns ghost modules to their disconnected state. No schema changes in this section (persistence migration happened in Section 01). Data already written to SkillTracker / ICAttribution by the wires is harmless -- it accumulates but nothing reads it without the downstream sections (07, 08, 11) also being active.
