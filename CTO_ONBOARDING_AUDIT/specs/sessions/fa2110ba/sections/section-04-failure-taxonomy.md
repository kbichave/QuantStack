# Section 04: Failure Mode Taxonomy

## Purpose

Every trade loss currently enters the research queue as a generic `bug_fix` task with binary priority (5 or 7). This makes all losses look identical to the research system — a regime mismatch, a stale data feed, and a genuine thesis failure all receive the same treatment. This section introduces a structured failure classification system so that losses are categorized, prioritized proportionally, and routed to the correct investigation pathway.

## Dependencies

- **section-03-readpoint-wiring** must be complete. Specifically, the StrategyBreaker and trade hooks wiring (Wires 2-4) must be live so that trade close events flow through the hook pipeline where classification is triggered.
- The `strategy_outcomes` table must exist (it does — defined in `src/quantstack/db.py`, line ~910).
- The `research_queue` table must exist with its current schema (defined in `src/quantstack/db.py`, line ~1818).

## What This Section Blocks

- **section-05-loss-aggregation** depends on the `failure_mode` column and the `FailureMode` enum to group and rank losses by category.

---

## Tests First

All tests go in `tests/unit/test_failure_taxonomy.py`.

### FailureMode enum

```python
class TestFailureModeEnum:
    """Verify the enum is well-formed and string-serializable."""

    def test_all_seven_modes_are_valid(self):
        """FailureMode has exactly 7 members: REGIME_MISMATCH, FACTOR_CROWDING,
        DATA_STALE, TIMING_ERROR, THESIS_WRONG, BLACK_SWAN, UNCLASSIFIED."""

    def test_enum_is_str_compatible(self):
        """Each member serializes to its string value (e.g., FailureMode.REGIME_MISMATCH == 'regime_mismatch')."""
```

### Rule-based classifier

```python
class TestRuleBasedClassifier:
    """The deterministic classifier runs first and handles unambiguous cases."""

    def test_regime_mismatch_detected(self):
        """When regime_at_entry != regime_at_exit in strategy_outcomes, classify as REGIME_MISMATCH."""

    def test_data_stale_detected(self):
        """When any key data source timestamp exceeds the freshness threshold at entry time, classify as DATA_STALE."""

    def test_black_swan_detected(self):
        """When loss magnitude > 3 standard deviations from the strategy's historical loss distribution, classify as BLACK_SWAN."""

    def test_timing_error_detected(self):
        """When entry was within 2 bars of a key support/resistance level (from daily plan), classify as TIMING_ERROR."""

    def test_returns_unclassified_when_no_rule_matches(self):
        """When no deterministic rule fires, return UNCLASSIFIED."""

    def test_first_matching_rule_wins(self):
        """Rules are evaluated in priority order. A trade that matches both REGIME_MISMATCH and TIMING_ERROR gets the first match."""
```

### Async LLM fallback

```python
class TestAsyncLLMFallback:
    """UNCLASSIFIED trades get queued for async LLM classification (haiku tier)."""

    def test_unclassified_queued_for_llm(self):
        """When rule-based returns UNCLASSIFIED, an async classification task is enqueued."""

    def test_llm_does_not_block_trade_close(self):
        """The on_trade_close hook returns immediately; LLM classification is fire-and-forget."""

    def test_llm_result_backfills_failure_mode(self):
        """When the async LLM task completes, it UPDATEs strategy_outcomes.failure_mode for the trade."""

    def test_llm_timeout_leaves_unclassified(self):
        """If LLM classification fails or times out, the trade remains UNCLASSIFIED. No retry loop."""
```

### Research queue enhancement

```python
class TestResearchQueuePriority:
    """Priority is now proportional to cumulative impact, not binary."""

    def test_priority_uses_cumulative_loss_and_recency(self):
        """priority = min(9, int(cumulative_loss_30d * recency_weight * 10)),
        where recency_weight = 0.95^days_ago."""

    def test_persistent_pattern_gets_higher_priority(self):
        """A strategy with 5 losses over 10 days gets higher priority than one with 1 loss yesterday."""

    def test_task_type_matches_failure_mode(self):
        """The research_queue INSERT uses the classified failure_mode as task_type, not 'bug_fix'."""

    def test_unclassified_still_queued(self):
        """Losses classified as UNCLASSIFIED still get queued (with task_type='unclassified')."""
```

---

## Implementation Details

### 1. FailureMode Enum

**File:** `src/quantstack/learning/failure_taxonomy.py` (new file)

Define a string enum with 7 members:

```python
class FailureMode(str, Enum):
    REGIME_MISMATCH = "regime_mismatch"
    FACTOR_CROWDING = "factor_crowding"
    DATA_STALE = "data_stale"
    TIMING_ERROR = "timing_error"
    THESIS_WRONG = "thesis_wrong"
    BLACK_SWAN = "black_swan"
    UNCLASSIFIED = "unclassified"
```

The `str` mixin ensures the enum values serialize directly to strings in JSON and SQL without `.value` access. This matters because the value is stored as `TEXT` in PostgreSQL and as a key in research queue context payloads.

### 2. Rule-Based Classifier

**File:** `src/quantstack/learning/failure_taxonomy.py` (same file as enum)

Implement a function with this signature:

```python
def classify_failure(
    realized_pnl_pct: float,
    regime_at_entry: str,
    regime_at_exit: str,
    strategy_id: str,
    symbol: str,
    entry_price: float,
    exit_price: float,
    data_freshness: dict[str, float] | None = None,
    key_levels: list[float] | None = None,
) -> FailureMode:
    """Classify a losing trade using deterministic rules.

    Rules are evaluated in priority order (first match wins):
    1. REGIME_MISMATCH — regime_at_entry != regime_at_exit
    2. DATA_STALE — any data source freshness exceeds threshold
    3. BLACK_SWAN — loss > 3 std devs from strategy's historical distribution
    4. TIMING_ERROR — entry within 2 bars of a key support/resistance level
    5. UNCLASSIFIED — no rule matched

    The historical loss distribution for BLACK_SWAN detection is queried from
    strategy_outcomes (trailing 252 trading days). If fewer than 20 historical
    losses exist, BLACK_SWAN cannot be determined (skip the rule).
    """
```

**Rule evaluation order rationale:** Regime mismatch is checked first because it is the most structurally informative — the entire strategy thesis may have been invalid for the market state. Data staleness is second because it indicates an infrastructure failure, not a strategy failure. Black swan is third because it is magnitude-based and context-independent. Timing error is last because it is the most ambiguous and depends on key level data that may not always be available.

**BLACK_SWAN threshold computation:** Query `strategy_outcomes` for the strategy's realized losses over the trailing 252 trading days. Compute mean and standard deviation. If `abs(realized_pnl_pct - mean) > 3 * std`, classify as BLACK_SWAN. With fewer than 20 historical losses, the distribution estimate is unreliable — skip this rule.

**DATA_STALE detection:** The `data_freshness` parameter is a dict mapping data source names to their staleness in minutes at the time of trade entry. If any source exceeds a threshold (e.g., 60 minutes for intraday, 1440 minutes for daily), classify as DATA_STALE. The thresholds should be defined as constants in the module.

**TIMING_ERROR detection:** The `key_levels` parameter is a list of support/resistance prices from the daily plan. If `entry_price` is within 2 ATR-bars (or a fixed percentage like 0.5%) of any key level, classify as TIMING_ERROR. This catches entries at inflection points where the direction is most uncertain.

### 3. Async LLM Classification for UNCLASSIFIED Trades

**File:** `src/quantstack/learning/failure_taxonomy.py`

Add a function:

```python
async def classify_failure_llm(
    trade_context: dict,
    trade_id: int,
) -> None:
    """Async LLM classification for trades the rule-based classifier couldn't categorize.

    Fires asynchronously from the trade close hook — must never block.
    Uses haiku tier for cost efficiency. Classifies into FACTOR_CROWDING
    or THESIS_WRONG (the two modes that require semantic understanding).

    On success, UPDATEs strategy_outcomes.failure_mode for the given trade_id.
    On failure (timeout, LLM error, invalid response), logs a warning and
    leaves the trade as UNCLASSIFIED.
    """
```

The trade context passed to the LLM includes: entry rationale, market conditions at entry, signal values at entry, realized P&L, holding period, and regime information. The LLM prompt should ask it to choose between FACTOR_CROWDING (the trade was crowded — too many participants on the same side) and THESIS_WRONG (the fundamental thesis was incorrect). If the LLM cannot determine either, it should return UNCLASSIFIED.

**Critical constraint:** This function is called via `asyncio.create_task()` (or equivalent fire-and-forget pattern) from the trade close hook. The hook itself must return immediately. If the system is not running an async event loop (e.g., in tests or synchronous contexts), the LLM classification is silently skipped.

### 4. Schema Change: Add failure_mode Column

**File:** `src/quantstack/db.py`

Add an idempotent column addition to the `_migrate_strategy_outcomes_pg` function:

```python
conn.execute(
    "ALTER TABLE strategy_outcomes "
    "ADD COLUMN IF NOT EXISTS failure_mode TEXT DEFAULT 'unclassified'"
)
```

This follows the existing migration pattern in `db.py` (see the `ALTER TABLE research_queue ADD COLUMN IF NOT EXISTS topic TEXT` example at line ~1847). Existing rows get the default value `'unclassified'`.

### 5. Research Queue Enhancement

**File:** `src/quantstack/hooks/trade_hooks.py`

The current research queue INSERT (lines 118-143) uses hardcoded `task_type='bug_fix'` and binary priority `(7 if realized_pnl_pct < -3.0 else 5)`.

Replace with:

1. Call `classify_failure()` with the trade's parameters to get the `FailureMode`.
2. Store the failure mode on the `strategy_outcomes` row (UPDATE with the classified mode).
3. Compute priority: `priority = min(9, int(cumulative_loss_30d * recency_weight * 10))` where:
   - `cumulative_loss_30d` = sum of absolute losses for this strategy in the trailing 30 days (query `strategy_outcomes`)
   - `recency_weight` = `0.95 ** days_ago` where `days_ago` is days since the most recent loss in the pattern
4. INSERT into `research_queue` with `task_type` set to the failure mode string value instead of `'bug_fix'`.
5. If the failure mode is `UNCLASSIFIED`, fire the async LLM classification.

**research_queue CHECK constraint update:** The existing `research_queue.task_type` column has a CHECK constraint limiting values to `('ml_arch_search', 'rl_env_design', 'bug_fix', 'strategy_hypothesis')`. This must be updated to include the new failure mode values. Add an `ALTER TABLE` migration that drops the old constraint and creates a new one including all 7 failure modes plus the existing 4 task types:

```python
conn.execute("""
    ALTER TABLE research_queue DROP CONSTRAINT IF EXISTS research_queue_task_type_check
""")
conn.execute("""
    ALTER TABLE research_queue ADD CONSTRAINT research_queue_task_type_check
    CHECK (task_type IN (
        'ml_arch_search', 'rl_env_design', 'bug_fix', 'strategy_hypothesis',
        'regime_mismatch', 'factor_crowding', 'data_stale',
        'timing_error', 'thesis_wrong', 'black_swan', 'unclassified'
    ))
""")
```

This migration goes in `_migrate_research_queue_pg()` after the existing table creation.

### 6. Wire Classification into on_trade_close

**File:** `src/quantstack/hooks/trade_hooks.py`

In the `on_trade_close` function, after the raw reflection block (step 1) and before the research queue INSERT (step 2), add the classification step for all losing trades (not just losses > 1%):

```python
# 1.5. Classify failure mode for losing trades
failure_mode = FailureMode.UNCLASSIFIED
if realized_pnl_pct < 0:
    try:
        failure_mode = classify_failure(
            realized_pnl_pct=realized_pnl_pct,
            regime_at_entry=regime_at_entry,
            regime_at_exit=regime_at_exit,
            strategy_id=strategy_id,
            symbol=symbol,
            entry_price=entry_price,
            exit_price=exit_price,
        )
        # Persist to strategy_outcomes
        from quantstack.db import db_conn
        with db_conn() as conn:
            conn.execute(
                "UPDATE strategy_outcomes SET failure_mode = %s "
                "WHERE strategy_id = %s AND symbol = %s AND exit_price = %s "
                "ORDER BY closed_at DESC LIMIT 1",
                [failure_mode.value, strategy_id, symbol, exit_price],
            )
    except Exception as exc:
        logger.warning(f"[hooks] failure classification failed: {exc}")
```

Note: classification runs for all losses (< 0%) to populate the column, but research queue insertion still only triggers for losses > 1% (the existing threshold). This ensures the `failure_mode` column has data for the loss aggregation batch (section-05) even for small losses.

---

## File Summary

| File | Action | What Changes |
|------|--------|-------------|
| `src/quantstack/learning/failure_taxonomy.py` | **Create** | FailureMode enum, classify_failure(), classify_failure_llm() |
| `src/quantstack/db.py` | **Modify** | Add failure_mode column to strategy_outcomes, update research_queue CHECK constraint |
| `src/quantstack/hooks/trade_hooks.py` | **Modify** | Import taxonomy, call classify_failure() in on_trade_close, replace hardcoded bug_fix with classified mode, replace binary priority with cumulative formula |
| `tests/unit/test_failure_taxonomy.py` | **Create** | All tests listed above |

## Rollback

Set all new `failure_mode` values to `'unclassified'` in `strategy_outcomes`. Revert `trade_hooks.py` to use hardcoded `task_type='bug_fix'` and binary priority. The research queue CHECK constraint expansion is backward-compatible (existing values still valid) and does not need to be rolled back.
