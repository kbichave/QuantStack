# Section 13: Options Refresh Expansion (Item 8.8)

## Background

QuantStack's data pipeline refreshes options chain data during scheduled intraday cycles. The current implementation in `src/quantstack/data/scheduled_refresh.py` refreshes options for watched symbols plus the top 30 universe symbols. The cap of 30 is hardcoded and may miss important symbols during high-activity periods. Additionally, there is no mechanism to ensure options data is fresh before a trading decision that depends on it.

This section makes three targeted improvements:

1. Extract the hardcoded `30` into a configurable environment variable
2. Add strategy-aware symbol selection so that symbols tied to active options strategies are always refreshed
3. Add a pre-trade freshness check that triggers an inline options refresh when stale data would otherwise be used for a trading decision

This section has no dependencies on other sections and can be implemented in parallel with everything else.

---

## Tests (Write First)

All tests go in a new file: `tests/unit/test_options_refresh.py`

```python
# tests/unit/test_options_refresh.py

# --- Configurable top-N ---

# Test: OPTIONS_REFRESH_TOP_N env var controls number of symbols refreshed (default 30)
#   Setup: Do NOT set OPTIONS_REFRESH_TOP_N in env. Mock the refresh function.
#   Assert: The symbol list passed to the options refresh call has at most 30 entries.

# Test: setting OPTIONS_REFRESH_TOP_N=50 refreshes 50 symbols
#   Setup: Set OPTIONS_REFRESH_TOP_N="50" in env (via monkeypatch).
#   Assert: The symbol list passed to the options refresh call has at most 50 entries.

# Test: OPTIONS_REFRESH_TOP_N=0 disables top-N refresh (only strategy-aware symbols refreshed)
#   Setup: Set OPTIONS_REFRESH_TOP_N="0". No active strategies.
#   Assert: No symbols passed to refresh (or only watched symbols).

# --- Strategy-aware refresh ---

# Test: strategy-aware refresh includes symbols from active options strategies
#   Setup: Create mock active strategies that use options signals (options_flow, put_call_ratio).
#          These strategies reference symbols AAPL, TSLA.
#   Assert: AAPL and TSLA appear in the refresh symbol list regardless of their universe ranking.

# Test: strategy-aware symbols are deduplicated with top-N symbols
#   Setup: AAPL is both in top-N and in active options strategies.
#   Assert: AAPL appears exactly once in the refresh list.

# --- Pre-trade refresh ---

# Test: pre-trade refresh triggers when options data is older than current trading day
#   Setup: Mock data_metadata for symbol XYZ with options last_timestamp = yesterday.
#          Call the pre-trade freshness check.
#   Assert: An options chain fetch is triggered for XYZ.

# Test: pre-trade refresh does NOT trigger when options data is from today
#   Setup: Mock data_metadata for symbol XYZ with options last_timestamp = today.
#   Assert: No fetch triggered.

# Test: pre-trade refresh has timeout to avoid blocking trading pipeline
#   Setup: Mock the options fetch to sleep longer than the timeout.
#   Assert: The pre-trade check returns (does not hang), and a warning is logged.

# --- Rate limit budget ---

# Test: rate limit budget respected when refreshing expanded symbol list
#   Setup: Set OPTIONS_REFRESH_TOP_N=100. Verify that the refresh function
#          respects AV rate limits (does not fire more than 75 calls/min).
#   Assert: Calls are throttled appropriately (mock the rate limiter and verify calls).
```

---

## Implementation Details

### 1. Configurable Top-N (Environment Variable)

**File to modify:** `src/quantstack/data/scheduled_refresh.py`

Locate the hardcoded `30` used to cap the number of universe symbols for options refresh. Replace it with:

```python
import os

OPTIONS_REFRESH_TOP_N = int(os.environ.get("OPTIONS_REFRESH_TOP_N", "30"))
```

Read this value at module level. Use it wherever the top-N cap is applied to the options refresh symbol list. The default of 30 preserves current behavior — no change for existing deployments.

**File to modify:** `.env.example`

Add:

```bash
OPTIONS_REFRESH_TOP_N=30    # Number of top universe symbols to refresh options data for (default: 30)
```

### 2. Strategy-Aware Symbol Selection

**File to modify:** `src/quantstack/data/scheduled_refresh.py`

After building the top-N symbol list, query for symbols that are referenced by active strategies using options-related signals. The relevant signal types are `options_flow`, `options_flow_collector`, and `put_call_ratio`.

The logic is:

1. Query the database for active strategies (status in `('paper_ready', 'forward_testing', 'live')`)
2. Filter to strategies whose signal configuration references options-derived collectors
3. Extract the symbols from those strategies
4. Union with the top-N list (deduplicate)

This ensures that any symbol with an active options-dependent strategy always has fresh options data, even if it falls outside the top-N by volume or other ranking criteria.

The database query should look something like:

```python
def get_options_strategy_symbols() -> list[str]:
    """Return symbols from active strategies that depend on options signals."""
    # Query strategies table for active strategies using options_flow / put_call_ratio signals
    # Return deduplicated list of symbols
    ...
```

### 3. Pre-Trade Options Freshness Check

**File to modify:** The trading graph's entry scanner node (likely in `src/quantstack/graphs/trading/`)

Before the entry scanner performs options-dependent analysis on a symbol, add a freshness gate:

1. Check `data_metadata` for the symbol's `options_chains` last_timestamp
2. If the timestamp is before the current trading day, trigger an inline fetch of the options chain
3. The fetch must have a timeout (suggested: 10 seconds) to avoid blocking the trading pipeline
4. If the fetch times out, log a warning and proceed with stale data (stale data is better than no trade evaluation at all)

This is a lightweight addition — a single function call before the options analysis step. It should look roughly like:

```python
async def ensure_fresh_options(symbol: str, timeout_seconds: float = 10.0) -> bool:
    """Fetch options chain if stale. Returns True if data is fresh (or was refreshed).
    
    Returns False if refresh timed out (caller should proceed with stale data
    and log the condition).
    """
    ...
```

### 4. Rate Limit Budget

When `OPTIONS_REFRESH_TOP_N` is increased beyond 30, the expanded symbol list must still respect Alpha Vantage rate limits (75 calls/min for premium). The existing rate limiter in `AlphaVantageClient` handles this — no new rate limiting code is needed. However, if the refresh is routed through the provider registry (from section 10), the registry's own tracking will apply. Either way, the constraint is that the options refresh cycle must not monopolize the AV rate budget to the point where other scheduled refreshes are starved.

If the expanded list exceeds what can be fetched in the refresh window, prioritize:
1. Watched symbols (positions held)
2. Strategy-aware symbols (active options strategies)
3. Top-N universe symbols (by whatever ranking is used)

This ordering ensures the most trading-critical data is always fresh.

---

## Files Summary

| File | Action | What Changes |
|------|--------|-------------|
| `tests/unit/test_options_refresh.py` | Create | All tests for this section |
| `src/quantstack/data/scheduled_refresh.py` | Modify | Extract hardcoded 30 to env var, add strategy-aware symbol selection |
| `.env.example` | Modify | Add `OPTIONS_REFRESH_TOP_N` |
| Trading graph entry scanner node | Modify | Add pre-trade options freshness check with timeout |

---

## Acceptance Criteria

1. Setting `OPTIONS_REFRESH_TOP_N=50` causes 50 universe symbols to be refreshed (instead of 30)
2. Symbols from active options strategies appear in the refresh list even if not in the top-N
3. A symbol with stale options data (older than today) gets refreshed inline before options-dependent trade analysis
4. The pre-trade refresh times out gracefully (does not block the trading pipeline)
5. Rate limits are respected even with an expanded symbol list
6. All tests in `tests/unit/test_options_refresh.py` pass
