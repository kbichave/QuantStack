# Section 05: Staleness Rejection in Collectors

## Overview

This section adds `check_freshness()` calls to all signal engine collectors (except `ml_signal`) so that each collector short-circuits with `{}` when its underlying data is too old. This prevents the system from generating confident signals on stale data. The synthesis engine already handles empty collector outputs by redistributing weight, so no changes to synthesis are needed for the normal case. An edge case where ALL collectors return `{}` simultaneously requires a safety check in synthesis to avoid division-by-zero or misleading output.

**Dependency:** Section 01 (`section-01-staleness-helper`) must be completed first. It creates `src/quantstack/signal_engine/staleness.py` with the `check_freshness()` function and ensures the `data_metadata` table is populated for all data source types.

---

## Background

The signal engine in `src/quantstack/signal_engine/engine.py` dispatches 22+ collectors concurrently (lines 217-245). Each collector reads from the local `DataStore` and returns a `dict[str, Any]`. If a collector fails or has no data, it returns `{}`, and the synthesis engine (`src/quantstack/signal_engine/synthesis.py`) handles that by treating missing keys as absent signals.

Today, no collector checks whether its underlying data is current. A technical collector computing RSI on 3-week-old OHLCV data returns a confident signal that is meaningless for current trading decisions. The fix is to add a freshness gate at the top of each collector's main function.

---

## Tests (Write First)

Create `tests/unit/test_staleness_collectors.py`. These tests validate that collectors correctly reject stale data and that synthesis handles the all-stale edge case.

### Per-Collector Staleness Tests

```python
# tests/unit/test_staleness_collectors.py

"""
Tests for staleness rejection across all signal engine collectors.

Each test verifies that a collector returns {} when its data source
exceeds the configured staleness threshold, and computes normally
when data is fresh.
"""

# Test: technical collector returns {} when OHLCV data is 5 days old (exceeds 4-day threshold)
# Test: technical collector computes normally when OHLCV data is 1 day old
# Test: macro collector returns {} when macro_indicators data is 50 days old (exceeds 45-day threshold)
# Test: macro collector computes normally when macro_indicators data is 30 days old
# Test: sentiment collector returns {} when news_sentiment data is 10 days old (exceeds 7-day threshold)
# Test: fundamentals collector returns {} when company_overview data is 100 days old (exceeds 90-day threshold)
# Test: ml_signal collector does NOT check staleness (no max_staleness defined)
```

The test strategy is to mock `check_freshness()` return values rather than populating `data_metadata` rows, keeping tests fast and isolated. For each collector category, write at least one stale test (returns `{}`) and one fresh test (computes normally). Use `unittest.mock.patch` on `quantstack.signal_engine.staleness.check_freshness`.

### All-Stale Edge Case Tests

```python
# Test: when ALL collectors return {} due to staleness, synthesis returns a valid SymbolBrief
#   - No division by zero
#   - Conviction is 0.0 or near-zero
#   - Bias label is "neutral" or equivalent
# Test: when all collectors return {}, brief includes an explicit low-confidence indicator
#   (e.g., conviction < 0.1 or a collector_failures list containing all collector names)
```

These tests call the synthesis engine's `synthesize()` method directly, passing `{}` for every collector input. The goal is to verify graceful degradation, not specific signal values.

### Performance Benchmark Test

```python
# Test: signal engine latency with staleness checks enabled adds < 2s overhead
#   across 50 symbols with all data_metadata rows populated.
#   This is a regression guard — if check_freshness starts doing expensive queries,
#   this test catches it.
```

This test is optional for CI (mark with `@pytest.mark.slow`) but must pass before merge.

---

## Staleness Thresholds Reference

Each collector maps to a data source table and has a maximum acceptable data age in calendar days:

| Collector Function | Engine Key | Data Table | Max Days | Rationale |
|---|---|---|---|---|
| `collect_technical` | `technical` | `ohlcv` | 4 | 2 trading days = up to 4 calendar days (covers 3-day weekends) |
| `collect_regime` | `regime` | `ohlcv` | 4 | Same price-derived logic |
| `collect_volume` | `volume` | `ohlcv` | 4 | Volume computed from OHLCV |
| `collect_risk` | `risk` | `ohlcv` | 4 | Risk metrics from price data |
| `collect_statarb` | `statarb` | `ohlcv` | 4 | Pair statistics from prices |
| `collect_flow` | `flow` | `ohlcv` | 30 | Flow signals reported with lag |
| `collect_options_flow_async` | `options_flow` | `options_chains` | 3 | Options chains update daily |
| `collect_put_call_ratio` | `put_call_ratio` | `options_chains` | 3 | Options-derived |
| `collect_sentiment` | `sentiment` (via Groq) | `news_sentiment` | 7 | News has longer relevance |
| `collect_sentiment_alphavantage` | `sentiment` (AV) | `news_sentiment` | 7 | Same category |
| `collect_social_sentiment` | `social` | `news_sentiment` | 7 | Social media sentiment |
| `collect_fundamentals` | `fundamentals` | `company_overview` | 90 | Quarterly data |
| `collect_quality` | `quality` | `company_overview` | 90 | Quality factors from fundamentals |
| `collect_earnings_momentum` | `earnings_momentum` | `earnings_history` | 90 | Quarterly earnings |
| `collect_macro` | `macro` | `macro_indicators` | 45 | Monthly macro indicators |
| `collect_commodity_signals` | `commodity` | `macro_indicators` | 45 | Commodity data via same pipeline |
| `collect_cross_asset` | `cross_asset` | `macro_indicators` | 45 | Cross-asset correlations |
| `collect_insider_signals` | `insider` | `insider_trades` | 30 | Reported with lag |
| `collect_short_interest` | `short_interest` | `short_interest` | 14 | Bi-monthly FINRA updates |
| `collect_sector` | `sector` | `ohlcv` | 7 | Weekly rotation signals |
| `collect_events` | `events` | `earnings_calendar` | 30 | Calendar-based events |
| `collect_ewf` | `ewf` | `ewf_forecasts` | 7 | External wave forecast freshness |
| `collect_ml_signal` | `ml_signal` | N/A | N/A | **Skip staleness check** (model-dependent) |

---

## Implementation

### Step 1: Add `check_freshness()` to Each Collector

For each of the 22 collectors listed above (excluding `ml_signal`), add a freshness check at the top of the main `collect_*` function, before any computation. The pattern is identical for every collector:

```python
from quantstack.signal_engine.staleness import check_freshness

async def collect_technical(symbol: str, store: DataStore) -> dict[str, Any]:
    """..."""
    if not check_freshness(store, symbol, table="ohlcv", max_days=4):
        return {}
    # ... existing computation unchanged ...
```

The key points for each collector:

- The `table` parameter must match the data source that collector depends on (see threshold table above).
- The `max_days` parameter comes from the threshold table.
- The check goes at the very top of the function, before any I/O or computation.
- The `check_freshness()` function (from section 01) handles logging internally: it logs a warning with collector context, symbol, actual data age, and the threshold when data is stale.
- Return `{}` immediately on stale data. This is the existing convention for "no signal available."

#### Collector Files to Modify

All files are in `src/quantstack/signal_engine/collectors/`:

1. `technical.py` — table=`ohlcv`, max_days=4
2. `regime.py` — table=`ohlcv`, max_days=4
3. `volume.py` — table=`ohlcv`, max_days=4
4. `risk.py` — table=`ohlcv`, max_days=4
5. `statarb.py` — table=`ohlcv`, max_days=4
6. `flow.py` — table=`ohlcv`, max_days=30
7. `options_flow.py` — table=`options_chains`, max_days=3 (note: this file exists but `options_flow_collector.py` is the one imported in engine.py)
8. `options_flow_collector.py` — table=`options_chains`, max_days=3
9. `put_call_ratio.py` — table=`options_chains`, max_days=3
10. `sentiment.py` — table=`news_sentiment`, max_days=7
11. `sentiment_alphavantage.py` — table=`news_sentiment`, max_days=7
12. `social_sentiment.py` — table=`news_sentiment`, max_days=7
13. `fundamentals.py` — table=`company_overview`, max_days=90
14. `quality.py` — table=`company_overview`, max_days=90
15. `earnings_momentum.py` — table=`earnings_history`, max_days=90
16. `macro.py` — table=`macro_indicators`, max_days=45
17. `commodity.py` — table=`macro_indicators`, max_days=45
18. `cross_asset.py` — table=`macro_indicators`, max_days=45
19. `insider_signals.py` — table=`insider_trades`, max_days=30
20. `short_interest.py` — table=`short_interest`, max_days=14
21. `sector.py` — table=`ohlcv`, max_days=7
22. `events.py` — table=`earnings_calendar`, max_days=30
23. `ewf_collector.py` — table=`ewf_forecasts`, max_days=7

**Do NOT modify** `ml_signal.py`. ML signal staleness is model-dependent and not suitable for a calendar-day threshold.

Note: `enhanced_sentiment.py` and `l2_microstructure.py` exist in the collectors directory but are not imported or dispatched by `engine.py`. Do not add staleness checks to unused collectors.

### Step 2: Handle the All-Stale Edge Case in Synthesis

When every collector returns `{}`, the synthesis engine in `src/quantstack/signal_engine/synthesis.py` must not crash or produce misleading output. Review the `synthesize()` method (line 185) and its helper `_compute_bias_and_conviction()` to verify behavior when all inputs are empty dicts.

The expected behavior when all collectors return `{}`:

- `conviction` should be 0.0 (no evidence to be convicted on)
- `bias` should be `"neutral"` (no directional signal)
- The method must not raise `ZeroDivisionError` from weight redistribution logic
- `collector_failures` should list all collector names that returned empty

If the current synthesis already handles this correctly (likely, since individual collector failures are common), document it with a test. If not, add a guard:

```python
# In synthesis.py, inside the bias/conviction computation:
if total_weight == 0:
    return "neutral", 0.0
```

This is a minimal, defensive change. The test from the "All-Stale Edge Case Tests" section above is the primary deliverable here.

---

## Verification Checklist

After implementation, verify:

1. Each of the 22 modified collectors imports `check_freshness` and calls it before computation
2. Each collector uses the correct `table` and `max_days` values per the threshold table
3. `ml_signal.py` is untouched
4. `enhanced_sentiment.py` and `l2_microstructure.py` are untouched (not used by engine)
5. All per-collector tests pass (mock `check_freshness` returning False -> collector returns `{}`)
6. All per-collector tests pass (mock `check_freshness` returning True -> collector computes normally)
7. All-stale synthesis test passes (all `{}` inputs -> valid low-confidence SymbolBrief)
8. Performance benchmark (if run) shows < 2s overhead for 50 symbols
9. No import cycles introduced by adding `from quantstack.signal_engine.staleness import check_freshness` to collectors

---

## Dependencies

- **Requires section-01-staleness-helper:** The `check_freshness()` function and `data_metadata` population must exist before this section can be implemented.
- **No downstream blockers:** No other section depends on this one.
