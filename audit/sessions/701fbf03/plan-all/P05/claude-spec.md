# P05 Spec: Adaptive Signal Synthesis — Gap Completion

## Context

QuantStack is an autonomous trading system using LangGraph with a signal synthesis engine that combines 7+ collector outputs (trend, RSI, MACD, BB, sentiment, ML, flow) into a weighted score per symbol. The system already has extensive P05 implementation from a CTO audit.

## What Already Exists

The prior audit already implemented:
- **IC-driven weights**: `ICAttributionTracker.get_weights_for_regime()` integrated into synthesis (flag-gated via `ic_driven_weights_enabled`)
- **Vol sub-regimes**: 8 sub-regime profiles in `_WEIGHT_PROFILES` (e.g., `trending_up_low_vol`)
- **Ensemble methods**: Weighted average, weighted median, trimmed mean — with hash-based A/B assignment
- **Transition dampening**: Score halved when `P(transition) > 0.3`
- **Multiplicative conviction**: 6 factors (ADX, stability, timeframe, regime agreement, ML confirmation, data quality)
- **IC gate**: Zeros out low-IC collectors
- **Correlation penalty**: Halves redundant collector weights

## What This Phase Must Deliver (Actual Gaps)

### 1. IC Weight Batch Job & Caching
**Problem**: Every synthesis call creates a new `ICAttributionTracker()`, loading all observations from DB. This is both slow and wasteful.
**Solution**: Weekly batch job that precomputes regime-conditioned weights into a `precomputed_ic_weights` table. Synthesis reads from this table (single query) instead of instantiating the full tracker.

### 2. Position Sizing During Regime Transitions
**Problem**: Score dampening (×0.5) reduces conviction but doesn't explicitly halve position sizes. Downstream sizing in `nodes.py` uses `signal_value * skill_adj * affinity` — conviction reduction partially achieves smaller sizes but doesn't guarantee 50% reduction.
**Solution**: Add `transition_zone: bool` flag to `SymbolBrief`. Consuming nodes apply explicit 0.5× position scalar when `transition_zone=True`.

### 3. Conviction Factor Calibration Job
**Problem**: Conviction factors use hardcoded parameters (ADX ramp range [15,50], stability range [0.85,1.05], timeframe penalty 0.80, etc.). These were set by intuition, not calibrated from data.
**Solution**: Quarterly job that computes optimal factor parameters by regressing realized trade outcomes against factor contributions. Store calibrated params in `conviction_calibration` table. Fall back to hardcoded defaults if insufficient data.

### 4. A/B Test Result Tracking
**Problem**: Hash-based ensemble A/B assignment exists but results are never compared. No way to know which method wins.
**Solution**: Track per-symbol-per-method conviction→outcome in `ensemble_ab_results` table. Weekly comparison job computes Sharpe/IC per method. After 30 days, promote winner.

### 5. Integration Test Suite
**Problem**: No end-to-end test for IC→weights→synthesis flow.
**Solution**: Integration test that exercises: record IC observations → compute weights → synthesize with IC weights → verify weights differ from static.

## Constraints

- All changes flag-gated (kill-switch per feature)
- Batch jobs run during overnight/weekend compute window
- DB writes use `db_conn()` context managers
- Synthesis latency budget: <50ms per symbol (batch precomputation enables this)
- Static weight fallback always available when IC data insufficient
