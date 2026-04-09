# P05 Implementation Plan: Adaptive Signal Synthesis — Gap Completion

## 1. Background

QuantStack's signal synthesis engine combines 7+ collector outputs (trend, RSI, MACD, BB, sentiment, ML, flow) into a weighted score per symbol. A prior CTO audit already implemented most of P05's scope: IC-driven weights, vol-conditioned sub-regimes, three ensemble methods with A/B assignment, transition dampening, and multiplicative conviction factors.

This plan addresses the remaining gaps: performance optimization (batch precomputation), position sizing enforcement during transitions, conviction factor calibration from realized data, A/B test result tracking, and integration testing.

## 2. Anti-Goals

- **Do NOT rewrite existing IC weight integration** — the `ICAttributionTracker.get_weights_for_regime()` → synthesis path works. We optimize how it's called, not what it does.
- **Do NOT add sub-regime granularity to IC weights** — base regime (4) granularity is sufficient for IC computation. Sub-regime profiles serve as static fallback. Data sparsity makes sub-regime IC unreliable.
- **Do NOT change the multiplicative conviction factor formulas** — the calibration job optimizes parameters (thresholds, ranges), not the multiplicative structure itself.
- **Do NOT build a full experimentation framework** — the A/B test tracking is minimal: table, weekly comparison, auto-promote. Not a general experimentation platform.
- **Do NOT modify the risk gate for transition handling** — transition zone is a signal quality judgment communicated via SymbolBrief, not a hard risk limit.

## 3. IC Weight Precomputation

### 3.1 Database Table

Add `precomputed_ic_weights` table:

```python
@dataclass
class PrecomputedICWeight:
    regime: str           # e.g. "trending_up"
    collector: str        # e.g. "trend", "rsi", "ml"
    weight: float         # normalized weight (0.0–1.0)
    ic_value: float       # underlying IC used
    computed_at: datetime
```

Schema: `(regime, collector)` as composite primary key, `computed_at` timestamp, `ic_value` for auditability.

### 3.2 Batch Job

New function `compute_and_store_ic_weights()` in `src/quantstack/learning/ic_attribution.py`:

1. For each base regime (trending_up, trending_down, ranging, unknown):
   a. Filter IC observations by regime
   b. Compute per-collector Spearman IC over 63-day window
   c. Apply IC gate (drop collectors with IC < 0.02)
   d. Apply ICIR consistency penalty (0.7× if ICIR < 0.1)
   e. Apply correlation penalty from `CrossSectionalICTracker`
   f. Check weight floor (total > 0.1, else skip this regime)
   g. Normalize positive-IC collectors to sum=1.0
   h. Upsert into `precomputed_ic_weights`
2. Log summary: which regimes updated, which fell back to static

Schedule: weekly, during overnight compute window (configured in `scripts/scheduler.py`).

### 3.3 Synthesis Integration

Replace the current pattern in `synthesis.py` (lines 530-544):

**Current:** Instantiate `ICAttributionTracker()` → call `get_weights_for_regime()` → fallback to static.

**New:** Single query to `precomputed_ic_weights` table filtered by current regime. If no rows or `computed_at` > 7 days old, fall back to static weights. No ICAttributionTracker instantiation during synthesis.

This reduces per-synthesis DB cost from "load all IC observations" to "one SELECT with WHERE regime = X".

**Staleness tolerance (review feedback):** Use 14-day staleness window (not 7) to tolerate one missed weekly run. Add retry + alert if the batch job fails.

**Sanity bounds (review feedback):** After computing weights, reject any regime where a single collector has weight > 0.80. Log warning and fall back to static for that regime. Add a `batch_id` column to `precomputed_ic_weights` to keep the last 2 batches for rollback.

**Weight concentration test:** Add test case where a single collector dominates IC — verify the 0.80 cap triggers static fallback.

## 4. Transition Zone Position Sizing

### 4.1 SymbolBrief Schema Change

Add `transition_zone: bool = False` field to `SymbolBrief` in `src/quantstack/shared/schemas.py`.

Set to `True` in `synthesis.py` when `transition_probability > 0.3` (same threshold as existing score dampening).

### 4.2 Downstream Consumption

In `src/quantstack/graphs/trading/nodes.py`, where position size is computed (the `signal_value * skill_adj * affinity` multiplication), add:

```python
if brief.transition_zone:
    position_scalar *= 0.5
```

This is a single multiplication in the existing sizing logic. The 0.5 scalar stacks with the existing score dampening (×0.5 on conviction), meaning total effective reduction during transitions is ~75% (0.5 conviction × 0.5 sizing ≈ 0.25 of normal). This is intentionally aggressive — regime transitions are high-uncertainty periods.

**Minimum position floor (review feedback):** Add a floor check after all scalars are applied: if final position size < $50 (or < 0.1% of portfolio), skip the trade entirely rather than taking a microscopic position. Document the full multiplication chain (affinity × skill × transition × forward_testing scalar) in a code comment at the sizing site so future developers see all active scalars.

### 4.3 Feature Flag

New flag `transition_position_sizing_enabled()` in `feedback_flags.py`. Default: `True`. Kill switch via env var `FEEDBACK_TRANSITION_POSITION_SIZING`.

## 5. Conviction Factor Calibration

### 5.1 Persist Factor Values

Enrich the `signals` INSERT in `synthesis.py` (around line 325) to include conviction factors in the `metadata` JSONB:

```python
metadata = {
    "votes": vote_scores,
    "weights": final_weights,
    "conviction_factors": conviction_factor_breakdown,  # NEW
}
```

This is a one-line change in the existing INSERT statement.

### 5.2 Calibration Table

Add `conviction_calibration` table:

```python
@dataclass
class ConvictionCalibration:
    factor_name: str      # e.g. "adx", "stability"
    param_name: str       # e.g. "lower_bound", "upper_bound", "scale"
    param_value: float
    calibrated_at: datetime
    sample_size: int
    r_squared: float      # regression fit quality
```

### 5.3 Calibration Job

New function `calibrate_conviction_factors()` in `src/quantstack/learning/ic_attribution.py`:

1. Join `signals.metadata.conviction_factors` with `closed_trades.realized_pnl_pct` on (symbol, signal_date ≈ entry_date)
2. For each factor, regress factor_value against trade_outcome
3. Optimize: find threshold/scale values that maximize correlation between factor-adjusted conviction and realized PnL
4. Store in `conviction_calibration` table
5. Synthesis reads calibrated params on startup (or weekly refresh)
6. Fall back to hardcoded defaults if calibration has <100 trades or R² < 0.01

**Survivorship bias fix (review feedback):** Use ALL generated signals (from `signals` table) correlated with forward returns from OHLCV, not just closed trades. This avoids selection bias from only analyzing traded signals and provides much more data.

Schedule: quarterly (90-day cycle), during overnight compute.

### 5.4 Synthesis Integration

In `_conviction_multiplicative()`, read calibrated parameters from `conviction_calibration` table instead of hardcoded values. Cache in a module-level `_CalibrationCache` with TTL:

```python
@dataclass
class _CalibrationCache:
    data: dict[str, dict[str, float]]
    loaded_at: datetime
```

On each call, check `loaded_at`. If > 7 days stale, re-query from DB. If query fails, keep stale cache. If no cache exists, use hardcoded defaults. Access synchronized via existing Lock pattern.

## 6. A/B Test Result Tracking

### 6.1 Results Table

Add `ensemble_ab_results` table:

```python
@dataclass
class EnsembleABResult:
    symbol: str
    signal_date: date
    method_name: str        # "weighted_avg", "weighted_median", "trimmed_mean"
    signal_value: float
    forward_return_5d: float | None  # filled by backfill job
    recorded_at: datetime
```

### 6.2 Recording

In `synthesis.py`, after ensemble computation, record the method used and signal value:

```python
# After ensemble_fn(scores, weights)
method_name = ensemble_fn.__name__.replace("_ensemble_", "")
# INSERT into ensemble_ab_results (symbol, date, method, signal_value)
```

### 6.3 Offline Evaluation (review feedback: A/B redesign)

**Changed approach:** Instead of hash-based live A/B (which gives N<17 per method with ~50 symbols), compute ALL three methods for every symbol on every signal and store all results. This is offline evaluation, not live A/B.

In `synthesis.py`, after computing weights, compute all three ensemble values:
```python
for fn in _ENSEMBLE_METHODS:
    method_name = fn.__name__.replace("_ensemble_", "")
    value = fn(scores, weights)
    # INSERT into ensemble_ab_results (symbol, date, method, signal_value)
```

Weekly job `evaluate_ensemble_ab()`:
1. Backfill `forward_return_5d` by joining `ensemble_ab_results` with OHLCV: for each (symbol, signal_date), compute `(close[date+5] - close[date]) / close[date]`. Skip rows where date+5 hasn't occurred yet. For Friday signals, use next Tuesday close (skip weekends).
2. Compute per-method IC (Spearman correlation of signal_value vs forward_return_5d)
3. If a non-default method has IC improvement > 0.01 sustained over 60+ days: update active method
4. Synthesis reads active method to select which ensemble function to use for the actual SymbolBrief

### 6.4 Active Method Selection

Use feature flag system (review feedback: no single-row config table). Add `ensemble_active_method()` to `feedback_flags.py` that reads env var `ENSEMBLE_ACTIVE_METHOD` (default: `"weighted_avg"`). The evaluation job sets this by writing to a `.env.ensemble` file or by updating the existing system_state table. This avoids a separate config table.

## 7. Integration with Existing Modules

### 7.1 ic_weights.py Weight Floor

Wire `check_weight_floor()` from `ic_weights.py` into the precomputation job. If total effective weight after IC factors falls below 0.1, the job skips that regime (static weights remain active).

### 7.2 CrossSectionalICTracker

The precomputation job uses `CrossSectionalICTracker.compute_pairwise_correlation()` to apply correlation penalties during weight computation — not during synthesis. This moves the expensive correlation computation from per-synthesis to weekly batch.

### 7.3 Scheduler Integration

Add three new scheduled tasks to `scripts/scheduler.py`:
- `weekly_ic_weight_precompute` — runs `compute_and_store_ic_weights()`
- `quarterly_conviction_calibration` — runs `calibrate_conviction_factors()`
- `weekly_ensemble_ab_evaluate` — runs `evaluate_ensemble_ab()`

All scheduled for overnight compute window (existing pattern in scheduler).

## 8. Code Hygiene (Review Feedback)

### 8.1 Fix Deferred Imports

Move all deferred imports in `synthesis.py` to module level:
- Lines 532-534: `from quantstack.config.feedback_flags import ic_driven_weights_enabled`
- Lines 548-551: `from quantstack.config.feedback_flags import ic_gate_enabled`
- Lines 564-568: `from quantstack.config.feedback_flags import correlation_penalty_enabled`
- Line 582: `from quantstack.config.feedback_flags import ensemble_ab_test_enabled`
- Line 591: `from quantstack.config.feedback_flags import transition_signal_dampening_enabled`

All flag imports go to module top. Runtime calls stay where they are.

### 8.2 Replace Silent Catch-and-Pass

Replace bare `except Exception: pass` blocks (lines 336, 544, 561, 579) with logged warnings:
```python
except Exception as exc:
    logger.warning("descriptive_context | error=%s", exc)
```

## 9. Schema Migrations

Three new tables, one schema enrichment (review feedback: removed ensemble_config single-row table):

1. `precomputed_ic_weights` — composite PK (regime, collector, batch_id). Keep last 2 batches for rollback.
2. `conviction_calibration` — composite PK (factor_name, param_name)
3. `ensemble_ab_results` — PK id, indexed on (signal_date, method_name)
4. `signals.metadata` — add `conviction_factors` to JSONB (no schema change needed, just data enrichment)

Migrations follow existing idempotent pattern (`CREATE TABLE IF NOT EXISTS`).

## 9. Testing Strategy

### Unit Tests
- IC weight precomputation: mock DB with known IC observations, verify correct weights computed
- Transition zone: verify SymbolBrief.transition_zone set correctly at various probabilities
- Conviction calibration: verify regression with synthetic data
- Ensemble A/B recording: verify method name and signal value persisted

### Integration Test
- End-to-end: seed IC observations → run precompute → synthesize → verify IC-driven weights used → verify transition_zone propagates to sizing

### Edge Cases
- All collectors have IC < 0.02 → static fallback
- Zero observations for a regime → static fallback
- Precomputed weights >14 days old → static fallback
- All ensemble methods tie → keep weighted_avg default
- Single collector weight > 0.80 → static fallback (sanity bound)
- Position size < $50 after all scalars → skip trade
- Calibration query fails → keep stale cache or hardcoded defaults
- Forward return not yet available (recent signals) → skip from evaluation
