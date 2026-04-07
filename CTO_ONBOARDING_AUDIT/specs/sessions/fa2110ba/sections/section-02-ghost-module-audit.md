# Section 2: Ghost Module API Audit

**Depends on:** section-01-persistence-migration (StrategyBreaker and ICAttributionTracker must already use PostgreSQL)
**Blocks:** section-03-readpoint-wiring (all 6 readpoints assume correct APIs)

---

## Background

Six fully-implemented learning modules exist under `src/quantstack/learning/` and `src/quantstack/execution/strategy_breaker.py`. They record data but have zero consumers -- no downstream code reads their output. Before wiring these modules into the live trading system (section-03), their APIs, math, and thresholds must be audited and fixed. Wiring broken formulas into live sizing decisions compounds the damage rather than fixing the feedback gap.

The modules and their current state:

| Module | File | Lines | Issue |
|--------|------|-------|-------|
| OutcomeTracker | `src/quantstack/learning/outcome_tracker.py` | ~350 | Affinity formula too slow to adapt |
| SkillTracker | `src/quantstack/learning/skill_tracker.py` | ~421 | ICIR adjustment unbounded before clamp |
| ICAttributionTracker | `src/quantstack/learning/ic_attribution.py` | ~420 | API is clean; verify scipy dependency |
| ExpectancyEngine | `src/quantstack/learning/expectancy_engine.py` | ~98 | Duplicates `core/kelly_sizing.py`; mark deprecated |
| StrategyBreaker | `src/quantstack/execution/strategy_breaker.py` | ~553 | Persistence migration handled in section-01 |
| TradeEvaluator | (writes to `trade_quality_scores`) | - | Needs a summary/aggregation read function |

---

## Tests (implement first)

Test file: `tests/unit/test_ghost_module_audit.py`

### OutcomeTracker affinity formula fix

```python
class TestOutcomeTrackerFormula:
    """Verify the updated affinity formula produces meaningful feedback."""

    def test_loss_step_size(self):
        """A -2% loss should produce a step of approximately -0.11, not -0.019."""
        # With step=0.15 and tanh divisor=2.0:
        # tanh(-2.0 / 2.0) = tanh(-1.0) ~ -0.762
        # step = 0.15 * -0.762 ~ -0.114
        ...

    def test_exponential_decay_halflife(self):
        """Outcome from 20 trades ago contributes half as much as the most recent."""
        # decay_weight = 0.5^(trades_since / 20)
        # At trades_since=20: weight = 0.5
        # At trades_since=0: weight = 1.0
        ...

    def test_cold_start_no_adjustment(self):
        """Fewer than 5 outcomes for a regime returns affinity 1.0 unchanged."""
        ...

    def test_affinity_bounds(self):
        """Affinity stays within [0.1, 1.0] after many consecutive wins or losses."""
        ...

    def test_recency_weighting(self):
        """A recent loss has more impact than an old loss of the same magnitude."""
        ...
```

### SkillTracker ICIR adjustment

```python
class TestSkillTrackerICIR:
    """Verify the simplified ICIR adjustment in get_confidence_adjustment()."""

    def test_high_icir_capped(self):
        """ICIR=3.0 must cap the IC adjustment at 0.3, not produce 0.6."""
        ...

    def test_adjustment_within_bounds(self):
        """get_confidence_adjustment() stays within [0.5, 1.5] for all edge cases."""
        # Test with: zero observations, perfect accuracy, zero accuracy,
        # extreme ICIR (positive and negative), DECAYING trend
        ...
```

### StrategyBreaker state integrity (post-migration)

```python
class TestBreakerPostMigration:
    """Verify breaker state survives restart after section-01 PostgreSQL migration."""

    def test_tripped_persists_across_restart(self):
        """A TRIPPED state read back from DB matches the original."""
        ...

    def test_concurrent_reads_no_block(self):
        """Multiple get_scale_factor() calls do not deadlock."""
        ...
```

### ICAttributionTracker verification

```python
class TestICAttributionVerification:
    """Verify ICAttributionTracker API correctness and scipy availability."""

    def test_scipy_spearman_available(self):
        """scipy.stats.spearmanr is importable (dependency in pyproject.toml)."""
        from scipy.stats import spearmanr
        assert callable(spearmanr)

    def test_data_persists_across_restart(self):
        """Data written to DB can be read back by a new instance."""
        ...
```

---

## Implementation Details

### 1. OutcomeTracker affinity formula fix

**File:** `src/quantstack/learning/outcome_tracker.py`

**Current formula (lines 62-66):**
```
_STEP = 0.05
_PNL_SCALE = 5.0
```
The formula `new_affinity = clip(current + 0.05 * tanh(pnl_pct / 5.0), 0.1, 1.0)` is too slow. A -2% loss produces `tanh(-0.4) ~ -0.38`, so the step is `0.05 * -0.38 = -0.019`. Starting from affinity 1.0, it takes approximately 47 consecutive losses of -2% to reach the floor of 0.1. This is not a meaningful feedback signal for a swing-trading system where a few losses should begin reducing allocation.

**Changes to hyperparameters:**

- `_STEP`: change from `0.05` to `0.15`
- `_PNL_SCALE`: change from `5.0` to `2.0`
- `_MIN_OUTCOMES_FOR_UPDATE`: change from `3` to `5` (cold-start safety)

**New: recency-weighted exponential decay with 20-trade halflife.** Each outcome's contribution decays by `0.5^(trades_since / 20)`. The last 20 trades contribute approximately 50% of the affinity signal, preventing stale outcomes from anchoring the value while smoothing single-trade noise. Add a constant `_HALFLIFE_TRADES = 20`.

**Changes to `apply_learning()` method (around line 190):**

The current logic computes a simple mean of all outcome weights per regime and applies one step. Replace this with a recency-weighted computation:

1. For each outcome in the regime, compute `outcome_weight = tanh(pnl_pct / _PNL_SCALE)`
2. Apply decay: `decayed_weight = outcome_weight * 0.5^(index_from_most_recent / _HALFLIFE_TRADES)`
3. Compute weighted mean: `weighted_mean = sum(decayed_weights) / sum(decay_factors)`
4. Apply step: `new_affinity = clip(current + _STEP * weighted_mean, _CLIP_MIN, _CLIP_MAX)`

**Cold-start rule:** If fewer than 5 outcomes exist for a regime, skip the update for that regime (affinity remains 1.0). The learning signal is too noisy with fewer observations.

**Result:** A -2% loss now produces `tanh(-1.0) ~ -0.76`, step = `0.15 * -0.76 ~ -0.11`. Reaching meaningful reduction from 1.0 takes approximately 8 consecutive losses instead of 47.

### 2. SkillTracker ICIR adjustment simplification

**File:** `src/quantstack/learning/skill_tracker.py`

**Current code (line 348):**
```python
icir_adj = max(-0.2, min(0.3, skill.icir * 0.2))
```

The intent is unclear. With ICIR=3.0 (high but possible for a good signal), `icir * 0.2 = 0.6`, which exceeds the intended max of 0.3. The outer `min(0.3, ...)` catches it, but the formula would be clearer if the multiplier itself prevented overshoot.

**Change to:**
```python
icir_adj = max(-0.2, min(0.3, skill.icir * 0.15))
```

With the new multiplier of 0.15: ICIR=2.0 produces 0.30 (hits cap exactly), ICIR=3.0 produces 0.45 but clamps to 0.30. The behavior is identical for ICIR values up to 2.0 and clearer in intent for higher values. The overall `get_confidence_adjustment()` clamp of [0.5, 1.5] remains unchanged.

### 3. ICAttributionTracker verification

**File:** `src/quantstack/learning/ic_attribution.py`

**scipy dependency:** Confirmed present in `pyproject.toml` (line 32: `"scipy>=1.10.0"`). The import at line 37 (`from scipy.stats import spearmanr`) is a hard module-level import -- if scipy were missing, the module would fail on import, which is the correct behavior (fail fast, not silently degrade).

**API review:** The `get_weights()` method (line 267) normalizes by IC > 0, which is correct. The `get_collector_ic()` method has a `min_observations` parameter defaulting to 20, which provides a reasonable cold-start guard. No changes needed to the API.

**Persistence migration:** Handled in section-01. After section-01, the `_persist()` and `_load()` methods will use PostgreSQL via `db_conn()` instead of `~/.quantstack/ic_attribution.json`. This section only verifies the API math is correct; it does not modify persistence.

### 4. ExpectancyEngine deprecation

**File:** `src/quantstack/learning/expectancy_engine.py`

This module duplicates functionality already provided by `src/quantstack/core/kelly_sizing.py` (specifically `regime_kelly_fraction()`). The Kelly sizing in the core module already incorporates IC and is more principled -- it uses regime-conditioned win rate and payoff ratio, while ExpectancyEngine uses a simpler unconditional calculation.

**Action:** Add a deprecation docstring at the top of `ExpectancyEngine.__init__()` and a module-level comment. Do not wire it into any new code paths. Do not delete it yet (existing callers may reference it).

```python
# DEPRECATED: This module duplicates core/kelly_sizing.py::regime_kelly_fraction().
# The Kelly sizing in core/ is more principled (regime-conditioned, IC-aware).
# Scheduled for removal after all callers migrate. See Phase 7 plan, Section 1.
```

### 5. TradeEvaluator read function

The TradeEvaluator writes 6-dimension LLM scores (execution_quality, thesis_accuracy, risk_management, timing_quality, sizing_quality, overall_score) to the `trade_quality_scores` table, but nothing reads from it. The daily planner in section-03 (Wire 6) will need aggregated pattern data, not individual scores.

**Action:** Add a standalone function (not on TradeEvaluator itself, since TradeEvaluator may not exist as a class -- the writes happen in trade hooks). Place it in a utility location accessible to the trading graph.

**Function signature:**
```python
def get_trade_quality_summary(
    strategy_id: str | None = None,
    window: int = 30,
) -> dict:
    """
    Compute rolling averages per quality dimension over the last `window` scored trades.

    Returns:
        {
            "dimensions": {"execution_quality": 6.2, "thesis_accuracy": 7.1, ...},
            "weakest": "execution_quality",
            "weakest_score": 6.2,
            "trade_count": 28,
        }
        Returns None if fewer than 5 scored trades exist.
    """
```

This function queries `trade_quality_scores`, groups by dimension, computes rolling averages, and identifies the weakest dimension. Section-03 (Wire 6) will call this from the `daily_plan` node.

---

## Verification Checklist

After implementation, verify:

1. `uv run pytest tests/unit/test_ghost_module_audit.py -x -q` passes
2. OutcomeTracker with 8 consecutive -2% losses reduces affinity from 1.0 to below 0.5
3. OutcomeTracker with 4 outcomes for a regime makes no adjustment (cold-start)
4. SkillTracker `get_confidence_adjustment()` never exceeds 1.5 or drops below 0.5 regardless of input
5. `from scipy.stats import spearmanr` succeeds in the project environment
6. ExpectancyEngine has deprecation comment; no new code paths reference it
7. `get_trade_quality_summary()` returns correct rolling averages for test data

---

## Rollback

Revert file changes via git. All changes in this section are to formula constants and a new utility function. No schema changes, no new tables, no persistence changes (those are in section-01). The ghost modules remain functional in their current disconnected state if reverted.
