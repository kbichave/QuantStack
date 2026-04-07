# Section 10: Conviction Calibration — Multiplicative Factors

## Overview

The current conviction logic in `RuleBasedSynthesizer._compute_bias_and_conviction()` (file: `src/quantstack/signal_engine/synthesis.py`, lines 383-420) uses additive adjustments to modify base conviction. For example, `+0.10` for ADX > 25, `-0.15` for weekly/daily contradiction, `-0.20` for collector failures. The problem: additive adjustments create non-proportional effects depending on the base conviction level. A `-0.15` penalty on a base conviction of `0.20` is a 75% reduction, but on a base of `0.80` it is only 19%. Multiplicative factors fix this by scaling proportionally regardless of base level.

This section converts the 6 existing additive conviction rules into multiplicative factors and replaces the additive block in the synthesizer. Each factor is a float multiplier centered around `1.0` (no effect). Factors below `1.0` reduce conviction; above `1.0` boost it.

**Dependencies:** None. This section is fully independent and can be implemented in isolation.

**Kill-switch:** Controlled by `FEEDBACK_CONVICTION_MULTIPLICATIVE` env var (default `false`). When `false`, the existing additive logic runs unchanged. When `true`, the multiplicative path activates. This allows safe rollout.

**Cold-start:** Not applicable. All factors use data already available at synthesis time (ADX, HMM stability, weekly trend, regime source, ML signal, collector failures). If any input is missing, the corresponding factor defaults to `1.0`.

---

## Tests (Write First)

**Test file:** `tests/unit/test_conviction_multiplicative.py`

### Individual Factor Tests

```python
class TestConvictionMultiplicativeFactors:
    """Test each of the 6 multiplicative conviction factors in isolation."""

    def test_adx_factor_at_threshold(self):
        """ADX=15 (weak trend boundary) -> factor exactly 1.0."""
        # factor = 1.0 + 0.15 * min(1.0, (ADX - 15) / 35)
        # At ADX=15: 1.0 + 0.15 * 0.0 = 1.0

    def test_adx_factor_strong_trend(self):
        """ADX=50 (strong trend) -> factor = 1.15."""
        # At ADX=50: 1.0 + 0.15 * min(1.0, 35/35) = 1.15

    def test_adx_factor_moderate(self):
        """ADX=32.5 (midpoint) -> factor = 1.075."""
        # At ADX=32.5: 1.0 + 0.15 * min(1.0, 17.5/35) = 1.075

    def test_adx_factor_none_input(self):
        """ADX is None -> factor defaults to 1.0."""

    def test_adx_factor_below_threshold(self):
        """ADX < 15 -> factor = 1.0 (no negative contribution)."""
        # The min(1.0, (ADX-15)/35) clamps negative values via the formula
        # Implementer: decide whether to clamp to 1.0 or allow slight sub-1.0.
        # Plan intent: factor = 1.0 for ADX <= 15.

    def test_stability_factor_zero(self):
        """HMM stability=0.0 -> factor = 0.85."""
        # factor = 0.85 + 0.20 * stability
        # At 0.0: 0.85

    def test_stability_factor_one(self):
        """HMM stability=1.0 -> factor = 1.05."""
        # At 1.0: 0.85 + 0.20 = 1.05

    def test_stability_factor_midpoint(self):
        """HMM stability=0.5 -> factor = 0.95."""

    def test_stability_factor_none(self):
        """HMM stability is None -> factor defaults to 1.0."""

    def test_timeframe_factor_contradicting(self):
        """Weekly trend contradicts daily regime -> factor = 0.80."""
        # weekly=bullish, daily=trending_down  OR  weekly=bearish, daily=trending_up

    def test_timeframe_factor_agreeing(self):
        """Weekly and daily agree -> factor = 1.0."""

    def test_timeframe_factor_unknown(self):
        """Either weekly or daily is 'unknown' -> factor = 1.0."""

    def test_regime_agreement_disagree(self):
        """HMM and rule-based disagree -> factor = 0.85."""

    def test_regime_agreement_agree(self):
        """HMM and rule-based agree -> factor = 1.0."""

    def test_ml_confirmation_confirms(self):
        """ML direction matches rule-based direction -> factor = 1.10."""

    def test_ml_confirmation_no_ml(self):
        """No ML signal available -> factor = 1.0."""

    def test_ml_confirmation_disagrees(self):
        """ML direction opposes rule-based -> factor = 1.0 (no penalty, just no boost)."""

    def test_data_quality_technical_failure(self):
        """'technical' in collector_failures -> factor = 0.75."""

    def test_data_quality_regime_failure(self):
        """'regime' in collector_failures -> factor = 0.75."""

    def test_data_quality_both_failures(self):
        """Both technical and regime failed -> factor = 0.75 * 0.75 = 0.5625."""
        # Each failed collector is a separate 0.75 factor, they multiply.

    def test_data_quality_no_failures(self):
        """No collector failures -> factor = 1.0."""
```

### Combined Behavior Tests

```python
class TestConvictionMultiplicativeCombined:
    """Test the full multiplicative pipeline: base * f1 * f2 * ... * f6."""

    def test_all_factors_worst_case(self):
        """All factors at their worst -> product approximately 0.43."""
        # 1.0 (ADX<=15) * 0.85 (stability=0) * 0.80 (contradicting) *
        # 0.85 (regime disagree) * 1.0 (no ML) * 0.75 (technical failed) = ~0.434
        # With base conviction 0.50: 0.50 * 0.434 = 0.217

    def test_all_factors_best_case(self):
        """All factors at their best -> product approximately 1.35."""
        # 1.15 (ADX=50) * 1.05 (stability=1.0) * 1.0 (agreeing) *
        # 1.0 (agree) * 1.10 (ML confirms) * 1.0 (no failures) = ~1.328

    def test_final_clip_lower_bound(self):
        """Extreme reduction clips to 0.05, never below."""
        # base=0.10, worst factors -> result clips to 0.05

    def test_final_clip_upper_bound(self):
        """Extreme boost clips to 0.95, never above."""
        # base=0.90, best factors -> result clips to 0.95

    def test_missing_inputs_default_to_unity(self):
        """When all optional inputs are None/missing, all factors = 1.0."""
        # Result: adjusted conviction == base conviction (clipped)

    def test_factor_logging(self):
        """All 6 factor values are logged for calibration/debugging."""
        # Verify logger.debug call includes each factor name and value.
```

### Config Flag Tests

```python
class TestConvictionConfigFlag:
    """Test the FEEDBACK_CONVICTION_MULTIPLICATIVE kill switch."""

    def test_flag_false_uses_additive(self):
        """FEEDBACK_CONVICTION_MULTIPLICATIVE=false -> existing additive logic runs."""
        # ADX > 25 should add +0.10 (not multiply by 1.15-ish)

    def test_flag_true_uses_multiplicative(self):
        """FEEDBACK_CONVICTION_MULTIPLICATIVE=true -> multiplicative factors used."""
        # ADX=50 should multiply by 1.15 (not add 0.10)

    def test_flag_missing_defaults_to_false(self):
        """Env var not set -> defaults to false (additive, backward compatible)."""
```

---

## Implementation Details

### File to Modify

`src/quantstack/signal_engine/synthesis.py`

### What Changes

The conviction scaling block inside `_compute_bias_and_conviction()` (currently lines 383-420) is replaced with a branching path controlled by the config flag.

### The 6 Multiplicative Factors

Each factor is a function of inputs already available at synthesis time. No new data sources needed.

**Factor 1 -- ADX strength:**
```
adx_factor = 1.0 + 0.15 * min(1.0, max(0.0, (adx - 15)) / 35)
```
Smooth ramp from `1.0` at ADX=15 to `1.15` at ADX=50+. Below ADX=15, factor is `1.0`. If ADX is None, factor is `1.0`.

**Factor 2 -- Regime stability:**
```
stability_factor = 0.85 + 0.20 * hmm_stability
```
Linear from `0.85` (stability=0) to `1.05` (stability=1). If hmm_stability is None, factor is `1.0`.

**Factor 3 -- Timeframe agreement:**
```
timeframe_factor = 0.80 if weekly contradicts daily else 1.0
```
Contradiction means: (weekly=bullish AND daily=trending_down) OR (weekly=bearish AND daily=trending_up). If either is "unknown", no contradiction -- factor is `1.0`.

**Factor 4 -- Regime source agreement:**
```
regime_agreement_factor = 0.85 if regime_disagreement else 1.0
```
Uses the existing `regime.get("regime_disagreement")` boolean.

**Factor 5 -- ML confirmation:**
```
ml_confirmation_factor = 1.10 if ML direction matches rule-based direction else 1.0
```
ML direction is derived from the ML vote score (positive = bullish, negative = bearish). Rule-based direction is derived from the weighted score. If ML is unavailable or score is zero, factor is `1.0`. Note: this is a boost-only factor -- ML disagreement does not penalize (the ML signal already has a weight in the vote, applying a double penalty via disagreement would be excessive).

**Factor 6 -- Data quality:**
```
data_quality_factor = 0.75 for each of "technical" or "regime" in collector_failures
```
If both failed, factor is `0.75 * 0.75 = 0.5625`. This is a per-failure multiplicative penalty. If neither failed, factor is `1.0`.

### Composition Formula

```python
adjusted = base_conviction * adx_factor * stability_factor * timeframe_factor * regime_agreement_factor * ml_confirmation_factor * data_quality_factor
conviction = round(max(0.05, min(0.95, adjusted)), 3)
```

### Branching on Config Flag

Read `FEEDBACK_CONVICTION_MULTIPLICATIVE` from environment (default `"false"`). The branch should look approximately like:

```python
import os

use_multiplicative = os.getenv("FEEDBACK_CONVICTION_MULTIPLICATIVE", "false").lower() == "true"

if use_multiplicative:
    # Compute 6 factors (each defaults to 1.0 if input is missing)
    # Multiply base conviction by all factors
    # Log individual factors for calibration
    # Clip to [0.05, 0.95]
else:
    # Existing additive logic (unchanged)
```

The additive block must remain intact for backward compatibility when the flag is off. Do not delete it.

### Logging for Calibration

When multiplicative mode is active, log all factor values at debug level:

```python
logger.debug(
    "conviction_factors | symbol={} base={:.3f} adx={:.3f} stability={:.3f} "
    "timeframe={:.3f} regime_agree={:.3f} ml_confirm={:.3f} data_quality={:.3f} "
    "adjusted={:.3f}",
    symbol, base_conviction, adx_factor, stability_factor,
    timeframe_factor, regime_agreement_factor, ml_confirmation_factor,
    data_quality_factor, adjusted,
)
```

This log line is critical for post-deployment calibration. It allows correlation of factor values with trade outcomes to determine whether coefficients need tuning.

### Calibration Plan (Future)

Store factor inputs and conviction outcomes alongside trade results. Quarterly, compute which factors actually improve conviction accuracy (correlation between factor-adjusted conviction and trade outcome) and tune the coefficients (the `0.15`, `0.20`, `0.85`, etc.). This is a downstream concern and not part of this section's implementation.

---

## Equivalence Check

The multiplicative factors were designed to approximate the additive effects at typical base conviction levels. Key equivalences:

| Scenario | Additive Effect | Multiplicative (at base ~0.75) | Match? |
|----------|----------------|-------------------------------|--------|
| ADX > 25 | +0.10 | 0.75 * 1.10 = 0.825 (+0.075) | Close |
| HMM stability > 0.8 | +0.05 | 0.75 * 1.01 = 0.758 (+0.008) | Weaker |
| Weekly contradicts | -0.15 | 0.75 * 0.80 = 0.60 (-0.15) | Exact |
| Regime disagree | -0.10 | 0.75 * 0.85 = 0.6375 (-0.1125) | Close |
| ML confirms | +0.05 | 0.75 * 1.10 = 0.825 (+0.075) | Stronger |
| Technical failure | -0.20 | 0.75 * 0.75 = 0.5625 (-0.1875) | Close |

The multiplicative approach is deliberately different at extreme base conviction levels -- that is the point. At base 0.20, a `-0.15` additive penalty nearly eliminates conviction (0.05), while the `0.80` multiplicative factor gives `0.16` -- still low but proportionally fair.

---

## Rollback

Set `FEEDBACK_CONVICTION_MULTIPLICATIVE=false` (or remove the env var). The additive path executes, behavior is identical to pre-change. No data migration needed since this section does not touch the database.
