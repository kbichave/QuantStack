# Section 7: IC Degradation to Weight Adjustment

## Overview

The signal engine currently uses static regime-conditional weights for its 14+ collectors. Even when a collector's Information Coefficient (IC) drops to zero, it retains its full static weight in synthesis. This section implements a continuous IC-based weight adjustment that multiplies static weights by an IC-derived factor, publishes degradation alerts, and rebalances weekly.

**Dependencies:**
- Section 03 (readpoint wiring): Wire 5 must be complete so ICAttributionTracker receives signal-outcome pairs
- Section 06 (eventbus extension): `SIGNAL_DEGRADATION` event type must exist in `EventType` enum

**Blocks:** Section 08 (signal correlation) builds on the same weight adjustment mechanism introduced here.

**Kill-switch flag:** `FEEDBACK_IC_WEIGHT_ADJUSTMENT` (env var, default `false`). When false, `ic_factor` always returns 1.0 -- data collection continues but weights are unaffected.

---

## Tests First

File: `tests/unit/test_ic_weight_adjustment.py`

### Sigmoid IC Factor Function

```python
class TestICFactorFunction:
    """Continuous sigmoid IC factor: 1 / (1 + exp(-50 * (ic - 0.02)))"""

    def test_healthy_ic_full_weight(self):
        """IC=0.05 should produce factor approximately 1.0."""
        # ic_factor(0.05) = 1 / (1 + exp(-50 * 0.03)) ≈ 1.0

    def test_threshold_ic_half_weight(self):
        """IC=0.02 (sigmoid center) should produce factor approximately 0.5."""
        # ic_factor(0.02) = 1 / (1 + exp(0)) = 0.5

    def test_zero_ic_near_zero_weight(self):
        """IC=0.00 should produce factor near 0.0."""
        # ic_factor(0.00) = 1 / (1 + exp(-50 * -0.02)) ≈ 0.0

    def test_negative_ic_near_zero_weight(self):
        """IC=-0.02 should produce factor effectively 0.0."""

    def test_smooth_transition_no_discrete_jumps(self):
        """Verify monotonic increase across IC range [-0.01, 0.05] with no jumps > 0.1 between adjacent points."""
        # Sample at 0.001 increments, assert each step < 0.1 change
```

### IC_IR Penalty

```python
class TestICIRPenalty:
    """When IC_IR (mean IC / std IC) < 0.1, apply 0.7x penalty for inconsistency."""

    def test_low_icir_applies_penalty(self):
        """IC_IR < 0.1 should multiply ic_factor by 0.7."""

    def test_healthy_icir_no_penalty(self):
        """IC_IR >= 0.1 should not apply penalty (multiplier = 1.0)."""
```

### Weight Floor Check

```python
class TestWeightFloorCheck:
    """If total effective weight across all collectors < 0.1, fall back to equal static weights."""

    def test_all_collectors_near_zero_ic_triggers_fallback(self):
        """When every collector has IC near zero, effective weights should equal static weights (no IC adjustment)."""

    def test_signal_degradation_event_published_on_floor_trigger(self):
        """SIGNAL_DEGRADATION event should be published via EventBus when floor is triggered."""

    def test_partial_degradation_no_fallback(self):
        """When some collectors have healthy IC, no fallback -- just let the healthy ones dominate."""
```

### Config Flag

```python
class TestICWeightConfigFlag:
    """FEEDBACK_IC_WEIGHT_ADJUSTMENT env var controls whether IC factors are applied."""

    def test_flag_false_ic_factor_always_one(self):
        """With flag=false, ic_factor() should return 1.0 regardless of IC value."""

    def test_flag_true_ic_factors_applied(self):
        """With flag=true, ic_factor() should return the sigmoid-computed value."""
```

### Cold-Start

```python
class TestICWeightColdStart:
    """Behavior when insufficient IC data exists."""

    def test_fewer_than_21_days_ic_data_returns_one(self):
        """With < 21 days of IC data for a collector, ic_factor should default to 1.0."""
```

---

## Implementation Details

### Core Formula

The weight adjustment multiplies each collector's static weight by an IC-derived factor:

```
effective_weight(collector, regime) = static_weight(collector, regime) * ic_factor(collector)
```

The `ic_factor` is a continuous sigmoid function of the collector's rolling 21-day IC:

```
ic_factor(ic) = 1 / (1 + exp(-50 * (ic - 0.02)))
```

This S-curve is centered at IC=0.02:
- IC > 0.04: factor approaches 1.0 (full weight)
- IC = 0.02: factor = 0.5 (half weight)
- IC < 0.00: factor approaches 0.0 (near-zero weight)

The continuous sigmoid avoids boundary oscillation. A collector hovering around IC=0.02 gets a stable ~0.5 factor instead of flipping between discrete tiers on successive days.

**IC_IR consistency penalty:** If `IC_IR = mean(IC) / std(IC) < 0.1`, multiply the factor by 0.7. This penalizes collectors whose IC is noisy even if the mean IC looks acceptable.

### IC Factor Helper Function

Create a helper function (location: `src/quantstack/signal_engine/ic_weights.py` or inline in synthesis) with signature:

```python
def compute_ic_factors(
    ic_data: dict[str, list[float]],
    min_observations: int = 21,
) -> dict[str, float]:
    """Compute IC-based weight adjustment factors for each collector.

    Args:
        ic_data: Mapping of collector_name -> list of daily IC values (most recent last).
                 Sourced from the nightly cross-sectional IC computation in signal_ic table.
        min_observations: Minimum number of IC observations required. Below this,
                          factor defaults to 1.0 (cold-start).

    Returns:
        Mapping of collector_name -> ic_factor in [0.0, 1.0].
    """
```

The function should:
1. For each collector, check if `len(ic_values) >= min_observations`. If not, return 1.0.
2. Compute `rolling_ic = mean(ic_values[-21:])` (most recent 21-day window).
3. Compute `ic_ir = mean(ic_values[-21:]) / std(ic_values[-21:])` if std > 0, else 0.0.
4. Apply sigmoid: `factor = 1 / (1 + exp(-50 * (rolling_ic - 0.02)))`.
5. If `ic_ir < 0.1`: `factor *= 0.7`.
6. Return the factor.

### IC Data Source

The primary IC source is the **nightly cross-sectional IC** stored in the `signal_ic` table (computed by the existing `run_ic_computation()` supervisor batch). This is unbiased because it covers all symbols with signals, not just traded symbols. ICAttributionTracker (wired in Section 03, Wire 5) provides supplementary per-trade granularity but does not drive weight adjustments alone -- this avoids survivorship bias from only tracking traded symbols.

### Synthesis Integration

File: `src/quantstack/signal_engine/synthesis.py`

Modify the synthesizer to accept an optional `ic_adjustments: dict[str, float]` parameter. When provided (and `FEEDBACK_IC_WEIGHT_ADJUSTMENT=true`), multiply each collector's static weight by its IC factor before normalizing weights.

The calling code (signal engine entry point) is responsible for:
1. Reading the latest IC data from `signal_ic` table
2. Calling `compute_ic_factors()` to produce the adjustment dict
3. Passing it to the synthesizer

When the flag is false or ic_adjustments is None, synthesis uses static weights unchanged.

### Weight Floor Safety Check

After applying IC factors to all collectors, sum the effective weights. If `total_effective_weight < 0.1`:
- Fall back to equal static weights (ignore IC adjustments entirely for this synthesis run)
- Publish a `SIGNAL_DEGRADATION` event via EventBus with payload:
  ```python
  {
      "type": "floor_triggered",
      "total_effective_weight": total_effective_weight,
      "collector_factors": {name: factor for each collector},
      "regime": current_regime,
  }
  ```
- Log at WARNING level: all collectors have degraded IC simultaneously, which likely indicates a data quality issue rather than genuine signal decay

This prevents division-by-zero or NaN conviction from pathological IC collapse.

### SIGNAL_DEGRADATION Event Publishing

Beyond the floor check, publish a `SIGNAL_DEGRADATION` event whenever a collector's IC drops below 0.02 from a previously healthy level (IC was >= 0.02 in the prior computation). Payload:

```python
{
    "type": "collector_degraded",
    "collector": collector_name,
    "current_ic": current_rolling_ic,
    "previous_ic": previous_rolling_ic,
    "regime": current_regime,
}
```

The research graph polls for these events and queues an investigation task to determine why the collector's predictive power has declined.

Requires `SIGNAL_DEGRADATION` to exist in the `EventType` enum (added by Section 06).

### Rebalancing Frequency

IC factors are **computed daily** (after the nightly `run_ic_computation()` batch), but synthesis weights are **updated weekly** to avoid excessive churn. Implementation:

- Store the last weight update timestamp
- On each synthesis run, check if 7+ calendar days have passed since the last update
- If yes, recompute IC factors from the latest IC data and cache them
- If no, reuse the cached factors from the last weekly computation
- Store each weekly weight snapshot (collector -> effective_weight) for audit purposes

This means a collector that degrades mid-week continues at its previous weight until the next weekly rebalance, which is acceptable because daily fluctuations in IC are noisy.

### Cold-Start Behavior

When a collector has fewer than 21 days of IC data in the `signal_ic` table, `ic_factor` defaults to 1.0 (full static weight). This means:
- New collectors get their full static weight until enough data accumulates
- After system initialization, all collectors run at static weights for the first 21 trading days
- The flag `FEEDBACK_IC_WEIGHT_ADJUSTMENT` should remain `false` during initial deployment anyway (safe rollout), so cold-start is doubly protected

### Integration with Section 08 (Signal Correlation)

Section 08 adds a `correlation_penalty` factor per collector. The final effective weight stacks both adjustments:

```
final_weight = static_weight * ic_factor * correlation_penalty
```

This section only implements the `ic_factor` portion. Section 08 extends the same mechanism with the correlation penalty. The synthesis integration point should be designed to accept both adjustment types (e.g., a single dict of combined multipliers, or separate dicts that get merged).

### Rollback

Set `FEEDBACK_IC_WEIGHT_ADJUSTMENT=false`. IC factors revert to 1.0 for all collectors. Data collection (IC computation, ICAttributionTracker) continues unaffected -- only the weight adjustment is disabled. No data migration or cleanup required.

---

## Files Modified

| File | Change |
|------|--------|
| `src/quantstack/signal_engine/synthesis.py` | Accept `ic_adjustments` param, apply IC factors to static weights before normalizing, weight floor check |
| `src/quantstack/signal_engine/ic_weights.py` (new) | `compute_ic_factors()` helper with sigmoid function, IC_IR penalty, cold-start check |
| Signal engine entry point (e.g., `engine.py` or the graph node calling synthesis) | Read IC data from `signal_ic` table, call `compute_ic_factors()`, pass to synthesizer |
| `tests/unit/test_ic_weight_adjustment.py` (new) | All tests listed above |

---

## Checklist

- [ ] Implement `compute_ic_factors()` with sigmoid function and IC_IR penalty
- [ ] Add cold-start guard (< 21 days -> factor 1.0)
- [ ] Modify synthesis to accept and apply IC adjustments
- [ ] Implement weight floor check with fallback to equal static weights
- [ ] Publish SIGNAL_DEGRADATION events (floor trigger and per-collector degradation)
- [ ] Implement weekly rebalancing with cached factors
- [ ] Wire config flag `FEEDBACK_IC_WEIGHT_ADJUSTMENT` (default false)
- [ ] Store weekly weight snapshots for audit
- [ ] Write all tests and verify they pass
