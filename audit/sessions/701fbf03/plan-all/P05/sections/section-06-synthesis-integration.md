# Section 06: Synthesis Integration

## Objective

Wire all P05 features from Sections 02-05 into the synthesis hot path in `src/quantstack/signal_engine/synthesis.py`. This section describes the exact integration points and ordering of changes within `_compute_bias_and_conviction()` and `synthesize()`.

## Dependencies

- **Section 01** (schema migrations): all 4 tables must exist
- **Section 02** (IC weight precompute): `get_precomputed_weights()` must be implemented
- **Section 03** (transition zone): `transition_zone` field on SymbolBrief, `transition_position_sizing_enabled()` flag
- **Section 04** (conviction calibration): metadata enrichment, `get_calibrated_conviction_params()`
- **Section 05** (ensemble A/B): recording logic, `_get_active_ensemble_method()`, `_ENSEMBLE_METHOD_MAP`

## Files to Modify

1. **`src/quantstack/signal_engine/synthesis.py`** -- all integration happens here

## Integration Map

The changes below are numbered in order of execution within the synthesis flow. Each change references the section that implements the underlying logic.

### Change 1: Replace ICAttributionTracker instantiation with precomputed lookup (Section 02)

**Location**: Lines 530-544 of `_compute_bias_and_conviction()`.

**Before:**
```python
        # Wire 5b: IC-driven regime-conditioned weights (static fallback if insufficient data)
        try:
            from quantstack.config.feedback_flags import ic_driven_weights_enabled
            if ic_driven_weights_enabled():
                from quantstack.learning.ic_attribution import ICAttributionTracker
                ic_weights = ICAttributionTracker().get_weights_for_regime(
                    trend, window=63, min_days=60,
                )
                if ic_weights:
                    weights = ic_weights
                    logger.debug(
                        "ic_driven_weights | regime=%s weights=%s", trend, ic_weights,
                    )
        except Exception:
            pass  # fall back to static weights silently
```

**After:**
```python
        # Wire 5b: IC-driven regime-conditioned weights (precomputed, Section 02)
        try:
            from quantstack.config.feedback_flags import ic_driven_weights_enabled
            if ic_driven_weights_enabled():
                from quantstack.learning.ic_attribution import get_precomputed_weights
                ic_weights = get_precomputed_weights(trend)
                if ic_weights:
                    weights = ic_weights
                    logger.debug(
                        "ic_driven_weights | regime=%s weights=%s (precomputed)", trend, ic_weights,
                    )
        except Exception:
            pass  # fall back to static weights silently
```

**Why**: `get_precomputed_weights()` is a single SELECT query with a 7-day staleness check. `ICAttributionTracker()` loads ALL observations from the DB and computes Spearman IC in-process. The precomputed path is O(1) vs O(n) where n is observation count.

**Fallback behavior**: If `get_precomputed_weights()` returns None (no data, stale, or error), weights are unchanged (static profile from `_get_weights()`). Same fallback as before.

### Change 2: IC gate and correlation penalty (unchanged)

**Location**: Lines 546-579. These remain as-is. They apply post-hoc adjustments to whatever weights are active (static or precomputed). The precomputed weights already incorporate these adjustments during batch computation (Section 02 steps 3-5), but the real-time IC gate provides an additional safety net if a collector degrades mid-week.

**Decision**: Keep the real-time IC gate active even when using precomputed weights. The weekly batch may miss a sudden collector degradation. The real-time gate catches it within 1 day.

### Change 3: Ensemble method selection from config (Section 05)

**Location**: Lines 581-587.

**Before:**
```python
        from quantstack.config.feedback_flags import ensemble_ab_test_enabled
        if ensemble_ab_test_enabled() and symbol:
            ensemble_fn = _ENSEMBLE_METHODS[hash(symbol) % len(_ENSEMBLE_METHODS)]
        else:
            ensemble_fn = _ensemble_weighted_avg
```

**After:**
```python
        from quantstack.config.feedback_flags import ensemble_ab_test_enabled
        ensemble_fn = _ensemble_weighted_avg
        if ensemble_ab_test_enabled() and symbol:
            active_method = _get_active_ensemble_method()
            ensemble_fn = _ENSEMBLE_METHOD_MAP.get(active_method, _ensemble_weighted_avg)
```

**Supporting code**: Add `_ENSEMBLE_METHOD_MAP`, `_get_active_ensemble_method()`, and `_active_ensemble_cache` as module-level definitions (see Section 05 for implementation).

### Change 4: Record ensemble A/B results (Section 05)

**Location**: After line 587 (after `score = ensemble_fn(scores, weights)`), before the transition dampening block.

Insert the recording block from Section 05, Step 1. All 3 methods' outputs are recorded for every symbol when A/B test is enabled.

### Change 5: Set transition_zone flag (Section 03)

**Location**: In `synthesize()`, before the SymbolBrief construction (line 368).

```python
        # P05 §5.2: Determine transition zone for downstream sizing
        in_transition_zone = False
        transition_prob = regime.get("transition_probability")
        if transition_prob is not None and transition_prob > 0.3:
            in_transition_zone = True
```

Then in the SymbolBrief constructor, add `transition_zone=in_transition_zone`.

### Change 6: Enrich signals metadata with conviction_factors (Section 04)

**Location**: Lines 332 of `synthesize()`, the metadata dict in the signals INSERT.

**Before:**
```python
_json_synth.dumps({"votes": vote_scores, "weights": final_weights}),
```

**After:**
```python
_json_synth.dumps({
    "votes": vote_scores,
    "weights": final_weights,
    "conviction_factors": conviction_factor_breakdown,
}),
```

The `conviction_factor_breakdown` variable is already available at this point (returned from `_compute_bias_and_conviction()` on line 303).

## Execution Order Summary

Within `_compute_bias_and_conviction()`:
1. Vote score computation (existing, unchanged)
2. Static weight profile selection (existing, unchanged)
3. **Precomputed IC weights** (Change 1, replaces ICAttributionTracker)
4. IC gate (existing, unchanged -- additional safety net)
5. Correlation penalty (existing, unchanged)
6. **Ensemble method from config** (Change 3)
7. Score computation via selected ensemble method (existing)
8. **Record A/B results** (Change 4)
9. Transition dampening (existing, unchanged)

Within `synthesize()`:
10. Signals INSERT with **enriched metadata** (Change 6)
11. SymbolBrief construction with **transition_zone** (Change 5)

## Flag Summary

All features are independently gated:

| Feature | Flag | Default | Section |
|---------|------|---------|---------|
| IC-driven weights | `FEEDBACK_IC_DRIVEN_WEIGHTS` | false | 02 |
| IC gate | `FEEDBACK_IC_GATE` | false | existing |
| Correlation penalty | `FEEDBACK_CORRELATION_PENALTY` | false | existing |
| Ensemble A/B test | `FEEDBACK_ENSEMBLE_AB_TEST` | false | 05 |
| Transition dampening | `FEEDBACK_TRANSITION_SIGNAL_DAMPENING` | false | existing |
| Transition position sizing | `FEEDBACK_TRANSITION_POSITION_SIZING` | false | 03 |
| Conviction calibration | (always collects; calibration is batch) | N/A | 04 |

Metadata enrichment (conviction_factors in signals INSERT) is unconditional -- it's a data collection change with zero runtime cost.

## Edge Cases

1. **Multiple flags enabled simultaneously**: Each feature operates independently. The weight pipeline is: static -> precomputed IC override -> IC gate -> correlation penalty -> ensemble method. Each step is additive/multiplicative.
2. **Precomputed weights + real-time IC gate conflict**: If precomputed weights include a collector that the real-time gate wants to zero out, the gate wins (it runs after). This is correct -- the gate is the safety net.
3. **Ensemble method change mid-day**: The 1-hour cache means synthesis may use the old method for up to 1 hour after a promotion. Acceptable.
4. **Import failures**: Every integration point is wrapped in try/except with silent fallback. Import errors (missing module) are caught. This matches existing patterns throughout the file.

## Tests

The integration tests for this section are covered by the individual section tests (02-05) and the integration test in Section 09. No additional unit tests needed for this section.

## Verification

After all sections are implemented, run synthesis for a test symbol and verify:
```python
from quantstack.signal_engine.synthesis import RuleBasedSynthesizer

synth = RuleBasedSynthesizer()
brief = synth.synthesize(
    symbol="AAPL",
    technical={"rsi_14": 45, "macd_hist": 0.5, "bb_pct": 0.5, "adx_14": 30, "close": 200},
    regime={"trend_regime": "trending_up", "transition_probability": 0.4},
    volume={}, risk={}, events={}, fundamentals={},
    collector_failures=[],
)
print(brief.transition_zone)       # True (transition_probability > 0.3)
print(brief.conviction_factors)    # dict with 6 factor values
```
