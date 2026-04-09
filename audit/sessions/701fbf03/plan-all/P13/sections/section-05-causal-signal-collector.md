# Section 05: Causal Signal Collector

## Objective

Build the signal engine collector that converts active causal factors into trading signals. Integrates with the existing `RuleBasedSynthesizer` as a new signal source alongside technical, sentiment, and ML signals.

## Dependencies

- Section 04 (Causal Factor Library) -- reads active factors from the library.

## Files to Create

### `src/quantstack/signal_engine/collectors/causal.py`
- **Async function `collect_causal_signal(symbol: str, store: DataStore) -> dict[str, Any]`**:
  - Follow the same collector interface as `collect_ml_signal` in `collectors/ml_signal.py`.
  - Load all active causal factors from `CausalFactorLibrary.get_active_factors()`.
  - For each active factor, compute the current CATE for the given symbol:
    - Fetch current feature values for the symbol (treatment variable value).
    - Look up the factor's ATE and weight by: `abs(ate) * regime_stability_score * refutation_confidence`.
    - Refutation confidence = `min(refutation_placebo_p, 1.0 - refutation_subset_cv)`.
  - Aggregate across all active factors into a single causal signal score.
  - Return dict with keys:
    - `causal_signal`: float -1 to 1 (negative = bearish causal evidence, positive = bullish)
    - `causal_confidence`: float 0-1 (weighted average of factor confidences)
    - `causal_active_factors`: int (number of active factors contributing)
    - `causal_top_factor`: str (name of highest-weight contributing factor)
    - `causal_regime_stability`: float (average regime stability across active factors)
  - Returns `{}` if no active causal factors exist (expected early on).
  - Never raises -- catches all exceptions and returns `{}` with a warning log.

## Files to Modify

### `src/quantstack/signal_engine/synthesis.py`
- Add `"causal"` to the weight profiles in `_WEIGHT_PROFILES`. Initial weight: 0.05 across all regime profiles (stolen proportionally from existing weights).
- Update the synthesis logic to incorporate `causal_signal` when present in the collector output.
- When causal signal is absent, redistribute its 0.05 weight proportionally to other active signals (same pattern as ML signal redistribution).

### `src/quantstack/signal_engine/collectors/__init__.py`
- Register `collect_causal_signal` in the collector registry so the signal engine invokes it.

### `src/quantstack/shared/schemas.py`
- Add optional `causal_signal` field to `SymbolBrief`:
  ```python
  causal_signal: float | None = Field(default=None, description="Causal factor signal (-1 to 1)")
  causal_confidence: float | None = Field(default=None, ge=0, le=1)
  causal_active_factors: int = Field(default=0)
  ```

## Implementation Details

1. **Collector Interface Compliance**: The collector must match the pattern established by `collect_ml_signal`:
   - Async entry point wrapping a sync implementation via `asyncio.to_thread`.
   - Accept `(symbol: str, store: DataStore)` parameters.
   - Return `dict[str, Any]` (empty on failure).
   - Never raise exceptions.

2. **CATE Computation**: For real-time signal generation, do NOT re-run full DML inference. Instead:
   - Use the stored ATE from the factor library as the base effect size.
   - Multiply by a binary indicator: 1 if the treatment condition is currently active for the symbol (e.g., insider_buy happened in the last N days), 0 otherwise.
   - Scale by confidence weights.

3. **Signal Aggregation**:
   - For each active factor where the treatment is currently present: `contribution = sign(ate) * abs(ate) * weight`.
   - Sum contributions and clip to [-1, 1].
   - Confidence = weighted average of individual factor confidences.

4. **Weight Profile Update**: Existing weights in `_WEIGHT_PROFILES` sum to 1.0. Subtract 0.05 proportionally from all existing keys and add `"causal": 0.05`. Example for `trending_up`: trend goes from 0.35 to ~0.332, etc.

5. **Graceful Bootstrap**: The collector returns `{}` until the first causal factors are discovered and activated. This is the expected state for weeks/months until the research graph populates the factor library.

## Test Requirements

- **Mock factor library**: Create mock active factors, verify collector computes correct signal values.
- **Empty library**: Verify `{}` returned when no active factors exist.
- **Treatment condition check**: Mock a factor for insider_buy. With insider_buy present, signal should be non-zero. Without, signal should be zero.
- **Signal aggregation**: Multiple factors with mixed signs. Verify correct aggregation and clipping.
- **Synthesis integration**: Verify `_WEIGHT_PROFILES` weights still sum to 1.0 after adding causal.
- **SymbolBrief schema**: Verify new causal fields are optional and default correctly.

## Acceptance Criteria

- [ ] `collect_causal_signal()` follows the standard collector interface (async, never raises)
- [ ] Returns correct signal values when active factors exist and treatment conditions are met
- [ ] Returns `{}` gracefully when no factors are active
- [ ] `_WEIGHT_PROFILES` updated with `"causal": 0.05`, all profiles still sum to 1.0
- [ ] `SymbolBrief` has optional causal signal fields
- [ ] Synthesis correctly incorporates/redistributes causal weight
- [ ] Tests pass: `uv run pytest tests/unit/signal_engine/test_causal_collector.py`
