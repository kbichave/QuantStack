# Section 15: Regime Transition Detection

## Purpose

The HMM regime model identifies 4 states (LOW_VOL_BULL, HIGH_VOL_BULL, LOW_VOL_BEAR, HIGH_VOL_BEAR) but the trading system does not use the transition uncertainty information that the HMM already produces. Most losses cluster around regime transitions -- periods when the model is uncertain about which state the market is in. During these periods, the system trades with full conviction, which is the opposite of what it should do.

This section exposes filtered transition probabilities from the HMM, wires them into position sizing, adds vol-conditioned sub-regimes for finer-grained strategy selection, and defines degraded-mode behavior when the HMM is unavailable.

## Dependencies

- No hard dependencies on other sections. This section is in Batch 1 (parallelizable).
- Section 16 (config-flags-integration) depends on this section for the `FEEDBACK_TRANSITION_SIZING` kill-switch wiring.
- The `risk_sizing` node in `src/quantstack/graphs/trading/nodes.py` is the primary integration point. Other sections (StrategyBreaker from Section 3, Sharpe demotion from Section 12) also modify sizing in that same node -- the factors multiply together.

## Current State of the Code

**HMM model** (`src/quantstack/core/hierarchy/regime/hmm_model.py`):
- `HMMRegimeModel.predict()` already calls `self.model.predict_proba(features)` and stores the posteriors in `HMMRegimeResult.state_probabilities`.
- `regime_stability` is computed as `float(current_probs[current_state_idx])` -- i.e., the max filtered probability. This is exactly what we need.
- The `transition_matrix` (static `transmat_`) is returned but is NOT the right signal for transition detection. The static matrix gives the same transition probability regardless of recent observations.

**Regime collector** (`src/quantstack/signal_engine/collectors/regime.py`):
- `collect_regime()` calls `_try_hmm_regime()` which calls `model.predict()`.
- The collector output already includes `hmm_stability` and `hmm_probabilities` when HMM succeeds.
- It does NOT include `transition_probability` or `most_likely_next_regime`.
- When HMM fails, it falls back to `rule_based` with no stability/probability information.

**Risk sizing** (`src/quantstack/graphs/trading/nodes.py`, `risk_sizing` method around line 517):
- Currently computes Kelly fraction and alpha signal.
- Does not read transition probability or apply any transition-based sizing factor.

## Tests

All tests go in `tests/unit/test_regime_transitions.py`.

### Filtered transition probability

```python
def test_transition_probability_uses_predict_proba_not_transmat():
    """
    Transition probability must come from the filtered state probabilities
    (predict_proba output), not the static transmat_ matrix. Verify that
    transition_probability = 1.0 - max(state_probabilities).
    """

def test_high_uncertainty_yields_high_transition_probability():
    """
    When max filtered probability < 0.5 (HMM uncertain about current state),
    transition_probability should be > 0.5.
    """

def test_confident_state_yields_low_transition_probability():
    """
    When max filtered probability > 0.9 (HMM very confident),
    transition_probability should be < 0.1.
    """
```

### Sizing response tiers

```python
def test_low_transition_probability_no_sizing_adjustment():
    """P(transition) < 0.10 -> factor = 1.0 (no reduction)."""

def test_moderate_transition_probability_mild_reduction():
    """P(transition) = 0.20 -> factor = 0.75 (25% reduction)."""

def test_elevated_transition_probability_half_reduction():
    """P(transition) = 0.40 -> factor = 0.50 (50% reduction)."""

def test_high_transition_probability_severe_reduction():
    """P(transition) = 0.60 -> factor = 0.25 (75% reduction)."""
```

### Degraded mode

```python
def test_hmm_failure_defaults_transition_probability_to_zero():
    """
    When HMM fit fails (convergence, insufficient data), the regime collector
    must return transition_probability = 0.0 so that downstream sizing
    applies no penalty.
    """

def test_risk_sizing_handles_none_transition_probability():
    """
    If transition_probability is missing or None in the signal data,
    the risk_sizing node must default to factor = 1.0 (no adjustment).
    """
```

### Minimum tradeable size floor

```python
def test_compound_factors_below_minimum_skips_trade():
    """
    When Kelly * breaker * transition * other factors produce a position
    value < $100, the trade should be skipped entirely.
    """

def test_compound_factors_above_minimum_places_trade():
    """
    When compound factors produce a position value >= $100, the trade
    should proceed normally.
    """
```

### Vol-conditioned sub-regimes

```python
def test_low_vol_sub_regime_label():
    """
    When 20-day realized vol is < 30th percentile of trailing 252-day
    distribution, the sub-regime suffix should be '_low_vol'.
    """

def test_high_vol_sub_regime_label():
    """
    When 20-day realized vol is > 70th percentile of trailing 252-day
    distribution, the sub-regime suffix should be '_high_vol'.
    """
```

### Config flag

```python
def test_config_flag_false_disables_transition_sizing():
    """
    FEEDBACK_TRANSITION_SIZING=false -> transition_factor always 1.0,
    regardless of actual transition probability.
    """
```

## Implementation Details

### 1. Add `transition_probability` to regime collector output

**File:** `src/quantstack/signal_engine/collectors/regime.py`

In the `_try_hmm_regime()` function, after computing `result = model.predict(df)`, add three new keys to the returned dict:

- `transition_probability`: computed as `1.0 - max(result.state_probabilities.values())`. This is the probability of NOT being in the most likely state, using filtered posteriors (not the static transition matrix). When the HMM is confident (e.g., 0.95 probability of LOW_VOL_BULL), this yields 0.05. When uncertain (e.g., 0.45 for the top state), this yields 0.55.
- `state_probabilities`: already present as `hmm_probabilities` -- no change needed.
- `most_likely_next_regime`: the state name with the second-highest filtered probability. Useful for downstream consumers that want to know which regime is being transitioned into.

In the HMM failure and insufficient-data fallback paths, set `transition_probability` to `0.0` so that sizing applies no penalty. The rationale: when we have no HMM output, we should not reduce sizing based on a nonexistent signal. The existing `confidence: 0.0` already signals low trust.

### 2. Sizing response function

**File:** New helper in `src/quantstack/signal_engine/collectors/regime.py` or in the risk sizing module.

Define a function that maps transition probability to a sizing factor using a tiered approach:

```python
def transition_sizing_factor(transition_probability: float) -> float:
    """
    Map transition probability to a position sizing multiplier.

    Tiers:
        P < 0.10  -> 1.0   (no adjustment)
        0.10-0.30 -> 0.75  (mild reduction)
        0.30-0.50 -> 0.50  (moderate reduction)
        P > 0.50  -> 0.25  (severe reduction, but never zero)
    """
```

The tier boundaries are chosen to be conservative: the system should still trade during transitions (factor never reaches 0), but with significantly reduced size. The 0.25 floor ensures that even in maximum uncertainty, some position is maintained if all other factors (breaker, Kelly, Sharpe) are healthy.

### 3. Wire into `risk_sizing` node

**File:** `src/quantstack/graphs/trading/nodes.py`, within the `risk_sizing` method.

After the Kelly fraction computation and after any StrategyBreaker factor (from Section 3), read `transition_probability` from the symbol's regime signal data. Compute the transition factor using the function above. Multiply into the final size:

```
final_size = kelly_size * breaker_factor * transition_factor
```

Guard with the config flag: if `FEEDBACK_TRANSITION_SIZING` is not set or is `false`, the transition factor defaults to 1.0.

Guard against missing data: if `transition_probability` is `None` or absent from the signal data, default to factor 1.0.

### 4. Minimum tradeable size floor

**File:** `src/quantstack/graphs/trading/nodes.py`, within the `risk_sizing` method.

After all multiplicative factors are applied (Kelly x breaker x transition x any Sharpe demotion factor), check whether the resulting dollar position value falls below $100. If so, skip the trade and log:

```
logger.info(f"[risk_sizing] {symbol}: position ${value:.0f} below $100 minimum after compound adjustments — skipping")
```

This prevents micro-orders where commission costs exceed expected edge. The $100 threshold is appropriate for equities; for options, a different floor may apply (left to the options execution path).

### 5. Vol-conditioned sub-regimes

**File:** `src/quantstack/signal_engine/collectors/regime.py`

Add a helper function that computes the volatility percentile and returns a sub-regime suffix:

```python
def _vol_sub_regime(df: pd.DataFrame) -> str:
    """
    Classify current volatility into a sub-regime using 20-day realized
    vol relative to trailing 252-day distribution.

    Returns: 'low_vol' | 'normal_vol' | 'high_vol'
    """
```

Computation:
1. Calculate 20-day realized volatility (annualized std of daily returns * sqrt(252)).
2. Calculate the 30th and 70th percentiles of the trailing 252-day realized volatility series.
3. If current vol < 30th percentile: `low_vol`. If > 70th percentile: `high_vol`. Otherwise: `normal_vol`.

Add the result to the collector output as `vol_sub_regime`. The combined regime label becomes e.g., `trending_up_low_vol`, `ranging_high_vol`, etc. This is exposed in the output dict so that synthesis weight profiles can key on it.

**Weight profile extension:** Rather than defining 12 independent weight profiles (4 base regimes x 3 vol tiers), extend the existing 4 profiles with vol-aware adjustments:
- Low vol in trending regimes: boost trend-following signal weights by 10%.
- High vol in trending regimes: reduce trend-following, boost options/hedging signals by 10%.
- High vol in ranging regimes: boost mean-reversion signals by 10%.

These adjustments are small and additive to the base profile. They can be implemented as a post-processing step in synthesis after the base regime weights are selected.

### 6. EventBus integration

The `REGIME_CHANGE` event type already exists in the EventBus. When the regime collector detects a transition (current regime differs from previous), it should publish this event with an enriched payload that includes:

```python
{
    "symbol": symbol,
    "previous_regime": previous_regime,
    "new_regime": current_regime,
    "transition_probability": transition_probability,
    "state_probabilities": state_probs,
}
```

This does not require a new event type -- it extends the existing one.

## Cold-Start Behavior

When the HMM has not been fit (fewer than 120 bars of data) or fails to converge:
- `transition_probability` defaults to `0.0`
- The sizing factor defaults to `1.0` (no adjustment)
- The system trades normally using rule-based regime detection
- Vol sub-regime computation still works (it only needs 252 bars of OHLCV, not HMM)

This ensures new symbols or symbols with limited data are not penalized by missing HMM output.

## Config Flag

`FEEDBACK_TRANSITION_SIZING` (default: `false`)

When `false`:
- The regime collector still computes and returns `transition_probability` (data accumulates for analysis).
- The `risk_sizing` node ignores the transition probability and uses factor 1.0.
- Vol sub-regimes are still computed and available in signal data.

When `true`:
- The `risk_sizing` node applies the tiered transition sizing factor.
- The minimum tradeable size floor is enforced.

This allows the transition probability data to be collected and observed in Langfuse traces before it affects live sizing.

## Rollback

Set `FEEDBACK_TRANSITION_SIZING=false`. The transition factor reverts to 1.0. Regime collector output retains the new fields (harmless extra data). No schema changes are needed for this section -- all output is added to existing dict returns and signal data, not new DB tables.

## Key Design Decisions

1. **Filtered posteriors, not static transition matrix.** The `transmat_` gives the same transition probability regardless of recent data. The filtered posteriors from `predict_proba()` reflect actual observation-conditioned uncertainty. The HMM model already computes these -- we just need to expose them.

2. **Tiered response, not continuous function.** A continuous mapping from probability to factor (e.g., `1 - 0.75 * p`) would be smoother but harder to reason about and calibrate. Discrete tiers make the behavior transparent: traders and debuggers can immediately understand why sizing was reduced by looking at which tier the probability fell into.

3. **Factor floor of 0.25, not zero.** The transition sizing should never block a trade entirely on its own. If all other factors (breaker, Kelly, Sharpe) indicate the trade is healthy, a regime transition should reduce but not prevent it. The minimum tradeable size floor ($100) is the final gate that catches micro-orders from compound factor multiplication.

4. **Vol sub-regimes as adjustments, not independent profiles.** Defining 12 independent weight profiles would require extensive calibration data we don't have. Vol-aware adjustments to existing profiles are safer to bootstrap and easier to tune incrementally.
