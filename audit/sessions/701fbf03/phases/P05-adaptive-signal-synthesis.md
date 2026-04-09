# P05: Adaptive Signal Synthesis

**Objective:** Replace static signal weight profiles with IC-driven adaptive weights, regime transition detection, and conviction calibration.

**Scope:** signal_engine/synthesis.py, learning/ic_attribution.py

**Depends on:** P01 (IC tracking must exist)

**Enables:** P10 (Meta-Learning)

**Effort estimate:** 1 week

---

## What Changes

### 5.1 IC-Driven Weight Adjustment
- Replace static `_WEIGHT_PROFILES` with dynamic weights from `ICAttribution.get_weights()`
- Update weights weekly from rolling 63-day IC per collector per regime
- Fall back to static profiles if insufficient data (<60 days)

### 5.2 Regime Transition Detection (QS-S7)
- Extract HMM transition probabilities: `P(transition) > 0.3 → transition zone`
- During transitions: reduce all signal weights by 50%, halve position sizes
- Add vol-conditioned sub-regimes: trending_up_low_vol vs trending_up_high_vol

### 5.3 Empirical Conviction Calibration (QS-S8)
- Replace additive adjustments (+0.10 for ADX, -0.15 for conflict) with multiplicative factors
- Calibrate factors quarterly from realized signal-to-return performance
- `adjusted = base * adx_factor * stability_factor * conflict_factor`

### 5.4 Signal Ensemble Methods
- Instead of weighted average, try: median (robust to outliers), trimmed mean (drop extremes)
- A/B test ensemble method vs weighted average for 30 days

## Files to Create/Modify

| File | Change |
|------|--------|
| `src/quantstack/signal_engine/synthesis.py` | Dynamic weights, transition detection, calibrated adjustments |
| `src/quantstack/learning/ic_attribution.py` | Weekly weight computation job |
| `src/quantstack/signal_engine/collectors/regime.py` | Expose transition probabilities |

## Acceptance Criteria

1. Signal weights differ from static profiles (driven by IC data)
2. Regime transition zones detected and exposure reduced
3. Conviction adjustments multiplicative, not additive
4. A/B test framework compares ensemble methods
