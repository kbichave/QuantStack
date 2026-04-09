# Section 06: Counterfactual Analysis

## Objective

Build the synthetic control and counterfactual analysis module for post-trade attribution. Answers: "How much of this trade's return was causal alpha vs market movement?" This runs after trade exit and feeds into research prioritization.

## Dependencies

None (can be implemented in parallel with Section 01).

## Files to Create

### `src/quantstack/core/causal/counterfactual.py`
- **Class `SyntheticControlBuilder`**: constructs a synthetic control portfolio for counterfactual analysis.
- **Method `build_control(target_symbol: str, control_pool: list[str], pre_period: tuple[date, date], store: DataStore) -> SyntheticControl`**:
  - Fetch price series for target and all control pool symbols over the pre-period.
  - Find optimal weights via constrained optimization: minimize `||target_returns - weighted_sum(control_returns)||` subject to weights >= 0, sum(weights) = 1.
  - Use `scipy.optimize.minimize` with SLSQP method.
  - Return `SyntheticControl` with weights, pre-period fit quality (R-squared), and the control symbols used.
- **Method `estimate_counterfactual(target_symbol: str, control: SyntheticControl, post_period: tuple[date, date], store: DataStore) -> CounterfactualResult`**:
  - Apply the pre-period weights to post-period control returns.
  - Compute counterfactual return (what target would have done without the trade signal).
  - Compute causal alpha = actual_return - counterfactual_return.
  - Statistical significance via permutation inference on pre-period residuals.
- **Function `run_counterfactual(trade_id: int) -> CounterfactualResult`**:
  - Convenience function: look up trade entry/exit dates from positions table, determine pre-period (2x trade duration before entry), build control from sector peers, estimate counterfactual.

### `src/quantstack/core/causal/models.py` (extend)
- **Dataclass `SyntheticControl`**: target_symbol, control_weights (dict[str, float]), pre_period_r_squared, control_symbols_used (list[str]).
- **Dataclass `CounterfactualResult`**: trade_id (int | None), actual_return (float), counterfactual_return (float), causal_alpha (float), p_value (float), pre_period_fit_r2 (float), control_weights (dict[str, float]).

## Files to Modify

### `src/quantstack/db.py`
- Extend the `trade_outcomes` or `positions` table with columns (if not already present, add via ALTER TABLE in ensure_tables):
  ```sql
  ALTER TABLE positions ADD COLUMN IF NOT EXISTS counterfactual_return REAL;
  ALTER TABLE positions ADD COLUMN IF NOT EXISTS causal_alpha REAL;
  ```

### `src/quantstack/execution/trade_service.py`
- In the trade exit/close logic, add a hook to trigger counterfactual analysis asynchronously after trade closure.
- Store `counterfactual_return` and `causal_alpha` in the position record.

## Implementation Details

1. **Synthetic Control Method**: Based on Abadie et al. (2010). The key idea: find a weighted combination of untreated units (stocks not traded) that closely matches the treated unit (stock traded) in the pre-intervention period. The post-intervention divergence is the treatment effect.

2. **Control Pool Selection**:
   - Default: same-sector stocks from the universe (via `universe.py`).
   - Filter: must have data for the full pre+post period, market cap within 0.5x-2x of target.
   - Minimum 5 control stocks required. If fewer available, widen to adjacent sectors.

3. **Pre-Period Calibration**:
   - Pre-period length: 2x the trade holding period (minimum 20 trading days).
   - Pre-period ends at trade entry date.
   - Good fit threshold: R-squared > 0.8. If below, flag the result as low confidence.

4. **Optimization**:
   - Objective: minimize MSE between target and synthetic control returns over pre-period.
   - Constraints: weights in [0, 1], sum to 1 (convex combination, no shorting the control).
   - Solver: SLSQP from scipy.

5. **Attribution Storage**: `causal_alpha` is the key metric for research prioritization. Strategies with consistently high causal_alpha get more research allocation. Strategies where alpha is mostly market movement get deprioritized.

6. **Async Execution**: Counterfactual analysis is computationally moderate but not latency-sensitive. Run via `asyncio.to_thread` after trade close. If it fails, log warning but do not block the trade exit flow.

## Test Requirements

- **Perfect synthetic control**: Create data where target = 0.5 * stock_A + 0.5 * stock_B. Verify weights recovered and counterfactual matches actual.
- **Known treatment effect**: Target diverges from control by exactly 5% after treatment date. Verify causal_alpha = 0.05.
- **Poor fit detection**: Control pool cannot match target (R-squared < 0.8). Verify low-confidence flag.
- **Edge cases**: Single control stock, very short pre-period (< 20 days), missing data in control pool.
- **Round-trip with DB**: Create a position, run counterfactual, verify causal_alpha stored in positions table.

## Acceptance Criteria

- [ ] `SyntheticControlBuilder.build_control()` finds optimal weights via SLSQP optimization
- [ ] Pre-period R-squared correctly measures fit quality
- [ ] `estimate_counterfactual()` correctly computes causal alpha = actual - counterfactual
- [ ] Known treatment effects in synthetic data are recovered
- [ ] Poor pre-period fit flagged (R-squared < 0.8)
- [ ] `positions` table extended with `counterfactual_return` and `causal_alpha` columns
- [ ] Trade exit flow triggers counterfactual analysis without blocking
- [ ] Tests pass: `uv run pytest tests/unit/core/causal/test_counterfactual.py`
