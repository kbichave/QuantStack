# Section 04: Iron Condor Harvesting

## Objective

Build an iron condor harvesting strategy that collects premium in ranging, elevated-IV environments. Uses P06 structure building for condor construction, with active management rules for rolling tested sides and taking profits.

## Files to Create

### `src/quantstack/core/strategy/condor_harvesting.py`

**Key components:**

1. **`CondorConfig` dataclass**:
   - `iv_rank_min: float = 50.0` -- minimum IV rank for entry
   - `regime_required: str = "SIDEWAYS"` -- trend regime gate (only enter in ranging)
   - `short_delta_target: float = 0.16` -- delta of short strikes (~1 SD OTM)
   - `wing_width: int = 5` -- dollar width of each spread (configurable per underlying price)
   - `profit_target_pct: float = 0.50` -- close at 50% of max profit
   - `management_trigger_delta: float = 0.30` -- roll when short strike delta exceeds this
   - `max_holding_days: int = 45` -- aligned with typical 30-45 DTE entry
   - `min_dte: int = 21` -- minimum days to expiry for entry
   - `max_dte: int = 45` -- maximum days to expiry for entry

2. **`CondorHarvestingStrategy(Strategy)`** class:
   - `on_bar(state: MarketState) -> list[TargetPosition]`:
     - Guard 1: `state.iv_rank` >= `iv_rank_min`
     - Guard 2: `state.regime.trend_regime == regime_required`
     - Select strikes using `short_delta_target` (short put, short call) and `wing_width` (long put, long call)
     - Return 4-leg condor as linked `TargetPosition` entries with shared `trade_group_id`
     - Metadata: `{"strategy_type": "condor", "structure": "iron_condor", "short_put_strike": X, "short_call_strike": Y, ...}`
   - `compute_management_action(state: MarketState, position_metadata: dict) -> dict | None`:
     - If short put delta > `management_trigger_delta`: return roll-down spec
     - If short call delta > `management_trigger_delta`: return roll-up spec
     - If unrealized P&L >= `profit_target_pct * max_credit`: return close spec
     - Otherwise: return None (hold)
   - `should_exit(state, entry_metadata) -> bool`:
     - Profit target reached
     - Time stop (DTE <= 5 remaining)
     - Regime shifted away from SIDEWAYS
     - Short strike breached (underlying crosses short strike)

3. **`select_condor_strikes(spot: float, chain_data: dict, short_delta: float, wing_width: float) -> dict`**:
   - Finds strikes closest to target delta from options chain
   - Returns: `{"short_put": K1, "long_put": K1-wing, "short_call": K2, "long_call": K2+wing}`
   - Handles missing strikes by snapping to nearest available

4. **`compute_condor_metrics(short_put, long_put, short_call, long_call, credit_received) -> dict`**:
   - Max profit: credit received
   - Max loss: wing_width - credit
   - Breakevens: short_put - credit, short_call + credit
   - Risk/reward ratio

## Files to Modify

### `src/quantstack/core/strategy/__init__.py`

Add export: `from quantstack.core.strategy.condor_harvesting import CondorHarvestingStrategy, CondorConfig`

## Implementation Details

- Iron condors are defined-risk structures. Max loss = wing width minus credit received. This makes risk gate approval simpler than naked strategies.
- Strike selection uses delta, not fixed dollar offsets. This adapts to different underlying prices and volatility levels.
- The management rules are critical for performance. Rolling the tested side (when price approaches a short strike) recenters the position and collects additional credit.
- Regime gate is strict: only enter in SIDEWAYS regime. If regime shifts to BULL or BEAR after entry, `should_exit` triggers.
- The 50% profit target is industry-standard for condors -- statistically improves win rate vs holding to expiry.
- DTE <= 5 exit avoids gamma risk acceleration near expiration.
- Uses the `vol_strategy_signals` table (from section 01) for signal persistence.

## Test Requirements

File: `tests/unit/strategy/test_condor_harvesting.py`

1. **Entry gates**: Blocked when IV rank < threshold or regime != SIDEWAYS. Allowed when both pass.
2. **Strike selection**: Given a mock chain, selects strikes closest to target delta with correct wing width.
3. **Condor metrics**: Known strikes and credit produce correct max profit, max loss, breakevens.
4. **Profit target exit**: Unrealized P&L >= 50% of max profit triggers exit.
5. **Management -- roll trigger**: Short strike delta exceeding threshold generates roll specification.
6. **Regime shift exit**: Regime changing from SIDEWAYS to BULL triggers `should_exit`.
7. **DTE exit**: Position with DTE <= 5 triggers `should_exit`.
8. **Short strike breach**: Underlying crossing short strike triggers exit.
9. **Edge case -- no valid strikes**: Chain missing strikes near target delta returns empty list (no entry).

## Acceptance Criteria

- [ ] `CondorHarvestingStrategy` extends `Strategy` base class
- [ ] Strict regime + IV rank gating prevents entry in wrong conditions
- [ ] Strike selection adapts to options chain data (delta-based, not fixed)
- [ ] Management actions (roll, close at profit) computed correctly
- [ ] Four exits work independently: profit target, time/DTE, regime shift, breach
- [ ] All 9 unit tests pass
- [ ] Condor metrics match manual calculation
