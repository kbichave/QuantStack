# Section 08: Unit Tests (Integration and Edge Cases)

**Depends on:** All prior sections (01-07)

## Objective

Comprehensive integration tests and edge-case coverage that validate the P08 subsystem works end-to-end. Individual unit tests are defined in each section; this section covers cross-section integration, shared edge cases, and regression scenarios.

## Files to Create

### `tests/unit/strategy/test_vol_strategy_integration.py`

**Integration tests across vol strategies:**

1. **Strategy selection by regime**: Given a `MarketState` with specific regime and IV data, verify the correct strategy is selected by the market-making agent logic:
   - SIDEWAYS + IV rank 60 -> `CondorHarvestingStrategy`
   - BULL + IV rank 70 + IV > RV -> `VolArbStrategy`
   - Any regime + RV > IV by threshold -> `GammaScalpingStrategy`
   - High implied correlation -> `DispersionStrategy`

2. **Strategy-to-hedge pipeline**: A `VolArbStrategy` entry produces a `TargetPosition` with `hedge_required: True`. Verify the `HedgingEngine` picks it up and produces a `HedgeAction` when delta drifts.

3. **Strategy-to-attribution pipeline**: A closed condor position flows through `attribute_vol_pnl` and produces correct `VolPnLComponents` with positive theta P&L and near-zero delta P&L.

4. **Mutual exclusion**: Two strategies do not fire simultaneously for the same symbol in the same cycle (e.g., vol arb and condor on the same symbol).

5. **Portfolio vega limit across strategies**: Three simultaneous vol positions approaching the $5,000 vega limit. Fourth proposal is blocked by risk gate.

### `tests/unit/strategy/test_vol_strategy_edge_cases.py`

**Shared edge cases:**

1. **No options chain data**: All four strategies return empty list when options chain is unavailable.
2. **Stale IV surface**: IV surface older than 1 trading day treated as unavailable.
3. **Zero volatility**: RV = 0 and IV = 0 -- no division by zero, no signal generated.
4. **Single-symbol universe**: All strategies function correctly with a one-symbol universe.
5. **Concurrent exits**: Two vol positions trigger exit simultaneously. Both exit signals generated correctly without interference.
6. **NaN/None in features**: Strategy handles None values in `features` dict gracefully (returns empty list, does not raise).
7. **Empty portfolio**: Market-making agent handles empty portfolio (no existing positions) without error.
8. **Regime transition mid-cycle**: Regime changes between entry evaluation and management check. Management uses current regime, not entry regime.

### `tests/unit/strategy/test_vol_schema.py`

**Database schema tests:**

1. **Table creation idempotent**: `ensure_schema()` can be called twice without error.
2. **`vol_strategy_signals` unique constraint**: Duplicate (symbol, date, strategy_type) raises appropriate error.
3. **`dispersion_trades` JSONB**: Components field correctly stores and retrieves JSON data.
4. **New columns on `strategy_outcomes`**: `delta_pnl`, `gamma_pnl`, `theta_pnl`, `vega_pnl`, `vol_pnl` columns exist and accept float values.

### `tests/unit/graphs/test_market_maker_integration.py`

**Agent integration tests:**

1. **Full cycle**: Market-making node receives state, scans for mispricings, selects strategy, generates proposal, passes through risk check. Verify the full data flow end-to-end with mocked market data.
2. **Tool resolution**: All tools listed in `options_market_maker` agent config resolve from `TOOL_REGISTRY`.
3. **Position management cycle**: Open vol position receives management action (roll, close) from the correct strategy's `compute_management_action`.

## Files to Modify

None -- this section only creates test files.

## Implementation Details

- All integration tests use mocked market data and database connections. No real API calls or database writes.
- Use `pytest` fixtures for common `MarketState` construction with vol-specific fields populated.
- Mock the options chain data with realistic strike/delta/IV distributions.
- The strategy selection integration test should instantiate the actual strategy classes and call `on_bar` to verify real behavior, not mock the strategy logic.
- Database schema tests can use a temporary in-memory database or mock the `db_conn` context manager.
- Tests should verify log output for warning cases (sparse IV, missing data) using `caplog` fixture.

## Test Requirements

This section IS the test requirements. Total test count:

| Test file | Tests |
|-----------|-------|
| `test_vol_strategy_integration.py` | 5 |
| `test_vol_strategy_edge_cases.py` | 8 |
| `test_vol_schema.py` | 4 |
| `test_market_maker_integration.py` | 3 |
| **Total** | **20** |

Combined with per-section unit tests:
- Section 01: 6 tests
- Section 02: 7 tests
- Section 03: 8 tests
- Section 04: 9 tests
- Section 05: 10 tests
- Section 06: 6 tests
- Section 07: 8 tests
- Section 08: 20 tests
- **Grand total: 74 tests**

## Acceptance Criteria

- [ ] All 20 integration/edge-case tests pass
- [ ] All 54 per-section unit tests pass (from sections 01-07)
- [ ] No test uses real API calls or database connections
- [ ] All tests run in < 30 seconds total
- [ ] Edge cases cover NaN, None, zero, empty, and concurrent scenarios
- [ ] Schema tests verify idempotent creation
- [ ] `pytest` collection discovers all test files without import errors
