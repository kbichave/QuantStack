# Section 08: Unit Tests

## Objective

Comprehensive unit test suite covering all P12 components. This section consolidates test requirements from sections 01-05 into a single implementation plan, adds edge case coverage, and ensures regression safety for existing functionality.

**Depends on:** section-02-futures-adapter, section-03-crypto-adapter, section-04-cross-asset-signals, section-05-risk-gate-multi-asset

## Files to Create

### `tests/unit/asset_classes/__init__.py`
Empty init for test package.

### `tests/unit/asset_classes/test_base.py`
Base framework tests:
- `AssetClass` cannot be instantiated directly (TypeError)
- Minimal concrete subclass satisfies all 6 abstract methods
- `TradingSchedule.is_open()`: in-hours, out-of-hours, weekend, DST spring-forward, DST fall-back
- `TradingSchedule.is_open()` with 24/7 schedule always returns True
- `PositionLimits` is frozen (AttributeError on assignment)
- `AssetClassType` enum has all 6 values

### `tests/unit/asset_classes/test_registry.py`
Registry tests:
- Register then retrieve -> same instance
- Get unregistered type -> KeyError with descriptive message
- `enabled()` returns only registered types (not all enum values)
- Double-register same type -> overwrites (not error)
- Thread safety: 100 concurrent registrations from threads -> no corruption

### `tests/unit/asset_classes/test_futures.py`
Futures adapter tests:
- All 6 ABC methods return correct types
- Trading hours: Sunday 18:00 ET (open), Saturday 12:00 (closed), Friday 17:01 ET (closed), Friday 16:59 ET (open)
- Position limits: 5% per position, correct max_positions
- Instrument list matches spec (ES, NQ, CL, GC, ZN)

### `tests/unit/asset_classes/test_futures_risk.py`
Futures risk model:
- Margin: ES=~$15k, NQ=~$18k, CL=~$8k, GC=~$10k, ZN=~$2k per contract
- Notional: 2 ES at 5000 = $500,000 (50 multiplier * 5000 * 2)
- Notional: 1 CL at 70 = $70,000 (1000 multiplier * 70)
- Validate: margin > available -> rejected
- Validate: notional > limit -> rejected
- Validate: within limits -> approved

### `tests/unit/asset_classes/test_crypto.py`
Crypto adapter tests:
- All 6 ABC methods return correct types
- Trading hours: any day/time -> is_open() = True
- Position limits: 2% per position, 6% total
- Instrument list: BTC, ETH, SOL

### `tests/unit/asset_classes/test_crypto_risk.py`
Crypto risk model:
- Margin is 100% of notional
- 3% equity order -> rejected
- 1.5% equity order -> approved
- Total crypto at 5.5% + new 1% order -> rejected (exceeds 6%)
- 24h move > 15% -> rejected regardless of size
- Order < $100 -> rejected (minimum size)

### `tests/unit/asset_classes/test_binance_data.py`
Binance data provider:
- Symbol mapping: BTC -> BTCUSDT, ETH -> ETHUSDT, SOL -> SOLUSDT
- Mock valid klines response -> correct DataFrame (columns: open, high, low, close, volume)
- Mock malformed response -> empty DataFrame (no crash)
- Mock 429 rate limit response -> retry with backoff
- Rate limiter: burst of 1201 requests -> last request delayed

### `tests/unit/signal_engine/test_futures_signals.py`
Futures signal collectors:
- Contango: front=5000, back=5025 -> pct=0.005
- Backwardation: front=70, back=68 -> pct=-0.0286 (approx)
- COT: 20 weeks mock data -> valid z-score in [-3, 3] range
- COT: insufficient data (< 10 weeks) -> returns empty dict
- Roll yield: 30 DTE, 0.5% contango -> annualized ~6.1%
- All collectors return dict[str, float] with expected keys

### `tests/unit/signal_engine/test_crypto_signals.py`
Crypto signal collectors:
- Funding positive for 3 periods -> positive z-score
- Funding near zero -> z-score near zero
- Volume spike (2x avg) -> momentum > 1.0
- Social score returns float in [0, 1] range
- Missing data -> empty dict (not crash)

### `tests/unit/signal_engine/test_equity_bond_correlation.py`
Equity-bond correlation:
- Synthetic correlated returns -> regime = "correlated_risk"
- Synthetic anti-correlated returns -> regime = "normal_hedge"
- Z-score computation matches manual calculation
- < 60 days of data -> returns empty dict

### `tests/unit/signal_engine/test_commodity_equity_leadlag.py`
Commodity-equity lead/lag:
- Synthetic gold-leads-equity pattern -> gold_equity_lead_5d significant
- Random data -> lead values near zero
- Missing GLD, USO available -> partial result with oil only
- Missing all commodity data -> empty dict

### `tests/unit/signal_engine/test_fx_carry.py`
FX carry:
- Strong DXY uptrend -> positive momentum
- DXY reversal -> carry_unwind_risk = True
- Missing UUP data -> empty dict

### `tests/unit/signal_engine/test_crypto_equity_correlation.py`
Crypto-equity correlation:
- Corr > 0.7 -> diversification_benefit = "low"
- Corr < 0.3 -> diversification_benefit = "high"
- 0.3 < corr < 0.7 -> diversification_benefit = "moderate"
- Trending correlation -> correct sign on correlation_trend

### `tests/unit/signal_engine/test_cross_asset_composite.py`
Composite cross-asset regime:
- All risk-off indicators -> "risk_off"
- All benign -> "risk_on"
- Mixed -> "divergent"
- All collectors fail -> "unknown"
- 2 of 4 collectors fail -> still produces regime from available data

### `tests/unit/execution/test_risk_gate_multi_asset.py`
Multi-asset risk gate:
- No asset_class param -> equity behavior unchanged (regression)
- Futures at 6% -> rejected, at 4% -> approved
- Crypto at 3% -> rejected, at 1.5% -> approved
- Total notional > 3x equity -> rejected
- Margin > 90% available -> rejected
- Existing equity-only tests still pass (backward compat)

### `tests/unit/execution/test_correlation_monitor.py`
Correlation monitor:
- Two uncorrelated assets -> acceptable
- Highly correlated portfolio + correlated new position -> rejected
- Cache hit within TTL -> returns cached (verify no recomputation)
- Cache miss after TTL -> recomputes
- Missing price data for one instrument -> excluded, not blocked

### `tests/unit/execution/test_portfolio_state_multi_asset.py`
Portfolio state:
- Mixed portfolio -> correct grouping by asset class
- Futures notional uses contract multiplier
- Empty portfolio -> all dicts empty (no KeyError)
- Unknown asset class in DB -> logged warning, treated as equity

### `tests/unit/execution/test_risk_state_multi_asset.py`
Per-asset-class halting:
- Crypto loss > 1% -> crypto halted, equity continues
- Futures loss > 1.5% -> futures halted, others continue
- Multiple classes halted -> each independent
- Next day -> all classes reset (unhalted)

## Implementation Details

1. Use `pytest` fixtures extensively. Create shared fixtures in `tests/unit/asset_classes/conftest.py` for mock asset classes, sample data, and DB setup.
2. All signal collector tests should use synthetic data (NumPy-generated) with known statistical properties, not mock API responses. This makes tests deterministic and fast.
3. Risk gate tests must import and run the EXISTING equity test suite first to verify backward compatibility. Use `pytest.mark.parametrize` for multi-asset-class variants.
4. Mock all external dependencies (IBKR, Binance, DB). No network calls in unit tests.
5. Use `freezegun` or `time_machine` for time-dependent tests (trading hours, DST).

## Acceptance Criteria

- [ ] All test files created with comprehensive coverage
- [ ] Shared conftest with reusable fixtures
- [ ] Zero network calls in any unit test
- [ ] Time-dependent tests use time mocking (no flaky clock-dependent tests)
- [ ] Signal collector tests use synthetic data with known properties
- [ ] All existing tests still pass (no regressions)
- [ ] Full suite runs in < 30 seconds
- [ ] `uv run pytest tests/unit/asset_classes/ tests/unit/signal_engine/test_futures_signals.py tests/unit/signal_engine/test_crypto_signals.py tests/unit/signal_engine/test_equity_bond*.py tests/unit/signal_engine/test_commodity*.py tests/unit/signal_engine/test_fx*.py tests/unit/signal_engine/test_crypto_equity*.py tests/unit/signal_engine/test_cross_asset*.py tests/unit/execution/test_risk_gate_multi_asset.py tests/unit/execution/test_correlation_monitor.py tests/unit/execution/test_portfolio_state_multi_asset.py tests/unit/execution/test_risk_state_multi_asset.py` all green
