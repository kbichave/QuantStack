# Section 02: Futures Adapter

## Objective

Implement the first concrete `AssetClass` for futures trading via IBKR. This covers data ingestion, signal collectors (contango/backwardation, COT, roll yield), execution adapter, and futures-specific risk model.

**Depends on:** section-01-asset-class-base

## Files to Create

### `src/quantstack/asset_classes/futures.py`
Concrete `AssetClass` implementation:

- **Instruments:** ES (S&P 500), NQ (Nasdaq 100), CL (Crude Oil), GC (Gold), ZN (10Y Treasury)
- **`get_data_providers()`** — returns IBKR historical data provider + optional Databento tick provider
- **`get_risk_model()`** — returns `FuturesRiskModel` (SPAN margin, notional limits)
- **`get_signal_collectors()`** — returns contango/backwardation, COT positioning, roll yield collectors
- **`get_execution_adapter()`** — returns IBKR broker adapter
- **`get_trading_hours()`** — `TradingSchedule(open="18:00", close="17:00", tz="US/Eastern", days=[0-4], is_24h=False)` (23h/day Sun-Fri)
- **`get_position_limits()`** — per-contract notional limits, max 5% equity per instrument

### `src/quantstack/asset_classes/providers/ibkr_data.py`
`DataProvider` implementation for IBKR historical data:

- `fetch_ohlcv(symbol, start, end)` — use `ib_insync` or IBKR REST API
- `fetch_quote(symbol)` — real-time quote via IBKR streaming
- Handle contract specification (expiry, multiplier) for each futures symbol
- Graceful fallback: if IBKR unavailable, log warning and return empty DataFrame

### `src/quantstack/asset_classes/risk/futures_risk.py`
`RiskModel` implementation:

- `margin_requirement(symbol, qty, price)` — SPAN margin lookup (use conservative estimates: ES=$15k, NQ=$18k, CL=$8k, GC=$10k, ZN=$2k per contract)
- `validate_order(order)` — check margin available, notional limit, max contracts
- Account for contract multiplier in notional calculation (ES=$50, NQ=$20, CL=$1000, GC=$100, ZN=$1000)

### `src/quantstack/signal_engine/collectors/futures_signals.py`
Three signal sub-collectors:

1. **Contango/Backwardation** — compare front-month vs. next-month price. Signal: `contango_pct = (far - near) / near`. Negative = backwardation (bullish for commodities).
2. **COT Positioning** — Commitment of Traders data. Track commercial vs. speculative net positioning. Signal: `cot_commercial_net`, `cot_spec_net`, `cot_z_score` (20-week z-score of spec positioning).
3. **Roll Yield** — annualized cost/benefit of rolling contracts. Signal: `roll_yield_annualized`. Positive = contango drag, negative = backwardation benefit.

Each returns a `dict[str, float]` compatible with `SignalBrief`.

## Files to Modify

### `src/quantstack/signal_engine/engine.py`
Register futures signal collectors when `MULTI_ASSET_ENABLED` and futures asset class is active. Use the existing collector registration pattern.

### `src/quantstack/universe.py`
Add `FUTURES_INSTRUMENTS` dict mapping symbol to contract spec (multiplier, exchange, typical margin).

## Implementation Details

1. Futures symbols use continuous contract notation (e.g., `ES` maps to the front-month contract). The provider must handle roll logic.
2. COT data can be fetched weekly from CFTC (public, no API key needed). Cache locally since it updates only on Fridays.
3. All IBKR calls must use connection pooling and handle disconnections gracefully (IBKR drops connections after idle timeout).
4. Contract multipliers are critical for notional calculation — a 1-lot ES at 5000 = $250k notional. Errors here are catastrophic.

## Test Requirements

### `tests/unit/asset_classes/test_futures.py`
- `FuturesAssetClass` returns valid providers, risk model, collectors, adapter.
- Trading hours: verify `is_open()` for Sunday evening (open), Saturday (closed), Friday 17:01 ET (closed).
- Position limits: verify 5% equity cap.

### `tests/unit/asset_classes/test_futures_risk.py`
- Margin requirement returns correct values for each instrument.
- Notional calculation: 2 lots ES at 5000 = $500k notional.
- `validate_order` rejects when margin exceeds available.
- `validate_order` rejects when notional exceeds position limit.

### `tests/unit/signal_engine/test_futures_signals.py`
- Contango calculator: front=5000, back=5025 -> contango_pct=0.005.
- Backwardation: front=70, back=68 -> contango_pct=-0.0286 (negative).
- COT z-score: mock 20 weeks of data, verify z-score computation.
- Roll yield: verify annualization is correct for different DTE values.

## Acceptance Criteria

- [ ] `FuturesAssetClass` is importable and passes all abstract method checks
- [ ] IBKR data provider handles disconnection gracefully (no crash)
- [ ] Contract multipliers are correct for all 5 instruments
- [ ] Margin requirements are conservative (never underestimate)
- [ ] All 3 signal collectors produce valid `dict[str, float]` output
- [ ] Signal collectors registered in engine when feature flag enabled
- [ ] All tests pass under `uv run pytest tests/unit/asset_classes/ tests/unit/signal_engine/test_futures_signals.py`
