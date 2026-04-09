# Section 03: Crypto Adapter

## Objective

Implement the concrete `AssetClass` for cryptocurrency trading via Binance API. Covers data ingestion, crypto-specific signal collectors (funding rates, on-chain metrics), execution adapter, and tighter risk model reflecting higher volatility.

**Depends on:** section-01-asset-class-base

## Files to Create

### `src/quantstack/asset_classes/crypto.py`
Concrete `AssetClass` implementation:

- **Instruments:** BTC, ETH, SOL (major only per anti-goals)
- **`get_data_providers()`** — returns Binance REST data provider + CoinGecko metadata provider
- **`get_risk_model()`** — returns `CryptoRiskModel` (2-3% max per position, tighter stops)
- **`get_signal_collectors()`** — returns funding rate, on-chain, social sentiment collectors
- **`get_execution_adapter()`** — returns Binance broker adapter
- **`get_trading_hours()`** — `TradingSchedule(is_24h=True, days=[0-6])` — 24/7
- **`get_position_limits()`** — 2% equity per position (conservative), 6% total crypto exposure

### `src/quantstack/asset_classes/providers/binance_data.py`
`DataProvider` implementation for Binance:

- `fetch_ohlcv(symbol, start, end)` — Binance REST `/api/v3/klines` endpoint. Rate limit: 1200 req/min.
- `fetch_quote(symbol)` — Binance `/api/v3/ticker/price`
- Handle symbol mapping: `BTC` -> `BTCUSDT`, `ETH` -> `ETHUSDT`, `SOL` -> `SOLUSDT`
- Implement rate limiter (token bucket, 1200/min) to avoid Binance IP bans

### `src/quantstack/asset_classes/providers/coingecko_metadata.py`
Metadata provider (not a `DataProvider` — supplementary):

- Fetch market cap, circulating supply, 24h volume from CoinGecko free API
- Cache aggressively (TTL=1h) since this data is slow-changing
- Used by risk model for liquidity assessment

### `src/quantstack/asset_classes/risk/crypto_risk.py`
`RiskModel` implementation:

- `margin_requirement(symbol, qty, price)` — 100% margin (no leverage for now; crypto vol is the leverage)
- `validate_order(order)` — enforce 2% max per position, 6% total crypto, minimum $100 order size
- Volatility-adjusted stop loss: use 20-day ATR * 2.5 as minimum stop distance
- Reject orders during extreme vol (24h move > 15%)

### `src/quantstack/signal_engine/collectors/crypto_signals.py`
Three signal sub-collectors:

1. **Funding Rate** — perpetual futures funding rate from Binance. Positive = longs pay shorts (crowded long). Signal: `funding_rate`, `funding_8h_annualized`, `funding_z_score`.
2. **On-Chain Metrics** — exchange inflows/outflows (proxy via CoinGecko volume delta). Signal: `exchange_flow_ratio`, `volume_momentum_7d`.
3. **Social Sentiment** — reuse existing social sentiment collector pattern from `src/quantstack/signal_engine/collectors/social_sentiment.py` but with crypto-specific sources. Signal: `crypto_social_score`, `mention_velocity`.

Each returns `dict[str, float]` compatible with `SignalBrief`.

## Files to Modify

### `src/quantstack/signal_engine/engine.py`
Register crypto signal collectors when `MULTI_ASSET_ENABLED` and crypto asset class is active.

### `src/quantstack/universe.py`
Add `CRYPTO_INSTRUMENTS` dict with symbol metadata (base/quote, min_order_size, tick_size).

## Implementation Details

1. Binance API requires API key for trading but not for market data. Data provider works without auth; execution adapter requires `BINANCE_API_KEY` and `BINANCE_SECRET_KEY` env vars.
2. Crypto is 24/7 — the scheduler and trading graph must handle this. No market-hours gating for crypto positions.
3. The 2% position limit is intentionally low. Crypto vol is 3-5x equity vol, so 2% crypto ~ 6-10% equity risk-equivalent.
4. Funding rates update every 8 hours on Binance. Fetch on 8h cadence, not per-minute.
5. All Binance timestamps are milliseconds since epoch — convert to UTC datetime consistently.

## Environment Variables

```
BINANCE_API_KEY         # Binance API key (required for execution, not for data)
BINANCE_SECRET_KEY      # Binance secret key
```

## Test Requirements

### `tests/unit/asset_classes/test_crypto.py`
- `CryptoAssetClass` returns valid providers, risk model, collectors, adapter.
- Trading hours: `is_open()` returns True for any datetime (24/7).
- Position limits: 2% per position, 6% total.

### `tests/unit/asset_classes/test_crypto_risk.py`
- Margin is 100% (no leverage).
- Order at 3% equity rejected.
- Order at 1.5% equity approved.
- Order rejected when 24h move > 15%.
- Total crypto exposure > 6% rejected.
- Minimum order size ($100) enforced.

### `tests/unit/asset_classes/test_binance_data.py`
- Mock Binance REST responses.
- Symbol mapping: `BTC` -> `BTCUSDT`.
- Rate limiter: 1201st request in 60s is delayed, not dropped.
- Malformed response handled gracefully (returns empty DataFrame).

### `tests/unit/signal_engine/test_crypto_signals.py`
- Funding rate: positive funding -> `funding_z_score` > 0 after sustained positive period.
- On-chain: volume spike -> `volume_momentum_7d` > 1.0.
- Social: all signals return float values in expected ranges.

## Acceptance Criteria

- [ ] `CryptoAssetClass` is importable and satisfies the ABC contract
- [ ] Binance data provider works without API key (market data only)
- [ ] Rate limiter prevents Binance IP bans
- [ ] Risk model enforces 2% per-position and 6% total crypto limits
- [ ] Extreme vol filter rejects orders during 15%+ 24h moves
- [ ] All 3 signal collectors produce valid output
- [ ] Symbol mapping handles all 3 instruments correctly
- [ ] All tests pass under `uv run pytest tests/unit/asset_classes/test_crypto*.py tests/unit/signal_engine/test_crypto_signals.py`
