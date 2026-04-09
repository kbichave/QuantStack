# P12 Implementation Plan: Multi-Asset Expansion

## 1. Background

With P02 (execution) and P07 (data architecture) in place, P12 expands from equity + options to futures, crypto, forex, and fixed income. Each asset class needs its own data pipeline, risk model, execution path, and signal collectors.

## 2. Anti-Goals

- **Do NOT add all asset classes simultaneously** — start with futures (highest diversification benefit)
- **Do NOT build custom execution for each asset** — use IBKR unified API for futures/forex, Binance for crypto
- **Do NOT allow position sizes that exceed risk tolerance** — crypto gets 2-3% max per position (higher vol)
- **Do NOT trade obscure instruments** — major instruments only (ES, NQ, BTC, ETH, EUR/USD)

## 3. Asset Class Framework

### 3.1 Base Class

New `src/quantstack/asset_classes/base.py`:

```python
class AssetClass(ABC):
    def get_data_providers(self) -> list[DataProvider]: ...
    def get_risk_model(self) -> RiskModel: ...
    def get_signal_collectors(self) -> list[Collector]: ...
    def get_execution_adapter(self) -> BrokerAdapter: ...
    def get_trading_hours(self) -> TradingSchedule: ...
    def get_position_limits(self) -> PositionLimits: ...
```

### 3.2 Priority: Futures First

Futures via IBKR:
- Instruments: ES (S&P), NQ (Nasdaq), CL (crude), GC (gold), ZN (10Y)
- Data: IBKR historical + Databento ($100/mo) for tick data
- Signal collectors: contango/backwardation, COT positioning, roll yield
- Risk: SPAN margin (already exists), notional limits
- Schedule: 23h/day Mon-Fri

### 3.3 Crypto Second

Crypto via Binance API:
- Instruments: BTC, ETH, SOL
- Data: Binance REST (free), CoinGecko for metadata
- Signal collectors: on-chain metrics, funding rates, social sentiment
- Risk: 2-3% max per position, higher vol → tighter stops
- Schedule: 24/7

### 3.4 Cross-Asset Signals

New signal collectors:
- Equity-bond correlation regime
- Commodity-equity lead/lag
- FX carry trade indicator
- Crypto-equity correlation

## 4. Risk Gate Extensions

Extend `risk_gate.py` to handle multi-asset:
- Per-asset-class position limits
- Cross-asset correlation exposure check
- Margin requirements per asset class
- Total portfolio notional limit

## 5. Schema

- `asset_class_config`: (class_name, enabled, position_limit_pct, instruments JSONB)
- Extend `positions` table with `asset_class` column

## 6. Testing

- AssetClass interface: each implementation returns valid providers/risk/execution
- Futures: mock IBKR data → signal collectors produce valid signals
- Risk gate: multi-asset position limits enforced
- Cross-asset signals: correlation computation correctness
