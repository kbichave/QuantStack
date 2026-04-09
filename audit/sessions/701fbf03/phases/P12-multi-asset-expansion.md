# P12: Multi-Asset Expansion

**Objective:** Expand from equity + options to futures, forex, crypto, and fixed income. Each asset class needs its own data pipeline, risk model, execution path, and signal collectors.

**Scope:** New: asset_classes/, data/providers/, execution/brokers/

**Depends on:** P02 (execution), P07 (data architecture)

**Effort estimate:** 3-4 weeks

---

## What Changes

### 12.1 Asset Class Framework
```python
# New: src/quantstack/asset_classes/base.py
class AssetClass(ABC):
    """Base class for asset-class-specific behavior."""
    @abstractmethod
    def get_data_providers(self) -> list[DataProvider]: ...
    @abstractmethod
    def get_risk_model(self) -> RiskModel: ...
    @abstractmethod
    def get_signal_collectors(self) -> list[Collector]: ...
    @abstractmethod
    def get_execution_adapter(self) -> BrokerAdapter: ...
    @abstractmethod
    def get_trading_hours(self) -> TradingSchedule: ...
```

### 12.2 Futures (CME via IBKR)
- **Data:** CME delayed via IBKR, live via Databento ($100/mo)
- **Instruments:** ES (S&P), NQ (Nasdaq), CL (crude), GC (gold), ZN (10Y)
- **Signal collectors:** Contango/backwardation, COT positioning, roll yield
- **Risk model:** Span margin (already exists!), notional-based position limits
- **Execution:** IBKR TWS/Gateway API
- **Schedule:** Nearly 24/5 (23 hours/day Mon-Fri)

### 12.3 Forex (via IBKR or OANDA)
- **Data:** OANDA REST API (free demo), IBKR for live
- **Instruments:** EUR/USD, GBP/USD, USD/JPY, AUD/USD, carry trade pairs
- **Signal collectors:** Interest rate differentials, PPP deviation, carry trade
- **Risk model:** Notional-based, correlation with equity portfolio
- **Execution:** IBKR or OANDA API
- **Schedule:** 24/5

### 12.4 Crypto (via Binance or Coinbase)
- **Data:** Binance API (free), CoinGecko
- **Instruments:** BTC, ETH, SOL, top 10 by market cap
- **Signal collectors:** On-chain metrics (active addresses, exchange flows), funding rates, social sentiment
- **Risk model:** Higher volatility → tighter position limits (2-3% max per position)
- **Execution:** Binance API or IBKR crypto
- **Schedule:** 24/7

### 12.5 Fixed Income (via IBKR)
- **Data:** FRED (yields, spreads), IBKR (bond prices)
- **Instruments:** Treasury ETFs (TLT, IEF, SHY), corporate bond ETFs (LQD, HYG)
- **Signal collectors:** Yield curve slope, credit spread, term premium
- **Risk model:** Duration-based position limits, convexity
- **Execution:** ETFs via Alpaca/IBKR
- **Schedule:** Standard equity hours (ETFs)

### 12.6 Cross-Asset Signals
- Equity-bond correlation regime (positive = risk-on, negative = flight-to-quality)
- Commodity-equity lead/lag (oil → energy stocks)
- FX-equity hedging (USDJPY as carry trade proxy)
- Crypto-equity correlation (BTC as risk-on/off indicator)

## Priority Order
1. **Futures** (highest Sharpe diversification benefit, IBKR already integrated)
2. **Crypto** (24/7, different correlation structure, free data)
3. **Forex** (carry trade diversification, 24/5)
4. **Fixed Income** (ETF-based, easiest to add but lowest alpha opportunity)

## Acceptance Criteria

1. AssetClass framework supports pluggable data/risk/execution per class
2. At least 1 futures contract tradeable in paper mode
3. Cross-asset signals contribute to existing SignalBrief
4. Risk gate handles multi-asset position limits
