# P12 Spec: Multi-Asset Expansion

## Deliverables

### D1: Asset Class Framework
- AssetClass ABC with uniform interface
- Implementations: EquityAssetClass, FuturesAssetClass, CryptoAssetClass
- Each returns appropriate providers, risk model, signal collectors, execution adapter, trading hours, position limits

### D2: Futures Trading (Priority 1)
- IBKR TWS API adapter for futures execution
- Instruments: ES, NQ, CL, GC, ZN
- Databento integration for tick data ($100/mo)
- Signal collectors: contango/backwardation, COT positioning, roll yield
- SPAN margin calculation

### D3: Crypto Trading (Priority 2)
- Binance REST API adapter for crypto execution
- Instruments: BTC, ETH, SOL
- Signal collectors: funding rates, on-chain metrics
- 2-3% max per position, 10% total crypto allocation
- 24/7 trading schedule

### D4: Cross-Asset Signals
- Equity-bond correlation regime
- Commodity-equity lead/lag
- FX carry trade indicator
- Crypto-equity correlation tracker

### D5: Risk Gate Multi-Asset Extension
- Per-asset-class position limits
- Cross-asset correlation exposure check
- Margin aggregation across asset classes
- Total portfolio notional limit

### D6: Schema Migrations
- `asset_class` column on positions table
- `asset_class_config` table
- Migration default: existing positions → 'equity'

## Dependencies
- P07 (Data Architecture): multi-provider pattern for new data sources
- P02 (Execution): execution pipeline extension for new brokers
