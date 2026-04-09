# P12 Research: Multi-Asset Expansion

## Codebase Research

### What Exists
- **Trading window enum**: `src/quantstack/` — TradingWindow system for instrument + time-horizon gating
- **Risk gate**: `src/quantstack/execution/risk_gate.py` — position limits, Greek checks, but equity-focused
- **Execution pipeline**: `src/quantstack/execution/trade_service.py` — Alpaca-based, equity + options
- **Signal engine**: full collector framework with synthesis — needs per-asset-class collectors
- **Universe**: `src/quantstack/universe.py` — currently equity-focused symbol universe
- **IBKR MCP**: referenced in codebase but has import errors — potential futures/forex execution path

### What's Needed (Gaps)
1. **Asset class abstraction**: No base class for asset types — need AssetClass ABC with providers, risk, signals, execution
2. **Futures adapter**: No futures data or execution — need IBKR integration + Databento for tick data
3. **Crypto adapter**: No crypto data or execution — need Binance API integration
4. **Cross-asset signals**: No correlation signals between asset classes
5. **Per-asset risk limits**: Risk gate is monolithic — needs per-asset-class limits
6. **Multi-asset schema**: positions table needs asset_class column

## Domain Research

### Futures Trading via IBKR
- IBKR TWS API provides unified access to futures, forex, options on futures
- Key instruments: ES (S&P 500), NQ (Nasdaq), CL (Crude Oil), GC (Gold), ZN (10Y)
- SPAN margin calculation — more capital-efficient than equity margin
- 23h/day trading Mon-Fri (CME Globex hours)
- Databento provides institutional-grade tick data at ~$100/mo

### Crypto Trading via Binance
- Binance REST API: free for spot data, maker fees 0.1%
- Key instruments: BTC, ETH, SOL
- 24/7 trading — requires separate scheduling logic
- Higher volatility → tighter position limits (2-3% max vs 5% for equities)
- Funding rate signals are unique to crypto (perpetual futures)

### Cross-Asset Diversification
- Primary benefit: uncorrelated return streams
- Equity-bond correlation regime changes (historically negative, recently positive)
- Commodity-equity lead/lag: oil prices predict energy sector moves
- Crypto-equity correlation has increased post-2020 but still provides diversification in some regimes
