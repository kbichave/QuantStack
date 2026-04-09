# P08: Options Market-Making

**Objective:** Add delta-neutral and volatility-based options strategies: market-making, vol arb, dispersion trading, gamma scalping.

**Scope:** New: core/options/market_making.py, strategies/options_mm/

**Depends on:** P02 (real execution), P06 (options desk)

**Enables:** None (terminal capability)

**Effort estimate:** 2-3 weeks

---

## What Changes

### 8.1 Delta-Neutral Strategy Framework
- **Vol arb:** Buy underpriced vol (IV < realized), sell overpriced vol. Delta-hedge continuously.
- **Dispersion trading:** Sell index vol (SPY), buy component vol (individual stocks). Profit when correlation < implied.
- **Gamma scalping:** Buy ATM straddles for long gamma, delta-hedge, profit from realized vol > implied vol.
- **Iron condor harvesting:** Sell OTM spreads when IV rank > 50th percentile, defined risk.

### 8.2 Vol Surface Arbitrage
- Detect vol surface anomalies: skew too steep, term structure inverted
- Calendar spread when term structure inverted (short front month, buy back month)
- Butterfly when wings overpriced relative to ATM

### 8.3 Hedging Engine
```python
class HedgingEngine:
    def compute_hedge_order(self, portfolio_greeks: PortfolioGreeks) -> list[HedgeOrder]:
        """Generate orders to neutralize delta, reduce gamma/vega."""
        # Delta hedge: buy/sell underlying shares
        # Gamma hedge: buy/sell options to flatten gamma
        # Vega hedge: trade options at different strikes/expirations
    
    def rebalance_schedule(self) -> HedgeSchedule:
        """When to rebalance: time-based (every 30min) or threshold-based (delta > $500)."""
```

### 8.4 Market-Making Agent
New agent in trading graph: `options_market_maker`
- Monitors IV surface for mispricings
- Generates quotes (bid/ask) for selected strikes
- Auto-hedges delta exposure
- Risk limits: max portfolio vega, max single-strike concentration

### 8.5 P&L Attribution for Options
- Decompose daily P&L: delta + gamma + theta + vega + rho + unexplained
- Track which Greeks contribute most to P&L over time
- Use for strategy selection: "gamma scalping works in high-vol, theta harvest in low-vol"

## Key Packages
- `QuantLib-Python` — vol surface, pricing, Greeks
- `py_vollib` — Black-Scholes (already installed)
- `FinancePy` — structured products pricing

## Acceptance Criteria

1. Delta-neutral strategies generate trades with auto-hedging
2. Vol surface anomalies detected and tradeable
3. Portfolio Greeks neutralized within configurable thresholds
4. P&L attribution decomposes by Greek contribution
5. Market-making agent generates quotes with risk limits
