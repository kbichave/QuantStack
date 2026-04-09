# P08 Implementation Plan: Options Market-Making

## 1. Background

With P02 (real execution), P06 (options desk: Greeks, hedging engine, IV surface, complex structures) in place, P08 adds delta-neutral and volatility-based strategies: vol arb, dispersion trading, gamma scalping, iron condor harvesting, and a market-making agent.

## 2. Anti-Goals

- **Do NOT build an actual market maker that quotes continuously** — this requires sub-second latency and dedicated infrastructure. Build the strategy logic; execution uses standard order flow.
- **Do NOT implement real-time vol surface streaming** — compute on-demand per cycle (P06 pattern).
- **Do NOT trade SPX index options** — stick with ETF/equity options available through Alpaca/IBKR.
- **Do NOT implement exotic pricing** — European + American (via financepy) is sufficient.

## 3. Delta-Neutral Strategy Framework

### 3.1 Vol Arb Strategy

New `src/quantstack/core/strategy/vol_arb_engine.py`:
- Signal: IV rank > 50th percentile AND (realized_vol_21d - IV) / IV > threshold
- Entry: Sell straddle/strangle when IV overpriced, buy when underpriced
- Hedge: Delta-hedge via HedgingEngine (P06)
- Exit: IV mean-reverts to realized vol, or time decay profit target hit

### 3.2 Dispersion Trading

New `src/quantstack/core/strategy/dispersion.py`:
- Signal: Implied correlation > realized correlation
- Entry: Sell index vol (SPY straddle), buy component vol (individual stock straddles)
- Hedge: Delta-neutral via underlying shares
- Risk: Correlation spike (left tail event)

### 3.3 Gamma Scalping

Extend P06 HedgingEngine with `GammaScalpingStrategy`:
- Entry: Buy ATM straddle for long gamma position
- Profit mechanism: Delta-hedge frequently, capture realized vol > implied vol
- Exit: Time decay overwhelms gamma profits (theta bleed)
- Schedule: Re-hedge every 30 min or on 0.5% underlying move

### 3.4 Iron Condor Harvesting

Use P06 StructureBuilder for condor construction:
- Signal: IV rank > 50th percentile, ranging regime
- Entry: Sell OTM put spread + OTM call spread (defined risk)
- Management: Roll tested side when short strike breached, close at 50% profit target
- Exit: Expiration or profit target

## 4. Hedging Engine Extensions

### 4.1 Gamma Hedging

Add `GammaHedgingStrategy` to P06's hedging framework:
- Target: Reduce portfolio gamma below threshold
- Method: Buy/sell options at different strikes to flatten gamma curve
- Constraint: Minimize cost while achieving target gamma

### 4.2 Vega Hedging

Add `VegaHedgingStrategy`:
- Target: Reduce portfolio vega below threshold
- Method: Trade options at different expiries (calendar-like adjustment)

## 5. Market-Making Agent

New agent `options_market_maker` in trading graph config:

### 5.1 Agent Definition

```yaml
# graphs/trading/config/agents.yaml
options_market_maker:
  role: "Options market-making and vol strategy execution"
  tools: [compute_greeks, get_iv_surface, score_trade_structure, simulate_trade_outcome]
  schedule: "every_cycle"
```

### 5.2 Agent Logic (Node)

In `graphs/trading/nodes.py`, new node:
1. Compute IV surface for universe symbols
2. Identify vol mispricings (IV vs realized vol divergence > threshold)
3. Select strategy based on regime (vol arb in trending, condor in ranging)
4. Generate trade proposals through risk gate
5. Monitor existing vol positions for management actions

### 5.3 Risk Limits

- Max portfolio vega: configurable (default $5,000)
- Max single-strike concentration: 20% of options allocation
- Max delta exposure before hedge: $500 (from P06 hedging engine)

## 6. P&L Attribution

Extend P06 Greek P&L attribution with strategy-level decomposition:
- Per-strategy: which strategies contribute delta vs gamma vs theta P&L
- Vol P&L: realized vol vs implied vol profit (core market-making metric)
- Use for strategy selection: "gamma scalping outperforms in high-vol regimes"

## 7. Schema

- `vol_strategy_signals`: (symbol, date, strategy_type, signal_value, iv_rank, realized_vol, iv)
- `dispersion_trades`: (index_symbol, components JSONB, implied_correlation, realized_correlation, entry_date)

## 8. Testing Strategy

### Unit Tests
- Vol arb signal: correct direction for IV > realized_vol
- Dispersion: correlation computation, entry/exit logic
- Gamma scalping: hedge frequency, P&L under various vol scenarios
- Condor: strike selection, management rules
- Market-making agent: strategy selection by regime

### Edge Cases
- IV surface too sparse for reliable vol arb → skip symbol
- Correlation data unavailable for dispersion → skip
- Gamma scalping theta bleed > gamma profit → auto-exit
