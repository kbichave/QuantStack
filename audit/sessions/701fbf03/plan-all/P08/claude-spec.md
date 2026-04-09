# P08 Spec: Options Market-Making

## Deliverables

### D1: Vol Arb Engine
- Compare IV rank vs realized vol percentile
- Entry when divergence > 1 std dev, sell straddle/strangle
- Delta-hedge via P06 HedgingEngine
- Exit on IV mean-reversion or profit target

### D2: Dispersion Trading
- Implied correlation vs realized correlation computation
- Sell index vol, buy component vol
- Correlation spike monitor with auto-exit at +0.15 threshold
- Position size capped at 2% per trade

### D3: Gamma Scalping
- Long gamma via ATM straddle
- Automated delta rehedge every 30min or 0.5% underlying move
- Theta bleed tracking with auto-exit when cumulative theta > gamma profit
- Uses P06 HedgingEngine for rebalancing

### D4: Iron Condor Harvesting
- Entry: IV rank > 50th percentile, ranging regime
- Management: roll tested side, 50% profit target, 200% loss limit
- Expiration risk: close within 2 DTE if near short strike
- Uses P06 StructureBuilder for construction

### D5: Market-Making Agent
- New trading graph node: options_market_maker
- Tools: compute_greeks, get_iv_surface, score_trade_structure, simulate_trade_outcome
- Strategy selection by regime (vol arb in trending, condor in ranging)
- All proposals through risk gate

## Dependencies
- P06 (Options Desk): hedging engine, IV surface, structure builder, Greeks
- P02 (Execution): order pipeline for options execution
