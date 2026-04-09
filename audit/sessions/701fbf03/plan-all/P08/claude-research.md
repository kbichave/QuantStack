# P08 Research: Options Market-Making

## Codebase Research

### What Exists
- **Options pricing**: `src/quantstack/core/options/engine.py` — Black-Scholes, Greeks, IV solver with vollib/financepy backends
- **IV surface**: `src/quantstack/core/options/iv_surface.py` — bilinear interpolation, ATM IV, skew metrics
- **Contract selector**: `src/quantstack/core/options/contract_selector.py` — rule-based selection by VolRegime/TrendRegime
- **Options tools**: `src/quantstack/tools/langchain/options_tools.py` — 2 implemented (fetch_options_chain, compute_greeks), 6 stubbed
- **Risk gate**: `src/quantstack/execution/risk_gate.py` — existing position limits, will need extension for vol strategies
- **Hedging engine**: planned in P06 — delta hedging infrastructure

### What's Needed (Gaps)
1. **Vol arb engine**: No vol arb strategy exists — need IV vs realized vol comparison, entry/exit logic
2. **Dispersion trading**: No cross-asset vol correlation tracking — need implied vs realized correlation
3. **Gamma scalping**: No automated rehedging — need periodic delta rebalance on gamma positions
4. **Iron condor harvesting**: No structured condor management — need roll logic, profit targets
5. **Market-making agent**: No options-specific agent in trading graph — need new node
6. **Vol P&L attribution**: Existing P&L doesn't decompose by Greek

## Domain Research

### Vol Arb Implementation Patterns
- Standard approach: compare IV rank to realized vol percentile, enter when divergence exceeds 1 std dev
- Hedging: delta-neutral via underlying shares, rebalance when delta drifts > threshold
- Risk: vol crush (sold straddle, vol drops = profit), vol spike (sold straddle, vol spikes = loss)

### Dispersion Trading
- Index vol typically overpriced vs component vol (correlation risk premium)
- Trade: sell index vol, buy component vol weighted by index composition
- Key risk: correlation spikes during crises — left tail event

### Market-Making vs Strategy
- True market-making requires sub-second execution — out of scope
- Strategy approach: identify mispricings, enter at limit orders, manage positions actively
- This is the approach P08 takes — options strategy execution, not market-making infrastructure
