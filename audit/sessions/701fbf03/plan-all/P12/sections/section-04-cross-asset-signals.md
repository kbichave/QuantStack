# Section 04: Cross-Asset Signal Collectors

## Objective

Build inter-market signal collectors that detect regime shifts and lead/lag relationships across asset classes. These signals feed into the existing `SignalBrief` and are available to all trading strategies regardless of asset class.

**Depends on:** section-02-futures-adapter, section-03-crypto-adapter

## Files to Create

### `src/quantstack/signal_engine/collectors/equity_bond_correlation.py`
Equity-bond correlation regime detector:

- Compute rolling 60-day correlation between SPY returns and TLT returns
- **Positive correlation** (both fall together) = risk-off regime, flight to quality failing
- **Negative correlation** (normal) = bonds provide hedge
- **Regime shift** = correlation crosses zero from negative to positive (danger signal)
- Signals: `eq_bond_corr_60d: float`, `eq_bond_regime: str` ("normal_hedge" | "correlated_risk" | "transitioning"), `eq_bond_corr_z: float` (z-score vs 2-year history)

### `src/quantstack/signal_engine/collectors/commodity_equity_leadlag.py`
Commodity-equity lead/lag indicator:

- Track 5-day rolling return of GLD and CL (from futures or ETF proxy GLD/USO) vs. SPY
- Compute Granger-causality-lite: correlation of commodity returns(t) with equity returns(t+1..t+5)
- Signals: `gold_equity_lead_5d: float`, `oil_equity_lead_5d: float`, `commodity_leading: bool`
- Use existing `DataStore` for all data (no new API calls)

### `src/quantstack/signal_engine/collectors/fx_carry.py`
FX carry trade indicator:

- Compute carry attractiveness using interest rate differential proxies (2Y Treasury yield proxy via SHY vs. foreign rate proxies)
- Track DXY (via UUP ETF) momentum as carry unwind signal
- Signals: `fx_carry_attractiveness: float` (-1 to 1), `dxy_momentum_20d: float`, `carry_unwind_risk: bool`
- This is a simplified version — full FX trading is a later phase. Signal is for cross-asset regime detection.

### `src/quantstack/signal_engine/collectors/crypto_equity_correlation.py`
Crypto-equity correlation tracker:

- Rolling 30-day correlation between BTC daily returns and SPY daily returns
- When correlation is high (>0.7), crypto is "risk-on proxy" — no diversification benefit
- When correlation is low (<0.3), crypto provides true diversification
- Signals: `btc_spy_corr_30d: float`, `crypto_diversification_benefit: str` ("high" | "moderate" | "low"), `correlation_trend: float` (delta over last 20 days)

## Files to Modify

### `src/quantstack/signal_engine/collectors/cross_asset.py`
Extend the existing cross-asset collector to import and orchestrate the four new collectors. The existing collector already handles SPY/QQQ/IWM/TLT/GLD — the new collectors add deeper inter-market analysis.

Add a `collect_multi_asset_signals(store, timeframe)` function that:
1. Calls each new collector
2. Merges results into a single dict
3. Adds a composite `cross_asset_regime` field: "risk_on" | "risk_off" | "divergent" | "unknown"

### `src/quantstack/signal_engine/engine.py`
Register the composite cross-asset collector. Gate behind `MULTI_ASSET_ENABLED` flag.

### `src/quantstack/signal_engine/synthesis.py`
Update signal synthesis to weight cross-asset signals appropriately. Cross-asset signals should have lower weight than primary signal sources (suggested: 0.1-0.15 weight in the composite score).

### `src/quantstack/universe.py`
Ensure `CROSS_ASSET_ETFS` includes UUP (DXY proxy) and SHY (2Y Treasury) if not already present.

## Implementation Details

1. All four collectors use only local DataStore data — no new API calls. The data pipeline must have these symbols loaded (SPY, TLT, GLD, UUP, SHY, BTC equivalent).
2. For crypto correlation, use BTC-USD data from the crypto data provider if available, or BITO ETF as proxy from equity data.
3. Granger-causality computation in the commodity-equity collector should be lightweight — just lagged correlation, not a full statistical test. Keep it simple and fast.
4. The composite regime determination logic: if equity-bond correlation is "correlated_risk" AND commodity is leading negative AND carry unwind risk is True -> "risk_off". If all signals are benign -> "risk_on". Mixed -> "divergent".
5. Handle missing data gracefully — if any sub-collector fails, return partial results with the available signals. Never block the entire signal pipeline on one missing input.

## Test Requirements

### `tests/unit/signal_engine/test_equity_bond_correlation.py`
- Positive correlation scenario (both SPY and TLT down) -> regime = "correlated_risk"
- Negative correlation (normal) -> regime = "normal_hedge"
- Z-score: verify against known data
- Missing TLT data -> returns empty dict (graceful degradation)

### `tests/unit/signal_engine/test_commodity_equity_leadlag.py`
- Gold rallying 5 days before equity pullback -> `gold_equity_lead_5d` is negative
- No lead/lag -> values near zero
- Missing GLD data -> partial result (oil-only)

### `tests/unit/signal_engine/test_fx_carry.py`
- Strong USD momentum + high carry -> `fx_carry_attractiveness` > 0.5
- DXY reversing -> `carry_unwind_risk` = True
- Missing UUP -> returns empty dict

### `tests/unit/signal_engine/test_crypto_equity_correlation.py`
- High BTC-SPY correlation (>0.7) -> `crypto_diversification_benefit` = "low"
- Low correlation (<0.3) -> "high"
- Correlation trending up -> positive `correlation_trend`

### `tests/unit/signal_engine/test_cross_asset_composite.py`
- All risk-off signals -> composite = "risk_off"
- All benign -> composite = "risk_on"
- Mixed signals -> composite = "divergent"
- Partial collector failure -> composite still produced from available signals

## Acceptance Criteria

- [ ] All four new collectors produce valid `dict[str, float|str|bool]` output
- [ ] Composite regime determination works with full and partial data
- [ ] Graceful degradation: missing data in any collector does not crash the pipeline
- [ ] Cross-asset signals integrated into `SignalBrief` via synthesis
- [ ] Weight of cross-asset signals in composite score is 0.10-0.15
- [ ] `CROSS_ASSET_ETFS` updated with any missing symbols
- [ ] All tests pass under `uv run pytest tests/unit/signal_engine/test_equity_bond*.py tests/unit/signal_engine/test_commodity*.py tests/unit/signal_engine/test_fx*.py tests/unit/signal_engine/test_crypto_equity*.py tests/unit/signal_engine/test_cross_asset*.py`
