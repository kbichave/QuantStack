# Strategy Registry

> Last updated: 2026-03-15
> Read at start of: /workshop, /decode, /meta, /trade, /reflect
> Update after: /workshop, /decode, /reflect

## Active Strategies

| ID | Name | Type | Regime Fit | Status | Sharpe | Max DD | Source | Last Validated |
|----|------|------|-----------|--------|--------|--------|--------|----------------|
| strat_3651b3242b6e | quality_rsimr_15d | equity (XOM,IBM,MSFT) | ranging/trending_down | forward_testing | 0.70 avg (0.90/0.67/0.55) | 0.56% | workshop v4 | 2026-03-15 (backtest) |
| strat_6af32bf683f6 | mtf_swing_rsimr | equity (XOM,MSFT) | ranging/trending_down | forward_testing | XOM 3.70, MSFT 1.17 | 0.42% | workshop v5 | 2026-03-15 (backtest) |
| strat_e01e2da6d772 | mtf_medium_rsimr | equity (MSFT only) | ranging/trending_down | forward_testing | MSFT 0.96 | 0.77% | workshop v5 | 2026-03-15 (backtest) |

## Failed Strategies

| ID | Name | Status | Sharpe | Trades | Failure Reason | Date |
|----|------|--------|--------|--------|----------------|------|
| strat_d46f97e01b38 | spy_swing_5d | failed | -0.07* | 43 | True Sharpe -0.07 (old 0.178 was RSI-only artifact). 5d hold too short. | 2026-03-15 |
| strat_037afa2a8570 | spy_swing_5d_v2 | failed | 0.063 | 58 | Relaxed thresholds diluted edge. PF dropped to 1.14. | 2026-03-15 |
| strat_493cb5448197 | spy_swing_10d_regime_gated | failed | 0.36 | 27 | Regime gate disproven (87% of oversold in trending_down). Best variant: RSI-only 15d pure time stop. Still below Sharpe 1.0. | 2026-03-15 |
| strat_be0a6ddaf86b | multi_stock_rsimr_15d | failed | -0.01 (avg) | 117 | RSI<35+SMA200 is stock-heterogeneous: works on XOM (0.90), MSFT (0.55) but anti-works on BA (-1.0), BAC (-0.99). Portfolio avg Sharpe ≈ 0. Only 2/13 symbols > 0.5. | 2026-03-15 |

*Note: v1 Sharpe was corrected after fixing backtest engine to evaluate all indicator rules (ADX, CCI, Stoch K, etc.)*

## Retired Strategies

| ID | Name | Retired Date | Reason |
|----|------|-------------|--------|
(none yet)

## Notes

- Strategies must pass backtest + walk-forward before "forward_testing"
- Minimum 3 weeks forward testing before promotion to "live"
- Record retirement reasons — learning data for /reflect
