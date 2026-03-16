# Strategy Registry

> Last updated: 2026-03-15
> Read at start of: /workshop, /decode, /meta, /trade, /reflect
> Update after: /workshop, /decode, /reflect

## Active Strategies

| ID | Name | Type | Regime Fit | Status | Sharpe | Max DD | Source | Last Validated |
|----|------|------|-----------|--------|--------|--------|--------|----------------|
(none yet)

## Failed Strategies

| ID | Name | Status | Sharpe | Trades | Failure Reason | Date |
|----|------|--------|--------|--------|----------------|------|
| strat_d46f97e01b38 | spy_swing_5d | failed | 0.178 | 48 | Sharpe <1.0, trades <60. Oversold extremes at SMA200 too rare for 5-day swing. | 2026-03-15 |
| strat_037afa2a8570 | spy_swing_5d_v2 | failed | 0.063 | 58 | Relaxed thresholds diluted edge. PF dropped to 1.14. | 2026-03-15 |

## Retired Strategies

| ID | Name | Retired Date | Reason |
|----|------|-------------|--------|
(none yet)

## Notes

- Strategies must pass backtest + walk-forward before "forward_testing"
- Minimum 3 weeks forward testing before promotion to "live"
- Record retirement reasons — learning data for /reflect
