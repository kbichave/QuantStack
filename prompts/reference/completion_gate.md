# Completion Gate

Referenced by `research_loop.md` and domain prompts. Two gates: PAPER_READY (intermediate) and TRADING_READY (full).

---

## PAPER_READY (intermediate gate)

Output `<promise>PAPER_READY</promise>` when ALL of:

| Criterion | Threshold |
|-----------|-----------|
| Validated strategies | >= 3 (any domain, passed Gates 0-3) |
| Regime coverage | >= 2 regimes have strategies |
| Walk-forward passed | For each validated strategy |
| ML champion model | >= 1 per domain attempted |
| Paper portfolio Sharpe | > 0.5 |
| Stress test max DD | < 20% |

PAPER_READY enables paper trading + daily-planner while deep research continues toward TRADING_READY.

---

## TRADING_READY (full gate)

Output `<promise>TRADING_READY</promise>` when ALL cross-domain criteria are met.

## Cross-Domain Criteria (all required)

| Criterion | Threshold |
|-----------|-----------|
| Equity investment strategies (time_horizon="investment") | >= 1 per cached symbol |
| Equity swing/position strategies | >= 1 per cached symbol |
| Options strategies with full reporting | >= 1 per cached symbol |
| Thesis type coverage | At least 3 equity investment types + 3 swing types + 3 options types |
| Regime coverage | Every regime has strategies across all three domains |
| Walk-forward | Passed for each strategy, PBO < 0.40 |
| ML models | Champion + challenger per symbol, avg OOS AUC > 0.58 |
| Beat SPY | Investment strategies must beat SPY on alpha-adjusted basis |
| Cross-instrument portfolio | HRP/risk-parity allocation across equity + options |
| Stacking ensemble | Built where it improves OOS |
| RL agents | Trained, recording in shadow mode |
| Portfolio Sharpe | > 0.7 after costs |
| Deflated Sharpe Ratio | DSR > 0 for portfolio-level returns (accounting for all strategies tested) |
| Factor diversification | No single factor explains > 50% of portfolio variance |
| Strategy correlation | No strategy pair with correlation > 0.70 in the final portfolio |
| Stress test max DD | < 15% (swing/options), < 18% (investment) |
| Negative result ledger | >= 20 documented failed hypotheses (proves research breadth) |
| Research velocity | >= 30 total hypotheses tested across the research program |
| `trading_sheets_monday.md` | Complete with investment, swing, AND options plans per symbol |
| Experiment history | Meaningful entries in `ml_experiments` and `breakthrough_features` |

---

## Per-Domain Thresholds

### Equity Investment

| Criterion | Threshold |
|-----------|-----------|
| Strategies | >= 1 per cached symbol, time_horizon="investment" |
| Thesis type coverage | At least 3 of 5 types (value, quality_growth, dividend, sector_rotation, earnings_catalyst) |
| Regime coverage | Every macro regime has at least one investment strategy |
| Walk-forward | Passed for each strategy, PBO < 0.40, 6-month OOS windows |
| ML models | Champion + challenger per symbol, avg OOS AUC > 0.58 |
| Beat SPY | Alpha-adjusted basis (not raw return with high beta) |
| Stacking ensemble | Built where it improves OOS |
| Portfolio Sharpe | > 0.7 after costs |
| Stress test max DD | < 18% |

### Equity Swing/Position

| Criterion | Threshold |
|-----------|-----------|
| Strategies | >= 1 per cached symbol |
| Strategy type coverage | At least 3 of 5 types (momentum, mean_reversion, breakout, stat_arb, event_driven) |
| Regime coverage | Every regime (trending, ranging, counter-trend) has a strategy |
| Walk-forward | Passed for each strategy, PBO < 0.40 |
| ML models | Champion + challenger per symbol, avg OOS AUC > 0.58 |
| Stacking ensemble | Built where it improves OOS |
| RL agents | Trained, recording in shadow mode |
| Portfolio Sharpe | > 0.7 after costs |
| Stress test max DD | < 15% |

### Options

| Criterion | Threshold |
|-----------|-----------|
| Strategies | >= 1 per cached symbol with full reporting (BTO/STC, premium, win rate, hold time, max loss, DTE) |
| Strategy type coverage | At least 3 of 5 types (directional, debit_spread, vrp_harvest, earnings_vol, skew_term_structure) |
| Regime coverage | Every regime has at least one options strategy |
| Walk-forward | Passed for each strategy, PBO < 0.40 |
| ML models | Vol prediction model per symbol, beating GARCH baseline |
| Portfolio Sharpe | > 0.7 after costs |
| Stress test | Survives VIX +50%, max DD < 15% |

---

## 30-Iteration Gap Analysis

After 30 iterations, count how many cross-domain criteria above are met:

- **≥ 50% met** → output `<promise>TRADING_READY</promise>`
- **< 50% met** → output `<promise>RESEARCH_BLOCKED</promise>` with:

```
RESEARCH BLOCKED — gap analysis (iteration N):
- Met: [list criteria that pass]
- Missing: [list criteria that fail with specific current vs. target values]
- Bottleneck: [single root cause — e.g. "insufficient data for symbol X", "backtesting tool returning errors"]
- Recommended fix: [specific actionable step for the human]
```

This surfaces what's actually broken rather than declaring readiness falsely.

**After 45 iterations in any single domain, output `<promise>TRADING_READY</promise>` for that domain regardless.**

Don't count iterations toward a number. Build until the portfolio is ready.
