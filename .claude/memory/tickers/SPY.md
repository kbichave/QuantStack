# SPY Research & Trading Memory

> Per-ticker memory file. Updated by research and trading loops.

## Evidence Map

_Last updated: 2026-03-26_

| Category | Key Findings | Tier | Direction |
|----------|-------------|------|-----------|
| Regime | trending_down, HIGH vol, ADX 46.83 (very strong), confidence 0.937 | tier_4 | bearish |
| HMM | HIGH_VOL_BULL (97.5%) -- disagrees with ADX trending_down | tier_4 | conflicting |
| Technicals | RSI bearish, at SMA200 resistance ($657.50), BB lower $645.72 | tier_1 | bearish |
| Volatility | GARCH cond vol 14.2% ann, EGARCH 15.4% ann, persistence 1.17 (explosive) | tier_2 | high vol expanding |
| GARCH forecast | 21d forecast: 13.8% ann (GARCH). RV(current) 13.0%. Vol regime: normal | tier_2 | vol slightly elevated |
| VRP | +11.45 pct pts (IV 25% vs RV 13.55%) as of 2026-03-20 -- extremely elevated | tier_2 | favors premium selling |
| Breadth | 13.3% (extreme washout) | tier_4 | bearish |
| Credit | Widening, dollar strengthening +2.29% | tier_4 | bearish |
| Capitulation | 0.163 (not ready -- selling pressure continues) | tier_3 | bearish (no bottom) |
| Options IV surface | Synthetic only -- flat 25% all strikes. No real skew/term structure | blocker | n/a |
| Price structure | At Low Volume Node -- expect fast moves. ATR_2x stop at $636.77 | tier_2 | volatile |

## Strategies

| ID | Name | Type | OOS Sharpe | Overfit Ratio | Status |
|----|------|------|-----------|--------------|--------|
| regime_momentum_v1 | Regime Momentum | position | 0.819 | 0.93 | validated |
| strat_dc4bd0297047 | spy_bear_put_debit_spread_v1 | options (bear put) | n/a | n/a | FAILED (IS Sharpe -1.44, OR-logic bug) |
| strat_64b8e34a30cf | spy_bear_put_spread_and_v2 | options (bear put) | n/a | n/a | FAILED (IS Sharpe -0.45, bidirectional contamination) |

## ML Models

| Date | Model | AUC (OOS) | IS-OOS Gap | Verdict |
|------|-------|-----------|------------|---------|
| 2026-03-21 | XGBoost | 0.7397 | 0.061 | champion |

**Breakthrough feature**: `macdext_histogram` is top SHAP in 4/5 ETF models.

## Options Research (2026-03-26, agent blitz_1_SPY_opt)

### GARCH/EGARCH Fit Results

| Model | omega | alpha | beta | Persistence | Cond Vol (ann) | AIC |
|-------|-------|-------|------|-------------|----------------|-----|
| GARCH(1,1) | 0.0544 | 0.0983 | 0.8298 | 0.928 | 14.2% | 1870.8 |
| EGARCH(1,1) | -0.004 | 0.2212 | 0.9507 | 1.172 | 15.4% | 1887.3 |

- GARCH persistence 0.928 (high but <1 = mean-reverting)
- EGARCH persistence 1.172 (>1 = explosive vol regime -- vol EXPANDS on down moves)
- EGARCH alpha=0.221 = strong ARCH effect (vol reacts sharply to shocks)
- GARCH 21d forecast: 13.8% annualized, converging toward long-run vol

### Bear Put Spread Structure Analysis (655/640, 30 DTE)

- Net debit: $6.17 per share ($617 per contract)
- Max profit: $8.83 per share ($883 per contract)
- Max loss: $6.17 per share ($617 per contract)
- Risk/reward: 1.43:1
- Break-even: $648.83 (1.2% below current price)
- Net delta: -0.123 (bearish)
- Probability of profit: 41.6% (structure analysis)

### Monte Carlo Simulation (5000 paths, 14d hold, 15% vol shock range)

| Metric | Value |
|--------|-------|
| Expected PnL | +$12.33 (near break-even) |
| Median PnL | -$42.15 (median outcome is small loss) |
| Probability of profit | 47.2% |
| VaR 95% | -$593.45 |
| P25 | -$394.45 |
| P75 | +$388.46 |
| P90 | +$687.95 |

### Tighter Config (21 DTE, 10d hold) Monte Carlo

| Metric | Value |
|--------|-------|
| Expected PnL | -$102.95 (negative) |
| Probability of profit | 40.2% |
| Conclusion | Shorter DTE accelerates theta, worse outcome |

### Key Finding

Put buying in downtrends has MARGINAL expected value. Time decay offsets directional gains in the median scenario. Edge exists only in tails (large moves >2%). This confirms QQQ finding from 2026-03-25.

## Lessons (SPY-specific)

1. **Regime-following works**: Hold long trending_up, short trending_down, flat ranging. OR-logic noise reduction prevents whipsaw.
2. **Post-2010 data only**: Pre-GFC data poisons walk-forward. Structural break at QE onset. Use 2010+ for validation.
3. **VRP elevated in bear**: VRP +11.45 pct pts favors premium selling, but EGARCH persistence >1.0 means vol expansion risk.
4. **EGARCH vol regime**: Persistence 1.172 = explosive. Volatility expanding, not mean-reverting.
5. **Bear put spreads are marginal**: Monte Carlo shows 47.2% POP, expected PnL near zero. Theta drag dominates in median scenario. Only profitable in large moves.
6. **Calendar spread requires real IV data**: Synthetic chain (flat 25% all strikes) makes calendar/term-structure strategies untestable.
7. **Rule engine cannot reliably backtest short-only options strategies**: OR-logic + bidirectional contamination generates LONG entries during bear markets. Use ML-gated entries or template backtests only.
8. **Options backtest tool broken**: run_backtest_options returns 0 trades (known FunctionTool bug). Cannot validate options strategies through standard pipeline.
