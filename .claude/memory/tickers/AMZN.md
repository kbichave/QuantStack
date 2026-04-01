# AMZN Ticker Research

> Last updated: 2026-03-26 (blitz_1_AMZN_swing session)
> Domain: equity swing

## Current Market State (2026-03-25)

- Price: $211.71
- Regime: ranging (ADX 12.38, very low)
- HMM: LOW_VOL_BULL (99.87% probability)
- RSI: 41.6 (approaching oversold, not extreme)
- StochRSI_K: 12.0 (deeply oversold)
- Stoch_K: 56.9 (mid-range)
- MACD histogram: 0.37 (positive, momentum turning up)
- ATR: 5.10 (~2.4% of price)
- BB lower: $204.14, SMA20: $211.09, SMA50: $218.65
- Capitulation: 0.20 (not ready)
- Credit: WIDENING (blocks bottom entries)
- Breadth: 13.3%
- VaR(95%): 3.10% daily, max DD -19.6% (90d)

## Strategies

### strat_1ae95475bfaa: AMZN_MeanRev_Ranging_v1 (RECOMMENDED)

**Status**: backtested (walk-forward validated)
**Type**: Mean reversion long in ranging regime
**Regime**: ranging (ADX < 20)
**Entry**: RSI < 35 (long, OR) + ADX < 20 (prerequisite) + Stoch_K < 30 (long, OR)
**Exit**: 10-day time stop + 8% trailing stop + RSI > 60
**Economic mechanism**: AMZN heavily indexed (SPY/QQQ/XLK) -- passive fund rebalancing + corporate buybacks ($10B+/yr) create oscillation around fair value. Counterparty: momentum traders/stop-loss sellers at extremes.

**IS (2010-2026)**: 331 trades, 55.9% WR, Sharpe 0.41, PF 1.30, MaxDD 6.85%, Calmar 2.28
**WF (3 folds, 2-year OOS windows)**:
| Fold | Test Period | OOS Sharpe | OOS Trades | OOS MaxDD |
|------|-------------|------------|------------|-----------|
| 1 | 2006-11 to 2008-04 | 0.13 | 27 | 3.58% |
| 2 | 2010-08 to 2011-12 | 0.24 | 26 | 2.32% |
| 3 | 2014-04 to 2015-09 | 1.12 | 28 | 1.21% |
- OOS Sharpe mean: 0.50, all 3/3 folds positive
- Overfit ratio: 0.46 (anti-overfit -- OOS > IS)

**WF (4 folds, 1-year OOS)**:
- OOS Sharpe mean: 0.63, 3/4 positive folds
- Overfit ratio: 0.37

**WF (4 folds, earliest config)**:
- OOS Sharpe mean: 0.94, 3/4 positive (fold 4 = 2014 trending year, expected failure)
- Overfit ratio: 0.036

**Verdict**: Consistently passes walk-forward across 3 different configurations. Anti-overfit in all cases. Trade count adequate (11-28/fold). MaxDD well controlled. Credit widening currently blocks live entry.

### strat_8ebbb0d86219: AMZN_OversoldBounce_v1

**Status**: backtested (walk-forward validated, conditional)
**Type**: Oversold bounce
**Entry**: RSI < 40 (long, OR) + MACD histogram > 0 (long, OR)
**Exit**: 7-day time stop + 6% trailing stop + RSI > 55

**IS (2010-2026)**: 221 trades, 54.3% WR, Sharpe 0.41, PF 1.36, MaxDD 4.93%, Calmar 2.99
**WF (4 folds)**: OOS Sharpe mean 0.96, overfit 0.089, 3/4 positive
**CONCERN**: OOS trades per fold = 4-8 (signal sparsity). High Sharpe may be sampling noise.

## Alpha Decay Analysis

- SMA_20 IC: 0.009-0.011, stable across 1-20 lags. No decay. Persistent but weak.
- Optimal holding period: 20 bars

## Template Backtest Baseline

- Mean reversion template (full history): 150 trades, Sharpe 0.29, WR 30.67%, PF 1.0
- Momentum template (full history): 180 trades, Sharpe 0.12, PF 0.95, negative return
- Momentum FAILS on AMZN -- confirms ranging regime thesis

## Key Findings

1. AMZN is a strong mean-reversion candidate with ADX at 12.38
2. Custom rule-based strategy with RSI+ADX+Stoch GENERATES TRADES (unlike QQQ custom rules)
3. Walk-forward shows consistent anti-overfit pattern across all configurations
4. Credit widening currently blocks live entry -- wait for credit_regime=stable
5. Momentum strategies lose money on AMZN in current regime

## Research Log

| Date | Hypothesis | Result | Root Cause |
|------|-----------|--------|------------|
| 2026-03-26 | H_AMZN_SWING_001: Mean-rev ranging | PASS (OOS 0.50-0.94) | Anti-overfit across configs |
| 2026-03-26 | H_AMZN_SWING_002: Oversold bounce | CONDITIONAL (sparse) | High OOS but 4-8 trades/fold |

## Next Steps

1. Wait for credit_regime=stable before creating entry alert
2. Test multi-symbol extension: run strat_1ae95475bfaa on MSFT, GOOG, AAPL
3. Consider ML model to enhance entry timing (SHAP on RSI x regime interaction)
4. Monitor ADX -- if rises above 20, switch to momentum strategies
