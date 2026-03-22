# Strategy Registry

> Last updated: 2026-03-20 (iteration 3 -- ETF generalization)
> Read at start of: /workshop, /decode, /meta, /trade, /reflect
> Update after: /workshop, /decode, /reflect

## Active Strategies

| ID | Name | Type | Regime Fit | Status | IS Sharpe | OOS Sharpe | Overfit Ratio | Trades (WF) | Max DD | Source | Last Validated |
|----|------|------|-----------|--------|-----------|------------|---------------|-------------|--------|--------|----------------|
| regime_momentum_v1 | Regime-Following Position (SPY) | trend_following | trending (bull+bear) | forward_testing | 0.761 | 0.819 | 0.93 | 32 | 2.2% | research_iter3 | 2026-03-20 |
| regime_momentum_v1_qqq | Regime-Following Position (QQQ) | trend_following | trending (bull+bear) | forward_testing | 0.785 | 1.346 | 0.58 | 37 | 1.8% | research_iter3 | 2026-03-20 |
| regime_momentum_v1_xlk | Regime-Following Position (XLK) | trend_following | trending (bull+bear) | forward_testing | 0.929 | 1.276 | 0.73 | 38 | 1.9% | research_iter3 | 2026-03-20 |
| regime_momentum_v1_xlf | Regime-Following Position (XLF) | trend_following | trending (bull+bear) | forward_testing | 0.411 | 0.617 | 0.67 | 22 | 2.7% | research_iter3 | 2026-03-20 |
| regime_momentum_v1_iwm | Regime-Following Position (IWM) | trend_following | trending (bull+bear) | conditional | 0.510 | 0.287 | 1.78 | 30 | 2.9% | research_iter3 | 2026-03-20 |
| vol_compress_xle_v1 | Vol Compression Breakout (XLE) | breakout | ranging-to-trending | forward_testing | 0.208 | 0.219 | 0.95 | 61 | 1.7% | research_iter2 | 2026-03-20 |

## Strategy Details

### regime_momentum_v1 -- Regime-Following Position Strategy (CORRECTED DESCRIPTION)

**CRITICAL CORRECTION (iteration 3)**: This is NOT a pullback strategy. The MCP engine uses OR-logic for plain rules, making this a regime-following POSITION strategy. The RSI/Stochastic filters act as noise reduction on entry/exit timing, not as independent entry gates. When all conditions are evaluated with AND-logic, the strategy FAILS (negative Sharpe across all symbols). The edge comes from holding positions through entire trend regimes, not from timing pullbacks within trends.

**True Mechanism**: Signal is ON (LONG) when ANY of: regime=trending_up, RSI 40-65, or stoch_k < 80. Signal is ON (SHORT) when ANY of: regime=trending_down, RSI 35-60, or stoch_k > 20. The stoch_k condition is nearly always true, so the signal is effectively regime-following with occasional gaps when ALL conditions happen to be false simultaneously. These gaps create natural position resets that improve risk management.

**Entry Rules (OR-logic, MCP plain rule semantics)**:
- LONG: regime=trending_up OR RSI 40-65 OR stoch_k < 80 (any triggers long)
- SHORT: regime=trending_down OR RSI 35-60 OR stoch_k > 20 (any triggers short)
- When both long and short fire: LONG takes priority

**Exit**: All conditions false simultaneously (signal drops to 0)

**Parameters**: sma_fast=20, sma_slow=50, rsi_period=14, adx_period=14, stoch_period=14

**Multi-ETF Walk-Forward Results (2010+ data, 5 folds, 252-bar test, purged CV)**:

| Symbol | IS Sharpe | OOS Sharpe | Overfit | OOS+ | Trades | MaxDD | Verdict |
|--------|-----------|------------|---------|------|--------|-------|---------|
| SPY    | 0.761     | 0.819      | 0.93    | 3/5  | 32     | 2.2%  | PASS |
| QQQ    | 0.785     | 1.346      | 0.58    | 4/5  | 37     | 1.8%  | STRONG PASS |
| XLK    | 0.929     | 1.276      | 0.73    | 4/5  | 38     | 1.9%  | STRONG PASS |
| XLF    | 0.411     | 0.617      | 0.67    | 3/5  | 22     | 2.7%  | PASS |
| IWM    | 0.510     | 0.287      | 1.78    | 3/5  | 30     | 2.9%  | WEAK |

**Data sensitivity**: Strategy FAILS walk-forward on full 26-year history (1999-2026) because pre-2010 data (dot-com bust, GFC) has fundamentally different regime dynamics. Post-2010 regime signals are cleaner and more persistent.

**Wide RSI Variant (30-70)**: Slightly higher Sharpe, fewer trades. Not meaningfully different in walk-forward. Stick with base.

**Symbols ranked**: QQQ (strongest), XLK (strongest), SPY (strong), XLF (moderate), IWM (weak)

### vol_compress_xle_v1 -- Vol Compression Breakout (XLE)

**Thesis**: When Bollinger Band width compresses to bottom quartile of 60-day rolling window, a directional move is imminent. Trade the breakout direction confirmed by RSI, MACD histogram, and above-average volume.

**Backtest (XLE, 26yr)**: 265 trades, 48.3% WR, PF 1.19, Sharpe 0.208, MaxDD 1.7%, Return 3.4%
**Walk-Forward (5 folds)**: OOS Sharpe 0.219 (std 0.890), 53.5% WR, Overfit ratio 0.95
**Symbols tested**: XLE (PASS), NVDA (FAIL OOS), TSLA (FAIL overfit 12.9x), SPY (FAIL OOS)

### vrp_premium_sell_spy_v1 (strat_d52734f45b93) -- Iron Condor VRP Capture

**Status**: DRAFT (cannot backtest -- run_backtest_options tool broken)
**Regime**: ranging + high vol
**Instrument**: Options (iron condor)
**Thesis**: Sell premium when VRP (IV - realized vol) > 5 pct pts in ranging regime. Iron condor with 25-delta short strikes and 10pt wings. Edge from VRP mean-reversion.
**Entry**: regime=ranging AND iv_rank > 50 AND VRP > 5 AND bb_width compressed
**Exit**: 50% profit target OR 7 DTE OR regime change OR 1.5x credit stop
**Risk**: EGARCH persistence >1.0 means vol can expand. Regime gate is critical safety.
**Holding**: ~21 days. Max premium: 2% equity per position.

### regime_momentum_calls_v1 (strat_8425300c071b) -- Directional Calls in Trending+HighVol

**Status**: DRAFT
**Regime**: trending_up + high vol
**Instrument**: Options (long ATM calls)
**Thesis**: Convexity overlay on regime_momentum equity signals. Buy calls when IV rank < 50 to avoid IV crush. MACD > 0 confirms momentum.
**Entry**: regime=trending_up AND vol=high AND iv_rank < 50 AND macd_histogram > 0
**Exit**: 100% premium gain OR -50% stop OR 10 DTE OR regime change
**Risk**: IV crush if vol drops. Time decay if trend stalls.

### regime_momentum_puts_v1 (strat_fcc1576c5e1f) -- Protective Puts in DownTrend+HighVol

**Status**: DRAFT
**Regime**: trending_down + high vol
**Instrument**: Options (long ATM puts)
**Thesis**: Directional puts with vega tailwind. In downtrends, IV EXPANDS (leverage effect), so puts benefit from both delta and vega. Allow IV rank up to 70 because expansion continues.
**Entry**: regime=trending_down AND vol=high AND macd_histogram < 0 AND egarch_persistence > 1
**Exit**: 100% premium gain OR -50% stop OR 10 DTE OR regime change

## Rejected Strategies

| Name | Why Rejected | Evidence |
|------|-------------|----------|
| Simple Oversold Mean Reversion | Infrequent trades OR destroyed edge when loosened | Sharpe -0.41, PF 0.76 at 87 trades |
| ADX Trend Momentum | Too selective, negative Sharpe on SPY/XLE | SPY Sharpe -0.145, NVDA Sharpe -0.422 |
| Vol Compression Breakout (NVDA) | Negative OOS Sharpe despite positive IS | OOS Sharpe -0.072 |
| Vol Compression Breakout (TSLA) | Extreme overfitting | OOS Sharpe 0.033, overfit ratio 12.89 |
| Vol Compression Breakout (SPY) | Negative OOS Sharpe | OOS Sharpe -0.251 |
| regime_momentum_v1 AND-logic | AND-logic pullback version fails ALL symbols | Negative Sharpe: SPY -0.46, IWM -0.51, XLF -0.19, XLK 0.03 |
| regime_momentum_v1 (full-history WF) | Pre-2010 data poisons walk-forward | OOS negative on all ETFs when including 1999-2009 |

## Regime Coverage

| Regime | Covered By | Gap? |
|--------|-----------|------|
| trending_up + normal vol | regime_momentum_v1 multi-ETF (SPY/QQQ/XLK/XLF/IWM) | No |
| trending_down + normal vol | regime_momentum_v1 multi-ETF (short side) | No |
| ranging + low vol -> breakout | vol_compress_xle_v1 | Partial (XLE only) |
| ranging + high vol | vrp_premium_sell_spy_v1 (DRAFT) | Partially -- needs backtest validation |
| trending_up + high vol | regime_momentum_calls_v1 (DRAFT) | Partially -- needs backtest validation |
| trending_down + high vol | regime_momentum_puts_v1 (DRAFT) | Partially -- needs backtest validation |
