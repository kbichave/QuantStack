# XLE Research & Trading Memory

## Evidence Map

_Last updated: 2026-03-26 (blitz_21_XLE_swing session)_

| Category | Key Findings | Tier | Direction |
|----------|-------------|------|-----------|
| Regime | trending_UP + HIGH vol, ADX 45.24, confidence 0.905 | tier_4 | bullish |
| Trend | +DI 37.15 vs -DI 13.29, strong directional momentum | tier_1 | bullish |
| Volatility | ATR percentile 86.5%, elevated vol in trend | tier_2 | neutral (risk) |
| HMM | LOW_VOL_BEAR (disagrees with ADX - lagging sector rotation) | tier_4 | bearish (stale) |
| Sector Rotation | growth_to_value confirmed, XLE +11.1% in 20d vs SPY -5% | tier_3 | bullish |
| Credit | widening (but XLE benefits from inflation/commodity cycle) | tier_4 | mixed |
| Breadth | 13.3% extreme washout - XLE above ALL MAs | tier_4 | bullish for XLE |
| IC (ADX) | IC=0.0168 at 10d, p=0.17 (not significant), half-life 1 bar | tier_1 | weak |
| IC (RSI) | IC=-0.0297 at 10d, p=0.016 (significant, negative = contrarian) | tier_1 | bearish at high RSI |
| IC (MACD hist) | IC=0.009 at 7d (p=0.46), IC=0.003 at 5d (p=0.84). Not significant. | tier_2 | noise |
| Options IV | ATM IV 25%, GARCH RV 16.5-21.9%, VRP positive (~3-8.5 vol pts) | tier_2 | supports selling vol |
| EGARCH | Persistence 1.04 (unit root), annualized vol 21.86%, leverage effect present | tier_2 | high vol clustering |
| Options Chain | Synthetic chain, 30 DTE, $2.50 strike spacing, no live OI/volume | tier_1 | data limitation |

## Strategies

| ID | Name | IS Sharpe | OOS Sharpe | Status | Notes |
|----|------|-----------|------------|--------|-------|
| vol_compress_xle_v1 | Vol Compression Breakout | 0.208 | 0.219 | forward_testing | Ranging-to-trending, works for XLE. ONLY validated XLE strategy. |
| strat_3c2de1b0b93f | xle_momentum_reaccel_long_v1 | 0.364 | 0.684 (3-fold) / -0.02 (4-fold) | backtested | Best XLE rule result in 8 trials. PBO=0. But IC=0.009 (noise), cost-fragile. Fold-boundary sensitive. |
| strat_0a87e9d159e1 | xle_bull_call_spread_trending_v1 | 1.122 (opts) | N/A (WF gap) | draft | 75 trades, 48% WR, PF 2.45, avg +37.5% premium return |
| strat_d35364563947 | xle_put_credit_spread_trending_v1 | -0.231 (opts) | N/A | FAILED | Tool limitation: buys puts not sells spreads |
| strat_ffc48e271d2f | xle_trend_momentum_swing_v2 | 0.126 | -0.077 to +0.273 | FAILED | OR-logic, DSR=0 |
| strat_29d4ef7cfefe | xle_trend_momentum_swing_v3 | 0.141 | -1.21 | FAILED | AND-logic ADX>25, worse than OR |
| strat_dea35ae9abdc | xle_trend_momentum_swing_v4 | 0.072 | -- | FAILED | AND-logic ADX>30, tighter = worse |
| strat_94da1778fdfe | xle_trend_momentum_swing_v5 | -0.005 | -- | FAILED | SMA20 cross, zero edge |
| strat_0a3cf418d7cb | xle_trend_momentum_swing_v1 | error | -- | FAILED | between condition bug |
| strat_3f7f3ffb26d7 | xle_uptrend_pullback_long_v1 | 0.076 | -- | FAILED | Stoch<35 pullback too loose: 249 trades, Sharpe 0.08 |
| strat_e13c1925bc7b | xle_uptrend_pullback_long_v2 | 0.087 | -- | FAILED | Stoch<25 slightly better but still noise-level Sharpe |
| strat_e467ad1be688 | xle_momentum_reaccel_long_v2 | 0.157 | -- | FAILED | RSI<55 too tight, diluted v1 edge |

## ML Models

| Date | Model | AUC (OOS) | IS-OOS Gap | Verdict |
|------|-------|-----------|------------|---------|
| 2026-03-21 | LightGBM | 0.5748 | 0.063 | champion (marginal) |

## Options Research (2026-03-26)

### Volatility Analysis
- **EGARCH model**: omega=0.107, alpha=0.227, beta=0.813, persistence=1.04
- **Annualized RV**: 21.86% (EGARCH), 16.54% (GARCH forecast), 20.28% (1d forecast annualized)
- **ATM IV**: 25% (synthetic chain)
- **VRP**: IV - RV = +3.14 to +8.46 vol points (positive = sell vol favored)
- **Vol regime**: GARCH says "low" (24th percentile), but ATR percentile says 86.5% (HIGH)
- **Interpretation**: Disconnect between intraday vol (low on GARCH daily returns) and range (high ATR). Energy sector has fat tails from oil shocks.

### Bull Call Spread Analysis (strat_0a87e9d159e1)
- **Structure**: Buy ATM call / Sell 5-point OTM call, 30 DTE
- **IS Backtest**: Sharpe 1.122, 75 trades, 48% WR, PF 2.45
- **Avg premium return**: +37.5% per trade
- **Avg capital return**: +0.74% per position (2% sizing)
- **IV at entry avg**: 20.6% (below current 25% -- current setup is slightly expensive)
- **IV crush**: 38.7% of trades experienced IV crush
- **Best periods**: Energy bull runs (2001, 2017, 2021-2022, 2024)
- **Worst periods**: Sector drawdowns (2008, 2012, 2014, 2023)
- **Key insight**: Winners average +120% premium return, losers always hit -50% stop. Asymmetric payoff.

### Put Credit Spread Analysis (strat_d35364563947) -- FAILED
- **Structure**: Sell OTM put / Buy further OTM put, 30 DTE
- **IS Backtest**: Sharpe -0.231, 80 trades, 42.5% WR, PF 0.85
- **Root cause**: Put buying backtest (not put selling). The options backtest engine buys ATM puts on signal, not sells put spreads. Wrong tool for credit spread evaluation.
- **Lesson**: run_backtest_options only supports single-leg long positions (calls or puts), not multi-leg credit spreads. Credit spread backtesting requires a different approach.

### Walk-Forward Gap (CRITICAL)
- run_walkforward uses equity backtest engine, NOT options pricing. Returns 0 trades for options strategies.
- run_backtest (equity) also returns 0 trades -- entry rules with prerequisite + OR logic not generating signals in equity engine for this strategy.
- Options backtest engine (run_backtest_options) has its OWN signal generation that works (75 trades).
- **Conclusion**: Cannot validate options strategies through standard walk-forward pipeline. Need manual fold analysis or wait for options-specific WF tool.

## Lessons (XLE-specific)

1. **Needs macro features** -- AUC 0.5748 barely above threshold. Energy sector is macro-driven (oil, yields, VIX).
2. Vol compression breakout works -- institutional rotation drives sustained moves after compression.
3. Only ETF with RSI near overbought during broad market selloff -- rotational beneficiary.
4. **Momentum rule-based strategies FAIL on XLE** (2026-03-26): 5 variants tested (v1-v5), all failed. Best OOS Sharpe 0.273 but DSR=0 (not significant). Root cause: energy sector has fundamentally different momentum dynamics than tech ETFs. Oil price crashes (2015, 2020) create regime breaks that destroy trend-following alpha. The regime_momentum_v1 pattern that works on QQQ/XLK/SPY does NOT generalize to XLE.
5. **ADX IC on XLE is weak** (0.0168, p=0.17) with 1-bar half-life. Not a useful signal for swing holding periods.
6. **RSI IC on XLE is negative** (-0.03, p=0.016): high RSI predicts negative returns (contrarian). Supports mean-reversion/overbought-exit, not momentum entry.
7. **Walk-forward reveals regime-conditional alpha**: OOS Sharpe +0.65 during energy bulls (2009-2013) but -0.48 to -1.42 during sector crises. Alpha is not persistent enough to survive DSR test.
8. **OR-logic > AND-logic for XLE** (same as other ETFs): v2 OR-logic (Sharpe 0.13) > v3 AND-logic (0.14) > v4 tighter AND (0.07). Prerequisite rules help but don't solve the fundamental problem.
9. **Path forward for XLE momentum**: ML-gated entry (LightGBM needs macro features), or sector-relative momentum (XLE/SPY spread), or options overlay (calls during trending_up). Rule-based momentum is a dead end.
10. **Bull call spread IS promising** (2026-03-26): Sharpe 1.12, PF 2.45 on 75 trades. BUT: (a) cannot validate OOS via walk-forward (tool gap), (b) current IV (25%) is above historical avg entry IV (20.6%), (c) need live OI/volume data before execution. Status: draft pending OOS validation.
11. **Put credit spread backtest is misleading** (2026-03-26): run_backtest_options buys puts, not sells put spreads. Tool limitation. Cannot evaluate credit strategies with current tooling.
12. **Options WF tool gap**: Neither run_walkforward nor run_walkforward_mtf can validate options strategies. Need options-specific walk-forward or manual time-window backtests.
13. **MACD histogram re-acceleration is best rule-based XLE signal** (2026-03-26, blitz_21): strat_3c2de1b0b93f (MACD hist crosses_above 0, ADX>25 + RSI<65 prerequisites) achieved IS Sharpe 0.364, WF 3-fold OOS 0.684 (all 3 folds positive), PBO=0. BUT: MACD hist IC=0.009 (noise), cost-fragile (67% Sharpe drop at 2x slippage), WF 4-fold OOS -0.02 (fold-boundary sensitive). Strategy works in energy bull periods, fails during commodity crashes. Not promotable but represents the ceiling for XLE rule-based swing.
14. **Pullback-in-uptrend has no edge on XLE** (2026-03-26, blitz_21): Stoch pullback (stoch_k<25-35) within ADX>25-30 uptrend produces near-zero Sharpe (0.08-0.09) despite 200+ trades. Pullbacks in energy trends are not reliably absorbed by structural flows -- commodity-specific shocks (OPEC, demand destruction) create pullbacks that DON'T recover.
15. **11 equity swing strategies tested on XLE, 0 pass full validation pipeline.** The only validated XLE strategy remains vol_compress_xle_v1 (OOS 0.219, forward_testing). Energy sector may be fundamentally unsuitable for rule-based swing strategies. ML with macro features or options overlay remain the only unexplored paths.

## Research Log

| Date | Hypothesis | Variants | Result | Key Finding |
|------|-----------|----------|--------|-------------|
| 2026-03-20 | Vol compression breakout | 1 | PASS (OOS 0.219) | Works for ranging regime |
| 2026-03-26 | H_XLE_SWING_001: Trend momentum long | 5 (v1-v5) | ALL FAILED | Energy momentum via rule engine has no edge. DSR=0. |
| 2026-03-26 | H_XLE_OPT_001: Bull call spread trending | 1 | PARTIAL PASS (IS 1.12) | Strong IS but OOS validation blocked by tool gap |
| 2026-03-26 | H_XLE_OPT_002: Put credit spread trending | 1 | FAILED | Tool limitation |
| 2026-03-26 | H_XLE_SWING_002: Pullback in uptrend | 2 (v1-v2) | ALL FAILED | Stoch pullback = noise, Sharpe 0.08-0.09 |
| 2026-03-26 | H_XLE_SWING_003: MACD re-acceleration | 2 (v1-v2) | BEST RESULT but not validated | v1 IS 0.364, WF 3-fold OOS 0.684, PBO=0. But IC=noise, cost-fragile, fold-sensitive. |
