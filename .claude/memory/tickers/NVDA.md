# NVDA Research Log

## Last Updated: 2026-03-26
## Agents: blitz_21_NVDA_opt (options), blitz_21_NVDA_swing (swing)

---

## Current State
- **Price**: $178.68
- **Regime**: trending_down, normal vol, ADX 21.4 (weak -- near ranging threshold at 25)
- **Market Regime**: SPY trending_down (ADX 46.8), credit WIDENING, breadth 13.3%
- **Technicals**: Below SMA20 (180.59), SMA50 (184.09), at SMA200 (178.95). RSI 44.2, MACD bearish.
- **Volatility**: GARCH RV 37.3% annualized, synthetic IV 25% (no live IV data available)
- **VRP Assessment**: IV < RV -- vol appears underpriced (favors buying puts)
- **Capitulation Score**: 0.188 (not_ready -- selling may continue)
- **Accumulation Score**: 0.50 (neutral -- no institutional signal)
- **Credit Regime**: WIDENING -- macro deteriorating

## Strategies Registered -- OPTIONS DOMAIN

### 1. NVDA_bear_put_spread_30d (strat_ab84dfa3d9c9) -- DRAFT/FAILED
- **Structure**: Buy 180P, Sell 170P, 30 DTE
- **Max Risk**: $383/spread, Max Profit: $617/spread, R:R 1.61
- **Score**: 60/100 (neutral recommendation, delta misalignment flag)
- **IS Backtest (equity signals)**: 736 trades, Sharpe 0.21, Win 49.9%, MaxDD 24.2%, PF 1.11
- **Walk-Forward**: OOS Sharpe MEAN = -0.30 (FAILED), 2/5 folds positive, degradation 167%
- **VERDICT**: FAILED Gate 3 (OOS). The underlying equity signals have no edge OOS.
- **Root Cause**: The entry rules (close < SMA50 + MACD bearish + RSI < 50) are tier_1 only signals that fire constantly -- no real selectivity. OOS Sharpe is negative, meaning the "bearish signal" is noise.

### 2. NVDA_iron_condor_ranging (strat_a1a438ec52e2) -- DRAFT/NOT TESTED
- **Structure**: Sell 165P/160P, Sell 190C/195C, 30 DTE
- **Net Credit**: $1.22/spread, Max Loss: $3.78, Breakevens: 163.78-191.22
- **Score**: 70/100 (buy recommendation, no risk flags)
- **Backtest**: Failed to run -- entry rule used "between" condition with list value (not supported by backtest engine)
- **VERDICT**: Registration succeeded but untestable with current backtest engine. Needs code fix for list-value conditions.

## Strategies Registered -- SWING DOMAIN

### 3. NVDA_bear_bounce_short_v1 (strat_6e67b7773144) -- DRAFT/FAILED
- **Type**: Bearish mean-reversion (short overbought bounces)
- **Entry**: RSI > 55, close > SMA_20 but < SMA_50, ADX > 18, MACD histogram < 0
- **Exit**: Stop 2 ATR, target 2.5 ATR, time stop 10d, RSI < 35
- **IS Backtest**: 1,135 trades, Sharpe 0.37, Win 51.0%, MaxDD 7.71%, PF 1.17
- **Walk-Forward**: OOS Sharpe MEAN = -0.054 (FAILED), 3/5 folds positive, degradation 114%
- **VERDICT**: FAILED Gate 3 (OOS). Negative OOS Sharpe with massive IS/OOS degradation.
- **Root Cause**: Strategy tries to catch bounce failures, but the bounce-then-fail pattern is not statistically reliable on NVDA. Fold 4 (2011-2012, bull period) was catastrophic at -2.02 OOS Sharpe. The mean-reversion entry (RSI > 55 bounce) has no predictive power for subsequent declines.

### 4. NVDA_bear_momentum_breakdown_v1 (strat_64e15a87e947) -- DRAFT/NOT PROMOTED
- **Type**: Bear momentum (short on breakdown in downtrend)
- **Entry**: close < SMA_20, SMA_20 < SMA_50 (death cross), ADX > 20, RSI < 45, MACD histogram < 0
- **Exit**: Stop 2.5 ATR, target 3 ATR, time stop 10d, RSI < 25
- **IS Backtest**: 1,211 trades, Sharpe 0.45, Win 51.1%, MaxDD 7.86%, PF 1.21
- **Walk-Forward**: OOS Sharpe MEAN = 0.49, Overfit Ratio = 1.03, PBO = 0.40, 3/5 folds positive
  - Folds 1-3 (bear periods): OOS Sharpe 1.07, 1.55, 0.98 (strong)
  - Folds 4-5 (bull periods): OOS Sharpe -0.57, -0.58 (expected losses in wrong regime)
- **Cost test (2x slippage)**: Sharpe drops to 0.27 (cost-fragile)
- **DSR**: 0.32, not significant at 95% with 2 trials
- **VERDICT**: NOT PROMOTED. Shows real regime-conditional edge (1.03 overfit ratio is excellent, PBO 0.40 passes) but IS Sharpe marginal (0.45 < 0.50) and cost-fragile. The strategy genuinely works in bear markets but the edge is too thin after costs for NVDA specifically.
- **Key insight**: This strategy has the best OOS profile of all NVDA strategies tested (4 hypotheses total). The 1.03 overfit ratio means it is NOT overfit -- the issue is the edge is simply small. Single stocks like NVDA have too much idiosyncratic noise.

## Failed Hypotheses

### Hypothesis 1 (Options): NVDA Bear Put Debit Spread
- **Pre-registration**: Directional bearish, buy 180P/sell 170P, 30 DTE. Expected Sharpe ~0.5-0.8.
- **Economic mechanism**: Structural selling pressure in bear markets (margin calls, index rebalancing). Counterparty: retail dip-buyers.
- **Failed at**: Gate 3 (Walk-Forward OOS). OOS Sharpe = -0.30 across 5 folds.
- **Root cause**: Entry signals are entirely tier_1 (close < SMA50, MACD < signal, RSI < 50). These fire ~50% of trading days, creating an always-on signal with no selectivity. The "edge" is just sampling variance in-sample.
- **What this rules out**: Simple momentum/trend-following signals (close < MA + MACD) are insufficient for NVDA options entries. Need tier_2+ signals (VRP, GEX, IV skew) for genuine edge.

### Hypothesis 2 (Swing): NVDA Bear Bounce Short
- **Pre-registration**: Short overbought bounces (RSI > 55 at SMA_20) in downtrend. Expected Sharpe ~0.5-0.8.
- **Economic mechanism**: Institutional sellers use bounces to reduce positions; retail dip-buyers provide counterparty.
- **Failed at**: Gate 3 (Walk-Forward OOS). OOS Sharpe = -0.054.
- **Root cause**: The bounce-failure pattern is not statistically reliable. RSI IC at 5-day horizon is 0.0005 (zero). The overbought-in-downtrend signal has no predictive power on NVDA specifically.
- **What this rules out**: RSI-based mean-reversion shorting on NVDA has no edge. Do not revisit without tier_3+ signal overlay.

### Hypothesis 3 (Swing): NVDA Bear Momentum Breakdown
- **Pre-registration**: Short breakdowns below SMA_20 in death cross structure. Expected Sharpe ~0.4-0.7.
- **Economic mechanism**: Momentum effect (Jegadeesh & Titman 1993), negative information cascades, herding.
- **Outcome**: Marginal pass on OOS (Sharpe 0.49, PBO 0.40, overfit ratio 1.03) but failed IS gate (0.45 < 0.50) and cost sensitivity (0.27 at 2x slippage).
- **Root cause**: Real but thin edge. The death cross + ADX + MACD filter is the best signal combination tested, but NVDA's idiosyncratic noise is too high for the signal to overcome costs reliably.
- **What this preserves**: The strategy structure is sound -- consider testing on bearish ETFs (QQQ, SMH, SOXX) where idiosyncratic risk is diversified away. The 1.03 overfit ratio strongly suggests a real effect.

## IC Analysis (Swing Signals)
- RSI IC (5-day forward): 0.0005 (p=0.97) -- no predictive power
- ADX IC (5-day forward): -0.024 (p=0.05) -- weak negative, barely significant
- MACD histogram IC (10-day forward): 0.01 (p=0.41) -- not significant
- RSI alpha decay: half-life 0.1 bars (noise), peak IC at lag 8-9 = 0.019
- MACD histogram alpha decay: half-life 1.2 bars, negative at short lags, weak positive at 10-13d

## Lessons Learned
1. **Synthetic chain limitation**: No live IV data means IV rank, VRP, and skew signals are unavailable. This severely limits options research quality. All strategies designed with synthetic IV (flat 25%) are unreliable.
2. **Tier_1 signal trap**: Bear entry rules using only price-vs-MA, MACD, and RSI have no edge OOS on NVDA. Confirmed across both options and swing domains (4 hypotheses).
3. **Iron condor backtest gap**: The backtest engine does not support "between" conditions with list values. This is a tool gap.
4. **ADX threshold ambiguity**: NVDA at ADX 21.4 is in a gray zone between trending and ranging.
5. **Single stock vs ETF**: NVDA has too much idiosyncratic noise for tier_1-only strategies. The bear momentum strategy (strat_64e15a87e947) has an excellent overfit ratio (1.03) suggesting a real effect, but the edge is too thin after costs. This pattern likely works better on sector ETFs (SMH, SOXX) where idiosyncratic noise is averaged out.
6. **Regime-conditional validation matters**: Strategy 2 (bear momentum) showed clear regime bifurcation -- strong in bear folds (Sharpe 1.0-1.6), negative in bull folds (-0.6). This is correct behavior for a short strategy. Whole-period Sharpe understates the edge in the target regime.

## Next Steps (for future sessions)
1. Acquire live options chain data to get real IV rank, skew, and VRP signals
2. Design entry rules using tier_2+ signals (GEX support, IV skew extremes, institutional accumulation > 0.55)
3. Fix backtest engine to support "between" conditions for iron condor testing
4. **HIGH PRIORITY**: Port the bear momentum breakdown strategy (strat_64e15a87e947 rules) to SMH or SOXX ETFs. The 1.03 overfit ratio strongly suggests a real signal that is drowned by NVDA-specific noise.
5. Revisit NVDA when ADX either strengthens above 30 (clearer trend) or weakens below 20 (clearer range)
6. If tier_3 institutional data becomes available (real GEX, insider cluster data), overlay onto the bear momentum structure for a higher-conviction entry
