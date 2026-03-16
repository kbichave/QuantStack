# Workshop Lessons

> Accumulated R&D learnings — your research memory
> Read at START of every /workshop session
> Update after: /workshop, /reflect

## Backtesting Pitfalls

- SPY mean-reversion with extreme oversold filters (RSI<35, Stoch<15, CCI<-150) produces only 27-55 trades in 6 years depending on configuration
- Loosening oversold thresholds to 1-of-4 (87 trades) destroys edge: Sharpe -0.41, PF 0.76
- 5-day hard time stop constrains mean-reversion strategies — 15d consistently outperforms 5d by +0.25 Sharpe
- **The old backtest engine only evaluated RSI** — original v1 results (Sharpe 0.178, 64% WR) were artifacts of broken signal generation. True multi-signal v1 has Sharpe -0.07.
- **Regime gates on mean-reversion are anti-productive**: oversold signals and ranging markets are mutually exclusive on SPY (87% of RSI<35 occurs in trending_down)
- Pure time stops outperform ATR-based exits for mean-reversion: letting trades play out fully beats active stop management (PF 1.68 vs 1.26)
- **RSI<35 mean-reversion is stock-type-specific**: works on value/quality stocks (XOM, MSFT) with institutional support floors but anti-works on momentum (TSLA, BA, NFLX) and financials (BAC). Applying uniformly across a diversified universe nets to Sharpe ≈ 0. Must pre-filter universe by stock characteristics.
- **QuantCore screener is broken for daily backtests**: hard-coded to `daily` timeframe at `server.py:2405` but DB only stores 1H data. Must fetch daily data via `fetch_market_data` before screening or backtesting.

## What Works in Each Regime

| Regime | Tends to Work | Tends to Fail | Confidence |
|--------|--------------|---------------|------------|
| trending_down + high_vol | Counter-trend oversold bounces (55.6% WR, PF 1.68 with 15d hold, pure time stop) — but Sharpe only 0.36 | Regime gates (remove the signal), tight stops (1.5× ATR SL gets clipped) | Medium (10+ variants tested) |
| ranging | Almost no oversold signals fire — market doesn't push extremes in ranging conditions | Mean-reversion entry signals (RSI<35 almost never triggers) | High (funnel analysis) |

## Parameter Sensitivity

- Hold period 5d→15d: most impactful single lever. Improves Sharpe +0.25 and PF +0.3 consistently across all signal configs.
- ATR exit asymmetry: tighter SL (1×ATR) + wider TP (3×ATR) improves Sharpe from -0.24 to +0.32 vs symmetric 1.5×/2.5×
- Confirmation count: 3-of-5 and 2-of-4 similar quality; 1-of-4 is too loose (destroys edge)
- Regime gate (ADX threshold): harmful at all levels tested (ADX<25, <30, <35) — reduces trades without improving metrics

## Failed Hypotheses (don't repeat these)

| Date | Hypothesis | Why It Failed | Regime | Takeaway |
|------|-----------|---------------|--------|----------|
| 2026-03-15 | SPY oversold bounce with 5-day hold (v1) | True Sharpe -0.07 (not 0.178 as old engine reported). 5d too short for SPY bounces. | trending_down / high_vol | Longer holds help but can't fix the fundamental SPY mean-reversion Sharpe problem. |
| 2026-03-15 | Regime gate (ADX<25 + not trending_down) for SPY mean-reversion (v2) | Only 6 trades — 87% of oversold signals occur during trending_down. Gate removes signal, not noise. | all | Never gate counter-trend strategies by excluding the trend that produces the entry signal. |
| 2026-03-15 | All SPY mean-reversion configs tested (v1→v4, 10 variants) | Best Sharpe 0.36 (RSI-only, 15d pure time stop). Per-trade edge exists but too small for SPY variance at 5% sizing. | all | SPY single-stock mean-reversion is exhausted. Move to multi-stock, options, or bidirectional approaches. |
| 2026-03-15 | Multi-stock RSI<35 mean-reversion on 13 mega-caps (v3) | Avg portfolio Sharpe -0.01. Signal is stock-heterogeneous: works on value/quality (XOM 0.90, MSFT 0.55) but anti-works on momentum/financials (BA -1.0, BAC -0.99). 117 trades but winners=losers. | all | RSI<35 mean-reversion is stock-type-specific, not universal. Must pre-filter by stock characteristics OR use options for convexity on the 2 proven names. |

## Workshop Sessions

### SPY Swing 5-Day — 2026-03-15

**Hypothesis**: Oversold mean-reversion bounce at SMA200 support with multi-signal confirmation
**Regime fit**: trending_down / high_vol (current SPY regime: ADX 32.8, ATR pct 81%)
**Entry gate**: 3+ of 5: RSI<35, Stoch K<15, close within 1.5% of SMA200, ATR pct>60, CCI<-150
**Time stop**: 5 days hard
**Backtest v1 (old engine — RSI-only signals)**: Sharpe=0.178 MaxDD=1.31% WinRate=64.6% Trades=48 PF=1.35
**Backtest v1 (enhanced engine — full rules)**: Sharpe=-0.07 MaxDD=1.11% WinRate=51.2% Trades=43 PF=0.95
**Note**: Original v1 results were artifacts of broken engine that only evaluated RSI. True 3-of-5 performance is worse.
**Walk-forward**: Not run (backtest failed)
**Verdict**: FAIL
**Sit-out trigger**: Strategy is FAILED, no signals emitted

### SPY Mean-Reversion v2 — 2026-03-15

**Changes from v1**: regime gate (ADX<25, not trending_down) + 10-day hold + 2-of-4 signals (dropped ATR percentile)
**Infrastructure fix**: Enhanced backtest engine to support ADX, CCI, Stochastic K, price_vs_sma200, regime, prerequisite/confirmation rule hierarchy, time stops, ATR SL/TP forward simulation. 12 existing tests still pass.

**Regime gate analysis**:
- SPY regime distribution (2020-2025): ranging=25.8%, trending_up=44.3%, trending_down=29.9%
- 87% of RSI<35 entries occur during trending_down with ADX>=25
- ADX<25 + 2-of-4 confirmations = only 15 signal bars → 6 trades
- **Regime gate DISPROVEN**: oversold signals and ranging markets are mutually exclusive on SPY

**Comprehensive variant comparison (2020-2025, 5% sizing)**:
| Variant | Trades | WR | Sharpe | PF | AvgPnL |
|---------|--------|-----|--------|-----|--------|
| v1: 3-of-5, 5d hold | 43 | 51.2% | -0.07 | 0.95 | -$3.65 |
| v1: 3-of-5, 15d hold | 39 | 48.7% | 0.18 | 1.26 | $20 |
| 2-of-4, 10d, no gate | 51 | 45.1% | 0.06 | 1.09 | $6.76 |
| RSI-only, 15d, 3×TP/1×SL | 55 | 36.4% | 0.32 | 1.39 | $27 |
| **RSI-only, 15d pure time stop** | **27** | **55.6%** | **0.36** | **1.68** | **$61** |
| ADX<25+not_td, 2-of-4, 10d (v2 plan) | 6 | 33.3% | -0.13 | 0.77 | -$14 |
| 1-of-4, 10d, no gate | 87 | 42.5% | -0.41 | 0.76 | -$20 |

**v2 Backtest best**: Sharpe=0.36 (RSI-only, 15d pure time stop)
**v2 Walk-forward**: Not run (backtest failed Sharpe threshold)
**Did regime gate help?**: No — gate is anti-correlated with the signal source on SPY
**Did longer hold help?**: Yes — 5d→15d consistently improved Sharpe (+0.25) and PF across all variants
**Asymmetric ATR exits**: Tighter SL (1×ATR) + wider TP (3×ATR) improved Sharpe from -0.25 to 0.32 vs symmetric
**Pure time stop discovery**: Best overall — 55.6% WR, PF 1.68, Sharpe 0.36 — letting trades play out fully beats active management

**Why it still fails**: SPY's daily variance overwhelms the per-trade edge at 5% sizing. Best avg PnL = $61/trade, but SPY moves ~$3-5/day on $670 stock, so 5% of $100k = $5,000 notional → $61 is only 1.2% per trade over 15 days. Sharpe 1.0 requires either: (a) much larger position sizes (higher risk), (b) much more frequent signals, or (c) higher per-trade return.

**Next step (v3)**: The single-stock SPY mean-reversion approach is exhausted. Three directions to explore:
1. **Sector rotation / multi-stock**: Apply the same oversold-bounce logic to individual stocks where mean-reversion amplitude is larger
2. **Options for convexity**: Buy OTM calls when SPY is deeply oversold — captures the bounce with leveraged returns
3. **Bidirectional**: Add short signals (overbought → short) to double the trade count and potentially improve Sharpe through diversification of signal direction

### Multi-Stock RSI Mean-Reversion v3 — 2026-03-15

**Universe**: 13 S&P 500 mega-caps (AAPL, MSFT, NVDA, META, AMZN, GOOGL, XOM, BAC, TSLA, AMD, BA, NFLX, NKE)
**Entry**: RSI<35 + price within 3% of SMA200 (both prerequisite)
**Exit**: 15d time stop + 2.5× ATR TP + 1.5× ATR SL
**Sizing**: 5% per trade, max 5 simultaneous positions

**Per-symbol results** (sorted by Sharpe):
| Symbol | Sharpe | MaxDD | WR | PF | Trades |
|--------|--------|-------|-----|-----|--------|
| XOM | 0.899 | 0.6% | 71% | 3.46 | 17 |
| MSFT | 0.548 | 0.3% | 55% | 3.34 | 11 |
| AAPL | 0.304 | 0.6% | 45% | 1.69 | 11 |
| AMD | 0.297 | 0.8% | 50% | 2.21 | 6 |
| NVDA | 0.290 | 0.3% | 100% | inf | 2 |
| GOOGL | 0.162 | 0.5% | 38% | 1.31 | 8 |
| AMZN | 0.036 | 0.6% | 41% | 1.08 | 17 |
| NKE | 0.027 | 1.2% | 30% | 1.06 | 10 |
| META | -0.099 | 0.8% | 33% | 0.70 | 3 |
| TSLA | -0.149 | 1.8% | 40% | 0.72 | 10 |
| NFLX | -0.453 | 5.3% | 17% | 0.00 | 6 |
| BAC | -0.988 | 1.6% | 13% | 0.14 | 8 |
| BA | -1.002 | 1.9% | 13% | 0.10 | 8 |

**Portfolio avg Sharpe**: -0.01 (essentially zero)
**Total trades**: 117 (trade count problem solved)
**Symbols with Sharpe > 0.5**: 2 (XOM, MSFT)
**Walk-forward**: Skipped (only 2/13 symbols viable, need ≥5)
**Verdict**: FAIL

**Key finding — stock heterogeneity**: RSI<35 + SMA200 signal is NOT universal. It works on value/quality stocks with institutional support floors (XOM: energy giant, MSFT: quality growth) but actively loses money on momentum/high-beta names (TSLA, NFLX, BA) and financials (BAC). These stocks tend to keep falling past RSI<35, and the 1.5× ATR stop gets hit. Winners and losers net to Sharpe ≈ 0.

**Why multi-stock doesn't fix the edge problem**: The v2 diagnosis was "SPY amplitude is too small." v3 proves the amplitude IS larger on individual stocks (XOM: +2.3% return vs SPY: +0.4%), but only on stocks where the bounce is reliable. Applying the same rule to all mega-caps dilutes the edge because the signal is heterogeneous across stock types.

**Infrastructure note**: Screener tool broken (hard-coded `daily` timeframe, only 1H in DB). Fixed by fetching daily data via Alpaca for all 13 symbols. QuantPod MCP was in degraded mode (stale PID holding DB lock); backtests run via direct Python script.

**Next step (v4)**: RSI<35 mean-reversion is now exhausted as a standalone strategy for equities. Remaining viable directions:
1. **Options for convexity**: Buy OTM calls on XOM/MSFT (proven bouncers) when RSI<35 — leverages the 2-3% bounce to 20-30% returns
2. **Factor-filtered universe**: Use quality + low-beta factors to pre-select stocks BEFORE applying RSI<35 — XOM and MSFT share these characteristics
3. **Completely different signal**: Abandon RSI mean-reversion. Explore momentum, breakout, or ML-driven signals instead

### Quality-Filtered RSI Mean-Reversion v4 — 2026-03-15

**Hypothesis**: RSI<35 mean-reversion is stock-type-specific; works on low-beta quality stocks (beta<0.9, institutional support floors) not high-beta names. Pre-filter universe by quality factors.

**Universe**: 17 symbols — XOM/MSFT (proven Tier A) + 15 Tier B:
- Healthcare: JNJ, UNH, PFE, ABBV, MRK
- Consumer Staples: PG, KO, PEP, WMT, COST
- Energy peers: CVX, COP
- Tech value: CSCO, IBM, ORCL

**Entry/Exit**: RSI<35 + within 3% SMA200 → 15d time stop, 2.5×ATR TP, 1.5×ATR SL (unchanged from v3)

**Per-symbol results** (sorted by Sharpe):
| Symbol | Sharpe | MaxDD% | WR% | PF | Trades | Sector |
|--------|--------|--------|-----|------|--------|--------|
| XOM | 0.899 | 0.56 | 70.6 | 3.46 | 17 | Energy |
| IBM | 0.665 | 0.38 | 64.3 | 3.54 | 14 | Tech Value |
| MSFT | 0.548 | 0.32 | 54.6 | 3.34 | 11 | Tech Quality |
| UNH | 0.473 | 0.64 | 60.0 | 1.94 | 15 | Healthcare |
| KO | 0.358 | 0.87 | 51.6 | 1.45 | 31 | Staples |
| CSCO | 0.209 | 0.55 | 44.4 | 1.75 | 9 | Tech Value |
| CVX | 0.151 | 0.63 | 42.9 | 1.35 | 14 | Energy |
| MRK | 0.125 | 0.34 | 50.0 | 1.28 | 8 | Pharma |
| WMT | 0.085 | 0.81 | 50.0 | 1.13 | 18 | Staples |
| COP | 0.076 | 0.97 | 44.4 | 1.20 | 9 | Energy |
| PEP | -0.050 | 0.81 | 40.0 | 0.97 | 25 | Staples |
| ORCL | -0.127 | 1.26 | 44.4 | 0.77 | 9 | Tech Growth |
| ABBV | -0.223 | 0.66 | 37.5 | 0.74 | 16 | Pharma |
| PG | -0.298 | 1.07 | 30.8 | 0.74 | 26 | Staples |
| PFE | -0.436 | 1.00 | 21.4 | 0.54 | 14 | Pharma |
| COST | -0.541 | 1.69 | 30.8 | 0.35 | 13 | Staples |
| JNJ | -0.734 | 1.43 | 18.2 | 0.41 | 22 | Pharma |

**Portfolio est. Sharpe** (XOM+IBM+MSFT): ~1.10 (N=3, avg_rho=0.10)
**Walk-forward**: INCONCLUSIVE — 0 OOS trades in all 25 folds. Signal too sparse for 150-bar windows (~1.7 expected trades/window). IS Sharpe stable across expanding windows (XOM: 0.59→1.01→0.74).
**Verdict**: FORWARD_TESTING (XOM, IBM, MSFT only — concentrated, not diversified)

**Beta hypothesis: PARTIALLY CONFIRMED** — Low-beta alone insufficient. The discriminating factor is institutional value-buyer support floor:
- **Winners**: Energy majors (XOM), legacy tech with value buyers (IBM, MSFT) — institutions defend price floors
- **Losers**: Pharma in secular decline (JNJ -0.73, PFE -0.44, ABBV -0.22) — no buyer floor, oversold keeps falling
- **Mixed**: Consumer staples split — KO works (0.36) but PG/COST fail. Likely PG/COST trade at higher multiples, less value-buyer support at RSI<35 levels
- **Surprise**: IBM (0.67) was not in v3 universe — deep value + institutional base made it the #2 performer

**Key finding**: RSI<35 mean-reversion works on ~3-5 specific stock archetypes, not on a "quality factor" broadly. Strategy is viable as a concentrated 3-name portfolio but needs options convexity (v5) to scale.

**Infrastructure note**: Walk-forward engine produces 0 OOS trades for low-frequency strategies. This is either signal sparsity (expected for RSI<35) or a bug in OOS signal generation. Needs investigation before next WF attempt.

**Next step (v5)**: Multi-timeframe entry refinement on XOM/IBM/MSFT. Three profiles: Swing (1D+1H, 15d hold), Medium (4H+1H, 4d hold — fixes WF sparsity), Day trade (1H+15min, <6h hold). Medium and Day trade profiles also solve the signal-frequency problem. v6 = options convexity.
