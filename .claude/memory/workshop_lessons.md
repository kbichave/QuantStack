# Workshop Lessons

> Accumulated R&D learnings — your research memory
> Read at START of every /workshop session
> Update after: /workshop, /reflect

## Backtesting Pitfalls

- SPY mean-reversion with extreme oversold filters (RSI<35, Stoch<15, CCI<-150) produces only 48 trades in 6 years — insufficient for reliable Sharpe estimation
- Loosening oversold thresholds to increase trade count dilutes edge — v2 (RSI<40, CCI<-100) had worse Sharpe (0.063 vs 0.178) despite 21% more trades
- 5-day hard time stop constrains mean-reversion strategies — bounces from deep oversold may need 10-15 days to fully play out

## What Works in Each Regime

| Regime | Tends to Work | Tends to Fail | Confidence |
|--------|--------------|---------------|------------|
| trending_down + high_vol | TBD — mean-reversion bounce has positive win rate (64%) but insufficient Sharpe | Tight-threshold oversold bounces (too few signals) | Low (1 test) |

## Parameter Sensitivity

- RSI threshold 35→40: increased trades 48→58 but Sharpe dropped 65% (0.178→0.063). The strict threshold was the edge; relaxing it admitted noise.
- SMA200 proximity 1.5%→3%: contributed to signal dilution in v2.

## Failed Hypotheses (don't repeat these)

| Date | Hypothesis | Why It Failed | Regime | Takeaway |
|------|-----------|---------------|--------|----------|
| 2026-03-15 | SPY oversold bounce (RSI<35 + Stoch<15 + CCI<-150 at SMA200) with 5-day hold | Only 48 trades in 6 years (too few), Sharpe 0.178 (too low). Win rate 64% and PF 1.35 are promising but avg P&L too small ($14/trade on 5% sizing). | trending_down / high_vol | Mean-reversion on SPY needs either longer hold (>5d) to let bounce develop, or different instrument (higher-vol single stocks). The 5-day constraint kills the edge. Consider: (1) extending to 10-day hold, (2) using leveraged ETFs (SPXL/UPRO), (3) options for convexity. |

## Workshop Sessions

### SPY Swing 5-Day — 2026-03-15

**Hypothesis**: Oversold mean-reversion bounce at SMA200 support with multi-signal confirmation
**Regime fit**: trending_down / high_vol (current SPY regime: ADX 32.8, ATR pct 81%)
**Entry gate**: 3+ of 5: RSI<35, Stoch K<15, close within 1.5% of SMA200, ATR pct>60, CCI<-150
**Time stop**: 5 days hard
**Backtest v1**: Sharpe=0.178 MaxDD=1.31% WinRate=64.6% Trades=48 PF=1.35
**Backtest v2** (relaxed): Sharpe=0.063 MaxDD=1.44% WinRate=63.8% Trades=58 PF=1.14
**Walk-forward**: Not run (backtest failed)
**Verdict**: FAIL — both iterations below Sharpe target (1.0) and trade count target (60)
**Key finding**: The multi-signal oversold gate works (64% win rate, PF>1.3 in v1) but the edge per trade is too small for a 5-day hold on SPY. The strict oversold conditions that produce the edge are inherently rare.
**Watch out for**: Don't retry this exact approach — the 5-day constraint is the fundamental problem, not the signal selection
**Sit-out trigger**: Strategy is FAILED, no signals emitted
