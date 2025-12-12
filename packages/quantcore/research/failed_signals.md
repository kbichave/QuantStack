# Failed Signals Documentation

This document tracks signals and strategies that were tested but ultimately **did not work**. Documenting failures is crucial for:
1. Avoiding repeated mistakes
2. Understanding market dynamics
3. Demonstrating research rigor
4. Learning from negative results

---

## Template for Failed Signal

```markdown
### Signal Name
**Hypothesis**: [What we expected]
**Implementation**: [How we tested it]
**Results**: [What actually happened]
**Why It Failed**: [Root cause analysis]
**Lessons Learned**: [What we learned]
**Date Tested**: [When]
```

---

## Failed Signals

### 1. Simple RSI Mean Reversion (RSI < 30 → Long)

**Hypothesis**: Stocks with RSI < 30 are oversold and will revert to mean.

**Implementation**:
- Entry: Long when RSI(14) < 30
- Exit: RSI > 50 or 5-day holding period
- Universe: S&P 500 constituents
- Period: 2015-2023

**Results**:
- Gross Sharpe: 0.45
- Net Sharpe (after costs): -0.12
- Win Rate: 52%
- Average holding: 3.2 days
- Turnover: 400%/year

**Why It Failed**:
1. **Transaction costs**: High turnover destroyed edge
2. **Crowded trade**: Signal too well-known, alpha decayed
3. **Regime dependence**: Only worked in bull markets
4. **No confirmation**: No momentum/trend confirmation

**Lessons Learned**:
- Simple indicators alone are not sufficient
- Must account for transaction costs from the start
- Need regime filtering for mean reversion
- Confirmation signals reduce false positives

**Date Tested**: January 2024

---

### 2. Overnight Gap Fade Strategy

**Hypothesis**: Large overnight gaps (>1%) will partially revert during the trading day.

**Implementation**:
- Entry: Short stocks gapping up >1% at open, Long stocks gapping down >1%
- Exit: Close of same day
- Universe: Liquid large caps (ADV > $10M)
- Period: 2018-2023

**Results**:
- Win Rate: 48%
- Sharpe: 0.22
- Average return per trade: -0.05%

**Why It Failed**:
1. **Momentum continuation**: Gaps often continued, especially on news
2. **Adverse selection**: Large gaps often had fundamental reasons
3. **Execution challenges**: Difficult to get good fills at open
4. **Slippage**: Opening auction spreads ate into edge

**Lessons Learned**:
- Need to distinguish information-driven vs. liquidity-driven gaps
- Opening execution is challenging and costly
- Intraday strategies face higher execution hurdles

**Date Tested**: March 2024

---

### 3. VIX Regime Timing

**Hypothesis**: Enter equity positions only when VIX < 20, exit when VIX > 25.

**Implementation**:
- Entry: Long SPY when VIX crosses below 20
- Exit: Close when VIX crosses above 25
- Period: 2010-2023

**Results**:
- Sharpe: 0.65 (vs 0.72 buy-and-hold)
- Missed upside: 15% of total return
- Whipsaw losses: 8% of capital

**Why It Failed**:
1. **Lagging indicator**: VIX spikes after moves already started
2. **Whipsaws**: VIX oscillates around thresholds causing frequent trading
3. **Missed recovery**: Best days often follow VIX spikes
4. **Parameter sensitivity**: Results highly sensitive to threshold choice

**Lessons Learned**:
- VIX better for hedging than timing
- Lagging indicators cause adverse selection
- Avoid binary rules on continuous indicators
- Missing best days is very costly

**Date Tested**: February 2024

---

### 4. Earnings Momentum (Post-Announcement Drift)

**Hypothesis**: Stocks beating earnings estimates continue to outperform for 60 days.

**Implementation**:
- Entry: Long on day after earnings beat (>5% surprise)
- Exit: 60 days later
- Universe: Russell 1000
- Period: 2015-2023

**Results**:
- Pre-2020 Sharpe: 0.85
- Post-2020 Sharpe: 0.15
- Overall: Alpha decayed significantly

**Why It Failed**:
1. **Alpha decay**: Well-documented anomaly, now crowded
2. **Faster incorporation**: Markets now price in surprises faster
3. **Options market**: Options pricing reflects expected surprises
4. **Factor exposure**: Much of return explained by momentum factor

**Lessons Learned**:
- Published anomalies decay after publication
- Need to check for factor exposure
- Faster markets reduce drift duration
- Original academic results often don't replicate in live trading

**Date Tested**: April 2024

---

### 5. Sector Rotation Based on Economic Indicators

**Hypothesis**: Leading economic indicators predict sector performance.

**Implementation**:
- Monthly rebalance based on ISM, housing starts, jobless claims
- Overweight cyclicals when indicators improving
- Overweight defensives when indicators deteriorating
- Period: 2010-2023

**Results**:
- Information Ratio: 0.15
- Tracking Error: 8%
- Turnover: 200%/year

**Why It Failed**:
1. **Signal delay**: Economic data lagged market movements
2. **Revisions**: Initial releases often revised substantially
3. **Nonlinear relationships**: Regime breaks invalidated relationships
4. **Crowding**: Macro hedge funds trade same signals

**Lessons Learned**:
- Economic indicators lag markets
- Need real-time, unrevised data
- Macro timing is extremely difficult
- Consider contrarian macro bets instead

**Date Tested**: May 2024

---

## Signals Requiring More Research

### Under Investigation

1. **Cross-asset momentum spillovers**: Does commodity momentum predict equity sector returns?
2. **Options-implied skew signals**: Can risk-neutral skew predict future returns?
3. **Insider trading patterns**: Do cluster insider buys predict outperformance?

### Abandoned (Insufficient Data)

1. **Private market signals**: Lack of reliable, timely data
2. **Social media sentiment**: Data quality issues, survivorship bias
3. **Alternative data (satellite, credit card)**: Too expensive to evaluate

---

## Key Takeaways from Failed Research

1. **Transaction costs matter more than expected**
   - Always model costs from the start
   - High turnover strategies rarely survive costs

2. **Published anomalies decay**
   - Academic research is backward-looking
   - Assume 50%+ alpha decay post-publication

3. **Regime changes break relationships**
   - No strategy works in all regimes
   - Need regime detection and adaptation

4. **Simple is rarely enough**
   - Single-indicator strategies are crowded
   - Confirmation and filtering required

5. **Execution is underestimated**
   - Slippage compounds over many trades
   - Opening/closing auction execution is hard

