# QQQ Research & Trading Memory

## Strategies

| ID | Name | OOS Sharpe | Overfit Ratio | Status |
|----|------|-----------|--------------|--------|
| regime_momentum_v1 | Regime Momentum | 1.346 | 0.58 | validated — STRONGEST ETF |

## ML Models

| Date | Model | AUC (OOS) | IS-OOS Gap | Verdict |
|------|-------|-----------|------------|---------|
| 2026-03-21 | LightGBM | 0.7122 | 0.133 | champion |

## Options Research (2026-03-25)

### Market Context
| Field | Value |
|-------|-------|
| QQQ Price | $583.98 (as of 2026-03-24) |
| Trend Regime | trending_down |
| ADX | 31.21 (strong trend) |
| +DI / -DI | 12.65 / 33.29 (bearish confirmation) |
| ATR (daily) | $10.24 (68.7th percentile) |
| Market Regime | bear / normal vol / risk-off |
| ATM IV (30d) | 25% (synthetic) |
| Realized Vol (20d) | 13.25% |
| Vol Premium | ~11.75pp — options are RICH vs realized |

**Signal:** allows_short=true, allows_long=false. Strong trend, options expensive vs realized.

---

### Candidate Structures (30 DTE)

#### 1. ATM Bear Put Spread — 585P/570P ★ PREFERRED
| Metric | Value |
|--------|-------|
| Net Debit | ~$6.42/share ($642/contract) |
| Max Profit | ~$8.58/share ($858/contract) |
| Max Loss | ~$6.42/share ($642/contract) |
| R/R | 1.34:1 |
| Breakeven | $578.58 (needs only –0.93% move) |
| POP | 41.6% |
| Net Delta | ~–0.09 (put spread) |

**Why preferred:** Breakeven only $5.40 below spot. In a trending_down regime with ADX 31, this is a high-probability move. Spread structure mitigates expensive IV (sell rich vol on the $570 short leg).

#### 2. OTM Bear Put Spread — 575P/560P (high-conviction variant)
| Metric | Value |
|--------|-------|
| Net Debit | ~$5.04/share ($504/contract) |
| Max Profit | ~$9.96/share ($996/contract) |
| Max Loss | ~$5.04/share ($504/contract) |
| R/R | 1.97:1 |
| Breakeven | $569.96 (needs –2.40% move) |
| POP | 35% |

**When to use:** If momentum accelerates — e.g., close below $575 support with volume. ATR of $10.24 means QQQ can cover this in ~1.5 days on a panic day. Lower cost, better R/R. Use on regime deterioration or catalyst (FOMC/earnings).

#### 3. Long ATM Put — 585P (AVOID in current vol environment)
- Debit: ~$16.22/share ($1,622/contract)
- Theta burn: –$23.76/day (punishing at $10.24 ATR)
- IV 25% vs realized 13% = overpaying ~$4–5/share in vol premium
- Only viable if anticipating a vol spike (VIX regime shift to high)

---

### Position Sizing (Risk-Gated)
- Max options premium per position: 2% of equity
- $50K account → max 1 contract ATM spread ($642 risk)
- $100K account → max 3 contracts ATM spread ($1,926 risk)

### Entry Criteria
- Confirm trending_down regime on day of entry (ADX > 25, -DI > +DI)
- Enter on intraday bounce toward $585–590 resistance (better fill)
- 30 DTE minimum; avoid entering < 21 DTE

### Exit Rules
- **Take profit:** 50–75% of max gain → $429–$643 per contract (ATM spread)
- **Stop loss:** 50% of debit paid → –$321 per contract
- **Time stop:** Roll down/out at 21 DTE if still bearish

### Key Risk Factors
1. Vol crush if market stabilizes → hurts long put value
2. Whipsaw / bear trap rally → spread decays but loss is capped
3. Data is synthetic — live IV skew may show put-heavy skew, making spread more expensive. Verify before execution.

---

## Lessons (QQQ-specific)

1. **Best regime signal quality** of all ETFs tested. OOS 1.346 vs IS 0.785 — performs BETTER on recent data.
2. Tech-heavy composition creates clean momentum trends.
3. OOS > IS (overfit ratio 0.58) — genuine signal, not overfitting.
4. **Options insight (2026-03-25):** IV 25% vs realized 13% → always prefer spreads over naked puts for bearish plays. The vol premium is too large to justify buying naked options unless anticipating a vol regime shift.
