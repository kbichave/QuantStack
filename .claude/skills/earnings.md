---
name: earnings
description: Earnings event playbook — systematic pre/post-earnings analysis for options structures and equity swing positioning. Integrates IV analysis, historical moves, press releases, and analyst estimates.
user_invocable: true
---

# /earnings — Earnings Event Playbook

## Purpose

Systematic analysis before and after earnings announcements. This is the
bridge between /options (structure selection) and /trade (equity positioning)
when an earnings event is the primary catalyst.

**When to use:** 1-14 days before a symbol's earnings, or immediately after
an earnings print to evaluate continuation vs reversal.

**Holding period:** 1-5 days (pre-earnings positioning exits before or at earnings;
post-earnings trades are swing trades riding the gap direction).

---

## Workflow

### Step 0: Read Context
- Read `.claude/memory/trade_journal.md` — any open positions in this name?
- Read `.claude/memory/strategy_registry.md` — any earnings-specific strategies registered?
- Read `.claude/memory/workshop_lessons.md` — past earnings trade lessons

### Step 1: Earnings Calendar Confirmation

Call `mcp__quantpod__get_event_calendar(symbol, days_ahead=30)`:
- Confirm exact earnings date and time (pre-market / after-hours)
- Days until earnings (DTE_earnings)
- Is this a quarterly report (10-Q) or annual (10-K)?

| DTE_earnings | Phase | Available Actions |
|-------------|-------|-------------------|
| > 14 days | Too early | Monitor only, no position. Set alert. |
| 7-14 days | Pre-earnings setup | IV analysis, structure selection, small initial position |
| 1-7 days | Active positioning | Full structure deployment, final sizing |
| 0 (today) | Event day | No new entries. Monitor existing only. |
| -1 to -3 | Post-earnings | Gap analysis, continuation/reversal assessment |

### Step 2: Historical Earnings Analysis

Call `mcp__quantpod__get_earnings_data(symbol, limit=8)` for last 8 quarters:
- **Average absolute move**: mean(|surprise_pct|) — this is the expected move
- **Beat/miss ratio**: how often does this company beat estimates?
- **Directional bias after beats**: does stock rally or sell on beats? (some stocks sell on beats — "buy the rumor, sell the news")
- **Directional bias after misses**: how severe are miss selloffs?
- **Guidance matters more**: does guidance change override the EPS number?

Compute:
```
expected_move_pct = mean(|actual_move|) over last 4 quarters
beat_rate = count(surprise > 0) / total
post_beat_avg_return = mean(1-day return when surprise > 0)
post_miss_avg_return = mean(1-day return when surprise < 0)
```

### Step 3: Analyst Estimates Context

Call `mcp__quantpod__get_analyst_estimates(symbol)`:
- Consensus EPS estimate and range (high/low)
- Revenue estimate
- Number of analysts covering (>10 = well-covered, <5 = thin coverage → larger surprise risk)
- Recent revisions: were estimates revised up or down in last 30 days?

**Revision momentum matters:**
- Estimates revised UP into earnings → positive setup (analysts catching up to reality)
- Estimates revised DOWN → lowered bar, potential beat → but may already be priced in
- No revisions → stale estimates, higher surprise risk

### Step 4: Press Release & Management Tone

Call `get_earnings_press_releases(symbol, limit=3)`:
- Most recent press release tone: optimistic, cautious, or defensive?
- Any pre-announcements or guidance updates?
- Key themes: cost cutting, growth investments, margin expansion, headwinds?

Call `get_company_news(symbol, limit=10)`:
- Recent analyst upgrades/downgrades ahead of earnings?
- Sector peer earnings results (if peers already reported, did the sector beat or miss?)

### Step 5: IV Regime Analysis

Call `mcp__quantpod__get_iv_surface(symbol)`:
- `iv_rank`: current IV vs 52-week range (0-100)
- `atm_iv_30d`: front-month ATM implied volatility
- `skew_25d`: put skew (positive = bearish positioning)

Call `mcp__quantpod__get_options_chain(symbol)`:
- At-the-money straddle price → market's implied move
- Compare implied move vs historical average move:
  - Implied > historical × 1.3 → IV is OVERPRICED (favor selling premium)
  - Implied < historical × 0.7 → IV is UNDERPRICED (favor buying premium)
  - Within range → fairly priced

```
implied_move = atm_straddle_price / stock_price
historical_move = expected_move_pct from Step 2
iv_premium_ratio = implied_move / historical_move
```

### Step 6: Structure Decision Matrix

**Pre-earnings (DTE 1-14):**

| IV Premium Ratio | Signal Bias | → Structure |
|-----------------|-------------|-------------|
| > 1.3 (overpriced) | Neutral | Iron condor: sell straddle, buy wings at 1.5× expected move |
| > 1.3 (overpriced) | Directional | Credit spread against signal direction (collect premium) |
| < 0.7 (underpriced) | Neutral | Long straddle (cheap premium, expect large move) |
| < 0.7 (underpriced) | Directional | Debit spread in signal direction |
| 0.7-1.3 (fair) | Strong directional | Small debit spread in signal direction |
| 0.7-1.3 (fair) | Weak/neutral | Skip — no edge in options premium |

**Post-earnings (gap already occurred):**

| Gap Direction | Relative to Expected Move | Volume | → Action |
|--------------|--------------------------|--------|----------|
| Gap UP | < expected move | High | Continuation likely — buy pullback |
| Gap UP | > 2× expected move | Any | Exhaustion risk — wait 1 day, then reassess |
| Gap DOWN | < expected move | High | Support test — watch for reversal at key level |
| Gap DOWN | > 2× expected move | Any | Capitulation — wait for stabilization (2-3 days) |
| Flat (< 1%) | N/A | Low | Disappointment — IV crush destroys options, equity may drift |

### Step 7: Risk Rules (HARD — never violate)

- **Never sell naked options through earnings** (undefined risk on gap)
- **Max premium at risk: 2% of equity per position** (from risk_gate)
- **DTE at entry: 7-45 days** (avoid gamma acceleration and far-dated speculation)
- **Iron condor breakevens: outside 1.5× expected move** (not just 1× — earnings surprise harder)
- **No equity swing entries within 24h of earnings** — use options for defined risk instead
- **Position size: 50% of normal** when DTE_earnings < 7 (event risk premium)

### Step 8: Execute

**For options structures:** Follow /options Step 7 (register strategy + execute via broker).

**For post-earnings equity swings:** Follow /trade Steps 6-8 (trade plan + pre-flight + execute).

In both cases:
- `paper_mode=True` unless explicitly confirmed by human
- Record `strategy_id` linking to an earnings-specific strategy
- Log IV regime, implied vs historical move, and structure reasoning

### Step 9: Update Memory

`trade_journal.md` entry must include:
- Earnings date and time
- Expected move (historical) vs implied move (options) vs actual move (realized)
- IV premium ratio at entry
- Structure type and P&L
- Lesson: was IV overpriced or underpriced? Did direction match signal?

`workshop_lessons.md`: if the trade revealed a pattern (e.g., "NVDA always sells off
on beats" or "XLF earnings are consistently underpriced"), record it for future use.

---

## Earnings Season Awareness

During earnings season (Jan/Apr/Jul/Oct), multiple names report simultaneously.
Portfolio-level considerations:
- Max 3 active earnings positions at once (sector/correlation diversification)
- If 2+ correlated names report same week, pick the highest IV premium ratio
- Watch for sector contagion: if AAPL misses, MSFT/GOOGL/META may sell off sympathetically

---

## What This Skill Does NOT Do

- No position management after entry (use /review for that)
- No fundamental valuation (use /invest for long-term thesis)
- No intraday earnings trades (0DTE, gamma scalping — too speculative)
- No binary bet sizing (never risk >2% on a single earnings event)
