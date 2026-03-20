---
name: morning
description: Pre-market scan and opportunity ranking — systematic daily routine that produces a prioritized watchlist with entry levels, stops, and sizing.
user_invocable: true
---

# /morning — Pre-Market Scan & Opportunity Ranking

## Purpose

Systematic pre-market routine run at 08:30–09:00 ET before market open.
Produces a ranked opportunity table with entry levels, stops, and sizing.
Feeds directly into /trade sessions for the trading day.

**Cadence:** Daily, Mon–Fri
**Duration:** 5–10 minutes
**Output:** Ranked opportunity table + macro context summary

---

## Workflow

### Step 0: Read Context
- Read `.claude/memory/session_handoffs.md` — overnight handoffs, pending actions
- Read `.claude/memory/trade_journal.md` — open positions (need monitoring today?)
- Read `.claude/memory/strategy_registry.md` — active strategies and their regime fit
- Read `.claude/memory/regime_history.md` — is a regime transition in progress?

### Step 1: System Health
Call `get_system_status` via QuantPod MCP.
- Kill switch active → STOP, report reason
- Risk halted → STOP, daily loss limit breached yesterday
- Note broker mode (paper/live)

### Step 2: Portfolio State
Call `get_portfolio_state` via QuantPod MCP.
- Cash available, equity, gross exposure
- Open positions — note symbols already held (avoid doubling)
- Largest position % — flag if concentration risk is high (>15% single name)

### Step 3: Macro Context

**a) Broad market regime:**
- Call `get_regime(SPY)` — trend_regime, volatility_regime, confidence, ADX, ATR%
- If confidence < 0.60: macro regime is ambiguous → reduce all sizing to half

**b) Event calendar:**
- Call `mcp__quantcore__get_event_calendar(SPY, days_ahead=1)` for today's macro events
- FOMC, CPI, NFP today → flag HIGH event risk, reduce sizing 50% across the board
- Earnings season active → note, individual symbol checks in Step 5

**c) Volatility context:**
- Call `mcp__quantcore__compute_technical_indicators(SPY, "daily", ["atr", "adx"])`
- ATR > 2× its 20-day average → vol spike, reduce all sizes 50%
- ADX < 15 on SPY → choppy/directionless market, favor mean-reversion setups

### Step 4: Build Watchlist

Default watchlist (expand based on strategy_registry symbols):
```
SPY, QQQ, IWM,                    # Major indices/ETFs
AAPL, MSFT, NVDA, AMZN, GOOGL,   # Mega-cap tech
META, TSLA, AMD, AVGO,            # High-beta tech
JPM, GS, XLF,                     # Financials
XLE, XLV, XLY, XLP                # Sector ETFs
```

Remove from watchlist:
- Symbols already at max position size in portfolio
- Symbols with earnings TODAY (skip for swing trades; use /options instead)
- Symbols in `RISK_RESTRICTED_SYMBOLS`

### Step 5: Signal Scan

For each symbol in watchlist (up to 15):
- Call `get_signal_brief(symbol)` via QuantPod MCP (~2–5 sec each)
- Or call `run_multi_signal_brief(symbols)` for batch (up to 5 at a time, ~4–8 sec)

For each SignalBrief, extract:
- `market_bias` (bullish/bearish/neutral)
- `market_conviction` (0–1)
- `risk_environment` (low/normal/elevated/high)
- Per-symbol: `consensus_bias`, `consensus_conviction`, `pod_agreement`
- `collector_failures` — note if data is incomplete

### Step 6: Score and Rank

For each symbol with a non-neutral signal:

```
opportunity_score = consensus_conviction
                  × regime_alignment_factor
                  × (1 - event_risk_penalty)
                  × liquidity_factor

Where:
  regime_alignment_factor:
    1.0 if signal direction matches SPY regime (e.g., bullish + trending_up)
    0.7 if neutral regime
    0.3 if signal opposes regime (contrarian — needs very high conviction)

  event_risk_penalty:
    0.0 if no events within 3 days
    0.2 if earnings within 5 days
    0.5 if FOMC/CPI/NFP today

  liquidity_factor:
    1.0 for S&P 500 components
    0.9 for QQQ/ETFs
    0.7 for small-cap or < 1M ADV (flag for manual review)
```

Rank by `opportunity_score` descending. Take top 5.

### Step 7: Deep Dive on Top 5

For each of the top 5 ranked opportunities:

**a) Volume profile:**
- Call `mcp__quantcore__analyze_volume_profile(symbol, "daily", lookback_days=20)`
- Identify entry zones: High Volume Nodes (HVN) as support, Low Volume Nodes (LVN) as air pockets

**b) Multi-timeframe alignment:**
- Call `mcp__quantcore__compute_technical_indicators(symbol, "weekly", ["sma_20", "rsi", "adx"])`
- Entry valid only if weekly trend agrees with daily signal direction
- If weekly disagrees: downgrade to "watch" (not actionable today)

**c) Risk check:**
- Call `mcp__quantcore__check_risk_limits(symbol, proposed_size)` — does this fit in portfolio?
- If it would breach gross_exposure or position limits: reduce size or skip

### Step 8: Output

Present the morning brief as a structured table:

```
## Morning Brief — {date}

### Macro Context
- SPY Regime: {trend} + {vol} (confidence: {x}%)
- Events Today: {list or "none"}
- Vol Environment: {normal/elevated/spike}
- Sizing Adjustment: {none / half / quarter}

### Open Positions Check
| Symbol | P&L | Days Held | Status |
|--------|-----|-----------|--------|
| ...    | ... | ...       | HOLD/TIGHTEN/CLOSE |

### Opportunities (Ranked)
| Rank | Symbol | Bias | Conviction | Score | Entry Zone | Stop | Target | Size |
|------|--------|------|------------|-------|------------|------|--------|------|
| 1    | ...    | ...  | ...        | ...   | ...        | ...  | ...    | ...  |

### Watch List (not actionable today)
| Symbol | Reason | Trigger to Revisit |
|--------|--------|--------------------|
| ...    | ...    | ...                |

### Action Items
- [ ] /trade {symbol1} — highest conviction opportunity
- [ ] /review {symbol2} — position approaching stop
- [ ] /options {symbol3} — high IV rank, earnings in 5 days
```

### Step 9: Update Memory
- `.claude/memory/session_handoffs.md` — morning context for /trade sessions:
  - Macro regime summary
  - Top opportunities with entry levels
  - Symbols to avoid today (events, regime mismatch)
  - Any open positions needing attention

---

## When to Skip /morning

- Market holiday (no scan needed)
- Kill switch active (system halted)
- Weekend (no US equity trading)

---

## Notes

- /morning is a READ-ONLY analysis session — no trades are executed here
- If a clear opportunity emerges, log it in session_handoffs.md and run /trade separately
- The morning brief serves as the trading day's north star — reference it in /trade decisions
- Keep the watchlist focused: 15 symbols max for efficient scanning
- Expand watchlist gradually as new strategies are registered in strategy_registry.md
