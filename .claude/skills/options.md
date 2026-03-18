---
name: options
description: Short-term options trading session — event-driven, Greeks/IV-based, expiry cadence management. Requires Phase 1 live chain fetching (get_options_chain, get_iv_surface).
user_invocable: true
---

# /options — Short-Term Options Trading Session

## Purpose

Evaluate and execute short-term options structures. Decisions are driven by:
- Event calendar (earnings, FOMC) — determines structure type
- IV regime (iv_rank from get_iv_surface) — determines premium selling vs buying
- Market regime + signal bias — determines directional vs neutral structures

**Sizing metric:** Max premium at risk (not notional equity %).
**P&L driver:** Theta decay AND/OR directional move.
**Never trade naked options.** Minimum: defined-risk structure (spread or condor).

---

## Workflow

### Step 0: Context + Health

- Call `get_system_status()` — kill switch / risk halt. STOP if active.
- Read `.claude/memory/strategy_registry.md` — any active options strategies?
- Read `.claude/memory/trade_journal.md` — open options positions (DTE check)

### Step 1: Event Calendar

Call `get_event_calendar(symbol, days_ahead=30)`:

| Condition | Implication |
|-----------|-------------|
| Earnings within 7 days | Short premium (IV expansion) → iron condor or short strangle if IV rank > 50% |
| Earnings 7–21 days | Directional debit spread if signal strong; otherwise wait |
| Earnings > 21 days | IV regime drives structure |
| FOMC within 2 days | Reduce all sizes 50% or skip |

**Critical rule:** Do NOT sell premium through earnings (IV crush can be exceeded by gap).
The exception: iron condor where breakevens are > 2× expected move — explicit risk acceptance.

### Step 1.5: Earnings Press Release Analysis

If earnings within 14 days (from Step 1):
- Call `get_earnings_press_releases(symbol, limit=3)` for the 3 most recent press releases.
- Analyze management tone: are they guiding up or down?
  - Look for keywords: "headwinds," "challenging environment," "cautious outlook" (bearish lean)
  - Look for keywords: "accelerating," "raised guidance," "strong demand" (bullish lean)
- Check if recent press releases mention tailwinds/headwinds that the market hasn't priced in.
- This informs structure selection in Step 4:
  - Bearish tone → lean toward put-side structures (bear put spread, put credit spread)
  - Bullish tone → lean toward call-side structures (bull call spread, call credit spread)
  - Mixed/neutral → stick with IV-driven structure selection (iron condor, straddle)

### Step 2: Regime + Signal

Call `get_regime(symbol)` → trend_regime + volatility_regime.
Call `get_signal_brief(symbol)` → market_bias, conviction.

| Regime | Signal bias | → Directional lean |
|--------|-------------|---------------------|
| trending_up | bullish | call side / bull spread |
| trending_down | bearish | put side / bear spread |
| ranging | any | neutral structures |
| any | conviction < 0.5 | neutral only |

Also call `get_company_news(symbol, limit=5)` for recent news that might affect IV.
Breaking news (M&A rumors, FDA decisions, activist involvement) can cause IV to spike
or collapse independent of regime — factor this into IV rank interpretation in Step 4.

### Step 3: Live Options Chain

**Always use live data for execution:**
```
get_options_chain(symbol, expiry_min_days=7, expiry_max_days=45)
get_iv_surface(symbol)
```

Extract from `get_iv_surface`:
- `iv_rank`: current IV vs 52-week range (0–100)
- `atm_iv_30d`: front-month ATM IV
- `skew_25d`: positive = put skew (bearish demand)

**If source="synthetic":** note this in trade reasoning. Do NOT execute live orders
with synthetic chain data — first confirm via broker MCP (`get_option_chains` from
Alpaca or IBKR).

### Step 4: Structure Selection

Use this decision matrix:

| IV rank | Regime | Event | → Structure |
|---------|--------|-------|-------------|
| > 50% | ranging | earnings < 14d | Iron condor (short strangle + wings) |
| > 50% | any | no earnings | Credit spread (match short side to signal direction) |
| > 50% | trending | no earnings | Credit spread against the trend (fade fade) |
| < 30% | trending_up | no event | Bull call spread (buy ATM call, sell OTM call) |
| < 30% | trending_down | no event | Bear put spread (buy ATM put, sell OTM put) |
| < 30% | ranging | no event | Long straddle or strangle (low premium entry) |
| 30–50% | any | earnings 14–21d | Debit spread matching signal bias |
| 30–50% | any | no event | Calendar spread or skip |

**Never trade:**
- Naked options (undefined risk)
- DTE < 7 at entry (gamma acceleration)
- DTE > 60 at entry (speculative far-dated positions)

### Step 5: Structure Analysis

Once strikes and expiry are selected, call:
```
analyze_option_structure(structure_spec)
score_trade_structure(structure_spec, market_regime=regime)
```

**Minimum viability thresholds:**
- Credit spreads: risk/reward ≤ 3:1 (max risk ≤ 3× max credit)
- Debit spreads: debit ≤ 40% of strike width (implies ≥ 2.5:1 max profit/max loss)
- Iron condors: breakevens must be outside expected move (1 SD = atm_iv × spot × √(dte/365))
- score_trade_structure: total_score ≥ 50 to proceed

### Step 6: Risk Check

Call `get_risk_metrics()`. Check against options-specific limits:
- New position premium ≤ 2% of equity (`max_premium_at_risk_pct`)
- Total options premium outstanding ≤ 8% of equity (`max_total_premium_pct`)
- DTE at entry: 7–60 days

If premium at risk > 2%: scale down contracts to fit within limit.
If total portfolio options premium would exceed 8%: skip this trade.

### Step 7: Register Strategy + Execute

Register if this is a new options strategy type:
```python
register_strategy(
    name="options_credit_spread_ranging",
    entry_rules=[],
    exit_rules=[],
    parameters={"structure": "credit_spread", "max_dte": 45, "min_dte": 7},
    instrument_type="options",
    time_horizon="swing",
    holding_period_days=21,
    regime_affinity={"ranging": 0.9, "trending_up": 0.5},
)
```

Execute via broker MCP (Alpaca or IBKR multi-leg order). **Paper mode default.**

Record in audit trail:
- Structure definition (all legs with strikes and expiries)
- Entry credit/debit
- Breakevens
- Max profit / max loss
- Management rules (exit triggers)

### Step 8: State Management Rules in trade_journal.md

Every options entry must record exit rules explicitly:

**Credit spreads / condors:**
- Exit at 50% of max profit (standard — don't hold to expiry)
- Exit at 21 DTE regardless of P&L (avoid gamma acceleration)
- Exit at 2× credit received if position moves against you (risk management)

**Debit spreads:**
- Exit at 2× premium paid if wrong (stop loss)
- Exit at 80% of max profit if reached early (lock in gains)
- Exit at 21 DTE if not at 50% profit yet (time stop)

**Management cadence:** Review in `/review` every 3–5 days. Check theta bleed, DTE, and
proximity to breakeven.

### Step 9: Update Memory

`trade_journal.md` entry must include:
- IV regime at entry (iv_rank, atm_iv_30d)
- Structure type and all legs
- Entry credit/debit per contract
- Breakevens
- Explicit exit rules (see Step 8)
- Next mandatory review date (14 DTE or when P&L ±30%)

`strategy_registry.md`: confirm strategy registered with `instrument_type="options"`.

---

## Exit Criteria (enforced in /review)

Options positions are reviewed for exit on every `/review` session:

1. **DTE < 21**: EXIT regardless of P&L. Gamma risk dominates.
2. **50% of max profit reached**: EXIT for credit structures. Lock in.
3. **2× max credit loss**: EXIT. Stop loss triggered.
4. **IV spike > 50%** since entry: re-evaluate — short premium positions are hurting;
   consider early exit or rolling.
5. **Price within 1 ATR of short strike**: TIGHTEN or EXIT.

---

## What This Skill Does NOT Do

- No stock purchases or short sales — pure options only
- No undefined-risk positions (naked calls or puts)
- No DTE < 7 entries — gamma pins create binary outcomes
- No earnings straddles without explicit iron condor structure (defined risk)
- No same-day expiry (0DTE) — too speculative, too much gamma risk

---

## IV Rank Interpretation

| IV rank | Interpretation | Strategy bias |
|---------|---------------|---------------|
| > 70% | Very elevated | Sell premium aggressively |
| 50–70% | Elevated | Sell premium with defined risk |
| 30–50% | Normal | Debit spreads if directional; otherwise skip |
| < 30% | Compressed | Buy options (debit structures or straddles) |
| < 15% | Extremely low | Long straddle / calendar spread (mean reversion of vol) |
