---
name: review
description: Position and performance review — evaluate open positions, flag exits, check strategy health, propose promotions/retirements.
user_invocable: true
---

# /review — Position & Strategy Review Session

## Purpose

Review all open positions against their strategy specs. Check for exit signals,
regime mismatches, and strategy health. Propose promotions and retirements.

## Workflow

### Step 0: Read Context
- Read `.claude/memory/trade_journal.md` — recent trades
- Read `.claude/memory/strategy_registry.md` — active strategies
- Read `.claude/memory/regime_history.md` — current regime

### Step 1: System Status
Call `get_system_status` — confirm system is operational.
Call `get_portfolio_state` — all open positions.

**Check intraday monitor output (if available):**
The `IntradayMonitorFlow` runs every 30–60 min via cron. If it has fired since last
/review, check `.claude/memory/session_handoffs.md` for any handoff it wrote, or
check Discord for `[INTRADAY ALERT]` messages.
- `regime_reversals` in the report → symbols where regime flipped since entry (act immediately)
- `action_items` → review each before looking at individual positions

### Step 2: Position Review
For each open position:

**a) Position Monitor (Enhancement 5 — call first):**
- Call `get_position_monitor(symbol)` for a comprehensive snapshot.
- This returns: current_price, pnl_pct, days_held, current_regime, near_stop, near_target.
- Act immediately on flags:
  - `near_stop=True` → move to TIGHTEN or CLOSE
  - `near_target=True` → consider partial exit (present to user)
  - `days_held` > strategy max_holding_period → time stop triggered

**b) Strategy Check:**
- Which strategy opened it? (check trade_journal.md for the entry)
- Call `get_strategy` for the full spec.

**c) Exit Rule Check:**
- Has take_profit level been reached?
- Has stop_loss level been breached?
- Has the strategy's max holding period been exceeded?

**d) Regime Check:**
- `current_regime` from get_position_monitor is already computed.
- Is the current regime still within the strategy's `regime_affinity`?
- If regime shifted since entry, flag the mismatch.

**e) RL Check (if available):**
- Call `get_rl_recommendation` for position size adjustment signal.
- Note: this is advisory only (shadow mode).

### Step 3: Categorize Each Position

| Category | Condition | Action |
|----------|-----------|--------|
| **HOLD** | Within strategy parameters, regime fits | No action |
| **TIGHTEN** | Approaching exit level or regime softening | Note: move stop closer |
| **CLOSE** | Exit rule triggered, regime mismatch, or time stop | Execute close |
| **REVIEW** | Edge case needing human judgment | Present with analysis |

### Step 4: Execute Closures
For CLOSE positions:
- Present reasoning (which exit rule triggered, or why regime mismatch warrants close)
- Call `close_position` with detailed reasoning
- Report fill results

### Step 5: Strategy Performance Review
For each active strategy (`status in ["live", "forward_testing"]`):
- Call `get_strategy_performance(strategy_id, lookback_days=30)`
- Flag if:
  - Live Sharpe < 0.3 for 4+ weeks → retirement candidate
  - Live performance degraded > 30% from backtest → validation needed

### Step 6: Validate Flagged Strategies
For any strategy flagged in Step 5:
- Call `validate_strategy(strategy_id)` — re-run backtest and compare
- If `still_valid=False`: flag for retirement

### Step 7: Promotion & Retirement Candidates

**Promotion candidates (forward_testing → live):**
- 3+ weeks of forward testing
- Forward Sharpe > 0.8
- Max drawdown within spec (< backtest DD × 1.5)
- Present with evidence. DO NOT auto-promote. Human confirms.

**Retirement candidates (live → retired):**
- 4+ weeks underperformance (Sharpe < 0.3)
- OR regime_fit no longer matches recent regimes
- OR validate_strategy shows `still_valid=False`
- Call `retire_strategy` with detailed reason. Log in trade_journal.

**Reactivation note:**
- Retired strategies can return to forward_testing if:
  - Regime returns to the strategy's affinity
  - Fresh backtest via /workshop validates
  - Requires a new /workshop session

### Step 8: Regime Matrix Review
Call `update_regime_matrix_from_performance(lookback_days=60)`.
- Review proposed changes.
- If justified, apply via `set_regime_allocation`.

### Step 9: Fill Quality Audit (Enhancement 5 — run weekly)

For the last 20 fills, call `get_fill_quality(order_id)` for each:
- Track: average slippage_bps, fill_vs_vwap_bps, worst 3 fills (by slippage).
- Flag if avg slippage > 5 bps or any fill > 15 bps.
- Investigate: time-of-day pattern (avoid market open/close 5 minutes)?
  Illiquid symbols? Position sizes too large vs ADV?
- Log findings in `.claude/memory/agent_performance.md` under "Execution Quality".

### Step 10: Update Memory
- `.claude/memory/trade_journal.md` — closed positions, performance notes
- `.claude/memory/strategy_registry.md` — updated live stats, promotions, retirements
- `.claude/memory/agent_performance.md` — IC patterns + execution quality metrics
- `.claude/memory/session_handoffs.md` — findings relevant to other sessions

## Notes

- /review is a monitoring session, not a trading session. Its primary output
  is position management (closes) and strategy lifecycle decisions.
- Run weekly, or more often during volatile regimes.
- Promotions require human confirmation. Retirements can proceed autonomously
  with reasoning logged.
- If RL recommendations are available, present them alongside your analysis
  but make clear they are advisory (shadow mode).
