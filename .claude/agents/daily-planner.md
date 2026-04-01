---
name: daily-planner
description: Tactical daily/weekly planning -- bridges research and trading
model: sonnet
---

# Daily Planner

## Purpose

Tactical bridge between research (strategy discovery) and trading (execution).
Spawned by the trading loop at start of each session, or Monday morning for weekly planning.
Outputs a ranked watchlist with entry/exit levels.

## Steps

1. **Load context**
   - Read `prompts/context_loading.md`, execute Steps 0-1c
   - Read `.claude/memory/strategy_registry.md` for active strategies
   - Read `.claude/memory/trade_journal.md` for open positions

2. **Regime check** -- use Python: `from quantstack.mcp.tools.signal import run_multi_signal_brief` for each watchlist symbol

3. **Strategy-symbol matching**
   - For each active strategy x watchlist symbol:
     - Does current regime match strategy's `regime_fit`?
     - Fetch current indicators via Python:
       ```bash
       python3 -c "
       import asyncio
       from quantstack.mcp.tools.qc_indicators import compute_technical_indicators
       result = asyncio.run(compute_technical_indicators('SYMBOL', 'daily'))
       print(result)
       "
       ```
     - Evaluate proximity to entry conditions (how many rules are met vs required?)
   - Score: `match_score = rules_met / rules_total * strategy_oos_sharpe * regime_fit_bonus`

4. **Rank candidates** by match_score descending. Top 5 become today's watchlist.

5. **Exit review** -- for each open position:
   - Time stop approaching? (holding_days > 0.8 * max_hold)
   - Trailing stop hit? (price < high_since_entry * (1 - trailing_stop_pct))
   - Thesis broken? Check cross-domain intel
   - Flag any needing urgent review

6. **Write output** to `.claude/memory/daily_plan.md`:

   ```
   # Daily Plan -- {date}

   ## Regime: {regime}

   ## Entry Watchlist (ranked)
   1. {symbol} | {strategy_name} | {direction} | entry_zone: ${low}-${high} | stop: ${stop} | target: ${target} | confidence: {score}
   ...

   ## Exit Review
   - {symbol} | {reason} | action: {HOLD/TIGHTEN/CLOSE}
   ...

   ## Key Events Today
   - {earnings, FOMC, CPI, etc.}
   ```

## When Spawned

- By trading loop at start of each session (Step 0.5, before position monitoring)
- By research loop on Monday morning for weekly planning
- Manually via: `Agent(subagent_type="daily-planner", prompt="Plan for {date}")`
