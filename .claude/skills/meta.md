---
name: meta
description: Portfolio-level orchestration — allocate strategies to regimes, run parallel analysis, resolve conflicts, execute approved trades.
user_invocable: true
---

# /meta — Portfolio Meta-Orchestrator

## Purpose

Allocate capital across strategies based on the current regime, run analysis
for multiple symbols, resolve portfolio-level conflicts, and execute the
resulting trade plan. This is the highest-level trading session — it
coordinates everything below it.

## Workflow

### Step 0: Read Context
- Read `.claude/memory/strategy_registry.md` — active strategies and their status
- Read `.claude/memory/regime_history.md` — current and recent regimes
- Read `.claude/memory/trade_journal.md` — recent performance
- Read `.claude/memory/agent_performance.md` — IC trust weightings
- Read `.claude/memory/session_handoffs.md` — handoffs from /workshop, /decode, /reflect

### Step 1: System Health Check
Call `get_system_status`.
- If kill switch active: STOP.
- If risk halted: STOP.
- Note broker mode.

### Step 2: Regime Detection
Call `get_regime` for each symbol in the primary watchlist.
- Record regimes for all symbols.
- Identify the dominant regime across the watchlist.
- Flag any regime transitions since last session.

### Step 3: Strategy Inventory
Call `list_strategies(status in ["live", "forward_testing"])`.
- Note which strategies are eligible for capital allocation.
- Cross-reference with regime affinities.

### Step 4: Allocation
Use the regime-strategy matrix:
- Call `get_regime_strategies(regime)` for the current regime.
- Review allocations and adjust if strategy performance data warrants.
- Present the allocation plan with reasoning for each strategy:
  - Strategy name, capital %, mode (paper/live), regime score
  - Why this strategy fits the current regime
  - Any warnings (low confidence, forward_testing cap)

### Step 5: Multi-Symbol Analysis
For each allocated strategy, call `run_analysis` or `run_multi_analysis`
for the relevant symbols.
- Collect all DailyBriefs.
- Note which symbols have actionable signals.

### Step 6: Generate Candidate Trades
From each DailyBrief + strategy allocation:
- Build a candidate trade for each actionable signal
- Include: symbol, action, confidence, strategy_id, capital_pct

### Step 5a: Cross-Strategy Correlation Check (Enhancement 2)

Before running conflict resolution, assess portfolio-level risk from correlated signals:

**Pairwise correlation:**
- For all candidate trades, if two symbols are in the same sector or ETF category,
  treat their combined exposure as a single position for risk limit purposes.
- If two proposed trades are historically correlated > 0.7 (check sector/beta alignment):
  → Reduce both to "quarter" size, OR pick the higher-conviction signal and skip the other.

**Portfolio beta check:**
- Estimate the portfolio beta impact of all proposed trades combined.
  Use the DailyBrief regime confidence as a proxy for market sensitivity.
- If the combined beta of proposed trades would exceed 1.5 (high directional market bet):
  → Reduce the largest proposed position size by one step (full→half, half→quarter).

**Calendar event check:**
- Call `mcp__quantcore__get_event_calendar` for any symbols with earnings in 24 hours.
- Those symbols: force to paper_mode, or skip if portfolio already near gross_exposure limit.

**RL cross-portfolio sizing:**
- Call `get_rl_recommendation` for each proposed trade.
- If RL recommends REDUCE on multiple symbols simultaneously: respect the pattern.

### Step 7: Conflict Resolution
Call `resolve_portfolio_conflicts(proposed_trades)`.
- Review conflict resolution decisions.
- Present the final approved trade list with reasoning:
  - Kept trades: why they survived
  - Skipped trades: why they were filtered (conflict, low confidence)
  - Adjusted trades: what changed (position size, entry)

### Step 8: Pre-Flight
Call `get_risk_metrics` — confirm headroom for all proposed trades:
- daily_headroom_pct > 0
- gross_exposure + new trades < max_gross_exposure
- No single position exceeds max_position_pct

### Step 9: Execute
For each approved trade:
- `paper_mode=True` for **all** forward_testing strategies (always)
- `paper_mode=True` for live strategies UNLESS human explicitly confirms live
- Call `execute_trade` with full reasoning and strategy_id
- Report fills

### Step 10: Post-Execution Review
Check for strategy lifecycle actions:

**Promotion (forward_testing → live):**
- 3+ weeks of forward testing
- Forward Sharpe > 0.8
- Max drawdown within spec
- Requires human confirmation
- Action: flag for user, do NOT auto-promote

**Demotion (live → retired):**
- 4+ weeks of underperformance (Sharpe < 0.3)
- OR regime_fit no longer matches recent regimes
- Action: flag for /reflect session

**Reactivation (retired → forward_testing):**
- Regime returns and fresh backtest validates
- Requires /workshop session

### Step 11: Update Memory
- `.claude/memory/trade_journal.md` — all executions with strategy linkage
- `.claude/memory/strategy_registry.md` — update performance stats
- `.claude/memory/regime_history.md` — log any regime shifts
- `.claude/memory/session_handoffs.md` — portfolio-level observations

## Notes

- /meta is the most complex skill. Run it when you have time to review
  the full portfolio, not for quick single-symbol trades (use /trade instead).
- The allocation engine prioritizes strategies by regime_score × Sharpe.
  Forward_testing strategies are always capped at 10% allocation.
- Signal conflicts resolve conservatively: genuine disagreements → SKIP.
- Live execution requires explicit human confirmation for each trade.
  Paper mode is always safe to run autonomously.
