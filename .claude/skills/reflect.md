---
name: reflect
description: Review recent trade outcomes, update memory files, fix skills that aren't working. Run weekly or after 10+ trades.
user_invocable: true
---

# /reflect — Self-Improvement Session

## Purpose

Review recent outcomes. Update memory files. Fix skills that aren't working.
Run weekly, after 10+ trades, or when prompted.

## Workflow

### Step 1: Gather Data
- Read `.claude/memory/trade_journal.md` — all trades since last /reflect
- Read `.claude/memory/strategy_registry.md` — all active strategies
- Read `.claude/memory/agent_performance.md` — current IC ratings
- Read `.claude/memory/workshop_lessons.md` — current learnings
- Check: when was the last /reflect? (look for last `reflect:` git commit)

### Step 2: Trade Outcome Analysis
For each trade since last /reflect:
- Did the DailyBrief signal predict the correct direction?
- Was regime classification correct at time of entry?
- Were there IC dissent notes that turned out correct?
- Was position sizing appropriate?
- Did the exit rule work or would a different exit have been better?

Compute:
- Win rate since last /reflect
- Average winner vs average loser
- Sharpe approximation if enough trades
- Best/worst strategy by P&L

### Step 3: Pattern Identification
- Strategies underperforming in their regime_fit → flag for matrix update
- Strategies outperforming outside their regime_fit → flag for matrix expansion
- ICs consistently right in specific regimes → increase trust
- ICs consistently wrong in specific regimes → add to Known Biases
- Common failure modes → add to `workshop_lessons.md`
- Missing workflow steps → candidates for skill edits

### Step 3.5: Production Monitor Findings

**AlphaMonitor Discord alerts (check first):**
- Review Discord for `[ALPHA CRITICAL]` or `[ALPHA WARNING]` alerts since last /reflect.
- CRITICAL = rolling_ic_30 < 0 for an IC → that IC's signals are losing money.
  Action: reduce conviction weight on that IC, flag for retraining in `agent_performance.md`.
- WARNING = IC decaying toward 0 → watch closely, one more declining session → treat as CRITICAL.

**DegradationDetector (for active strategies):**
- If `StrategyValidationFlow` ran on Saturday, check for any quarantined or degraded strategies.
  These appear as `[DEGRADATION]` Discord alerts or in the FastAPI `/dashboard/anomalies` endpoint.
- IS/OOS ratio > 4 → strategy is quarantined, do NOT send new signals from it.
- IS/OOS ratio 2–4 → reduce position sizes per detector recommendation.
- Log findings in `strategy_registry.md` under the affected strategy.

### Step 3.6: ML Model Health (if any models in ml_model_registry.md)
- Read `.claude/memory/ml_model_registry.md`
- For each trained model: has live performance degraded vs OOS baseline?
- If degradation > 20%: flag for retraining — log in `session_handoffs.md`
- Check SHAP feature importance for drift: are top features still the same as at training?
  If new features are dominating, the market structure has changed

### Step 4: Update Memory Files
- `trade_journal.md`: add weekly summary section
- `agent_performance.md`: update IC accuracy, Known Biases
- `workshop_lessons.md`: add new findings
- `regime_history.md`: update duration stats if enough transitions
- `strategy_registry.md`: update live stats for active strategies

### Step 5: Review and Edit Skills
For each skill in `.claude/skills/`:
1. Step I keep doing manually that isn't in the skill? → Add to skill
2. Step I always skip? → Remove from skill
3. Thresholds still correct? → Adjust with evidence
4. Missing memory file reads? → Add them

ONLY edit skills with evidence from 3+ sessions. Log EVERY change in
`session_handoffs.md`.

### Step 6: Review Regime-Strategy Matrix
If 2+ weeks of data contradict a matrix entry:
- Edit `CLAUDE.md` matrix
- Log in `session_handoffs.md`

### Step 7: Commit
Git add all changed files. Commit: `reflect: [date] — [summary]`

### Step 8: Output Summary
Present to the user:
- Key findings
- Memory files updated
- Skills edited (if any) and why
- Strategies flagged for action
- Recommended next session type
