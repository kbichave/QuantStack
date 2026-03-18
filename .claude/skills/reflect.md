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
- Read `.claude/memory/agent_performance.md` — current collector accuracy ratings
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
- Collectors consistently right in specific regimes → increase weight in synthesis
- Collectors consistently wrong in specific regimes → add to Known Biases
- Common failure modes → add to `workshop_lessons.md`
- Missing workflow steps → candidates for skill edits

### Step 3.5: Production Monitor Findings

**AlphaMonitor Discord alerts (check first):**
- Review Discord for `[ALPHA CRITICAL]` or `[ALPHA WARNING]` alerts since last /reflect.
- CRITICAL = rolling_ic_30 < 0 for a collector → that collector's signals are losing money.
  Action: reduce conviction weight for that collector, flag for code fix in `agent_performance.md`.
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
- **Causal feature drift check:** re-run `CausalFilter` on recent data (last 60 days)
  and compare surviving features to the set used at training time. If the surviving set
  has changed materially (>30% different features), the causal structure has shifted —
  this is stronger evidence of regime change than SHAP drift alone. Log in `agent_performance.md`.

### Step 3.7: Desk Agent Performance (if desk agents were used)

For each desk agent invoked since last /reflect:
- Did market-intel's regime classification match the realized outcome?
- Did alpha-research's signal quality assessment predict win/loss correctly?
- Did risk desk's position sizing prevent outsized losses?
- Did execution desk's algo recommendation reduce slippage?

Scoring per desk agent:
- **Correct**: desk output aligned with realized outcome → +1
- **Neutral**: desk output was inconclusive or not actionable → 0
- **Wrong**: desk output contradicted realized outcome → -1

Log accuracy per desk in `agent_performance.md` under a "Desk Agent Accuracy" section:
```
## Desk Agent Accuracy
| Desk Agent | Sessions | Correct | Neutral | Wrong | Accuracy |
|------------|----------|---------|---------|-------|----------|
| market-intel | ... | ... | ... | ... | ...% |
| alpha-research | ... | ... | ... | ... | ...% |
| risk | ... | ... | ... | ... | ...% |
| execution | ... | ... | ... | ... | ...% |
```

### Step 4: Update Memory Files
- `trade_journal.md`: add weekly summary section
- `agent_performance.md`: update collector accuracy, Known Biases
- `workshop_lessons.md`: add new findings
- `regime_history.md`: update duration stats if enough transitions
- `strategy_registry.md`: update live stats for active strategies

### Step 4.5: Signal Quality Check
For each SignalEngine collector in `.claude/memory/agent_performance.md`:
- Is rolling accuracy < 50% for 2+ consecutive sessions?
- Does Known Biases list have 3+ items?
- Did any collector output contribute to a trade loss or missed signal?

If YES for any collector: identify the source file in `packages/quant_pod/signal_engine/collectors/`:
- `technical.py` — trend, momentum, RSI, MACD, ADX
- `regime.py` — regime classification
- `volume.py` — OBV, VWAP deviation
- `risk.py` — VaR, drawdown, liquidity
- `sentiment.py` — news sentiment scoring
- `fundamentals.py` — P/E, ROE, FCF yield
- `events.py` — earnings calendar, FOMC dates

Fix the collector code directly. Signal quality issues are code bugs, not prompt problems.

Add to `session_handoffs.md`: "Collector [name] needs fix — evidence: [what you found], file: [path]"

### Step 5: Review and Edit Skills
For each skill in `.claude/skills/`:
1. Step I keep doing manually that isn't in the skill? → Add to skill
2. Step I always skip? → Remove from skill
3. Thresholds still correct? → Adjust with evidence
4. Missing memory file reads? → Add them

5. Collector source code still correct? → Fix in `packages/quant_pod/signal_engine/collectors/`
6. Desk agent prompts still relevant? → Update `.claude/agents/*.md` if systematic errors found

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
