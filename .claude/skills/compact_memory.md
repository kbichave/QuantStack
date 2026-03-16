---
name: compact_memory
description: Compact memory files to remove stale/redundant entries and prevent context window bloat. Run after 5+ sessions or when a memory file exceeds 200 lines.
user_invocable: true
---

# /compact-memory — Memory Compaction Session

## Purpose

Distill `.claude/memory/*.md` files to keep only actionable findings.
Remove noise, redundant iterations, and resolved/stale context.
This is a destructive operation — commit before and after.

## When to Run

- Any memory file exceeds 200 lines
- After 5+ sessions without compaction
- Before a long workshop session (clear space in context window)
- Run `/reflect` first if there are unprocessed session findings

---

## Step 0: Commit Current State

```bash
git add .claude/memory/
git commit -m "memory: snapshot before compaction $(date +%Y-%m-%d)"
```

This is your rollback point.

---

## Step 1: Compact `workshop_lessons.md`

**Keep:**
- Final verdict per strategy (one row per hypothesis in Failed Hypotheses table)
- The single most important finding per session (one sentence)
- Infrastructure workarounds that are still relevant (e.g. screener broken)
- "What Works in Each Regime" table — always keep, update entries

**Remove:**
- Detailed per-variant comparison tables (already summarized in the verdict)
- Repeated mentions of the same finding across multiple sessions
- Rationale for discarded approaches once the finding is captured in one place
- Sessions > 6 months old where the infrastructure has changed

**Target: ≤ 100 lines**

---

## Step 2: Compact `trade_journal.md`

**Keep:**
- Last 30 trades (most recent by date)
- Any trade with an unusual outcome worth remembering (large loss, surprise fill, regime detection failure)
- One-sentence summary of resolved patterns ("RSI overshoot on financials during rate hikes — confirmed 3×")

**Remove:**
- Trades older than 90 days with no unusual characteristics
- Duplicate "hold" decisions with identical reasoning
- Any session where `get_system_status()` was the only action

**Target: ≤ 150 lines**

---

## Step 3: Compact `regime_history.md`

**Keep:**
- Regime transitions (date, from, to, trigger) — these are the signal
- Periods where regime and strategy diverged — learning signal
- Current regime at end of last session

**Remove:**
- Consecutive sessions with identical regime (compress to "regime stable for N days")
- Verbose descriptions — keep the table, drop the prose

**Target: ≤ 80 lines**

---

## Step 4: Compact `agent_performance.md`

**Keep:**
- IC accuracy table (one row per IC, latest rolling window only)
- Known Biases list — maximum 5 items per IC (the top 5 by evidence count)
- Flags for `/tune` sessions with supporting evidence

**Remove:**
- Historical IC accuracy from more than 2 months ago (superseded by rolling window)
- Bias entries that have been addressed in a `/tune` session
- Observations without evidence count (speculation, not pattern)

**Target: ≤ 100 lines**

---

## Step 5: Compact `session_handoffs.md`

**Keep:**
- Pending actions (anything not yet done)
- Modified files log for the last 30 days
- Cross-session context that the next session MUST know

**Remove:**
- Completed actions (mark as `✓` then remove on the next compaction)
- Modified files log older than 30 days
- Context that is now visible in the code (e.g. "added X to Y file" — just read the file)

**Target: ≤ 80 lines**

---

## Step 6: `strategy_registry.md` — No Compaction

Do NOT compact the strategy registry. Every strategy (including failed ones) must be
preserved for audit purposes. The registry grows linearly and that's intentional.

If a strategy is `failed`, it should stay in the registry as `status: failed` with the
failure reason — this prevents re-testing the same hypothesis.

---

## Step 7: `ml_model_registry.md`

**Keep:**
- One row per model (model type, features, OOS accuracy, last validated date)
- Status: active / stale / retired

**Remove:**
- Training run details (keep the final metrics only)
- Deprecated models (status: retired > 3 months ago with no active successors)

---

## Step 8: Commit Compacted State

```bash
git add .claude/memory/
git commit -m "memory: compact $(date +%Y-%m-%d) — [N lines removed across N files]"
```

Log in `session_handoffs.md`:
```
[DATE] compact-memory: removed N lines from workshop_lessons.md, N from trade_journal.md,
N from regime_history.md. Rollback: <prev commit hash>
```

---

## What NOT to Remove

- Any finding that is referenced in `CLAUDE.md` (e.g., regime-strategy matrix entries)
- Any finding that motivated a code change (link to commit if possible)
- Failure records in the strategy registry (keep all failures)
- The "Failed Hypotheses" table in `workshop_lessons.md` — this is the institutional memory that prevents re-running the same dead ends

---

## Size Targets Summary

| File | Max Lines | Notes |
|------|-----------|-------|
| `workshop_lessons.md` | 100 | Compress per-session detail, keep verdicts |
| `trade_journal.md` | 150 | Rolling 90-day window |
| `regime_history.md` | 80 | Compress stable periods |
| `agent_performance.md` | 100 | Latest window only |
| `session_handoffs.md` | 80 | Pending actions + last 30d |
| `strategy_registry.md` | unlimited | Never compact |
| `ml_model_registry.md` | 60 | One row per model |
