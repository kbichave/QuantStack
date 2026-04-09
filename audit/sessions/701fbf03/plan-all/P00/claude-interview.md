# P00 Interview: Wire Learning Modules

**Date:** 2026-04-07
**Context:** Research revealed 4/6 wires already operational; questions focused on remaining decisions.

---

## Q1: Feature Flag Activation Strategy

**Question:** Wire 2/4/5b are implemented but behind feature flags (default: off). What's your activation strategy for these feedback loops?

**Answer:** Enable all at once. System is paper-only, risk is bounded. No need for sequential rollout.

**Implication:** Remove the flag-gating from the code path — make these loops always-on. The flags can remain as a kill switch but should default to `true`.

---

## Q2: Wire 4 Consumer Location (SkillTracker → Conviction)

**Question:** Where should conviction adjustment be applied?

**Answer:** Both layers — apply at agent executor output AND signal synthesis. Double-dampening for consistently wrong agents.

**Implication:** Two integration points needed:
1. `agent_executor.py` — scale individual agent conviction output by `get_confidence_adjustment(agent_id)`
2. `synthesis.py` — apply agent-level confidence as an additional weight factor during aggregation

---

## Q3: OutcomeTracker Auto-Trigger

**Question:** When should `apply_learning()` run?

**Answer:** After every trade close. Most responsive — affinity updates within the same session.

**Implication:** Add `outcome_tracker.apply_learning(strategy_id)` call inside `on_trade_close` hook, after `record_exit()`. Updates regime_affinity in real-time.

---

## Q4: TradeEvaluator Dependency Strategy

**Question:** Should `openevals` be mandatory or optional?

**Answer:** Optional with heuristic fallback. Try openevals, fall back to rule-based scoring if unavailable. No hard dependency.

**Implication:** Build a `score_trade_heuristic()` function that produces TradeQualityScore from P&L, timing, and sizing data without LLM. Use as fallback when openevals import fails.
