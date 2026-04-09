# Opus Review

**Model:** claude-opus-4-6
**Generated:** 2026-04-07T21:30:00Z

---

## Critical Finding: Sections 2, 3, 4, and 5 Are Already Implemented

The plan's central claim is that Wires 2, 4, and 5b need to be "activated" or "completed." But reading the actual code tells a different story:

**Wire 2 (regime affinity sizing)** is already fully wired in `nodes.py` at lines 742-746. The code reads `regime_affinity_lookup`, floors at 0.1, and multiplies `signal_value`. Section 2 describes exactly what already exists.

**Wire 4 Layer 1 (skill confidence in executor)** is already wired at lines 718-730 and 748-752 of nodes.py. `SkillTracker.get_confidence_adjustment()` is called, results stored in `skill_adjustments`, and applied to `signal_value` in the candidate loop.

**Wire 5b (IC-driven weights)** is already wired in `synthesis.py` at lines 530-544. The IC weight lookup, fallback to static, and regime conditioning are all present.

**Wire 7 (auto-trigger apply_learning)** is already called at line 429 of `trade_hooks.py` inside `_on_trade_fill()`.

**The only remaining work is Section 1 (flag defaults) and Section 6 (TradeEvaluator producer).**

## Section 1: Flag Flip Risk — Simultaneous Activation

Compound multiplication concern: With Wire 2 (affinity) and Wire 4 (skill) both active, the effective minimum is 0.1 * 0.5 = 0.05, not 0.25 as stated. The existing code comment says "Enable one at a time after verifying data accumulation."

**Recommendation:** Enable one flag per week with specific metrics.

## Section 3: Wrong File, Wrong Architecture

The plan proposes modifying `agent_executor.py`, but Wire 4 is already in `nodes.py` risk_sizing node. agent_executor.py is a generic LLM tool-calling loop with no concept of conviction scores.

## Section 4: Collector-to-Agent Mapping Is Fragile

- Hardcoded coupling between signal engine and agent framework
- agent_id mismatch: SkillTracker data uses debate_verdict names, not clean names like "ml_scientist"
- Boundary unclear for semi-deterministic collectors (regime, sentiment)

**Recommendation:** Drop this section or fix the agent_id mismatch.

## Section 6a: Heuristic Scorer Issues

- execution_quality needs OHLC data not available in hook context
- thesis_accuracy conflates outcome with thesis quality (intentional debt, should be documented)

## Section 6b: Import-Based Feature Detection

Import success doesn't guarantee runtime success. Use explicit try/except around actual evaluation call, not the import. Module-level imports per CLAUDE.md.

## Section 7: Redundancy

`_on_trade_fill` already calls `apply_learning()`. Adding same call to `on_trade_close` means double DB round-trips per close.

## Missing Considerations

1. **No observability plan** — No metrics or alerts for loop health
2. **No cold-start strategy** — No monitoring for when loops start having effect
3. **Concurrency on apply_learning** — No row lock for concurrent trade closes
4. **No integration test** — Only unit tests; no end-to-end learning loop test
5. **Silent exception swallowing** — `except Exception: pass` in synthesis.py IC fallback violates CLAUDE.md principles

## Summary

Plan is well-structured with sound rollback strategy. However, ~60% of proposed work already exists. Revise to:
1. Acknowledge what's already wired, remove duplicate sections
2. Focus on Section 1 (staged flag rollout) and Section 6 (TradeEvaluator producer)
3. Add observability for all feedback loops
4. Fix agent_id mismatch between skill tracking producer and consumer
5. Add integration test for end-to-end learning loop
