# Integration Notes — Opus Review Feedback

## Integrating

### Critical Issue 1: Dual LLM Config Systems — INTEGRATING
**Verified:** `src/quantstack/llm_config.py` exists with IC/Pod/Assistant/Decoder/Workshop tiers, separate from `src/quantstack/llm/config.py` (heavy/medium/light). This is a real gap. Adding a prerequisite item 5.0 to consolidate these before LiteLLM deployment.

### Critical Issue 2: Compaction Failure Modes — INTEGRATING
Valid point about latency and failure on critical path. Adding: (a) deterministic compaction as v1 default, LLM-based as v2 optional, (b) explicit timeout and fallback code path, (c) cost/benefit analysis.

### Critical Issue 3: get_chat_model() Temperature Signature — INTEGRATING
Valid. Specifying the exact signature change and backward compatibility.

### Major Issue 4: Dataclasses vs Pydantic — INTEGRATING
Correct — `with_structured_output()` needs Pydantic BaseModel. Fixing schema definitions.

### Major Issue 5: EWF Cache Injection Mechanism — INTEGRATING
Valid ambiguity. Specifying module-level cache dict pattern (simplest for this codebase).

### Major Issue 6: Memory Decay Query Strategy — INTEGRATING
Valid. Specifying SQL-based weighting with over-fetch.

### Major Issue 7: Downstream Brief Consumption — INTEGRATING
Valid gap. Adding specification for how agents consume briefs.

### Major Issue 8: LiteLLM Tier Naming — INTEGRATING
Simplifying to use same tier names (`heavy`, `medium`, `light`) in LiteLLM.

### Minor Issue 12: CI Check for Hardcoded Strings — INTEGRATING
Adding pytest-based regression test.

### Suggestion 1: Deterministic Compaction First — INTEGRATING
Strong argument. Typed state → typed brief is extractable without LLM.

### Suggestion 4: Structured Output Failure Detection — INTEGRATING
Adding Pydantic ValidationError handling.

## NOT Integrating

### Minor Issue 10: Tier/Temperature Inconsistencies
The `hypothesis_critic` at medium tier with 0.7 temperature is intentional — it scores hypotheses and benefits from diverse scoring perspectives without needing heavy-tier reasoning. Will add a comment explaining the rationale.

### Minor Issue 11: Rollback Plan for 5.8
The plan already says "archive, not delete." Down-migration is an implementation detail for the migration file, not the plan.

### Minor Issue 13: Budget Race Condition
Budget tracking scope (per-executor-loop) is already clear enough. Tools that make their own LLM calls are out of scope for agent-level budgets.

### Suggestion 3: Gate on Baseline Data
Good practice but impractical in a spec-ordered implementation. Cost tracking (5.2) comes after compaction (5.1). We'll note that 5.2 should be validated before measuring impact of other items, but won't block implementation order.

### Minor Issue 9: SummarizationNode
Correct that we diverge — typed schemas require custom nodes, not generic summarization. Will add the rationale.
