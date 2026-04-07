# Opus Review

**Model:** claude-opus-4-6
**Generated:** 2026-04-06T00:00:00Z

---

# Implementation Plan Review: Phase 10 — Advanced Research

## Executive Summary

The plan is well-structured, covers all spec requirements, and demonstrates strong understanding of the existing codebase. However, it has a critical incompatibility with the existing state model, underestimates several integration risks, and is missing operational detail in areas that will trip up an implementer. Below are specific findings, ordered by severity.

---

## 1. Completeness

**Coverage is strong.** All 8 AR items from the spec are addressed (AR-1 through AR-10, skipping unused numbers), plus the AutoResearchClaw upgrades, event bus extensions, and database migrations. The cross-cutting concerns (prompt caching, testing) are covered.

**Gap: Prompt caching is missing from the plan.** The spec (Cross-Cutting Concerns section) explicitly calls out enabling Anthropic prompt caching on all API calls for 70-90% input token savings. The plan mentions it in passing in Section 10 ("~4K cached system prompt") but never dedicates a section to implementing it. This is a prerequisite for hitting the $10/day target — without it, the Haiku strategy agents at 78 cycles/day will cost significantly more than projected. The plan needs a concrete section covering: where caching is enabled (the LLM provider layer in `src/quantstack/llm/provider.py`), what gets cached (system prompts, tool definitions), and the expected cache hit rate.

**Gap: The spec's Validation Plan (bottom of spec) defines specific acceptance gates per sub-phase.** The plan's testing section (Integration & Testing Strategy) covers unit/integration/regression but does not map to the spec's validation milestones. For example, the spec says "10A (week 3): Active vs. stub tool count (target: 50+ active)" — there is no mechanism in the plan to measure this.

---

## 2. Technical Soundness

### Critical: `extra="forbid"` on State Models (Section 3)

The plan proposes adding `token_budget_remaining`, `cost_budget_remaining`, `tokens_consumed`, and `cost_consumed` fields to `ResearchState` and `TradingState`. Both models use `ConfigDict(extra="forbid")`. This means **any node that returns a dict with a key not in the model definition will raise a ValidationError at runtime**. The plan acknowledges these are new fields, which is correct — but it doesn't mention that every existing node's return dict must be verified to NOT accidentally include these new field names, and more importantly, that the graph checkpoint serialization may need updating if the checkpointer has stored schema assumptions.

The real problem: adding 4 fields to ResearchState and 4 to TradingState means every LangGraph checkpoint for in-flight runs will be incompatible after deployment. **The plan needs a migration strategy for in-flight graph runs** — likely a clean restart of all graph services with empty checkpoint state.

### Concern: Knowledge Graph Embedding Cost (Section 7)

The plan calls for `text-embedding-3-small` embeddings (1536 dimensions) for every KG node. The plan routes this through "existing LLM routing" — the current LLM routing is configured for Bedrock primary with Anthropic models. **OpenAI's embedding API is a different provider entirely.** The plan needs to specify: is this Amazon Titan embeddings via Bedrock? OpenAI directly? A local model? This is an unresolved dependency that blocks KG implementation.

### Concern: DSPy Integration (Section 9)

The meta_prompt_optimizer proposes using DSPy's MIPROv2. This is a significant new dependency. DSPy requires its own LLM configuration, has its own execution model (compiling programs, not just generating text), and MIPROv2 specifically needs a "teacher" model (typically more expensive). The plan says "~$2 per optimization run" but doesn't specify which model DSPy will use as its teacher, how DSPy integrates with the existing Bedrock-based LLM routing, or whether DSPy's execution model is compatible with the Docker sandbox. This needs a spike / proof-of-concept before committing to the timeline.

---

## 3. Risk Assessment

### Unaddressed Failure Mode: Overnight Runner Crash Recovery

Section 4 describes an 8-hour overnight loop. If the process crashes at hour 4, what happens? The plan says experiments log to `autoresearch_experiments`, so completed work is preserved. But there is no mention of: (a) how the runner detects it was interrupted and resumes, (b) whether the $10 budget tracker persists across restarts, (c) whether the morning validator at 04:00 runs regardless of whether the overnight loop completed.

**Suggestion:** Budget tracking should be DB-persisted. The morning validator should run unconditionally.

### Unaddressed Failure Mode: Mandate Staleness (Section 10)

The CIO agent produces a mandate at 09:00 ET. What if the CIO agent fails? Strategy agents would either operate with yesterday's stale mandate or be blocked entirely.

**Suggestion:** Define a fallback: if no mandate exists for today by 09:30, use a conservative default mandate (max_new_positions=0, all existing positions in "monitor" mode).

### Unaddressed Failure Mode: Feature Factory Runaway

Section 5 describes programmatic enumeration producing 500+ candidates from cross-interactions of all feature pairs. If there are N base features, cross-interactions produce O(N^2) candidates. With 30 base features, that's potentially 20,000+ candidates before Haiku-generated ones.

**Suggestion:** Add a hard cap on enumerated candidates (e.g., 2000).

---

## 4. Dependency Issues

### Hidden Coupling: AR-9 Budget and AR-1 Overnight Runner

Section 3 adds per-cycle budget ($0.50/cycle). Section 4 has overnight budget ($10/night). The plan doesn't clarify how these interact.

**Suggestion:** Explicitly state the overnight runner overrides per-cycle budgets with per-experiment budget.

### Missing Dependency: pgvector Extension

Section 7 mentions `CREATE EXTENSION IF NOT EXISTS vector`. This requires pgvector to be installed. Standard `postgres:16` Docker image does NOT include it.

**Suggestion:** Specify pgvector-enabled PostgreSQL image in docker-compose.yml.

---

## 5. Scalability Concerns

### 96 Experiments/Night Clarification

The 5-minute budget per experiment: is it a timeout or a sleep interval? If timeout, throughput could be hundreds/night. If sleep, you waste 98% of compute.

**Suggestion:** Clarify as timeout, run experiments back-to-back.

---

## 6. Missing Details

- No error handling strategy for LLM failures in new pipelines
- No monitoring/alerting for 5+ new scheduled processes
- No rollback plan/feature flags for sub-phase capabilities
- Thresholds currently hardcoded — extraction is a prerequisite refactor

---

## 7. Ordering Issues

- Within 10D: governance (AR-4) should precede meta agents (AR-2) since meta agents need to target the NEW agent hierarchy
- KG schema creation could be pulled into 10B so features are recorded as nodes from day one

---

## 8. Over/Under Engineering

- Consensus subgraph (Section 8): ship with feature flag for disable
- Event schema validation: inline Pydantic models simpler than separate file
- Loss analyzer: define deterministic rules exhaustively, Haiku only for "unknown" bucket
- Prioritization formula cold-start problem: acknowledge explicitly

---

## Summary of Actionable Recommendations

| Priority | Issue | Section | Action |
|----------|-------|---------|--------|
| **P0** | `extra="forbid"` breaks with new state fields | 3 | Add migration note: clean restart required |
| **P0** | Embedding provider unspecified | 7 | Decide: Bedrock Titan, OpenAI, or local |
| **P0** | pgvector not in standard Docker image | 7, 13 | Specify pgvector-enabled PostgreSQL image |
| **P1** | Overnight runner crash recovery | 4 | Persist budget to DB, unconditional morning validator |
| **P1** | Mandate failure fallback missing | 10 | Conservative default mandate on CIO failure |
| **P1** | AR-9 / AR-1 budget interaction undefined | 3, 4 | Explicitly scope per-cycle vs. per-night budgets |
| **P1** | Prompt caching section missing | All | Add dedicated section |
| **P1** | DSPy integration risk | 9 | Spike/PoC before committing |
| **P2** | Feature enumeration could produce 20K+ | 5 | Add hard cap |
| **P2** | 5-minute budget semantics unclear | 4 | Clarify as timeout |
| **P2** | Governance before meta agents in 10D | 9, 10 | Reorder |
| **P2** | Validation milestones not mapped | Testing | Add acceptance gates |
| **P3** | Consensus needs feature flag | 8 | Add env var toggle |
| **P3** | Threshold extraction prerequisite | 9 | Add refactor task |
