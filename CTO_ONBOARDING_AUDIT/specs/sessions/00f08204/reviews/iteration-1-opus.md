# Opus Review

**Model:** claude-opus-4-6
**Generated:** 2026-04-06

---

## Overall Assessment

The plan is well-structured, thorough, and demonstrates good understanding of the codebase. However, there are several gaps, a significant blind spot regarding a parallel LLM config system, and some architectural decisions that need more scrutiny. The plan is implementable but would benefit from addressing the issues below before an engineer starts work.

---

## Critical Issues

### Issue 1: Two Parallel LLM Config Systems -- Plan Only Addresses One

- **Severity:** Critical
- **Location:** Sections 4 (5.3), 6 (5.4), 7 (5.6), 8 (5.7)
- **Issue:** The plan focuses on `src/quantstack/llm/provider.py` and `src/quantstack/llm/config.py` as the LLM routing layer. However, the codebase has a **second, independent LLM config system** at `src/quantstack/llm_config.py` (note: no `llm/` subdirectory). This file defines its own tier system (IC, Pod, Assistant, Decoder, Workshop) with its own provider fallback chain, its own env var overrides (`LLM_MODEL_IC`, `LLM_MODEL_POD`, etc.), and its own model string resolution. The plan never mentions this file. Hardcoded strings like `groq/llama-3.3-70b-versatile` in `hypothesis_agent.py` and `sentiment.py` reference this system. If 5.6 only replaces strings that should use `get_chat_model()` from `llm/provider.py`, but callers actually use the `llm_config.py` tier system, the replacements will break.
- **Recommendation:** Before implementation, audit which files import from `llm/provider.py` vs `llm_config.py`. The plan needs an explicit strategy: either consolidate the two systems first (recommended), or address both in each item. This is a prerequisite blocker for items 5.3, 5.6, and 5.7.

### Issue 2: Compaction Node Adds Latency and a Failure Point on the Critical Trading Path

- **Severity:** Critical
- **Location:** Section 2 (5.1)
- **Issue:** Adding an LLM call (even Haiku) to the critical path between merge and execution introduces latency (200-800ms per compaction call, 2 calls per cycle) and a new failure mode. The plan's fallback (Section 2.7) says "fall back to passing raw state" on parse failure, but doesn't address: (a) what happens if the LLM call itself times out or errors (network failure, 429, etc.), (b) whether the fallback path is tested, (c) whether downstream agents can actually handle both brief format AND raw format (they'd need conditional logic). The plan also doesn't quantify the cost of the compaction calls themselves -- two Haiku calls per cycle at ~65-120KB input could cost $0.01-0.02/cycle, which at 6-12 cycles/hour adds up.
- **Recommendation:** (a) Add an explicit timeout for compaction LLM calls (e.g., 5 seconds). (b) Define the fallback as a code path, not just prose -- downstream agents should always read from brief keys, with the fallback populating those keys from raw data via a deterministic extractor (no LLM). (c) Add cost/benefit analysis of the compaction calls themselves. (d) Consider whether a deterministic compaction (Python code that extracts structured fields from typed state) would achieve 80% of the benefit without any LLM call.

### Issue 3: `get_chat_model()` Currently Has No Temperature Parameter

- **Severity:** Critical
- **Location:** Section 5 (5.4)
- **Issue:** The plan says "Pass the temperature through to `get_chat_model()` which passes it to the LLM constructor." But `get_chat_model(tier, thinking)` currently accepts only `tier` and `thinking`. The `ModelConfig` dataclass does have a `temperature` field (defaulting to 0.0), but it is never set from a parameter -- it always uses the default. The plan describes the wiring conceptually but doesn't specify the actual signature change to `get_chat_model()` needed, nor does it address backward compatibility for the ~30+ call sites that currently call `get_chat_model(tier)` without a temperature argument.
- **Recommendation:** Explicitly define the new signature: `get_chat_model(tier: str, thinking: dict | None = None, temperature: float | None = None)`. State that `temperature=None` means "use ModelConfig default (0.0)" for backward compatibility. List the files that call `get_chat_model()` and confirm none will break.

---

## Major Issues

### Issue 4: Brief Schemas Use Dataclasses, Not Pydantic

- **Severity:** Major
- **Location:** Section 2.3
- **Issue:** The schemas are defined as `@dataclass` but the plan repeatedly references "Pydantic models," "Pydantic-typed brief," and "with_structured_output() to force Pydantic compliance." LangChain's `with_structured_output()` requires Pydantic `BaseModel` subclasses, not dataclasses. The code as written won't work with the described approach.
- **Recommendation:** Change all brief schemas from `@dataclass` to Pydantic `BaseModel` subclasses. Use `Field()` for validation constraints (e.g., `signal_strength: float = Field(ge=0.0, le=1.0)`).

### Issue 5: EWF Cache Injected into Tool via Unclear "Tool Context" Mechanism

- **Severity:** Major
- **Location:** Section 6.5
- **Issue:** The plan says `get_ewf_analysis` should "Accept an optional `ewf_cache` parameter (injected from graph state via tool context)" but doesn't explain how graph state gets injected into a `@tool`-decorated function. LangChain tools don't have automatic access to graph state. The plan needs to specify the injection mechanism: either (a) use `InjectedState` annotation from LangGraph, (b) modify the tool to accept the cache explicitly and have the agent executor pass it, or (c) use a closure/factory pattern. Each has different implications for the tool registry and agent executor.
- **Recommendation:** Specify the exact injection mechanism. The cleanest approach for this codebase (given `TOOL_REGISTRY` in `tools/registry.py`) is likely to use LangGraph's `InjectedState` or to have the `data_refresh` node write to a module-level cache dict that the tool reads. Document which approach and why.

### Issue 6: Memory Temporal Decay Math Applied at Wrong Layer

- **Severity:** Major
- **Location:** Section 9.2
- **Issue:** The plan says to apply `weight = 0.5^(age_days / half_life_days)` at retrieval time in `read_recent()` and `read_as_context()`. But looking at `blackboard.py`, `read_recent()` does a SQL `ORDER BY created_at DESC LIMIT ?` query. The decay weighting would need to happen after fetching rows, which means you'd need to over-fetch (fetch more than `limit` rows) and then re-sort by weighted score, then truncate to `limit`. The plan doesn't address this over-fetch requirement, and applying it naively would either (a) still return only the N most recent rows (defeating the purpose), or (b) require fetching all rows and sorting in Python (performance issue at scale).
- **Recommendation:** Specify the query strategy: either compute the weight in SQL (`ORDER BY POW(0.5, age_days / half_life) * some_relevance_score DESC`) or define an explicit over-fetch ratio (e.g., fetch 3x limit, apply weighting in Python, return top limit). State the performance implications.

### Issue 7: No Specification for How Downstream Agents Consume Briefs

- **Severity:** Major
- **Location:** Section 2.4, 2.5
- **Issue:** The plan says "Downstream nodes read the brief key instead of raw upstream keys" but doesn't specify how. The `execute_entries` agent currently has a system prompt and tool set that presumably references raw state. The plan doesn't mention: (a) which agent prompts need updating to reference brief fields instead of raw state, (b) how the brief is injected into the agent's context (system message? tool result? state key that the executor auto-includes?), (c) what happens to the `reflect` agent that needs both the brief AND the outcomes.
- **Recommendation:** Audit the system prompts for `risk_sizing`, `execute_entries`, `portfolio_review`, `analyze_options`, and `reflect` agents. Document which state keys each currently reads and how the brief replaces them. Specify the injection mechanism in `agent_executor.py`.

### Issue 8: LiteLLM Config Uses Namespace Collision with Existing Tier Names

- **Severity:** Major
- **Location:** Section 8.4
- **Issue:** The plan defines LiteLLM model names as `quantstack-heavy`, `quantstack-medium`, `quantstack-light`. But the existing `get_chat_model()` uses tiers `heavy`, `medium`, `light`. The plan's section 8.5 says to "map tier names to LiteLLM logical names" but doesn't specify where this mapping lives or how the existing agent configs (which use `heavy`/`medium`/`light`) translate. A renaming layer without clear ownership will confuse debugging.
- **Recommendation:** Either keep the same names in LiteLLM (just `heavy`, `medium`, `light`) to avoid a translation layer, or explicitly define the mapping function and its location. Simpler is better here.

---

## Minor Issues

### Issue 9: Spec Mentions `langmem.short_term.SummarizationNode` but Plan Doesn't Use It

- **Severity:** Minor
- **Location:** Section 2.4
- **Issue:** The spec and research document both recommend using `langmem`'s `SummarizationNode` pattern. The plan instead implements custom compaction nodes. The research found that `SummarizationNode` is a proven pattern with built-in token counting triggers.
- **Recommendation:** Either use `SummarizationNode` (if it fits) or add a sentence explaining why custom implementation is preferred (likely: merge points need typed schemas, not just summaries).

### Issue 10: `agents.yaml` Tier Assignments Inconsistent Between Plan and Research

- **Severity:** Minor
- **Location:** Section 4.2
- **Issue:** The plan lists `execution_researcher` under Heavy tier. The research document lists it under Light tier. The plan's temperature table (5.4) gives `hypothesis_critic` 0.7, which is unusual for a Medium-tier agent doing "structured extraction."
- **Recommendation:** Reconcile the tier assignments. If `hypothesis_critic` needs 0.7 temperature for ideation diversity, it may actually need Heavy tier.

### Issue 11: Rollback Plan for 5.8 is Destructive

- **Severity:** Minor
- **Location:** Section 10.3
- **Issue:** "Drop archive table columns, disable pruning job" is destructive DDL without a down-migration. Archived rows wouldn't be restored.
- **Recommendation:** Write an explicit down-migration that moves archived rows back and drops the new artifacts.

### Issue 12: Missing CI/CD Integration for 5.6

- **Severity:** Minor
- **Location:** Section 7.3
- **Issue:** Validation is a manual grep. "grep-based CI check" mentioned in 10.4 is unspecified.
- **Recommendation:** Add a concrete pytest test that greps for hardcoded model strings and fails if any are found.

### Issue 13: Budget Enforcement Race Condition

- **Severity:** Minor
- **Location:** Section 3.5
- **Issue:** Budget tracking per-agent-executor-loop doesn't account for nested LLM calls from tools.
- **Recommendation:** Clarify scope: per-agent-executor-loop only, or inclusive of tool LLM calls.

---

## Suggestions

### Suggestion 1: Consider Deterministic Compaction First

Implement deterministic compaction (Python code that extracts structured fields from typed state) as v1. Add LLM-based compaction as v2 only if v1 doesn't achieve the 40% reduction target.

### Suggestion 2: Consolidate `llm_config.py` and `llm/config.py` as Item 5.0

Add a prerequisite item to consolidate the two LLM config systems before starting optimization work.

### Suggestion 3: Add a Cost Dashboard Before Optimization

Gate items 5.3-5.8 on having at least 3 days of baseline cost data from 5.2.

### Suggestion 4: `with_structured_output()` Failure Modes

Use Pydantic `ValidationError` as the detection mechanism. Wrap structured output in try/except and fall back. Log malformed responses.

---

## Summary

| Severity | Count |
|----------|-------|
| Critical | 3 |
| Major | 5 |
| Minor | 5 |
| Suggestion | 4 |
