# Section 2: Agent Output Schema Validation

## Problem

`agent_executor.py`'s `parse_json_response()` (line 645) returns `{}` or an empty fallback on parse failure. When `daily_plan` returns `{}`, the Trading Graph trades blind. When `fund_manager` returns `[]`, rejected entries are treated as approved. There is no retry mechanism, no schema validation against Pydantic models, and no structured fallback to conservative defaults.

The codebase already has significant schema infrastructure that is **not yet wired into the node functions**:

- `src/quantstack/graphs/schemas/trading.py` — Pydantic output schemas for 11 trading agents (e.g., `DailyPlanOutput`, `EntrySignalOutput`, `SafetyCheckOutput`)
- `src/quantstack/graphs/schemas/__init__.py` — `AGENT_OUTPUT_SCHEMAS` registry mapping agent names to schema classes, plus `AGENT_FALLBACKS` with conservative defaults
- `src/quantstack/graphs/trading/models.py` — Node-level output models with `safe_default()` classmethods for circuit breaker use
- `src/quantstack/graphs/agent_executor.py` line 726 — `parse_and_validate()` function exists but is **never imported or called** by any node function

All 30+ call sites across `trading/nodes.py`, `research/nodes.py`, and `supervisor/nodes.py` still call `parse_json_response()` directly, bypassing all schema validation.

## Goal

Wire the existing schema infrastructure into the agent execution pipeline so that every agent output is validated against its Pydantic schema, with retry-with-hint on first failure and conservative safe defaults on second failure. Add `with_structured_output()` support for providers that handle it natively (Anthropic/Bedrock).

## Dependencies

- None. This section is parallelizable with sections 01, 03, 04, and 06.

## Tests

```python
# tests/graphs/test_agent_output_schemas.py

# --- Schema structure tests ---
# Test: DailyPlanOutput validates correct JSON structure
# Test: EntrySignalOutput validates correct JSON structure
# Test: PositionReviewOutput validates correct JSON structure
# Test: FundManagerVerdict (EntrySignalOutput) validates correct JSON structure
# Test: RiskSizingOutput validates correct JSON structure
# Test: SafetyCheckOutput defaults to halted=True (fail closed)

# --- Retry and fallback tests ---
# Test: parse failure on first attempt retries with schema hint
# Test: parse failure on second attempt returns safe default (not {})
# Test: safe default for fund_manager is empty signals list (no entries approved)
# Test: safe default for entry_scan is empty list (no entries)
# Test: safe default for daily_plan is conservative plan with no candidates
# Test: safe default for safety_check is halted=True (fail closed)
# Test: raw LLM output logged as warning on fallback to safe default

# --- Provider integration tests ---
# Test: with_structured_output works with json_schema method (Anthropic/Bedrock)
# Test: with_structured_output fallback to json_mode works (Groq)
# Test: provider detection correctly routes to json_schema vs json_mode path
```

## Implementation Details

### Step 1: Wire `parse_and_validate()` into node functions

The function `parse_and_validate()` at `agent_executor.py:726` already exists and does JSON parsing + Pydantic validation. It accepts an `output_schema` parameter. Currently, every node function calls `parse_json_response()` directly instead.

**What to change:** In each node function that calls `parse_json_response()`, replace the call with `parse_and_validate()`, passing the appropriate schema from `AGENT_OUTPUT_SCHEMAS` and the fallback from `AGENT_FALLBACKS`.

**Files to modify:**

- `src/quantstack/graphs/trading/nodes.py` — ~12 call sites. Each currently looks like:
  ```python
  parsed = parse_json_response(text, {})
  ```
  Replace with:
  ```python
  from quantstack.graphs.schemas import AGENT_OUTPUT_SCHEMAS, AGENT_FALLBACKS
  parsed, _ = parse_and_validate(
      text,
      fallback=AGENT_FALLBACKS.get("daily_planner", {}),
      output_schema=AGENT_OUTPUT_SCHEMAS.get("daily_planner"),
      agent_name="daily_planner",
      graph_name="trading",
  )
  ```

- `src/quantstack/graphs/research/nodes.py` — ~11 call sites. Same pattern.
- `src/quantstack/graphs/supervisor/nodes.py` — ~8 call sites. Same pattern.

**Import change for all three files:** Add `parse_and_validate` to the existing import from `quantstack.graphs.agent_executor`, and add the `AGENT_OUTPUT_SCHEMAS`/`AGENT_FALLBACKS` import from `quantstack.graphs.schemas`.

### Step 2: Add retry-with-hint to `parse_and_validate()`

Currently `parse_and_validate()` returns the fallback immediately on validation failure. It should instead support a retry mechanism.

**What to change in `parse_and_validate()` at `agent_executor.py:726`:**

Add an optional `retry_fn` callback parameter. When validation fails on the first attempt:

1. Build a schema hint string from the Pydantic model's `model_json_schema()` method
2. Call `retry_fn(hint)` which re-invokes the LLM with the schema appended to the prompt
3. Parse and validate the retry response
4. On second failure: log a WARNING with the raw LLM output, write to DLQ, return the conservative fallback

The `retry_fn` is provided by the calling node function, which has access to the LLM and conversation context. Signature:

```python
retry_fn: Callable[[str], Awaitable[str]] | None = None
```

When `retry_fn` is None (default), behavior is unchanged from current -- fail to fallback immediately. This preserves backward compatibility for callers that don't pass it.

The schema hint string should be concise:

```python
hint = (
    f"Your previous response could not be parsed. "
    f"Respond with valid JSON matching this schema:\n"
    f"{json.dumps(output_schema.model_json_schema(), indent=2)}"
)
```

### Step 3: Add `with_structured_output()` support in agent executor

For Anthropic/Bedrock providers, LangChain's `with_structured_output(schema, method="json_schema")` instructs the model to produce structured output natively. This is more reliable than post-hoc parsing.

**What to change in `run_agent()` at `agent_executor.py:298`:**

Before the main agent loop, check if the agent has a registered output schema and the provider supports structured output:

```python
output_schema = AGENT_OUTPUT_SCHEMAS.get(config.name)
if output_schema and provider in ("anthropic", "bedrock"):
    structured_llm = llm.with_structured_output(output_schema, method="json_schema")
```

For Groq/Llama models, use `method="json_mode"` with post-hoc Pydantic validation as fallback:

```python
elif output_schema and provider == "groq":
    structured_llm = llm.with_structured_output(output_schema, method="json_mode")
```

The structured LLM is used only for the final response extraction, not during tool-calling rounds. Tool-calling rounds continue to use the base LLM.

**File:** `src/quantstack/graphs/agent_executor.py`

### Step 4: Verify and complete the schema registry

The `AGENT_OUTPUT_SCHEMAS` registry in `schemas/__init__.py` already maps all 21 agents. Verify that:

1. Every agent name in `graphs/*/config/agents.yaml` has a corresponding entry in `AGENT_OUTPUT_SCHEMAS`
2. Every entry in `AGENT_FALLBACKS` is conservative (fail closed for safety-critical agents)
3. The `SafetyCheckOutput` schema defaults to `halted=True` (already verified in the code)

**Critical safe default behaviors to verify/enforce:**

| Agent Node | Schema Class | Safe Default Behavior |
|-----------|-------------|----------------------|
| `daily_planner` | `DailyPlanOutput` | Conservative plan, no candidates |
| `fund_manager` | `EntrySignalOutput` | Empty signals list (no entries approved) |
| `position_monitor` | `PositionReviewOutput` | Empty analyses (no exits triggered) |
| `safety_check` | `SafetyCheckOutput` | `halted=True` (FAIL CLOSED) |
| `executor` | `ExecutionOrderOutput` | Empty orders (no trades executed) |

These are already defined in `AGENT_FALLBACKS`. The key invariant is: **on parse failure, no agent should default to a permissive state** (approving entries, allowing trading, executing orders).

**File:** `src/quantstack/graphs/schemas/__init__.py` — verify completeness, no changes expected unless gaps found.

### Step 5: Groq compatibility benchmark (deferred verification)

The plan calls for benchmarking Groq Llama 3.3 70B structured output across all 5 critical schemas with 100 sample prompts. If error rate > 5%, keep those agents on Haiku.

This is a verification step, not a code change. Create a benchmark script:

**New file:** `scripts/benchmark_groq_schemas.py`

The script should:
1. Load each of the 5 critical schemas
2. Send 100 sample prompts to Groq Llama 3.3 70B with `json_mode`
3. Attempt Pydantic validation on each response
4. Report error rate per schema
5. If any schema exceeds 5% error rate, output a recommendation to keep that agent on Haiku

This script is run manually as a one-time validation, not wired into CI.

## Files Summary

| File | Action | What Changes |
|------|--------|-------------|
| `src/quantstack/graphs/agent_executor.py` | Modify | Add retry-with-hint to `parse_and_validate()`. Add `with_structured_output()` routing in `run_agent()`. |
| `src/quantstack/graphs/trading/nodes.py` | Modify | Replace ~12 `parse_json_response()` calls with `parse_and_validate()` + schema + fallback. |
| `src/quantstack/graphs/research/nodes.py` | Modify | Replace ~11 `parse_json_response()` calls with `parse_and_validate()`. |
| `src/quantstack/graphs/supervisor/nodes.py` | Modify | Replace ~8 `parse_json_response()` calls with `parse_and_validate()`. |
| `src/quantstack/graphs/schemas/__init__.py` | Verify | Confirm all 21 agents mapped, fallbacks conservative. No changes expected. |
| `src/quantstack/graphs/schemas/trading.py` | Verify | Confirm schema fields match what nodes actually produce. Adjust if needed. |
| `src/quantstack/tools/models.py` | No change | Existing tool I/O models are separate from agent output schemas. |
| `scripts/benchmark_groq_schemas.py` | Create | One-time Groq structured output benchmark script. |

## Verification Checklist

After implementation, confirm:

1. No node function in trading/research/supervisor `nodes.py` calls `parse_json_response()` directly (all go through `parse_and_validate()`)
2. A deliberately malformed LLM response triggers retry-with-hint, then falls back to conservative default
3. The `safety_check` agent returns `halted=True` on any parse failure (never fails open)
4. The `fund_manager` agent returns empty signals on any parse failure (never approves by default)
5. Langfuse traces show `agent_dlq` entries for parse failures with raw LLM output preserved
6. Anthropic/Bedrock agents use `with_structured_output(method="json_schema")` 
7. Groq agents use `with_structured_output(method="json_mode")` with post-hoc validation
