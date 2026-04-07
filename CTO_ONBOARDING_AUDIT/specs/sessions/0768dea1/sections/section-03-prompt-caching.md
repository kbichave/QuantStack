# Section 03: Prompt Caching with Explicit Breakpoints (Item 0.2)

## Overview

QuantStack runs 21 LLM-backed agents cycling every 5-10 minutes across three LangGraph StateGraphs. Every LLM call currently pays full input token price for system prompts and tool definitions because there are zero `cache_control` references in the codebase. At approximately $126/day, this is the single largest cost waste.

Anthropic's prompt caching reduces cached input token cost by 90% (cache reads cost 0.1x base price). The cache key is computed as a hash of the full request prefix up to each breakpoint, following the hierarchy `tools -> system -> messages`. By adding `cache_control: {"type": "ephemeral"}` breakpoints at the right positions, second-and-subsequent calls with identical tool definitions and system prompts will hit cache instead of re-ingesting the full prefix.

This section depends on **section-02-tool-ordering** being complete. Without deterministic tool ordering, the cache key changes on every process restart, defeating the purpose of caching.

## Goal

- Add `cache_control` breakpoints on the last tool definition and on system messages
- Make breakpoints provider-aware (Anthropic/Bedrock only; skip for OpenAI/Ollama)
- Add cache hit rate observability via Langfuse
- Target: 50%+ reduction in prompt token cost, 80%+ cache hit rate for stable agents

## Dependencies

| Dependency | Why |
|------------|-----|
| section-02-tool-ordering | Tools must be sorted deterministically before cache breakpoints are meaningful. If tool order changes between calls, the cache key changes and breakpoints are useless. |

---

## Tests First

**Test file:** `tests/unit/test_prompt_caching.py`

```python
# Test: build_system_message returns structured content with cache_control for Anthropic provider
# Setup: Call build_system_message with provider="anthropic" and a valid AgentConfig
# Assert: returned SystemMessage.content is a list with one dict containing
#         keys "type", "text", and "cache_control" where cache_control == {"type": "ephemeral"}

# Test: build_system_message returns plain string for OpenAI provider
# Setup: Call build_system_message with provider="openai"
# Assert: returned SystemMessage.content is a plain string, no cache_control anywhere

# Test: build_system_message returns plain string for Ollama provider
# Setup: Call build_system_message with provider="ollama"
# Assert: returned SystemMessage.content is a plain string

# Test: build_system_message returns structured content for Bedrock provider
# Setup: Call build_system_message with provider="bedrock"
# Assert: returned SystemMessage.content includes cache_control

# Test: last tool in Anthropic dict list has cache_control set
# Setup: Convert 3 sorted tools to Anthropic dicts via the caching path
# Assert: only the last dict has a "cache_control" key; first two do not

# Test: non-Anthropic tool path does not add cache_control
# Setup: Convert tools for a non-Anthropic provider path
# Assert: no tool dict has a "cache_control" key
```

---

## Implementation

Three changes across three files, described below.

### 3a. Tool-Level Cache Breakpoint

**File:** `src/quantstack/tools/registry.py`

**What to change:** After the tool list is sorted (from section-02) and converted to Anthropic dict format, add `cache_control: {"type": "ephemeral"}` to the **last** tool definition in the list. This creates a cache boundary: all tool definitions up to and including this breakpoint are cached as one block.

**Where in the code:** The function `get_tools_for_agent_with_search()` (starts at line 338) builds `tools_for_api` by iterating `tool_names` and calling `tool_to_anthropic_dict()` for each. After the loop completes but before appending `TOOL_SEARCH_TOOL`:

1. If `tools_for_api` is non-empty, add `cache_control` to the last regular tool dict:
   ```python
   tools_for_api[-1]["cache_control"] = {"type": "ephemeral"}
   ```
2. Then append `TOOL_SEARCH_TOOL` as before.

The `TOOL_SEARCH_TOOL` dict is a meta-tool (type `tool_search_bm25_2025_04_15`) and should NOT get `cache_control`. The cache breakpoint goes on the last real tool.

**`tool_to_anthropic_dict` itself** (in `src/quantstack/graphs/tool_search_compat.py`, line 44) does not need modification. It converts a BaseTool to an Anthropic dict and optionally adds `defer_loading`. The cache breakpoint is a list-level concern (only the last item gets it), not a per-tool concern.

### 3b. System Message Cache Breakpoint (Provider-Aware)

**File:** `src/quantstack/graphs/agent_executor.py`

**What to change:** Modify `build_system_message()` (line 152) to accept a `provider` parameter and return a structured `SystemMessage` with `cache_control` when the provider supports it.

**Current signature:**
```python
def build_system_message(
    config: AgentConfig,
    graph_name: str | None = None,
) -> SystemMessage:
```

**New signature:**
```python
def build_system_message(
    config: AgentConfig,
    graph_name: str | None = None,
    provider: str | None = None,
) -> SystemMessage:
```

**Provider-aware return logic** (at the end of the function, where it currently returns `SystemMessage(content=base)`):

For Anthropic or Bedrock providers, return a structured content block:
```python
SystemMessage(content=[{"type": "text", "text": base, "cache_control": {"type": "ephemeral"}}])
```

For OpenAI, Ollama, or unknown/None providers, return the plain string form:
```python
SystemMessage(content=base)
```

The provider check should be a simple string comparison: `if provider in ("anthropic", "bedrock"):`.

**Callers to update:** `build_system_message` is called in two places within `agent_executor.py`:

1. `run_agent()` at line 212: `build_system_message(config)` -- the `llm` parameter is available here. Determine the provider from the LLM class name: `type(llm).__name__` is `"ChatAnthropic"` for Anthropic direct, `"ChatBedrock"` for Bedrock. Map to the provider string and pass it through.
2. `run_agent_bigtool()` at line 437: `build_system_message(config)` -- same approach, check `type(llm).__name__`.

A simple helper to extract the provider string from the LLM object:
```python
def _detect_provider(llm: BaseChatModel) -> str:
    """Infer provider name from LLM class for cache_control decisions."""
    cls_name = type(llm).__name__
    if cls_name == "ChatAnthropic":
        return "anthropic"
    if cls_name == "ChatBedrock":
        return "bedrock"
    if "openai" in cls_name.lower():
        return "openai"
    return "other"
```

**Minimum token constraint:** Anthropic requires 1,024-2,048 tokens minimum for caching to activate. Most agent system prompts in QuantStack are well above this threshold (role + goal + backstory + tool guidance). If an agent's prompt is very short, caching will be silently skipped by the API (no error). This is acceptable -- short prompts are cheap anyway. No code changes needed for this constraint.

### 3c. Cache Hit Rate Observability

**File:** `src/quantstack/observability/tracing.py`

**What to add:** A new function that logs prompt cache metrics from the Anthropic API response's `usage` object. The Anthropic API returns `cache_read_input_tokens` and `cache_creation_input_tokens` in `response.usage`. These fields tell you how many tokens were served from cache vs. newly cached.

New function signature:
```python
def trace_prompt_cache_metrics(
    agent_name: str,
    model_name: str,
    cache_read_tokens: int,
    cache_creation_tokens: int,
    total_input_tokens: int,
) -> None:
    """Log prompt cache hit/miss metrics to Langfuse.

    Enables monitoring of:
    - Cache hit rate per agent over time
    - Detection of cache regressions (non-deterministic prompt content)
    - Cost reduction verification against the 50%+ target
    """
```

This function should create a Langfuse trace with tags `["prompt_cache", "cost"]` and metadata containing the token counts plus a computed `cache_hit_rate` (= `cache_read_tokens / total_input_tokens` when `total_input_tokens > 0`).

**Caller integration** in `src/quantstack/graphs/agent_executor.py`: In `run_agent()`, after each LLM call (around line 248 where `usage_metadata` is extracted), check for cache-specific fields:

```python
# After the existing usage extraction block:
um = response.usage_metadata or {}
cache_read = um.get("cache_read_input_tokens", 0)
cache_creation = um.get("cache_creation_input_tokens", 0)
if cache_read or cache_creation:
    trace_prompt_cache_metrics(
        agent_name=config.name,
        model_name=model_name,
        cache_read_tokens=cache_read,
        cache_creation_tokens=cache_creation,
        total_input_tokens=um.get("input_tokens", 0),
    )
```

Only log when there are actual cache metrics (non-zero values) to avoid noise from providers that don't support caching.

---

## Bedrock Considerations

Prompt caching is GA on Bedrock -- no `anthropic_beta` header needed. The `cache_control` syntax works identically via the Bedrock InvokeModel API.

**Verification required after implementation:** Confirm that `langchain_aws.ChatBedrock` passes `cache_control` content blocks through to the API rather than stripping them. If `ChatBedrock` uses the Converse API internally, `cache_control` may be silently dropped (Converse uses a different `cachePoint` syntax). After deploying, verify that Bedrock-routed calls show `cache_read_input_tokens > 0`, not just Anthropic-direct calls. If Bedrock strips `cache_control`, the fix is either to switch Bedrock to the InvokeModel API path or to use the Converse `cachePoint` syntax instead.

---

## Verification Plan

After deployment, run 3+ consecutive trading cycles and check:

1. `cache_read_input_tokens > 0` in Langfuse traces for the second cycle onward
2. Cache hit rate logs (from `trace_prompt_cache_metrics`) show >80% cache reads for stable agents
3. Total prompt token cost drops 50%+ vs. pre-deployment baseline (compare Langfuse cost dashboard day-over-day)

If cache hit rates are low, investigate:
- Are tools still non-deterministically ordered? (section-02 regression)
- Is any dynamic content injected into system prompts that changes per-call?
- Is Bedrock silently stripping `cache_control`? Check Bedrock CloudWatch for cache metrics.

---

## File Change Summary

| File | Change |
|------|--------|
| `src/quantstack/tools/registry.py` | Add `cache_control` to last tool dict in `get_tools_for_agent_with_search()` |
| `src/quantstack/graphs/agent_executor.py` | Add `provider` param to `build_system_message()`, add `_detect_provider()` helper, pass provider at call sites, add cache metric extraction after LLM calls |
| `src/quantstack/observability/tracing.py` | Add `trace_prompt_cache_metrics()` function |
| `tests/unit/test_prompt_caching.py` | New test file with 6 tests covering provider-aware caching |
