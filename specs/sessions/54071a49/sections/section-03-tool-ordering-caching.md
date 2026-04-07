# Section 3: Deterministic Tool Ordering & Prompt Caching

## Problem Statement

Two related issues cause unnecessary LLM cost:

1. **Non-deterministic tool ordering (finding MC1):** Tool definitions injected from `TOOL_REGISTRY` in dict iteration order means tool ordering can vary between runs. Since Anthropic's prompt cache key includes tool definitions, non-deterministic ordering invalidates the entire cache prefix on every request.

2. **No prompt caching configured (finding MC0c):** Despite Anthropic/Bedrock supporting prompt caching (90% cost reduction on cached prefixes with a 5-minute TTL that auto-refreshes on hit), zero caching infrastructure was wired up. The system prompt contains a static agent persona + tool definitions (cacheable) followed by dynamic state (not cacheable). Without a cache breakpoint separating these, every request pays full input token cost.

**Expected cost impact:** At ~$126/day in system prompt tokens for the heavy tier, prompt caching at 80%+ hit rate saves ~$100/day. Combined with the existing Groq hybrid routing, total daily cost drops from ~$40-60/day to ~$5-7/day.

## Current State Assessment

Before implementing, verify the current state of each file. Based on code review, several of these changes may already be partially or fully addressed in the codebase:

### Tool Ordering

**File:** `src/quantstack/tools/registry.py`

The function `get_tools_for_agent()` (around line 437) already sorts tools alphabetically by name (`tools.sort(key=lambda t: t.name)` at line 468). The docstring explicitly states this is for prompt cache stability. Similarly, `get_tools_for_agent_with_search()` iterates over `sorted(tool_names)` when building the API-ready dicts (line 501).

**Verdict:** Tool ordering is already deterministic. No changes needed here unless a code path bypasses these functions.

### Prompt Caching in System Messages

**File:** `src/quantstack/graphs/agent_executor.py`

The function `build_system_message()` (around line 258) already checks `if provider in ("anthropic", "bedrock")` and returns a `SystemMessage` with structured content containing `cache_control: {"type": "ephemeral"}` (lines 288-293). Non-Anthropic providers get a plain string `SystemMessage`.

**Verdict:** System message caching is already implemented.

### Bedrock Beta Header

**File:** `src/quantstack/llm/provider.py`

The `_instantiate_chat_model()` function already wires `BEDROCK_PROMPT_CACHING_BETA` into Bedrock model kwargs when `config.prompt_caching` is True (line 257). For Anthropic direct, it adds the beta header via `extra_headers` (lines 274-276).

**Verdict:** Bedrock/Anthropic prompt caching headers are already wired.

### Cache Control on Tool Definitions

**File:** `src/quantstack/tools/registry.py`

The function `get_tools_for_agent_with_search()` already marks the last tool definition with `cache_control: {"type": "ephemeral"}` (line 508), so Anthropic caches all tool definitions as one block.

**Verdict:** Tool definition caching is already implemented for the Anthropic native path.

### Prompt Caching Feature Flag

**File:** `src/quantstack/llm/config.py`

`PROMPT_CACHING_ENABLED_DEFAULT = True` is set, and `get_chat_model()` in `provider.py` respects the `PROMPT_CACHING_ENABLED` env var with this default.

**Verdict:** Feature flag exists and defaults to enabled.

## Implementation Approach

This section has two sequential steps. Step 1 (tool ordering) is a prerequisite for Step 2 (prompt caching) because tool ordering changes invalidate the cache prefix.

### Step 1: Deterministic Tool Ordering

Sort the tool list alphabetically by tool name before passing to the LLM. This ensures the cache prefix is stable across requests.

**Files to modify:**
- `src/quantstack/tools/registry.py` -- Ensure `get_tools_for_agent()` sorts tools by name
- `src/quantstack/graphs/tool_binding.py` -- Verify sorted tools flow through to `llm.bind_tools()`

**What to verify (may already be done):**
- `get_tools_for_agent()` returns tools in deterministic alphabetical order
- `get_tools_for_agent_with_search()` builds API dicts in sorted order
- `_bind_all_tools()` in `tool_binding.py` calls `get_tools_for_agent()` (which sorts), so its output is also deterministic
- No code path exists where unsorted tools reach `llm.bind_tools()` or `llm.bind(tools=...)`

### Step 2: Prompt Caching

Add `cache_control: {"type": "ephemeral"}` to the `SystemMessage`'s content for Anthropic/Bedrock providers. The cache breakpoint goes after the static content (persona + tool definitions) so that dynamic per-request state is not cached.

**Files to modify:**
- `src/quantstack/graphs/agent_executor.py` (lines 258-295) -- Add `cache_control` to `SystemMessage` for Anthropic/Bedrock
- `src/quantstack/llm/provider.py` -- Wire `BEDROCK_PROMPT_CACHING_BETA` into Bedrock model kwargs
- `src/quantstack/llm/config.py` -- Define `BEDROCK_PROMPT_CACHING_BETA` constant and `PROMPT_CACHING_ENABLED_DEFAULT`

**What to verify (may already be done):**
- `build_system_message()` returns structured content with `cache_control` for anthropic/bedrock
- `_instantiate_chat_model()` passes `anthropic_beta` header when `config.prompt_caching` is True
- `get_chat_model()` resolves `prompt_caching` flag from explicit arg, env var, or default
- Cache control only applies to Anthropic/Bedrock (heavy tier). Groq/Llama/Ollama do not support it.

## Tests

Tests go in two files. Write these first, then verify whether they already pass against the current codebase.

### Tool Ordering Tests

```python
# tests/tools/test_tool_ordering.py

"""Tests for deterministic tool ordering.

Tool ordering must be stable across calls and independent of registry
insertion order. This is critical for Anthropic prompt cache stability
since the cache key includes tool definitions.
"""

# Test: get_tools_for_agent returns tools sorted alphabetically by name
#   - Register tools with names ["zebra", "alpha", "mango"]
#   - Call get_tools_for_agent(["zebra", "alpha", "mango"])
#   - Assert returned list has names ["alpha", "mango", "zebra"]

# Test: two consecutive calls produce identical tool ordering
#   - Call get_tools_for_agent() twice with same args
#   - Assert both return lists have identical name sequences

# Test: tool ordering is deterministic regardless of registry insertion order
#   - Insert tools in order [C, A, B] into registry
#   - Call get_tools_for_agent(["C", "A", "B"])
#   - Assert returned order is [A, B, C]

# Test: get_tools_for_agent_with_search returns API dicts in sorted order
#   - Call with tool_names=["zebra", "alpha"], always_loaded=["alpha"]
#   - Assert tools_for_api names are in ["alpha", "zebra"] order

# Test: bind_tools_to_llm passes sorted tools to llm.bind_tools
#   - Mock LLM with bind_tools that captures args
#   - Call _bind_all_tools()
#   - Assert captured tools are sorted by name
```

### Prompt Caching Tests

```python
# tests/graphs/test_prompt_caching.py

"""Tests for prompt caching configuration.

Prompt caching reduces Anthropic/Bedrock costs by ~90% when the cache
prefix (system prompt + tool definitions) is stable across requests.
"""

# Test: build_system_message includes cache_control for anthropic provider
#   - Call build_system_message(config, provider="anthropic")
#   - Assert returned SystemMessage.content is a list (structured content)
#   - Assert content[0] has "cache_control": {"type": "ephemeral"}

# Test: build_system_message includes cache_control for bedrock provider
#   - Call build_system_message(config, provider="bedrock")
#   - Assert returned SystemMessage.content is a list with cache_control

# Test: cache_control NOT applied to non-Anthropic providers
#   - Call build_system_message(config, provider="groq")
#   - Assert returned SystemMessage.content is a plain string (no structured content)
#   - Call build_system_message(config, provider="openai")
#   - Assert same plain string behavior

# Test: Bedrock model instantiation includes anthropic_beta header when prompt_caching=True
#   - Create ModelConfig with provider="bedrock", prompt_caching=True
#   - Call _instantiate_chat_model(config)
#   - Assert model_kwargs contains "anthropic_beta" with BEDROCK_PROMPT_CACHING_BETA

# Test: Bedrock model does NOT include anthropic_beta when prompt_caching=False
#   - Create ModelConfig with provider="bedrock", prompt_caching=False
#   - Call _instantiate_chat_model(config)
#   - Assert model_kwargs does NOT contain "anthropic_beta"

# Test: cache key stability -- N consecutive calls with same config produce same prefix
#   - Call build_system_message() 5 times with identical AgentConfig
#   - Assert all 5 return identical SystemMessage content
#   - This validates that no randomness or timestamp leaks into the cached prefix

# Test: get_chat_model respects PROMPT_CACHING_ENABLED env var
#   - Set PROMPT_CACHING_ENABLED=false
#   - Call get_chat_model("heavy")
#   - Assert returned model has prompt_caching=False in its config

# Test: prompt_caching defaults to True when env var not set
#   - Ensure PROMPT_CACHING_ENABLED is not in env
#   - Call get_chat_model("heavy")
#   - Assert prompt caching is enabled
```

## Verification After Deployment

After implementation, verify caching is working by checking Langfuse traces:

1. Look for `cache_creation_input_tokens` field in trace metadata -- this appears on the first request that creates a cache entry.
2. Look for `cache_read_input_tokens` field -- this appears on subsequent requests that hit the cache.
3. Cache hit rate should exceed 80% within the first hour. The 5-minute TTL auto-refreshes on each cache hit, so active agents keep their cache warm.
4. If cache hit rate is low, check whether dynamic content (timestamps, market data snapshots) is leaking into the system message before the cache breakpoint.

**Langfuse query approach:**
- Filter traces by `agent_name` and check the `usage` metadata for cache-related fields.
- Compare `cache_read_input_tokens / (cache_read_input_tokens + input_tokens)` as the hit rate metric.

## Dependencies

- **No upstream dependencies.** This section can be implemented independently and in parallel with sections 01, 02, 04, and 06.
- **Downstream:** Section 10 (Signal & Data Quality Gates) depends on this section for tool registry changes. However, the dependency is specifically on the `ACTIVE_TOOLS` / `PLANNED_TOOLS` split in the registry, not on tool ordering or caching per se.

## Rollback

If prompt caching causes issues (e.g., stale cached responses, unexpected billing):
- Set `PROMPT_CACHING_ENABLED=false` in `.env` to disable without code changes.
- Tool ordering changes are safe to keep regardless -- deterministic ordering is strictly better even without caching.
