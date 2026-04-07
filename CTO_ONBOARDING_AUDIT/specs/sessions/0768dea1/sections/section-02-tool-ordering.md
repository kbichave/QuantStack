# Section 02: Deterministic Tool Ordering (Item 0.1)

## Problem

The `TOOL_REGISTRY` in `src/quantstack/tools/registry.py` is a Python dict populated from 19+ import sources. Dict iteration order depends on insertion order, which depends on module import sequencing. This means the tool list sent to the LLM API can differ between process restarts.

Anthropic's prompt cache key is a hash of the full request prefix. The `tools` field is the first level in the cache hierarchy (`tools -> system -> messages`). If tool definitions differ in order between calls, the cache key changes and the entire prefix becomes a cache miss. With 21 agents cycling every 5-10 minutes, this wastes 30-50% of prompt token spend.

Fixing tool ordering is also a prerequisite for Section 03 (Prompt Caching), which adds `cache_control` breakpoints to the last tool in the sorted list. Without deterministic ordering, the breakpoint lands on a different tool each restart, defeating caching entirely.

## Scope

This change only affects the **return order** of tools from two functions. It does not change:
- Tool registration
- Tool definitions
- The `TOOL_REGISTRY` dict itself
- Any tool behavior

## Files to Modify

- `src/quantstack/tools/registry.py` -- two functions
- `tests/unit/test_tool_ordering.py` -- new file

## Dependencies

- **None.** This section can be implemented independently.
- **Blocks:** Section 03 (Prompt Caching) depends on this section being complete.

## Tests First

**Test file:** `tests/unit/test_tool_ordering.py`

All tests mock the `TOOL_REGISTRY` to avoid importing the full tool dependency tree. Use `unittest.mock.patch` to replace the registry with a controlled dict.

```python
"""Tests for deterministic tool ordering in registry.py."""

import pytest
from unittest.mock import patch, MagicMock
from langchain_core.tools import BaseTool


def _make_mock_tool(name: str) -> BaseTool:
    """Create a mock BaseTool with a given name."""
    tool = MagicMock(spec=BaseTool)
    tool.name = name
    tool.description = f"Description for {name}"
    return tool


# Test: get_tools_for_agent returns tools sorted alphabetically by name
# Setup: Register 3 tools with names "zebra_tool", "alpha_tool", "mid_tool"
# Action: Call get_tools_for_agent(["zebra_tool", "alpha_tool", "mid_tool"])
# Assert: returned list has names in order ["alpha_tool", "mid_tool", "zebra_tool"]


# Test: get_tools_for_agent_with_search returns sorted deferred and always-loaded lists
# Setup: Register mix of deferred and always-loaded tools with unsorted names
# Action: Call get_tools_for_agent_with_search(tool_names, always_loaded)
# Assert: both tools_for_api and tools_for_execution are sorted by name


# Test: TOOL_SEARCH_TOOL remains last after sorting
# Setup: Register tools where TOOL_SEARCH_TOOL name would sort to middle alphabetically
# Action: Call get_tools_for_agent_with_search
# Assert: TOOL_SEARCH_TOOL is the last item in tools_for_api list


# Test: sorting is stable across multiple calls
# Setup: Call get_tools_for_agent twice with same tool names
# Assert: identical order both times (object identity or name equality)


# Test: tool_to_anthropic_dict preserves sort order from input
# Setup: Pass pre-sorted tool list through the conversion path
# Assert: output dicts maintain same order as input
```

## Implementation

### Function 1: `get_tools_for_agent()` (line 320)

**Current behavior:** Iterates `tool_names` in caller-provided order, appends each resolved `BaseTool` to a list, returns it.

**Change:** After resolving all tools, sort the result list by `tool.name` before returning.

Sort key: `lambda t: t.name`

```python
def get_tools_for_agent(tool_names: list[str] | tuple[str, ...]) -> list[BaseTool]:
    """Resolve tool name strings to tool objects, sorted by name for cache stability."""
    tools = []
    for name in tool_names:
        if name not in TOOL_REGISTRY:
            available = sorted(TOOL_REGISTRY.keys())
            raise KeyError(
                f"Tool '{name}' not found in TOOL_REGISTRY. "
                f"Available tools: {available}"
            )
        tools.append(TOOL_REGISTRY[name])
    tools.sort(key=lambda t: t.name)
    return tools
```

### Function 2: `get_tools_for_agent_with_search()` (line 338)

**Current behavior:** Iterates `tool_names` in caller-provided order. For each name, calls `tool_to_anthropic_dict()` and appends the result dict to `tools_for_api`. Finally appends `TOOL_SEARCH_TOOL` to the end.

**Critical detail:** The `tools_for_api` dict list is built by iterating `tool_names`. If the iteration order is unsorted, the API dicts will also be unsorted even if `tools_for_execution` is sorted separately. The fix must sort the input `tool_names` before the iteration loop, not just the outputs.

**Change:** Sort `tool_names` before the loop that builds `tools_for_api`. The `tools_for_execution` list is already sorted by `get_tools_for_agent()` (from Change 1). After building the sorted `tools_for_api`, append `TOOL_SEARCH_TOOL` as the last item.

```python
def get_tools_for_agent_with_search(
    tool_names: list[str] | tuple[str, ...],
    always_loaded: list[str] | tuple[str, ...],
) -> tuple[list[dict], list[BaseTool]]:
    """Partition tools into deferred/always-loaded and return API-ready dicts.
    
    Tools are sorted by name for prompt cache stability.
    TOOL_SEARCH_TOOL is always appended last.
    """
    always_loaded_set = set(always_loaded)
    tool_names_set = set(tool_names)
    invalid = always_loaded_set - tool_names_set
    if invalid:
        raise ValueError(
            f"always_loaded contains tools not in tool_names: {sorted(invalid)}"
        )

    tools_for_execution = get_tools_for_agent(tool_names)
    tool_map = {t.name: t for t in tools_for_execution}

    tools_for_api: list[dict] = []
    for name in sorted(tool_names):                          # <-- sorted
        defer = name not in always_loaded_set
        tools_for_api.append(tool_to_anthropic_dict(tool_map[name], defer=defer))

    tools_for_api.append(TOOL_SEARCH_TOOL)                   # always last

    return tools_for_api, tools_for_execution
```

### TOOL_SEARCH_TOOL Handling

`TOOL_SEARCH_TOOL` is a meta-tool dict (not a `BaseTool`) imported from `quantstack.graphs.tool_search_compat`. It enables deferred tool search and is not a regular tool. It must always be the **last** item in `tools_for_api` after sorting. The current code already appends it after the loop, and this behavior is preserved.

## Edge Cases

- **Tools with identical names:** Not possible -- dict keys are unique, and `TOOL_REGISTRY` is a dict.
- **Dynamic tools added at runtime (via `_try_import`):** Sorted on next call to either function; insertion order is irrelevant.
- **Agent YAML configs listing tools in a specific intentional order:** There is no semantic meaning to tool order in agent configs. The LLM receives tool schemas as an unordered set.
- **Empty tool list:** `sorted([])` returns `[]`. No special handling needed.
- **Single tool:** Sorting a single-element list is a no-op. Correct.

## Verification

After implementing, verify with a quick smoke test:

1. Call `get_tools_for_agent(["web_search", "fetch_market_data", "signal_brief"])` and confirm the returned list is ordered `[fetch_market_data, signal_brief, web_search]` by `.name`.
2. Call `get_tools_for_agent_with_search(...)` and confirm `TOOL_SEARCH_TOOL` is the last element of `tools_for_api`.
3. Restart the process and repeat -- order must be identical.
