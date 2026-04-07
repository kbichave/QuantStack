# Section 08: Tool Access Control

## Problem

No runtime enforcement exists for tool access boundaries between graphs. Any agent can call any tool registered in `TOOL_REGISTRY`. The only constraint is which tools are listed in an agent's `agents.yaml` config, but nothing enforces this at the invocation layer. A misconfigured `agents.yaml` could allow `hypothesis_critic` (research graph) to call `execute_order` (execution tool). This is a security and safety gap -- research agents should never be able to place orders, and supervisor agents should never execute trades.

## Dependencies

- **section-03-pydantic-state-migration**: The Pydantic state models must be in place before this section, since the `blocked_tools` configuration loading integrates with the same config pipeline that the Pydantic migration touches.
- **Blocked by**: section-14-integration-tests (this section is a prerequisite for integration testing).

## Design

### Configuration: `blocked_tools` per graph

Add a top-level `blocked_tools` key to each graph's `agents.yaml`. This is a graph-level key (not per-agent) because tool access is a graph boundary concern, not an agent boundary concern. A research agent should never execute orders regardless of which specific research agent it is.

**Research graph** (`src/quantstack/graphs/research/config/agents.yaml`):
```yaml
blocked_tools:
  - execute_order
  - cancel_order
  - activate_kill_switch
```

**Trading graph** (`src/quantstack/graphs/trading/config/agents.yaml`):
```yaml
blocked_tools:
  - register_strategy
  - train_model
```

**Supervisor graph** (`src/quantstack/graphs/supervisor/config/agents.yaml`):
```yaml
blocked_tools:
  - execute_order
  - cancel_order
  - activate_kill_switch
  - submit_option_order
  - close_position
```

The supervisor graph blocks all execution tools. It is read-only by design.

### Enforcement: Guard in `agent_executor.py`

A ~5-line guard in `src/quantstack/graphs/agent_executor.py` at the tool invocation point, inside the `run_agent` function's tool execution loop (around line 315, where `for tool_call in response.tool_calls:` iterates). Before calling any tool, check if the tool name appears in the current graph's `blocked_tools` list.

When a blocked tool is called:
1. Return an error message to the agent (not a Python exception -- let the LLM handle it gracefully and choose a different approach).
2. Log a security event via Langfuse with: agent name, tool name, graph name, timestamp.
3. Do NOT circuit-break the agent. The violation is a config bug or LLM hallucination, not an agent health failure. Tripping the circuit breaker for a tool access violation would punish the agent for something that is not a reliability issue.

### Loading `blocked_tools` from YAML

The `load_agent_configs()` function in `src/quantstack/graphs/config.py` currently parses only agent-level entries from the YAML. It needs a companion function (or modification) to also extract the graph-level `blocked_tools` key. The `ConfigWatcher` already hot-reloads `agents.yaml` on file changes (dev mode) and SIGHUP (prod mode). The `blocked_tools` list should be reloaded alongside agent configs so that access control changes take effect without restart.

Two approaches for loading:

**Option A** -- Add a `blocked_tools` return to `load_agent_configs()`:
Modify `load_agent_configs()` to return both the agent configs dict and the blocked_tools list. This changes the function signature, which means callers need updating.

**Option B** -- Separate loader function:
Add `load_blocked_tools(yaml_path: Path) -> frozenset[str]` that reads the same YAML and extracts only the `blocked_tools` key. The `ConfigWatcher` calls this alongside `load_agent_configs()` and stores both.

Option B is cleaner -- it avoids changing the existing function contract and keeps concerns separated. The `ConfigWatcher` gains a `get_blocked_tools() -> frozenset[str]` method.

### Passing `blocked_tools` to the agent executor

The `run_agent()` function needs access to the blocked tools list. Two options:

1. Pass `blocked_tools: frozenset[str]` as a parameter to `run_agent()`. This is explicit and testable.
2. Look up the graph name from `_AGENT_TEAMS` and query a module-level registry. This couples the executor to global state.

Option 1 is preferred. The caller (the graph node function) already has access to the `ConfigWatcher` instance and knows which graph it belongs to. It passes `blocked_tools` when calling `run_agent()`.

### Bypass prevention

The guard must be at the invocation layer inside `run_agent()`, not at the YAML config level. If someone calls a tool through `TOOL_REGISTRY` directly (bypassing the agent executor), the guard does not apply -- but that path is only used by deterministic node code, not by LLM agents. The YAML `tools` list already constrains which tools an agent sees, and the `blocked_tools` guard is a defense-in-depth layer for when the YAML `tools` list is misconfigured or when the LLM hallucinates a tool name that happens to exist in the registry.

## Tests

All tests go in `tests/unit/test_tool_access_control.py`.

```python
# tests/unit/test_tool_access_control.py
"""Tests for tool access control (Section 08)."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool

from quantstack.graphs.agent_executor import run_agent
from quantstack.graphs.config import AgentConfig


# --- Fixtures and helpers ---

# Helper to build an AgentConfig for testing
# (reuse the pattern from test_agent_executor.py)

# Helper to build a mock BaseTool

# Helper to build an AIMessage with tool_calls


# --- Test: blocked tool returns error to agent ---

# Test: research agent calls execute_order -> blocked, error message returned to agent
# Setup: run_agent with blocked_tools=frozenset({"execute_order"}), LLM's first response
#   has a tool_call for "execute_order", second response is final text.
# Assert: the ToolMessage returned to the LLM contains an error about blocked access.
# Assert: the agent is NOT terminated -- it gets a chance to respond after the error.


# --- Test: security event logged on violation ---

# Test: research agent calls execute_order -> security event logged
# Setup: same as above, but patch the Langfuse logging function.
# Assert: Langfuse log called with agent_name, tool_name="execute_order",
#   graph_name="research", and a timestamp.


# --- Test: trading agent blocked from research tools ---

# Test: trading agent calls register_strategy -> blocked
# Setup: run_agent with blocked_tools=frozenset({"register_strategy"}), LLM tries
#   to call register_strategy.
# Assert: error returned to agent, security event logged.


# --- Test: supervisor blocked from all execution tools ---

# Test: supervisor agent calls any execution tool -> blocked
# Setup: blocked_tools includes execute_order, cancel_order, activate_kill_switch, etc.
# Assert: each one is blocked with error + log.


# --- Test: allowed tool proceeds normally ---

# Test: trading agent calls fetch_portfolio (allowed) -> execution proceeds
# Setup: run_agent with blocked_tools=frozenset({"register_strategy"}), LLM calls
#   fetch_portfolio which is NOT in blocked_tools.
# Assert: tool executes normally, result returned to LLM.


# --- Test: blocked tool does NOT circuit-break the agent ---

# Test: blocked tool call does not trip circuit breaker
# Setup: run_agent with blocked_tools, LLM calls blocked tool.
# Assert: no circuit breaker state change. The agent continues normally after
#   receiving the error message.


# --- Test: bypass via TOOL_REGISTRY directly -> still blocked at invocation layer ---

# Test: tool access guard is at the run_agent invocation layer
# Setup: call run_agent with a tool that IS in the tool_map but IS in blocked_tools.
# Assert: the guard catches it before tool_map[tool_name].ainvoke() is called.
# Note: this test verifies the guard placement, not a literal TOOL_REGISTRY bypass
#   (which would skip run_agent entirely and is outside the enforcement boundary).


# --- Test: ConfigWatcher hot-reload of blocked_tools ---

# Test: ConfigWatcher hot-reload of blocked_tools -> new blocks take effect
# Setup: create a tmp_path agents.yaml with blocked_tools: [execute_order].
#   Load via ConfigWatcher. Then modify the file to add cancel_order to blocked_tools.
#   Trigger reload. Verify get_blocked_tools() returns the updated set.
```

## Implementation Details

### File: `src/quantstack/graphs/config.py`

Add a new function to load blocked tools from YAML:

```python
def load_blocked_tools(yaml_path: Path) -> frozenset[str]:
    """Load graph-level blocked_tools list from agents.yaml.

    Returns an empty frozenset if the key is not present.
    """
    # Parse yaml_path, extract top-level 'blocked_tools' key
    # Return frozenset of tool name strings
```

The function reads the YAML, checks for a top-level `blocked_tools` key, validates it is a list of strings, and returns a `frozenset`. If the key is absent, returns an empty `frozenset`.

### File: `src/quantstack/graphs/config_watcher.py`

Extend `ConfigWatcher` to also load and store `blocked_tools`:

- In `__init__`, call `load_blocked_tools()` alongside `load_agent_configs()`.
- Store as `self._blocked_tools: frozenset[str]`.
- Add `get_blocked_tools() -> frozenset[str]` method (thread-safe, under same lock).
- In `_stage_reload()`, reload blocked_tools alongside agent configs.
- In `apply_pending_reload()`, swap both atomically.

### File: `src/quantstack/graphs/agent_executor.py`

Add `blocked_tools: frozenset[str] = frozenset()` parameter to `run_agent()` (and `run_agent_bigtool()` for consistency).

Inside the tool execution loop (the `for tool_call in response.tool_calls:` block), add the guard before the existing `if tool_name not in tool_map:` check:

```python
# Tool access control guard
if tool_name in blocked_tools:
    result = json.dumps({
        "error": "tool_access_denied",
        "tool": tool_name,
        "message": f"Tool '{tool_name}' is not available in this graph context.",
    })
    tool_ok = False
    # Log security event (Langfuse)
    # ... log with agent_name=config.name, tool_name, graph_name, timestamp
    messages.append(ToolMessage(content=result, tool_call_id=tool_id))
    continue
```

The guard goes BEFORE the `tool_name not in tool_map` check. This ensures that even if the tool exists in `tool_map`, the block is enforced. The `continue` skips to the next tool_call without executing the blocked tool.

### File: `src/quantstack/graphs/*/config/agents.yaml` (all three)

Add the `blocked_tools` key at the top level of each file (before the first agent definition).

### Graph node callers

Each graph's node functions that call `run_agent()` need to pass `blocked_tools` from the `ConfigWatcher`. The specific files depend on how each graph's runner passes config to nodes, but the pattern is:

```python
# In a graph node function:
blocked = config_watcher.get_blocked_tools()
response = await run_agent(llm, tools, agent_config, prompt, blocked_tools=blocked)
```

## Verification Checklist

- [ ] `blocked_tools` key added to all 3 `agents.yaml` files
- [ ] `load_blocked_tools()` function in `config.py`
- [ ] `ConfigWatcher` loads, stores, and hot-reloads `blocked_tools`
- [ ] `run_agent()` accepts `blocked_tools` parameter
- [ ] Guard inserted before tool execution in `run_agent()`
- [ ] Blocked tool returns JSON error to agent (not a Python exception)
- [ ] Security event logged via Langfuse on violation
- [ ] Blocked tool does NOT trip circuit breaker
- [ ] All 8 test cases pass
- [ ] Hot-reload test passes (ConfigWatcher picks up changes to `blocked_tools`)
