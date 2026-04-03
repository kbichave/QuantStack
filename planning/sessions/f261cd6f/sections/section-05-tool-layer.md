# Section 05: Tool Layer Refactor

## Overview

The existing tool layer has two parallel implementations that both need migration:

1. **`src/quantstack/crewai_tools/`** (24 files) — Thin wrappers using CrewAI's `@tool` decorator. Each wraps an async MCP function via a sync `run_async()` bridge. These are purely LLM-facing tools.

2. **`src/quantstack/tools/mcp_bridge/`** (8 files) — Class-based tools subclassing `crewai_compat.BaseTool` with a `_run()` method. These call MCP server functions through an `MCPBridge` class.

Additionally, 7 files in `src/quantstack/tools/` (top-level) import from `crewai_compat.BaseTool` and need the same treatment.

The migration replaces both patterns with two new directories:
- `src/quantstack/tools/langchain/` — LLM-facing tools using LangChain's `@tool` decorator (natively async, no sync bridge needed)
- `src/quantstack/tools/functions/` — Node-callable plain async functions (no decorator, called directly by graph nodes)

A `TOOL_REGISTRY` maps string names (referenced in agent YAML configs) to tool objects, enabling YAML-driven tool assignment in graph builders.

After migration, `src/quantstack/crewai_tools/` is deleted entirely, `src/quantstack/crewai_compat.py` is deleted, and all `crewai_compat.BaseTool` subclasses in `mcp_bridge/` are converted.

## Dependencies

- **Depends on**: Section 01 (scaffolding — directory creation, dependency installation)
- **Blocks**: Sections 06, 07, 08 (all graph builders consume tools from the registry)

## Tests (Write Before Implementation)

All tests go in `tests/unit/test_crewai_tools/` (rename directory to `tests/unit/test_tools/` as part of this work).

```python
# tests/unit/test_tools/test_tool_layer_contract.py

import importlib
import inspect
import pkgutil

import pytest


class TestLLMFacingToolContract:
    """Every tool in tools/langchain/ must satisfy the LangChain @tool contract."""

    def _get_tool_modules(self):
        """Discover all modules in quantstack.tools.langchain."""
        import quantstack.tools.langchain as pkg
        return [
            importlib.import_module(f"{pkg.__name__}.{info.name}")
            for info in pkgutil.iter_modules(pkg.__path__)
            if not info.name.startswith("_")
        ]

    def test_all_llm_tools_have_tool_decorator(self):
        """All public callables in tools/langchain/ must have the @tool decorator.

        LangChain's @tool sets a .tool attribute or makes the object an instance
        of BaseTool. Verify this property for every exported tool.
        """
        ...

    def test_all_llm_tools_have_nonempty_description(self):
        """LLM-facing tools need descriptions so the model knows when to call them."""
        ...

    def test_all_llm_tools_are_async(self):
        """LangChain @tool supports async natively. All our tools must be async
        to avoid blocking the event loop during graph execution."""
        ...

    def test_all_llm_tools_return_str(self):
        """LLM-facing tools must return JSON-serialized strings for the model to parse."""
        ...


class TestNodeCallableFunctionContract:
    """Every function in tools/functions/ must be a plain async function with type hints."""

    def _get_function_modules(self):
        """Discover all modules in quantstack.tools.functions."""
        import quantstack.tools.functions as pkg
        return [
            importlib.import_module(f"{pkg.__name__}.{info.name}")
            for info in pkgutil.iter_modules(pkg.__path__)
            if not info.name.startswith("_")
        ]

    def test_all_node_functions_are_async(self):
        """Node-callable functions must be coroutine functions."""
        ...

    def test_all_node_functions_have_type_hints(self):
        """Every parameter and return value must have type annotations."""
        ...

    def test_no_tool_decorator_on_node_functions(self):
        """Node-callable functions must NOT have @tool — they are called directly."""
        ...


class TestNoCrewAIImports:
    """After migration, no tool file should reference crewai or crewai_compat."""

    def test_no_crewai_imports(self):
        """Scan all .py files under tools/ for crewai imports. None should exist."""
        ...

    def test_crewai_compat_deleted(self):
        """crewai_compat.py must not exist after migration is complete."""
        ...


class TestToolRegistry:
    """TOOL_REGISTRY maps string names to tool objects for YAML-driven assignment."""

    def test_registry_contains_all_yaml_referenced_tools(self):
        """Every tool name referenced in any agents.yaml config must exist in TOOL_REGISTRY."""
        ...

    def test_registry_values_are_callable(self):
        """All registry entries must be callable (either @tool objects or async functions)."""
        ...

    def test_signal_brief_in_registry(self):
        """Smoke test: signal_brief tool exists and returns valid JSON with expected keys."""
        ...

    def test_risk_tools_in_registry(self):
        """Smoke test: risk-related tools exist in the registry."""
        ...

    def test_data_tools_in_registry(self):
        """Smoke test: data fetch tools exist in the registry."""
        ...
```

## Implementation Details

### Step 1: Create Directory Structure

```
src/quantstack/tools/
  langchain/
    __init__.py
    signal_tools.py
    analysis_tools.py
    research_tools.py
    data_tools.py
    execution_tools.py
    portfolio_tools.py
    options_tools.py
    risk_tools.py
    strategy_tools.py
    backtest_tools.py
    ml_tools.py
    intelligence_tools.py
    fundamentals_tools.py
    learning_tools.py
    web_tools.py
  functions/
    __init__.py
    risk_functions.py
    data_functions.py
    backtest_functions.py
    execution_functions.py
    system_functions.py
  registry.py           # TOOL_REGISTRY dict
```

### Step 2: Classification Heuristic

Audit every tool and classify it as **LLM-facing** or **node-callable** using this rule:

- **LLM-facing**: The tool is called by an agent node where the LLM decides *whether* and *when* to invoke it. The LLM needs the tool's name and description to make that decision. Examples: `get_signal_brief`, `run_backtest`, `fetch_fundamentals`, `search_knowledge_base`.

- **Node-callable**: The tool is always called unconditionally by a specific graph node. The node code calls it directly — no LLM decision involved. Examples: `get_system_status` (always called by `safety_check` node), `execute_order` (always called by `execute_entries` node), `save_strategy` (always called by `strategy_registration` node).

When in doubt, classify as LLM-facing. It is easy to demote an LLM-facing tool to a node-callable function later, but the reverse requires adding descriptions and decorator metadata.

### Step 3: Migrate `crewai_tools/` (LLM-Facing Pattern)

The existing pattern in `crewai_tools/` wraps async MCP functions with a sync `run_async()` bridge because CrewAI tools were synchronous:

```python
# OLD (crewai_tools/signal_tools.py)
from crewai.tools import tool
from quantstack.crewai_tools._async_bridge import run_async

@tool("Get Signal Brief")
def get_signal_brief_tool(symbol: str) -> str:
    result = run_async(get_signal_brief(symbol=symbol))
    return json.dumps(result, default=str)
```

The new pattern uses LangChain's natively async `@tool`:

```python
# NEW (tools/langchain/signal_tools.py)
from langchain_core.tools import tool

from quantstack.mcp.tools.signal import get_signal_brief, run_multi_signal_brief

@tool
async def signal_brief(symbol: str) -> str:
    """Get a technical signal brief for a symbol including trend, momentum, and key levels.

    Returns JSON with technical, fundamental, momentum, and regime signals.
    Call this first when evaluating any symbol.
    """
    result = await get_signal_brief(symbol=symbol)
    return json.dumps(result, default=str)
```

Key differences from the old pattern:
- `async def` instead of `def` + `run_async()` — no sync bridge needed
- `await` instead of `run_async()` — native async, no `nest_asyncio` dependency
- `@tool` (no string name argument) — LangChain infers the name from the function name
- Docstring is the tool description (LangChain reads it from the function docstring)
- Error handling: wrap in try/except and return JSON error object (same pattern as before)

Apply this conversion to all 24 files in `crewai_tools/`. The underlying MCP function imports stay the same — only the wrapper changes.

### Step 4: Migrate `mcp_bridge/` Tools (Class-Based Pattern)

The `mcp_bridge/` files use a class-based pattern subclassing `crewai_compat.BaseTool`:

```python
# OLD (tools/mcp_bridge/tools_analysis.py)
from quantstack.crewai_compat import BaseTool

class ComputeIndicatorsTool(BaseTool):
    name: str = "compute_indicators"
    description: str = "Compute technical indicators..."
    args_schema: type[BaseModel] = IndicatorsInput

    def _run(self, symbol: str, ...) -> str:
        result = _run_async(self._bridge.call_quantcore("compute_indicators", ...))
        return json.dumps(result, default=str)
```

For each class, determine if it is LLM-facing or node-callable, then convert:

**If LLM-facing** (most of them):
```python
# NEW (tools/langchain/analysis_tools.py)
@tool
async def compute_indicators(symbol: str, indicators: list[str] | None = None) -> str:
    """Compute technical indicators (RSI, MACD, ATR, Bollinger Bands, etc.) for a symbol."""
    result = await mcp_compute_indicators(symbol=symbol, indicators=indicators)
    return json.dumps(result, default=str)
```

**If node-callable**:
```python
# NEW (tools/functions/data_functions.py)
async def fetch_market_data(symbol: str, start_date: str, end_date: str) -> dict:
    """Fetch OHLCV market data for the given symbol and date range."""
    return await mcp_fetch_market_data(symbol=symbol, start_date=start_date, end_date=end_date)
```

Node-callable functions return native Python types (dict, list), not JSON strings. They have no `@tool` decorator and no Pydantic `args_schema`.

### Step 5: Migrate Top-Level `tools/` Files

Seven files in `src/quantstack/tools/` import `crewai_compat.BaseTool`:
- `alphavantage_tools.py`
- `options_flow_tools.py`
- `knowledge_tools.py`
- `memory_tools.py`

Apply the same classification and conversion. Move LLM-facing tools into `tools/langchain/`, node-callable functions into `tools/functions/`.

### Step 6: Build the Tool Registry

Create `src/quantstack/tools/registry.py`:

```python
"""Central tool registry mapping string names to tool objects.

Graph builders look up tools by name from agent YAML configs, enabling
YAML-driven tool assignment without hardcoded imports in graph code.
"""

from langchain_core.tools import BaseTool

# Import all LLM-facing tools
from quantstack.tools.langchain.signal_tools import signal_brief, multi_signal_brief
from quantstack.tools.langchain.analysis_tools import compute_indicators, ...
# ... (all LLM-facing tools)

TOOL_REGISTRY: dict[str, BaseTool] = {
    "signal_brief": signal_brief,
    "multi_signal_brief": multi_signal_brief,
    "compute_indicators": compute_indicators,
    # ... every LLM-facing tool, keyed by the name used in agents.yaml
}


def get_tools_for_agent(tool_names: list[str]) -> list[BaseTool]:
    """Resolve a list of tool name strings to tool objects.

    Raises KeyError with a clear message if any tool name is not found
    in the registry, listing both the missing name and available names.
    """
    ...
```

The registry only contains LLM-facing tools. Node-callable functions are imported directly by node implementations and do not appear in the registry.

### Step 7: Delete Old Code

After all tools are migrated and tests pass:

1. Delete `src/quantstack/crewai_tools/` entirely (24 files + `__init__.py` + `_async_bridge.py`)
2. Delete `src/quantstack/crewai_compat.py`
3. Remove `_run_async()` from `tools/mcp_bridge/_bridge.py` (or delete the file if no longer needed)
4. Remove `run_async` from any remaining imports
5. Verify no file in the codebase imports from `crewai_tools` or `crewai_compat`

### Step 8: Update `mcp_bridge/` Internals

The `MCPBridge` class in `tools/mcp_bridge/_bridge.py` may still be useful as the communication layer to MCP servers. However, the tool *wrappers* (`tools_analysis.py`, etc.) that subclass `BaseTool` should be deleted after their functions are extracted into `langchain/` or `functions/`. The bridge's `call_quantcore()` and `call_etrade()` methods become the underlying async functions that the new tools call.

If the `MCPBridge` class is only used by tool wrappers that are being deleted, it can be simplified or removed too. Evaluate during implementation.

## File Inventory

### Files to Create

| File | Purpose |
|------|---------|
| `src/quantstack/tools/langchain/__init__.py` | Package init, re-exports all tools |
| `src/quantstack/tools/langchain/signal_tools.py` | Signal analysis tools (from crewai_tools/signal_tools.py) |
| `src/quantstack/tools/langchain/analysis_tools.py` | Technical analysis tools (from mcp_bridge/tools_analysis.py + crewai_tools/analysis_tools.py) |
| `src/quantstack/tools/langchain/data_tools.py` | Data fetch tools (from crewai_tools/data_tools.py + mcp_bridge/tools_data.py) |
| `src/quantstack/tools/langchain/execution_tools.py` | Trade execution tools (from crewai_tools/execution_tools.py + mcp_bridge/tools_etrade.py) |
| `src/quantstack/tools/langchain/risk_tools.py` | Risk analysis tools (from crewai_tools/risk_tools.py + mcp_bridge/tools_risk.py) |
| `src/quantstack/tools/langchain/options_tools.py` | Options analysis tools (from crewai_tools/options_tools.py) |
| `src/quantstack/tools/langchain/portfolio_tools.py` | Portfolio management tools (from crewai_tools/portfolio_tools.py) |
| `src/quantstack/tools/langchain/strategy_tools.py` | Strategy registration tools (from crewai_tools/strategy_tools.py) |
| `src/quantstack/tools/langchain/backtest_tools.py` | Backtesting tools (from crewai_tools/backtest_tools.py) |
| `src/quantstack/tools/langchain/ml_tools.py` | ML experiment tools (from crewai_tools/ml_tools.py) |
| `src/quantstack/tools/langchain/research_tools.py` | Research tools (from crewai_tools/research_tools.py) |
| `src/quantstack/tools/langchain/intelligence_tools.py` | Market intelligence tools (from crewai_tools/intelligence_tools.py) |
| `src/quantstack/tools/langchain/fundamentals_tools.py` | Fundamental analysis tools (from crewai_tools/fundamentals_tools.py) |
| `src/quantstack/tools/langchain/learning_tools.py` | Learning/feedback tools (from crewai_tools/learning_tools.py + feedback_tools.py) |
| `src/quantstack/tools/langchain/web_tools.py` | Web search tools (from crewai_tools/web_tools.py) |
| `src/quantstack/tools/functions/__init__.py` | Package init |
| `src/quantstack/tools/functions/risk_functions.py` | Risk gate, safety check functions |
| `src/quantstack/tools/functions/data_functions.py` | Market data fetch functions |
| `src/quantstack/tools/functions/backtest_functions.py` | Backtest execution functions |
| `src/quantstack/tools/functions/execution_functions.py` | Order execution functions |
| `src/quantstack/tools/functions/system_functions.py` | System status, kill switch functions |
| `src/quantstack/tools/registry.py` | TOOL_REGISTRY + get_tools_for_agent() |
| `tests/unit/test_tools/test_tool_layer_contract.py` | Contract tests for all tool types |

### Files to Delete

| File | Reason |
|------|--------|
| `src/quantstack/crewai_tools/` (entire directory, 26 files) | Replaced by tools/langchain/ |
| `src/quantstack/crewai_compat.py` | BaseTool stub no longer needed |
| `src/quantstack/tools/mcp_bridge/tools_analysis.py` | Replaced by tools/langchain/analysis_tools.py |
| `src/quantstack/tools/mcp_bridge/tools_data.py` | Replaced by tools/langchain/data_tools.py |
| `src/quantstack/tools/mcp_bridge/tools_etrade.py` | Replaced by tools/langchain/execution_tools.py |
| `src/quantstack/tools/mcp_bridge/tools_market.py` | Replaced by tools/langchain/ (various) |
| `src/quantstack/tools/mcp_bridge/tools_risk.py` | Replaced by tools/langchain/risk_tools.py |
| `src/quantstack/tools/mcp_bridge/_factories.py` | Factory for BaseTool instances — no longer needed |
| `src/quantstack/tools/mcp_bridge/_schemas.py` | Pydantic schemas — inline into @tool parameter types or keep if shared |

### Files to Modify

| File | Change |
|------|--------|
| `src/quantstack/tools/__init__.py` | Update exports to reference langchain/ and functions/ |
| `src/quantstack/tools/mcp_bridge/_bridge.py` | Remove `_run_async()`, keep `MCPBridge` if still useful |
| `src/quantstack/tools/mcp_bridge/__init__.py` | Update exports |
| `src/quantstack/tools/alphavantage_tools.py` | Remove BaseTool subclass, convert to @tool or plain function |
| `src/quantstack/tools/options_flow_tools.py` | Remove BaseTool subclass, convert to @tool or plain function |
| `src/quantstack/tools/knowledge_tools.py` | Remove BaseTool subclass, convert to @tool or plain function |
| `src/quantstack/tools/memory_tools.py` | Remove BaseTool subclass, convert to @tool or plain function |

## Gotchas and Edge Cases

1. **`_schemas.py` in mcp_bridge**: Contains Pydantic models used as `args_schema` for BaseTool subclasses. LangChain's `@tool` infers schemas from function signatures, so most of these schemas become unnecessary. However, if any schema has complex validation logic (nested models, custom validators), that logic needs to move into the function body or remain as a shared model.

2. **`_async_bridge.py` and `nest_asyncio`**: The `run_async()` function exists because CrewAI tools were synchronous. LangChain `@tool` is natively async, so `run_async()` is no longer needed. However, do NOT remove `nest_asyncio` from dependencies yet — other code (outside the tool layer) may still use it. That cleanup happens in Section 12.

3. **Error handling pattern**: The existing tools return JSON error objects on failure rather than raising exceptions. Preserve this pattern for LLM-facing tools (the LLM needs to see the error in a parseable format). Node-callable functions should raise exceptions (the graph's error handling and retry_policy deal with them).

4. **`MCPBridge` class**: The bridge pattern (class with `call_quantcore()` / `call_etrade()` methods) was needed because CrewAI tools were class instances that needed a shared bridge object. With plain async functions, each tool can import and call MCP functions directly. The bridge class may still be useful for centralized connection management — evaluate during implementation.

5. **Tool naming**: LangChain's `@tool` uses the function name as the tool name by default. Choose function names carefully — they become the identifiers that agents see and that YAML configs reference. Use snake_case, be descriptive, and avoid collisions. The TOOL_REGISTRY keys must match exactly.

6. **Duplicate tools across crewai_tools/ and mcp_bridge/**: Some functionality exists in both directories (e.g., signal analysis in `crewai_tools/signal_tools.py` and `mcp_bridge/tools_analysis.py`). During migration, deduplicate — keep the version that calls the underlying MCP function most cleanly and discard the other.
