# Section 3: CrewAI Tool Wrappers

## Overview

This section covers the creation of ~60 CrewAI tool wrappers across 22 modules in `src/quantstack/crewai_tools/`. Each wrapper is a thin adapter between CrewAI's `@tool` decorator interface and the existing async functions in `src/quantstack/mcp/tools/`. The wrappers handle async-to-sync bridging, JSON serialization, and error containment.

**Dependencies:** Section 01 (project scaffolding) must be complete -- the `src/quantstack/crewai_tools/` directory and `nest_asyncio` dependency must exist.

**Blocks:** Sections 04 (agent definitions) and 05 (crew workflows) cannot proceed until tool wrappers are available for agents to reference.

---

## Background

The existing tool layer lives in `src/quantstack/mcp/tools/`. Functions there are decorated with `@tool_def()` and `@domain(...)`, are mostly `async`, and return Python dicts. CrewAI tools have different requirements:

1. Must use CrewAI's `@tool("Tool Name")` decorator from `crewai.tools`
2. Must return **strings** (not dicts)
3. Must be **synchronous** (CrewAI calls tools synchronously from its internal executor)
4. Must have non-empty docstrings (CrewAI uses the docstring as the tool description shown to the LLM)

The wrappers bridge these gaps without modifying any existing code.

---

## Tests (Write These First)

All test files go in `tests/unit/test_crewai_tools/`.

### `tests/unit/test_crewai_tools/test_tool_wrapper_contract.py`

Tests that validate the structural contract every tool wrapper must satisfy. These run against all discovered tool modules via parametrization.

```python
"""Tests for CrewAI tool wrapper structural contracts."""
import importlib
import json
import pkgutil

import pytest

import quantstack.crewai_tools as tools_pkg


def _all_tool_modules():
    """Discover all modules in quantstack.crewai_tools."""
    return [
        name
        for _, name, _ in pkgutil.iter_modules(tools_pkg.__path__)
        if not name.startswith("_")
    ]


def _all_tool_functions():
    """Yield (module_name, func_name, func) for every @tool-decorated function."""
    for mod_name in _all_tool_modules():
        mod = importlib.import_module(f"quantstack.crewai_tools.{mod_name}")
        for attr_name in dir(mod):
            obj = getattr(mod, attr_name)
            # CrewAI @tool decorator sets a .name attribute on the wrapper
            if callable(obj) and hasattr(obj, "name"):
                yield mod_name, attr_name, obj


@pytest.mark.parametrize(
    "mod_name,func_name,func",
    list(_all_tool_functions()),
    ids=lambda x: x if isinstance(x, str) else "",
)
class TestToolContract:
    def test_has_nonempty_docstring(self, mod_name, func_name, func):
        """CrewAI uses the docstring as the tool description for the LLM."""
        assert func.description, f"{mod_name}.{func_name} has empty description"

    def test_has_tool_name(self, mod_name, func_name, func):
        """Every tool must have a human-readable name via @tool('Name')."""
        assert func.name, f"{mod_name}.{func_name} missing tool name"
```

### `tests/unit/test_crewai_tools/test_signal_tools.py`

Representative test for one tool module. Each module follows the same pattern -- this serves as the template.

```python
"""Tests for signal_tools CrewAI wrappers."""
import json
from unittest.mock import AsyncMock, patch

import pytest


class TestGetSignalBriefTool:
    """get_signal_brief_tool calls the underlying async function and returns JSON."""

    @patch("quantstack.crewai_tools.signal_tools.get_signal_brief", new_callable=AsyncMock)
    def test_calls_underlying_function(self, mock_fn):
        """Tool passes symbol parameter to the underlying async function."""
        from quantstack.crewai_tools.signal_tools import get_signal_brief_tool

        mock_fn.return_value = {"symbol": "AAPL", "composite_score": 0.72}
        result = get_signal_brief_tool.run(symbol="AAPL")
        mock_fn.assert_called_once_with(symbol="AAPL")

    @patch("quantstack.crewai_tools.signal_tools.get_signal_brief", new_callable=AsyncMock)
    def test_returns_json_string(self, mock_fn):
        """Tool serializes dict result to JSON string."""
        from quantstack.crewai_tools.signal_tools import get_signal_brief_tool

        mock_fn.return_value = {"symbol": "AAPL", "score": 0.72}
        result = get_signal_brief_tool.run(symbol="AAPL")
        parsed = json.loads(result)
        assert parsed["symbol"] == "AAPL"

    @patch("quantstack.crewai_tools.signal_tools.get_signal_brief", new_callable=AsyncMock)
    def test_handles_exception_gracefully(self, mock_fn):
        """Tool returns error JSON instead of crashing when underlying function raises."""
        from quantstack.crewai_tools.signal_tools import get_signal_brief_tool

        mock_fn.side_effect = RuntimeError("DB connection lost")
        result = get_signal_brief_tool.run(symbol="AAPL")
        parsed = json.loads(result)
        assert parsed["error"] is not None
        assert "DB connection lost" in parsed["error"]


class TestRunMultiSignalBriefTool:
    """run_multi_signal_brief_tool handles list input and returns JSON."""

    @patch("quantstack.crewai_tools.signal_tools.run_multi_signal_brief", new_callable=AsyncMock)
    def test_passes_symbols_list(self, mock_fn):
        from quantstack.crewai_tools.signal_tools import run_multi_signal_brief_tool

        mock_fn.return_value = [{"symbol": "SPY"}, {"symbol": "QQQ"}]
        result = run_multi_signal_brief_tool.run(symbols="SPY,QQQ")
        # The tool should parse the comma-separated string into a list
        mock_fn.assert_called_once()
```

### `tests/unit/test_crewai_tools/test_risk_tools.py`

Validates the risk context tool returns the full context bundle (critical for LLM-reasoned risk).

```python
"""Tests for risk_tools CrewAI wrappers."""
import json
from unittest.mock import AsyncMock, patch

import pytest


class TestGetPortfolioContextTool:
    """get_portfolio_context_tool must return the full context bundle."""

    @patch("quantstack.crewai_tools.risk_tools._build_portfolio_context")
    def test_returns_full_context_bundle(self, mock_build):
        from quantstack.crewai_tools.risk_tools import get_portfolio_context_tool

        mock_build.return_value = {
            "equity": 20000.0,
            "cash": 15000.0,
            "exposure_by_symbol": {"AAPL": 0.08},
            "daily_pnl_pct": -0.005,
            "volatility": {"vix": 18.5},
            "regime": "trending_up",
        }
        result = get_portfolio_context_tool.run()
        parsed = json.loads(result)
        required_keys = ["equity", "cash", "exposure_by_symbol", "daily_pnl_pct", "volatility", "regime"]
        for key in required_keys:
            assert key in parsed, f"Missing required context key: {key}"
```

### `tests/unit/test_crewai_tools/test_async_bridge.py`

Tests that the async bridging mechanism works correctly with `nest_asyncio`.

```python
"""Tests for the async-to-sync bridge used by all tool wrappers."""
import asyncio

import pytest


class TestAsyncBridge:
    def test_nest_asyncio_applied(self):
        """nest_asyncio must be applied so tools can call asyncio.run inside an existing loop."""
        import nest_asyncio
        # After nest_asyncio.apply(), calling asyncio.run() inside a running loop should not raise
        nest_asyncio.apply()

        async def inner():
            return 42

        # Simulate being inside an event loop (as CrewAI does)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(inner())
            assert result == 42
        finally:
            loop.close()

    def test_run_async_helper_returns_result(self):
        """The shared run_async helper correctly bridges async-to-sync."""
        from quantstack.crewai_tools._async_bridge import run_async

        async def sample():
            return {"status": "ok"}

        result = run_async(sample())
        assert result == {"status": "ok"}

    def test_run_async_propagates_exceptions(self):
        """Exceptions from async functions propagate through the bridge."""
        from quantstack.crewai_tools._async_bridge import run_async

        async def failing():
            raise ValueError("bad input")

        with pytest.raises(ValueError, match="bad input"):
            run_async(failing())
```

### `tests/unit/test_crewai_tools/test_rag_tools.py`

Tests for the new RAG tools (not wrapping existing MCP tools).

```python
"""Tests for RAG tool wrappers."""
import json
from unittest.mock import MagicMock, patch

import pytest


class TestSearchKnowledgeBaseTool:
    @patch("quantstack.crewai_tools.rag_tools._get_chromadb_client")
    def test_returns_results_as_json(self, mock_client):
        from quantstack.crewai_tools.rag_tools import search_knowledge_base_tool

        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "documents": [["Past AAPL trade had 12% return"]],
            "metadatas": [[{"ticker": "AAPL", "date": "2026-03-15"}]],
        }
        mock_client.return_value.get_collection.return_value = mock_collection

        result = search_knowledge_base_tool.run(query="AAPL momentum trades")
        parsed = json.loads(result)
        assert "results" in parsed
        assert len(parsed["results"]) > 0

    @patch("quantstack.crewai_tools.rag_tools._get_chromadb_client")
    def test_handles_empty_collection(self, mock_client):
        from quantstack.crewai_tools.rag_tools import search_knowledge_base_tool

        mock_collection = MagicMock()
        mock_collection.query.return_value = {"documents": [[]], "metadatas": [[]]}
        mock_client.return_value.get_collection.return_value = mock_collection

        result = search_knowledge_base_tool.run(query="nonexistent topic")
        parsed = json.loads(result)
        assert parsed["results"] == []


class TestRememberKnowledgeTool:
    @patch("quantstack.crewai_tools.rag_tools._get_chromadb_client")
    def test_writes_to_correct_collection(self, mock_client):
        from quantstack.crewai_tools.rag_tools import remember_knowledge_tool

        mock_collection = MagicMock()
        mock_client.return_value.get_or_create_collection.return_value = mock_collection

        result = remember_knowledge_tool.run(
            content="AAPL momentum works in trending regimes",
            collection="strategy_knowledge",
            metadata='{"ticker": "AAPL"}'
        )
        mock_collection.add.assert_called_once()
```

---

## Implementation Details

### Directory Structure

```
src/quantstack/crewai_tools/
    __init__.py               # Package init, applies nest_asyncio
    _async_bridge.py          # Shared async-to-sync helper
    signal_tools.py           # Wraps signal.py
    strategy_tools.py         # Wraps strategy.py + _impl.py
    backtest_tools.py         # Wraps backtesting.py
    ml_tools.py               # Wraps ml.py
    risk_tools.py             # Wraps qc_risk.py
    execution_tools.py        # Wraps execution.py
    portfolio_tools.py        # Wraps portfolio.py
    intelligence_tools.py     # Wraps capitulation.py, institutional_accumulation.py, macro_signals.py, cross_domain.py
    coordination_tools.py     # Wraps coordination.py
    research_tools.py         # Wraps qc_research.py
    data_tools.py             # Wraps qc_data.py + qc_indicators.py
    fundamentals_tools.py     # Wraps qc_fundamentals.py
    options_tools.py          # Wraps qc_options.py + options_execution.py
    nlp_tools.py              # Wraps nlp.py
    attribution_tools.py      # Wraps attribution.py
    feedback_tools.py         # Wraps feedback.py
    learning_tools.py         # Wraps learning.py
    meta_tools.py             # Wraps meta.py
    intraday_tools.py         # Wraps intraday.py
    analysis_tools.py         # Wraps analysis.py
    rag_tools.py              # NEW: RAG query and storage (ChromaDB)
    web_tools.py              # NEW: Web search for market-intel
```

### `__init__.py` -- Package Initialization

This file must apply `nest_asyncio` at import time so that all wrapper modules can safely call `asyncio.run()` even when CrewAI has already started an event loop.

```python
"""CrewAI tool wrappers for QuantStack.

Applies nest_asyncio at import time so async-to-sync bridging works
inside CrewAI's internal event loop.
"""
import nest_asyncio

nest_asyncio.apply()
```

### `_async_bridge.py` -- Shared Async-to-Sync Helper

Every wrapper module uses this helper to call the underlying async function. Centralizing it avoids duplicating the event loop management logic in 22 files.

```python
"""Async-to-sync bridge for CrewAI tool wrappers."""
import asyncio


def run_async(coro):
    """Run an async coroutine synchronously.

    Safe to call inside an already-running event loop because nest_asyncio
    is applied at package import time (see __init__.py).
    """
    return asyncio.run(coro)
```

Note: some existing functions in `coordination.py` (e.g., `publish_event`, `poll_events`) are synchronous, not async. The wrapper should call those directly without the async bridge. Each wrapper author must check whether the underlying function is `async def` or `def` and choose accordingly.

### Wrapper Pattern (async underlying function)

Every wrapper that wraps an `async` function follows this exact pattern:

```python
import json

from crewai.tools import tool

from quantstack.crewai_tools._async_bridge import run_async
from quantstack.mcp.tools.signal import get_signal_brief


@tool("Get Signal Brief")
def get_signal_brief_tool(symbol: str) -> str:
    """Generate a comprehensive signal analysis for a stock symbol.

    Returns technical, fundamental, momentum, and regime signals as a
    JSON object. Call this first when evaluating any symbol.
    """
    try:
        result = run_async(get_signal_brief(symbol=symbol))
        return json.dumps(result, default=str)
    except Exception as exc:
        return json.dumps({"error": str(exc), "tool": "get_signal_brief", "symbol": symbol})
```

### Wrapper Pattern (sync underlying function)

For synchronous functions (e.g., those in `coordination.py`):

```python
@tool("Publish Event")
def publish_event_tool(event_type: str, source: str, payload: str = "{}") -> str:
    """Publish an event to the inter-loop event bus.

    Used after significant state changes (strategy promoted, model trained, trade executed).
    payload must be a JSON string.
    """
    try:
        payload_dict = json.loads(payload) if payload else {}
        result = publish_event(event_type=event_type, source=source, payload=payload_dict)
        return json.dumps(result, default=str)
    except Exception as exc:
        return json.dumps({"error": str(exc), "tool": "publish_event"})
```

### Error Handling Convention

Every wrapper catches `Exception` at the outermost level and returns a JSON error object instead of raising. This prevents a single tool failure from crashing the entire CrewAI agent loop. The error JSON always includes:

- `"error"` -- the exception message
- `"tool"` -- which tool failed (for debugging)
- Any input parameters that help reproduce the issue

This is the one place where a broad `except Exception` is acceptable: the wrapper is a boundary layer between two frameworks, and the alternative (an unhandled exception) kills the agent's entire task chain.

### Parameter Type Constraints

CrewAI tools receive all parameters as strings from the LLM. Complex types need special handling:

- **Lists** (e.g., `symbols` in `run_multi_signal_brief`): Accept as comma-separated string, split in the wrapper. Document this in the docstring.
- **Dicts** (e.g., `parameters` in `register_strategy`): Accept as JSON string, parse with `json.loads` in the wrapper. Document this in the docstring.
- **Optional parameters**: Use string defaults like `""` or `"null"`, parse to Python `None` in the wrapper.
- **Numeric parameters**: Accept as string, cast to `int` or `float` in the wrapper.

Example for a tool with complex params:

```python
@tool("Register Strategy")
def register_strategy_tool(
    name: str,
    parameters: str,
    entry_rules: str,
    exit_rules: str,
    description: str = "",
    asset_class: str = "equities",
    symbol: str = "",
) -> str:
    """Register a new trading strategy in the persistent catalog.

    parameters: JSON string of indicator settings, e.g. '{"rsi_period": 14}'
    entry_rules: JSON string of rule list, e.g. '[{"indicator": "rsi_14", "condition": "crosses_below", "value": 30}]'
    exit_rules: JSON string of exit rule list.
    """
    try:
        result = run_async(register_strategy(
            name=name,
            parameters=json.loads(parameters),
            entry_rules=json.loads(entry_rules),
            exit_rules=json.loads(exit_rules),
            description=description,
            asset_class=asset_class,
            symbol=symbol or None,
        ))
        return json.dumps(result, default=str)
    except Exception as exc:
        return json.dumps({"error": str(exc), "tool": "register_strategy"})
```

### Complete Tool-to-Source Mapping

Each wrapper module and the tools it must expose:

| Wrapper Module | Source Module(s) | Tool Functions to Create |
|---|---|---|
| `signal_tools.py` | `signal.py` | `get_signal_brief_tool`, `run_multi_signal_brief_tool` |
| `strategy_tools.py` | `strategy.py`, `_impl.py` | `register_strategy_tool`, `get_strategy_tool`, `list_strategies_tool`, `update_strategy_status_tool` |
| `backtest_tools.py` | `backtesting.py` | `run_backtest_tool`, `run_walkforward_tool`, `run_combinatorial_cv_tool` |
| `ml_tools.py` | `ml.py` | `train_ml_model_tool`, `predict_ml_signal_tool`, `analyze_model_shap_tool`, `check_concept_drift_tool` |
| `risk_tools.py` | `qc_risk.py` | `get_portfolio_context_tool` (returns full context bundle: equity, exposure, P&L, volatility, regime) |
| `execution_tools.py` | `execution.py` | `execute_trade_tool`, `close_position_tool`, `get_fills_tool` |
| `portfolio_tools.py` | `portfolio.py` | `get_portfolio_state_tool`, `get_regime_tool`, `get_daily_equity_tool` |
| `intelligence_tools.py` | `capitulation.py`, `institutional_accumulation.py`, `macro_signals.py`, `cross_domain.py` | `get_capitulation_score_tool`, `get_institutional_accumulation_tool`, `get_credit_market_signals_tool`, `get_cross_domain_intel_tool` |
| `coordination_tools.py` | `coordination.py` | `record_heartbeat_tool`, `publish_event_tool`, `poll_events_tool`, `get_system_status_tool` |
| `research_tools.py` | `qc_research.py` | `compute_information_coefficient_tool`, `compute_alpha_decay_tool`, `compute_deflated_sharpe_ratio_tool` |
| `data_tools.py` | `qc_data.py`, `qc_indicators.py` | `fetch_market_data_tool`, `compute_all_features_tool`, `compute_technical_indicators_tool` |
| `fundamentals_tools.py` | `qc_fundamentals.py` | `get_company_facts_tool`, `get_financial_statements_tool` |
| `options_tools.py` | `qc_options.py`, `options_execution.py` | `get_options_chain_tool`, `compute_greeks_tool`, `submit_options_order_tool` |
| `nlp_tools.py` | `nlp.py` | `analyze_sentiment_tool`, `extract_entities_tool` |
| `attribution_tools.py` | `attribution.py` | `run_attribution_analysis_tool` |
| `feedback_tools.py` | `feedback.py` | `record_tool_error_tool`, `get_open_bugs_tool` |
| `learning_tools.py` | `learning.py` | `run_reflexion_tool`, `calibrate_model_tool` |
| `meta_tools.py` | `meta.py` | `generate_daily_digest_tool`, `auto_promote_eligible_tool` |
| `intraday_tools.py` | `intraday.py` | `get_intraday_signals_tool` |
| `analysis_tools.py` | `analysis.py` | `run_analysis_tool` |
| `rag_tools.py` | N/A (new) | `search_knowledge_base_tool`, `remember_knowledge_tool` |
| `web_tools.py` | N/A (new) | `web_search_tool`, `web_fetch_tool` |

**Total: ~60 tool functions across 22 modules.**

### Risk Gate Transformation

The existing `src/quantstack/execution/risk_gate.py` uses hardcoded thresholds (`position_pct > 0.10`, `daily_loss_pct > 0.02`, etc.). In the CrewAI system, the risk tool wrapper does NOT replicate those checks. Instead:

1. `get_portfolio_context_tool` in `risk_tools.py` aggregates all the data the old risk gate used (portfolio equity, per-symbol exposure, daily P&L, volatility, ADV, regime) and returns it as a JSON string.
2. The risk agent (defined in Section 04) receives this context and reasons about whether the proposed trade is appropriate.
3. The old `risk_gate.py` is preserved as an outer programmatic safety boundary (Section 12 covers the safety envelope). It is NOT deleted.

The `_build_portfolio_context` helper in `risk_tools.py` should query the same data sources that `risk_gate.py` queries today: Alpaca positions, daily P&L from the database, VIX/volatility from signals, and regime from the regime detector.

### New Tools: RAG (`rag_tools.py`)

These do not wrap existing MCP tools. They are new tools backed by ChromaDB (Section 06 covers the RAG pipeline).

- `search_knowledge_base_tool(query, collection="", ticker="", n_results=5)` -- queries ChromaDB, returns top-N documents with metadata as JSON. The `collection` parameter selects which ChromaDB collection to search (`trade_outcomes`, `strategy_knowledge`, `market_research`). If omitted, searches all three.
- `remember_knowledge_tool(content, collection, metadata="")` -- writes a new document to the specified ChromaDB collection. `metadata` is a JSON string with keys like `ticker`, `strategy_name`, `date`.

Both tools need a ChromaDB `HttpClient` connection. Use a module-level lazy singleton (created on first call) pointing to `CHROMADB_URL` env var (default `http://chromadb:8000` for Docker, `http://localhost:8000` for local dev).

### New Tools: Web (`web_tools.py`)

These provide market-intel and community-intel agents with web access.

- `web_search_tool(query)` -- uses a search API (DuckDuckGo or SerpAPI) to find relevant URLs. Returns top 5 results as JSON with title, URL, snippet.
- `web_fetch_tool(url)` -- fetches a URL and returns the text content (HTML stripped). Truncated to 10,000 characters to stay within LLM context limits.

Implementation note: prefer `duckduckgo-search` Python package (no API key required) for `web_search_tool`. For `web_fetch_tool`, use `httpx` with a 10-second timeout and basic HTML-to-text conversion via `html2text` or simple regex stripping.

### Dependency: `nest_asyncio`

Add `nest_asyncio` to the `[crewai]` optional dependency group in `pyproject.toml`. This is required because CrewAI's internals may already be running inside an event loop when tool functions are called. Without `nest_asyncio`, calling `asyncio.run()` inside a running loop raises `RuntimeError: This event loop is already running`.

The `apply()` call happens once at package import time in `__init__.py`. It patches the global event loop policy and is safe to call multiple times.

### `json.dumps` with `default=str`

All wrappers use `json.dumps(result, default=str)` rather than plain `json.dumps(result)`. This handles datetime objects, UUID objects, Decimal values, and other non-JSON-native types that frequently appear in database query results. Without this, wrappers would crash on any result containing these types.

---

## Implementation Checklist

1. Create `src/quantstack/crewai_tools/__init__.py` with `nest_asyncio.apply()`
2. Create `src/quantstack/crewai_tools/_async_bridge.py` with the `run_async` helper
3. For each of the 22 wrapper modules:
   a. Read the source module to identify the public functions and their signatures
   b. Check whether each function is `async def` or `def`
   c. Write the wrapper with correct parameter types (strings for complex types, with JSON parsing)
   d. Include a descriptive docstring (this is what the LLM sees)
   e. Wrap the body in try/except returning error JSON on failure
4. Write tests for each module following the pattern in `test_signal_tools.py`
5. Run the contract test (`test_tool_wrapper_contract.py`) to verify all wrappers meet structural requirements
6. Add `nest_asyncio`, `duckduckgo-search`, and `html2text` to the `[crewai]` dependency group in `pyproject.toml`
