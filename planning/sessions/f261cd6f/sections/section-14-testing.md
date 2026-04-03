# Section 14: Testing — Final Validation Suite

## Purpose

This section defines the final validation layer for the CrewAI-to-LangGraph migration. It covers six categories of tests that run after all graphs, runners, and safety checks are implemented (sections 01-13). These tests verify the system works end-to-end, meets performance budgets, produces correct observability data, and does not regress from the pre-migration behavior.

**Dependencies**: Section 11 (Runners), Section 13 (Risk & Safety). All graphs must be buildable and invocable before these tests pass.

---

## Test Categories

1. **Regression tests** — replay captured CrewAI cycle I/O through new LangGraph graphs
2. **Timing benchmarks** — graph invocations complete within cycle budgets
3. **LangFuse trace structure assertions** — every node, tool call, and LLM call is traced correctly
4. **End-to-end smoke tests** — build and invoke each graph with minimal state
5. **Config validation tests** — YAML agent configs parse correctly and cross-reference tool registry
6. **Tool contract tests** — all LLM-facing and node-callable tools meet interface contracts

---

## 1. Regression Tests

### Background

Before migration, capture 3-5 real cycle I/O pairs from each CrewAI crew (research, trading, supervisor). Store these as JSON fixtures in `tests/fixtures/regression/`. Each fixture contains:

- `input_state`: the initial state passed to the graph (regime, portfolio context, cycle number, etc.)
- `expected_outputs`: key fields from the final state (decisions made, orders placed, strategies registered)
- `metadata`: timestamp, market conditions, crew version

The regression tests replay the same input states through the new LangGraph implementations and assert that outputs are equivalent (or document accepted divergences).

### File: `tests/integration/test_regression_replay.py`

```python
"""Regression tests: replay captured CrewAI I/O through LangGraph graphs.

Each fixture in tests/fixtures/regression/ contains an input_state and
expected_outputs from a real CrewAI cycle. We feed the same inputs into
the new graphs and verify outputs match.
"""

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "regression"


def load_fixtures(graph_name: str) -> list[dict]:
    """Load all regression fixtures for a given graph."""
    graph_dir = FIXTURE_DIR / graph_name
    if not graph_dir.is_dir():
        return []
    return [
        json.loads(f.read_text())
        for f in sorted(graph_dir.glob("*.json"))
    ]


@pytest.fixture()
def mock_llm():
    """Mock LLM that returns deterministic responses for regression replay."""
    # Implementation: return canned responses keyed by prompt content
    ...


class TestResearchRegression:
    """Replay research cycle I/O through ResearchGraph."""

    @pytest.mark.parametrize(
        "fixture",
        load_fixtures("research"),
        ids=lambda f: f.get("metadata", {}).get("fixture_id", "unknown"),
    )
    async def test_replay_research_cycle(self, fixture, mock_llm):
        """Research graph produces equivalent output for captured input."""
        ...

    def test_at_least_3_fixtures_exist(self):
        """Ensure we have minimum regression coverage."""
        fixtures = load_fixtures("research")
        assert len(fixtures) >= 3, (
            f"Need >= 3 research regression fixtures, found {len(fixtures)}. "
            "Capture them before migration with scripts/capture_regression_baseline.py"
        )


class TestTradingRegression:
    """Replay trading cycle I/O through TradingGraph."""

    @pytest.mark.parametrize(
        "fixture",
        load_fixtures("trading"),
        ids=lambda f: f.get("metadata", {}).get("fixture_id", "unknown"),
    )
    async def test_replay_trading_cycle(self, fixture, mock_llm):
        """Trading graph produces equivalent output for captured input."""
        ...

    def test_at_least_3_fixtures_exist(self):
        fixtures = load_fixtures("trading")
        assert len(fixtures) >= 3


class TestSupervisorRegression:
    """Replay supervisor cycle I/O through SupervisorGraph."""

    @pytest.mark.parametrize(
        "fixture",
        load_fixtures("supervisor"),
        ids=lambda f: f.get("metadata", {}).get("fixture_id", "unknown"),
    )
    async def test_replay_supervisor_cycle(self, fixture, mock_llm):
        """Supervisor graph produces equivalent output for captured input."""
        ...
```

### Regression Fixture Capture Script: `scripts/capture_regression_baseline.py`

This script must be run against the **existing CrewAI system** before migration begins. It:

1. Constructs the same initial state the runner builds (regime, portfolio, cycle number)
2. Calls `crew.kickoff()` and records the structured output
3. Saves `{input_state, expected_outputs, metadata}` as a JSON fixture
4. Stores fixtures in `tests/fixtures/regression/{graph_name}/cycle_{n}.json`

The script is a one-time capture tool. It does not need tests itself. Fixture structure:

```json
{
  "metadata": {
    "fixture_id": "research_cycle_001",
    "captured_at": "2026-04-02T10:00:00Z",
    "crew_version": "crewai",
    "market_regime": "trending_up"
  },
  "input_state": {
    "cycle_number": 42,
    "regime": "trending_up"
  },
  "expected_outputs": {
    "selected_domain": "swing",
    "hypothesis_generated": true,
    "validation_passed": true,
    "strategy_registered": true,
    "errors": []
  }
}
```

### Accepted Divergences

Some outputs will differ between CrewAI and LangGraph because LLM reasoning is non-deterministic. The regression tests should assert on **structural correctness** (right fields present, right types, right flow), not on exact LLM prose. Specifically:

- `hypothesis` text will differ -- assert it is a non-empty string
- `daily_plan` text will differ -- assert it contains expected sections
- `decisions` list entries will differ in `reasoning` -- assert the `action` field matches
- Numeric values (position sizes, confidence scores) may differ slightly -- use approximate comparisons

---

## 2. Timing Benchmarks

### Background

Each graph invocation must complete within its cycle interval:

| Graph | Budget | Interval context |
|-------|--------|-----------------|
| Trading | < 5 min (300s) | Market hours cycle |
| Research | < 10 min (600s) | Market hours cycle |
| Supervisor | < 5 min (300s) | Always |

LangGraph adds overhead from state serialization, checkpointing (PostgreSQL writes after every node), and callback dispatch (LangFuse). This overhead must be measured.

### File: `tests/benchmarks/test_graph_timing.py`

```python
"""Timing benchmarks: verify graph invocations complete within cycle budgets.

These tests use mock LLMs with realistic response delays to measure
the overhead of LangGraph orchestration, checkpointing, and tracing.
"""

import time

import pytest


# Mock LLM adds realistic latency per call (simulates API round-trip)
MOCK_LLM_LATENCY_SECONDS = 0.5

# Budgets in seconds — leave 20% headroom below cycle interval
TIMING_BUDGETS = {
    "trading": 240,     # 4 min (cycle is 5 min)
    "research": 480,    # 8 min (cycle is 10 min)
    "supervisor": 240,  # 4 min (cycle is 5 min)
}


class TestTradingGraphTiming:
    """Trading graph completes within budget."""

    @pytest.mark.slow
    async def test_trading_graph_within_budget(self, mock_checkpointer, mock_llm):
        """Full trading graph invocation finishes under 4 minutes with mock LLM."""
        ...


class TestResearchGraphTiming:
    """Research graph completes within budget."""

    @pytest.mark.slow
    async def test_research_graph_within_budget(self, mock_checkpointer, mock_llm):
        """Full research graph invocation finishes under 8 minutes with mock LLM."""
        ...


class TestSupervisorGraphTiming:
    """Supervisor graph completes within budget."""

    @pytest.mark.slow
    async def test_supervisor_graph_within_budget(self, mock_checkpointer, mock_llm):
        """Full supervisor graph invocation finishes under 4 minutes with mock LLM."""
        ...
```

### Implementation Notes

- Use `MemorySaver` (in-memory checkpointer) for benchmarks to isolate graph overhead from PostgreSQL latency. Optionally run a separate benchmark with `AsyncPostgresSaver` to measure DB overhead.
- Mock LLMs should add a configurable delay (default 0.5s) to simulate real API latency. This ensures the benchmark reflects realistic conditions rather than instant responses.
- Mark these tests with `@pytest.mark.slow` and exclude from the default test run. Run them explicitly: `pytest tests/benchmarks/ -m slow`.
- Record wall-clock time via `time.monotonic()` before and after `await graph.ainvoke(...)`.
- If timing tests fail consistently, the cause is likely one of: (a) excessive checkpointing, (b) serial tool calls that should be parallel, (c) LangFuse callback overhead. Profile with `cProfile` or LangFuse spans to identify the bottleneck.

---

## 3. LangFuse Trace Structure Assertions

### Background

LangFuse traces are the audit trail for a capital-handling system. Every graph invocation must produce traces with the correct structure: one span per node, tool invocations nested under their parent node, correct session and thread IDs.

### File: `tests/integration/test_langfuse_traces.py`

```python
"""LangFuse trace structure assertions.

Verifies that graph invocations produce the expected trace hierarchy.
Uses a mock LangFuse callback handler that captures spans in-memory.
"""

from unittest.mock import MagicMock, patch

import pytest


class FakeCallbackHandler:
    """Captures trace events for assertion without a running LangFuse server."""

    def __init__(self):
        self.spans: list[dict] = []
        self.session_id: str | None = None
        self.thread_id: str | None = None

    # Implement the CallbackHandler interface methods to capture spans
    ...


class TestTradingGraphTraces:
    """Trading graph produces correct trace structure."""

    async def test_every_node_appears_as_span(self, fake_handler):
        """Each node in the trading graph emits a trace span."""
        ...

    async def test_tool_calls_nested_under_parent_node(self, fake_handler):
        """Tool invocations appear as children of the node that called them."""
        ...

    async def test_session_id_matches_config(self, fake_handler):
        """Session ID in traces matches the one passed in graph config."""
        ...

    async def test_thread_id_matches_config(self, fake_handler):
        """Thread ID in traces matches the one passed in graph config."""
        ...


class TestResearchGraphTraces:
    """Research graph produces correct trace structure."""

    async def test_every_node_appears_as_span(self, fake_handler):
        ...

    async def test_conditional_edge_traced(self, fake_handler):
        """When signal_validation fails, trace shows routing to END, not backtest."""
        ...


class TestSupervisorGraphTraces:
    """Supervisor graph produces correct trace structure."""

    async def test_every_node_appears_as_span(self, fake_handler):
        ...
```

### Implementation Notes

- Use a `FakeCallbackHandler` that records spans to a list. Pass it in the `callbacks` config to `graph.ainvoke()`.
- For trading graph, expected node span names: `safety_check`, `daily_plan`, `position_review`, `entry_scan`, `execute_exits`, `merge_parallel`, `risk_sizing`, `portfolio_review`, `options_analysis`, `execute_entries`, `reflection`.
- For research graph, expected node span names: `context_load`, `domain_selection`, `hypothesis_generation`, `signal_validation`, `backtest_validation`, `ml_experiment`, `strategy_registration`, `knowledge_update`.
- For supervisor graph, expected node span names: `health_check`, `diagnose_issues`, `execute_recovery`, `strategy_lifecycle`, `scheduled_tasks`.
- Verify nested structure: when an agent node calls a tool, the tool span's `parent_run_id` matches the node span's `run_id`.

---

## 4. End-to-End Smoke Tests

### Background

Replace the existing `tests/integration/test_e2e_smoke.py` which tests CrewAI crew assembly. The new version builds each LangGraph graph and invokes it with minimal test state.

### File: `tests/integration/test_e2e_smoke.py`

```python
"""E2E smoke tests: build and invoke each graph with minimal state.

Replaces the old CrewAI crew assembly test. Verifies that graphs compile,
accept initial state, execute all nodes, and return a valid final state.
"""

import pytest
from langgraph.checkpoint.memory import MemorySaver


class TestTradingGraphSmoke:
    """Trading graph builds and completes a minimal cycle."""

    async def test_graph_compiles(self, mock_config_watcher):
        """build_trading_graph() returns a compiled graph."""
        from quantstack.graphs.trading.graph import build_trading_graph
        graph = build_trading_graph(mock_config_watcher, MemorySaver())
        assert graph is not None

    async def test_graph_completes_without_error(self, mock_config_watcher, mock_llm):
        """Graph invocation with minimal state completes without raising."""
        from quantstack.graphs.trading.graph import build_trading_graph
        graph = build_trading_graph(mock_config_watcher, MemorySaver())
        initial_state = {
            "cycle_number": 1,
            "regime": "trending_up",
            "portfolio_context": {"positions": [], "cash_available": 25000.0},
        }
        result = await graph.ainvoke(
            initial_state,
            config={"configurable": {"thread_id": "smoke-test-001"}},
        )
        assert "errors" in result
        assert isinstance(result["errors"], list)

    async def test_halted_system_routes_to_end(self, mock_config_watcher, mock_llm):
        """When system status is halted, graph terminates at safety_check."""
        ...


class TestResearchGraphSmoke:
    """Research graph builds and completes a minimal cycle."""

    async def test_graph_compiles(self, mock_config_watcher):
        from quantstack.graphs.research.graph import build_research_graph
        graph = build_research_graph(mock_config_watcher, MemorySaver())
        assert graph is not None

    async def test_graph_completes_without_error(self, mock_config_watcher, mock_llm):
        from quantstack.graphs.research.graph import build_research_graph
        graph = build_research_graph(mock_config_watcher, MemorySaver())
        initial_state = {
            "cycle_number": 1,
            "regime": "trending_up",
        }
        result = await graph.ainvoke(
            initial_state,
            config={"configurable": {"thread_id": "smoke-test-002"}},
        )
        assert "errors" in result

    async def test_signal_validation_fail_routes_to_end(self, mock_config_watcher, mock_llm):
        """When signal_validation fails, graph terminates without backtesting."""
        ...


class TestSupervisorGraphSmoke:
    """Supervisor graph builds and completes a minimal cycle."""

    async def test_graph_compiles(self, mock_config_watcher):
        from quantstack.graphs.supervisor.graph import build_supervisor_graph
        graph = build_supervisor_graph(mock_config_watcher, MemorySaver())
        assert graph is not None

    async def test_graph_completes_without_error(self, mock_config_watcher, mock_llm):
        from quantstack.graphs.supervisor.graph import build_supervisor_graph
        graph = build_supervisor_graph(mock_config_watcher, MemorySaver())
        initial_state = {"cycle_number": 1}
        result = await graph.ainvoke(
            initial_state,
            config={"configurable": {"thread_id": "smoke-test-003"}},
        )
        assert "errors" in result
```

### Shared Test Fixtures: `tests/integration/conftest.py`

Add these fixtures to the existing integration conftest:

```python
@pytest.fixture()
def mock_config_watcher():
    """ConfigWatcher with test YAML agent configs loaded."""
    ...

@pytest.fixture()
def mock_llm():
    """Mock BaseChatModel that returns deterministic responses."""
    ...

@pytest.fixture()
def mock_checkpointer():
    """In-memory checkpointer (MemorySaver) for test isolation."""
    from langgraph.checkpoint.memory import MemorySaver
    return MemorySaver()
```

---

## 5. Config Validation Tests

### Background

Replace `test_crew_workflows.py` (if it exists) with tests that validate the new YAML-based agent configuration system. These tests ensure YAML files parse correctly, all required fields are present, tool references resolve against the tool registry, and LLM tiers are valid.

### File: `tests/unit/test_agent_config_validation.py`

```python
"""Config validation tests: YAML agent configs parse and cross-reference correctly.

Validates every agents.yaml file in src/quantstack/graphs/*/config/.
"""

import pathlib

import pytest
import yaml

from quantstack.graphs.config import AgentConfig, load_agent_configs


GRAPHS_DIR = pathlib.Path(__file__).resolve().parents[2] / "src" / "quantstack" / "graphs"
VALID_TIERS = {"heavy", "medium", "light"}


def _all_yaml_configs() -> list[pathlib.Path]:
    """Discover all agents.yaml files across graph modules."""
    return sorted(GRAPHS_DIR.glob("*/config/agents.yaml"))


class TestYamlParsing:
    """Every agents.yaml file parses into valid AgentConfig objects."""

    @pytest.mark.parametrize("yaml_path", _all_yaml_configs(), ids=lambda p: p.parent.parent.name)
    def test_yaml_parses_without_error(self, yaml_path):
        configs = load_agent_configs(yaml_path)
        assert len(configs) > 0

    @pytest.mark.parametrize("yaml_path", _all_yaml_configs(), ids=lambda p: p.parent.parent.name)
    def test_all_agents_have_required_fields(self, yaml_path):
        configs = load_agent_configs(yaml_path)
        for name, config in configs.items():
            assert config.role, f"{name} missing role"
            assert config.goal, f"{name} missing goal"
            assert config.backstory, f"{name} missing backstory"
            assert config.llm_tier in VALID_TIERS, f"{name} has invalid llm_tier: {config.llm_tier}"
            assert config.max_iterations > 0, f"{name} has non-positive max_iterations"
            assert config.timeout_seconds > 0, f"{name} has non-positive timeout_seconds"


class TestToolRegistryCrossRef:
    """Tool names in agent configs resolve against TOOL_REGISTRY."""

    @pytest.mark.parametrize("yaml_path", _all_yaml_configs(), ids=lambda p: p.parent.parent.name)
    def test_all_tool_refs_resolve(self, yaml_path):
        from quantstack.tools.langchain import TOOL_REGISTRY
        configs = load_agent_configs(yaml_path)
        for name, config in configs.items():
            for tool_name in config.tools:
                assert tool_name in TOOL_REGISTRY, (
                    f"Agent '{name}' references tool '{tool_name}' not in TOOL_REGISTRY"
                )


class TestNoDuplicateAgentNames:
    """No two agents share a name within the same YAML file."""

    @pytest.mark.parametrize("yaml_path", _all_yaml_configs(), ids=lambda p: p.parent.parent.name)
    def test_no_duplicates(self, yaml_path):
        raw = yaml.safe_load(yaml_path.read_text())
        # YAML dicts silently drop duplicates; check by counting keys in raw text
        text = yaml_path.read_text()
        top_level_keys = [
            line.split(":")[0].strip()
            for line in text.splitlines()
            if line and not line.startswith(" ") and not line.startswith("#") and ":" in line
        ]
        assert len(top_level_keys) == len(set(top_level_keys)), (
            f"Duplicate agent names in {yaml_path}"
        )
```

---

## 6. Tool Contract Tests

### Background

Replace `tests/unit/test_crewai_tools/test_tool_wrapper_contract.py` which validates CrewAI `Tool` instances. The new tests validate two tool categories:

- **LLM-facing tools** in `src/quantstack/tools/langchain/`: must have `@tool` decorator, non-empty description, be async, return `str`
- **Node-callable functions** in `src/quantstack/tools/functions/`: must be async, have type hints, no `@tool` decorator

### File: `tests/unit/test_tool_contracts.py`

```python
"""Tool contract tests: verify all tools meet interface requirements.

Replaces test_tool_wrapper_contract.py (CrewAI era).
Discovers all tool modules via pkgutil and validates structural contracts.
"""

import asyncio
import importlib
import inspect
import pkgutil

import pytest

import quantstack.tools.langchain as langchain_pkg
import quantstack.tools.functions as functions_pkg


def _discover_tools(package) -> list[tuple[str, str, object]]:
    """Yield (module_name, attr_name, obj) for public callables in a package."""
    results = []
    for _, mod_name, _ in pkgutil.iter_modules(package.__path__):
        if mod_name.startswith("_"):
            continue
        mod = importlib.import_module(f"{package.__name__}.{mod_name}")
        for attr_name in dir(mod):
            if attr_name.startswith("_"):
                continue
            obj = getattr(mod, attr_name)
            if callable(obj) and not isinstance(obj, type):
                results.append((mod_name, attr_name, obj))
    return results


_langchain_tools = _discover_tools(langchain_pkg)
_function_tools = _discover_tools(functions_pkg)


class TestLangchainToolContracts:
    """All LLM-facing tools in tools/langchain/ meet the @tool contract."""

    @pytest.mark.parametrize(
        "mod,name,tool",
        _langchain_tools,
        ids=[f"{m}.{n}" for m, n, _ in _langchain_tools],
    )
    def test_has_nonempty_description(self, mod, name, tool):
        assert getattr(tool, "description", None), f"{mod}.{name} has no description"

    @pytest.mark.parametrize(
        "mod,name,tool",
        _langchain_tools,
        ids=[f"{m}.{n}" for m, n, _ in _langchain_tools],
    )
    def test_is_async(self, mod, name, tool):
        """LangChain @tool must wrap an async function for LangGraph compatibility."""
        coroutine = getattr(tool, "coroutine", None)
        assert coroutine is not None or asyncio.iscoroutinefunction(tool), (
            f"{mod}.{name} must be async"
        )


class TestFunctionToolContracts:
    """All node-callable functions in tools/functions/ meet the interface contract."""

    @pytest.mark.parametrize(
        "mod,name,func",
        _function_tools,
        ids=[f"{m}.{n}" for m, n, _ in _function_tools],
    )
    def test_is_async(self, mod, name, func):
        assert asyncio.iscoroutinefunction(func), f"{mod}.{name} must be async"

    @pytest.mark.parametrize(
        "mod,name,func",
        _function_tools,
        ids=[f"{m}.{n}" for m, n, _ in _function_tools],
    )
    def test_has_type_hints(self, mod, name, func):
        sig = inspect.signature(func)
        assert sig.return_annotation != inspect.Parameter.empty, (
            f"{mod}.{name} missing return type hint"
        )

    @pytest.mark.parametrize(
        "mod,name,func",
        _function_tools,
        ids=[f"{m}.{n}" for m, n, _ in _function_tools],
    )
    def test_no_tool_decorator(self, mod, name, func):
        """Node-callable functions must NOT have @tool decorator."""
        assert not hasattr(func, "description") or not hasattr(func, "args_schema"), (
            f"{mod}.{name} appears to have @tool decorator -- should be in tools/langchain/"
        )


class TestNoCrewaiImports:
    """No tool file imports from crewai or crewai_compat after migration."""

    @pytest.mark.parametrize(
        "pkg",
        [langchain_pkg, functions_pkg],
        ids=["langchain", "functions"],
    )
    def test_no_crewai_imports(self, pkg):
        import ast
        pkg_path = pkg.__path__[0]
        for _, mod_name, _ in pkgutil.iter_modules([pkg_path]):
            mod_file = f"{pkg_path}/{mod_name}.py"
            tree = ast.parse(open(mod_file).read())
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    module = getattr(node, "module", "") or ""
                    names = [a.name for a in getattr(node, "names", [])]
                    combined = module + " ".join(names)
                    assert "crewai" not in combined.lower(), (
                        f"{mod_name}.py still imports crewai: {combined}"
                    )
```

---

## Files to Create

| File | Purpose |
|------|---------|
| `tests/integration/test_regression_replay.py` | Regression tests replaying captured CrewAI I/O |
| `tests/benchmarks/test_graph_timing.py` | Timing benchmarks for cycle budget enforcement |
| `tests/integration/test_langfuse_traces.py` | LangFuse trace structure assertions |
| `tests/integration/test_e2e_smoke.py` | End-to-end smoke tests (replaces existing CrewAI version) |
| `tests/unit/test_agent_config_validation.py` | YAML agent config parsing and cross-referencing |
| `tests/unit/test_tool_contracts.py` | Tool interface contract validation (replaces CrewAI version) |
| `tests/fixtures/regression/research/` | Directory for research regression fixtures (JSON) |
| `tests/fixtures/regression/trading/` | Directory for trading regression fixtures (JSON) |
| `tests/fixtures/regression/supervisor/` | Directory for supervisor regression fixtures (JSON) |
| `scripts/capture_regression_baseline.py` | One-time script to capture CrewAI cycle I/O as fixtures |

## Files to Delete

| File | Reason |
|------|--------|
| `tests/unit/test_crewai_tools/test_tool_wrapper_contract.py` | Replaced by `test_tool_contracts.py` |
| `tests/unit/test_crewai_tools/` (entire directory) | All CrewAI tool tests replaced by new tool contract and integration tests |

## Files to Modify

| File | Change |
|------|--------|
| `tests/integration/conftest.py` | Add `mock_config_watcher`, `mock_llm`, `mock_checkpointer` fixtures |
| `tests/integration/test_e2e_smoke.py` | Complete rewrite from CrewAI crew assembly to LangGraph graph invocation |
| `pyproject.toml` | Add `slow` marker to `[tool.pytest.ini_options]` for benchmark tests |

---

## Pytest Configuration

Add the `slow` marker so benchmarks can be excluded from CI fast runs:

```toml
# In pyproject.toml [tool.pytest.ini_options]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
]
```

Default CI command: `uv run pytest tests/ -m "not slow"`

Benchmark-only command: `uv run pytest tests/benchmarks/ -m slow`

---

## Execution Checklist

1. Capture regression baselines by running `scripts/capture_regression_baseline.py` against the live CrewAI system (must happen before any migration code lands)
2. Create fixture directories under `tests/fixtures/regression/`
3. Write the 6 test files listed above
4. Add shared fixtures to `tests/integration/conftest.py`
5. Add `slow` marker to `pyproject.toml`
6. Delete `tests/unit/test_crewai_tools/` directory
7. Run `uv run pytest tests/ -m "not slow"` -- all non-benchmark tests should pass
8. Run `uv run pytest tests/benchmarks/ -m slow` -- benchmark tests should pass with mock LLMs
9. Verify LangFuse trace tests pass with the `FakeCallbackHandler`
10. Verify regression tests pass (or document accepted divergences in fixture metadata)
