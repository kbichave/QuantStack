# Research Findings: CrewAI to LangGraph Migration

## Part 1: Codebase Analysis

### CrewAI Footprint Summary

The codebase has a **clean, well-isolated CrewAI footprint**. Core trading logic, MCP tools, safety boundaries, and schemas are pure Python.

### Crew Definitions (3 crews)

**ResearchCrew** (`src/quantstack/crews/research/crew.py`):
- Process: `hierarchical` (manager = quant_researcher, workers = ml_scientist, strategy_rd, community_intel)
- 4 agents, 8 tasks (load_context → domain_selection → hypothesis → signal_validation → backtest → ml_experiment → strategy_registration → knowledge_update)
- YAML configs: `config/agents.yaml`, `config/tasks.yaml`
- Manager pattern: quant_researcher orchestrates 3 worker agents

**TradingCrew** (`src/quantstack/crews/trading/crew.py`):
- Process: `sequential` (flat, no manager)
- 10 agents: daily_planner, position_monitor, trade_debater, risk_analyst, fund_manager, options_analyst, earnings_analyst, market_intel, trade_reflector, executor
- 11 tasks with dependency DAG and async_execution on some (position_review, entry_scan)
- Most complex crew with branching dependencies

**SupervisorCrew** (`src/quantstack/crews/supervisor/crew.py`):
- Process: `sequential`
- 3 agents: health_monitor, self_healer, strategy_promoter
- 5 tasks: health_check → diagnose_issues → execute_recovery, strategy_lifecycle, scheduled_tasks

### LLM Provider Tier System

| Tier | Used By | Example Model |
|------|---------|---------------|
| heavy | fund-manager, quant-researcher, trade-debater, risk, ml-scientist, strategy-rd, options-analyst | bedrock/anthropic.claude-sonnet-4-* |
| medium | earnings-analyst, position-monitor, daily-planner, market-intel, trade-reflector | bedrock/anthropic.claude-sonnet-4-* |
| light | community-intel, supervisor, execution agents | bedrock/anthropic.claude-haiku-4-5-* |
| embedding | memory, RAG | ollama/mxbai-embed-large |

Fallback chain: bedrock → anthropic → openai → ollama. Functions: `get_model(tier)` returns string, `get_model_with_fallback(tier)` tries chain.

### CrewAI Tool Wrappers (25 files in `src/quantstack/crewai_tools/`)

All follow this pattern:
```python
from crewai.tools import tool
from quantstack.crewai_tools._async_bridge import run_async

@tool("Tool Name")
def tool_name_tool(arg: str) -> str:
    """Description for LLM."""
    result = run_async(mcp_tool_function(arg))
    return json.dumps(result, default=str)
```

Async bridge (`_async_bridge.py`): uses `nest_asyncio.apply()` + `loop.run_until_complete()` to call async MCP tools from CrewAI's sync context.

Tool domains: signal, risk, data, portfolio, rag, execution, strategy, analysis, backtest, ml, options, fundamentals, research, attribution, web, coordination, intelligence, nlp, intraday, learning, meta, feedback.

### Runners (3 continuous loops)

All follow the same pattern:
```python
def factory():
    return SomeCrew().crew()
run_loop(factory, shutdown, crew_name="name")
```

`run_loop()` calls `crew.kickoff()` each cycle, writes heartbeat + checkpoint to PostgreSQL.

Cycle intervals: trading=5min (market hours), research=10min, supervisor=5min always.

### Observability

- `instrumentation.py`: Calls `CrewAIInstrumentor().instrument()` — **CrewAI-specific, must replace**
- `crew_tracing.py`: Custom Langfuse trace helpers (provider_failover, strategy_lifecycle, self_healing, capital_allocation, safety_boundary) — **Pure Langfuse, keep and rename**
- `tracing.py`: Lazy-init Langfuse client, TracingSpan wrapper — **Pure Langfuse, keep**
- `flush_util.py`: Langfuse flush — **Keep**

### Testing

- Framework: **pytest**
- `test_crew_workflows.py`: Validates YAML configs (agents.yaml/tasks.yaml) have required fields
- `test_crewai_risk_safety.py`: Tests SafetyGate — **pure Python, keep**
- `test_crewai_tools/`: Tests `@tool` decorator contract, async bridge — **must rewrite**
- `test_agent_definitions.py`, `test_scaffolding.py`: Reference CrewAI structure

### Pure Python (NO CrewAI dependency — keep unchanged)

- `crews/risk/safety_gate.py` — SafetyGateLimits, RiskDecision, RiskVerdict, SafetyGate
- `crews/decoder_crew.py` — Pure Python strategy decoder (4 ICs + synthesizer)
- `crews/schemas.py` — Re-exports from shared.schemas (Pydantic)
- `crews/registry.py` — IC_REGISTRY, POD_MANAGER_REGISTRY metadata
- `crewai_compat.py` — BaseTool stubs used by ~50 tool files in tools/mcp_bridge

### Dependencies (pyproject.toml)

CrewAI is an optional dependency group:
```toml
crewai = [
    "crewai[tools]>=0.100.0",
    "chromadb>=0.5.0",
    "ollama>=0.4.0",
    "nest-asyncio>=1.6.0",
    "openinference-instrumentation-crewai>=0.1.0",
    "duckduckgo-search>=6.0.0",
]
```

### Docker

3 crew services (trading-crew, research-crew, supervisor-crew) depend on postgres, ollama, chromadb, langfuse.

### Migration Impact Summary

| Category | Files | Lines | Action |
|----------|-------|-------|--------|
| Crew definitions | 3 | ~250 | Rewrite as StateGraphs |
| Tool wrappers | 25 | ~2000 | Replace @tool with LangChain @tool or direct |
| Runners | 3 | ~50 lines changed | Adapt kickoff → graph.invoke |
| Instrumentation | 1 | ~50 | Replace CrewAIInstrumentor with Langfuse callback |
| Tests | ~6 | ~200 | Rewrite for LangGraph patterns |
| Pure Python (keep) | 50+ | ~6000+ | Unchanged |

---

## Part 2: LangGraph Best Practices (Web Research)

### 1. State Design with TypedDict

The fundamental building block — typed state flows through the graph:

```python
from typing import TypedDict, List, Annotated
import operator

class ResearchState(TypedDict):
    messages: List[str]
    context: str
    hypothesis: str
    validation_result: str
    backtest_output: str
    strategy_id: str
    iteration: int
    error_log: Annotated[List[str], operator.add]  # append-only reducer
```

`Annotated[..., operator.add]` tells LangGraph to **append** rather than overwrite — critical for accumulating data across nodes.

### 2. Sequential Pipeline Pattern

```python
workflow = StateGraph(ResearchState)
workflow.add_node("context_load", context_load_node)
workflow.add_node("hypothesis", hypothesis_node)
workflow.add_edge(START, "context_load")
workflow.add_edge("context_load", "hypothesis")
# ... etc
app = workflow.compile()
```

Each node is a plain function: `def node(state: State) -> dict` returning only changed fields.

### 3. Conditional Branching (Risk Gate)

```python
def risk_gate_router(state: TradingState) -> str:
    if state["risk_verdict"].approved:
        return "execute"
    return "reject"

workflow.add_conditional_edges("risk_check", risk_gate_router,
    {"execute": "execute_entries", "reject": END})
```

### 4. Sub-Graph Composition

Compiled subgraphs can be added as nodes to parent graphs. Two patterns:
- **Shared state**: `parent.add_node("research", research_subgraph)` — same TypedDict
- **Different schemas**: Wrapper function transforms state between schemas

### 5. Agent-as-Node vs Tool-as-Node

- **Agent-as-node**: LLM call with system prompt + tools. For reasoning-heavy steps (hypothesis, trade analysis).
- **Tool-as-node**: Pure Python function. For deterministic computation (risk calc, backtest, data fetch).
- **Recommendation**: Use tool-as-node wherever possible to minimize token cost and latency.

### 6. Supervisor/Router Pattern (replaces Process.hierarchical)

```python
def supervisor(state: AgentState) -> str:
    # Deterministic routing replaces manager LLM delegation
    return "swing_researcher" if state["task_type"] == "swing" else "options_researcher"

workflow.add_conditional_edges("supervisor", supervisor, {...})
```

### 7. LangFuse Integration

**Setup**: `from langfuse.langchain import CallbackHandler`

**Usage**:
```python
result = graph.invoke(state, config={
    "callbacks": [langfuse_handler],
    "metadata": {
        "langfuse_session_id": "trading-loop-2026-04-02",
        "langfuse_tags": ["research", "swing"]
    }
})
```

- Tool calls, LLM invocations, and node executions are **automatically traced**
- Use shared `trace_id` to consolidate subgraph traces
- Call `get_client().shutdown()` on process exit to flush

### 8. PostgreSQL Checkpointing

**Package**: `langgraph-checkpoint-postgres` (v3.0.5, MIT license)

**Setup**:
```python
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

async with AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer:
    await checkpointer.setup()  # Creates tables (idempotent)
    graph = workflow.compile(checkpointer=checkpointer)
    result = await graph.ainvoke(state, config={"configurable": {"thread_id": "research-001"}})
```

- Creates 4 tables: `checkpoints`, `checkpoint_blobs`, `checkpoint_writes`, `checkpoint_migrations`
- Coexists with existing application tables in the `quantstack` database
- Requires `psycopg` (v3, not psycopg2) with `autocommit=True` and `dict_row` factory
- **Performance**: Keep state lean (IDs + summaries), store heavy data in app tables

### 9. CrewAI → LangGraph Concept Mapping

| CrewAI | LangGraph | Notes |
|--------|-----------|-------|
| Agent (role, goal, backstory) | Node function with system prompt | Extract prompt from backstory |
| Task (description, expected_output) | State transitions + edges | Output becomes state field |
| Crew (agents, tasks, process) | StateGraph (nodes, edges, compile) | Orchestration is explicit topology |
| Process.sequential | Linear edge chain | `add_edge("a", "b")` |
| Process.hierarchical | Conditional edges + supervisor | Manager replaced by routing function |
| Implicit data flow (conversation) | Explicit TypedDict state | Biggest migration effort |
| memory=True | PostgresSaver checkpointer | Explicit state persistence |
| crew.kickoff() | graph.invoke(state) | Execution entry point |

### 10. Migration Strategy

1. **State schema design first** — audit what data flows between agents
2. **Hybrid approach available** — wrap existing CrewAI crews in LangGraph nodes during transition
3. **Incremental migration** — orchestration layer first, then individual agent nodes
4. **Test each node in isolation** — pure functions of state are easy to unit test

**Sources**:
- LangGraph Documentation (docs.langchain.com)
- Langfuse LangGraph Cookbook (langfuse.com/guides/cookbook/integration_langgraph)
- langgraph-checkpoint-postgres (pypi.org)
- LangGraph v0.2 Checkpointer Libraries (blog.langchain.com)
- Multiple comparison articles (xcelore.com, aidevdayindia.org, zenml.io)
