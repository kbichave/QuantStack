# Complete Specification: CrewAI to LangGraph Migration

## 1. Goal

Completely remove CrewAI from QuantStack and replace it with LangGraph. Drop ChromaDB in favor of pgvector. Keep LangFuse for observability. Clean-cut rewrite — no hybrid/incremental approach.

## 2. Motivation

- CrewAI adoption declining; LangGraph winning the multi-agent orchestration space
- LangGraph's explicit state graph model fits QuantStack's deterministic trading workflows (risk gates, regime checks, sequential pipelines)
- CrewAI's magic (decorators, YAML configs, hidden prompt injection, implicit memory) makes debugging hard
- LangGraph has first-class LangFuse integration via callback handlers
- LangGraph supports streaming, checkpointing (PostgreSQL), and human-in-the-loop natively
- Opportunity to consolidate vector storage (ChromaDB → pgvector) and reduce infrastructure complexity

## 3. Current Architecture (What Changes)

### 3.1 CrewAI Crews → LangGraph StateGraphs

| Crew | Agents | Tasks | Process | LangGraph Equivalent |
|------|--------|-------|---------|---------------------|
| ResearchCrew | 4 (quant_researcher as manager + 3 workers) | 8 (linear pipeline) | hierarchical | StateGraph with conditional edges for manager routing |
| TradingCrew | 10 | 11 (DAG with async branches) | sequential | StateGraph with parallel branches + conditional risk gate |
| SupervisorCrew | 3 | 5 | sequential | StateGraph, simplest graph |

### 3.2 Tool Wrappers (Hybrid Split)

**25 files in `src/quantstack/crewai_tools/`** — all use `@tool` decorator from crewai + `run_async()` bridge.

Decision: **Split by caller**:
- **~15-20 LLM-facing tools** (hypothesis, analysis, signal interpretation) → `langchain_core.tools.tool` decorator. LLM agents discover and call these.
- **~10 deterministic tools** (risk calc, backtest, data fetch) → Plain async functions called directly by graph nodes. No wrapper abstraction.

**50 files in `src/quantstack/tools/mcp_bridge/`** — use `crewai_compat.py` BaseTool stubs. Same split:
- LLM-invoked → LangChain BaseTool
- Node-invoked → plain async functions
- Delete `crewai_compat.py` entirely after migration

### 3.3 Runners

3 runners (`trading_runner.py`, `research_runner.py`, `supervisor_runner.py`) currently:
```python
crew = SomeCrew().crew()
crew.kickoff()
```

Will become:
```python
graph = build_some_graph(config)
await graph.ainvoke(initial_state, config={"configurable": {"thread_id": "..."}})
```

### 3.4 Agent Configuration

**Current**: YAML files (`agents.yaml`, `tasks.yaml`) per crew, parsed by CrewAI decorators.

**Target**: YAML → dataclass → graph builder:
1. YAML files define agent profiles (role, goal, backstory, llm tier, timeouts)
2. Loader parses into `@dataclass AgentConfig` with validation
3. Graph builder accepts `AgentConfig`, builds nodes programmatically
4. **Hot-reload**: file-watch (dev) + SIGHUP (prod) — config changes without restart

### 3.5 LLM Provider

**Current**: `get_model(tier)` returns model string for CrewAI `Agent(llm=...)`.

**Target**: Returns `ChatModel` instance (via LiteLLM or langchain-anthropic) for LangGraph node functions. Keep tier system (heavy/medium/light/embedding) and fallback chain.

### 3.6 Memory & RAG

**Current**: ChromaDB for vector storage (embeddings, knowledge base search). CrewAI `memory=True` for agent memory.

**Target**:
- **Graph state**: LangGraph PostgreSQL checkpointer (`langgraph-checkpoint-postgres` + `AsyncPostgresSaver`)
- **Vector storage**: pgvector extension on existing `quantstack` PostgreSQL database
- **Migration**: One-time script to export ChromaDB embeddings → pgvector tables
- **Drop**: ChromaDB service from docker-compose

### 3.7 Observability

**Current**: `CrewAIInstrumentor().instrument()` + custom Langfuse trace helpers.

**Target**:
- Remove `openinference-instrumentation-crewai`
- Use `langfuse.langchain.CallbackHandler` passed to every `graph.invoke()` config
- Keep custom trace helpers (rename `crew_tracing.py` → `graph_tracing.py` or merge into `tracing.py`)
- Automatic tracing of nodes, LLM calls, tool invocations

## 4. What Stays Unchanged

- **Risk gate** (`src/quantstack/execution/risk_gate.py`) — LAW, never modify
- **Safety gate** (`src/quantstack/crews/risk/safety_gate.py`) — pure Python
- **Decoder crew** (`src/quantstack/crews/decoder_crew.py`) — pure Python IC analyzer
- **Pydantic schemas** (`src/quantstack/shared/schemas.py`) — framework-agnostic
- **MCP tool implementations** (`src/quantstack/mcp/tools/`) — underlying async functions
- **Signal engine** (`src/quantstack/signal_engine/`) — pure Python
- **Autonomous runner** (`src/quantstack/autonomous/`) — pure Python
- **Database layer** (`src/quantstack/db.py`) — framework-agnostic
- **Claude agent definitions** (`.claude/agents/`) — separate system
- **Prompts** (`prompts/`) — drive Claude-native loops, not CrewAI

## 5. What Gets Deleted

- `docs/crewai_docs_md/` — ~180 crawled CrewAI doc files
- `src/quantstack/crewai_compat.py` — after all tools migrated
- `src/quantstack/crewai_tools/` — entire directory (replaced by LangGraph tool layer)
- CrewAI crew YAML configs — replaced by new YAML format for LangGraph agents
- `crewai` optional dependency group in pyproject.toml
- ChromaDB service from docker-compose.yml
- `openinference-instrumentation-crewai` dependency
- `nest-asyncio` dependency (LangGraph is natively async)

## 6. New Dependencies

```toml
[project.optional-dependencies]
langgraph = [
    "langgraph>=0.4.0",
    "langchain-core>=0.3.0",
    "langchain-anthropic>=0.3.0",     # or langchain-litellm
    "langgraph-checkpoint-postgres>=3.0.0",
    "psycopg[binary]>=3.1.0",         # async PG driver for checkpointer
    "psycopg-pool>=3.1.0",            # connection pooling
    "pgvector>=0.3.0",                # vector similarity search
    "watchdog>=4.0.0",                # file-watch for config hot-reload
]
```

## 7. LangGraph Architecture

### 7.1 State Schemas (TypedDict per graph)

Each graph gets an explicit typed state. Nodes return only changed fields. Use `Annotated[List, operator.add]` for append-only fields.

### 7.2 Node Types

- **Agent nodes**: LLM call with system prompt + tools. For reasoning-heavy steps.
- **Tool nodes**: Pure Python function, no LLM. For deterministic computation.
- **Router nodes**: Conditional edges based on state inspection (risk gate, regime check).

### 7.3 Graph Composition

Supervisor graph composes research and trading subgraphs. Subgraphs can be compiled independently for testing.

### 7.4 Checkpointing

`AsyncPostgresSaver` on existing `quantstack` database. Thread IDs map to loop iterations (e.g., `trading-2026-04-02-cycle-42`). Keep state lean — store heavy data (DataFrames, signal arrays) in app tables, reference by ID in graph state.

### 7.5 Concurrency

Trading graph has parallel branches (position_review || entry_scan). LangGraph supports this via `Send()` API or parallel node execution.

## 8. Constraints

- Risk gate enforcement is non-negotiable — mandatory conditional edge in every trading graph path
- Paper mode default preserved
- Audit trail via LangFuse (every decision logged with reasoning)
- Two-loop architecture (research + trading running independently) preserved
- Existing MCP tool implementations must not be modified — only the wrapper layer changes
- Claude Code agent spawning system (`.claude/agents/`) is separate and unaffected
