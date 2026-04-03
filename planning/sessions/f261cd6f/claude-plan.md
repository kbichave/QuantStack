# Implementation Plan: CrewAI → LangGraph Migration

## Overview

QuantStack is an autonomous trading platform with two continuous loops (research and trading) orchestrated by CrewAI crews. This plan removes all CrewAI dependencies and replaces them with LangGraph state graphs, migrates vector storage from ChromaDB to pgvector, and preserves the existing LangFuse observability layer.

The migration is a **clean-cut rewrite** — no hybrid/incremental approach. All three crews (research, trading, supervisor) are rebuilt as LangGraph StateGraphs in a single pass.

### Why LangGraph

CrewAI's implicit orchestration (conversation-based handoff, hidden prompt injection, magic decorators) makes debugging production trading systems unreasonably hard. LangGraph's explicit state graph model gives us:

- **Typed state**: Every field flowing between nodes is declared in a TypedDict. No conversation archaeology to find what data an agent received.
- **Deterministic routing**: Risk gates and regime checks become conditional edges — testable Python functions, not LLM delegation.
- **Native async**: No `nest_asyncio` hacks. LangGraph nodes are async-first.
- **PostgreSQL checkpointing**: Built-in state persistence on our existing database.
- **LangFuse integration**: Callback handler traces every node, LLM call, and tool invocation automatically.

### Why pgvector (replacing ChromaDB)

ChromaDB is an extra service (Docker container, network hop, separate failure domain). We already run PostgreSQL. pgvector gives us vector similarity search inside the same database — one fewer service to operate, one fewer point of failure, transactional consistency with our application data.

---

## Section 1: Dependency & Scaffolding Changes

### 1.1 pyproject.toml

Remove the `crewai` optional dependency group entirely:
```
crewai[tools]>=0.100.0
chromadb>=0.5.0
ollama>=0.4.0  (keep if still used for local LLM)
nest-asyncio>=1.6.0
openinference-instrumentation-crewai>=0.1.0
duckduckgo-search>=6.0.0
```

Add a `langgraph` optional dependency group:
```
langgraph>=0.4.0,<0.5.0
langchain-core>=0.3.0,<0.4.0
langchain-anthropic>=0.3.0
langgraph-checkpoint-postgres>=3.0.0
psycopg[binary]>=3.1.0
psycopg-pool>=3.1.0
pgvector>=0.3.0
watchdog>=4.0.0
```

**Important**: Pin LangGraph and langchain-core to minor version ranges. LangGraph is pre-1.0 and has had breaking changes between minor versions.

Keep `ollama>=0.4.0` in a separate group if still needed for local LLM fallback.

**Note on `nest-asyncio`**: Do NOT remove `nest-asyncio` from dependencies in Phase 1. It is still needed by `mcp_bridge/` tools that use `run_async()`. Remove it only after all mcp_bridge tools are migrated (end of Phase 4).

### 1.2 Dockerfile

Change the install command from `pip install -e ".[all]"` (which includes `crewai`) to install the new `langgraph` group. Ensure `psycopg` binary dependencies are available in the base image.

### 1.3 docker-compose.yml

- **Remove**: `chromadb` service entirely
- **Keep**: postgres, langfuse, ollama
- **Rename**: `trading-crew` → `trading-graph`, `research-crew` → `research-graph`, `supervisor-crew` → `supervisor-graph` (cosmetic, clarifies architecture)
- **Update** runner commands: `python -m quantstack.runners.trading_runner` (same entry points, different internals)
- **Add** pgvector: Ensure PostgreSQL image includes pgvector extension (use `pgvector/pgvector:pg16` or add extension to existing image)

### 1.4 Delete CrewAI Documentation

Remove `docs/crewai_docs_md/` entirely (~180 files). This was crawled reference material, not project documentation.

### 1.5 Directory Structure Changes

```
src/quantstack/
  graphs/                          # NEW — replaces crews/ for orchestration
    __init__.py
    state.py                       # TypedDict state schemas
    config.py                      # AgentConfig dataclass + YAML loader
    config_watcher.py              # Hot-reload (watchdog + SIGHUP)
    research/
      __init__.py
      graph.py                     # build_research_graph()
      nodes.py                     # Node functions
      config/
        agents.yaml                # Agent profiles (adapted format)
    trading/
      __init__.py
      graph.py                     # build_trading_graph()
      nodes.py
      config/
        agents.yaml
    supervisor/
      __init__.py
      graph.py                     # build_supervisor_graph()
      nodes.py
      config/
        agents.yaml
  tools/                           # REFACTORED — split by caller type
    langchain/                     # LLM-facing tools (@tool decorator)
      __init__.py
      signal_tools.py
      analysis_tools.py
      research_tools.py
      ...                          # ~15-20 files
    functions/                     # Node-callable plain async functions
      __init__.py
      risk_functions.py
      data_functions.py
      backtest_functions.py
      ...                          # ~10 files
    mcp_bridge/                    # EXISTING — refactor BaseTool → split
      ...                          # ~50 files, audit and split
  crews/                           # KEEP pure Python only
    __init__.py                    # Re-exports schemas (unchanged)
    decoder_crew.py                # Pure Python IC analyzer (unchanged)
    registry.py                    # Metadata registry (unchanged)
    schemas.py                     # Schema re-exports (unchanged)
    risk/
      safety_gate.py               # Pure Python safety boundary (unchanged)
  rag/                             # REFACTORED — ChromaDB → pgvector
    embeddings.py                  # Switch to pgvector backend
    ...
```

**Delete entirely:**
- `src/quantstack/crewai_tools/` (25 files)
- `src/quantstack/crewai_compat.py`
- `docs/crewai_docs_md/` (~180 files)
- CrewAI crew.py files (3 files — replaced by graphs/)

---

## Section 2: LLM Provider Refactor

### 2.1 Current State

`src/quantstack/llm/provider.py` has `get_model(tier) → str` which returns a model identifier string (e.g., `"bedrock/anthropic.claude-sonnet-4-20250514-v1:0"`). This string was consumed by CrewAI's `Agent(llm=...)`.

### 2.2 Target

Add a new function `get_chat_model(tier) → BaseChatModel` that returns a LangChain ChatModel instance. Keep `get_model()` for backward compatibility (other code may use the string).

The function should:
1. Read the tier → model string mapping (existing logic)
2. Instantiate the appropriate LangChain chat model class based on provider:
   - `bedrock` → `ChatBedrock`
   - `anthropic` → `ChatAnthropic`
   - `openai` → `ChatOpenAI`
   - `ollama` → `ChatOllama`
3. Apply the fallback chain (try providers in order until one succeeds)

### 2.3 Config

Add to `src/quantstack/llm/config.py`:

```python
@dataclass(frozen=True)
class ModelConfig:
    provider: str
    model_id: str
    tier: str
    max_tokens: int = 4096
    temperature: float = 0.0
```

The `get_chat_model()` function returns a configured `BaseChatModel` instance with appropriate parameters for the tier.

---

## Section 3: Agent Configuration System

### 3.1 YAML Format

Adapt the existing YAML agent definitions to a new format that separates orchestration concerns from LLM configuration:

```yaml
# agents.yaml (per graph)
quant_researcher:
  role: "Senior Quantitative Researcher"
  goal: "Discover and validate alpha-generating strategies"
  backstory: |
    You are a senior quant researcher...
  llm_tier: heavy
  max_iterations: 20
  timeout_seconds: 600
  tools:
    - signal_brief
    - multi_signal_brief
    - fetch_market_data
    - compute_features
```

### 3.2 AgentConfig Dataclass

```python
@dataclass
class AgentConfig:
    name: str
    role: str
    goal: str
    backstory: str
    llm_tier: str        # "heavy", "medium", "light"
    max_iterations: int
    timeout_seconds: int
    tools: list[str]     # tool registry keys
```

### 3.3 Config Loader

A loader function reads YAML, validates against `AgentConfig` schema, and returns a mapping of `{agent_name: AgentConfig}`. Called at startup and on hot-reload.

### 3.4 Hot-Reload

Two mechanisms:
- **Dev (file-watch)**: `watchdog` library monitors YAML files. On change, reload configs and rebuild graphs on next cycle.
- **Prod (SIGHUP)**: Signal handler triggers config reload. Runners check a reload flag at the start of each cycle.

Implementation: A `ConfigWatcher` class that:
1. Loads initial configs from YAML
2. Exposes `get_config(agent_name) → AgentConfig`
3. Watches for changes (file or signal)
4. Thread-safe reload (atomic swap of config dict)

The runner passes `ConfigWatcher` to the graph builder. Each cycle, the builder reads current configs and constructs the graph. No graph caching — graphs are cheap to build.

---

## Section 4: State Schema Design

### 4.1 Design Principles

Keep state lean. Store IDs and summaries in graph state; store heavy data (DataFrames, signal arrays, backtest results) in PostgreSQL application tables and reference by ID.

**Append-only fields**: Fields declared as `Annotated[list[T], operator.add]` accumulate across nodes. Nodes that don't append must either return an empty list `[]` for that field or omit the field entirely from the return dict. Returning `None` will fail. This is a common LangGraph pitfall — all node functions must be aware of which fields are append-only.

**Explicit vs implicit context**: CrewAI's `memory=True` provided implicit conversational memory — agents could reference what earlier agents said in raw text. LangGraph replaces this with explicit typed state. This is a deliberate tradeoff: we lose "conversation archaeology" but gain predictable, debuggable data flow. Before building each graph, audit the existing `tasks.yaml` `context` fields to ensure no task relies on the raw conversational output of a prior task (as opposed to its structured result).

### 4.2 Research Graph State

```python
class ResearchState(TypedDict):
    # Input
    cycle_number: int
    regime: str                              # current market regime
    # Pipeline
    context_summary: str                     # loaded context (text summary)
    selected_domain: str                     # "swing", "investment", "options"
    selected_symbols: list[str]
    hypothesis: str                          # generated hypothesis
    validation_result: dict                  # signal validation output
    backtest_id: str                         # reference to backtest results in DB
    ml_experiment_id: str                    # reference to ML experiment in DB
    registered_strategy_id: str              # newly registered strategy
    # Accumulation
    errors: Annotated[list[str], operator.add]
    decisions: Annotated[list[dict], operator.add]  # audit trail
```

### 4.3 Trading Graph State

```python
class TradingState(TypedDict):
    # Input
    cycle_number: int
    regime: str
    portfolio_context: dict                  # positions, cash, exposure
    # Pipeline
    daily_plan: str                          # daily planner output
    position_reviews: list[dict]             # per-position HOLD/TRIM/CLOSE
    exit_orders: list[dict]                  # executed exits
    entry_candidates: list[dict]             # scanned candidates
    risk_verdicts: list[dict]                # per-candidate risk sizing
    fund_manager_decisions: list[dict]       # APPROVED/REJECTED per candidate
    options_analysis: list[dict]             # options structures if applicable
    entry_orders: list[dict]                 # executed entries
    reflection: str                          # trade reflector output
    # Accumulation
    errors: Annotated[list[str], operator.add]
    decisions: Annotated[list[dict], operator.add]
```

### 4.4 Supervisor Graph State

```python
class SupervisorState(TypedDict):
    cycle_number: int
    health_status: dict                      # system health check results
    diagnosed_issues: list[dict]             # issues found
    recovery_actions: list[dict]             # actions taken
    strategy_lifecycle_actions: list[dict]   # promotions/retirements
    scheduled_task_results: list[dict]
    errors: Annotated[list[str], operator.add]
```

---

## Section 5: Graph Builders

### 5.1 Research Graph

**Topology**: Linear pipeline with one conditional branch.

```
START → context_load → domain_selection → hypothesis_generation
      → signal_validation → [PASS?] → backtest_validation
                            [FAIL?] → END (with error logged)
      → ml_experiment → strategy_registration → knowledge_update → END
```

**Process mapping**: The current hierarchical process (quant_researcher as manager) becomes a linear graph. The "manager" logic was really just task ordering — LangGraph's edges handle this explicitly. No supervisor node needed.

**Nodes**:
- `context_load`: Tool node (deterministic). Calls `get_system_status()`, `get_regime()`, loads recent session handoffs. Returns context summary.
- `domain_selection`: Agent node. LLM decides which research domain to pursue based on context, regime, and strategy gaps.
- `hypothesis_generation`: Agent node. LLM generates testable hypothesis for selected domain.
- `signal_validation`: Tool node. Calls signal computation functions. Returns pass/fail.
- `backtest_validation`: Tool node. Runs backtest, stores results in DB, returns backtest_id.
- `ml_experiment`: Agent node. LLM designs and interprets ML experiment. Calls ML tools.
- `strategy_registration`: Tool node. Registers validated strategy in DB.
- `knowledge_update`: Tool node. Updates knowledge base (pgvector).

**Conditional edge after signal_validation**: Router function checks `validation_result["passed"]` → routes to `backtest_validation` or `END`.

### 5.2 Trading Graph

**Topology**: Sequential with two parallel branches and a mandatory risk gate.

```
START → safety_check → daily_plan
      → [PARALLEL: position_review → execute_exits ─┐
                    entry_scan ─────────────────────┘
      → merge_parallel → risk_sizing
      → [RISK GATE: pass/fail]
      → portfolio_review → options_analysis → execute_entries
      → reflection → END
```

**Parallel execution**: Add dual edges from `daily_plan` to both `position_review` and `entry_scan`:

```python
graph.add_edge("daily_plan", "position_review")
graph.add_edge("daily_plan", "entry_scan")
```

LangGraph executes both concurrently since they share a predecessor and write to different state fields. Both branches converge at a **join node** (`merge_parallel`) before proceeding:

```python
graph.add_edge("position_review", "execute_exits")
graph.add_edge("execute_exits", "merge_parallel")
graph.add_edge("entry_scan", "merge_parallel")
# merge_parallel → risk_sizing → ...
```

**Note**: Do NOT use `Send()` here — `Send()` is for map-reduce patterns (fanning out the same node over a collection). Dual edges are the correct pattern for running two different nodes concurrently.

**Risk gate**: Mandatory conditional edge after `risk_sizing`. Calls `SafetyGate.validate()` — pure Python, no LLM. Routes to `portfolio_review` on pass, `END` (with violations logged) on fail.

**Nodes**:
- `safety_check`: Tool node. Calls `get_system_status()`. If system halted, routes to END immediately.
- `daily_plan`: Agent node (daily_planner). LLM generates plan based on regime, calendar, portfolio state.
- `position_review`: Agent node (position_monitor). Reviews each open position. Returns HOLD/TRIM/CLOSE per position.
- `entry_scan`: Agent node (trade_debater). Scans for entry candidates. Returns structured candidates list.
- `execute_exits`: Tool node. Executes exit orders via Alpaca API. Returns order confirmations.
- `merge_parallel`: Tool node (no-op). Join point for parallel branches. Ensures both position_review and entry_scan have completed before proceeding.
- `risk_sizing`: Tool+Agent node. Computes position sizes via Kelly criterion (tool), then risk analyst reviews (agent).
- `portfolio_review`: Agent node (fund_manager). Batch review of all proposed entries for correlation, allocation, diversity.
- `options_analysis`: Agent node (options_analyst). If any candidates are options-eligible, selects structures.
- `execute_entries`: Tool node. Executes approved entries via Alpaca API.
- `reflection`: Agent node (trade_reflector). Analyzes completed trades, extracts lessons.

**Note**: No `persist_state` node needed — `AsyncPostgresSaver` automatically checkpoints state after every node when configured (see Section 9.5).

### 5.3 Supervisor Graph

**Topology**: Linear, simplest graph.

```
START → health_check → diagnose_issues → execute_recovery
      → strategy_lifecycle → scheduled_tasks → END
```

**Nodes**: All are relatively simple — health_monitor and self_healer are light-model agents, strategy_promoter is medium-model.

### 5.4 Graph Builder Pattern

Each graph module exports a `build_*_graph(config_watcher, checkpointer) → CompiledStateGraph` function:

```python
def build_research_graph(
    config_watcher: ConfigWatcher,
    checkpointer: AsyncPostgresSaver,
) -> CompiledStateGraph:
    """Build the research pipeline graph."""
```

The builder:
1. Reads current agent configs from `config_watcher`
2. Constructs `ChatModel` instances via `get_chat_model(tier)`
3. Creates node functions with agent configs bound (via closures or partial)
4. Defines the StateGraph topology
5. Compiles with the checkpointer

---

## Section 6: Tool Layer Refactor

### 6.1 Audit and Split

Every tool in `src/quantstack/crewai_tools/` and `src/quantstack/tools/mcp_bridge/` must be classified:

**LLM-facing** (agent node calls this via tool-calling): Needs `@tool` decorator from `langchain_core.tools`. These tools have descriptions that help the LLM decide when/how to call them.

**Node-callable** (graph node calls this directly): Plain async function. No decorator, no description. Called explicitly in node code.

Classification heuristic:
- If the tool is called by a reasoning agent that decides *whether* to call it → LLM-facing
- If the tool is always called unconditionally by a specific node → node-callable

### 6.2 LLM-Facing Tools (new `src/quantstack/tools/langchain/`)

Pattern:
```python
from langchain_core.tools import tool

@tool
async def get_signal_brief(symbol: str) -> str:
    """Get a technical signal brief for a symbol including trend, momentum, and key levels."""
    result = await mcp_signal_brief(symbol)
    return json.dumps(result, default=str)
```

Key difference from CrewAI: LangChain's `@tool` is natively async. No `run_async()` bridge needed.

### 6.3 Node-Callable Functions (new `src/quantstack/tools/functions/`)

Pattern:
```python
async def fetch_market_data(symbol: str, start_date: str, end_date: str) -> dict:
    """Fetch OHLCV market data for the given symbol and date range."""
    return await mcp_fetch_market_data(symbol, start_date, end_date)
```

No decorator, no JSON serialization overhead. Called directly in node functions.

### 6.4 mcp_bridge Migration

The files in `tools/mcp_bridge/` (exact count TBD — audit during implementation) currently subclass `crewai_compat.BaseTool`. Each file needs:
1. Audit: is this tool LLM-facing or node-callable?
2. If LLM-facing: convert to `@tool` decorated async function in `tools/langchain/`
3. If node-callable: convert to plain async function in `tools/functions/`
4. Delete the original mcp_bridge file after conversion

After all 50 are migrated, delete `crewai_compat.py`.

### 6.5 Tool Registry

A registry maps tool names (from YAML agent configs) to actual tool objects:

```python
TOOL_REGISTRY: dict[str, BaseTool] = {
    "signal_brief": get_signal_brief,
    "multi_signal_brief": run_multi_signal_brief,
    ...
}
```

Graph builders look up tools by name from agent configs, enabling YAML-driven tool assignment.

---

## Section 7: RAG Pipeline Migration (ChromaDB → pgvector)

### 7.1 Database Setup

Add pgvector extension to the `quantstack` PostgreSQL database:
```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

Create embeddings table:
```python
class EmbeddingRecord:
    id: str
    collection: str           # "research", "trading", "strategy"
    content: str              # original text
    embedding: list[float]    # vector (dimension matches model)
    metadata: dict            # ticker, date, source, etc.
    created_at: datetime
```

### 7.2 Embedding Backend

Refactor `src/quantstack/rag/embeddings.py`:
- Remove ChromaDB client initialization
- Add `psycopg` connection pool for pgvector operations
- Implement `store_embedding()`, `search_similar()`, `delete_collection()` using pgvector SQL queries
- Keep the embedding model unchanged (ollama/mxbai-embed-large)

### 7.3 Migration Script

One-time script (`scripts/migrate_chromadb_to_pgvector.py`):
1. Connect to ChromaDB
2. Export all collections (embeddings + metadata)
3. Verify embedding vector dimensions are consistent per collection
4. Create pgvector tables with matching dimensions
5. Batch-insert embeddings
6. Verify counts match per collection
7. Run sample similarity searches on 5-10 queries and compare results between ChromaDB and pgvector (sanity check that vectors transferred correctly)
8. Print migration report with counts, dimensions, and similarity comparison results

### 7.4 Knowledge Base Interface

Update `src/quantstack/tools/knowledge_tools.py` (or wherever RAG search is exposed):
- Replace ChromaDB query calls with pgvector similarity search
- Keep the same public API (`search_knowledge_base(query, collection, n_results)`)

---

## Section 8: Observability Changes

### 8.1 Remove CrewAI Instrumentation

In `src/quantstack/observability/instrumentation.py`:
- Remove `CrewAIInstrumentor().instrument()` call
- Remove `openinference-instrumentation-crewai` import
- Add LangFuse callback handler initialization

### 8.2 LangFuse Callback Handler

Create a shared callback handler factory:

```python
def get_langfuse_handler(session_id: str, tags: list[str]) -> CallbackHandler:
    """Create a LangFuse callback handler for a graph invocation."""
```

Each runner creates a handler per cycle, passing it to `graph.invoke()` config. This automatically traces all nodes, LLM calls, and tool invocations.

### 8.3 Custom Trace Helpers

Rename `crew_tracing.py` → merge into `tracing.py`. Keep all 5 trace functions:
- `trace_provider_failover()`
- `trace_strategy_lifecycle()`
- `trace_self_healing_event()`
- `trace_capital_allocation()`
- `trace_safety_boundary_trigger()`

Update docstrings to remove "CrewAI" references. No functional changes — these are pure LangFuse calls.

### 8.4 Flush on Shutdown

Ensure `GracefulShutdown` handler calls `get_client().shutdown()` to flush pending LangFuse events before process exit. This is already partially implemented in `flush_util.py`.

---

## Section 9: Runner Refactor

### 9.1 Common Pattern

All three runners follow the same lifecycle:
1. `setup_instrumentation()` — initialize LangFuse
2. `ConfigWatcher(yaml_path)` — load agent configs
3. `AsyncPostgresSaver.from_conn_string(DB_URI)` → checkpointer
4. `build_*_graph(config_watcher, checkpointer)` → compiled graph
5. Loop: `graph.ainvoke(initial_state, config)` per cycle

### 9.2 Runner Loop Changes (Sync → Async Migration)

The current `run_loop()` is fully synchronous (`time.sleep()`, `crew.kickoff()`). It must become async to call `graph.ainvoke()`. This is a non-trivial change:

**Entry point**: Each runner's `main()` function wraps the async loop with `asyncio.run()`:
```python
def main() -> None:
    asyncio.run(async_main())

async def async_main() -> None:
    # ... setup ...
    await run_loop(graph_builder, shutdown, graph_name="trading")
```

**Loop internals**: Replace `time.sleep(interval)` with `asyncio.sleep(interval)`. Replace `crew.kickoff()` with `await graph.ainvoke(state, config)`.

**AgentWatchdog compatibility**: The current watchdog uses `threading.Timer`. In an async context, replace with `asyncio.wait_for(graph.ainvoke(...), timeout=seconds)`. This is simpler and more reliable than a thread-based timer in async code.

**GracefulShutdown compatibility**: Signal handlers registered via `signal.signal()` work in async contexts, but the handler must use `loop.call_soon_threadsafe()` to set the shutdown flag from a signal context. Alternatively, use `loop.add_signal_handler()` which is designed for asyncio.

**The `run_loop()` function needs**:
- Accept a graph builder function instead of a crew factory
- Build initial state from current market conditions / portfolio
- Pass LangFuse callback handler in config
- Pass `thread_id` for checkpointing (e.g., `f"trading-{date}-cycle-{n}"`)
- Use `asyncio.wait_for()` for timeout instead of AgentWatchdog threading

### 9.3 Cycle Intervals (Unchanged)

Keep the existing interval logic:
- Trading: 5 min (market hours), 30 min (after hours)
- Research: 10 min (market hours), 30 min (after hours), 2 hours (weekend)
- Supervisor: 5 min always

### 9.4 Watchdog Integration

Keep `AgentWatchdog` with same timeouts. It wraps the `graph.ainvoke()` call. On timeout, it kills the cycle and logs to LangFuse.

### 9.5 Checkpoint Persistence

Replace the manual `save_checkpoint()` SQL call with LangGraph's built-in checkpointing. The `AsyncPostgresSaver` handles this automatically — each `graph.ainvoke()` with a `thread_id` persists state.

Keep the `crew_checkpoints` table for backward compatibility during transition, but new cycles write to LangGraph's checkpoint tables.

---

## Section 10: Testing Strategy

### 10.1 Node Unit Tests

Each node function is a pure function of state → state update. Test in isolation:

```python
async def test_context_load_node():
    """context_load returns a context summary given cycle inputs."""
```

### 10.2 Graph Integration Tests

Compile the graph with `MemorySaver` (in-memory checkpointer for tests) and invoke with test state. Assert final state matches expected output.

### 10.3 Tool Contract Tests

Replace `test_crewai_tools/test_tool_wrapper_contract.py`:
- For LLM-facing tools: verify each has `@tool` decorator, non-empty description, returns string
- For node-callable functions: verify each is async, has type hints, returns expected type

### 10.4 Config Validation Tests

Replace `test_crew_workflows.py`:
- Parse YAML agent configs
- Validate against `AgentConfig` schema
- Cross-check tool references against tool registry

### 10.5 Risk Safety Tests

Keep `test_crewai_risk_safety.py` unchanged (rename file to `test_risk_safety.py`). It tests pure Python `SafetyGate` — no framework dependency.

### 10.6 Regression Tests

Before migrating each graph, capture I/O from 3-5 real cycles of the current CrewAI system:
- Record input state (market conditions, portfolio, regime)
- Record output state (decisions made, orders placed, strategies registered)
- Replay identical inputs through the new LangGraph implementation
- Assert outputs are equivalent (or document why they differ)

This is critical for a capital-handling system. Without regression baselines, correctness is assumed, not verified.

### 10.7 Timing Benchmarks

Each graph invocation must complete within its cycle interval (trading: 5 min, research: 10 min, supervisor: 5 min). Write benchmark tests that measure wall-clock time for `graph.ainvoke()` with realistic state. LangGraph overhead (state serialization, checkpointing, callback dispatch) is non-zero.

### 10.8 LangFuse Trace Assertion Tests

Assert specific trace structure in integration tests:
- Every node execution appears as a span
- Tool invocations are nested under their parent node span
- Session ID and thread ID match expected values
- Use a test LangFuse project or mock the callback handler

### 10.9 Integration Smoke Tests

Update `tests/integration/test_e2e_smoke.py`:
- Build each graph
- Invoke with minimal test state
- Assert graph completes without error
- Verify LangFuse traces were emitted

---

## Section 11: Migration Execution Order

### Phase 1: Foundation (no behavior change)

1. Add LangGraph dependencies to pyproject.toml
2. Create `src/quantstack/graphs/` directory structure
3. Create state schemas (TypedDict) for all 3 graphs
4. Create `AgentConfig` dataclass and YAML loader
5. Add `get_chat_model()` to LLM provider
6. Set up pgvector extension in PostgreSQL
7. Write ChromaDB → pgvector migration script

### Phase 2: Supervisor Graph (simplest, proves the pattern)

1. Audit supervisor's tools — classify as LLM-facing or node-callable. Migrate the tools this graph needs.
2. Build `SupervisorGraph` with 5 nodes + error handling (retry_policy on agent nodes)
3. Refactor `supervisor_runner.py` to async (`asyncio.run()` entry, `asyncio.wait_for()` timeout)
4. Write node unit tests + graph integration test
5. Validate LangFuse tracing works end-to-end
6. Run supervisor in paper mode alongside old system to verify behavior parity

### Phase 3: Research Graph

1. Audit research tools — classify and migrate tools this graph needs
2. Build `ResearchGraph` with 8 nodes + conditional edge + error handling
3. Refactor `research_runner.py` to async
4. Write tests (node unit + graph integration)
5. Capture regression baseline: record 3-5 current research cycle I/O pairs, replay through new graph

### Phase 4: Trading Graph (most complex)

1. Audit trading tools — classify and migrate tools this graph needs
2. Build `TradingGraph` with 11 nodes + parallel branches (dual edges, join node) + risk gate
3. Refactor `trading_runner.py` to async
4. Write tests (node unit + graph integration + risk gate edge test)
5. Validate risk gate conditional edge with both pass/fail cases
6. **Shadow-run phase**: Run new trading graph on paper trades for at least 2 full trading days alongside the old system. Compare decisions, timing, and LangFuse traces.
7. Audit remaining mcp_bridge files not yet migrated. Migrate stragglers. Delete `crewai_compat.py`.

### Phase 5: Cleanup & Observability

1. Delete `src/quantstack/crewai_tools/` entirely
2. Delete crew.py files from `src/quantstack/crews/` (keep pure Python modules)
3. Remove CrewAI instrumentation, update to LangFuse callback handler
4. Merge `crew_tracing.py` into `tracing.py`
5. Delete `docs/crewai_docs_md/`
6. Remove `nest-asyncio` from dependencies (all tools now migrated)
7. Update Docker configs (remove ChromaDB, update service names)
8. Update `start.sh` / tmux setup if service names changed
9. Run full test suite + timing benchmarks (ensure all graphs complete within cycle intervals)

### Phase 6: Config Hot-Reload

1. Implement `ConfigWatcher` with watchdog (dev) + SIGHUP (prod)
2. Ensure reload only takes effect at cycle boundaries (not mid-graph-execution)
3. Wire into all three runners
4. Write tests for config reload

---

## Section 12: Risk & Safety Considerations

### 12.1 Error Handling Strategy

LangGraph supports `retry_policy` on nodes. Apply it based on node type:

- **Agent nodes** (LLM calls): Retry up to 2 times with exponential backoff. LLM calls can fail transiently (rate limits, network). After retries exhausted, append error to `errors` state field and route to END.
- **Tool nodes** (deterministic): Retry once. Deterministic failures are usually permanent (bad data, missing DB row). After retry, append error and continue or route to END depending on criticality.
- **Critical nodes** (safety_check, risk_sizing): No retry. Fail fast. If the safety check or risk gate fails, the graph must terminate — retrying a safety check that returns "halted" is wrong.

Each graph should have an error-handling edge from any node to END that logs the error to the `errors` state field and to LangFuse. The runner inspects the final state's `errors` field to determine cycle status.

### 12.2 Risk Gate Enforcement

The `SafetyGate` (pure Python) is wired as a **mandatory conditional edge** in the trading graph. There is no path from `risk_sizing` to `execute_entries` that bypasses the gate. This is enforced by graph topology — not by convention.

### 12.2 Paper Mode

The existing `ALPACA_PAPER=true` / `USE_REAL_TRADING=false` environment variables are checked in the execution tools, not in the orchestration layer. No changes needed — the tools enforce paper mode regardless of which framework calls them.

### 12.3 System Halt (Kill Switch)

The `safety_check` node at the start of every trading graph cycle calls `get_system_status()`. If halted, the graph routes to END immediately. This replaces the CrewAI task that did the same check.

### 12.4 Audit Trail

LangFuse callback handler automatically traces every node execution, LLM call, and tool invocation. Combined with custom trace helpers (strategy_lifecycle, capital_allocation, safety_boundary), the audit trail is more complete than with CrewAI.

### 12.6 Shadow-Run Acceptance Criteria

Before cutting over to LangGraph for live trading:
1. All three graphs pass unit tests and integration tests
2. Trading graph completes a shadow-run of at least 2 full trading days on paper trades
3. Shadow-run decisions match or exceed the quality of CrewAI decisions (manual review)
4. All graph invocations complete within cycle interval budgets (timing benchmarks pass)
5. LangFuse traces show complete coverage (every node, LLM call, tool invocation traced)
6. No regression in risk gate enforcement (verified by specific test cases)

### 12.7 Rollback Plan

If the migration fails mid-flight:
- The old crew code is in git history
- PostgreSQL checkpoint tables coexist with existing `crew_checkpoints`
- pgvector tables coexist with ChromaDB data (ChromaDB service can be re-added to docker-compose)
- Reverting is: `git revert` + restore ChromaDB service + restart old runners
- **Note**: Database schema changes (pgvector extension, checkpoint tables) are additive and do not need rollback — they coexist safely with the old system

---

## Section 13: Docker & Infrastructure

### 13.1 PostgreSQL Image

Switch to `pgvector/pgvector:pg16` (or add pgvector extension to existing image). This provides the `vector` extension needed for embedding storage.

### 13.2 Service Changes

| Before | After |
|--------|-------|
| trading-crew | trading-graph |
| research-crew | research-graph |
| supervisor-crew | supervisor-graph |
| chromadb | (removed) |
| postgres | postgres (with pgvector) |
| langfuse | langfuse (unchanged) |
| ollama | ollama (unchanged) |

### 13.3 Health Checks

Update Docker health checks to ping the new runner processes. The runners still write heartbeats to PostgreSQL — the health check mechanism doesn't change.

### 13.4 Environment Variables

No new env vars required. Remove any CrewAI-specific vars from `.env.example` (e.g., `CREWAI_TELEMETRY_ENABLED`). The LangFuse vars (`LANGFUSE_SECRET_KEY`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_HOST`) are already set.

Add to `.env.example`:
```
# LangGraph checkpointing (uses same TRADER_PG_URL)
# No additional config needed — checkpointer uses the existing database
```
