# TDD Plan: CrewAI → LangGraph Migration

**Testing framework**: pytest (existing project convention)
**Test locations**: `tests/unit/`, `tests/integration/`
**Fixtures**: `conftest.py` per directory
**Run command**: `pytest tests/`

For each section of the implementation plan, these tests should be written BEFORE the implementation.

---

## Section 1: Dependency & Scaffolding Changes

```python
# Test: pyproject.toml has langgraph in optional dependencies
# Test: pyproject.toml does NOT have crewai in any dependency group
# Test: langgraph version pinned to >=0.4.0,<0.5.0
# Test: nest-asyncio still present (needed until Phase 4 complete)
# Test: src/quantstack/graphs/ directory exists with expected structure
# Test: src/quantstack/crewai_tools/ directory does NOT exist (after cleanup)
# Test: docs/crewai_docs_md/ directory does NOT exist (after cleanup)
```

---

## Section 2: LLM Provider Refactor

```python
# Test: get_chat_model("heavy") returns a BaseChatModel instance
# Test: get_chat_model("medium") returns a BaseChatModel instance
# Test: get_chat_model("light") returns a BaseChatModel instance
# Test: get_chat_model("invalid") raises ValueError
# Test: get_chat_model falls back through provider chain when primary unavailable
# Test: get_chat_model("heavy") returns ChatBedrock when LLM_PROVIDER=bedrock
# Test: get_chat_model("heavy") returns ChatAnthropic when LLM_PROVIDER=anthropic
# Test: get_model() still works (backward compat) — returns string, not ChatModel
# Test: ModelConfig dataclass is frozen (immutable)
```

---

## Section 3: Agent Configuration System

```python
# Test: AgentConfig loads from valid YAML — all fields populated
# Test: AgentConfig rejects YAML missing required fields (role, goal, llm_tier)
# Test: AgentConfig rejects invalid llm_tier values
# Test: config loader returns dict[str, AgentConfig] for multi-agent YAML
# Test: config loader raises on duplicate agent names within same file
# Test: tool references in AgentConfig cross-validate against TOOL_REGISTRY
# Test: ConfigWatcher.get_config() returns current config
# Test: ConfigWatcher detects file change and reloads (dev mode)
# Test: ConfigWatcher handles SIGHUP and reloads (prod mode)
# Test: ConfigWatcher reload is atomic — concurrent reads don't see partial state
# Test: ConfigWatcher reload only applies at cycle boundary, not mid-execution
```

---

## Section 4: State Schema Design

```python
# Test: ResearchState TypedDict has all required fields
# Test: TradingState TypedDict has all required fields
# Test: SupervisorState TypedDict has all required fields
# Test: append-only fields (errors, decisions) accumulate across nodes — verify operator.add reducer
# Test: node returning empty list for append-only field does not error
# Test: node omitting append-only field from return dict does not error
# Test: node returning None for append-only field raises (verify this fails as expected)
```

---

## Section 5: Graph Builders

### 5.1 Research Graph

```python
# Test: build_research_graph() returns CompiledStateGraph
# Test: research graph has expected node count (8)
# Test: research graph START connects to context_load
# Test: signal_validation routes to backtest_validation on pass
# Test: signal_validation routes to END on fail
# Test: all nodes in graph are callable (no missing implementations)
# Test: research graph invocation with mock LLM produces valid final state
# Test: context_load node returns context_summary field
# Test: domain_selection node returns selected_domain field
```

### 5.2 Trading Graph

```python
# Test: build_trading_graph() returns CompiledStateGraph
# Test: trading graph has expected node count (12 including merge_parallel)
# Test: safety_check routes to END when system halted
# Test: safety_check routes to daily_plan when system ok
# Test: daily_plan has two outgoing edges (position_review AND entry_scan)
# Test: position_review and entry_scan execute concurrently (both complete before merge_parallel)
# Test: merge_parallel waits for both branches before proceeding
# Test: risk_sizing routes to portfolio_review when SafetyGate approves
# Test: risk_sizing routes to END when SafetyGate rejects (with violations logged)
# Test: risk gate edge is mandatory — no path from risk_sizing to execute_entries bypasses it
# Test: full graph invocation with mock LLM produces valid final state
# Test: entry_orders is empty when risk gate rejects
```

### 5.3 Supervisor Graph

```python
# Test: build_supervisor_graph() returns CompiledStateGraph
# Test: supervisor graph is linear (5 nodes, no branches)
# Test: full graph invocation with mock LLM produces valid final state
```

### 5.4 Graph Builder Pattern

```python
# Test: graph builder accepts ConfigWatcher and checkpointer
# Test: graph builder reads agent configs from ConfigWatcher
# Test: graph builder creates ChatModel instances for each agent tier
# Test: rebuilding graph with new config produces different node behavior
```

---

## Section 6: Tool Layer Refactor

```python
# Test: all LLM-facing tools in tools/langchain/ have @tool decorator
# Test: all LLM-facing tools have non-empty description
# Test: all LLM-facing tools are async (coroutine functions)
# Test: all LLM-facing tools return str (JSON serialized)
# Test: all node-callable functions in tools/functions/ are async
# Test: all node-callable functions have type hints
# Test: no tool file imports from crewai or crewai_compat
# Test: TOOL_REGISTRY contains all tools referenced in agent YAML configs
# Test: signal_brief tool returns valid JSON with expected keys
# Test: risk tool returns valid JSON with expected keys
# Test: data tool returns valid JSON with expected keys
```

---

## Section 7: RAG Pipeline Migration (ChromaDB → pgvector)

```python
# Test: pgvector extension is installed in test database
# Test: embeddings table created with correct schema (id, collection, content, embedding, metadata)
# Test: store_embedding() inserts record with correct vector dimension
# Test: search_similar() returns top-N results ordered by cosine similarity
# Test: search_similar() filters by collection
# Test: search_similar() filters by metadata (e.g., ticker)
# Test: delete_collection() removes all records for a collection
# Test: no code imports chromadb (after migration)
# Test: migration script exports and imports embeddings with matching counts
# Test: migration script verifies vector dimensions are consistent
# Test: migration script sample similarity comparison produces equivalent results
```

---

## Section 8: Observability Changes

```python
# Test: setup_instrumentation() does not import CrewAIInstrumentor
# Test: setup_instrumentation() initializes Langfuse callback handler
# Test: get_langfuse_handler() returns CallbackHandler with session_id and tags
# Test: graph invocation with callback handler produces trace spans (mock LangFuse)
# Test: trace_provider_failover() still works (pure LangFuse, no CrewAI)
# Test: trace_strategy_lifecycle() still works
# Test: trace_safety_boundary_trigger() still works
# Test: crew_tracing.py does not exist (merged into tracing.py)
# Test: no code imports openinference.instrumentation.crewai
```

---

## Section 9: Runner Refactor

```python
# Test: trading_runner.main() calls asyncio.run()
# Test: research_runner.main() calls asyncio.run()
# Test: supervisor_runner.main() calls asyncio.run()
# Test: run_loop() is async (coroutine function)
# Test: run_loop() uses asyncio.wait_for() for timeout
# Test: run_loop() passes LangFuse callback handler in graph config
# Test: run_loop() passes thread_id in graph config
# Test: GracefulShutdown works in async context (signal stops the loop)
# Test: cycle interval logic unchanged (5 min market, 30 min after hours, etc.)
# Test: runner rebuilds graph each cycle (picks up config changes)
```

---

## Section 10: Testing Strategy (Meta-Tests)

```python
# Test: regression baseline captured for research graph (3+ cycles)
# Test: regression baseline captured for trading graph (3+ cycles)
# Test: graph invocation timing within cycle budget (trading < 5 min, research < 10 min)
# Test: LangFuse traces have correct structure (every node as span, nested tool calls)
```

---

## Section 12: Risk & Safety

```python
# Test: error retry_policy on agent nodes retries up to 2 times
# Test: error retry_policy on tool nodes retries up to 1 time
# Test: critical nodes (safety_check, risk_sizing) have no retry — fail fast
# Test: node error appends to errors state field
# Test: SafetyGate.validate() called as conditional edge (not as node)
# Test: no path through trading graph bypasses risk gate
# Test: paper mode enforced by execution tools regardless of graph framework
# Test: system halt detected by safety_check node → graph terminates
```

---

## Section 13: Docker & Infrastructure

```python
# Test: docker-compose.yml does not reference chromadb service
# Test: docker-compose.yml postgres image supports pgvector
# Test: .env.example does not contain CREWAI_ prefixed variables
# Test: Dockerfile installs langgraph optional dependency group
```
