<!-- PROJECT_CONFIG
runtime: python-uv
test_command: uv run pytest
END_PROJECT_CONFIG -->

<!-- SECTION_MANIFEST
section-01-scaffolding
section-02-llm-provider
section-03-agent-config
section-04-state-schemas
section-05-tool-layer
section-06-supervisor-graph
section-07-research-graph
section-08-trading-graph
section-09-rag-migration
section-10-observability
section-11-runners
section-12-docker-cleanup
section-13-risk-safety
section-14-testing
END_MANIFEST -->

# Implementation Sections Index

## Dependency Graph

| Section | Depends On | Blocks | Parallelizable |
|---------|------------|--------|----------------|
| section-01-scaffolding | - | all | Yes |
| section-02-llm-provider | 01 | 06, 07, 08 | Yes |
| section-03-agent-config | 01 | 06, 07, 08 | Yes |
| section-04-state-schemas | 01 | 06, 07, 08 | Yes |
| section-05-tool-layer | 01 | 06, 07, 08 | Yes |
| section-06-supervisor-graph | 02, 03, 04, 05 | 11 | Yes |
| section-07-research-graph | 02, 03, 04, 05 | 11 | Yes |
| section-08-trading-graph | 02, 03, 04, 05 | 11 | No |
| section-09-rag-migration | 01 | 07 | Yes |
| section-10-observability | 01 | 11 | Yes |
| section-11-runners | 06, 07, 08, 10 | 14 | No |
| section-12-docker-cleanup | 01 | - | Yes |
| section-13-risk-safety | 08 | 14 | No |
| section-14-testing | 11, 13 | - | No |

## Execution Order

1. **Batch 1**: section-01-scaffolding (no dependencies)
2. **Batch 2**: section-02-llm-provider, section-03-agent-config, section-04-state-schemas, section-05-tool-layer, section-09-rag-migration, section-10-observability, section-12-docker-cleanup (all depend only on 01, parallelizable)
3. **Batch 3**: section-06-supervisor-graph, section-07-research-graph, section-08-trading-graph (depend on 02-05, supervisor/research parallelizable)
4. **Batch 4**: section-11-runners, section-13-risk-safety (depend on graphs)
5. **Batch 5**: section-14-testing (final validation)

## Section Summaries

### section-01-scaffolding
Dependency changes (pyproject.toml), directory structure creation (src/quantstack/graphs/), file deletions (crewai_tools/, crewai_docs_md/), Dockerfile and docker-compose skeleton updates.

### section-02-llm-provider
Add `get_chat_model(tier) → BaseChatModel` to LLM provider. Keep `get_model()` for backward compat. Add `ModelConfig` dataclass. Wire up provider-specific ChatModel classes (ChatBedrock, ChatAnthropic, ChatOpenAI, ChatOllama).

### section-03-agent-config
New YAML format for agent profiles. `AgentConfig` dataclass with validation. Config loader. `ConfigWatcher` class with file-watch (dev) + SIGHUP (prod) hot-reload. Atomic config swap, cycle-boundary reload semantics.

### section-04-state-schemas
TypedDict state schemas for all 3 graphs (ResearchState, TradingState, SupervisorState). Append-only field patterns with operator.add reducers.

### section-05-tool-layer
Audit and split crewai_tools/ and mcp_bridge/ tools. Create tools/langchain/ (LLM-facing @tool) and tools/functions/ (node-callable async). Build TOOL_REGISTRY. Delete crewai_compat.py.

### section-06-supervisor-graph
Build SupervisorGraph StateGraph (5 nodes, linear). Node implementations for health_check, diagnose_issues, execute_recovery, strategy_lifecycle, scheduled_tasks. Error handling with retry_policy. Unit tests for each node. Graph integration test.

### section-07-research-graph
Build ResearchGraph StateGraph (8 nodes + conditional edge). Node implementations. Conditional routing after signal_validation (pass → backtest, fail → END). Unit tests + integration test. Regression baseline capture.

### section-08-trading-graph
Build TradingGraph StateGraph (12 nodes + parallel branches + risk gate). Dual edges for position_review || entry_scan. merge_parallel join node. Mandatory risk gate conditional edge via SafetyGate.validate(). Node implementations. Unit tests + integration test. Shadow-run preparation.

### section-09-rag-migration
pgvector extension setup. Embeddings table schema. Refactor rag/embeddings.py from ChromaDB to pgvector. Migration script (ChromaDB → pgvector with dimension verification + sample similarity comparison). Update knowledge_tools.py.

### section-10-observability
Remove CrewAIInstrumentor. Add LangFuse CallbackHandler factory. Merge crew_tracing.py into tracing.py. Update docstrings. Ensure flush on shutdown.

### section-11-runners
Refactor all 3 runners from sync to async (asyncio.run entry, asyncio.wait_for timeout, loop.add_signal_handler for GracefulShutdown). Replace crew.kickoff() with graph.ainvoke(). Pass LangFuse callback + thread_id in config. Rebuild graph each cycle for hot-reload.

### section-12-docker-cleanup
Remove chromadb service. Switch postgres image to pgvector/pgvector:pg16. Rename crew services to graph services. Update .env.example. Update start.sh/tmux if needed. Remove CrewAI env vars.

### section-13-risk-safety
Validate risk gate is mandatory conditional edge in trading graph. Error handling strategy (retry_policy per node type). Shadow-run acceptance criteria verification. Paper mode enforcement verification. Kill switch verification.

### section-14-testing
Regression tests (replay captured I/O through new graphs). Timing benchmarks (graph invocations within cycle budgets). LangFuse trace structure assertions. Full end-to-end smoke tests. Config validation tests. Tool contract tests.
