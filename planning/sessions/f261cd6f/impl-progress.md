# Implementation Progress

## Section Checklist
- [x] section-01-scaffolding
- [x] section-02-llm-provider
- [x] section-03-agent-config
- [x] section-04-state-schemas
- [x] section-05-tool-layer
- [x] section-06-supervisor-graph
- [x] section-07-research-graph
- [x] section-08-trading-graph
- [x] section-09-rag-migration
- [x] section-10-observability
- [x] section-11-runners
- [x] section-12-docker-cleanup
- [x] section-13-risk-safety
- [x] section-14-testing

## Error Log
| Timestamp | Section | Error | Attempt | Resolution |
|-----------|---------|-------|---------|------------|
| 2026-04-02 | section-08 | LangGraph ValueError: node name collides with state key | 1 | Renamed daily_plan→plan_day, options_analysis→analyze_options, reflection→reflect |
| 2026-04-03 | docker-runtime | `RetryPolicy` import from `langgraph.pregel` fails in v1.1.4 | 1 | Changed to `from langgraph.types import RetryPolicy` in all 3 graph files |
| 2026-04-03 | docker-runtime | `ConfigWatcher()` missing required `yaml_path` arg | 1 | All 3 runners now resolve yaml_path via `Path(__file__).resolve().parent.parent / "graphs" / {name} / "config" / "agents.yaml"` |
| 2026-04-03 | docker-runtime | Langfuse v3 requires ClickHouse | 1 | Pinned image to `langfuse/langfuse:2` |
| 2026-04-03 | docker-runtime | Langfuse healthcheck: curl not in container, Next.js binds to container IP not localhost | 1 | Changed to `wget -q --spider http://$(hostname -i):3000/api/public/health` |
| 2026-04-03 | docker-runtime | Docker disk full during torch CUDA dep install | 1 | `docker system prune -f` freed 14.75GB |

## Session Log
- Completed section-01-scaffolding: deps updated (crewai→langgraph), graphs/ structure created, crewai_tools/ + crewai_compat.py + crewai_docs_md/ deleted, docker-compose + Dockerfile + .env.example updated. 44/44 tests pass.
- Completed section-02-llm-provider: Added ModelConfig dataclass, get_chat_model() with provider dispatch (anthropic/bedrock/openai/gemini/ollama), fallback chain. 27/27 tests pass.
- Completed section-04-state-schemas: ResearchState, TradingState, SupervisorState TypedDicts with operator.add reducers. 7/7 tests pass.
- Completed section-03-agent-config: AgentConfig dataclass, YAML loader with duplicate detection, ConfigWatcher with file-watch + SIGHUP + cycle-boundary reload. 3 YAML files created. 22/22 tests pass.
- Completed section-12-docker-cleanup: Done as part of section-01.
- Completed section-10-observability: Removed CrewAI instrumentation, merged crew_tracing.py into tracing.py, added langfuse_trace_context() for LangGraph cycle tracing (Langfuse v4 uses @observe/OTEL not CallbackHandler), updated flush_util to call shutdown(). 17/17 tests pass.
- Completed section-06-supervisor-graph: 5-node linear StateGraph (health_check→diagnose_issues→execute_recovery→strategy_lifecycle→scheduled_tasks). Node factory pattern with closures binding LLM + config. RetryPolicy per node. Error catch → errors append. 6/6 tests pass.
- Completed section-05-tool-layer: Created tools/langchain/ (10 files, 18 @tool wrappers), tools/functions/ (4 files, node-callable async fns), TOOL_REGISTRY with get_tools_for_agent(). All YAML-referenced tools covered. Slimmed mcp_bridge/__init__ and tools/__init__ to avoid crewai_compat imports. Fixed stale docker+scaffolding tests. 16/16 new tests pass, 158/158 total pass.
- Completed section-07-research-graph: 8-node StateGraph (context_load→domain_selection→hypothesis_generation→signal_validation→[conditional]→backtest_validation→ml_experiment→strategy_registration→knowledge_update). Node factory pattern (make_X returning async closures). Conditional edge routes failed validations to END. 10/10 tests pass.
- Completed section-08-trading-graph: 11-node StateGraph with parallel branches and risk gate. Nodes renamed (plan_day/analyze_options/reflect) to avoid LangGraph state key collision. Parallel branches: plan_day→position_review+entry_scan→merge_parallel. SafetyGate enforced structurally via conditional edge (no bypass path). _safety_check_router and _risk_gate_router. 15/15 tests pass. 2311/2311 total pass (no regressions).
- Completed section-13-risk-safety: Renamed test_crewai_risk_safety.py→test_risk_safety.py with preserved SafetyGate unit tests. Added migration-specific tests: retry policy inspection (agent=3, tool=2, critical=none), structural risk gate enforcement (BFS all paths verify risk_sizing), SafetyGate pure-Python verification, kill switch routing. 19/19 tests pass.
- Completed section-11-runners: Rewrote all 3 runners to async (asyncio.run entry, graph.ainvoke, asyncio.wait_for timeout, asyncio.sleep). Shared run_loop() in trading_runner.py accepts graph_builder + initial_state_builder. GracefulShutdown.install_async() added for event loop signal handlers. Langfuse trace context per cycle. Thread ID for checkpointing. Legacy save_checkpoint preserved for transition. 30/30 tests pass.
- Completed section-14-testing: E2E smoke tests for all 3 graphs (compile + invoke), integration conftest with mock fixtures. 6/6 smoke tests pass. 86/86 total migration tests pass across all new test files.
- Completed section-09-rag-migration: Rewrote query.py (ChromaDB→pgvector with psycopg2), embeddings.py (removed ChromaDB interface), ingest.py (pgvector storage), migrate_memory.py (pgvector backend). Created scripts/migrate_chromadb_to_pgvector.py. Updated test_rag_pipeline.py (37 tests), test_memory_migration.py (14 tests), test_rag_degradation.py (3 tests). Fixed Dockerfile (added COPY src/), start.sh/status.sh (removed chromadb, renamed crew→graph), entrypoint.sh (passthrough mode). 54/54 RAG tests pass. 2399 total pass (0 regressions).
