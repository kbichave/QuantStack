<!-- PROJECT_CONFIG
runtime: python-uv
test_command: uv run pytest tests/
END_PROJECT_CONFIG -->

<!-- SECTION_MANIFEST
section-01-scaffolding
section-02-llm-providers
section-03-tool-wrappers
section-04-agent-definitions
section-05-crew-workflows
section-06-rag-pipeline
section-07-self-healing
section-08-observability
section-09-runners
section-10-scripts
section-11-memory-migration
section-12-risk-safety
section-13-testing
section-14-docker-resources
END_MANIFEST -->

# Implementation Sections Index

## Dependency Graph

| Section | Depends On | Blocks | Parallelizable |
|---------|------------|--------|----------------|
| section-01-scaffolding | - | all | Yes |
| section-02-llm-providers | 01 | 04, 05, 09 | Yes |
| section-03-tool-wrappers | 01 | 04, 05 | Yes |
| section-04-agent-definitions | 02, 03 | 05 | No |
| section-05-crew-workflows | 04 | 09 | No |
| section-06-rag-pipeline | 01 | 04, 11 | Yes |
| section-07-self-healing | 01 | 09 | Yes |
| section-08-observability | 01 | 09 | Yes |
| section-09-runners | 05, 07, 08 | 10 | No |
| section-10-scripts | 09 | 14 | No |
| section-11-memory-migration | 06 | 13 | No |
| section-12-risk-safety | 03 | 05, 13 | Yes |
| section-13-testing | 09, 11, 12 | - | No |
| section-14-docker-resources | 10 | - | No |

## Execution Order

1. **Batch 1:** section-01-scaffolding (no dependencies — project structure, Docker Compose, pyproject.toml)
2. **Batch 2:** section-02-llm-providers, section-03-tool-wrappers, section-06-rag-pipeline, section-07-self-healing, section-08-observability, section-12-risk-safety (all depend only on 01, parallel)
3. **Batch 3:** section-04-agent-definitions, section-11-memory-migration (depend on batch 2 outputs)
4. **Batch 4:** section-05-crew-workflows (depends on 04 + 12)
5. **Batch 5:** section-09-runners (depends on 05, 07, 08)
6. **Batch 6:** section-10-scripts, section-13-testing (depend on 09, 11)
7. **Batch 7:** section-14-docker-resources (final, depends on 10)

## Section Summaries

### section-01-scaffolding
Project structure, Docker Compose stack (8 services), Dockerfile, pyproject.toml with CrewAI dependencies, `.env.example`.

### section-02-llm-providers
`src/quantstack/llm/provider.py` and `config.py`. Provider selection, model tier map, fallback chain (Bedrock → Anthropic → OpenAI → Ollama), env var validation.

### section-03-tool-wrappers
All 22 CrewAI tool wrapper modules in `src/quantstack/crewai_tools/` (~60 tools). Wraps existing async functions with `@tool` decorator, `nest_asyncio`, JSON serialization.

### section-04-agent-definitions
YAML agent configs for TradingCrew (10 agents), ResearchCrew (4 agents), SupervisorCrew (3 agents). Role, goal, backstory, tool assignments, model tier assignment.

### section-05-crew-workflows
Task definitions and crew classes. TradingCrew (11 sequential tasks with parallel sub-tasks), ResearchCrew (8 hierarchical tasks), SupervisorCrew (5 sequential tasks + scheduler).

### section-06-rag-pipeline
`src/quantstack/rag/` — ChromaDB client setup, Ollama embedding wrapper, ingestion pipeline (chunking, metadata), retrieval functions, 3 collections (trade_outcomes, strategy_knowledge, market_research).

### section-07-self-healing
`src/quantstack/health/` — heartbeat, watchdog, graceful shutdown, exponential backoff retry wrapper, DB reconnect, degraded mode detection.

### section-08-observability
Langfuse integration — CrewAIInstrumentor setup, `@observe` decorators, provider failover tracing, cost tracking, event bus listener, flush-on-shutdown.

### section-09-runners
`src/quantstack/runners/` — trading_runner.py, research_runner.py, supervisor_runner.py. Continuous loop with fresh crew instances, market hours detection, heartbeat writing, checkpoint persistence.

### section-10-scripts
Updated start.sh (13-step Docker Compose launch), stop.sh (graceful shutdown), status.sh (container health + crew status + Langfuse URL).

### section-11-memory-migration
One-time ingestion of `.claude/memory/` files into ChromaDB. Markdown parsing, chunking, metadata extraction, idempotent upsert.

### section-12-risk-safety
Programmatic safety boundary — preserved `risk_gate.py` as outer envelope (15% position, -3% daily halt, 200K ADV, 200% exposure). Structured JSON output for risk decisions. Temperature 0. Kill switch persistence.

### section-13-testing
E2E smoke test, provider fallback test, graceful shutdown test, watchdog test, RAG degradation test, soak test spec. 48-hour verification phase configuration.

### section-14-docker-resources
Docker resource limits per service, log rotation config, Langfuse retention policy, cost estimation validation.
