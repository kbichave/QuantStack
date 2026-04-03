# TDD Plan: QuantStack CrewAI Migration

Testing framework: **pytest** + **pytest-asyncio** (existing project conventions). Tests in `tests/unit/` and `tests/integration/`.

---

## Section 1: Project Scaffolding and Docker Compose Stack

### Tests Before Implementation
```python
# Test: docker-compose.yml is valid YAML and defines all 8 services
# Test: each service has a health check defined
# Test: each crew service depends on infrastructure services
# Test: all services share the same Docker network
# Test: named volumes are defined for postgres, ollama, chromadb, langfuse-db
# Test: Dockerfile builds successfully (docker build --target test)
# Test: pyproject.toml includes crewai optional dependency group with all required packages
```

---

## Section 2: LLM Provider Management

### Tests Before Implementation
```python
# Test: get_model("heavy") returns correct model string for each provider (bedrock, anthropic, openai, gemini, ollama)
# Test: get_model("medium") returns correct model string per provider
# Test: get_model("light") returns correct model string per provider
# Test: get_model with unknown tier raises ValueError
# Test: LLM_PROVIDER env var selects primary provider
# Test: LLM_PROVIDER defaults to "bedrock" when unset
# Test: fallback chain activates when primary provider raises exception
# Test: fallback chain tries providers in order: bedrock -> anthropic -> openai -> ollama
# Test: fallback chain raises after all providers fail
# Test: provider config validates required env vars per provider (e.g., AWS keys for bedrock)
```

---

## Section 3: CrewAI Tool Wrappers

### Tests Before Implementation
```python
# For each tool wrapper module (22 modules, ~60 tools):

# Test: tool has @tool decorator with descriptive name
# Test: tool docstring is non-empty (CrewAI uses it as tool description)
# Test: tool calls underlying async function correctly (mock the async function)
# Test: tool returns JSON string (not dict)
# Test: tool handles underlying function raising exception (returns error JSON, doesn't crash)
# Test: nest_asyncio is applied before any tool wrapper runs

# Specific tool tests:
# Test: get_signal_brief_tool passes symbol param to get_signal_brief
# Test: execute_trade_tool passes all params (symbol, side, quantity, etc.)
# Test: get_portfolio_context_tool returns full context bundle (exposure, P&L, volatility, regime)
# Test: search_knowledge_base_tool passes query and optional filters to ChromaDB
# Test: remember_knowledge_tool writes to correct ChromaDB collection
```

---

## Section 4: Agent Definitions (YAML Configs)

### Tests Before Implementation
```python
# Test: agents.yaml for each crew is valid YAML
# Test: each agent has required fields: role, goal, backstory, llm
# Test: agent llm field references a valid tier variable ("{heavy_model}", "{medium_model}", "{light_model}")
# Test: no agent has allow_delegation=true (prevent circular delegation)
# Test: each agent lists only tools that exist in crewai_tools/
# Test: TradingCrew has all 10 required agents
# Test: ResearchCrew has all 4 required agents
# Test: SupervisorCrew has all 3 required agents
# Test: backstory for risk_analyst mentions "reason about" not "check threshold"
# Test: backstory for fund_manager mentions "correlation" and "concentration"
```

---

## Section 5: Task Definitions and Crew Workflows

### Tests Before Implementation
```python
# Test: tasks.yaml for each crew is valid YAML
# Test: each task has required fields: description, expected_output, agent
# Test: TradingCrew tasks are in correct order (safety_check first, persist_state last)
# Test: task context dependencies form a valid DAG (no cycles)
# Test: TradingCrew instantiates without error with mocked LLM
# Test: ResearchCrew instantiates without error with mocked LLM
# Test: SupervisorCrew instantiates without error with mocked LLM
# Test: TradingCrew.kickoff() completes one cycle with mock tools (all return canned JSON)
# Test: ResearchCrew.kickoff() completes one cycle with mock tools
# Test: position_review task has async_execution=True
# Test: entry_scan task has async_execution=True
```

---

## Section 6: RAG Pipeline (ChromaDB + Ollama)

### Tests Before Implementation
```python
# Test: OllamaEmbeddingFunction calls ollama.embed with correct model name
# Test: OllamaEmbeddingFunction returns list of float lists
# Test: ingest_memory_files reads all .md files from a test directory
# Test: ingest_memory_files chunks text with correct size/overlap
# Test: ingest_memory_files writes to correct ChromaDB collection per file type
# Test: ingest_memory_files is idempotent (second call doesn't duplicate)
# Test: search_knowledge_base returns top-N results with metadata
# Test: search_knowledge_base filters by ticker metadata when provided
# Test: search_knowledge_base handles empty collection gracefully
# Test: remember_knowledge writes to ChromaDB with correct metadata
# Test: ChromaDB PersistentClient uses absolute path (not relative)

# Use in-memory ChromaDB client for unit tests
```

---

## Section 7: Self-Healing System

### Tests Before Implementation
```python
# Test: GracefulShutdown handler sets should_stop=True on SIGTERM
# Test: GracefulShutdown handler sets should_stop=True on SIGINT
# Test: AgentWatchdog triggers callback after timeout_seconds
# Test: AgentWatchdog.end_cycle() cancels the timer
# Test: AgentWatchdog doesn't trigger if cycle completes in time
# Test: write_heartbeat creates/updates heartbeat file with current timestamp
# Test: check_health returns True when heartbeat is fresh
# Test: check_health returns False when heartbeat is stale (> max_age)
# Test: check_health returns False when heartbeat file doesn't exist
# Test: resilient_crew_kickoff retries on RateLimitError with exponential backoff
# Test: resilient_crew_kickoff fails after max_retries
# Test: resilient_crew_kickoff doesn't retry on non-retryable errors
# Test: DB reconnect wrapper retries on OperationalError
```

---

## Section 8: Observability (Self-Hosted Langfuse)

### Tests Before Implementation
```python
# Test: CrewAIInstrumentor.instrument() is called before any crew operations
# Test: langfuse.flush() is called in shutdown handler
# Test: @observe decorator is applied to runner main function
# Test: provider failover events are logged to Langfuse (mock Langfuse client)
# Test: strategy lifecycle events include full reasoning text
# Test: Langfuse environment variables are set in Docker Compose
```

---

## Section 9: Continuous Runner Architecture

### Tests Before Implementation
```python
# Test: runner creates fresh crew instance each cycle (not reusing)
# Test: runner respects should_stop flag (exits loop when True)
# Test: runner writes heartbeat after each successful cycle
# Test: runner sleeps for correct interval based on market hours
# Test: runner skips sleep if cycle took longer than interval
# Test: is_market_hours returns True during NYSE hours (Mon-Fri 9:30-16:00 ET)
# Test: is_market_hours returns False on weekends
# Test: is_market_hours returns False on NYSE holidays
# Test: runner catches exceptions per cycle without crashing the loop
# Test: runner logs cycle duration and result to Langfuse
```

---

## Section 10: Start/Stop/Status Scripts

### Tests Before Implementation
```python
# Test: start.sh checks for Docker and docker compose
# Test: start.sh checks for .env file existence
# Test: start.sh starts infrastructure before crew services
# Test: stop.sh sends graceful shutdown (not kill -9)
# Test: status.sh displays container status, heartbeats, and position count
# Test: startup waits for all health checks before starting crews
```

---

## Section 11: Memory Migration

### Tests Before Implementation
```python
# Test: ingestion reads strategy_registry.md and chunks correctly
# Test: ingestion reads workshop_lessons.md and tags as negative results
# Test: ingestion reads ticker-specific files and tags with ticker metadata
# Test: ingestion skips if collections are already non-empty
# Test: ingested content is retrievable via search query
# Test: ingested strategy knowledge returns relevant results for strategy-related queries
```

---

## Section 12: LLM-Reasoned Risk

### Tests Before Implementation
```python
# Test: risk agent receives complete context bundle (portfolio, market, RAG)
# Test: programmatic safety gate rejects position > 15% of equity
# Test: programmatic safety gate rejects when daily loss > 3%
# Test: programmatic safety gate rejects when ADV < 200K
# Test: programmatic safety gate rejects when gross exposure > 200%
# Test: programmatic safety gate PASSES valid LLM recommendations
# Test: risk decision uses temperature=0
# Test: risk decision output is valid JSON with required fields (symbol, recommended_size_pct, reasoning)
# Test: daily loss halt persists across process restarts (DB sentinel)
# Test: kill switch check runs before every crew cycle
```

---

## Section 13: Testing Strategy

### Tests Before Implementation
```python
# Test: E2E smoke test runs one full TradingCrew cycle with mock LLM and test DB
# Test: provider fallback test simulates primary failure and verifies secondary
# Test: graceful shutdown test sends SIGTERM and verifies state persistence
# Test: watchdog test triggers after configured timeout
# Test: RAG degradation test verifies crews continue when ChromaDB is down
```

---

## Section 14: Docker Resource Limits and Cost Estimation

### Tests Before Implementation
```python
# Test: docker-compose.yml defines memory limits for each service
# Test: total memory limits sum to < 10GB
# Test: logging configuration specifies max-size and max-file for all services
# Test: Langfuse retention cleanup function removes traces older than 30 days
```
