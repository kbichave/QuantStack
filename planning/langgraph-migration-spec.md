# CrewAI to LangGraph Migration Spec

## Goal

Completely remove CrewAI from QuantStack and replace it with LangGraph. Keep LangFuse for observability. The result should be a cleaner, more maintainable agent orchestration layer using LangGraph's state-machine paradigm instead of CrewAI's crew/agent/task abstraction.

## Why

- CrewAI adoption is declining; LangGraph (by LangChain) is winning the multi-agent orchestration space
- LangGraph's explicit state graph model is a better fit for QuantStack's deterministic trading workflows (risk gates, regime checks, sequential pipeline stages)
- CrewAI's magic (decorators, YAML configs, hidden prompt injection) makes debugging hard
- LangGraph has first-class LangFuse integration (we already use LangFuse)
- LangGraph supports streaming, checkpointing, and human-in-the-loop natively

## Current CrewAI Footprint

### Source code (must be replaced with LangGraph equivalents)
- `src/quantstack/crews/` — 4 crews: research, risk, trading, supervisor
  - Each has `crew.py` + `config/agents.yaml` + `config/tasks.yaml`
  - `crews/__init__.py` re-exports Pydantic schemas (these stay, they're pure Python)
  - `crews/decoder_crew.py`, `crews/registry.py`, `crews/schemas.py` — pure Python utilities
- `src/quantstack/crewai_tools/` — 20+ tool wrapper files that bridge async MCP tools to CrewAI's sync `BaseTool`
  - `_async_bridge.py` — nest_asyncio hack for CrewAI's event loop
  - Each file wraps one domain (signal, risk, data, etc.) as CrewAI `BaseTool` subclasses
- `src/quantstack/crewai_compat.py` — compatibility shim
- `src/quantstack/observability/crew_tracing.py` — CrewAI-specific Langfuse trace helpers (keep the trace functions, remove CrewAI references in docstrings/names)
- `src/quantstack/llm/provider.py` — returns model strings "for a CrewAI Agent's llm parameter" (refactor for LangGraph)

### Runners (must be refactored)
- `src/quantstack/runners/research_runner.py` — imports `ResearchCrew`, calls `.crew()` in a loop
- `src/quantstack/runners/trading_runner.py` — similar pattern for trading
- `src/quantstack/runners/supervisor_runner.py` — supervisor crew runner

### Tests (must be updated/rewritten)
- `tests/unit/test_crewai_tools/` — 6 test files for CrewAI tool wrappers
- `tests/unit/test_crewai_risk_safety.py`
- `tests/unit/test_agent_definitions.py`
- `tests/unit/test_scaffolding.py` — likely references CrewAI structure
- `tests/integration/` — e2e smoke, conftest reference CrewAI

### Config/Infra (must be cleaned)
- `pyproject.toml` — remove `crewai` and `crewai-tools` dependencies, add `langgraph`, `langchain-core`, `langchain-anthropic` (or `langchain-litellm`)
- `Dockerfile` — may reference CrewAI
- `docker-compose.yml` — may reference CrewAI
- `.env.example` — CrewAI-specific env vars

### Docs (delete entirely)
- `docs/crewai_docs_md/` — ~180 files of crawled CrewAI documentation (not our code, just reference)
- `docs/architecture/quant_pod.md` — references CrewAI architecture

### Memory/planning (clean up references)
- Previous planning session at `planning/sessions/e4625d34/` can be archived or deleted

## What STAYS (do not touch)

- **LangFuse** — all observability stays. `src/quantstack/observability/tracing.py`, `instrumentation.py`, `flush_util.py`, `health/langfuse_retention.py`
- **Pydantic schemas** — `src/quantstack/shared/schemas.py` and re-exports
- **Core trading logic** — `src/quantstack/execution/`, `src/quantstack/signal_engine/`, `src/quantstack/autonomous/`
- **MCP tools** — `src/quantstack/mcp/tools/` (the actual implementations the CrewAI wrappers call)
- **Risk gate** — `src/quantstack/execution/risk_gate.py` (LAW — never modify)
- **Database layer** — `src/quantstack/db.py`
- **Claude agent definitions** — `.claude/agents/*.md` (these are for Claude Code, not CrewAI)
- **Prompts** — `prompts/` (trading_loop.md, research_loop.md — these drive the Claude-native loops)

## LangGraph Target Architecture

### Graphs replace Crews
Each CrewAI crew becomes a LangGraph `StateGraph`:
- `ResearchGraph` — research pipeline (context loading -> domain selection -> hypothesis -> validation -> backtest -> ML experiment -> registration -> community scan)
- `TradingGraph` — trading pipeline (market scan -> signal generation -> risk check -> debate -> fund approval -> execution)
- `RiskGraph` — risk assessment (position sizing, VaR, correlation, stress test)
- `SupervisorGraph` — orchestrates research + trading graphs, handles scheduling

### Tool layer
- Delete `src/quantstack/crewai_tools/` entirely
- LangGraph nodes call the existing MCP tool implementations directly (`src/quantstack/mcp/tools/`)
- Or use LangChain's `@tool` decorator for simple wrappers where needed
- No more `nest_asyncio` hack — LangGraph is natively async

### LLM provider
- Refactor `src/quantstack/llm/provider.py` to return LangChain `ChatModel` instances instead of model strings
- Keep the tier system (heavy/light) and fallback chain
- LiteLLM integration via `langchain-litellm` or direct `ChatLiteLLM`

### Observability
- LangFuse already works with LangChain/LangGraph via `langfuse.callback.CallbackHandler`
- Replace CrewAI auto-instrumentation with LangGraph callback handler
- Keep custom trace functions in `crew_tracing.py` (rename file to `graph_tracing.py` or just `tracing.py`)

### State management
- LangGraph `TypedDict` state replaces CrewAI's implicit task result passing
- Checkpointing via LangGraph's built-in `MemorySaver` or PostgreSQL checkpointer (we already have PG)

## Constraints

- Must preserve the "two loops" architecture (research loop + trading loop running independently)
- Risk gate enforcement is non-negotiable — must be a mandatory node in every trading graph
- Paper mode default must be preserved
- Audit trail (every decision logged with reasoning) must be maintained via LangFuse
- The Claude Code agent spawning system (`.claude/agents/`) is SEPARATE from this and must not be affected
- Existing MCP tool implementations must not be modified — only the wrapper layer changes
