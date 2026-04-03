# CrewAI Migration Spec: QuantStack Orchestration Overhaul

## Goal

Convert the QuantStack trading system from Claude Code-based orchestration to a fully autonomous CrewAI-based system with observability, RAG, and local memory — capable of running unattended for weeks while continuously researching strategies and paper-trading via Alpaca.

## Current State

- **Orchestration**: Claude Code CLI sessions spawned by `start.sh` in tmux windows (trading, research, supervisor, scheduler). Each loop runs as a fresh `claude` invocation every 2-5 minutes.
- **Agents**: Defined as markdown files in `.claude/agents/` (fund-manager, market-intel, ml-scientist, quant-researcher, options-analyst, risk, trade-debater, etc.)
- **Prompts**: System prompts in `prompts/` (research_loop.md, trading_loop.md, context_loading.md, per-domain researcher prompts)
- **Tools**: Python functions in `src/quantstack/mcp/tools/` — called via `python3 -c "import ..."` from Claude Bash tool
- **State**: PostgreSQL database (`quantstack`) for strategies, positions, signals, etc.
- **Memory**: Markdown files in `.claude/memory/` — session handoffs, strategy registry, trade journal
- **Execution**: Alpaca API (paper mode default, live behind `USE_REAL_TRADING=true`)
- **Data**: Alpha Vantage (primary), Alpaca IEX (intraday), FinancialDatasets

## Target State

### 1. CrewAI Orchestration
- Replace all Claude Code CLI orchestration with CrewAI crews and agents
- Convert each `.claude/agents/*.md` agent definition to a CrewAI Agent with appropriate role, goal, backstory, and tools
- Convert `prompts/*.md` loop definitions to CrewAI Tasks with proper sequencing and delegation
- Implement two persistent CrewAI Crews: **ResearchCrew** and **TradingCrew** (mirroring current two-loop architecture)
- Add a **SupervisorCrew** for meta-oversight, health checks, and self-healing

### 2. Tool Compatibility
- Wrap all existing Python tools (`src/quantstack/mcp/tools/`) as CrewAI-compatible tools using `@tool` decorator or `BaseTool`
- Maintain the same function signatures and behavior
- Ensure risk_gate.py remains inviolable — wrap as a mandatory pre-execution tool

### 3. Observability with Langfuse
- Integrate Langfuse for full LLM call tracing, cost tracking, and evaluation
- Track: agent decisions, tool calls, strategy performance, research outcomes
- Set up dashboards for: daily P&L, strategy hit rate, agent error rate, API costs

### 4. RAG Components
- Add RAG pipeline for: research papers, strategy documentation, historical trade outcomes, market reports
- Use vector store (ChromaDB or similar) for document embeddings
- Enable agents to query historical context when making decisions

### 5. Local Memory with Ollama
- Run Ollama locally for embedding generation and lightweight inference
- Use for: RAG embeddings, sentiment analysis, document summarization
- Keep heavy reasoning on cloud LLMs (Claude/GPT) but offload commodity tasks to local models

### 6. Persistent Memory
- Replace `.claude/memory/*.md` files with structured CrewAI memory (short-term, long-term, entity memory)
- Back memory with PostgreSQL or vector store for durability
- Ensure session continuity without relying on markdown handoff files

### 7. Start/Stop/Status Scripts
- Update `start.sh` to launch CrewAI processes (not tmux + claude CLI)
- Add `stop.sh` for graceful shutdown with state persistence
- Add `status.sh` for health checks, active positions, crew status
- Implement proper process management (systemd services or supervisor)

### 8. Continuous Autonomous Operation
- System must run unattended for days/weeks
- Self-healing: detect and recover from crashes, API failures, rate limits
- Automatic strategy promotion pipeline: research -> backtest -> paper trade -> evaluate
- Alerting: critical events (large losses, system errors) sent via notification (email/Slack/webhook)

## Constraints

- Paper trading only by default (Alpaca paper mode)
- Must preserve all existing strategy logic and risk gates
- PostgreSQL remains the system of record
- Alpha Vantage rate limits (75/min premium) must be respected
- Must work on macOS (developer machine)

## Non-Goals (for this phase)

- Live trading enablement
- Multi-machine deployment
- Custom model fine-tuning
- Mobile app or web UI (CLI/dashboards only)
