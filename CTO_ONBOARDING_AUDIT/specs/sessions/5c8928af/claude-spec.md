# Phase 1: Safety Hardening — Synthesized Specification

## Mission

Eliminate all P0 existential risks in QuantStack before real capital is deployed. Ten items covering execution safety, data durability, LLM security, infrastructure hardening, and inter-graph coordination. System is currently in paper mode with no open positions — clean-slate deployment.

---

## 1.1 Mandatory Stop-Loss Enforcement

### Problem
`trade_service.py:212-223` allows `stop_price=None`. If any of 4 conditions fails (including Alpaca API hiccup), bracket order silently degrades to plain market order with zero protection. `alpaca_broker.py:execute_bracket()` catches all exceptions and falls back to plain `execute()` (line 223).

### Requirements
- Reject `OrderRequest` if `stop_price is None` at trade_service validation
- Enforce at OMS level — `order_lifecycle.py` rejects orders without stop
- If bracket order fails, place SL as separate contingent order — NEVER fall back to plain
- Implement using broker's native bracket order API where available
- Verify bracket legs after submission — query broker for active child orders
- Persist bracket leg IDs to DB (currently in-memory `Fill` only, lost on crash)
- **All brokers**: Retrofit E*Trade and PaperBroker with the same bracket/contingent-SL pattern (not just Alpaca)
- On startup: reconcile all open positions have active SL orders at broker
- **Auto-fix reconciliation**: If position missing stop, submit SL automatically, log event, continue trading (don't halt)

### Design Decisions
- Use `client_order_id` for idempotent order submission: `{strategy_id}_{symbol}_{timestamp}_{leg_type}`
- Bracket state machine separate from broker adapter: tracks intent (entry, stop, target) and statuses
- "Stop-loss priority" circuit breaker: when broker API degraded, stop new entries but continue retrying SL submissions
- No migration needed for existing positions — system has no open positions (paper mode, fresh start)

### Key Files
`src/quantstack/execution/trade_service.py`, `src/quantstack/execution/order_lifecycle.py`, `src/quantstack/brokers/alpaca_broker.py`, `src/quantstack/execution/execution_monitor.py`

---

## 1.2 Automated Database Backups

### Problem
ALL system state lives in PostgreSQL (60+ tables). Docker volumes are `driver: local` (single host). No `pg_dump` scheduled. No WAL archiving. Disk failure = total data loss.

### Requirements
- Daily `pg_dump` → local backup directory
- Enable WAL archiving for point-in-time recovery
- **Local + WAL for now** — offsite (S3) deferred to later phase
- Test restore procedure and document it
- Add backup verification to supervisor health checks
- Alerting if backup job fails

### Key Files
`docker-compose.yml`, new backup script, supervisor health checks

---

## 1.3 Prompt Injection Defense

### Problem
Portfolio context, knowledge base entries, and market data API responses injected directly into LLM prompts via f-strings with no sanitization. Agent executor (`agent_executor.py`) builds system messages from config and user messages with `json.dumps(portfolio_ctx, default=str)` directly interpolated. All 21 agents across trading/research/supervisor graphs are vulnerable.

### Requirements
- Replace f-string interpolation with structured XML-tagged templates
- Validate and escape all interpolated data at prompt boundaries
- Use field-level extraction instead of raw JSON dumps
- Add shared sanitization function (`graphs/prompt_safety.py`)
- **Dual LLM separation**: Enforce that research-facing LLMs (which process external market data, news, knowledge base) never have direct access to execution tools. Non-LLM graph node code mediates between research outputs and execution inputs.

### Design Decisions
- Research LLM produces structured recommendations only (symbol, direction, confidence, rationale)
- Trading graph's execution path validates recommendations via risk gate without passing raw research text to execution LLM
- Sanitization function strips common injection patterns before any external text reaches LLM context

### Key Files
`src/quantstack/graphs/trading/nodes.py:80-92`, `src/quantstack/graphs/research/nodes.py:219-224`, `src/quantstack/graphs/agent_executor.py`, all nodes that build prompts

---

## 1.4 Output Schema Validation with Retry

### Problem
All 21 agents return JSON. `parse_json_response()` (agent_executor.py:474-521) silently returns `{}` or `[]` on failure with no retry. Critical impacts: no plan → trades blind; entries missed; positions unmonitored; rejected entries treated as approved; risk assessment skipped.

### Requirements
- **One Pydantic model per agent** — 21 distinct models for maximum type safety and independent evolution
- On parse failure, retry once with "Please respond with valid JSON matching this schema: ..."
- Log all fallback events as warnings
- Add `agent_dead_letters` table: `(agent_name, cycle_id, raw_output, parse_error, timestamp)`
- Monitor DLQ frequency per agent — high rate = prompt quality issue

### Design Decisions
- 21 distinct Pydantic models, not grouped by shape — each agent evolves independently
- Retry includes the Pydantic model's JSON schema in the retry prompt
- Dead letter queue enables post-mortem analysis of LLM output quality

### Key Files
`src/quantstack/graphs/agent_executor.py`, `src/quantstack/tools/models.py`

---

## 1.5 Run Containers as Non-Root

### Problem
All containers run as root. Container compromise = root privileges.

### Requirements
- Add `USER` directive to Dockerfile
- Add `RUN useradd -r quantstack && chown -R quantstack:quantstack /app`
- Verify application still functions with reduced privileges (volume mounts, TA-Lib, etc.)

### Key Files
`Dockerfile`

---

## 1.6 Durable Checkpoints (PostgresSaver)

### Problem
All three graphs use LangGraph's `MemorySaver` — in-process memory only (`runners/trading_runner.py:161-167`). Container crash mid-cycle loses all intermediate state. Crash during `execute_entries` could leave approved trades never executed with no record.

### Requirements
- Switch to `PostgresSaver` from `langgraph-checkpoint-postgres` v3.x
- **Migrate entire db.py from psycopg2 to psycopg3** — eliminates maintaining two drivers
- Use `psycopg_pool.ConnectionPool` (not `from_conn_string` single connection) for production
- Run `setup()` as deployment step (not at app startup) to avoid race conditions
- Pin to `~=3.0` to avoid breaking schema changes

### Design Decisions
- Full psycopg3 migration (db.py) rather than parallel driver approach
- Connection pool sizing: min_size=4, max_size=10, max_lifetime=3600, max_idle=600
- `autocommit=True` and `prepare_threshold=0` required for PostgresSaver compatibility
- Crash recovery validated with integration test: kill container mid-cycle, restart, verify resume

### Key Files
`src/quantstack/runners/trading_runner.py`, `src/quantstack/runners/research_runner.py`, `src/quantstack/runners/supervisor_runner.py`, `src/quantstack/db.py`

---

## 1.7 Trading Graph Polls EventBus

### Problem
Supervisor publishes events (`IC_DECAY`, `DEGRADATION_DETECTED`, `REGIME_CHANGE`) to EventBus. Trading Graph never polls them. Trading continues on decayed strategies.

### Requirements
- Add `bus.poll()` at `safety_check` node for `IC_DECAY`, `RISK_EMERGENCY`
- **Also poll before `execute_entries` and `execute_exits`** — catches events mid-cycle (trading cycles ~5 min)
- Add `KILL_SWITCH_TRIGGERED` event in `kill_switch.trigger()`
- All graph loops poll `KILL_SWITCH_TRIGGERED` at cycle start

### Design Decisions
- Poll at cycle start + before irreversible actions (entries, exits) — not every node
- Max latency for critical events = time between safety_check and execute_entries (~2-3 min worst case)

### Key Files
`src/quantstack/graphs/trading/nodes.py`, `src/quantstack/execution/kill_switch.py`, all graph runners

---

## 1.8 Kill Switch Publishes to EventBus

### Problem
Kill switch sets DB flag + sentinel file but never publishes `KILL_SWITCH_TRIGGERED` event. Supervisor can't detect via normal polling loop.

### Requirements
- Add event publication in `kill_switch.trigger()`
- Event payload: reason, timestamp, trigger source

### Key Files
`src/quantstack/execution/kill_switch.py`

---

## 1.9 Containerize Scheduler

### Problem
Scheduler runs 5 critical jobs as a bare process in tmux. Crash = all jobs stop. No restart supervisor. Also has broken import chain (`ibkr_mcp` dependency).

### Requirements
- Add `scheduler` service to `docker-compose.yml` with health check and `unless-stopped` restart policy
- Health check: verify APScheduler is running and jobs are registered
- **Fix the ibkr_mcp import chain properly** — remove or isolate the dependency so scheduler imports cleanly

### Key Files
`docker-compose.yml`, `scripts/scheduler.py`

---

## 1.10 DB Transaction Isolation for Positions

### Problem
Execution monitor (async) and trading graph can race on the same position symbol. Default `READ COMMITTED` allows both to read stale state; one update overwrites the other.

### Requirements
- Use `SELECT FOR UPDATE` on position rows during modification
- **Only two concurrent writers**: execution monitor + trading graph (supervisor reads only)
- Verify no lost updates with integration test

### Design Decisions
- Row-level locking (`SELECT FOR UPDATE`) preferred over SERIALIZABLE isolation (less contention)
- Only position update queries need locking, not all DB queries

### Key Files
Position update queries across the codebase, `src/quantstack/db.py`

---

## Cross-Cutting Concerns

### psycopg3 Migration (affects 1.6, 1.10, and all DB-touching items)
The decision to migrate db.py from psycopg2 to psycopg3 is a foundational change that affects:
- Connection pool management
- Query placeholder syntax (`%s` vs `$1` or psycopg3's `%s` compatibility)
- Transaction semantics
- All existing tests that use DB fixtures

This should be sequenced early (before PostgresSaver and transaction isolation work).

### Testing Strategy
- pytest framework with markers (slow, integration, benchmark, requires_api)
- Existing test directory: `tests/{integration, core, graphs, coordination, unit, fixtures, regression, benchmarks, property}`
- Each item needs both unit tests and integration tests per acceptance criteria
- Crash recovery tests for 1.1 (bracket failure), 1.6 (checkpoint resume), 1.10 (concurrent writes)

### Parallelization
Items 1.1, 1.3 are the largest (2-3 days each). Items 1.5, 1.8 are smallest (0.5 day each). psycopg3 migration (part of 1.6) must precede 1.10. Items 1.7+1.8 are tightly coupled. Scheduler fix (1.9) is independent.
