# Integration Notes: Opus Review Feedback

## Integrating

### 1. Hard programmatic risk backstop (Critical) — INTEGRATING
The reviewer is right. Even with full LLM reasoning, there must be an outer boundary of hard limits that the LLM cannot exceed. This is defense-in-depth, not a contradiction of the LLM-first philosophy. The LLM reasons within bounds; the bounds prevent catastrophic outcomes from hallucination.

**Change:** Add Section 12.5 "Programmatic Safety Boundary" — preserve `risk_gate.py` as an outer envelope. LLM reasons about sizing, but risk gate rejects anything exceeding hard outer limits (15% per position, -3% daily halt, 200K ADV, 200% gross exposure). Temperature 0 + structured JSON output for risk decisions.

### 2. Cycle interval correction (High) — INTEGRATING
60-second cycles for an 11-task pipeline is physically impossible. Each LLM call takes 5-30 seconds.

**Change:** Update Section 9.2 to 5-minute trading cycles (market hours), 30-minute off-hours. This also cuts LLM costs by ~80%.

### 3. Complete tool module mapping (Medium) — INTEGRATING
The reviewer correctly identified ~15 tool modules without CrewAI wrappers.

**Change:** Add comprehensive tool mapping table in Section 3.2 covering ALL modules in `mcp/tools/`.

### 4. Scheduler and startup sequence (Medium) — INTEGRATING
The plan missed the scheduler (cron jobs) and the full start.sh sequence (13+ steps).

**Change:** Add scheduler tasks to SupervisorCrew (weekly community intel, monthly execution audit, data freshness checks). Expand Section 10.1 with full startup sequence.

### 5. `nest_asyncio` for async/sync boundary (Medium) — INTEGRATING
`asyncio.run()` will fail inside an existing event loop.

**Change:** Add `nest_asyncio` to dependencies. Apply patch at runner startup.

### 6. Shadow mode before cutover (Critical) — PARTIALLY INTEGRATING
The reviewer wants a parallel running period. The user explicitly chose big-bang. Compromise: add a "verification phase" where CrewAI runs in read-only/paper mode for 48 hours before enabling execution, but don't maintain the old Claude CLI system.

**Change:** Add Section 13.4 "Verification Phase" — 48-hour read-only mode where crews run but execution tools are no-ops.

### 7. Docker resource limits and machine specs (Medium) — INTEGRATING
**Change:** Add resource limits to docker-compose.yml definition and specify minimum 16GB RAM.

### 8. Langfuse key generation and log management (Medium) — INTEGRATING
**Change:** Add concrete Langfuse setup procedure and log rotation config.

### 9. Cost estimation (Medium) — INTEGRATING
**Change:** Add cost estimation section. With 5-minute cycles, cost drops to ~$50-100/month.

### 10. Testing improvements (Medium) — INTEGRATING
**Change:** Add E2E smoke test, soak test spec, fallback chain test, shutdown test.

## NOT Integrating

### 1. Parallel research workers
The current system runs 2 research workers as a workaround for Claude CLI session limits. CrewAI can use `async_execution=True` on tasks within a single crew, achieving parallelism without multiple processes. If capacity is insufficient, we can add a second research-crew container later. Not worth the complexity upfront.

### 2. Full rollback plan to tmux/Claude CLI
The user explicitly chose big-bang. Maintaining the old system as a rollback target creates maintenance burden and divided attention. The verification phase (48-hour read-only) is the compromise.

### 3. Separate Postgres for Langfuse
The reviewer noted "three separate Postgres instances." Langfuse requires its own DB schema. Keeping it in a separate Postgres container is the cleanest approach. The overhead is minimal (Langfuse-db is lightweight). Not changing this.

### 4. CrewAI memory storage backend concern
CrewAI's unified Memory uses its configured embedding provider (Ollama in our case). It doesn't add another database — it uses the embedder directly. No change needed.
