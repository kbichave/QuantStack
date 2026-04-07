# Phase 1: Safety Hardening — Deep Plan Spec

**Timeline:** Week 1-2
**Effort:** 12-14 days (parallelizable to ~8 days with 2 engineers)
**Gate:** Every position has a broker-held stop order. DB backed up. Prompts sanitized.

---

## Context

This spec is part of the QuantStack CTO Onboarding Audit implementation plan (164 findings, overall grade C-). Phase 1 addresses **existential risks to capital, data, and legal standing**. If any of these remain unfixed when real money is deployed, a single bad day could wipe the account, corrupt all system state, or expose the system to adversarial manipulation.

**Full audit reference:** [`CTO_ONBOARDING_AUDIT/`](../README.md)
**Primary audit section:** [`01_SAFETY_HARDENING.md`](../01_SAFETY_HARDENING.md)
**Supporting sections:** [`04_OPERATIONAL_RESILIENCE.md` §4.5](../04_OPERATIONAL_RESILIENCE.md) (transaction isolation)

---

## Objective

Eliminate all P0 existential risks: unprotected positions, data loss, prompt injection, silent parse failures, exposed database, crash-state corruption, broken inter-graph communication, uncontainerized scheduler.

---

## Items

### 1.1 Mandatory Stop-Loss Enforcement

- **Findings:** CTO C1, C2, C3, DO-2 | **Severity:** CRITICAL | **Effort:** 2-3 days
- **Audit section:** [`01_SAFETY_HARDENING.md` §1.1](../01_SAFETY_HARDENING.md)
- **Problem:** `trade_service.py:212-223` allows `stop_price=None`. If any of 4 conditions fails (including Alpaca API hiccup), bracket order silently degrades to plain market order with zero protection. `alpaca_broker.py:execute_bracket()` catches all exceptions and falls back to plain `execute()`.
- **Fix:**
  1. Reject `OrderRequest` if `stop_price is None` at `trade_service.py` validation
  2. Enforce at OMS level — `order_lifecycle.py` rejects orders without stop
  3. If bracket order fails, place SL as separate contingent order — NEVER fall back to plain
  4. Implement `execute_bracket()` using Alpaca's native bracket order API
  5. Verify bracket legs after submission — query broker for active child orders
  6. Persist bracket leg IDs to DB (currently in-memory `Fill` only, lost on crash)
  7. On startup: reconcile all open positions have active SL orders at broker
- **Key files:** `src/quantstack/execution/trade_service.py`, `src/quantstack/execution/order_lifecycle.py`, `src/quantstack/brokers/alpaca_broker.py`, `src/quantstack/execution/execution_monitor.py`
- **Acceptance criteria:**
  - [ ] No `OrderRequest` can be created with `stop_price=None`
  - [ ] Bracket failure results in separate contingent SL order, never plain order
  - [ ] Startup reconciliation verifies all open positions have broker-side stop orders
  - [ ] Unit tests: trigger kill switch → verify `execute_order` rejects; set risk gate violation → verify order blocked

### 1.2 Automated Database Backups

- **Finding:** CTO OC1 | **Severity:** CRITICAL | **Effort:** 1 day
- **Audit section:** [`01_SAFETY_HARDENING.md` §1.2](../01_SAFETY_HARDENING.md)
- **Problem:** ALL system state lives in PostgreSQL (60+ tables). Docker volumes are `driver: local` (single host). No `pg_dump` scheduled. No WAL archiving. Disk failure = total data loss.
- **Fix:**
  1. Add daily `pg_dump` → local backup directory
  2. Upload to S3 (or equivalent offsite storage)
  3. Enable WAL archiving for point-in-time recovery
  4. Test restore monthly — document the procedure
  5. Add backup verification to supervisor health checks
- **Key files:** `docker-compose.yml`, new backup script, supervisor health checks
- **Acceptance criteria:**
  - [ ] Daily automated `pg_dump` running and verified
  - [ ] Backups stored offsite (S3 or equivalent)
  - [ ] Restore procedure documented and tested at least once
  - [ ] Alerting if backup job fails

### 1.3 Prompt Injection Defense

- **Finding:** CTO LC1 | **Severity:** CRITICAL | **Effort:** 2-3 days
- **Audit section:** [`01_SAFETY_HARDENING.md` §1.3](../01_SAFETY_HARDENING.md)
- **Problem:** Portfolio context, knowledge base entries, and market data API responses injected directly into LLM prompts via f-strings with no sanitization. Adversarial knowledge base entry or compromised API response could inject instructions that manipulate trading decisions.
- **Fix:**
  1. Replace f-string interpolation with structured XML-tagged templates
  2. Validate and escape all interpolated data at prompt boundaries
  3. Use field-level extraction instead of raw JSON dumps
  4. Add input sanitization function shared across all nodes (new `graphs/prompt_safety.py`)
- **Key files:** `src/quantstack/graphs/trading/nodes.py:80-92`, `src/quantstack/graphs/research/nodes.py:219-224`, all nodes that build prompts
- **Acceptance criteria:**
  - [ ] No raw f-string interpolation of external data into prompts
  - [ ] Structured templates with clear field boundaries for all agent prompts
  - [ ] Sanitization function applied to all DB-sourced and API-sourced data before prompt inclusion

### 1.4 Output Schema Validation with Retry

- **Finding:** CTO LC2 | **Severity:** CRITICAL | **Effort:** 2 days
- **Audit section:** [`01_SAFETY_HARDENING.md` §1.4](../01_SAFETY_HARDENING.md)
- **Problem:** All 21 agents return JSON. Parse failure → `parse_json_response()` silently returns `{}` or `[]` with no retry. Critical impacts: no plan → trades blind; entries missed entirely; active positions unmonitored; rejected entries treated as approved; risk assessment silently skipped.
- **Fix:**
  1. Add Pydantic models per agent output (partially exists in `tools/models.py`)
  2. On parse failure, retry once with "Please respond with valid JSON matching this schema: ..."
  3. Log all fallback events as warnings
  4. Add `agent_dead_letters` table: `(agent_name, cycle_id, raw_output, parse_error, timestamp)`
  5. Monitor DLQ frequency per agent — high rate = prompt quality issue
- **Key files:** Agent executor, `src/quantstack/tools/models.py`, `parse_json_response()` utility
- **Acceptance criteria:**
  - [ ] Every agent has a Pydantic output model
  - [ ] Parse failures trigger one retry with schema hint
  - [ ] All fallback events logged and queryable
  - [ ] Dead letter queue populated for post-mortem analysis

### 1.5 Run Containers as Non-Root

- **Finding:** QS-I1 | **Severity:** CRITICAL | **Effort:** 0.5 day
- **Audit section:** [`01_SAFETY_HARDENING.md` §1.5](../01_SAFETY_HARDENING.md)
- **Problem:** All containers run as root. Container compromise = root privileges.
- **Fix:**
  1. Add `USER` directive to Dockerfile
  2. Add `RUN useradd -r quantstack && chown -R quantstack:quantstack /app`
- **Key files:** `Dockerfile`
- **Acceptance criteria:**
  - [ ] All containers run as non-root user
  - [ ] Application still functions correctly with reduced privileges

### 1.6 Durable Checkpoints (PostgresSaver)

- **Finding:** CTO GC1 | **Severity:** CRITICAL | **Effort:** 1 day
- **Audit section:** [`01_SAFETY_HARDENING.md` §1.6](../01_SAFETY_HARDENING.md)
- **Problem:** All three graphs use LangGraph's `MemorySaver` — in-process memory only. Container crash mid-cycle loses all intermediate state. Crash during `execute_entries` could leave approved trades never executed with no record.
- **Fix:** Switch to `PostgresSaver` for durable checkpointing. LangGraph supports this natively.
- **Key files:** Graph runner initialization for all 3 graphs
- **Acceptance criteria:**
  - [ ] All three graph runners use `PostgresSaver` (or equivalent durable checkpointer)
  - [ ] Container crash during any node results in clean resume from last checkpoint
  - [ ] Verified with integration test: kill container mid-cycle, restart, verify state consistent

### 1.7 Trading Graph Polls EventBus

- **Findings:** CTO AC1, AC2 | **Severity:** CRITICAL | **Effort:** 1 day
- **Audit section:** [`01_SAFETY_HARDENING.md` §1.7](../01_SAFETY_HARDENING.md)
- **Problem:** Supervisor publishes events (`IC_DECAY`, `DEGRADATION_DETECTED`, `REGIME_CHANGE`) to EventBus. Trading Graph never polls them. Supervisor shouting into void. Trading continues on decayed strategies for up to 5 minutes.
- **Fix:**
  1. Add `bus.poll()` at `safety_check` node for `IC_DECAY`, `RISK_EMERGENCY` (~5-10 lines)
  2. Add `KILL_SWITCH_TRIGGERED` event in `kill_switch.trigger()` (~3 lines)
  3. All graph loops poll `KILL_SWITCH_TRIGGERED` at cycle start (~5 lines)
- **Key files:** `src/quantstack/graphs/trading/nodes.py`, `src/quantstack/execution/kill_switch.py`, all graph runners
- **Acceptance criteria:**
  - [ ] Trading graph receives and acts on supervisor events within one cycle
  - [ ] Kill switch trigger publishes event visible to all graphs
  - [ ] IC_DECAY event halts trading of the affected strategy

### 1.8 Kill Switch Publishes to EventBus

- **Finding:** CTO AC2 | **Severity:** CRITICAL | **Effort:** 0.5 day
- **Audit section:** [`01_SAFETY_HARDENING.md` §1.7](../01_SAFETY_HARDENING.md) (combined with 1.7)
- **Problem:** Kill switch sets DB flag + sentinel file but never publishes `KILL_SWITCH_TRIGGERED` event. Supervisor can't detect via normal polling loop.
- **Fix:** Add event publication in `kill_switch.trigger()`
- **Key files:** `src/quantstack/execution/kill_switch.py`
- **Acceptance criteria:**
  - [ ] Kill switch trigger publishes `KILL_SWITCH_TRIGGERED` event

### 1.9 Containerize Scheduler

- **Finding:** CTO OC2 | **Severity:** CRITICAL | **Effort:** 1 day
- **Audit section:** [`01_SAFETY_HARDENING.md` §1.8](../01_SAFETY_HARDENING.md)
- **Problem:** Scheduler runs 13 critical jobs as a bare process in tmux. Crash = all jobs stop (data refresh, strategy promotion, EOD sync). No restart supervisor.
- **Fix:** Add `scheduler` service to `docker-compose.yml` with health check and `unless-stopped` restart policy. Health check: verify APScheduler is running and jobs are registered.
- **Key files:** `docker-compose.yml`, `scripts/scheduler.py`
- **Acceptance criteria:**
  - [ ] Scheduler runs as Docker container with `unless-stopped` restart
  - [ ] Health check verifies APScheduler process is alive and jobs registered
  - [ ] Crash of scheduler container → automatic restart within 60s

### 1.10 DB Transaction Isolation for Positions

- **Finding:** QS-I3 | **Severity:** CRITICAL | **Effort:** 1 day
- **Audit section:** [`04_OPERATIONAL_RESILIENCE.md` §4.5](../04_OPERATIONAL_RESILIENCE.md)
- **Problem:** Two agents simultaneously read and update a position (execution monitor tightening stop while trading graph sizing new entry on same symbol). Default `READ COMMITTED` allows both to read stale state; one update overwrites the other.
- **Fix:** Use `SELECT FOR UPDATE` on position rows during modification. Alternatively, set isolation to `SERIALIZABLE` for position update connection pool.
- **Key files:** Position update queries across the codebase
- **Acceptance criteria:**
  - [ ] Position updates use row-level locking (`SELECT FOR UPDATE`)
  - [ ] Concurrent position modifications serialized correctly
  - [ ] No lost updates verified with integration test

---

## Dependencies

- **Depends on:** Phase 0 (quick wins should be done first, but not hard dependency)
- **Blocks:** Phase 2 (statistics), Phase 3 (ops), Phase 4 (agent arch) — all assume safety layer is in place

---

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| 1.1: Existing positions with `stop_price=None` break on new validation | Add migration to compute default ATR-based stops for existing positions |
| 1.3: Prompt injection defense may change agent behavior (different formatting) | Run parallel: old prompts + new prompts for 2 cycles, compare outputs |
| 1.5: Non-root containers may have permission issues with volume mounts | Test with Docker volumes first; fix ownership in Dockerfile |
| 1.6: PostgresSaver migration from MemorySaver | No existing checkpoint data to migrate; clean cutover is safe |

---

## Validation Plan

1. **Stop-loss (1.1):** Submit order with `stop_price=None` → must reject. Submit bracket → kill Alpaca API mid-call → verify separate SL placed.
2. **Backups (1.2):** Run `pg_dump`, corrupt DB, restore, verify all 60+ tables intact.
3. **Prompt injection (1.3):** Insert adversarial knowledge base entry with "IGNORE ALL INSTRUCTIONS" → verify sanitization strips it.
4. **Output validation (1.4):** Feed malformed JSON to `parse_json_response()` → verify retry fires and DLQ populated.
5. **Security (1.5):** `docker exec` into running container → verify `whoami` returns non-root.
6. **Checkpoints (1.6):** Start trading cycle → `docker kill` mid-cycle → restart → verify resume from last node.
7. **EventBus (1.7-1.8):** Trigger kill switch → verify all 3 graphs receive event within 1 cycle.
8. **Scheduler (1.9):** `docker kill quantstack-scheduler` → verify auto-restart + jobs resume.
9. **Transaction isolation (1.10):** Two concurrent position updates on same symbol → verify no lost writes.

---

## Parallelization Plan

```
Engineer A (Week 1):       Engineer B (Week 1):
  1.1 Stop-loss (3 days)     1.3 Prompt injection (3 days)
  1.7+1.8 EventBus (1 day)   1.4 Output validation (2 days)

Engineer A (Week 2):       Engineer B (Week 2):
  1.2 DB backups (1 day)     1.5 Non-root (0.5 day)
  1.6 PostgresSaver (1 day)  1.9 Scheduler (1 day)
  1.10 Tx isolation (1 day)  Integration testing (1 day)
```
