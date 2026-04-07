# 01 — Safety Hardening: Must Fix Before Real Capital

**Priority:** P0 — Existential Risk
**Timeline:** Week 1-2
**Gate:** Every position has a broker-held stop order. DB backed up. Prompt injection mitigated.

---

## Why This Section Comes First

These findings represent existential risk to capital, data, and legal standing. If any of these remain unfixed when real money is deployed, a single bad day could wipe the account, corrupt all system state, or expose the system to adversarial manipulation. Nothing else matters until these are resolved.

---

## 1.1 Mandatory Stop-Loss Enforcement

**Finding IDs:** CTO C1, CTO C2, CTO C3, DO-2
**Severity:** CRITICAL
**Effort:** 2-3 days

### The Problem

The `ATRPositionSizer` requires a stop_loss parameter, but the entry point in `trade_service.py` allows `stop_price=None`. An LLM agent can submit a trade with no stop-loss and it passes the risk gate. For an autonomous system with no human oversight, this is an existential risk.

**The code path today:**

```python
# trade_service.py:212-223
# ALL FOUR conditions must be true for bracket order:
if (stop_price is not None                              # ← default is None
    and target_price is not None                         # ← default is None
    and getattr(broker, "supports_bracket_orders", ...)()
    and hasattr(broker, "execute_bracket")):
    fill = broker.execute_bracket(order, stop_price, target_price)
else:
    fill = broker.execute(order)  # ← plain market order, ZERO protection
```

If ANY condition fails — including a simple Alpaca API hiccup — the system silently falls back to a plain order with no stop-loss, no take-profit. The agent thinks protection is in place. It isn't.

**Additionally:** `alpaca_broker.py:execute_bracket()` catches all exceptions and falls back to plain `execute()`:

```python
except Exception as e:
    logger.warning(f"Bracket submission failed ({e}), falling back to plain order")
    return self.execute(req)  # ← ANY hiccup drops protection silently
```

### The Fix

| Step | Action | Location |
|------|--------|----------|
| 1 | Reject `OrderRequest` if `stop_price is None` | `trade_service.py` — validation at entry |
| 2 | Enforce at OMS level — `order_lifecycle.py` rejects orders without stop | `order_lifecycle.py` |
| 3 | If bracket order fails, place SL as separate contingent order — NEVER fall back to plain | `alpaca_broker.py` |
| 4 | Implement `execute_bracket()` using Alpaca's native bracket order API | `alpaca_broker.py` |
| 5 | Verify bracket legs after submission — query broker for active child orders | `trade_service.py` |
| 6 | Persist bracket leg IDs to DB (currently in-memory `Fill` only, lost on crash) | `trade_service.py` |
| 7 | On startup: reconcile all open positions have active SL orders at broker | `execution_monitor.py` |

### Acceptance Criteria

- [ ] No `OrderRequest` can be created with `stop_price=None`
- [ ] Bracket failure results in separate contingent SL order, never plain order
- [ ] Startup reconciliation verifies all open positions have broker-side stop orders
- [ ] Unit tests: trigger kill switch → verify `execute_order` rejects; set risk gate violation → verify order blocked

---

## 1.2 Automated Database Backups

**Finding ID:** CTO OC1
**Severity:** CRITICAL
**Effort:** 1 day

### The Problem

ALL system state lives in PostgreSQL — 60+ tables including positions, strategies, fills, signal state, research queue, knowledge base. Docker volumes are `driver: local` (single host). No `pg_dump` scheduled. No backup documentation. No WAL archiving.

If the Docker host disk fails, everything is lost. This is the single highest-risk infrastructure finding in the entire audit.

### The Fix

| Step | Action |
|------|--------|
| 1 | Add daily `pg_dump` → local backup directory |
| 2 | Upload to S3 (or equivalent offsite storage) |
| 3 | Enable WAL archiving for point-in-time recovery |
| 4 | Test restore monthly — document the procedure |
| 5 | Add backup verification to supervisor health checks |

### Acceptance Criteria

- [ ] Daily automated `pg_dump` running and verified
- [ ] Backups stored offsite (S3 or equivalent)
- [ ] Restore procedure documented and tested at least once
- [ ] Alerting if backup job fails

---

## 1.3 Prompt Injection Defense

**Finding ID:** CTO LC1
**Severity:** CRITICAL
**Effort:** 2-3 days

### The Problem

Portfolio context, knowledge base entries, and market data API responses are injected directly into LLM prompts via f-strings with no sanitization:

```python
# trading/nodes.py:80-92 — VULNERABLE
prompt = f"Portfolio: {json.dumps(portfolio_ctx, default=str)}\n"

# research/nodes.py:219-224 — VULNERABLE
prefetched_context = f"{str(knowledge_text)[:2000]}\n"  # From DB, unsanitized
```

An adversarial knowledge base entry, compromised API response, or malicious portfolio state could inject instructions that manipulate trading decisions. For an autonomous system managing real capital, this is an existential risk.

### The Fix

| Step | Action | Location |
|------|--------|----------|
| 1 | Replace f-string interpolation with structured XML-tagged templates | `trading/nodes.py`, `research/nodes.py` |
| 2 | Validate and escape all interpolated data at prompt boundaries | All nodes that build prompts |
| 3 | Use field-level extraction instead of raw JSON dumps | `trading/nodes.py:80-92` |
| 4 | Add input sanitization function shared across all nodes | New `graphs/prompt_safety.py` |

### Acceptance Criteria

- [ ] No raw f-string interpolation of external data into prompts
- [ ] Structured templates with clear field boundaries for all agent prompts
- [ ] Sanitization function applied to all DB-sourced and API-sourced data before prompt inclusion

---

## 1.4 Output Schema Validation with Retry

**Finding ID:** CTO LC2
**Severity:** CRITICAL
**Effort:** 2 days

### The Problem

All 21 agents return JSON. When parsing fails, `parse_json_response()` silently returns a fallback (`{}` or `[]`) with no retry. Critical impacts:

| Node | Expected Output | On Parse Failure | Impact |
|------|----------------|-----------------|--------|
| daily_plan | `{"plan": "..."}` | Returns `{}` | No plan — trades blind |
| entry_scan | `[{symbol, signal_strength}]` | Returns `[]` | Entries missed entirely |
| position_review | `[{symbol, thesis_intact}]` | Returns `[]` | Active positions unmonitored |
| fund_manager | `[{symbol, verdict}]` | Returns `[]` | Rejected entries treated as approved |
| risk_sizing | `[{symbol, action}]` | Returns `[]` | Risk assessment silently skipped |

### The Fix

| Step | Action |
|------|--------|
| 1 | Add Pydantic models per agent output (partially exists in `tools/models.py`) |
| 2 | On parse failure, retry once with "Please respond with valid JSON matching this schema: ..." |
| 3 | Log all fallback events as warnings |
| 4 | Add `agent_dead_letters` table: `(agent_name, cycle_id, raw_output, parse_error, timestamp)` |
| 5 | Monitor DLQ frequency per agent — high rate = prompt quality issue |

### Acceptance Criteria

- [ ] Every agent has a Pydantic output model
- [ ] Parse failures trigger one retry with schema hint
- [ ] All fallback events logged and queryable
- [ ] Dead letter queue populated for post-mortem analysis

---

## 1.5 PostgreSQL Security Hardening

**Finding ID:** QS-I1, QS-I2
**Severity:** CRITICAL
**Effort:** 1 day

### The Problem

```yaml
# docker-compose.yml
ports:
  - "5434:5432"    # Exposed to localhost (and possibly network)
environment:
  POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-quantstack}  # Default: "quantstack"
```

If `.env` is not set, the database uses password "quantstack" and is accessible on port 5434. All system state is exposed. Additionally, all containers run as root — if any container is compromised, the attacker has root privileges.

### The Fix

| Step | Action | Location |
|------|--------|----------|
| 1 | Bind PostgreSQL to localhost only: `127.0.0.1:5434:5432` | `docker-compose.yml` |
| 2 | Remove default passwords — require `.env` to exist with non-default values | `docker-compose.yml` |
| 3 | Add `.env` validation to `start.sh` — fail if passwords are defaults | `start.sh` |
| 4 | Add `USER` directive to Dockerfile — run as non-root | `Dockerfile` |
| 5 | Add `RUN useradd -r quantstack && chown -R quantstack:quantstack /app` | `Dockerfile` |

### Acceptance Criteria

- [ ] PostgreSQL port bound to 127.0.0.1 only
- [ ] Default password "quantstack" rejected at startup
- [ ] All containers run as non-root user
- [ ] `.env` validation prevents startup with insecure defaults

---

## 1.6 Durable Checkpoints (Crash Recovery)

**Finding ID:** CTO GC1
**Severity:** CRITICAL
**Effort:** 1 day

### The Problem

All three graphs use LangGraph's `MemorySaver` for mid-cycle checkpointing. This is in-process memory only. If a container crashes mid-cycle, all intermediate state is lost. A crash during `execute_entries` (after risk approval, before order submission) could leave the system in an inconsistent state — approved trades never executed, no record of the approval.

### The Fix

Switch to `PostgresSaver` or `SqliteSaver` for durable checkpointing. LangGraph supports this natively. Enables resumption from the exact node where the crash occurred.

### Acceptance Criteria

- [ ] All three graph runners use `PostgresSaver` (or equivalent durable checkpointer)
- [ ] Container crash during any node results in clean resume from last checkpoint
- [ ] Verified with integration test: kill container mid-cycle, restart, verify state consistent

---

## 1.7 Trading Graph Polls EventBus + Kill Switch Publishes

**Finding IDs:** CTO AC1, CTO AC2
**Severity:** CRITICAL
**Effort:** 1 day (5-10 lines each)

### The Problem

The Supervisor publishes events (`IC_DECAY`, `DEGRADATION_DETECTED`, `REGIME_CHANGE`) to the EventBus. But the Trading Graph never polls them. The supervisor is shouting into the void. Trading continues on decayed strategies for up to 5 minutes after supervisor detects the problem.

Additionally, when the kill switch fires, it sets a DB flag and writes a sentinel file — but never publishes a `KILL_SWITCH_TRIGGERED` event. The supervisor can't detect kill switch activation via its normal polling loop.

### The Fix

| Step | Action | Location | Lines |
|------|--------|----------|-------|
| 1 | Add `bus.poll()` at `safety_check` node for `IC_DECAY`, `RISK_EMERGENCY` | `trading/nodes.py` | ~5-10 |
| 2 | Add `KILL_SWITCH_TRIGGERED` event in `kill_switch.trigger()` | `execution/kill_switch.py` | ~3 |
| 3 | All graph loops poll `KILL_SWITCH_TRIGGERED` at cycle start | All runners | ~5 |

### Acceptance Criteria

- [ ] Trading graph receives and acts on supervisor events within one cycle
- [ ] Kill switch trigger publishes event visible to all graphs
- [ ] IC_DECAY event halts trading of the affected strategy

---

## 1.8 Containerize the Scheduler

**Finding ID:** CTO OC2
**Severity:** CRITICAL
**Effort:** 1 day

### The Problem

The scheduler runs 13 critical jobs (data refresh, strategy pipeline, community intel, etc.) as a bare process in a tmux session. If it crashes, all jobs stop — no data refresh, no strategy promotion, no EOD sync. No supervisor (systemd, Docker) to restart it.

### The Fix

Add `scheduler` service to `docker-compose.yml` with health check and restart policy (`unless-stopped`). Same Dockerfile as graph services. Health check: verify APScheduler is running and jobs are registered.

### Acceptance Criteria

- [ ] Scheduler runs as Docker container with `unless-stopped` restart
- [ ] Health check verifies APScheduler process is alive and jobs registered
- [ ] Crash of scheduler container → automatic restart within 60s

---

## 1.9 Deterministic Tool Ordering (1-Line Fix, 30-50% Cost Savings)

**Finding ID:** CTO MC1
**Severity:** CRITICAL
**Effort:** 30 minutes

### The Problem

When agents bind tools from `TOOL_REGISTRY`, tool definitions are injected into the system prompt. If ordering varies between invocations (dict ordering, import order), the system prompt hash changes → prompt cache breaks → full-price input tokens on every call.

With 21 agents making calls every 5-10 minutes, this is costing 30-50% more on API spend than necessary.

### The Fix

Sort tool definitions alphabetically by name before injection in `tool_binding.py`. One-line change:

```python
tools = sorted(tools, key=lambda t: t.name)
```

### Acceptance Criteria

- [ ] Tool definitions always injected in deterministic alphabetical order
- [ ] Verified: identical prompts produce identical cache keys across cycles

---

## 1.10 Enable Prompt Caching

**Finding ID:** CTO MC0c
**Severity:** CRITICAL
**Effort:** 1 hour

### The Problem

Claude's prompt caching (90% cost reduction on cached input tokens) is not enabled. No `cache_control`, `CacheControl`, or `ephemeral` references found in the entire codebase. Every call pays full input token price.

At Sonnet $3/MTok input, with ~20K tokens of system prompt per agent, 21 agents, ~100 cycles/day: **~$126/day in system prompt tokens. With caching: ~$12.60/day.** We're paying 10x what we should.

### The Fix

Add `cache_control` breakpoints to system message construction:

```python
SystemMessage(content=base, additional_kwargs={"cache_control": {"type": "ephemeral"}})
```

For Bedrock: use `anthropic_beta: ["prompt-caching-2024-07-31"]` header.

### Acceptance Criteria

- [ ] Prompt caching enabled for all Anthropic API and Bedrock calls
- [ ] System prompt tokens show cache hits in Langfuse traces after first call per cycle
- [ ] Cost reduction of 50%+ on system prompt tokens verified

---

## Summary: Week 1-2 Delivery Checklist

| # | Item | Effort | Impact |
|---|------|--------|--------|
| 1.1 | Mandatory stop-loss | 2-3 days | Prevents unprotected positions |
| 1.2 | Automated DB backups | 1 day | Prevents total data loss |
| 1.3 | Prompt injection defense | 2-3 days | Prevents adversarial manipulation |
| 1.4 | Output schema validation | 2 days | Prevents silent failures |
| 1.5 | PostgreSQL security | 1 day | Prevents unauthorized access |
| 1.6 | Durable checkpoints | 1 day | Prevents inconsistent state on crash |
| 1.7 | EventBus wiring | 1 day | Closes supervisor → trading feedback |
| 1.8 | Containerize scheduler | 1 day | Prevents silent job failures |
| 1.9 | Deterministic tool ordering | 30 min | 30-50% prompt cost savings |
| 1.10 | Enable prompt caching | 1 hour | 10x reduction in system prompt cost |

**Total estimated effort: 10-12 engineering days.**
**Total estimated annual savings from 1.9 + 1.10 alone: $7,000-$40,000.**
