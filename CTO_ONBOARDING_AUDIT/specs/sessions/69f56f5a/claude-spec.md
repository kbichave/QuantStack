# Phase 3: Operational Resilience — Synthesized Specification

**Timeline:** Week 3-5 | **Effort:** 14-15 days (parallelizable to ~8 days with 2 engineers)
**Gate:** CI/CD active. Logs aggregated. Kill switch auto-recovers (broker failures only). Migrations versioned.

---

## Context

QuantStack is an autonomous trading system running as 3 LangGraph StateGraphs in Docker Compose (trading, research, supervisor) plus supporting services (PostgreSQL+pgvector, Langfuse, Ollama, dashboard, finrl-worker). Phase 3 transforms it from market-hours-only attended operation to 24/7 unattended resilience.

**Existing infrastructure strengths:**
- Kill switch with 4 auto-triggers (broker failures, SPY halt, drawdown, model drift)
- File-based heartbeat health checks with Docker health check integration
- Prometheus metrics exposed at `/metrics` (trades, risk rejections, latency, NAV, PnL, kill switch state)
- Discord webhook alerting + daily digest (portfolio, strategy lifecycle, loop health, risk)
- PostgreSQL advisory-lock-protected idempotent migrations (37+ tables)
- Atomic daily API quota tracking in `system_state` table
- Structured JSONL logging with Loguru (50MB rotation, 30-day retention)
- Graceful shutdown with SIGTERM handling and cleanup callbacks
- Supervisor graph with health check / diagnose / recover pipeline

**Critical gaps being addressed:**
- CI/CD completely disabled (`.yml.disabled` files)
- No centralized log shipping (logs lost on host failure)
- Kill switch requires manual reset for all conditions (no auto-recovery)
- No schema_migrations table (can't track which migrations ran)
- Per-process rate limiter (3 containers × 75 = 225 req/min to AV, limit is 75)
- Health checks only verify heartbeat freshness (no cycle success rate, error counts)
- No OOM monitoring or alerting
- No container-level metrics collection

---

## 3.1 Re-enable CI/CD Pipeline

**Approach:** Re-enable existing `ci.yml.disabled` and `release.yml.disabled` as-is, then fix whatever breaks. Do not modernize in this phase (no ghcr.io, no pip-audit, no smoke test — those come later).

**Existing pipeline structure (to re-enable):**
- Ruff lint/format check
- MyPy type checking on trading execution path
- pytest with coverage (target 60%+), matrix: ubuntu/macos, Python 3.11/3.12
- Bandit security scan (HIGH severity, skips RL module)
- Trivy Docker image scan (main branch only)
- All-checks gate blocking merges

**Key files:** `.github/workflows/ci.yml.disabled`, `.github/workflows/release.yml.disabled`, `pyproject.toml` (pytest config)

**Acceptance criteria:**
- [ ] `ci.yml` re-enabled, all checks passing on main
- [ ] Failed CI blocks merge (branch protection rule)
- [ ] Known flaky tests triaged: fix blockers, mark flaky with `@pytest.mark.skip(reason="...")`

---

## 3.2 Log Aggregation + Alerting

**Stack:** Fluent Bit (collector, ~2MB) → Grafana Loki (backend, ~200MB) → Grafana (UI/alerting)

**Architecture:**
- Fluent Bit sidecar in docker-compose reads container logs from `/var/lib/docker/containers/`
- Ships to Loki (single-binary mode, label-based indexing — no full-text index, low storage)
- Grafana provides log search UI + LogQL alerting rules
- Grafana contact points: Discord webhook (immediate), email via Google mail bot (4h+ escalation — TODO for user to set up credentials)

**Alert rules (LogQL in Grafana):**
- `rate({job=~".*-graph"} |= "CRITICAL" [5m]) > 0` → immediate Discord
- `rate({job=~".*-graph"} |= "ERROR" [5m]) > 5` → Discord warning
- Kill switch triggered → immediate Discord (separate alert on `quantstack_kill_switch_active` Prometheus metric)

**Docker Compose additions:**
- `fluent-bit` service: image `fluent/fluent-bit:3.2`, read-only mount of Docker container logs
- `loki` service: image `grafana/loki:3.7.0`, persistent volume for log data
- `grafana` service: image `grafana/grafana:11.4.0`, pre-configured Loki + Prometheus datasources

**Retention:** Loki `retention_period: 720h` (30 days). Trade audit logs (fills, position changes) tagged for longer retention.

**Key files:** `docker-compose.yml`, new `config/fluent-bit.conf`, new `config/loki-config.yaml`, new `config/grafana/provisioning/`

**Acceptance criteria:**
- [ ] All container logs searchable in Grafana via Loki
- [ ] CRITICAL events trigger immediate Discord notification
- [ ] Error rate anomaly detection active via LogQL
- [ ] Logs survive host container restart (Loki persistent volume)

---

## 3.3 Kill Switch Auto-Recovery

**Principle:** Auto-reset ONLY when the condition is purely external and infrastructure-level with no information content about model or market state.

**Auto-reset safe: Broker failures only**
- Condition: `KILL_MAX_BROKER_FAILURES` consecutive API failures
- Recovery flow: 0min → Discord alert; 5min → auto-investigate (is broker API responding now?); 15min → if broker responsive, auto-reset with 50% sizing for first cycle; subsequent cycles ramp back to 100% sizing over 3 successful cycles
- Logging: every auto-reset logged with reason, broker status check result, sizing scalar

**NEVER auto-reset (require manual reset with reason logged):**
- **Drawdown (3-day rolling)** — systematic signal, not transient. Requires human review of what drove it.
- **Model drift (>50%)** — broken model at half size is a slower bleed. Needs investigation.
- **SPY halt / circuit breaker** — post-halt open is often the most volatile 15 minutes. Manual decision on whether post-halt regime is tradeable.

**Escalation tiers (all kill switch types):**
- 0min → Discord alert with trigger reason + current positions
- 4h (still triggered) → Email escalation (Google mail bot — TODO for user to configure)
- 24h (still triggered) → Emergency Discord alert with full position summary + P&L

**Depends on:** Phase 1 items 1.7+1.8 (EventBus wiring)

**Key files:** `src/quantstack/execution/kill_switch.py`, supervisor graph nodes

**Acceptance criteria:**
- [ ] Kill switch trigger sends immediate Discord alert
- [ ] Broker failures auto-recover after investigation with 50% sizing
- [ ] Drawdown, drift, SPY halt require manual `reset(reset_by=..., reason=...)`
- [ ] Escalation at 4h (email) and 24h (emergency)
- [ ] Auto-reset ramps sizing back to 100% over 3 successful cycles

---

## 3.4 Migration Versioning — Full Alembic

**Approach:** Migrate from 37+ `CREATE TABLE IF NOT EXISTS` calls to full Alembic migration framework.

**Implementation:**
1. Install Alembic, configure `alembic.ini` pointing to QuantStack's PostgreSQL
2. Generate initial migration from current schema (baseline) — represents all existing tables
3. Mark baseline as applied on existing databases
4. Convert future schema changes to Alembic migrations (up/down)
5. Startup calls `alembic upgrade head` instead of `run_migrations()`
6. Advisory lock (`_MIGRATION_ADVISORY_LOCK = 42`) retained for multi-container safety

**Migration from current system:**
- Existing `_migrate_*_pg()` functions become the baseline migration
- Schema introspection or `CREATE TABLE IF NOT EXISTS` dump generates baseline
- Existing databases stamped as "at baseline" without re-running DDL

**Key files:** `src/quantstack/db.py`, new `alembic/`, new `alembic.ini`

**Acceptance criteria:**
- [ ] Alembic configured with PostgreSQL connection
- [ ] Baseline migration represents all 37+ existing tables
- [ ] `alembic upgrade head` is idempotent on existing databases
- [ ] Future schema changes go through `alembic revision --autogenerate`
- [ ] Multi-container safety via advisory lock

---

## 3.5 Langfuse Retention (Deferred — Config-Only)

**Approach:** Add retention cleanup as a disabled-by-default feature. Do NOT implement actual cleanup now.

**Implementation:**
- Add `LANGFUSE_RETENTION_ENABLED=false` and `LANGFUSE_RETENTION_DAYS=30` env vars
- Add scheduler job stub that checks the flag and exits if disabled
- Log message: "Langfuse retention cleanup is disabled. Set LANGFUSE_RETENTION_ENABLED=true to enable."
- When enabled (future): cleanup via Langfuse API or direct SQL against langfuse-db

**Key files:** `scripts/scheduler.py`, env var validation

**Acceptance criteria:**
- [ ] Config flag exists, defaults to OFF
- [ ] Scheduler job registered but skips execution when disabled
- [ ] Documentation notes the feature exists and how to enable

---

## 3.6 OOM Monitoring

**Stack:** cAdvisor (container metrics) + Prometheus (storage) + Grafana (alerting)

**Docker Compose additions:**
- `cadvisor` service: image `gcr.io/cadvisor/cadvisor:v0.49.1`, mounts `/var/run/docker.sock`, `/sys`, `/var/lib/docker`
- `prometheus` service: image `prom/prometheus:v2.53.0`, scrapes cAdvisor + existing QuantStack `/metrics` endpoints
- Grafana (from 3.2) gets Prometheus datasource alongside Loki

**Key metrics:**
- `container_memory_working_set_bytes` / `container_spec_memory_limit_bytes` — use working_set (what OOM killer uses), not usage_bytes
- `container_oom_events_total` — counter increments on OOMKilled

**Alert rules:**
- `container_memory_working_set_bytes / container_spec_memory_limit_bytes > 0.80` for 2min → Warning
- `container_memory_working_set_bytes / container_spec_memory_limit_bytes > 0.90` for 1min → Critical
- `increase(container_oom_events_total[5m]) > 0` → Critical (OOMKilled)

**Consideration:** research-graph at 1GB may need increase — monitor before changing.

**Key files:** `docker-compose.yml`, new `config/prometheus.yml`

**Acceptance criteria:**
- [ ] Memory usage per container visible in Grafana
- [ ] Alert at 80% threshold (warning) and 90% (critical)
- [ ] OOMKilled events alerted immediately
- [ ] All QuantStack Prometheus metrics also scraped by Prometheus

---

## 3.7 Health Check Granularity

**Approach:** Extend supervisor graph health check nodes beyond heartbeat freshness.

**New metrics to track:**
- Cycle success rate (last 10 cycles) — alert if < 70%
- Error count per cycle — alert if > 3
- Strategy generation rate — alert if 0 new strategies in 7 days
- IC trend — alert if IC < 0 for 5 consecutive days, disable collector
- Research queue depth — alert if > 50 pending items

**Data sources:** `graph_checkpoints` table (has cycle status, error_message), `strategies` table, `signal_state` table

**Implementation:**
- Add `_check_cycle_metrics()` to supervisor health check node
- Query `graph_checkpoints` for last 10 cycles per graph
- Compute success rate, error count
- Write results to `system_state` table for dashboard consumption
- Fire Discord alerts when thresholds breached

**Key files:** `src/quantstack/graphs/supervisor/nodes.py`, `src/quantstack/health/heartbeat.py`

**Acceptance criteria:**
- [ ] Supervisor reports cycle success rate, error count, IC trend
- [ ] Alerts fire when metrics breach thresholds
- [ ] Health status queryable from system_state table

---

## 3.8 Shared Rate Limiter

**Approach:** PL/pgSQL token bucket function in PostgreSQL, replacing per-process in-memory counter.

**Implementation:**
1. Create `rate_limit_buckets` table: `(bucket_key TEXT PK, tokens NUMERIC, max_tokens NUMERIC, refill_rate NUMERIC, last_refill TIMESTAMPTZ)`
2. Create `consume_token(bucket_key, cost)` PL/pgSQL function:
   - `SELECT ... FOR UPDATE` row lock
   - Calculate refilled tokens based on elapsed time
   - Consume or reject atomically
   - Return BOOLEAN
3. Initialize bucket: `alpha_vantage`, 75 tokens, refill 1.25/sec (75/min)
4. Replace per-process `_wait_for_rate_limit()` with call to `consume_token()`
5. Retry with backoff if token not available
6. Daily quota tracking stays as-is (already atomic via `INSERT ... ON CONFLICT`)

**Performance:** 75/min is trivial for PostgreSQL. `FOR UPDATE` row locks handle this without measurable overhead.

**Key files:** `src/quantstack/data/fetcher.py`, Alembic migration for new table + function

**Acceptance criteria:**
- [ ] Rate limiting shared across all containers
- [ ] Total AV calls never exceed 75/min regardless of container count
- [ ] Daily quota tracking unchanged (already correct)

---

## 3.9 Secrets Management

**Approach:** Harden `.env` only. No Docker secrets or vault.

**Implementation:**
1. `start.sh`: check `.env` permissions, warn if not 600, offer to fix
2. Add `.env.example` with all vars documented (types, ranges, defaults)
3. Document credential rotation procedure in `docs/ops-runbook.md`

**Key files:** `start.sh`, `.env.example`, `docs/ops-runbook.md`

**Acceptance criteria:**
- [ ] `start.sh` validates `.env` permissions
- [ ] `.env.example` documents all vars
- [ ] Rotation procedure documented

---

## 3.10 SBOM Scanning

**Depends on:** 3.1 (CI/CD pipeline re-enabled)

**Approach:** Deferred to post-3.1 iteration. The existing ci.yml already includes Trivy for Docker image scanning. Add `pip-audit` and `cyclonedx-py` as a follow-up once CI is green.

**Acceptance criteria:**
- [ ] `pip audit` runs in CI on every push (after CI stabilized)
- [ ] Weekly vulnerability scan with alerting

---

## 3.11 Env Var Type Validation

**Approach:** Hard crash on invalid values at startup. No warnings, no defaults for critical vars.

**Implementation:**
1. Create `src/quantstack/config/validation.py` with typed validators
2. Call at service startup before any graph initialization
3. Validate:
   - Required vars exist: `TRADER_PG_URL`, `ALPHA_VANTAGE_API_KEY`, `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`
   - Numeric vars parse correctly: `RISK_MAX_POSITION_PCT`, `AV_DAILY_CALL_LIMIT`, `FORWARD_TESTING_SIZE_SCALAR`
   - Boolean vars are `true`/`false`: `ALPACA_PAPER`, `USE_REAL_TRADING`, `USE_FORWARD_TESTING_FOR_ENTRIES`
   - URL vars are valid URLs: `TRADER_PG_URL`, `LANGFUSE_HOST`
4. On failure: log error with var name, expected type, actual value, then `sys.exit(1)`

**Key files:** New `src/quantstack/config/validation.py`, runner entry points

**Acceptance criteria:**
- [ ] All critical env vars validated at startup
- [ ] Invalid values produce clear error messages and crash
- [ ] Service refuses to start with misconfigured environment

---

## Dependencies

- **3.3 depends on Phase 1 items 1.7+1.8** (EventBus wiring)
- **3.10 depends on 3.1** (CI pipeline)
- **3.8 migration depends on 3.4** (Alembic for rate_limit_buckets table)
- **3.6 shares Grafana with 3.2** (single Grafana instance)

## Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Log stack | Fluent Bit + Loki + Grafana | Free, ~200MB, label-based queries, pairs with Prometheus |
| Kill switch auto-reset | Broker failures only | Only infrastructure-level conditions carry no market/model signal |
| Rate limiter | PL/pgSQL token bucket | Single atomic call, no Redis needed, 75/min is trivial for PG |
| Migration framework | Full Alembic | Proper up/down, auto-generate, version chain |
| CI/CD scope | Re-enable as-is, fix breakage | Get green first, modernize later |
| Env validation | Hard crash | Better to not start than trade with bad config |
| Langfuse cleanup | Config flag, default OFF | Defer actual cleanup, just wire the option |
| Secrets | Harden .env only | Sufficient for single-host |
| Monitoring | cAdvisor + Prometheus + Grafana in docker-compose | Fully self-contained, unified dashboard |
| Email escalation | Google mail bot (TODO for user) | No AWS; user will set up Gmail credentials separately |
