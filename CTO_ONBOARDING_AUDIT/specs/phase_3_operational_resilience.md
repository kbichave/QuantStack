# Phase 3: Operational Resilience — Deep Plan Spec

**Timeline:** Week 3-5
**Effort:** 14-15 days (parallelizable to ~8 days with 2 engineers)
**Gate:** CI/CD active. Logs aggregated. Kill switch auto-recovers.

---

## Context

This spec is part of the QuantStack CTO Onboarding Audit implementation plan (164 findings, overall grade C-). Phase 3 transforms the system from a market-hours-only, attended operation to one that can survive 24/7 unattended. Every failure mode must have an automated recovery path, every critical event must alert, and every deploy must be tested.

**Full audit reference:** [`CTO_ONBOARDING_AUDIT/`](../README.md)
**Primary audit section:** [`04_OPERATIONAL_RESILIENCE.md`](../04_OPERATIONAL_RESILIENCE.md)

---

## Objective

Build the operational backbone for 24/7 autonomous operation: CI/CD pipeline, centralized logging with alerting, kill switch auto-recovery, database migration versioning, and infrastructure monitoring.

---

## Items

### 3.1 Enable CI/CD Pipeline

- **Finding:** CTO OC3 | **Severity:** CRITICAL | **Effort:** 2 days
- **Audit section:** [`04_OPERATIONAL_RESILIENCE.md` §4.1](../04_OPERATIONAL_RESILIENCE.md)
- **Problem:** `.github/workflows/ci.yml.disabled` and `release.yml.disabled` — CI/CD completely off. No automated tests on push. No image registry. No staging. No rollback. Deploying untested code to a system trading real capital is existential risk.
- **Fix:**
  1. Re-enable `ci.yml` — run test suite on every push to main
  2. Add type checking (mypy/pyright) to CI
  3. Add `pip audit` for dependency vulnerability scanning
  4. Build Docker image in CI and push to registry
  5. Add smoke test: start containers, verify health checks, stop
- **Key files:** `.github/workflows/ci.yml.disabled`, `.github/workflows/release.yml.disabled`, `Dockerfile`
- **Acceptance criteria:**
  - [ ] Every push to main runs test suite + type check + build
  - [ ] Failed CI blocks merge
  - [ ] Docker images versioned and stored in registry
  - [ ] Rollback = deploy previous image version

### 3.2 Log Aggregation + Alerting

- **Finding:** CTO OH2 | **Severity:** HIGH | **Effort:** 3 days
- **Audit section:** [`04_OPERATIONAL_RESILIENCE.md` §4.2](../04_OPERATIONAL_RESILIENCE.md)
- **Problem:** Logs go to Docker json-file driver (50MB rotation, 5 files max) and local JSONL files. No centralized shipping. No alerting on ERROR/CRITICAL volume spikes. Host goes down = all logs lost.
- **Fix:**
  1. Add Filebeat/Fluentd sidecar to Docker Compose
  2. Ship to Elasticsearch or CloudWatch
  3. Create alerts: error rate > 5/hour, CRITICAL any occurrence, kill switch triggered
  4. Real-time Discord webhook for CRITICAL events (extend `daily_digest.py`)
- **Key files:** `docker-compose.yml`, logging configuration, `scripts/daily_digest.py`
- **Acceptance criteria:**
  - [ ] All container logs shipped to centralized system
  - [ ] CRITICAL events trigger immediate Discord/Slack notification
  - [ ] Error rate anomaly detection active
  - [ ] Logs survive host failure

### 3.3 Kill Switch Auto-Recovery

- **Finding:** CTO OH3 | **Severity:** HIGH | **Effort:** 2 days
- **Depends on:** Phase 1 items 1.7+1.8 (EventBus wiring)
- **Audit section:** [`04_OPERATIONAL_RESILIENCE.md` §4.3](../04_OPERATIONAL_RESILIENCE.md)
- **Problem:** Once triggered, kill switch requires manual `reset()`. No auto-recovery for transient conditions. No escalation. Could sit triggered for days if owner is AFK.
- **Fix:** Tiered recovery: 0min → Discord alert; 5min → auto-investigate (broker down? SPY halted? drawdown?); 15min → if transient resolved, auto-reset with 50% sizing; 4h → email/SMS escalation; 24h → emergency alert with full position summary.
- **Key files:** `src/quantstack/execution/kill_switch.py`, supervisor graph
- **Acceptance criteria:**
  - [ ] Kill switch trigger sends immediate alert
  - [ ] Transient conditions auto-recover after investigation
  - [ ] Persistent conditions escalate through notification tiers
  - [ ] Auto-reset uses reduced sizing for first cycle after recovery

### 3.4 Migration Versioning

- **Finding:** QS-I4 | **Severity:** HIGH | **Effort:** 1 day
- **Audit section:** [`04_OPERATIONAL_RESILIENCE.md` §4.4](../04_OPERATIONAL_RESILIENCE.md)
- **Problem:** 30+ migration functions called sequentially in `db.py:517-553`. No `schema_version` table. Can't track which migrations ran, failed, or need re-running. If migration fails mid-way, next startup re-runs all.
- **Fix:**
  1. Add `schema_migrations` table: `(version INT, name TEXT, applied_at TIMESTAMP)`
  2. Check before running each migration: skip if already applied
  3. Log migration outcomes
- **Key files:** `src/quantstack/db.py`
- **Acceptance criteria:**
  - [ ] `schema_migrations` table tracks all applied migrations
  - [ ] Migrations are idempotent and skip if already applied
  - [ ] Migration failure leaves clear error state

### 3.5 Langfuse Retention Cleanup

- **Finding:** CTO OH1 | **Severity:** HIGH | **Effort:** 0.5 day
- **Audit section:** [`04_OPERATIONAL_RESILIENCE.md` §4.6](../04_OPERATIONAL_RESILIENCE.md)
- **Problem:** Langfuse keeps all traces indefinitely. `langfuse-db` has 256MB memory. 3 graphs producing traces every 5-10 min will fill disk within weeks.
- **Fix:** Add retention cleanup job to scheduler: delete traces > 30 days. Run weekly.
- **Key files:** Scheduler jobs, Langfuse DB access
- **Acceptance criteria:**
  - [ ] Weekly cleanup deletes Langfuse traces > 30 days old
  - [ ] Disk usage monitored and alerted at 80%

### 3.6 OOM Monitoring

- **Finding:** CTO OH5 | **Severity:** HIGH | **Effort:** 1 day
- **Audit section:** [`04_OPERATIONAL_RESILIENCE.md` §4.7](../04_OPERATIONAL_RESILIENCE.md)
- **Problem:** Docker memory limits are hard caps. Container exceeds limit → Docker kills silently (OOMKilled). No alerting.
- **Fix:**
  1. Add container stats monitoring (Docker stats or cAdvisor)
  2. Alert on memory usage > 80% of limit
  3. Alert on any OOMKilled event
  4. Consider increasing research-graph from 1GB
- **Key files:** `docker-compose.yml`, monitoring configuration
- **Acceptance criteria:**
  - [ ] Memory usage per container monitored
  - [ ] Alert at 80% threshold
  - [ ] OOMKilled events logged and alerted

### 3.7 Health Check Granularity

- **Finding:** CTO OH4 | **Severity:** HIGH | **Effort:** 2 days
- **Audit section:** [`04_OPERATIONAL_RESILIENCE.md` §4.8](../04_OPERATIONAL_RESILIENCE.md)
- **Problem:** Health checks only verify heartbeat file presence/age. Don't validate cycle success rate, strategy generation, trading performance, or error counts.
- **Fix:** Add granular metrics: cycle success rate (last 10) < 70% → alert; error count/cycle > 3 → alert; 0 new strategies in 7 days → alert; IC < 0 for 5 days → disable collector; research queue > 50 → alert.
- **Key files:** Supervisor graph health check nodes
- **Acceptance criteria:**
  - [ ] Supervisor reports cycle success rate, error count, IC trend
  - [ ] Alerts fire when metrics breach thresholds
  - [ ] Health dashboard or query available

### 3.8 Shared Rate Limiter

- **Finding:** DO-9 | **Severity:** HIGH | **Effort:** 1 day
- **Audit section:** [`04_OPERATIONAL_RESILIENCE.md` §4.10](../04_OPERATIONAL_RESILIENCE.md)
- **Problem:** Per-process rate limiter. 3 containers × 75 req/min each = 225 req/min to AV (limit 75/min). Daily quota guard has race conditions across processes.
- **Fix:** Shared rate limiter via PostgreSQL advisory lock or token bucket backed by `system_state` table with atomic increment.
- **Key files:** `src/quantstack/data/fetcher.py`, rate limiting logic
- **Acceptance criteria:**
  - [ ] Rate limiting shared across all containers
  - [ ] Total AV calls never exceed 75/min regardless of container count

### 3.9 Secrets Management

- **Finding:** CTO (MEDIUM) | **Severity:** MEDIUM | **Effort:** 1 day
- **Audit section:** [`04_OPERATIONAL_RESILIENCE.md` §4.9](../04_OPERATIONAL_RESILIENCE.md)
- **Problem:** API keys, DB credentials, broker secrets as plain text in `.env`. No encryption, rotation, or access auditing.
- **Fix:**
  1. `.env` permissions set to 600
  2. Add env var type validation at startup (reject `RISK_MAX_POSITION_PCT=ten`)
  3. Document credential rotation procedure
- **Key files:** `.env`, `start.sh`, startup validation
- **Acceptance criteria:**
  - [ ] `.env` file permissions restricted
  - [ ] Startup validates env var types and rejects invalid values
  - [ ] Credential rotation procedure documented

### 3.10 SBOM Scanning

- **Finding:** QS-I8 | **Severity:** MEDIUM | **Effort:** 0.5 day
- **Depends on:** 3.1 (CI/CD pipeline)
- **Audit section:** [`04_OPERATIONAL_RESILIENCE.md` §4.11](../04_OPERATIONAL_RESILIENCE.md)
- **Problem:** No SBOM. No vulnerability scanning. CVE in numpy/pandas/httpx could be exploited without awareness.
- **Fix:** Add `pip audit` to CI. Generate SBOM with `cyclonedx-py`. Run weekly vulnerability scan.
- **Key files:** CI pipeline configuration
- **Acceptance criteria:**
  - [ ] `pip audit` runs in CI on every push
  - [ ] Weekly vulnerability scan with alerting

### 3.11 Env Var Type Validation

- **Finding:** CTO (MEDIUM) | **Severity:** MEDIUM | **Effort:** 0.5 day
- **Audit section:** [`04_OPERATIONAL_RESILIENCE.md` §4.9](../04_OPERATIONAL_RESILIENCE.md) (combined with 3.9)
- **Problem:** `RISK_MAX_POSITION_PCT=ten` silently fails. No type validation at startup.
- **Fix:** Add startup validation function that checks types, ranges, and required values for all critical env vars.
- **Key files:** Startup initialization
- **Acceptance criteria:**
  - [ ] All critical env vars validated at startup
  - [ ] Invalid values produce clear error messages and prevent startup

---

## Dependencies

- **Depends on:** Phase 1 (safety hardening)
- **3.3 depends on Phase 1 items 1.7+1.8** (EventBus wiring for kill switch events)
- **3.10 depends on 3.1** (needs CI pipeline to run scans)
- **Runs parallel with:** Phase 5 (Cost Optimization) and Phase 8 (Data Pipeline)

---

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| 3.1: Re-enabling CI may expose many existing test failures | Triage: fix blockers, skip known-flaky, track tech debt |
| 3.2: Log aggregation adds infrastructure cost | Start with CloudWatch (pay-per-use), evaluate Elasticsearch if volume grows |
| 3.3: Kill switch auto-recovery could re-enable trading during genuine emergency | Conservative defaults: auto-reset only for broker timeout, never for drawdown |

---

## Validation Plan

1. **CI/CD (3.1):** Push intentionally broken code → verify CI blocks merge.
2. **Logging (3.2):** Search centralized logs for specific error → verify accessible.
3. **Kill switch (3.3):** Trigger kill switch, simulate broker recovery → verify auto-reset with reduced sizing.
4. **Migrations (3.4):** Run `db.py` startup twice → verify no duplicate migration attempts.
5. **Rate limiter (3.8):** Start 3 containers simultaneously → verify AV calls stay under 75/min.
