# 04 — Operational Resilience: Survive 24/7 Unattended

**Priority:** P2
**Timeline:** Week 3-5
**Gate:** CI/CD enabled, log aggregation active, kill switch auto-recovery, migration versioning.

---

## Why This Section Matters

The system currently runs as a market-hours-only, attended operation. A human starts it (`./start.sh`), monitors tmux sessions, and manually resets the kill switch after incidents. For 24/7 autonomous operation, every failure mode must have an automated recovery path, every critical event must alert, and every deploy must be tested before it touches production.

---

## 4.1 Enable CI/CD Pipeline

**Finding ID:** CTO OC3
**Severity:** CRITICAL
**Effort:** 2 days

### The Problem

`.github/workflows/ci.yml.disabled` and `release.yml.disabled` — CI/CD is completely turned off. No automated tests on push. No image registry. No staging environment. No rollback procedure. For a system trading real capital, deploying untested code is an existential risk.

### The Fix

| Step | Action |
|------|--------|
| 1 | Re-enable `ci.yml` — run test suite on every push to main |
| 2 | Add type checking (mypy/pyright) to CI |
| 3 | Add `pip audit` for dependency vulnerability scanning |
| 4 | Build Docker image in CI and push to registry |
| 5 | Add smoke test: start containers, verify health checks, stop |

### Acceptance Criteria

- [ ] Every push to main runs test suite + type check + build
- [ ] Failed CI blocks merge
- [ ] Docker images versioned and stored in registry
- [ ] Rollback = deploy previous image version

---

## 4.2 Log Aggregation and Alerting

**Finding ID:** CTO OH2
**Severity:** HIGH
**Effort:** 3 days

### The Problem

Logs go to Docker json-file driver (50MB rotation, 5 files max) and local JSONL files. No shipping to any centralized system. No alerting on ERROR/CRITICAL volume spikes. If the host goes down, all logs are lost.

### The Fix

| Step | Action |
|------|--------|
| 1 | Add Filebeat/Fluentd sidecar to Docker Compose |
| 2 | Ship to Elasticsearch or CloudWatch (whichever is cheaper at scale) |
| 3 | Create alerts: error rate > 5/hour, CRITICAL any occurrence, kill switch triggered |
| 4 | Real-time Discord webhook for CRITICAL events (extend `daily_digest.py`) |

### Acceptance Criteria

- [ ] All container logs shipped to centralized system
- [ ] CRITICAL events trigger immediate Discord/Slack notification
- [ ] Error rate anomaly detection active
- [ ] Logs survive host failure

---

## 4.3 Kill Switch Auto-Recovery

**Finding ID:** CTO OH3
**Severity:** HIGH
**Effort:** 2 days

### The Problem

Once triggered, the kill switch requires manual `reset()` call. No auto-recovery for transient conditions. No escalation. Could sit triggered for days if owner is AFK.

### The Fix

Tiered recovery:

| Time | Action |
|------|--------|
| 0 min | Discord alert: kill switch triggered, reason, positions affected |
| 5 min | Auto-investigate: is the trigger condition still true? (broker down? SPY halted? drawdown?) |
| 15 min | If transient condition resolved (broker back, SPY resumed): auto-reset with reduced sizing (50%) |
| 4 hours | If still triggered: escalate to email/SMS |
| 24 hours | If still triggered: emergency alert with full position summary |

### Acceptance Criteria

- [ ] Kill switch trigger sends immediate alert
- [ ] Transient conditions (broker timeout, brief SPY halt) auto-recover after investigation
- [ ] Persistent conditions escalate through notification tiers
- [ ] Auto-reset uses reduced sizing for first cycle after recovery

---

## 4.4 Database Migration Versioning

**Finding ID:** QS-I4
**Severity:** HIGH
**Effort:** 1 day

### The Problem

30+ migration functions called sequentially in `db.py:517-553`. No `schema_version` table. Can't answer: "Which migrations have run? Did #17 fail? Is the schema consistent?" If a migration fails mid-way, next startup re-runs all. Idempotent `CREATE TABLE IF NOT EXISTS` helps, but `ALTER TABLE` may fail or duplicate.

### The Fix

| Step | Action |
|------|--------|
| 1 | Add `schema_migrations` table: `(version INT, name TEXT, applied_at TIMESTAMP)` |
| 2 | Check before running each migration: skip if already applied |
| 3 | Log migration outcomes |
| 4 | Consider Alembic for future migrations (but don't over-engineer now) |

### Acceptance Criteria

- [ ] `schema_migrations` table tracks all applied migrations
- [ ] Migrations are idempotent and skip if already applied
- [ ] Migration failure leaves clear error state

---

## 4.5 Database Transaction Isolation for Positions

**Finding ID:** QS-I3
**Severity:** CRITICAL
**Effort:** 1 day

### The Problem

When two agents simultaneously read and update a position (execution monitor tightening stop while trading graph sizing new entry on same symbol), default `READ COMMITTED` isolation allows both to read stale state. One update overwrites the other.

### The Fix

Use `SELECT FOR UPDATE` on position rows during modification. Alternatively, set isolation to `SERIALIZABLE` for the position update connection pool.

### Acceptance Criteria

- [ ] Position updates use row-level locking (`SELECT FOR UPDATE`)
- [ ] Concurrent position modifications serialized correctly
- [ ] No lost updates verified with integration test

---

## 4.6 Langfuse Retention Cleanup

**Finding ID:** CTO OH1
**Severity:** HIGH
**Effort:** 0.5 day

### The Problem

Langfuse keeps all traces indefinitely. The `langfuse-db` container has only 256MB memory. With 3 graphs producing traces every 5-10 minutes, the DB will fill disk within weeks.

### The Fix

Add retention cleanup job to scheduler: delete traces older than 30 days. Run weekly.

### Acceptance Criteria

- [ ] Weekly cleanup deletes Langfuse traces > 30 days old
- [ ] Disk usage monitored and alerted at 80%

---

## 4.7 OOM and Resource Monitoring

**Finding ID:** CTO OH5
**Severity:** HIGH
**Effort:** 1 day

### The Problem

Docker memory limits are hard caps. If a container exceeds its limit, Docker kills it silently (OOMKilled). No alerting, no logging.

### The Fix

| Step | Action |
|------|--------|
| 1 | Add container stats monitoring (Docker stats or cAdvisor) |
| 2 | Alert on memory usage > 80% of limit |
| 3 | Alert on any OOMKilled event |
| 4 | Consider increasing research-graph from 1GB (tight for ML training) |

### Acceptance Criteria

- [ ] Memory usage per container monitored
- [ ] Alert at 80% threshold
- [ ] OOMKilled events logged and alerted

---

## 4.8 Health Check Granularity

**Finding ID:** CTO OH4
**Severity:** HIGH
**Effort:** 2 days

### The Problem

Health checks only verify heartbeat file presence and age. They don't validate cycle success rate, strategy generation rate, trading performance trends, or error counts.

### The Fix

Add granular health metrics to supervisor:

| Metric | Threshold | Action |
|--------|-----------|--------|
| Cycle success rate (last 10) | < 70% | Alert |
| Error count per cycle | > 3 | Alert |
| Strategy pipeline throughput | 0 new strategies in 7 days | Alert |
| IC trend (any collector) | IC < 0 for 5 consecutive days | Disable collector |
| Research queue depth | > 50 pending tasks | Alert (backlog building) |

### Acceptance Criteria

- [ ] Supervisor health check reports cycle success rate, error count, IC trend
- [ ] Alerts fire when metrics breach thresholds
- [ ] Health dashboard (or query) available for human review

---

## 4.9 Secrets Management

**Finding ID:** CTO OC (MEDIUM)
**Severity:** MEDIUM
**Effort:** 1 day

### The Problem

API keys, DB credentials, and broker secrets stored as plain text in `.env` on disk. No encryption, no rotation, no access auditing.

### The Fix (pragmatic, not over-engineered)

| Step | Action |
|------|--------|
| 1 | `.env` permissions set to 600 (owner read/write only) |
| 2 | Add `.env` to `.gitignore` (verify) |
| 3 | Add env var type validation at startup (reject `RISK_MAX_POSITION_PCT=ten`) |
| 4 | Document credential rotation procedure |
| 5 | Consider AWS Secrets Manager for production deployment |

### Acceptance Criteria

- [ ] `.env` file permissions restricted
- [ ] Startup validates env var types and rejects invalid values
- [ ] Credential rotation procedure documented

---

## 4.10 Rate Limiter: Shared Across Processes

**Finding ID:** DO-9
**Severity:** HIGH
**Effort:** 1 day

### The Problem

`data/fetcher.py` has a per-process rate limiter. Three Docker containers × 75 req/min each = 225 req/min to Alpha Vantage (limit is 75/min). The daily quota guard in `system_state` table helps but has race conditions across processes.

### The Fix

Shared rate limiter via PostgreSQL advisory lock or a token bucket backed by the `system_state` table with atomic increment. Replace `time.sleep()` with `asyncio.sleep()` to avoid blocking.

### Acceptance Criteria

- [ ] Rate limiting shared across all containers
- [ ] Total API calls to AV never exceed 75/min regardless of container count
- [ ] No blocking sleep on the event loop

---

## 4.11 SBOM and Dependency Scanning

**Finding ID:** QS-I8
**Severity:** MEDIUM
**Effort:** 0.5 day

### The Problem

No Software Bill of Materials. No vulnerability scanning. A CVE in numpy, pandas, or httpx could be exploited without awareness.

### The Fix

Add `pip audit` to CI pipeline. Generate SBOM with `cyclonedx-py`. Run weekly vulnerability scan.

### Acceptance Criteria

- [ ] `pip audit` runs in CI on every push
- [ ] SBOM generated and stored
- [ ] Weekly vulnerability scan with alerting

---

## 4.12 Multi-Mode Operation (24/7)

**Finding ID:** CTO 24/7 Readiness
**Severity:** MEDIUM
**Effort:** 3-5 days

### The Problem

The system has no concept of extended hours or overnight modes. Trading and research graphs simply don't run outside market hours.

### Required Modes

| Mode | Hours (ET) | Activity |
|------|-----------|----------|
| **Market Hours** | 9:30-16:00 Mon-Fri | Full trading + research pipelines |
| **Extended Hours** | 16:00-20:00, 04:00-09:30 | Position monitoring only, no new entries, earnings processing |
| **Overnight/Weekend** | 20:00-04:00, weekends | Heavy research (ML training), data backfill, community intel |

### The Fix

Add time-of-day mode detection to graph runners. Each mode loads different agent configurations and tool bindings.

### Acceptance Criteria

- [ ] Graph runners detect current mode and adjust behavior
- [ ] Extended hours: no new entries, monitoring only
- [ ] Overnight: research graph gets full compute budget

---

## Summary: Operational Resilience Delivery

| # | Item | Effort | Priority |
|---|------|--------|----------|
| 4.1 | CI/CD pipeline | 2 days | HIGH |
| 4.2 | Log aggregation + alerting | 3 days | HIGH |
| 4.3 | Kill switch auto-recovery | 2 days | HIGH |
| 4.4 | Migration versioning | 1 day | HIGH |
| 4.5 | Transaction isolation | 1 day | CRITICAL |
| 4.6 | Langfuse retention | 0.5 day | HIGH |
| 4.7 | OOM monitoring | 1 day | HIGH |
| 4.8 | Health check granularity | 2 days | HIGH |
| 4.9 | Secrets management | 1 day | MEDIUM |
| 4.10 | Shared rate limiter | 1 day | HIGH |
| 4.11 | SBOM scanning | 0.5 day | MEDIUM |
| 4.12 | Multi-mode operation | 3-5 days | MEDIUM |

**Total estimated effort: 18-22 engineering days.**
