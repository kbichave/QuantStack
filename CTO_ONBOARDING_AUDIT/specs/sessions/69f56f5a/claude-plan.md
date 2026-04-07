# Implementation Plan: Phase 3 — Operational Resilience

## Overview

QuantStack is an autonomous trading system running 3 LangGraph graphs (trading, research, supervisor) as Docker Compose services, backed by PostgreSQL+pgvector, Langfuse observability, and Ollama for local LLM inference. The system currently requires human attendance during market hours and has no automated recovery, no CI/CD, no centralized logging, and no container-level monitoring.

This plan transforms QuantStack into a 24/7 unattended system by implementing 11 operational resilience items across CI/CD, logging, kill switch recovery, migration versioning, monitoring, rate limiting, and configuration hardening.

**Key architectural decisions (from stakeholder interview):**
- Log stack: Fluent Bit → Loki → Grafana (free, ~200MB, unified with Prometheus)
- Kill switch auto-reset: broker failures ONLY (all other conditions require human clearance — drawdown, drift, and market halt carry signal about model/market state)
- Rate limiter: PL/pgSQL token bucket (single atomic DB call, no Redis)
- Migrations: Full Alembic (proper up/down, hand-written migrations — no SQLAlchemy models for autogenerate)
- CI/CD: Re-enable existing disabled workflows and fix breakage — no modernization yet
- Env validation: Hard crash on invalid (no silent defaults)
- Monitoring: cAdvisor + Prometheus + Grafana fully in docker-compose.yml

---

## Section 1: CI/CD Pipeline Re-enablement

### Context

Two GitHub Actions workflows exist but are disabled: `ci.yml.disabled` (lint, type check, test, security scan, Docker build) and `release.yml.disabled` (tag-based release to PyPI). The test suite uses pytest with 30+ unit tests, coverage target 60%, markers for slow/integration/regression/requires_api/requires_gpu.

### What to Build

**1.1 Rename disabled workflows back to `.yml`**

Remove the `.disabled` suffix from both files. This immediately activates CI on the next push.

**1.2 Triage and fix CI failures**

The existing workflow has 6 stages: ruff lint, mypy type check, pytest, bandit security scan, trivy Docker scan, and an all-checks gate. Expect failures in some stages since CI has been disabled while development continued.

Triage approach:
- **Ruff failures:** Auto-fix with `ruff check --fix` and `ruff format`. Commit the fixes.
- **MyPy failures:** Focus on the trading execution path (`src/quantstack/execution/`). Type stubs or `# type: ignore` for third-party libraries without stubs. Do not chase 100% type coverage.
- **Pytest failures:** Run locally first. Fix genuine bugs. Mark legitimately flaky tests with `@pytest.mark.skip(reason="flaky: <description>")`. Track skipped tests as tech debt.
- **Bandit failures:** Review HIGH-severity findings. Fix real issues, suppress false positives with `# nosec` + comment explaining why.
- **Trivy failures:** Fix CRITICAL/HIGH vulnerabilities by updating dependencies. Suppress known-unfixable CVEs with `.trivyignore`.

**1.3 Branch protection**

Enable GitHub branch protection on `main`:
- Require all status checks to pass before merge
- Do not require PR reviews (solo developer)

### Key Files

- `.github/workflows/ci.yml.disabled` → `.github/workflows/ci.yml`
- `.github/workflows/release.yml.disabled` → `.github/workflows/release.yml`
- `pyproject.toml` (pytest config, ruff config)

### Risks

The biggest risk is a cascade of failures that makes CI feel intractable. Mitigation: fix one stage at a time, starting with ruff (mechanical), then pytest (most value), then mypy, then bandit/trivy. Get the all-checks gate green even if individual stages have suppressed issues.

**1.4 Post-deployment smoke test**

Add to `start.sh` after `docker compose up -d`:
- Wait for Docker health checks to pass (poll `docker compose ps` for healthy status, timeout 120s)
- Log "System healthy" or fail with diagnostic output (`docker compose logs --tail=50`)
- This catches configuration errors and missing env vars immediately after deployment.

---

## Section 2: Log Aggregation + Alerting

### Context

Current logging: Loguru writes to stderr (colorized) and `~/.quantstack/logs/quantstack.jsonl` (structured, 50MB rotation, 30-day retention). Docker json-file driver rotates at 50MB × 5 files per container. No centralized log shipping. Logs are lost if the host fails.

Existing alerting: Discord webhook client (`src/quantstack/tools/discord/client.py`) and daily digest (`src/quantstack/coordination/daily_digest.py`) already post to Discord.

### What to Build

**2.1 Fluent Bit collector sidecar**

Add a `fluent-bit` service to `docker-compose.yml`:
- Mount `/var/lib/docker/containers/` read-only to read container JSON logs
- Parse Docker JSON format, extract container name as label
- Forward to Loki via the `loki` output plugin (Fluent Bit has native Loki support)
- Configuration file at `config/fluent-bit/fluent-bit.conf`

Input config: `[INPUT]` section using `tail` plugin on `/var/lib/docker/containers/*/*.log` with `Docker_Mode On` to handle multi-line logs. Tag with container name using `Tag_Regex`.

Output config: `[OUTPUT]` section using `loki` plugin pointing to `http://loki:3100/loki/api/v1/push` with labels for `job` (container name), `level` (parsed from JSON), `graph` (trading/research/supervisor).

**2.2 Loki log backend**

Add a `loki` service to `docker-compose.yml`:
- Single-binary mode (sufficient for single-host)
- Persistent volume for log data (`loki-data:/loki`)
- Retention: `retention_period: 720h` (30 days)
- Health check: HTTP GET to `/ready`
- Memory limit: 256MB (sufficient for label-only indexing)

Configuration at `config/loki/loki-config.yaml` with:
- `schema_config` using `boltdb-shipper` + `filesystem` storage
- `limits_config` with `retention_period: 720h`
- `compactor` enabled for retention enforcement

**2.3 Grafana dashboard and alerting**

Add a `grafana` service to `docker-compose.yml`:
- Pre-provisioned datasources: Loki (logs) + Prometheus (metrics from section 5)
- Pre-provisioned alert rules via YAML provisioning
- Exposed on port 3000
- Anonymous auth enabled for local access (single-host, not internet-facing)
- Persistent volume for dashboards and alert state

Provisioning directory structure:
```
config/grafana/provisioning/
  datasources/
    datasources.yaml      # Loki + Prometheus
  alerting/
    alerts.yaml           # LogQL and PromQL alert rules
  dashboards/
    dashboards.yaml       # Dashboard discovery config
    quantstack.json       # Pre-built system dashboard
```

**Alert rules:**

| Alert | Query | Severity | Channel |
|-------|-------|----------|---------|
| CRITICAL log | `rate({job=~".*-graph"} \|= "CRITICAL" [5m]) > 0` | Critical | Discord immediate |
| Error spike | `rate({job=~".*-graph"} \|= "ERROR" [5m]) > 5` | Warning | Discord |
| Kill switch active | `quantstack_kill_switch_active == 1` | Critical | Discord immediate |
| Container restart | `changes(container_start_time_seconds[5m]) > 0` | Warning | Discord |

**Contact points:**
- Discord webhook (primary, immediate)
- Email via Google mail bot (escalation, 4h+ — credentials to be configured by user separately, mark as TODO)

**Important: Grafana alerting is not the sole alerting path.** The existing supervisor graph Discord alerting (kill switch notifications, daily digest) must remain as a parallel path. Grafana adds log-based and metric-based alerts, but if Grafana goes down, the supervisor's direct Discord webhooks still fire for critical operational events.

**2.4 Integration with existing logging**

No changes to existing Loguru setup. Fluent Bit reads the Docker json-file driver output (which captures everything written to stdout/stderr by containers). The JSONL file (`~/.quantstack/logs/quantstack.jsonl`) continues as a local backup.

**2.5 Loki volume protection**

The `loki-data` named volume must NOT be included in any cleanup scripts. Document explicitly in `start.sh` and `stop.sh` that `docker compose down -v` destroys log history. The `stop.sh` script should use `docker compose down` (without `-v`).

### Key Files

- `docker-compose.yml` (3 new services)
- `config/fluent-bit/fluent-bit.conf` (new)
- `config/loki/loki-config.yaml` (new)
- `config/grafana/provisioning/` (new directory tree)

### Risks

Fluent Bit reading `/var/lib/docker/containers/` requires the Docker socket path to be correct on the host. On macOS (development), Docker Desktop uses a VM and paths differ from Linux (production). Mitigation: document the Docker socket/log path difference and provide both configurations.

---

## Section 3: Kill Switch Auto-Recovery

### Context

The kill switch (`src/quantstack/execution/kill_switch.py`, 455 lines) is a thread-safe singleton with a sentinel file (`~/.quantstack/KILL_SWITCH_ACTIVE`) for cross-process persistence. It has 4 auto-trigger conditions via `AutoTriggerMonitor`: consecutive broker failures, SPY halt, 3-day rolling drawdown, and model drift. Currently, ALL conditions require manual `reset(reset_by=...)`. There is no auto-recovery and no escalation.

### What to Build

**3.1 Auto-recovery for broker failures**

Add an `AutoRecoveryManager` class to `kill_switch.py` (or a new `kill_switch_recovery.py` module):

```python
class AutoRecoveryManager:
    """Manages tiered recovery for broker failure kill switch triggers.

    Only broker failures qualify for auto-recovery. All other kill switch
    conditions (drawdown, drift, SPY halt) require manual reset because
    they carry signal about model or market state.
    """
```

Recovery flow:
1. **0 min** — Kill switch triggers. Discord alert sent with: trigger reason, current open positions, unrealized P&L.
2. **5 min** — Auto-investigate: call broker health check endpoint. Is the API responding? Are other market data sources working?
3. **15 min** — If broker is responsive AND trigger was broker failures: auto-reset with `sizing_scalar=0.5`. Log the auto-reset with full context.
4. **Subsequent cycles** — Ramp sizing back: 0.5 → 0.75 → 1.0 over 3 successful trading cycles. Track ramp state in `system_state` table.
5. **If broker still down at 15 min** — Do not reset. Continue escalation.

**Safety cap: `MAX_AUTO_RESETS_PER_DAY = 2`.** If the system auto-resets twice in one day and triggers again, something structural is wrong. Stay halted and escalate. Track daily reset count in `system_state` table (key: `auto_resets_{YYYY-MM-DD}`).

The recovery manager runs as a background task in the supervisor graph. It checks kill switch state every cycle (5 min) and manages the recovery timeline. No EventBus dependency — the manager polls kill switch state directly via the singleton API and calls methods on it. This is a simple polling loop, not an event-driven architecture.

**3.2 Escalation tiers (all kill switch types)**

Add `KillSwitchEscalationManager`:
- Track `triggered_at` timestamp
- Escalation schedule:
  - **0 min** → Discord: trigger reason + positions + P&L
  - **4 hours** → Email (Google mail bot): full status report + what to do. If email not configured, send enhanced Discord message noting "email escalation would fire here but is not configured."
  - **24 hours** → Emergency Discord: all-caps header, full position summary, daily P&L, recommended actions

The escalation manager uses the supervisor graph cycle (every 5 min) to check elapsed time since trigger and fire appropriate notifications. It stores `last_escalation_tier` in `system_state` to avoid duplicate notifications.

**3.3 Sizing ramp-back mechanism**

After auto-reset from broker failure:
- Write `sizing_scalar` to `system_state` table (key: `kill_switch_sizing_scalar`)
- Write `successful_cycles_since_reset` counter
- Trading graph reads `sizing_scalar` before position sizing
- After 3 successful cycles: set scalar back to 1.0 and remove the key

The trading graph already has a position sizing pipeline. The scalar multiplies the final position size. Integration point: wherever `FORWARD_TESTING_SIZE_SCALAR` is applied, also apply kill switch sizing scalar.

**3.4 Reset reason logging**

Modify `reset()` to accept an optional `reason` parameter (in addition to existing `reset_by`):

```python
def reset(self, reset_by: str, reason: str = "") -> None:
    """Reset kill switch. Reason is logged for audit trail. Warn if empty."""
```

Making `reason` optional avoids breaking existing callers. Log a warning when reason is empty to encourage providing context.

Log the reason to both the application log and a new `kill_switch_events` table (or append to `system_state` with a timestamped key).

### Key Files

- `src/quantstack/execution/kill_switch.py` (modify)
- `src/quantstack/graphs/supervisor/nodes.py` (add recovery/escalation checks)
- `src/quantstack/coordination/daily_digest.py` (add kill switch event formatting)

### Risks

The auto-recovery background task must not interfere with the kill switch's thread safety. The `AutoRecoveryManager` should only call `reset()` through the existing API — it must not directly modify state. The `reset()` call itself handles locking and sentinel file cleanup.

Race condition: supervisor graph cycle fires recovery at the same time as a new trigger condition. Mitigation: `reset()` already uses a Lock(). After reset, the auto-trigger monitor may immediately re-trigger if the condition persists. The recovery manager should detect this (reset_at very recent + triggered again) and back off.

---

## Section 4: Alembic Migration Framework

### Context

`src/quantstack/db.py` (2396 lines) contains 37+ `_migrate_*_pg()` functions, each running `CREATE TABLE IF NOT EXISTS`. They're called sequentially from `run_migrations()` at startup, protected by PostgreSQL advisory lock `42`. There is no `schema_migrations` table and no way to track which migrations have run, failed, or need re-running.

### What to Build

**4.1 Alembic setup**

Install `alembic` as a dependency. Create standard Alembic directory structure:

```
alembic/
  env.py            # Migration environment (reads TRADER_PG_URL)
  script.py.mako    # Migration template
  versions/         # Migration files
alembic.ini         # Configuration (sqlalchemy.url from env var)
```

`env.py` must:
- Read `TRADER_PG_URL` from environment (same as `db.py`)
- Acquire advisory lock `42` before running migrations (same lock as current system, prevents concurrent migration)
- Release advisory lock after completion

**4.2 Baseline migration**

Create a single baseline migration that represents all 37+ existing tables. This migration should:
- Use `CREATE TABLE IF NOT EXISTS` for every table (same as current code)
- Be marked as "already applied" on existing databases via `alembic stamp head`
- On fresh databases, actually create all tables

Approach: extract the DDL from each `_migrate_*_pg()` function into the baseline migration's `upgrade()` function. The `downgrade()` function drops all tables in reverse dependency order (for development use only — never run in production).

**4.3 Startup integration**

Replace the current startup flow:
```
# Current: run_migrations(conn) → 37 _migrate_*_pg() calls
# New: alembic.command.upgrade(config, "head")
```

The `run_migrations()` function in `db.py` becomes a thin wrapper around `alembic upgrade head`. It should:
- Create Alembic config programmatically (no need to read `alembic.ini` at runtime)
- Acquire advisory lock 42 (same as current)
- Run `upgrade("head")`
- Release lock

Keep the `_migrations_done` module flag to prevent re-running within the same process.

**4.4 Migration for existing databases**

For databases that already have all tables (production):
1. `alembic upgrade head` runs the baseline migration
2. Baseline migration uses `IF NOT EXISTS` — no-ops on existing tables
3. Alembic records the baseline version in `alembic_version` table

This is safe because the baseline migration is idempotent.

**4.5 Alembic fallback for first deployment**

Add `USE_ALEMBIC=true` env flag. During the transition period:
- If `USE_ALEMBIC=true`: run `alembic upgrade head`
- If `USE_ALEMBIC=false` (or unset): fall back to the old `run_migrations()` path

This allows rolling back to the old migration system if Alembic introduces issues on the first production deployment. Remove the fallback after 1 week of stable operation.

**4.6 Future migration workflow**

After Alembic is in place, migrations are hand-written (the codebase uses raw SQL, not SQLAlchemy ORM models, so `--autogenerate` will not work):
```bash
# Create a new migration manually
alembic revision -m "add rate_limit_buckets table"
# Edit the generated file: write upgrade() and downgrade() SQL

# Apply
alembic upgrade head

# Verify
alembic current
```

### Key Files

- `src/quantstack/db.py` (modify `run_migrations()`)
- `alembic/` (new directory)
- `alembic.ini` (new)
- `pyproject.toml` (add alembic dependency)

### Risks

The biggest risk is the baseline migration not exactly matching the current schema. If a `_migrate_*_pg()` function does something beyond `CREATE TABLE IF NOT EXISTS` (e.g., adding columns, creating indexes), the baseline migration must capture that. Mitigation: introspect the running database schema and diff against the baseline DDL before declaring it correct.

---

## Section 5: Container Monitoring Stack

### Context

QuantStack already exposes Prometheus metrics at `/metrics` (trades, risk rejections, latency, NAV, P&L, kill switch state) but no Prometheus server scrapes them. Docker Compose sets memory limits on all containers but has no monitoring of actual usage or OOMKilled events.

### What to Build

**5.1 cAdvisor service**

Add to `docker-compose.yml`:
- Image: `gcr.io/cadvisor/cadvisor:v0.49.1`
- Mounts: `/var/run/docker.sock`, `/sys`, `/var/lib/docker` (all read-only)
- Port: 8080 (internal only, scraped by Prometheus)
- Memory limit: 128MB

Provides container-level metrics: CPU, memory (working set + cache), network, disk I/O, and critically `container_oom_events_total`.

**5.2 Prometheus service**

Add to `docker-compose.yml`:
- Image: `prom/prometheus:v2.53.0`
- Persistent volume for time-series data
- Port: 9090 (internal, accessible via Grafana)
- Memory limit: 256MB

Scrape config at `config/prometheus/prometheus.yml`:

```yaml
scrape_configs:
  - job_name: cadvisor
    static_configs:
      - targets: ['cadvisor:8080']
  - job_name: trading-graph
    static_configs:
      - targets: ['trading-graph:8000']
    metrics_path: /metrics
  - job_name: research-graph
    static_configs:
      - targets: ['research-graph:8000']
  - job_name: supervisor-graph
    static_configs:
      - targets: ['supervisor-graph:8000']
```

Retention: `--storage.tsdb.retention.time=15d` (container metrics are high-cardinality; 15 days is sufficient for operational monitoring).

**5.3 Grafana datasource and alerting**

The Grafana service from Section 2 gets a Prometheus datasource alongside Loki.

Alert rules (PromQL):

| Alert | Expression | For | Severity |
|-------|-----------|-----|----------|
| Memory warning | `container_memory_working_set_bytes{name=~".*-graph"} / container_spec_memory_limit_bytes{name=~".*-graph"} > 0.80` | 2m | Warning |
| Memory critical | `... > 0.90` | 1m | Critical |
| OOMKilled | `increase(container_oom_events_total{name=~".*-graph\|postgres\|langfuse"}[5m]) > 0` | 0m | Critical |
| Container restart | `changes(container_start_time_seconds[5m]) > 0` | 0m | Warning |

Use `container_memory_working_set_bytes` (not `container_memory_usage_bytes`) because working set excludes reclaimable cache and is what the OOM killer evaluates.

**5.4 Grafana dashboard with alert panels**

Create a `quantstack.json` dashboard with panels:
- **Active alerts panel** — shows all currently firing alerts with severity badges (Grafana built-in "Alert list" panel type)
- **Alert history panel** — time series of alert state transitions over last 24h (shows when alerts fired and resolved)
- Container memory usage (% of limit) — bar gauge per service
- Container CPU usage — time series per service
- OOM events — stat panel (should be 0)
- Trading metrics: trades executed, risk rejections, agent latency, portfolio NAV, daily P&L, kill switch state (from existing Prometheus metrics)
- Log volume by level — from Loki
- **Kill switch status** — stat panel showing current state (active/inactive) with color coding (red=active, green=inactive)

The alert panels should be at the top of the dashboard so the most critical information is visible first. The dashboard should be the first thing checked when investigating an issue.

### Key Files

- `docker-compose.yml` (add cAdvisor, Prometheus; Grafana already from Section 2)
- `config/prometheus/prometheus.yml` (new)
- `config/grafana/provisioning/alerting/alerts.yaml` (extend with PromQL rules)
- `config/grafana/provisioning/dashboards/quantstack.json` (new)

### Risks

cAdvisor on macOS Docker Desktop may not report all metrics correctly (it's primarily Linux-oriented). Memory metrics should work. CPU metrics may differ. Mitigation: test on Linux deployment target and document macOS limitations for development.

---

## Section 6: Health Check Granularity

### Context

The supervisor graph runs every 5 minutes with a `health_check` node that checks kill switch status and heartbeat freshness for trading-graph (120s max) and research-graph (600s max). It classifies each as healthy/degraded/critical. There's no tracking of cycle success rate, error trends, or strategy generation velocity.

### What to Build

**6.1 Extended health metrics collector**

Add a `collect_health_metrics()` function to the supervisor health check node that queries:

- **Cycle success rate** (per graph, last 10 cycles): query `graph_checkpoints` table for `status` column. Calculate `success_count / total_count`.
- **Error count per cycle** (last cycle): count of `status='error'` in `graph_checkpoints` for the most recent cycle.
- **Strategy generation velocity**: count of strategies created in the last 7 days from `strategies` table.
- **Research queue depth**: count of pending research tasks from relevant table/queue.

Store computed metrics in `system_state` table with keys like `health_metric_trading_success_rate`, `health_metric_research_error_count`, etc.

**6.2 Threshold alerting**

Integrate with the Grafana alerting from Section 2/5. Write health metrics as Prometheus gauges (extend `src/quantstack/observability/metrics.py`):

```python
def register_health_metrics():
    """Register health-check Prometheus gauges."""
    # cycle_success_rate: Gauge, labels=[graph_name]
    # cycle_error_count: Gauge, labels=[graph_name]
    # strategy_generation_7d: Gauge
    # research_queue_depth: Gauge
```

The supervisor graph updates these gauges each cycle. Grafana alerts on:
- Cycle success rate < 70% for 15 min → Warning
- Error count per cycle > 3 → Warning
- 0 new strategies in 7 days → Info (notification, not pager)
- Research queue > 50 → Warning

**6.3 IC trend monitoring (DEFERRED)**

IC trend monitoring is deferred to a later phase. Computing rolling IC (correlation between signal scores and future returns) is a research-grade statistical computation that doesn't belong in an operational resilience phase. It requires specifying lookback windows, sample size requirements, and re-enablement criteria — all of which are research decisions, not ops decisions.

For now, the health check covers: cycle success rate, error counts, strategy generation velocity, and research queue depth. These are pure operational metrics derivable from existing tables with simple queries.

### Key Files

- `src/quantstack/graphs/supervisor/nodes.py` (extend health_check node)
- `src/quantstack/observability/metrics.py` (add health gauges)
- `config/grafana/provisioning/alerting/alerts.yaml` (add health alert rules)

---

## Section 7: Shared Rate Limiter

### Context

`src/quantstack/data/fetcher.py` has a per-process rate limiter: in-memory counter that resets every 60s, 75 req/min limit. With 3 containers (trading, research, supervisor), effective rate is 225 req/min to Alpha Vantage (real limit: 75/min). Daily quota tracking is already atomic via PostgreSQL `INSERT ... ON CONFLICT DO UPDATE`.

### What to Build

**7.1 Rate limit bucket table and PL/pgSQL function**

Create via Alembic migration (depends on Section 4):

Table `rate_limit_buckets`:
- `bucket_key TEXT PRIMARY KEY` — e.g., `'alpha_vantage'`
- `tokens NUMERIC NOT NULL` — current available tokens
- `max_tokens NUMERIC NOT NULL` — bucket capacity
- `refill_rate NUMERIC NOT NULL` — tokens per second
- `last_refill TIMESTAMPTZ NOT NULL DEFAULT now()`

Function `consume_token(p_key TEXT, p_cost NUMERIC DEFAULT 1) RETURNS BOOLEAN`:
1. `SELECT * FROM rate_limit_buckets WHERE bucket_key = p_key FOR UPDATE` (row lock)
2. Calculate elapsed time since last_refill using `clock_timestamp()` (NOT `now()` — `now()` returns transaction start time, which would freeze the refill calculation if wrapped in a larger transaction)
3. Refill tokens: `LEAST(max_tokens, tokens + elapsed * refill_rate)`
4. If tokens >= cost: consume and return TRUE
5. Else: update refill timestamp (prevent drift) and return FALSE

Seed data: `INSERT INTO rate_limit_buckets VALUES ('alpha_vantage', 75, 75, 1.25, now())` — 75 tokens, refill at 1.25/sec = 75/min.

**7.2 Python integration**

Replace per-process `_wait_for_rate_limit()` in `fetcher.py` with a non-blocking function that:
1. Calls `SELECT consume_token('alpha_vantage')` via a short-lived DB connection (not holding a connection open during backoff)
2. If TRUE: proceed with the API call
3. If FALSE: return to caller with a "rate limited" signal. The caller uses `asyncio.sleep(1)` and retries (up to 60 retries = 60s max wait). This avoids holding a DB connection for up to 60s during backoff.

**Circuit breaker:** If `consume_token` raises a database exception (connection error, not a FALSE return), fall back to the old per-process in-memory rate limiter for that call. Log WARNING. This prevents a DB outage from blocking all API calls permanently — the per-process limiter is less accurate but functional.

Remove the in-memory `_call_count` and `_minute_start` as the primary path, but keep them as the fallback limiter behind the circuit breaker.

The daily quota check stays unchanged (already uses PostgreSQL atomically).

**7.3 Backpressure signal**

When `consume_token` returns FALSE, the caller should log at DEBUG level: "Rate limit: waiting for token (bucket: alpha_vantage)". This is normal operation, not an error. If all 60 retries exhaust, log WARNING and skip the call (same as current behavior when daily quota is exceeded).

### Key Files

- `src/quantstack/data/fetcher.py` (modify rate limit logic)
- `alembic/versions/` (new migration for `rate_limit_buckets` + `consume_token` function)

### Risks

If the PostgreSQL connection is down, the rate limiter blocks all API calls (can't acquire token). This is actually desirable — if the DB is down, the system has bigger problems and shouldn't be making external API calls. The existing health check will detect the DB outage.

---

## Section 8: Langfuse Retention (Config Stub)

### Context

Langfuse keeps all traces indefinitely. `langfuse-db` has 256MB memory. 3 graphs producing traces every 5-10 min will fill disk within weeks. However, the user wants to defer actual cleanup — just add the configuration wiring.

### What to Build

**8.1 Configuration flags**

Add to `.env.example`:
```
LANGFUSE_RETENTION_ENABLED=false
LANGFUSE_RETENTION_DAYS=30
```

Add to env var validation (Section 9): validate `LANGFUSE_RETENTION_ENABLED` as boolean and `LANGFUSE_RETENTION_DAYS` as positive integer, but both are optional with defaults.

**8.2 Scheduler job stub**

Add a `langfuse_retention_cleanup` job to `scripts/scheduler.py`:
- Schedule: weekly (Sunday 02:00 ET)
- Implementation: check `LANGFUSE_RETENTION_ENABLED`. If false, log "Langfuse retention cleanup is disabled. Set LANGFUSE_RETENTION_ENABLED=true to enable." and return.
- If true: placeholder that logs "Langfuse retention cleanup: would delete traces older than {LANGFUSE_RETENTION_DAYS} days (implementation pending)".

This wires the scheduler and config. Actual deletion logic will be implemented when the user is ready to enable it.

### Key Files

- `scripts/scheduler.py` (add job)
- `.env.example` (add vars)

---

## Section 9: Secrets and Env Var Hardening

### Context

API keys, DB credentials, and broker secrets are plain text in `.env`. No permissions enforcement, no type validation, no rotation documentation.

### What to Build

**9.1 .env permissions check in start.sh**

Add to `start.sh` before `docker compose up`:
- Check if `.env` exists
- Check permissions: `stat -c '%a' .env` (Linux) or `stat -f '%Lp' .env` (macOS)
- If not 600: warn and offer to fix (`chmod 600 .env`)
- If `.env` doesn't exist: error with instructions to copy from `.env.example`

**9.2 Env var type validation module**

Create `src/quantstack/config/validation.py`:

```python
def validate_environment() -> None:
    """Validate all critical environment variables at startup.

    Raises SystemExit(1) on any validation failure.
    """
```

Validation tiers:

**Required (crash if missing or invalid):**
- `TRADER_PG_URL` — must be valid PostgreSQL URL
- `ALPHA_VANTAGE_API_KEY` — must be non-empty string
- `ALPACA_API_KEY`, `ALPACA_SECRET_KEY` — must be non-empty strings

**Typed (crash if present but invalid type):**
- `RISK_MAX_POSITION_PCT` — must parse as float, range 0.0-1.0
- `AV_DAILY_CALL_LIMIT` — must parse as positive integer
- `FORWARD_TESTING_SIZE_SCALAR` — must parse as float, range 0.0-1.0
- `ROLLING_DRAWDOWN_MULTIPLIER` — must parse as positive float

**Boolean (crash if present but not true/false):**
- `ALPACA_PAPER`, `USE_REAL_TRADING`, `USE_FORWARD_TESTING_FOR_ENTRIES`, `LANGFUSE_RETENTION_ENABLED`

**Optional (warn if missing, use defaults):**
- `GROQ_API_KEY`, `DISCORD_WEBHOOK_URL`, `RESEARCH_SYMBOL_OVERRIDE`

On failure: log the exact var name, expected type/range, actual value (redacted if it looks like a secret), and exit.

Call `validate_environment()` at the top of each runner (`trading_runner.py`, `research_runner.py`, `supervisor_runner.py`) before any graph initialization.

**9.3 .env.example documentation**

Ensure `.env.example` has every variable with:
- Description comment
- Expected type and range
- Whether required or optional
- Default value if optional

**9.4 Credential rotation documentation**

Add a "Credential Rotation" section to `docs/ops-runbook.md`:
- Alpha Vantage: regenerate at alphavantage.co, update `.env`, restart containers
- Alpaca: regenerate in Alpaca dashboard, update `.env`, restart containers
- PostgreSQL: update password in PG, update `TRADER_PG_URL`, restart containers
- Langfuse: regenerate keys in Langfuse settings, update `.env`, restart
- Discord: regenerate webhook URL, update `.env`, restart

### Key Files

- `start.sh` (add permissions check)
- `src/quantstack/config/validation.py` (new)
- Runner entry points (add `validate_environment()` call)
- `.env.example` (update documentation)
- `docs/ops-runbook.md` (add rotation section)

---

## Section 10: SBOM Scanning (Deferred)

### Context

Depends on Section 1 (CI/CD re-enabled and green). The existing `ci.yml` already includes Trivy for Docker image scanning.

### What to Build

After CI is green (Section 1 complete):

**10.1 pip-audit in CI**

Add a step to `ci.yml`:
- Uses `pypa/gh-action-pip-audit@v1.1.0`
- Scans installed packages against Python Packaging Advisory Database
- Fails the build on CRITICAL/HIGH vulnerabilities
- Suppresses known-unfixable CVEs via `.pip-audit-known-vulnerabilities` file

**10.2 SBOM generation**

Add `cyclonedx-py` as a dev dependency. Add CI step to generate SBOM:
- `cyclonedx-py environment --output sbom.json --output-format json`
- Upload as build artifact for audit trail

This section is intentionally minimal — it depends on CI being stable first.

### Key Files

- `.github/workflows/ci.yml` (add pip-audit + cyclonedx steps)
- `pyproject.toml` (add cyclonedx-py to dev dependencies)

---

## Section 11: Implementation Order and Dependencies

### Memory Budget

After all additions, the Docker Compose stack (14 services):

| Service | Memory Limit | Status |
|---------|-------------|--------|
| postgres | 512MB | existing |
| langfuse-db | 256MB | existing |
| ollama | 4GB | existing |
| langfuse | 512MB | existing |
| trading-graph | 1GB | existing |
| research-graph | 1GB | existing |
| supervisor-graph | 512MB | existing |
| dashboard | 256MB | existing |
| finrl-worker | 2GB | existing |
| fluent-bit | 64MB | **new** |
| loki | 256MB | **new** |
| grafana | 256MB | **new** |
| cadvisor | 128MB | **new** |
| prometheus | 256MB | **new** |
| **Total** | **~11.0 GB** | |

Ensure the host has at least 16GB RAM to leave headroom for the OS and burst usage.

### Dependency Graph

```
Section 1 (CI/CD) ─────────────────────────────────── Section 10 (SBOM)
                                                         (after CI green)

Section 2 (Log Aggregation) ──┐
                               ├── Grafana shared ──── Section 5 (Monitoring)
Section 5 (Monitoring) ────────┘

Section 4 (Alembic) ──────────── Section 7 (Rate Limiter migration)

Section 3 (Kill Switch Recovery) ── no external dependencies (polls state directly)

Sections 6, 8, 9 ── independent
```

### Recommended Implementation Order

1. **Section 9** (Env var validation) — foundation, no dependencies, small
2. **Section 4** (Alembic) — migration framework needed by Section 7
3. **Section 1** (CI/CD) — re-enable and fix, parallel with Section 4
4. **Section 7** (Rate limiter) — uses Alembic from Section 4
5. **Section 2** (Log aggregation) — Fluent Bit + Loki + Grafana
6. **Section 5** (Monitoring) — cAdvisor + Prometheus, shares Grafana from Section 2
7. **Section 6** (Health check granularity) — uses Prometheus from Section 5
8. **Section 3** (Kill switch recovery) — most complex, no external blockers
9. **Section 8** (Langfuse retention stub) — trivial, do anytime
10. **Section 10** (SBOM) — after CI is green

Parallelization: Sections 1+4+9 can run simultaneously. Sections 2+5 should be done together (shared Grafana).
