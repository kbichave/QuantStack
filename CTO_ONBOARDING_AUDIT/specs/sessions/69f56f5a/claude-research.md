# Research Findings — Phase 3: Operational Resilience

## Part 1: Codebase Analysis

### 1. Docker Compose Setup

**File:** `docker-compose.yml`

| Service | Image | Memory Limit | Health Check |
|---------|-------|-------------|--------------|
| postgres | pgvector/pgvector:pg16 | 512m | pg_isready every 10s |
| langfuse-db | postgres:16-alpine | 256m | pg_isready every 10s |
| ollama | ollama/ollama | 4g | ollama list every 15s |
| langfuse | langfuse/langfuse:2 | 512m | HTTP every 15s |
| trading-graph | Custom | 1g | Heartbeat file check |
| research-graph | Custom | 1g | Heartbeat file check |
| supervisor-graph | Custom | 512m | Heartbeat file check |
| dashboard | Custom | 256m | HTTP /api/status |
| finrl-worker | Custom (Dockerfile.finrl) | 2g | HTTP check |

- **Logging:** All services use `json-file` driver with 50MB max, 5 files rotation
- **Volumes:** Shared bind mounts for code, models, logs, memory, execution state
- **Networks:** Single `quantstack-net` bridge
- **Restart:** `unless-stopped` for all
- **Graceful shutdown:** 90s `stop_grace_period` for graph services

### 2. CI/CD Workflows (DISABLED)

**Files:** `.github/workflows/ci.yml.disabled`, `.github/workflows/release.yml.disabled`

CI pipeline structure when enabled:
- **Lint & Type Check:** Ruff linter/formatter, MyPy on trading execution path
- **Unit Tests:** pytest with coverage (target 60%+), multi-OS (ubuntu/macos), Python 3.11/3.12
- **Integration Tests:** Nightly schedule, real DB
- **Security Scan:** Bandit (HIGH severity only), skips RL module
- **Docker Build & Scan:** Trivy vulnerability scan on main branch
- **All Checks Gate:** Blocks merges until lint, test, security pass

**Test Config (pyproject.toml):**
```
pythonpath = ["src"]
testpaths = ["tests", "src/quantstack/core/tests"]
asyncio_mode = "auto"
markers: slow, integration, regression, requires_api, requires_gpu
```

30+ unit tests in `tests/unit/`. Fixtures in `tests/conftest.py` provide `trading_ctx`, `signal_cache`, `risk_state`, `portfolio`, `paper_broker`, `kill_switch`, `tick_executor`, `sample_ohlcv_df`.

### 3. Kill Switch

**File:** `src/quantstack/execution/kill_switch.py` (455 lines)

Architecture:
- Singleton with thread-safe Lock()
- Sentinel file: `~/.quantstack/KILL_SWITCH_ACTIVE` (survives restarts)
- Cross-process safe via file fallback

**AutoTriggerMonitor — 4 auto-triggers:**
1. **Consecutive broker failures** — 3 failures (configurable `KILL_MAX_BROKER_FAILURES`)
2. **Market circuit breaker** — SPY halted detection
3. **Rolling 3-day drawdown** — 3x daily loss limit (default 6%)
4. **Model drift** — >50% of strategies show drift

**Reset:** Explicit `reset(reset_by=...)` required. No auto-reset exists. Records timestamp + who triggered. Atomically deletes sentinel file.

**Registration:** `register_position_closer(lambda: broker.close_all())`, signal handlers for SIGTERM/SIGINT.

### 4. Database Migrations

**File:** `src/quantstack/db.py` (2396 lines)

- **37+ migration functions** called sequentially at startup
- All use `CREATE TABLE IF NOT EXISTS` (idempotent)
- Advisory lock (`_MIGRATION_ADVISORY_LOCK = 42`) prevents concurrent migrations
- Module flag `_migrations_done` prevents re-running in same process
- **No `schema_migrations` table** — no tracking of which migrations ran

Tables span: positions, cash_balance, closed_trades, fills, signal_state, strategies, regime_strategy_matrix, system_state, loop_events, loop_cursors, loop_heartbeats, graph_checkpoints, agent_memory, agent_skills, daily_equity, ohlcv, universe, and 20+ more.

### 5. Rate Limiting

**File:** `src/quantstack/data/fetcher.py`

**Per-process tracking:**
- In-memory counter `_call_count`, `_minute_start`
- 75 req/min limit, resets every 60s, sleeps if hit

**Daily quota tracking (persistent in DB):**
- `system_state` table, key: `av_daily_calls_{YYYY-MM-DD}`
- Atomic increment: `INSERT ... ON CONFLICT DO UPDATE SET value = (CAST(value AS INTEGER) + 1)::TEXT`
- 25,000 calls/day limit
- Priority-based throttling: low priority skips at 50%, normal at 80%, critical always proceeds

**Problem:** Per-minute limit is per-process. 3 containers × 75 = 225 req/min to AV (limit is 75).

### 6. Health Checks

**Files:** `src/quantstack/health/heartbeat.py`, `src/quantstack/health/watchdog.py`

**Heartbeat:** File-based (`/tmp/{crew_name}-heartbeat`), writes unix timestamp after each cycle.
- Trading: 120s max age
- Research: 600s max age
- Supervisor: 360s max age

**Watchdog:** `AgentWatchdog(timeout_seconds=600, on_timeout=handle_stuck)` — detects stuck cycles.

**Supervisor health_check node:** Checks kill switch status + heartbeat freshness for trading-graph and research-graph. Classifies as healthy/degraded/critical.

### 7. Logging

**File:** `src/quantstack/observability/logging.py`

Dual output:
1. **Stderr** — colorized, human-readable, INFO+
2. **JSONL file** — `~/.quantstack/logs/quantstack.jsonl`, DEBUG+, 50MB rotation, 30-day retention

Fields: timestamp, level, message, module, function, line, trace_id (Langfuse), extra fields via `logger.bind()`.

Uses **Loguru** framework. Docker json-file driver provides container-level rotation (50MB × 5 files).

**No log shipping** — logs stay on host.

### 8. Existing Alerting & Monitoring

**Discord:** `src/quantstack/tools/discord/client.py` — async REST client with rate limit handling.

**Daily Digest:** `src/quantstack/coordination/daily_digest.py` — generates DigestReport with portfolio, strategy lifecycle, loop health, risk, ML stats. Output as markdown for Discord + JSON.

**Prometheus Metrics:** `src/quantstack/observability/metrics.py` — counter/gauge/histogram metrics exposed at `/metrics`:
- `quantstack_trades_executed_total`, `quantstack_risk_rejections_total`
- `quantstack_agent_latency_seconds`, `quantstack_signal_staleness_seconds`
- `quantstack_portfolio_nav_dollars`, `quantstack_daily_pnl_dollars`
- `quantstack_kill_switch_active`

**Langfuse:** Per-cycle tracing, instrumented at runner startup.

### 9. Scheduler

**File:** `scripts/scheduler.py` — APScheduler with CronTrigger

| Job | Schedule | Purpose |
|-----|----------|---------|
| data_refresh | 08:00 Mon-Fri ET | Refresh daily OHLCV, options, fundamentals |
| eod_data_refresh | 16:30 Mon-Fri ET | Close bar, options chains, news, macro |
| strategy_lifecycle_weekly | 18:00 Sun ET | Gap analysis, backtest, promote drafts |
| strategy_lifecycle_monthly | 09:00 1st of month | Validate live, retire degraded |

2-hour timeout per job. No LLM calls in scheduler.

### 10. Graceful Shutdown

**File:** `src/quantstack/health/shutdown.py`

`GracefulShutdown` class: catches SIGTERM/SIGINT, runs cleanup callbacks (Langfuse flush, DB close) with 10s timeout per callback. 90s stop_grace_period in Docker Compose.

---

## Part 2: Web Research — Best Practices

### 1. GitHub Actions CI/CD for Python Docker Projects

**Python testing:**
- `actions/setup-python@v5` with `cache: 'pip'` for fast installs
- Matrix builds for Python 3.11/3.12
- Ruff for lint/format (`ruff check --output-format=github`)
- MyPy as separate step (doesn't block test results)
- `--junitxml` for structured test reports

**Docker images:**
- **ghcr.io** preferred for private projects (uses `GITHUB_TOKEN`, no extra credentials)
- `docker/build-push-action@v6` with `cache-from: type=gha` / `cache-to: type=gha,mode=max` for layer caching
- Tag with git SHA + branch name
- Only push on main; build-only on PRs

**Vulnerability scanning:**
- `pypa/gh-action-pip-audit@v1.1.0` — scans against Python Packaging Advisory Database + OSV
- `cyclonedx-py environment --output sbom.json` for OWASP CycloneDX SBOM generation
- Run pip-audit as merge gate (exit 1 on vulnerabilities)

**Smoke testing:**
```yaml
- run: |
    docker compose up -d
    sleep 10
    curl --fail http://localhost:8000/health || exit 1
    docker compose down
```

**Branch protection:** Require all checks, 1 approval, block force push, linear history.

### 2. Log Aggregation for Docker Compose

**Collector choice:**

| Collector | Memory | Recommendation |
|-----------|--------|---------------|
| **Fluent Bit** | ~1-2 MB | Best for single-host Docker Compose |
| Fluentd | ~40-100 MB | Overkill for single-host |
| Filebeat | ~30-60 MB | Only if already running Elasticsearch |

**Backend choice:**

| Backend | Cost | Memory | Recommendation |
|---------|------|--------|---------------|
| **Grafana Loki** | Free (self-hosted) | ~200 MB | Best for single-host, label-based queries |
| Elasticsearch | Free but RAM-hungry | 2+ GB | Overkill |
| CloudWatch | ~$0.50/GB ingestion | N/A | AWS-native alternative |

**Recommended stack:** Fluent Bit → Loki → Grafana (~200 MB total)

**Alerting via LogQL in Grafana:**
```
rate({job="trading-graph"} |= "CRITICAL" [5m]) > 0
rate({job="trading-graph"} |= "ERROR" [5m]) > 5
```
Grafana has built-in Discord/Slack contact points.

**Retention:** Loki `retention_period` configurable. 30 days for operational logs, 90+ days for trade audit.

### 3. PostgreSQL Shared Rate Limiting

**Token bucket in PostgreSQL** (recommended over advisory locks for rate control):

```sql
CREATE TABLE rate_limit_buckets (
    bucket_key    TEXT PRIMARY KEY,
    tokens        NUMERIC NOT NULL,
    max_tokens    NUMERIC NOT NULL,
    refill_rate   NUMERIC NOT NULL,  -- tokens per second
    last_refill   TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

- Use `SELECT ... FOR UPDATE` row lock (not advisory locks) for token bucket
- `consume_token()` PL/pgSQL function: refill based on elapsed time, atomic consume-or-reject
- Initialize: 75 tokens, refill at 1.25/sec (75/min)

**Daily quota:** `INSERT ... ON CONFLICT DO UPDATE` is already atomic and race-free (current implementation is correct).

**Performance:** 75/min is trivial for PostgreSQL. Advisory locks handle millions of ops/sec. Bottleneck will never be the rate limiter.

**PostgreSQL vs Redis:** At 75 calls/min, PostgreSQL is the right choice. No additional infrastructure. Only add Redis if >1000 req/sec.

**Advisory locks are better suited for:** singleton coordination (e.g., only one container runs scheduler).

### 4. Docker Container OOM Monitoring

**Recommended stack:** cAdvisor + Prometheus + Grafana (~150 MB overhead)

**Key metric:** `container_oom_events_total{name="trading-graph"}`

**Alert tiers:**
| Threshold | Action |
|-----------|--------|
| >70% of limit | Warning — Discord notification |
| >85% of limit | Critical — trigger GC, reduce batch sizes |
| >95% of limit | Emergency — graceful shutdown before OOM |
| OOMKilled event | Post-mortem — restart + capture context |

**Use `container_memory_working_set_bytes`** (not `container_memory_usage_bytes`) — working set excludes reclaimable cache and is what the OOM killer actually uses.

**OOM detection methods:**
1. cAdvisor `container_oom_events_total` (best — Prometheus alert)
2. `docker inspect --format='{{.State.OOMKilled}}'` (scripted polling)
3. `docker events --filter event=oom` (real-time stream → webhook)

**Health checks beyond ping:** Endpoint should report cycle age, error rate, positions monitored, kill switch state, DB connectivity. Return 503 when unhealthy.

**Since Grafana already runs for Langfuse:** Add Prometheus + cAdvisor to existing Grafana. Zero additional UI overhead.

---

## Implications for Phase 3 Implementation

### Already in good shape (existing code):
- Kill switch auto-triggers (4 conditions) — just needs auto-recovery + escalation
- Heartbeat system — needs granularity expansion
- Prometheus metrics — needs cAdvisor for container-level
- Discord alerting + daily digest — needs real-time CRITICAL webhook
- Advisory lock for migrations — needs schema_migrations table on top
- Atomic daily quota tracking — needs shared per-minute rate limiting

### Key decisions for the plan:
1. **Log stack:** Fluent Bit + Loki + existing Grafana (not Elasticsearch)
2. **Rate limiter:** PostgreSQL token bucket with `FOR UPDATE` (not Redis)
3. **OOM monitoring:** cAdvisor + Prometheus + existing Grafana
4. **CI registry:** ghcr.io (not DockerHub)
5. **Kill switch recovery:** Tiered escalation, auto-reset only for transient conditions
