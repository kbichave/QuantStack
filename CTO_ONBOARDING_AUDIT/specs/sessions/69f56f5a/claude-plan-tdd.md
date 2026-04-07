# TDD Plan: Phase 3 — Operational Resilience

**Testing framework:** pytest with asyncio, fixtures in `tests/conftest.py`
**Test location:** `tests/unit/` for unit tests, `tests/integration/` for DB-dependent tests
**Conventions:** Existing codebase uses `test_*.py` naming, `conftest.py` fixtures provide `trading_ctx`, `signal_cache`, `risk_state`, `portfolio`, `paper_broker`, `kill_switch`. Markers: `slow`, `integration`, `regression`, `requires_api`, `requires_gpu`.

---

## Section 1: CI/CD Pipeline Re-enablement

No new test code to write — this section re-enables existing CI and fixes breakage. The testing work IS the implementation:

- Run `pytest tests/` locally, identify all failures
- Categorize: genuine bug vs. flaky vs. missing fixture vs. environment-dependent
- Fix genuine bugs, mark flaky with `@pytest.mark.skip(reason="...")`
- Verify `ruff check` and `mypy` pass or suppressions are documented

---

## Section 2: Log Aggregation + Alerting

Primarily infrastructure (Docker Compose config files). Limited unit test surface.

```python
# tests/integration/test_log_aggregation.py (marker: @pytest.mark.integration)

# Test: Fluent Bit config parses without errors (run fluent-bit --dry-run)
# Test: Loki accepts log push via HTTP API (/loki/api/v1/push)
# Test: Grafana provisions datasources on startup (GET /api/datasources returns loki + prometheus)
# Test: Grafana alert rules are provisioned (GET /api/v1/provisioning/alert-rules returns expected rules)
# Test: Discord contact point is configured (GET /api/v1/provisioning/contact-points)
```

These are integration tests that require running Docker Compose services. Mark with `@pytest.mark.integration`.

---

## Section 3: Kill Switch Auto-Recovery

```python
# tests/unit/test_kill_switch_recovery.py

# --- AutoRecoveryManager ---
# Test: manager does nothing when kill switch is not active
# Test: manager does nothing when kill switch triggered by drawdown (not broker failure)
# Test: manager does nothing when kill switch triggered by drift
# Test: manager does nothing when kill switch triggered by SPY halt
# Test: manager initiates investigation at 5 min after broker failure trigger
# Test: manager auto-resets at 15 min when broker is responsive
# Test: manager does NOT auto-reset at 15 min when broker still down
# Test: manager sets sizing_scalar=0.5 on auto-reset
# Test: manager respects MAX_AUTO_RESETS_PER_DAY=2 (third reset blocked)
# Test: manager backs off when reset followed by immediate re-trigger

# --- Sizing ramp-back ---
# Test: sizing_scalar starts at 0.5 after auto-reset
# Test: sizing_scalar ramps 0.5 → 0.75 → 1.0 over 3 successful cycles
# Test: sizing_scalar resets to 0.5 if kill switch re-triggers during ramp
# Test: sizing_scalar key removed from system_state after reaching 1.0

# --- KillSwitchEscalationManager ---
# Test: escalation sends Discord at 0 min
# Test: escalation sends email at 4 hours (or enhanced Discord if email not configured)
# Test: escalation sends emergency Discord at 24 hours
# Test: escalation does not send duplicate notifications for same tier
# Test: escalation resets tier tracking when kill switch is cleared

# --- reset() signature ---
# Test: reset() works with reason provided (audit trail logged)
# Test: reset() works without reason (backward compatible, warning logged)
# Test: reset event written to kill_switch_events table
```

---

## Section 4: Alembic Migration Framework

```python
# tests/integration/test_alembic_migrations.py (marker: @pytest.mark.integration)

# --- Baseline migration ---
# Test: alembic upgrade head on empty database creates all tables
# Test: alembic upgrade head on existing database is idempotent (no errors)
# Test: alembic current shows correct version after upgrade
# Test: alembic downgrade base drops all tables (dev only)

# --- Advisory lock ---
# Test: concurrent alembic upgrade from two connections — one proceeds, other waits/skips
# Test: advisory lock released even if migration fails (cleanup in finally block)

# --- Fallback ---
# Test: USE_ALEMBIC=false uses old run_migrations() path
# Test: USE_ALEMBIC=true uses alembic upgrade path
# Test: both paths result in identical schema (compare pg_dump output)

# --- Startup integration ---
# Test: run_migrations() only runs once per process (_migrations_done flag)
```

---

## Section 5: Container Monitoring Stack

Primarily infrastructure. Limited unit test surface, mostly integration verification.

```python
# tests/integration/test_monitoring_stack.py (marker: @pytest.mark.integration)

# Test: Prometheus scrapes cAdvisor metrics (query: up{job="cadvisor"} == 1)
# Test: Prometheus scrapes trading-graph metrics (query: up{job="trading-graph"} == 1)
# Test: Grafana Prometheus datasource configured (GET /api/datasources)
# Test: OOM alert rule exists (query Grafana alerting API)
# Test: Memory warning alert rule exists
# Test: Alert list panel exists on dashboard
# Test: Alert history panel exists on dashboard
```

---

## Section 6: Health Check Granularity

```python
# tests/unit/test_health_metrics.py

# --- collect_health_metrics() ---
# Test: computes cycle success rate from graph_checkpoints (10 cycles, 7 success = 0.70)
# Test: returns 0.0 success rate when no checkpoints exist
# Test: computes error count for most recent cycle
# Test: counts strategies created in last 7 days
# Test: counts pending research queue items

# --- Prometheus gauge updates ---
# Test: cycle_success_rate gauge updated with computed value
# Test: cycle_error_count gauge updated with computed value
# Test: strategy_generation_7d gauge updated

# --- Threshold alerting ---
# Test: alert fires when success rate < 70% (verify Grafana rule expression)
# Test: alert fires when error count > 3
# Test: info notification when 0 strategies in 7 days
```

---

## Section 7: Shared Rate Limiter

```python
# tests/integration/test_rate_limiter.py (marker: @pytest.mark.integration)

# --- consume_token PL/pgSQL function ---
# Test: consume_token returns TRUE when tokens available
# Test: consume_token returns FALSE when bucket empty
# Test: consume_token refills tokens based on elapsed time (clock_timestamp)
# Test: consume_token is atomic under concurrent access (3 threads, 75 tokens, verify exactly 75 consumed)
# Test: consume_token uses clock_timestamp not now() (verify in function source)
# Test: bucket initializes with correct values (75 tokens, 1.25/sec refill)

# --- Python integration ---
# Test: fetcher acquires token before API call
# Test: fetcher retries with backoff when token not available (verify async sleep, not blocking)
# Test: fetcher does not hold DB connection during backoff
# Test: fetcher circuit breaker falls back to per-process limiter on DB error
# Test: per-process fallback limiter enforces 75/min independently

# --- Daily quota (unchanged) ---
# Test: daily quota increment is atomic (existing behavior, regression test)
# Test: priority-based throttling still works with new rate limiter
```

---

## Section 8: Langfuse Retention (Config Stub)

```python
# tests/unit/test_langfuse_retention.py

# Test: LANGFUSE_RETENTION_ENABLED defaults to false
# Test: scheduler job exists for langfuse_retention_cleanup
# Test: job logs "disabled" message when LANGFUSE_RETENTION_ENABLED=false
# Test: job logs "would delete" message when LANGFUSE_RETENTION_ENABLED=true
# Test: LANGFUSE_RETENTION_DAYS defaults to 30
```

---

## Section 9: Secrets and Env Var Hardening

```python
# tests/unit/test_env_validation.py

# --- Required vars ---
# Test: validate_environment() passes with all required vars set
# Test: validate_environment() exits with code 1 when TRADER_PG_URL missing
# Test: validate_environment() exits when ALPHA_VANTAGE_API_KEY missing
# Test: validate_environment() exits when ALPACA_API_KEY missing

# --- Typed vars ---
# Test: RISK_MAX_POSITION_PCT="0.05" passes validation
# Test: RISK_MAX_POSITION_PCT="ten" causes exit with clear error
# Test: RISK_MAX_POSITION_PCT="1.5" causes exit (out of range 0.0-1.0)
# Test: AV_DAILY_CALL_LIMIT="25000" passes
# Test: AV_DAILY_CALL_LIMIT="-1" causes exit

# --- Boolean vars ---
# Test: USE_REAL_TRADING="true" passes
# Test: USE_REAL_TRADING="True" passes (case insensitive)
# Test: USE_REAL_TRADING="yes" causes exit (not true/false)

# --- Optional vars ---
# Test: missing GROQ_API_KEY logs warning but does not exit
# Test: missing DISCORD_WEBHOOK_URL logs warning but does not exit

# --- Redaction ---
# Test: error message for API key vars redacts the actual value
# Test: error message for non-secret vars shows the actual value
```

---

## Section 10: SBOM Scanning (Deferred)

No test code — this adds CI steps, not application code. Verification is that the CI pipeline passes with pip-audit and cyclonedx steps added.

---

## Testing Infrastructure Notes

**Fixtures needed:**
- `test_db` — fresh PostgreSQL database for integration tests (create/drop per test session)
- `mock_broker` — mock Alpaca broker that can simulate failures for kill switch recovery tests
- `mock_discord` — mock Discord webhook endpoint for escalation tests
- `env_override` — context manager that sets/unsets env vars for validation tests

**Running tests:**
```bash
# Unit tests only (fast, no DB needed)
pytest tests/unit/ -m "not integration"

# Integration tests (needs Docker Compose up)
pytest tests/integration/ -m integration

# All tests
pytest tests/
```
