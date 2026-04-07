# Opus Review

**Model:** claude-opus-4
**Generated:** 2026-04-06

---

## Overall Assessment

The plan is well-structured, technically grounded, and demonstrates clear understanding of the codebase. The stakeholder decisions (broker-only auto-reset, PL/pgSQL rate limiter, Loki over Elasticsearch) are sound. However, there are several issues ranging from a critical dependency phantom to missing operational safety guardrails.

## Actionable Items

| Priority | Section | Action |
|----------|---------|--------|
| **Critical** | 3, 11 | Remove phantom EventBus dependency — Section 3 can be implemented now |
| **Critical** | 7 | Use `clock_timestamp()` not `now()` in PL/pgSQL function |
| **High** | 3 | Add `MAX_AUTO_RESETS_PER_DAY` safety cap (e.g., 2) |
| **High** | 4 | Note that `--autogenerate` requires SQLAlchemy models; plan for hand-written migrations |
| **High** | 7 | Make rate limiter non-blocking; don't hold DB connections during sleep retries |
| **High** | 7 | Add circuit breaker fallback to per-process limiter on DB errors |
| **High** | All | Add testing requirements per section |
| **Medium** | 3.4 | Make `reason` parameter optional to avoid breaking existing callers |
| **Medium** | 4 | Add Alembic fallback flag (`USE_ALEMBIC`) for first deployment |
| **Medium** | 6.3 | Defer IC trend monitoring to a later phase |
| **Medium** | 11 | Add total memory budget table for 14-service Docker Compose |
| **Medium** | 2+5 | Note that Grafana is SPOF for alerting; keep supervisor Discord alerts as parallel path |
| **Low** | 5.4 | Don't pre-build dashboard JSON; provision datasources only, build dashboard interactively |
| **Low** | 8 | Fix cross-reference: "Section 10" should say "Section 9" (env validation) |
| **Low** | 4 | Specify concrete rollback procedure for failed migrations |
| **Low** | 1 | Add post-deployment smoke test to `start.sh` |

## Detailed Findings

### 1. Completeness
- Spec 3.2 AC "Logs survive host container restart" — plan mentions Loki volume but should explicitly note `loki-data` must NOT be in any cleanup script
- Spec 3.7 AC "Alerts fire when metrics breach thresholds" — plan needs to clarify that supervisor gauges appear on supervisor's own `/metrics` endpoint (scraped by Prometheus)

### 2. Technical Correctness

**Section 3 — Phantom EventBus Dependency (CRITICAL):** Zero EventBus matches in source code. AutoRecoveryManager as designed needs no pub/sub — it polls state and calls methods. Remove this blocker.

**Section 3.4 — Breaking `reset()` signature:** Adding required `reason` breaks existing callers. Make it optional with default `""`.

**Section 4 — Autogenerate requires SQLAlchemy models:** Current codebase uses raw SQL. Without `MetaData` object, autogenerate produces empty migrations. Note this limitation; use hand-written migrations.

**Section 7 — `now()` vs `clock_timestamp()`:** `now()` returns transaction start time. Use `clock_timestamp()` for actual wall-clock time in rate limiter.

**Section 7 — Retry loop holds DB connections:** Thread holds connection for up to 60s while sleeping. Return FALSE and let caller manage backoff with `asyncio.sleep`.

### 3. Missing Risks
- Grafana as single point of alerting failure — keep supervisor Discord alerts as parallel path
- Docker Compose goes from 9 to 14 services, approaching 10-11GB total memory — add memory budget table
- Alembic baseline fidelity — use `pg_dump --schema-only` to verify baseline migration

### 4. Dependency Ordering
- Section 3 should NOT depend on Phase 1 EventBus (phantom)
- Section 8 references "Section 10" for env validation but means Section 9

### 5. Gaps
- No rollback plan for failed Alembic migrations
- No post-deployment smoke test in start.sh
- Section 6.3 IC trend monitoring under-specified (lookback window, re-enable criteria)
- No testing strategy specified for new code

### 6. Over-engineering
- Section 6.3 IC monitoring is research-grade, not operational resilience — defer
- Pre-building Grafana dashboard JSON (Section 5.4) is premature — provision datasources, build dashboard interactively

### 7. Operational Safety
- Section 4: Add Alembic fallback flag (`USE_ALEMBIC`) for first deployment
- Section 7: Add circuit breaker — if `consume_token` raises exception (not returns FALSE), fall back to per-process limiter
- Section 3: Add `MAX_AUTO_RESETS_PER_DAY` limit (e.g., 2) to prevent runaway auto-recovery
