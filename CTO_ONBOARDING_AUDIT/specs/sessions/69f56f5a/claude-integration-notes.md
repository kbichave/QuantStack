# Integration Notes — Opus Review Feedback

## Integrating

| Item | Why |
|------|-----|
| **Remove EventBus dependency (Section 3)** | Correct — EventBus doesn't exist in codebase. AutoRecoveryManager polls state directly. No blocker needed. |
| **`clock_timestamp()` not `now()` (Section 7)** | Correct — `now()` returns transaction start time. Critical correctness fix for rate limiter. |
| **`MAX_AUTO_RESETS_PER_DAY` cap (Section 3)** | Excellent safety net. Prevents runaway auto-recovery if broker keeps failing. Cap at 2/day. |
| **Hand-written Alembic migrations (Section 4)** | Correct — no SQLAlchemy models exist. Autogenerate won't work. Note this explicitly. |
| **Non-blocking rate limiter (Section 7)** | Correct — caller should manage backoff, not hold DB connection during sleep. |
| **Circuit breaker for rate limiter (Section 7)** | Good safety — fall back to per-process limiter if PL/pgSQL function raises exception. |
| **`reason` parameter optional (Section 3.4)** | Pragmatic — avoids breaking existing callers. Default to empty string, log warning. |
| **Grafana alerting + supervisor Discord as parallel paths (Section 2)** | Critical resilience — don't make Grafana SPOF for alerts. Keep existing supervisor Discord path. |
| **Memory budget table (Section 11)** | Useful for capacity planning — 14 services need explicit memory accounting. |
| **Defer IC trend monitoring (Section 6.3)** | Correct — research-grade computation doesn't belong in operational resilience phase. |
| **Fix Section 8 cross-reference** | Typo fix. |
| **Post-deployment smoke test in start.sh (Section 1)** | Good operational practice. |
| **Alembic fallback flag (Section 4)** | Good safety — `USE_ALEMBIC=true` env flag for first deployment. |
| **Dashboard should show alerts** | User request — Grafana dashboard panels should include active alert status and alert history. |

## NOT Integrating

| Item | Why |
|------|-----|
| **Don't pre-build dashboard JSON (Section 5.4)** | Disagree — deep-implement needs a concrete spec to build against. We'll specify panel requirements (including alert panels per user request) but keep the JSON as implementation detail, not hand-written. The implementer generates it from Grafana API after building interactively. |
| **Alembic baseline via `pg_dump --schema-only` diff** | Over-specified mitigation. The baseline migration uses `IF NOT EXISTS` — it's safe regardless. Schema drift from the dump would be informational but not blocking. |
| **Testing strategy per section** | This belongs in the TDD plan (Step 16), not the implementation plan. The TDD step explicitly creates test specifications for every section. |
