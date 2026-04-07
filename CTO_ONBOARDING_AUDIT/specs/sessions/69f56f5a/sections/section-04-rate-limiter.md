# Section 4: Shared Rate Limiter

## Problem

`src/quantstack/data/fetcher.py` uses a per-process in-memory rate limiter: an `_call_count` integer and `_minute_start` timestamp that reset every 60 seconds, capping at 75 requests/min. Three Docker containers (trading-graph, research-graph, supervisor-graph) each run their own fetcher instance, so the effective aggregate rate to Alpha Vantage is up to 225 req/min against a real limit of 75/min. The daily quota tracking is already atomic (PostgreSQL `INSERT ... ON CONFLICT DO UPDATE`), but the per-minute rate limiting is not coordinated.

## Solution

A PostgreSQL-backed token bucket implemented as a PL/pgSQL function. A single `consume_token()` call atomically checks, refills, and consumes tokens. No Redis, no distributed coordination -- just one row lock per API call. Python integration is non-blocking: if no token is available, the caller releases the DB connection and retries with async sleep.

## Dependency

This section depends on **section-02-alembic-migrations**. The `rate_limit_buckets` table and `consume_token()` function are created via an Alembic migration. Alembic must be set up and the baseline migration applied before this migration can run.

---

## Tests First

All tests go in `tests/integration/test_rate_limiter.py` with marker `@pytest.mark.integration` (requires a running PostgreSQL instance).

### consume_token PL/pgSQL function

```python
# tests/integration/test_rate_limiter.py

import pytest

@pytest.mark.integration
class TestConsumeToken:
    """Tests for the PL/pgSQL consume_token() function."""

    def test_returns_true_when_tokens_available(self, test_db):
        """consume_token('alpha_vantage') returns TRUE when bucket has tokens."""

    def test_returns_false_when_bucket_empty(self, test_db):
        """After consuming all 75 tokens, the next call returns FALSE."""

    def test_refills_tokens_based_on_elapsed_time(self, test_db):
        """After draining the bucket and waiting ~1 second, ~1.25 tokens refill."""

    def test_atomic_under_concurrent_access(self, test_db):
        """3 threads race to consume from a 75-token bucket.
        Exactly 75 succeed, the rest fail. No over-consumption."""

    def test_uses_clock_timestamp_not_now(self, test_db):
        """Verify the function source contains clock_timestamp(), not now().
        now() returns transaction-start time and would freeze refill calculation
        inside a larger transaction."""

    def test_bucket_initializes_with_correct_values(self, test_db):
        """Seed row has tokens=75, max_tokens=75, refill_rate=1.25."""
```

### Python integration

```python
@pytest.mark.integration
class TestFetcherRateLimiter:
    """Tests for the Python-side rate limiter integration in fetcher.py."""

    def test_acquires_token_before_api_call(self, test_db, mock_av_server):
        """Fetcher calls consume_token before issuing HTTP request."""

    def test_retries_with_async_sleep_when_token_unavailable(self, test_db):
        """When consume_token returns FALSE, fetcher sleeps (async, not blocking)
        and retries. Does not hold a DB connection during the sleep."""

    def test_does_not_hold_db_connection_during_backoff(self, test_db):
        """Verify the DB connection is released before the sleep/retry loop."""

    def test_circuit_breaker_falls_back_on_db_error(self, test_db):
        """When consume_token raises a database exception (not FALSE),
        fetcher falls back to the per-process in-memory limiter."""

    def test_per_process_fallback_enforces_75_per_min(self):
        """The fallback in-memory limiter still caps at 75/min independently."""
```

### Daily quota (regression)

```python
@pytest.mark.integration
class TestDailyQuotaRegression:
    """Ensure existing daily quota behavior is unaffected."""

    def test_daily_quota_increment_is_atomic(self, test_db):
        """Concurrent daily count increments produce the correct total."""

    def test_priority_throttling_works_with_new_rate_limiter(self, test_db):
        """Priority-based throttling (critical vs. normal requests) still
        functions correctly when the new per-minute limiter is active."""
```

---

## Implementation Details

### 4.1 Alembic Migration: Table and Function

Create a new Alembic migration file in `alembic/versions/`. The migration creates both the table and the PL/pgSQL function.

**File:** `alembic/versions/<timestamp>_add_rate_limit_buckets.py`

**upgrade():**

Table `rate_limit_buckets`:

| Column | Type | Constraints |
|--------|------|-------------|
| `bucket_key` | `TEXT` | `PRIMARY KEY` |
| `tokens` | `NUMERIC` | `NOT NULL` |
| `max_tokens` | `NUMERIC` | `NOT NULL` |
| `refill_rate` | `NUMERIC` | `NOT NULL` |
| `last_refill` | `TIMESTAMPTZ` | `NOT NULL DEFAULT now()` |

Seed row:

```sql
INSERT INTO rate_limit_buckets (bucket_key, tokens, max_tokens, refill_rate, last_refill)
VALUES ('alpha_vantage', 75, 75, 1.25, now());
```

The values mean: 75-token capacity, refill at 1.25 tokens/second (= 75 tokens/minute).

Function `consume_token(p_key TEXT, p_cost NUMERIC DEFAULT 1) RETURNS BOOLEAN`:

The function must:

1. `SELECT ... FROM rate_limit_buckets WHERE bucket_key = p_key FOR UPDATE` -- acquires a row-level lock so concurrent callers serialize on this single row.
2. Compute elapsed seconds since `last_refill` using **`clock_timestamp()`** (not `now()`). This is critical: `now()` returns the transaction start time, which would freeze the refill calculation if the call is wrapped in a larger transaction. `clock_timestamp()` returns wall-clock time at the moment of execution.
3. Refill tokens: `new_tokens = LEAST(max_tokens, tokens + elapsed * refill_rate)`.
4. If `new_tokens >= p_cost`: subtract cost, update `tokens` and `last_refill` to `clock_timestamp()`, return `TRUE`.
5. If `new_tokens < p_cost`: still update `tokens` to `new_tokens` and `last_refill` to `clock_timestamp()` (prevents refill drift on repeated failed checks), return `FALSE`.

**downgrade():**

```sql
DROP FUNCTION IF EXISTS consume_token(TEXT, NUMERIC);
DROP TABLE IF EXISTS rate_limit_buckets;
```

### 4.2 Python Integration in fetcher.py

**File:** `src/quantstack/data/fetcher.py`

Replace the current `_wait_for_rate_limit()` method. The new implementation has two paths: a primary DB-backed path and a circuit-breaker fallback.

**Primary path (DB-backed):**

The method should:

1. Open a short-lived DB connection (via `db_conn()` context manager).
2. Execute `SELECT consume_token('alpha_vantage')`.
3. Close the connection immediately (exit the context manager).
4. If result is `TRUE`: return, proceed with API call.
5. If result is `FALSE`: log at DEBUG level ("Rate limit: waiting for token (bucket: alpha_vantage)"), sleep 1 second, retry. Maximum 60 retries (60 seconds total). After 60 retries, log WARNING and skip the call.

The key design point: the DB connection is **not held** during the sleep/retry loop. Each retry opens a new short-lived connection, calls `consume_token`, and closes it. This prevents connection pool exhaustion during backoff.

**Circuit breaker fallback:**

If `consume_token` raises a database exception (connection error, timeout -- not a `FALSE` return), fall back to the existing per-process in-memory rate limiter for that call. Log at WARNING level. This prevents a DB outage from blocking all API calls. The per-process limiter is less accurate (each container enforces 75/min independently) but keeps the system functional.

The existing `_call_count` and `_minute_start` fields remain on the class but become the fallback limiter, not the primary path. Rename them for clarity (e.g., `_fallback_call_count`, `_fallback_minute_start`) and extract the current logic into a `_wait_for_rate_limit_fallback()` method.

**Backpressure signal:**

When `consume_token` returns `FALSE`, log at DEBUG (this is normal contention, not an error). If all 60 retries exhaust without a token, log at WARNING and skip the call -- same behavior as the current daily-quota-exceeded path.

### 4.3 Integration Points

The `_call_count += 1` lines scattered across every fetch method in `fetcher.py` (there are 15+ call sites) should be reviewed. Under the new scheme, token consumption happens in `_wait_for_rate_limit()` before the call, not after. The post-call `_call_count += 1` increments become unnecessary for the primary path but should remain for the fallback path (the in-memory limiter still needs them). Guard these increments behind a flag that indicates which limiter path was used for the current call.

The daily quota tracking (`_increment_daily_count()`, `_get_daily_count()`) is **unchanged**. It already uses PostgreSQL atomically and operates on a separate concern (daily budget vs. per-minute rate).

---

## Key Files

| File | Action |
|------|--------|
| `alembic/versions/<timestamp>_add_rate_limit_buckets.py` | Create -- migration for table + function |
| `src/quantstack/data/fetcher.py` | Modify -- replace `_wait_for_rate_limit()`, restructure fallback |
| `tests/integration/test_rate_limiter.py` | Create -- all tests listed above |

## Risks and Mitigations

**Risk: DB connection overhead per API call.** Each rate-limited call now opens and closes a DB connection. Mitigation: the existing `db_conn()` uses a connection pool. The `consume_token` call is a single-statement round trip (~1ms on localhost). This adds negligible latency compared to the Alpha Vantage API call itself (~200-500ms).

**Risk: DB down blocks all API calls.** If PostgreSQL is unreachable, `consume_token` raises an exception. The circuit breaker catches this and falls back to the per-process limiter. This is the correct degraded mode: if the DB is down, the system has larger problems, but at least data fetching continues at reduced coordination.

**Risk: Row lock contention under high concurrency.** The `FOR UPDATE` row lock means concurrent callers serialize. With 3 containers making at most 75 calls/min total (1.25/sec), contention is negligible. The lock hold time is microseconds (single UPDATE statement). This would only become a problem at hundreds of concurrent consumers, which is not a realistic scenario for this system.

**Risk: clock_timestamp() vs now() confusion in future migrations.** Document in a code comment within the PL/pgSQL function body why `clock_timestamp()` is used. If someone changes it to `now()`, the refill calculation breaks silently inside transactions.
