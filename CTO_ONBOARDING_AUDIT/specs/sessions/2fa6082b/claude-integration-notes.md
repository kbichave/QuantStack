# Integration Notes — Opus Review Feedback

## Suggestions INTEGRATED

### 1a. TTLCache per-entry TTL gap (CRITICAL)
**Integrating.** The reviewer correctly identified that `shared/cache.py`'s TTLCache stores a single TTL, not per-entry. Two fixes needed: (1) modify TTLCache to store `(value, timestamp, ttl)` tuples, (2) add `shared/cache.py` to files-to-modify for Section 5.

### 1b. ABC over-specification
**Integrating.** Only `name()` should be abstract. All data-fetching methods should be non-abstract with default `return None`. This avoids 40+ `pass` stubs across providers.

### 1d. Circuit breaker on provider registry
**Integrating.** Add a cooldown check: if `consecutive_failures >= 3 AND last_failure_at > now() - 10 minutes`, skip directly to fallback. Uses existing `data_provider_failures` table — minimal code addition.

### 2a. Weekend/holiday calendar-day staleness
**Integrating.** Change price-derived collector thresholds from 2 to 4 calendar days. Document that this covers 3-day weekends. No trading calendar needed.

### 2b. data_metadata for non-OHLCV sources
**Integrating.** Add data_metadata population for all acquisition phases (macro, news, options, insider, etc.) as a prerequisite. Eliminates the expensive fallback MAX() queries.

### 2e. Cache invalidation race condition
**Integrating.** Document as a known limitation. The race window is narrow (seconds) and the outcome (one more stale brief) is strictly better than the current 55-minute window.

### 3a. Pin edgartools version
**Integrating.** Pin to a specific version and add a startup health check.

### 3b. FRED as fallback initially (strangler fig)
**Integrating.** Smart suggestion. Make AV primary for macro initially, FRED as fallback. Swap after 2+ weeks of stable FRED operation. Configuration-driven swap via env var or registry config.

### 3c. Drop pg_partman, manual partitions only
**Integrating.** Simpler approach: manual partition creation via startup hook. Remove pg_partman conditional path.

### 4a. Parallelize 8.1 and 8.4
**Integrating.** Remove the false dependency. Both can run in Week 1.

### 4b. EDGAR MVP milestone
**Integrating.** Define a minimal EDGAR: CIK resolution + Form 4 parsing first, unblocking 8.6 early. XBRL financials as follow-on.

### 5a. All-collectors-stale test
**Integrating.** Add test for synthesis handling when all 22 collectors return `{}`.

### 5d. Staleness check performance benchmark
**Integrating.** Add benchmark test. If overhead exceeds 2s, batch freshness checks.

### 6c. Provider initialization validation
**Integrating.** Providers validate config at init. Missing optional providers excluded from registry with warning log.

### 6d. NotImplementedError vs None convention
**Integrating.** Unimplemented methods raise `NotImplementedError` (registry catches, doesn't count as failure). `None` means "tried, found nothing." Empty DataFrame means "tried, no results."

### 6e. Structured provider observability
**Integrating.** Log provider_name, data_type, symbol, latency_ms, success, fallback_used.

## Suggestions NOT integrated

### 1c. Explain why LISTEN/NOTIFY rejected
**Not integrating as a plan change.** Adding a rationale note is reasonable but doesn't change the plan's substance. The simple invalidation approach is clearly correct for 50 symbols. If the plan document gets a "design decisions" addendum, I'd include this.

### 2c. Advisory lock during migration
**Not integrating.** The plan already requires Docker services stopped. Adding advisory locks on top of that is belt-and-suspenders for a 15-minute weekend maintenance window. The real protection is stopping services.

### 2d. Enumerate all indexes
**Not integrating as a plan change.** This is implementation detail for the migration script. The script will introspect existing indexes and recreate them. No need to enumerate in the plan.

### 3b (alternate). Keep AV primary permanently
**Partially integrating.** FRED will become primary for macro eventually, but the reviewer's strangler fig suggestion (start as fallback) is the right path. See integration above.

### 5b. Concurrent fallback load test
**Not integrating.** EDGAR's 10 req/sec handles 50 sequential calls in 5 seconds. The acquisition pipeline is already sequential per-symbol. No concurrent burst scenario exists in the current architecture.

### 5c. Audit OHLCV query patterns
**Not integrating in plan.** Will be done during implementation as part of the migration script's pre-flight checks. Good practice but not plan-level.

### 6a. Collector count discrepancy
**Not integrating.** The exact count is an implementation detail that will be resolved when touching each file. The staleness table covers all categories.

### 6b. sec_filings upsert conflict
**Not integrating in plan.** Implementation detail: ON CONFLICT (accession_number) DO UPDATE SET fetched_at = NOW(). Standard pattern.
