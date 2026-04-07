# Opus Review

**Model:** claude-opus-4
**Generated:** 2026-04-06T00:00:00Z

---

## Review: Phase 8 — Data Pipeline Hardening Implementation Plan

### 1. Architectural Concerns

**1a. Section 5 (Drift Pre-Cache): TTLCache does not support per-entry TTL — plan assumes it does.**

This is the most concrete technical gap. The plan (Section 5) proposes `cache.put(symbol, brief, ttl=300)` for CRITICAL drift. But the underlying `TTLCache` in `src/quantstack/shared/cache.py` stores entries as `(value, timestamp)` tuples and checks expiry against a single `self._ttl` set at construction time. There is no per-entry TTL support. The `set()` method signature is `set(self, key: str, value: Any) -> None` — no TTL parameter.

The plan acknowledges this in Section 5 ("Extend it to accept an optional `ttl_override` parameter") but files-to-modify only lists `signal_engine/cache.py`. The actual change must happen in `shared/cache.py` (the `TTLCache` class itself), which is shared infrastructure used by other subsystems (IC output cache per the docstring). Modifying a shared cache class to support per-entry TTL requires storing TTL per entry and changing the `get()` expiry check. This is doable but the blast radius is wider than stated — any other consumer of `TTLCache` could be affected.

**Recommendation:** Either (a) modify `TTLCache` to store `(value, timestamp, ttl)` tuples with a fallback to `self._ttl` when per-entry TTL is `None`, or (b) use a separate short-lived cache instance for drift-degraded entries. Option (a) is cleaner but add `shared/cache.py` to the files-to-modify list and add a test verifying existing consumers aren't broken.

**1b. Section 4 (Provider Abstraction): The ABC is over-specified for day-one needs.**

The `DataProvider` ABC defines 8 abstract methods. But FRED only implements `fetch_macro_indicator`, and EDGAR only implements `fetch_insider_transactions`, `fetch_institutional_holdings`, `fetch_fundamentals`, and `fetch_sec_filings`. The AV adapter would need to implement all 8. With `@abstractmethod` on all of them, every provider must implement every method or use `pass`/`raise NotImplementedError`.

The plan says "Default implementations return `None`" — but if these are `@abstractmethod`, there ARE no default implementations. The ABC in `base.py` needs non-abstract default methods that return `None`, with only `name()` as abstract. Clarify that only `name()` is abstract; the rest are default-None.

**1c. Research findings suggest Dual-TTL (soft/hard) and LISTEN/NOTIFY — plan ignores both.**

The research (Part 2) recommends a soft-TTL/hard-TTL pattern and PostgreSQL LISTEN/NOTIFY for event-driven invalidation. The plan opts for simple per-symbol invalidation after refresh. This is the right call for now — LISTEN/NOTIFY adds a persistent connection requirement and the current system doesn't need it. But the plan should explicitly state WHY it rejected these options. A one-liner saying "LISTEN/NOTIFY adds connection complexity disproportionate to 50-symbol universe; revisit at 500+ symbols" would suffice.

**1d. Section 4: No circuit breaker on the provider registry.**

The registry tries primary, then fallback, and tracks consecutive failures. But there's no circuit breaker — if AV is slow (5s timeouts) rather than down, every single call pays the timeout penalty before falling through to FRED/EDGAR. At 50 symbols x 14 phases, that's potentially 50 * 5s = 250 seconds of wasted time per refresh cycle.

**Recommendation:** After N consecutive failures (the plan already tracks this), skip the failing provider entirely for a cooldown period (e.g., 10 minutes). The `data_provider_failures` table already has the data; just add a check: if `consecutive_failures >= 3 AND last_failure_at > now() - interval '10 minutes'`, skip directly to fallback.

### 2. Missing Edge Cases

**2a. Section 3 (Staleness): Weekend/holiday data age calculations.**

The staleness thresholds for price-derived collectors are "2 days." But on Monday morning before market open, the most recent daily data is from Friday — 3 calendar days ago. On a 3-day weekend, it's 4 calendar days. The `check_freshness()` helper computes `(now() - meta.last_timestamp).days` which is calendar days, not trading days.

The spec says "2 trading days" but the implementation uses calendar days. Either the threshold needs to be bumped to 4 calendar days (to cover 3-day weekends), or the helper needs a trading-day-aware calculation. The simpler fix: use `max_days=4` for price-derived collectors and document that it covers weekends. Don't build a trading calendar just for this.

**2b. Section 3: `data_metadata` might not exist for all collector data sources.**

The plan says the helper checks `data_metadata.last_timestamp`. But `data_metadata` is keyed on `(symbol, timeframe)` — this works for OHLCV-dependent collectors. What about collectors that read from `macro_indicators`, `news_sentiment`, `options_chains`, or `insider_trades`? These tables may not have corresponding `data_metadata` rows.

The plan mentions "or direct MAX(date/timestamp) query for tables without metadata rows" in the helper docstring — good. But this fallback query runs on every collector invocation for every symbol. That's 22 collectors x 50 symbols = 1,100 queries per signal engine cycle. If even half hit the fallback path, that's 550 MAX() queries, which could be slow on unindexed columns.

**Recommendation:** Populate `data_metadata` for ALL data sources during acquisition (not just OHLCV). This is a prerequisite that should be called out explicitly. Alternatively, maintain a lightweight in-memory freshness cache updated during refresh cycles.

**2c. Section 7 (OHLCV Partitioning): Concurrent writes during migration.**

The plan says "Docker services stopped" during the weekend migration. But what about the data that arrives between the last write and service shutdown? More critically: what if someone restarts services before the migration completes? The migration script should acquire an advisory lock and verify no active connections before proceeding.

**2d. Section 7: Index creation timing.**

Step 7 says "Create indexes: Recreate any indexes on the new partitioned table." But the primary key is already defined in the partition DDL. Are there additional indexes? The plan doesn't enumerate them. If there are non-PK indexes (e.g., on `(symbol, timeframe)` alone for metadata queries), they need to be listed. Also, `CREATE INDEX CONCURRENTLY` cannot run inside a transaction — the migration script needs to handle this.

**2e. Section 2 (Cache Invalidation): Race condition between refresh and signal engine.**

If the signal engine starts a run for AAPL, the refresh cycle completes and invalidates AAPL's cache, and then the signal engine's run completes and calls `_cache_put(symbol, brief)` — the stale brief gets re-cached. This is a narrow window but real under concurrent operation. The plan doesn't address it.

**Mitigation:** Either (a) accept this as a one-TTL-cycle staleness window (it's better than the current 55-minute window), or (b) add a generation counter to the cache that's bumped on invalidation and checked before put. Option (a) is fine — just document the known limitation.

### 3. Dependency Risks

**3a. `edgartools` library stability.**

The research says 3,680+ commits and active maintenance. But `edgartools` is a volunteer-maintained open-source library that wraps SEC's free API. The SEC periodically changes their endpoint structure. If `edgartools` breaks, insider data and fundamentals fallback both go down simultaneously.

**Recommendation:** Pin `edgartools` to a specific version. Add a health check that verifies a known CIK resolves correctly on startup. Have a plan for what happens if the library breaks.

**3b. FRED API key as a hard dependency for macro data.**

The plan makes FRED the primary source for macro indicators, displacing AV. If the FRED API key expires, is revoked, or the free tier changes, macro data stops flowing with no automatic fallback to AV (since AV is now the fallback, and the registry tries primary first).

**Recommendation:** Make AV the primary for macro for the initial rollout, with FRED as fallback. Once FRED proves stable over 2+ weeks of production operation, swap the routing. This follows the strangler fig pattern.

**3c. pg_partman availability is a bigger deal than stated.**

Without pg_partman, you need a manual partition creation mechanism. The plan proposes "a utility function in `db.py`" — but who calls it?

**Recommendation:** Just build the manual approach from day one. Four lines of SQL to create next month's partition in a startup hook is simpler than conditionally depending on an extension. Remove the pg_partman path entirely to reduce plan complexity.

### 4. Ordering/Sequencing Issues

**4a. Item 8.4 depends on 8.1 — but why?**

The plan (Section 9) says "8.4 after 8.1 cache changes." But 8.4 modifies `engine.py` and `cache.py` while 8.1 modifies `scheduled_refresh.py`. These touch different files with no code-level dependency.

**Recommendation:** Start 8.4 in parallel with 8.1. They don't conflict. This shortens the critical path.

**4b. Item 8.6 depends on 8.3's EDGAR provider — correct, but the dependency is load-bearing.**

If the EDGAR provider takes longer than expected, 8.6 is completely blocked.

**Recommendation:** Define an "EDGAR MVP" milestone: ticker-to-CIK resolution + Form 4 parsing (the simplest EDGAR data). Ship that first to unblock 8.6, then add XBRL financials and 13F parsing as follow-ons.

### 5. Testing Gaps

**5a. No test for the "all collectors stale" scenario.**

What happens when every collector returns `{}` due to staleness? The synthesis engine "redistributes weight" — but redistributing weight across zero contributors is a division-by-zero or an empty brief.

**5b. No test for provider registry fallback under concurrent load.**

The validation plan tests AV failure → FRED/EDGAR fallback. But it doesn't test what happens when multiple symbols trigger fallback simultaneously. If the EDGAR rate limiter (10 req/sec) is shared, 50 symbols falling back simultaneously could exceed it.

**5c. No test for OHLCV partition pruning with actual query patterns.**

Before migrating, audit every query that hits the `ohlcv` table and verify each includes a timestamp filter. Queries without one would scan ALL partitions — potentially slower than unpartitioned.

**5d. No load/stress testing for signal engine with staleness checks.**

Adding `check_freshness()` to 22 collectors means 22 additional DB queries per symbol per run. At 50 symbols, that's 1,100 extra queries. Will this push the 2-6s wall-clock to 10s+?

**Recommendation:** Add a benchmark test. If overhead exceeds 2s, batch freshness checks into a single query at the engine level.

### 6. Specific Improvements

**6a. Collector count discrepancy.**

The research lists 24 collectors, the plan says 22. Enumerate exactly which collectors get staleness checks and which are excluded.

**6b. Section 6: Missing upsert conflict handling for `sec_filings` table.**

The `sec_filings` table uses `accession_number` as PK. If re-fetched, what columns get updated? Define the conflict resolution explicitly.

**6c. Section 4: Missing error handling for provider initialization.**

If `FRED_API_KEY` is not set, the FRED provider should fail at startup with a clear error. Optional providers that are misconfigured should be excluded from the registry with a warning.

**6d. Section 4: The `None` vs empty DataFrame convention needs enforcement.**

Use `NotImplementedError` for unimplemented methods (caught by registry), and `None` for "tried and found nothing." The registry can then distinguish "not supported" from "failed."

**6e. Add structured observability for the provider registry from day one.**

At minimum log: provider name, data type, symbol, latency, success/failure, and which provider in the chain served the request.

### Summary

The plan is well-structured and addresses real problems. The two highest-priority fixes are:

1. **The TTLCache per-entry TTL gap** (Section 5) — this will fail at implementation time if not addressed.
2. **The weekend calendar-day staleness bug** (Section 3) — this will cause false staleness rejections every Monday morning for price-derived collectors.

The provider abstraction (Section 4) is the riskiest section due to scope. Consider cutting scope to FRED-only for Week 1 and EDGAR as a follow-on.
