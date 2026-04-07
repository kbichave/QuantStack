# Section 04: Drift Detection Pre-Cache

## Background

In the QuantStack signal engine, `src/quantstack/signal_engine/engine.py` orchestrates 22 collectors, synthesizes their outputs into a `SignalBrief`, and caches the result. A drift detector (`DriftDetector().check_drift_from_brief()`) runs after synthesis to check whether the current feature distribution has shifted from the historical baseline. The drift report includes a severity level (`NONE`, `WARNING`, `CRITICAL`) and a PSI (Population Stability Index) score.

**The bug:** Drift detection currently runs after synthesis but the cache write (`_cache_put(symbol, brief)`) happens unconditionally at the end of `engine.py`'s `run()` method (line 179). A brief with CRITICAL drift is cached at full confidence with the default 1-hour TTL. Any downstream consumer hitting the cache within that hour gets a high-confidence signal that the system itself has flagged as unreliable. The only mitigation today is `brief.drift_warning = True`, which consumers may or may not check.

**The fix:** Reorder the code so drift detection runs before the cache write, and use drift severity to (1) penalize confidence and (2) shorten cache TTL. This requires the TTLCache to support per-entry TTL overrides, which is delivered by section-03 (dependency).

## Dependencies

- **section-03-ttlcache-per-entry** must be completed first. That section modifies `TTLCache.set()` to accept an optional `ttl` parameter and stores `(value, timestamp, entry_ttl)` tuples. Without it, there is no way to cache a drifted brief with a shorter TTL.

## Tests (Write Before Implementing)

File: `tests/unit/test_drift_pre_cache.py`

### TTLCache per-entry TTL integration tests

These verify the section-03 TTLCache changes work correctly in the drift context:

```python
def test_ttlcache_set_without_ttl_uses_default():
    """TTLCache.set(key, value) with no ttl parameter uses default TTL (backward compatible)."""

def test_ttlcache_set_with_custom_ttl():
    """TTLCache.set(key, value, ttl=300) uses 300s TTL for that entry only."""

def test_ttlcache_get_respects_per_entry_ttl():
    """TTLCache.get() returns None for entry past its per-entry TTL even if default TTL hasn't expired."""

def test_ttlcache_get_returns_value_within_per_entry_ttl():
    """TTLCache.get() returns value for entry within its per-entry TTL."""

def test_existing_ttlcache_consumers_unaffected():
    """IC output cache and other existing consumers still work correctly after per-entry TTL change."""
```

### Drift-to-cache behavior tests

```python
def test_drift_none_caches_with_default_ttl_no_penalty():
    """NONE drift -> brief cached with default TTL (3600s), no confidence penalty applied."""

def test_drift_warning_halves_ttl_and_penalizes_confidence():
    """WARNING drift -> brief cached with half TTL (1800s), confidence reduced by 0.10."""

def test_drift_critical_short_ttl_and_heavy_penalty():
    """CRITICAL drift -> brief cached with 300s TTL, confidence reduced by 0.30."""

def test_drift_critical_inserts_system_event():
    """CRITICAL drift -> DRIFT_CRITICAL event inserted into system_events table."""

def test_drift_warning_no_system_event():
    """WARNING drift -> no event inserted into system_events (log only)."""

def test_drift_check_runs_before_cache_put():
    """Verify via mock ordering that drift check executes BEFORE cache.put()."""

def test_confidence_penalty_floors_at_zero():
    """Confidence penalty does not push overall_confidence below 0.0."""
```

## Implementation Details

### Files to modify

1. **`src/quantstack/signal_engine/cache.py`** — Pass `ttl` parameter through to underlying `TTLCache`
2. **`src/quantstack/signal_engine/engine.py`** — Reorder drift detection before cache write; apply confidence penalty and TTL adjustment

### 1. Signal engine cache passthrough (`src/quantstack/signal_engine/cache.py`)

The `put()` function currently delegates to `_cache.set(symbol.upper(), brief)`. It needs an optional `ttl` parameter that passes through to the underlying `TTLCache.set()`:

```python
def put(symbol: str, brief: SignalBrief, ttl: int | None = None) -> None:
    """Store brief with optional per-entry TTL override.
    
    When ttl is None, the default SIGNAL_ENGINE_CACHE_TTL is used.
    When ttl is provided, that entry expires after ttl seconds regardless
    of the default.
    """
    if _enabled:
        _cache.set(symbol.upper(), brief, ttl=ttl)
```

This is a backward-compatible change. All existing callers that pass no `ttl` get the same behavior as before.

### 2. Engine code reordering (`src/quantstack/signal_engine/engine.py`)

The current flow in `run()` (lines ~120-180) is:

1. Run collectors
2. Build brief (synthesis)
3. Drift detection (best-effort, sets `brief.drift_warning`)
4. Log results
5. Cache the brief (`_cache_put(symbol, brief)`)
6. Return

The modified flow:

1. Run collectors (unchanged)
2. Build brief (unchanged)
3. Drift detection (unchanged try/except, still best-effort)
4. **Apply confidence penalty based on drift severity**
5. **Determine cache TTL based on drift severity**
6. Log results (unchanged)
7. **Cache the brief with determined TTL** (`_cache_put(symbol, brief, ttl=cache_ttl)`)
8. Return

### Drift severity mapping

Define this mapping as a module-level constant in `engine.py`:

| Drift Severity | Confidence Penalty | Cache TTL | Action |
|---------------|-------------------|-----------|--------|
| `NONE` (or drift check failed) | 0 | `None` (use default 3600s) | No action |
| `WARNING` | -0.10 | 1800s (half default) | Log warning only |
| `CRITICAL` | -0.30 | 300s (5 minutes) | Insert `DRIFT_CRITICAL` into `system_events` |

The confidence penalty is applied as:

```python
brief.overall_confidence = max(0.0, brief.overall_confidence - penalty)
```

The `max(0.0, ...)` guard prevents negative confidence values.

### DRIFT_CRITICAL system event

When drift severity is CRITICAL, insert a row into `system_events` so the supervisor graph can detect and act on it. This is a separate DB write from the existing `research_queue` insert (which queues an ML arch search). The system event is for operational alerting; the research queue entry is for automated remediation.

```sql
INSERT INTO system_events (event_type, symbol, severity, details, created_at)
VALUES ('DRIFT_CRITICAL', :symbol, 'critical', :drift_report_json, NOW())
```

The supervisor graph already monitors `system_events` for actionable items, so no additional wiring is needed.

### Handling drift check failures

The drift check is wrapped in a try/except and labeled "best-effort, never blocks brief delivery." This behavior is preserved. If the drift check raises an exception, the brief is cached with default TTL and no confidence penalty — identical to current behavior. The only change is that successful drift checks now influence caching behavior.

### Interaction with existing drift handling

The current code already does two things on CRITICAL drift:
- Sets `brief.drift_warning = True`
- Inserts an `ml_arch_search` task into `research_queue`

Both of these remain. This section adds:
- Confidence penalty on the brief
- Shorter cache TTL
- `DRIFT_CRITICAL` event in `system_events`

These are additive — nothing is removed or replaced.

### Pseudocode for the modified `run()` method

The key structural change in `engine.py`'s `run()` method. After the brief is built and drift is detected, but before logging and caching:

```python
# After drift detection try/except block:
cache_ttl: int | None = None  # None = use default

if drift_report is not None:
    if drift_report.severity == "WARNING":
        brief.overall_confidence = max(0.0, brief.overall_confidence - 0.10)
        cache_ttl = 1800
    elif drift_report.severity == "CRITICAL":
        brief.overall_confidence = max(0.0, brief.overall_confidence - 0.30)
        cache_ttl = 300
        # Insert DRIFT_CRITICAL system event
        try:
            with db_conn() as conn:
                conn.execute(
                    """INSERT INTO system_events 
                       (event_type, symbol, severity, details, created_at)
                       VALUES ('DRIFT_CRITICAL', %s, 'critical', %s, NOW())""",
                    [symbol, json.dumps({...drift details...})],
                )
        except Exception:
            logger.debug("system_events insert failed (non-critical)")

# ... logging ...

_cache_put(symbol, brief, ttl=cache_ttl)
return brief
```

Note that `drift_report` must be initialized to `None` before the drift detection try/except so it is available in scope for the subsequent conditional. If the drift check fails entirely, `drift_report` remains `None` and no penalty or TTL adjustment is applied.

## Verification

After implementation, verify:

1. **Ordering:** Mock `DriftDetector` to return CRITICAL, mock `cache.put`, assert `put` is called with `ttl=300`
2. **Confidence floor:** Set `brief.overall_confidence = 0.15`, apply CRITICAL penalty (-0.30), verify result is `0.0` not `-0.15`
3. **System event:** Mock `db_conn`, verify `DRIFT_CRITICAL` insert SQL is executed on CRITICAL severity
4. **No regression:** Run existing signal engine tests to confirm the reordering does not break the happy path
5. **Backward compatibility:** Verify that `cache.put(symbol, brief)` (no ttl) still works identically to before
