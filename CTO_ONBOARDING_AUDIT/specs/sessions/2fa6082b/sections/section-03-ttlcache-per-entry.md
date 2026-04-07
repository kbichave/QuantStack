# Section 03: Per-Entry TTL Support in TTLCache

## Background

QuantStack's shared `TTLCache` (`src/quantstack/shared/cache.py`) is an in-memory key-value store used by the signal engine cache and the IC output cache. Currently it stores entries as `(value, timestamp)` tuples and checks expiry against a single instance-wide `self._ttl`. Every entry lives for the same duration regardless of context.

The drift detection system (section-04) needs to cache SignalBriefs with shorter TTLs when drift is detected (e.g., 300s for CRITICAL drift instead of the default 3600s). To support this, `TTLCache` must allow callers to override TTL on a per-entry basis while remaining fully backward-compatible for existing consumers that never pass a TTL.

This is a prerequisite for section-04-drift-pre-cache, which depends on this change. No other sections depend on this one.

## Current Implementation

The cache is at `src/quantstack/shared/cache.py`. Key details:

- `__init__` takes `ttl_seconds` (default 1800) and stores it as `self._ttl`
- Internal store is `dict[str, tuple[Any, float]]` mapping key to `(value, timestamp)`
- `set(key, value)` stores `(value, time.monotonic())`
- `get(key)` checks `time.monotonic() - ts > self._ttl` to determine expiry
- `clear_expired()` iterates all entries and removes those past `self._ttl`
- `__contains__` also checks `self._ttl` for membership tests

The signal engine cache (`src/quantstack/signal_engine/cache.py`) wraps `TTLCache`:
- Creates a module-level `TTLCache(ttl_seconds=_ttl)` where `_ttl` comes from `SIGNAL_ENGINE_CACHE_TTL` env var (default 3600)
- `put(symbol, brief)` calls `_cache.set(symbol.upper(), brief)`
- `get(symbol)` calls `_cache.get(symbol.upper())`

## Tests (Write First)

Create `tests/unit/test_ttlcache_per_entry.py`. All tests use `time.monotonic` patching or `time.sleep` for small intervals.

```python
# tests/unit/test_ttlcache_per_entry.py

# Test 1: Default TTL backward compatibility
# TTLCache.set(key, value) with no ttl parameter uses the instance default TTL.
# Create a cache with ttl_seconds=10. Set a key without per-entry TTL.
# Verify get() returns the value before 10s and returns None after 10s.
# This proves existing consumers are unaffected by the change.

# Test 2: Per-entry TTL override (shorter than default)
# TTLCache.set(key, value, ttl=2) on a cache with ttl_seconds=10.
# After 3 seconds, get() should return None (entry's own TTL expired),
# even though the instance default of 10s has not elapsed.

# Test 3: Per-entry TTL override (longer than default)
# TTLCache.set(key, value, ttl=20) on a cache with ttl_seconds=10.
# After 15 seconds, get() should still return the value (entry TTL is 20s),
# even though the instance default of 10s has elapsed.

# Test 4: Mixed entries with different TTLs
# Set key "a" with no override, key "b" with ttl=2, key "c" with ttl=30.
# Advance time past 2s: "b" expired, "a" and "c" still alive.
# Advance time past default: "a" expired, "c" still alive.
# Advance time past 30s: all expired.

# Test 5: __contains__ respects per-entry TTL
# Set key with ttl=2. Before 2s: key in cache is True. After 2s: False.

# Test 6: clear_expired() respects per-entry TTL
# Set entries with different per-entry TTLs.
# Call clear_expired() after some entries should have expired.
# Verify only the actually-expired entries are removed.

# Test 7: Existing signal engine cache still works
# Instantiate TTLCache the way signal_engine/cache.py does (ttl_seconds from env).
# Perform set/get cycles. Verify identical behavior to before.
# This is a regression guard.
```

## Implementation

### File: `src/quantstack/shared/cache.py`

The change is additive. Modify the internal tuple from `(value, timestamp)` to `(value, timestamp, entry_ttl)` where `entry_ttl` is `None` when the caller does not override.

**Changes to `set()`:**

Add an optional `ttl: int | None = None` parameter. Store `(value, time.monotonic(), ttl)` instead of `(value, time.monotonic())`.

```python
def set(self, key: str, value: Any, ttl: int | None = None) -> None:
    """Store *value* under *key* with the current monotonic timestamp.
    
    Args:
        ttl: Optional per-entry TTL in seconds. If None, the instance
             default (self._ttl) is used at read time.
    """
```

**Changes to `get()`:**

Unpack the third element. Use the entry's TTL if set, otherwise fall back to `self._ttl`.

```python
def get(self, key: str) -> Any | None:
    # Unpack (value, timestamp, entry_ttl)
    # effective_ttl = entry_ttl if entry_ttl is not None else self._ttl
    # Check time.monotonic() - ts > effective_ttl
```

**Changes to `__contains__()`:**

Same logic as `get()` — unpack three elements, use entry TTL if present.

**Changes to `clear_expired()`:**

Same pattern — each entry's effective TTL must be checked individually rather than using `self._ttl` for all.

**Changes to `_store` type hint:**

Update from `dict[str, tuple[Any, float]]` to `dict[str, tuple[Any, float, int | None]]`.

### File: `src/quantstack/signal_engine/cache.py`

Add a `ttl` pass-through parameter to the `put()` function so the engine can specify per-entry TTLs for drift-adjusted briefs.

```python
def put(symbol: str, brief: SignalBrief, ttl: int | None = None) -> None:
    """Store brief with optional per-entry TTL override."""
    if _enabled:
        _cache.set(symbol.upper(), brief, ttl=ttl)
```

This is the only change to this file. Section-04 will call `cache.put(symbol, brief, ttl=300)` for CRITICAL drift.

## Blast Radius

`TTLCache` is used by:
1. **Signal engine cache** (`src/quantstack/signal_engine/cache.py`) — modified here to pass TTL through
2. **IC output cache** — uses default `set(key, value)` with no TTL override, so behavior is identical

The change is backward-compatible: all existing call sites pass no `ttl` argument, which defaults to `None`, which falls back to `self._ttl`. The regression test (Test 7) explicitly verifies this.

## Dependencies

- **Depends on:** Nothing. This section has no prerequisites.
- **Blocks:** section-04-drift-pre-cache, which needs per-entry TTL to cache drift-adjusted briefs with shorter lifetimes.
