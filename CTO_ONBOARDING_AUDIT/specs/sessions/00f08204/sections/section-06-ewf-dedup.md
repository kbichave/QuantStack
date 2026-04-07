# Section 06: EWF Deduplication via Module-Level Cache

**Plan Reference:** Item 5.5
**Dependencies:** None
**Blocks:** None

---

## Problem

5 agents independently call `get_ewf_analysis` for the same symbols in the same cycle. 3-5x redundant DB queries per symbol.

## Solution

Module-level cache in `ewf_tools.py`. Populate once at cycle start, clear at cycle end. Zero API change — tool signatures unchanged.

---

## Tests (Write First)

Create `tests/unit/test_ewf_cache.py`:

```python
# --- Cache population ---
# Test: populate_ewf_cache stores results for all symbol:timeframe combos
# Test: populate_ewf_cache handles empty symbol list gracefully
# Test: clear_ewf_cache empties the module-level cache

# --- Cache lookup ---
# Test: get_ewf_analysis returns cached result when cache hit
# Test: get_ewf_analysis queries DB when cache miss
# Test: get_ewf_blue_box_setups returns cached result when cache hit
# Test: cache key differentiates timeframes (SPY:4h != SPY:daily)

# --- Cycle lifecycle ---
# Test: cache is empty before populate_ewf_cache called
# Test: cache is empty after clear_ewf_cache called
```

---

## Implementation

### 1. Module-Level Cache

In `src/quantstack/tools/langchain/ewf_tools.py`:

```python
_ewf_cycle_cache: dict[str, dict] = {}

def populate_ewf_cache(symbols: list[str], timeframes: list[str] = ["4h", "daily"]) -> None:
    """Called by data_refresh node at cycle start."""

def clear_ewf_cache() -> None:
    """Called at cycle end."""
```

### 2. Cache Lookup in Tools

Modify `get_ewf_analysis`:
- Check `_ewf_cycle_cache[f"{symbol}:{timeframe}"]` first
- Fall back to DB query on cache miss
- Same pattern for `get_ewf_blue_box_setups`

### 3. Trading Graph Integration

- `data_refresh` node: call `populate_ewf_cache(watchlist_symbols)`
- `reflect` node: call `clear_ewf_cache()`

### 4. Research Graph Integration

- `context_load` node: call `populate_ewf_cache(research_symbols)`
- Final node: call `clear_ewf_cache()`

---

## Rollback

Remove cache population calls. Cache misses fall back to DB — zero behavior change.

## Files

| File | Change |
|------|--------|
| `src/quantstack/tools/langchain/ewf_tools.py` | Modify — add cache |
| `src/quantstack/graphs/trading/nodes.py` | Modify — populate/clear |
| `src/quantstack/graphs/research/nodes.py` | Modify — populate/clear |
| `tests/unit/test_ewf_cache.py` | **Create** |
