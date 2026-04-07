# Section 13: Regime Flip Forced Review

## Overview

When a position is entered during a specific market regime (e.g., `trending_up`) and the regime subsequently flips (e.g., to `trending_down` or `ranging`), the system currently logs an alert via `risk_gate.monitor()` but takes no protective action. The position stays open in a hostile regime until the next manual or scheduled review decides to close it.

This section adds three capabilities:

1. **Regime-at-entry persistence** -- store the regime when a position is opened (in both the `MonitoredPosition` dataclass and the positions DB table).
2. **Severity-based automated response** -- severe mismatches trigger auto-exit orders; moderate mismatches tighten stops.
3. **Stop floor enforcement** -- repeated tightening is capped at a minimum distance to prevent stops from being pushed into bid/ask spread noise.

## Dependencies

- **Section 01 (DB Migration)**: The `regime_at_entry` column on the positions table and backfill of existing rows to `'unknown'` must be in place before this section can persist regime data.
- No other section dependencies.

## Current State of the Code

The regime flip detection already exists in `src/quantstack/execution/risk_gate.py` inside `monitor()` (around line 785). It:

- Accepts `current_regimes: dict[str, dict]` and `entry_regimes: dict[str, str]` as arguments.
- Classifies opposite-direction flips as `CRITICAL` and lateral flips as `WARNING`.
- Appends a `MonitorAlert` with `recommended_action` of `"evaluate_exit"` (critical) or `"reduce"` (warning).
- Takes **no automated action** beyond the alert.

The `MonitoredPosition` dataclass lives in `src/quantstack/execution/execution_monitor.py` (line 75). It has no `regime_at_entry` field. Fields include `stop_price: float | None`, `entry_atr: float`, and `entry_price: float`, all of which are relevant to stop tightening math.

## Tests (Write First)

All tests go in `tests/unit/test_regime_flip.py`.

```python
# --- Persistence tests ---

# Test: regime_at_entry stored in DB on position creation
#   Create a position with regime_at_entry="trending_up", query DB, assert column value matches.

# Test: regime_at_entry loaded into MonitoredPosition from DB on restart
#   Insert a position row with regime_at_entry="ranging" directly in DB.
#   Reconstruct MonitoredPosition from DB row. Assert mp.regime_at_entry == "ranging".

# Test: existing positions backfilled with regime_at_entry = 'unknown'
#   Insert a position row WITHOUT regime_at_entry (simulating pre-migration data).
#   Verify the column defaults/backfills to 'unknown'.

# --- Severe mismatch tests ---

# Test: trending_up -> trending_down (severe) -> auto-exit order generated with reason
#   MonitoredPosition with regime_at_entry="trending_up", current regime="trending_down".
#   Call regime flip logic. Assert an exit order is returned with reason="regime_flip_severe".

# Test: trending_down -> trending_up (severe for short positions) -> auto-exit
#   Same pattern but for a short position. Assert exit order generated.

# Test: auto-exit flows through normal execution pipeline (not a bypass)
#   Verify the generated exit order is appended to the exit_orders list,
#   not executed directly. It must pass through risk_gate.check() like any other exit.

# --- Moderate mismatch tests ---

# Test: trending_up -> ranging (moderate) -> stop tightened by 50%
#   MonitoredPosition: entry_price=100, current_price=100, stop_price=90.
#   After moderate flip: new stop_price = 95 (distance halved from 10 to 5).

# Test: stop tightening math -- $100 current, $90 stop -> new stop $95
#   Explicit arithmetic verification of the 50% distance reduction.

# Test: ranging -> trending_* (moderate) -> stop tightened by 50%
#   Verify lateral-to-directional flips also trigger moderate tightening.

# Test: any -> unknown (moderate) -> stop tightened by 50%
#   Regime going to "unknown" is treated as moderate, not ignored.

# --- Stop floor tests ---

# Test: minimum stop floor -- after tightening, stop distance >= max(2x ATR, 1% price)
#   MonitoredPosition: current_price=100, stop_price=99, entry_atr=0.80.
#   Floor = max(2*0.80, 100*0.01) = max(1.60, 1.00) = 1.60.
#   50% tightening would give distance 0.50, but floor enforces 1.60.
#   New stop = 100 - 1.60 = 98.40.

# Test: repeated tightening capped at floor -- two flips don't push stop into noise
#   Apply moderate flip twice. Verify stop never goes below floor distance.

# Test: stop_price = None -> stop SET at floor distance (not tightened)
#   MonitoredPosition with stop_price=None, entry_atr=1.0, current_price=100.
#   Floor = max(2*1.0, 100*0.01) = max(2.0, 1.0) = 2.0.
#   New stop = 100 - 2.0 = 98.0. Assert stop is SET, not skipped.
```

## Implementation Details

### 1. Add `regime_at_entry` to `MonitoredPosition`

**File**: `src/quantstack/execution/execution_monitor.py`

Add a new field to the `MonitoredPosition` dataclass:

```python
regime_at_entry: str = "unknown"
```

This field is populated from the DB when reconstructing positions on restart, and set from the current regime when a new position is opened.

### 2. Add `regime_at_entry` Column to Positions Table

**Handled by Section 01 (DB Migration)**. The column is:

- `regime_at_entry TEXT` (nullable)
- Existing rows backfilled to `'unknown'`
- DB is source of truth; `MonitoredPosition` is runtime cache

### 3. Populate `regime_at_entry` on Position Creation

Wherever a new position is persisted to the DB (the position creation path in the execution pipeline), include the current regime value. The exact insertion point depends on how positions are written -- look for the INSERT into the positions table and add `regime_at_entry` to the column list.

### 4. Regime Comparison Logic

**File**: `src/quantstack/execution/risk_gate.py` (extend `monitor()` method, around line 785)

Replace the current alert-only logic with severity-based actions. The severity matrix:

| Entry Regime | Current Regime | Severity | Action |
|---|---|---|---|
| `trending_up` | `trending_down` | Severe | Auto-exit within 1 cycle |
| `trending_down` | `trending_up` | Severe (short positions) | Auto-exit within 1 cycle |
| `trending_up` | `ranging` | Moderate | Tighten stops by 50% |
| `trending_down` | `ranging` | Moderate | Tighten stops by 50% |
| `ranging` | `trending_*` | Moderate | Tighten stops by 50% |
| Any | `unknown` | Moderate | Tighten stops by 50% |

The existing `opposites` set and severity classification in `monitor()` already handles severe vs. moderate. The change is to add action logic after the alert is appended.

### 5. Auto-Exit (Severe Mismatch)

When severity is `CRITICAL` (opposite-direction flip):

- Generate an exit order dict with `reason: "regime_flip_severe"` and the position's symbol, side, and quantity.
- Append it to the cycle's `exit_orders` list so it flows through the normal execution pipeline (including `risk_gate.check()`).
- This is NOT a direct execution bypass. The exit order enters the same pipeline as any other exit.

The `monitor()` method return type (`MonitorReport`) may need an additional field for generated exit orders, or the caller must translate `CRITICAL` alerts with `recommended_action="evaluate_exit"` into actual exit orders. Choose the approach that keeps the boundary clean -- `monitor()` recommends, the calling node acts.

### 6. Stop Tightening (Moderate Mismatch)

When severity is `WARNING` (lateral flip or flip to `unknown`):

- Compute the current stop distance: `distance = current_price - stop_price` (for longs; reversed for shorts).
- New distance = `distance * 0.5` (halve the distance).
- Compute the floor: `floor = max(2 * entry_atr, 0.01 * current_price)`.
- Enforce: `new_distance = max(new_distance, floor)`.
- New stop: `current_price - new_distance` (for longs; `current_price + new_distance` for shorts).
- Update `stop_price` on the `MonitoredPosition` and persist to DB.

### 7. Handle `stop_price = None`

If a position has no existing stop (`stop_price is None`), the tightening formula has nothing to halve. In this case:

- SET a stop at the floor distance: `stop = current_price - max(2 * entry_atr, 0.01 * current_price)` (for longs).
- Positions without stops in hostile regimes are the highest-risk scenario -- they need one.
- This applies to both severe and moderate flips. A position with no stop facing a severe flip gets both: a stop is set AND an auto-exit order is generated (belt and suspenders).

### 8. Persist Stop Changes to DB

After modifying `stop_price` on a `MonitoredPosition`, the updated value must be written back to the positions table. Use the same DB write pattern used elsewhere in `execution_monitor.py` for position updates.

## Key Design Decisions

**Exits flow through the normal pipeline**: Auto-exits from regime flips are not executed directly. They enter `exit_orders` and pass through `risk_gate.check()` like any other exit. This preserves the invariant that the risk gate is the single enforcement point for all trades.

**Stop floor prevents death by a thousand cuts**: Without the floor, two consecutive moderate flips reduce stop distance to 25% of original ($10 -> $5 -> $2.50). Three flips: 12.5%. The `max(2x ATR, 1%)` floor prevents stops from entering bid/ask spread noise territory.

**`unknown` regime is treated as moderate, not ignored**: When the regime detector can't determine the current regime, that's a signal of uncertainty. Treating it as moderate (tighten stops) is the conservative response. Ignoring it would leave positions unprotected during periods of maximum uncertainty.

**DB is source of truth for `regime_at_entry`**: The `MonitoredPosition` dataclass is a runtime cache. On restart, positions are reconstructed from DB rows. The `regime_at_entry` column ensures no regime context is lost across restarts.

## Files Modified

| File | Change |
|------|--------|
| `src/quantstack/execution/execution_monitor.py` | Add `regime_at_entry: str = "unknown"` field to `MonitoredPosition` dataclass |
| `src/quantstack/execution/risk_gate.py` | Extend `monitor()` regime flip block (~line 785) with auto-exit generation and stop tightening logic |
| Position creation path (locate the INSERT) | Add `regime_at_entry` to position INSERT statements |
| `tests/unit/test_regime_flip.py` | **NEW** -- all tests listed above |

## Implementation Checklist

1. Write all tests in `tests/unit/test_regime_flip.py` (they should fail initially).
2. Add `regime_at_entry: str = "unknown"` to the `MonitoredPosition` dataclass.
3. Verify Section 01 DB migration adds the `regime_at_entry` column with `'unknown'` backfill.
4. Add `regime_at_entry` to position creation INSERT path.
5. Add `regime_at_entry` to position reconstruction (DB -> `MonitoredPosition`) path.
6. Implement stop tightening helper function with floor enforcement.
7. Implement `stop_price = None` handling (set stop at floor distance).
8. Extend `monitor()` to generate exit orders for severe mismatches.
9. Extend `monitor()` to call stop tightening for moderate mismatches.
10. Persist stop changes to DB after tightening.
11. Run tests, verify all pass.
