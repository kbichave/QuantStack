# Section 02: Fill Legs Recording & VWAP Helper

## Overview

The `fills` table uses `order_id` as its primary key -- one row per order. When partial fills arrive (or when TWAP/VWAP child fills land in later sections), previous fill data is overwritten. You cannot reconstruct average fill price, execution VWAP, or fill trajectory. This section adds a `fill_legs` table for granular per-fill tracking and a dual-write pattern that keeps the existing `fills` summary backward-compatible.

**Depends on:** section-01-schema-foundation (the `fill_legs` table DDL must already exist)

**Blocks:** section-05-audit-trail, section-06-tca-ewma, section-07-algo-scheduler-core, section-08-twap-vwap

---

## Current State

### `fills` table (one row per order)

The paper broker writes a single row per order in `_record_fill()`:

```
src/quantstack/execution/paper_broker.py  (lines ~294-319)
```

Fields: `order_id, symbol, side, requested_quantity, filled_quantity, fill_price, slippage_bps, commission, partial, rejected, reject_reason, filled_at, session_id`.

The `Fill` Pydantic model lives in `src/quantstack/execution/paper_broker.py` (line ~87). `OrderRequest` is at line ~73. Both `PaperBroker` and `AlpacaBroker` produce `Fill` objects through their `execute()` methods.

The Alpaca broker path is in `src/quantstack/execution/alpaca_broker.py` -- it polls for fills via `_poll_for_fill()` and converts to `Fill` via `_to_fill()`.

The OMS in `src/quantstack/execution/order_lifecycle.py` has `record_fill()` (line ~294) and `record_partial_fill()` (line ~266) which update `Order` state but delegate persistence to the broker layer.

### Key constraint

All existing code that reads from `fills` must continue working. The `fill_legs` table is additive; `fills` remains the summary view.

---

## Tests (Write First)

Place tests in `tests/unit/execution/test_fill_legs.py`.

### Schema & Data Layer

```python
# Test: inserting a fill leg with all fields succeeds
# Test: unique constraint on (order_id, leg_sequence) prevents duplicates
# Test: inserting fill leg with nonexistent order_id fails (FK constraint)
```

### Fill Recording

```python
# Test: single fill creates one fill_leg row and updates fills summary
# Test: two partial fills for same order create two legs with correct sequences
# Test: fills summary row has VWAP of all legs after multiple partials
# Test: VWAP computation: 50 shares @ $100 + 50 shares @ $102 = avg $101.00
# Test: fill recording works for both paper_broker and alpaca_broker paths
```

### VWAP Helper

```python
# Test: compute_fill_vwap returns correct VWAP for multiple legs
# Test: compute_fill_vwap returns single price for single-leg fills
# Test: compute_fill_vwap raises for nonexistent order_id
```

All tests should use the existing in-memory SQLite fixture pattern from the `PaperBroker` tests. The schema-level tests (FK constraint) may need a PostgreSQL fixture or conditional skip if SQLite does not enforce FKs by default (SQLite requires `PRAGMA foreign_keys = ON`).

---

## Implementation

### 1. FillLeg dataclass

Add to `src/quantstack/execution/paper_broker.py` (or a new shared models file if preferred):

```python
@dataclass
class FillLeg:
    leg_id: int          # auto-increment PK
    order_id: str        # FK to orders
    leg_sequence: int    # 1, 2, 3... per order
    quantity: int
    price: float
    timestamp: datetime
    venue: str | None    # "paper" | "alpaca" -- used by audit trail (section-05)
```

The `(order_id, leg_sequence)` pair is unique-constrained in the DB (created by section-01). Index on `order_id` for fast leg lookups.

### 2. Modify `PaperBroker._record_fill()`

**File:** `src/quantstack/execution/paper_broker.py`

The current method does a single INSERT into `fills`. Change it to a dual-write:

1. Determine `leg_sequence` for this order: query `SELECT COALESCE(MAX(leg_sequence), 0) + 1 FROM fill_legs WHERE order_id = ?`.
2. INSERT into `fill_legs` with the individual fill details and `venue = "paper"`.
3. Compute running VWAP across all legs for this order: `SELECT SUM(quantity * price) / SUM(quantity) FROM fill_legs WHERE order_id = ?`.
4. Compute total filled quantity: `SELECT SUM(quantity) FROM fill_legs WHERE order_id = ?`.
5. UPSERT the `fills` summary row with the cumulative VWAP as `fill_price` and cumulative quantity as `filled_quantity`.

The existing `fills` INSERT becomes an UPSERT (INSERT ... ON CONFLICT(order_id) DO UPDATE) so that the first fill creates the row and subsequent partials update it.

Both the `fill_legs` INSERT and the `fills` UPSERT should happen inside the same `self._lock` block that currently protects the write, using a single transaction if possible.

### 3. Modify `AlpacaBroker` fill recording

**File:** `src/quantstack/execution/alpaca_broker.py`

Apply the same dual-write pattern. In `_to_fill()` or wherever the Alpaca broker persists fill data, add the `fill_legs` INSERT with `venue = "alpaca"` and update the `fills` summary via UPSERT.

The Alpaca broker may receive partial fills through its polling mechanism (`_poll_for_fill`). Each poll response that reports a new partial fill should create a new leg. Track the last known `filled_quantity` for the order and only create a leg for the incremental fill (new_filled - last_filled).

### 4. VWAP computation helper

Add a standalone function (not a method on any broker) so it can be called by TCA (section-06), algo performance (section-08), and audit trail (section-05):

```python
def compute_fill_vwap(conn, order_id: str) -> float:
    """Return volume-weighted average price across all fill legs for an order.

    Raises ValueError if no legs exist for the given order_id.
    """
    ...
```

**Location:** `src/quantstack/execution/fill_utils.py` (new file) or add to an existing shared module like `src/quantstack/execution/__init__.py`. A dedicated `fill_utils.py` is cleaner since multiple modules will import it.

The implementation queries `fill_legs` for the given `order_id`, computes `SUM(quantity * price) / SUM(quantity)`, and returns the result. If no rows exist, raise `ValueError(f"No fill legs for order {order_id}")`.

### 5. Record partial fills from OMS

**File:** `src/quantstack/execution/order_lifecycle.py`

The OMS method `record_partial_fill()` (line ~266) currently updates the `Order` object in memory. After updating the order state, it should call into the broker's fill recording path (which now does the dual-write). Verify that the existing call chain already does this -- if `record_partial_fill` delegates to the broker, no change is needed. If it writes to `fills` directly, add the `fill_legs` INSERT there as well.

---

## File Summary

| File | Action |
|------|--------|
| `src/quantstack/execution/paper_broker.py` | Modify `_record_fill()` for dual-write; add `FillLeg` dataclass |
| `src/quantstack/execution/alpaca_broker.py` | Modify fill recording for dual-write with `venue = "alpaca"` |
| `src/quantstack/execution/fill_utils.py` | New file: `compute_fill_vwap()` helper |
| `src/quantstack/execution/order_lifecycle.py` | Verify partial fill path uses broker's recording (may need no change) |
| `tests/unit/execution/test_fill_legs.py` | New file: all tests listed above |

---

## Edge Cases

- **Single-fill orders (IMMEDIATE):** The common case today. Creates exactly one leg with `leg_sequence = 1`. The `fills` summary row is identical to the single leg. No behavioral change from the caller's perspective.
- **Rejected orders:** The paper broker currently calls `_record_fill()` even for rejected fills (with `filled_quantity = 0`). Do NOT create a fill leg for rejected orders -- guard with `if fill.rejected or fill.filled_quantity == 0: return` before the `fill_legs` INSERT. The `fills` summary row can still be written for rejected orders to preserve existing behavior.
- **Concurrent partial fills:** The `self._lock` (threading.RLock) in `PaperBroker` serializes writes. For `AlpacaBroker`, ensure the leg_sequence query and INSERT are atomic (single transaction) to avoid race conditions on the sequence number.
- **SQLite vs PostgreSQL:** The paper broker uses SQLite (in-memory for tests, file for dev). Production uses PostgreSQL. The UPSERT syntax differs: SQLite uses `INSERT ... ON CONFLICT ... DO UPDATE`, PostgreSQL uses the same syntax. Both support it. Ensure the `fill_legs` table creation (from section-01) uses compatible DDL for both backends.

---

## Verification Criteria

After implementation, these must hold:

1. `SELECT COUNT(*) FROM fill_legs WHERE order_id = ?` equals the number of partial fills received for that order.
2. `compute_fill_vwap(conn, order_id)` matches `SELECT fill_price FROM fills WHERE order_id = ?` (the summary row tracks the same VWAP).
3. All existing tests that read from `fills` pass without modification (backward compatibility).
4. `sum(leg.quantity for leg in legs) == fills.filled_quantity` for every order (invariant).
