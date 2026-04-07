# Section 05: Best Execution Audit Trail

## Overview

SEC Rule 606 and FINRA Rule 5310 require broker-dealers to demonstrate best execution. Even for a paper-trading-first system, building the audit trail now establishes the data model and collection habits so that when live trading begins, every fill is already documented with NBBO context, price improvement metrics, and algo selection rationale.

This section implements the `execution_audit` table and the fill-time hooks that populate it. It covers IMMEDIATE orders today and is designed so that future TWAP/VWAP child fills (section-08) slot in without schema changes.

**Dependencies:** section-02-fill-legs (the `fill_legs` table and `fill_leg_id` FK). The `execution_audit` table itself is created in section-01-schema-foundation, but this section owns the logic that populates it.

**Blocks:** section-14-integration-tests (audit trail assertions in cross-section tests).

---

## Tests (implement first)

File: `tests/unit/execution/test_audit_trail.py`

```python
# --- Schema & Recording ---

# Test: IMMEDIATE order fill creates exactly one execution_audit row
#   Setup: submit and fill an IMMEDIATE order via paper_broker
#   Assert: query execution_audit WHERE order_id = X returns 1 row

# Test: audit row captures NBBO bid/ask at fill time
#   Setup: mock NBBO source to return bid=99.90, ask=100.10
#   Fill order at 100.05
#   Assert: audit row has nbbo_bid=99.90, nbbo_ask=100.10, nbbo_midpoint=100.00

# Test: price_improvement_bps computed correctly for a buy (positive = favorable)
#   Setup: midpoint=100.00, fill_price=99.98 (bought below mid)
#   Assert: price_improvement_bps = (100.00 - 99.98) / 100.00 * 10000 = 0.2 bps

# Test: price_improvement_bps negative when fill is worse than midpoint
#   Setup: midpoint=100.00, fill_price=100.05 (bought above mid)
#   Assert: price_improvement_bps = (100.00 - 100.05) / 100.00 * 10000 = -0.5 bps

# Test: price_improvement_bps sign flipped for sells (selling above mid = favorable)
#   Setup: midpoint=100.00, fill_price=100.03 (sold above mid)
#   Assert: price_improvement_bps > 0

# Test: algo_selected and algo_rationale populated
#   Setup: order with exec_algo=IMMEDIATE
#   Assert: audit row algo_selected="IMMEDIATE", algo_rationale contains "ADV" context

# Test: audit row created even when NBBO fetch fails (with null NBBO fields)
#   Setup: mock NBBO source to raise exception
#   Assert: audit row exists with nbbo_bid=None, nbbo_ask=None, fill_price populated

# Test: fill_leg_id is None for IMMEDIATE single-fill orders
#   Assert: audit row fill_leg_id IS NULL (no fill_legs for single fills)

# Test: fill_leg_id populated for child fills (future TWAP/VWAP)
#   Setup: create a fill_leg row, call audit recorder with that leg_id
#   Assert: audit row fill_leg_id = the leg's id

# --- Query Interface ---

# Test: query "fills worse than NBBO midpoint" returns correct results
#   Setup: insert 3 audit rows: 2 with negative price_improvement_bps, 1 positive
#   Assert: WHERE price_improvement_bps < 0 returns exactly 2 rows

# Test: average price improvement by algo type
#   Setup: insert audit rows for IMMEDIATE and TWAP algos
#   Assert: GROUP BY algo_selected returns correct averages
```

---

## Data Model

The `execution_audit` table is created by section-01-schema-foundation. The schema for reference:

```python
@dataclass
class ExecutionAudit:
    audit_id: int           # auto-increment PK
    order_id: str           # FK to orders
    fill_leg_id: int | None # FK to fill_legs (None for IMMEDIATE, set for child fills)
    nbbo_bid: float | None  # None if NBBO fetch failed
    nbbo_ask: float | None
    nbbo_midpoint: float | None
    fill_price: float
    fill_venue: str | None  # "alpaca" | "paper"
    price_improvement_bps: float | None  # None if no NBBO available
    algo_selected: str      # "IMMEDIATE" | "TWAP" | "VWAP"
    algo_rationale: str     # human-readable, e.g. "TWAP: 0.5% ADV, 30-min window"
    timestamp_ns: int       # nanosecond-precision timestamp (time.time_ns())
```

Index on `order_id` for joining with orders table. Index on `timestamp_ns` for time-range queries.

---

## NBBO Capture

### Source

Alpaca IEX quotes (15-minute delayed). Acceptable for paper trading audit trail per interview decision. The NBBO fetcher is a thin wrapper around the existing Alpaca data client.

### Implementation

File: `src/quantstack/execution/audit_trail.py` (new file)

This module contains two components:

**1. `NBBOFetcher` class**

Responsible for fetching the current best bid/ask for a symbol. Signature:

```python
class NBBOFetcher:
    """Fetches NBBO quotes from Alpaca IEX for audit trail purposes."""

    def fetch(self, symbol: str) -> tuple[float | None, float | None]:
        """Return (bid, ask) or (None, None) if unavailable.

        Uses Alpaca IEX endpoint. Never raises -- returns None pair on any failure
        and logs the error with symbol context.
        """
        ...
```

Design decisions:
- Never raises. NBBO is supplementary data -- a fetch failure must not block fill recording. Log the failure, return Nones, and let the audit row record the fill without NBBO context.
- No caching. Each fill gets a fresh quote. For IMMEDIATE orders this is one call per fill. For TWAP/VWAP children, calls are naturally spaced by the scheduling interval (5+ minutes), so no rate limiting concern.

**2. `AuditRecorder` class**

Responsible for computing price improvement and persisting the audit row. Signature:

```python
class AuditRecorder:
    """Records best-execution audit trail for every fill event."""

    def __init__(self, nbbo_fetcher: NBBOFetcher):
        ...

    def record(
        self,
        order_id: str,
        symbol: str,
        side: str,           # "buy" | "sell"
        fill_price: float,
        fill_venue: str,
        algo_selected: str,
        algo_rationale: str,
        fill_leg_id: int | None = None,
    ) -> None:
        """Capture NBBO, compute price improvement, persist audit row.

        Called after every fill event -- both IMMEDIATE and child fills.
        Catches all exceptions internally; audit recording failure must
        never propagate to the fill path.
        """
        ...
```

### Price Improvement Calculation

```
midpoint = (bid + ask) / 2

For buys:  price_improvement_bps = (midpoint - fill_price) / midpoint * 10_000
For sells: price_improvement_bps = (fill_price - midpoint) / midpoint * 10_000
```

Positive = favorable (bought below mid, or sold above mid). Negative = adverse.

If NBBO is unavailable (fetch returned None), `price_improvement_bps` is stored as None.

### Algo Rationale String

The `algo_rationale` field captures *why* this algo was selected. Format:

- IMMEDIATE: `"IMMEDIATE: {qty} shares = {pct:.2f}% ADV, below TWAP threshold"`
- TWAP: `"TWAP: {qty} shares = {pct:.2f}% ADV, {duration}-min window, {n_children} slices"`
- VWAP: `"VWAP: {qty} shares = {pct:.2f}% ADV, volume-weighted {duration}-min window"`

The rationale is constructed at order submission time (in `order_lifecycle.py` where algo selection happens at lines 455-478) and stored on the `Order` object so it flows through to the audit recorder at fill time.

---

## Integration Points

### Hook into Fill Recording

The audit recorder is called from two places, both in the existing fill recording path:

**1. `OrderLifecycle.record_fill()` in `order_lifecycle.py`**

After the fill is persisted to the `fills` table (and `fill_legs` if applicable), call:

```python
# After fill persistence, before returning:
self._audit_recorder.record(
    order_id=order_id,
    symbol=order.symbol,
    side=order.side,
    fill_price=fill_price,
    fill_venue=fill_venue,
    algo_selected=order.exec_algo,
    algo_rationale=order.algo_rationale,
    fill_leg_id=fill_leg_id,  # None for IMMEDIATE
)
```

**2. `OrderLifecycle.record_partial_fill()`**

Same call, with the `fill_leg_id` from the newly created fill leg.

The `AuditRecorder` is injected into `OrderLifecycle` at construction. If not provided (backward compatibility), audit recording is silently skipped.

### Adding `algo_rationale` to Order

The `Order` dataclass in `order_lifecycle.py` needs a new field:

```python
algo_rationale: str = ""  # populated at algo selection time
```

The algo selection logic (lines 455-478) already computes `quantity / adv` -- add a rationale string at the same point:

```python
# In the algo selection block, after determining exec_algo:
order.algo_rationale = f"{exec_algo}: {quantity} shares = {pct_adv:.2f}% ADV, ..."
```

This is a minimal change to an existing code path.

---

## Error Handling

The audit trail is a secondary write. The invariant: **audit recording failure must never prevent or delay a fill from being recorded.** 

The `AuditRecorder.record()` method wraps its entire body in a try/except that:
1. Catches all exceptions
2. Logs the error with full context (order_id, symbol, fill_price, the exception)
3. Returns without raising

This means a database connectivity issue, a malformed quote response, or any other transient failure degrades gracefully -- the fill proceeds, the audit row is simply missing. The missing audit can be detected by a periodic reconciliation query:

```sql
SELECT o.order_id FROM orders o
LEFT JOIN execution_audit ea ON o.order_id = ea.order_id
WHERE o.status = 'FILLED' AND ea.audit_id IS NULL;
```

---

## File Summary

| File | Action | What Changes |
|------|--------|-------------|
| `src/quantstack/execution/audit_trail.py` | **Create** | `NBBOFetcher` and `AuditRecorder` classes |
| `src/quantstack/execution/order_lifecycle.py` | **Modify** | Add `algo_rationale` field to `Order`. Add rationale string in algo selection block. Call `AuditRecorder.record()` in `record_fill()` and `record_partial_fill()`. |
| `tests/unit/execution/test_audit_trail.py` | **Create** | All tests listed above |

---

## Audit Queries (Reference)

These are the key queries consumers of the audit trail will use. They do not require new code -- they are standard SQL against the `execution_audit` table. Documenting them here so the implementer can use them in tests and as acceptance criteria.

**Fills worse than NBBO midpoint:**
```sql
SELECT * FROM execution_audit
WHERE price_improvement_bps < 0
ORDER BY timestamp_ns DESC;
```

**Average price improvement by algo type:**
```sql
SELECT algo_selected, AVG(price_improvement_bps) AS avg_pi_bps, COUNT(*) AS fills
FROM execution_audit
WHERE price_improvement_bps IS NOT NULL
GROUP BY algo_selected;
```

**Execution quality over time (daily):**
```sql
SELECT DATE(to_timestamp(timestamp_ns / 1e9)) AS fill_date,
       AVG(price_improvement_bps) AS avg_pi_bps,
       COUNT(*) AS fills
FROM execution_audit
WHERE price_improvement_bps IS NOT NULL
GROUP BY fill_date
ORDER BY fill_date;
```

**Missing audit rows (reconciliation):**
```sql
SELECT o.order_id, o.symbol, o.status, o.updated_at
FROM orders o
LEFT JOIN execution_audit ea ON o.order_id = ea.order_id
WHERE o.status IN ('FILLED', 'PARTIALLY_FILLED')
  AND ea.audit_id IS NULL;
```
