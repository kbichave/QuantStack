# Section 07: Algo Scheduler Core (EMS)

## Overview

The OMS in `order_lifecycle.py` selects execution algorithms (IMMEDIATE, TWAP, VWAP, POV) based on `quantity / adv` thresholds (lines 455-478), but every order is executed as a single instant fill regardless of algorithm selection. This section builds the Execution Management System (EMS) -- a new `algo_scheduler.py` module that manages the parent/child order lifecycle for time-sliced execution.

The OMS already declares in its own header that it is separate from the EMS. The scheduler is that EMS. It imports OMS types, creates parent/child order hierarchies, runs an async execution loop, handles failures, and cleans up after crashes.

This section covers only the scheduling core: types, state machines, async loop, cancellation, crash recovery, and POV fallback. The actual TWAP/VWAP scheduling algorithms and paper broker simulation enhancements are in section-08.

**File to create:** `src/quantstack/execution/algo_scheduler.py`

**Dependencies:** section-02-fill-legs (fill_legs table and recording must exist so child fills are tracked as individual legs).

**Blocks:** section-08-twap-vwap (which implements the actual TWAP/VWAP scheduling strategies on top of this core).

---

## Tests First

All tests go in `tests/unit/execution/test_algo_scheduler.py`.

### AlgoParentOrder and ChildOrder Types

```python
# Test: AlgoParentOrder initializes with status="pending", filled_quantity=0
# Test: ChildOrder initializes with status="pending", attempts=0, broker_order_id=None
# Test: AlgoParentOrder fields include parent_order_id, symbol, side, total_quantity,
#       algo_type, start_time, end_time, arrival_price, max_participation_rate, status,
#       filled_quantity, avg_fill_price
# Test: ChildOrder fields include child_id, parent_id, scheduled_time, target_quantity,
#       filled_quantity, fill_price, status, attempts, broker_order_id
```

### Parent State Machine

```python
# Test: parent starts PENDING, transitions to ACTIVE on first child submit
# Test: parent transitions to COMPLETING when end_time reached
# Test: parent transitions to COMPLETED when filled_quantity >= 99.5% of total_quantity
# Test: parent transitions from ACTIVE to CANCELLING to CANCELLED on cancel request
# Test: invalid transitions raise (e.g., COMPLETED -> ACTIVE)
# Test: parent avg_fill_price updated as VWAP of all child fills after each child completes
```

### Child State Machine

```python
# Test: child starts PENDING
# Test: child transitions PENDING -> SUBMITTED when broker order placed
# Test: child transitions SUBMITTED -> PARTIALLY_FILLED on partial fill
# Test: child transitions SUBMITTED -> FILLED on complete fill
# Test: child transitions SUBMITTED -> CANCELLED on cancel
# Test: child transitions SUBMITTED -> EXPIRED on timeout
# Test: child transitions SUBMITTED -> REJECTED on broker rejection
```

### Parent-Child Invariant

```python
# Test: invariant holds -- sum(child.filled_qty) == parent.filled_qty at all times
# Test: after 3 children fill (100, 150, 50), parent.filled_quantity == 300
# Test: parent.avg_fill_price == VWAP across all filled children
```

### POV Fallback

```python
# Test: POV order dispatched as VWAP with max_participation_rate capped at 5%
# Test: POV order's algo_type stored as "vwap" (not "pov") for scheduling purposes
# Test: POV fallback logged with explicit rationale
```

### Child Failure Handling

```python
# Test: child REJECTED with buying_power reason -> retry with 50% size, max 3 attempts
# Test: child REJECTED with invalid_params -> fail child immediately (no retry)
# Test: child timeout (no fill within 2x bucket duration) -> cancel child, qty redistributed
# Test: API error on child submit -> exponential backoff (1s, 2s, 4s), retry
# Test: 3 consecutive child failures -> ALL active parents paused, alert logged
# Test: redistributed quantity added to next scheduled child's target_quantity
```

### Cancellation Triggers

```python
# Test: kill switch activation -> all active parents cancelled, all open children cancelled
# Test: risk gate daily halt -> all active parents cancelled
# Test: execution monitor exit signal for symbol -> only that symbol's parent cancelled
# Test: manual cancellation via trade service -> specific parent cancelled
# Test: cancelled parent's remaining unfilled children are all cancelled
# Test: cancellation of parent with in-flight child orders -> child broker orders cancelled
```

### Crash Recovery

```python
# Test: startup_recovery finds ACTIVE parents in DB and cancels them
# Test: startup_recovery finds COMPLETING parents in DB and cancels them
# Test: startup_recovery attempts to cancel open child broker orders via broker API
# Test: startup_recovery logs full context (parent_id, filled_qty, remaining_qty)
# Test: startup_recovery marks parents as CANCELLED with reason "system_restart_recovery"
# Test: startup_recovery does NOT attempt to resume mid-execution
# Test: startup_recovery runs before the scheduling loop begins
```

### Async Execution Loop

```python
# Test: scheduler loop wakes at each child's scheduled_time and submits to broker
# Test: broker calls use loop.run_in_executor (sync broker wrapped in async)
# Test: scheduler never acquires OMS RLock directly inside a coroutine
# Test: OMS state updates go through OMS public methods (which handle their own locking)
# Test: loop exits cleanly when no active parent orders remain
# Test: each child fill records a fill_leg (delegates to fill recording from section-02)
# Test: each child fill triggers TCA EWMA update (delegates to section-06 hook)
# Test: each child fill writes audit trail row (delegates to section-05 hook)
```

### Algo Performance Tracking

```python
# Test: algo_performance row created after parent reaches COMPLETED
# Test: implementation_shortfall_bps = (avg_fill - arrival_price) / arrival_price * 10000
# Test: child counts (num_children, num_children_filled, num_children_failed) accurate
# Test: timing fields populated (decision_time, first_fill_time, last_fill_time)
# Test: actual_participation_rate computed from filled_qty / market_volume_during_window
```

### Database Persistence

```python
# Test: algo_parent_orders table stores full parent lifecycle
# Test: algo_child_orders table stores child lifecycle with FK to parent
# Test: algo_performance table stores post-completion metrics
# Test: parent status updates persisted to DB on every state transition
# Test: child status updates persisted to DB on every state transition
```

---

## Implementation Details

### New File: `src/quantstack/execution/algo_scheduler.py`

This is the EMS. It is responsible for scheduling and executing child orders on behalf of parent algo orders. It does NOT contain the TWAP/VWAP scheduling algorithms themselves (those are in section-08); it provides the execution framework they plug into.

### Data Types

Two core dataclasses:

**`AlgoParentOrder`** represents the top-level algo execution request:

```python
@dataclass
class AlgoParentOrder:
    parent_order_id: str       # matches the OMS order_id
    symbol: str
    side: str                  # "buy" | "sell"
    total_quantity: int
    algo_type: str             # "twap" | "vwap"
    start_time: datetime
    end_time: datetime
    arrival_price: float       # mid-price at signal time
    max_participation_rate: float  # default 0.02 (2%), POV fallback caps at 0.05
    status: str                # "pending" | "active" | "completing" | "completed" | "cancelled"
    filled_quantity: int = 0
    avg_fill_price: float = 0.0
```

**`ChildOrder`** represents a single time slice:

```python
@dataclass
class ChildOrder:
    child_id: str              # unique ID, e.g., "{parent_id}-C{seq}"
    parent_id: str             # FK to AlgoParentOrder
    scheduled_time: datetime   # when to submit to broker
    target_quantity: int       # how many shares this slice targets
    filled_quantity: int = 0
    fill_price: float = 0.0
    status: str = "pending"    # "pending" | "submitted" | "partially_filled" | "filled" | "cancelled" | "expired" | "rejected"
    attempts: int = 0
    broker_order_id: str | None = None
```

### State Machines

**Parent transitions:**
```
PENDING -> ACTIVE        (first child submitted)
ACTIVE -> COMPLETING     (end_time reached OR filled >= 99.5%)
COMPLETING -> COMPLETED  (all children terminal AND filled >= 99.5%)
ACTIVE -> CANCELLING     (cancel requested)
CANCELLING -> CANCELLED  (all children terminal)
```

The parent never jumps states. Each transition is validated -- invalid transitions raise `ValueError`. Every transition is persisted to the `algo_parent_orders` table.

**Child transitions:**
```
PENDING -> SUBMITTED          (broker order placed)
SUBMITTED -> PARTIALLY_FILLED (partial fill received)
SUBMITTED -> FILLED           (full fill received)
PARTIALLY_FILLED -> FILLED    (remaining quantity filled)
SUBMITTED -> CANCELLED        (cancelled by parent or timeout)
SUBMITTED -> EXPIRED          (no fill within 2x bucket duration)
SUBMITTED -> REJECTED         (broker rejected)
```

### POV Fallback

The OMS algo selection can produce four outcomes: IMMEDIATE, TWAP, VWAP, POV. The scheduler implements TWAP and VWAP. POV orders are dispatched as VWAP with `max_participation_rate` capped at 5% (0.05). This is explicitly logged:

```python
if algo_type == "pov":
    algo_type = "vwap"
    max_participation_rate = min(max_participation_rate, 0.05)
    log.info("POV fallback: dispatching as VWAP with max_participation=5%")
```

IMMEDIATE orders bypass the scheduler entirely and continue through the existing single-fill path in the OMS.

### Child Failure Handling

The scheduler implements graduated failure responses:

1. **REJECTED (buying power):** Reduce child target quantity by 50%, retry. Maximum 3 attempts per child. If all attempts fail, redistribute remaining quantity to the next scheduled child.

2. **REJECTED (invalid params):** Fail the child immediately. Do not retry. Log the rejection reason for debugging.

3. **Timeout:** If a child receives no fill within `2 * bucket_duration` after submission, cancel the broker order and redistribute the unfilled quantity to the next scheduled child.

4. **API/network error:** Exponential backoff: wait 1s, 2s, 4s between retries. Maximum 3 retries per child.

5. **3 consecutive child failures (across any parents):** Pause ALL active parent orders. This signals a network-level or broker-level problem affecting all orders. Log an alert. Resume only on manual intervention or after a configurable cooldown period.

**Quantity redistribution:** When a child fails or partially fills, its unfilled quantity is added to the next scheduled child's `target_quantity`. If no more children are scheduled, the quantity remains unfilled and is logged as execution shortfall in the `algo_performance` record.

### Cancellation Triggers

Four sources can cancel a parent order:

1. **Kill switch activation:** Cancels ALL active parents immediately. For each parent, cancels all open child broker orders via the broker API. This is the highest priority -- it takes precedence over any in-flight child processing.

2. **Risk gate daily halt:** Same as kill switch -- cancels all active parents.

3. **Execution monitor exit signal:** Cancels only the parent for the symbol that triggered the exit. For example, if the execution monitor fires a stop-loss on AAPL, only the AAPL parent (if any is active) gets cancelled.

4. **Manual cancellation:** Cancels a specific parent by `parent_order_id`.

In all cases, the cancellation flow is:
1. Set parent status to CANCELLING
2. For each child in PENDING state: set to CANCELLED
3. For each child in SUBMITTED state: send cancel to broker, wait for confirmation, set to CANCELLED
4. Once all children are terminal: set parent to CANCELLED
5. Persist all state changes to DB

### Crash Recovery: `startup_recovery()`

This method runs once on scheduler startup, before the scheduling loop begins. It handles the case where the process crashed mid-execution (e.g., during a 30-minute TWAP).

Logic:
1. Query `algo_parent_orders` for any rows with `status IN ('active', 'completing')`
2. For each orphaned parent:
   a. Query its children for any with `status = 'submitted'` and a non-null `broker_order_id`
   b. Attempt to cancel those broker orders via `broker.cancel_order(broker_order_id)`
   c. Log full context: parent_id, symbol, algo_type, filled_qty, remaining_qty, number of children in each state
   d. Set parent status to CANCELLED with `cancel_reason = "system_restart_recovery"`
   e. Set all non-terminal children to CANCELLED
3. Persist all changes

The scheduler does NOT attempt to resume mid-execution. After a crash, market conditions may have changed significantly. The trading graph will re-evaluate and potentially submit a new order if the signal is still valid.

### Async Execution Loop: `AlgoScheduler`

The `AlgoScheduler` class manages the execution loop. Design:

```python
class AlgoScheduler:
    """Execution Management System for time-sliced algo orders."""

    def __init__(self, broker, oms, db_url: str): ...

    async def startup_recovery(self) -> None:
        """Cancel orphaned parent orders from previous crash. Runs once on startup."""
        ...

    async def submit_parent(self, parent: AlgoParentOrder, children: list[ChildOrder]) -> None:
        """Accept a new parent order with pre-computed children. Enqueue for execution."""
        ...

    async def cancel_parent(self, parent_order_id: str, reason: str) -> None:
        """Cancel a parent and all its children."""
        ...

    async def cancel_all(self, reason: str) -> None:
        """Cancel all active parents. Called by kill switch / daily halt."""
        ...

    async def run(self) -> None:
        """Main loop. Processes children from priority queue by scheduled_time."""
        ...
```

**Execution flow within `run()`:**
1. Maintain a priority queue of pending children sorted by `scheduled_time`
2. Sleep until the next child's `scheduled_time`
3. Submit the child to the broker via `loop.run_in_executor()` (broker.execute is synchronous)
4. Process the fill result (or handle failure per the rules above)
5. Update parent state: increment `filled_quantity`, recompute `avg_fill_price` as VWAP of all filled children
6. Record fill leg (delegates to fill recording infrastructure from section-02)
7. Trigger post-fill hooks: TCA EWMA update (section-06), audit trail (section-05)
8. Check parent completion condition (`filled >= 99.5%`)
9. If no more active parents, the loop becomes idle (not spinning)

**Sync/async boundary:** The existing OMS uses `threading.RLock`. The scheduler runs as an async loop. The contract:
- Broker calls are synchronous, wrapped with `loop.run_in_executor()`
- OMS state updates go through the OMS's public methods, which handle their own locking
- The scheduler NEVER acquires the OMS RLock directly inside a coroutine
- This matches the pattern already used by `execution_monitor.py`

**Position updates during execution:** Each child fill updates the position incrementally via `PortfolioState.upsert_position()`. Partially filled positions are visible to the execution monitor, which can evaluate stop-loss rules against the running average cost.

### Database Schema

Three new tables (created in section-01-schema-foundation):

**`algo_parent_orders`:**
| Column | Type | Notes |
|--------|------|-------|
| parent_order_id | TEXT PK | Matches OMS order_id |
| symbol | TEXT NOT NULL | |
| side | TEXT NOT NULL | "buy" or "sell" |
| total_quantity | INTEGER NOT NULL | |
| algo_type | TEXT NOT NULL | "twap" or "vwap" |
| start_time | TIMESTAMP NOT NULL | |
| end_time | TIMESTAMP NOT NULL | |
| arrival_price | DOUBLE PRECISION | Mid at signal time |
| max_participation_rate | DOUBLE PRECISION | Default 0.02 |
| status | TEXT NOT NULL | State machine value |
| filled_quantity | INTEGER DEFAULT 0 | |
| avg_fill_price | DOUBLE PRECISION DEFAULT 0.0 | |
| cancel_reason | TEXT | Null unless cancelled |
| created_at | TIMESTAMP DEFAULT now() | |
| updated_at | TIMESTAMP DEFAULT now() | |

**`algo_child_orders`:**
| Column | Type | Notes |
|--------|------|-------|
| child_id | TEXT PK | "{parent_id}-C{seq}" |
| parent_id | TEXT NOT NULL | FK to algo_parent_orders |
| scheduled_time | TIMESTAMP NOT NULL | |
| target_quantity | INTEGER NOT NULL | |
| filled_quantity | INTEGER DEFAULT 0 | |
| fill_price | DOUBLE PRECISION DEFAULT 0.0 | |
| status | TEXT NOT NULL DEFAULT 'pending' | |
| attempts | INTEGER DEFAULT 0 | |
| broker_order_id | TEXT | Null until submitted |
| created_at | TIMESTAMP DEFAULT now() | |
| updated_at | TIMESTAMP DEFAULT now() | |

Index on `parent_id` for fast child lookups. Index on `(status, scheduled_time)` for the priority queue query.

**`algo_performance`:**
| Column | Type | Notes |
|--------|------|-------|
| parent_order_id | TEXT PK | FK to algo_parent_orders |
| symbol | TEXT NOT NULL | |
| side | TEXT NOT NULL | |
| algo_type | TEXT NOT NULL | |
| total_qty | INTEGER | |
| filled_qty | INTEGER | |
| arrival_price | DOUBLE PRECISION | |
| avg_fill_price | DOUBLE PRECISION | |
| benchmark_vwap | DOUBLE PRECISION | Null if unavailable |
| implementation_shortfall_bps | DOUBLE PRECISION | (avg_fill - arrival) / arrival * 10000 |
| vwap_slippage_bps | DOUBLE PRECISION | Null if no benchmark |
| delay_cost_bps | DOUBLE PRECISION | |
| market_impact_bps | DOUBLE PRECISION | |
| num_children | INTEGER | |
| num_children_filled | INTEGER | |
| num_children_failed | INTEGER | |
| max_participation_rate | DOUBLE PRECISION | |
| actual_participation_rate | DOUBLE PRECISION | |
| decision_time | TIMESTAMP | When OMS selected the algo |
| first_fill_time | TIMESTAMP | |
| last_fill_time | TIMESTAMP | |
| scheduled_end_time | TIMESTAMP | |

The `algo_performance` row is written once when a parent reaches COMPLETED or CANCELLED (with partial fills). It is a post-hoc summary used for analysis, not for real-time decisions.

### Integration with Existing Code

**`order_lifecycle.py` (OMS):** When an order has `exec_algo` in (TWAP, VWAP, POV), the OMS delegates to the scheduler instead of executing directly. The OMS creates the order record, the scheduler creates the parent/child hierarchy and manages execution. The OMS's `fill_order()` method is still called for each child fill to maintain the existing fills/positions flow.

**`execution_monitor.py`:** Already runs an async loop for price ticks. The algo scheduler runs as a separate async task in the same event loop. The execution monitor can trigger parent cancellation by calling `scheduler.cancel_parent(order_id, reason="exit_signal")`.

**`risk_gate.py`:** No changes needed for this section. The risk gate evaluates the parent order before it reaches the scheduler. The scheduler trusts that any order it receives has already passed risk checks.

**`paper_broker.py`:** The scheduler submits child orders to the broker. The paper broker enhancement for simulating TWAP/VWAP children against historical bars is in section-09. Until then, child orders submitted to the paper broker will fill using the existing instant-fill model.

**Kill switch (`kill_switch.py`):** Register a callback so that when the kill switch fires, it calls `scheduler.cancel_all(reason="kill_switch")`.

### Data Feed Staleness Guard

If the price feed becomes stale (no update for > 30 seconds), the scheduler pauses child submission. Children remain in PENDING state until the feed reconnects. If the feed is stale for > 5 minutes, the scheduler cancels the parent with reason "price_feed_stale". This prevents executing into unknown market conditions.

---

## Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| Sync/async impedance between OMS (RLock) and scheduler (asyncio) | Scheduler calls broker via `run_in_executor()`. OMS updates go through public methods with their own locking. Scheduler never acquires RLock in a coroutine. |
| Crash mid-execution leaves orphaned state | `startup_recovery()` cancels all ACTIVE parents on restart. Never resumes -- let trading graph re-evaluate. |
| Data feed failure during multi-minute execution | Staleness guard: pause at 30s stale, cancel parent at 5 min stale. |
| Child failure cascade | 3 consecutive failures pause all parents. Graduated retry logic per failure type. |
| Priority queue starvation if many parents active | Expected low concurrency (few active parents at once). If needed, round-robin across parents. |

---

## Checklist

- [ ] Define `AlgoParentOrder` and `ChildOrder` dataclasses
- [ ] Implement parent state machine with validated transitions
- [ ] Implement child state machine with validated transitions
- [ ] Implement POV-to-VWAP fallback with 5% participation cap
- [ ] Implement child failure handling (retry, backoff, redistribution, pause-all)
- [ ] Implement cancellation from kill switch, risk halt, exit signal, manual
- [ ] Implement `startup_recovery()` for crash recovery
- [ ] Implement `AlgoScheduler` async execution loop
- [ ] Implement data feed staleness guard
- [ ] Create `algo_parent_orders`, `algo_child_orders`, `algo_performance` tables
- [ ] Integrate with OMS: delegate TWAP/VWAP/POV orders to scheduler
- [ ] Register kill switch callback for `cancel_all`
- [ ] Write `algo_performance` record on parent completion
- [ ] Write all tests in `tests/unit/execution/test_algo_scheduler.py`
