# Section 10: DB Transaction Isolation for Positions

## Background

The QuantStack trading system has two concurrent writers that can modify the same position row simultaneously:

1. **Execution monitor** (`src/quantstack/execution/execution_monitor.py`) -- an async task that evaluates exit rules on every price tick. It updates trailing stop high-water marks, triggers exits, and modifies position state.
2. **Trading graph** (`src/quantstack/graphs/trading/nodes.py`) -- sizes new entries, updates position metadata (strategy, exit levels), and processes fills.

Both operate on the `positions` table. Under PostgreSQL's default READ COMMITTED isolation, both can read the same row, compute updates independently, and one overwrites the other. This is a classic lost-update race condition. The supervisor graph only reads positions -- it does not need locking.

There are five distinct write paths into the `positions` table that must all use the same locking discipline:

| Write Path | File | What It Does |
|------------|------|--------------|
| Broker fill handler | `src/quantstack/execution/alpaca_broker.py` | Calls `portfolio.update_position()` on entry/exit fills |
| Trade service metadata | `src/quantstack/execution/trade_service.py` (line ~253) | Updates strategy, exit levels via `UPDATE positions SET ... WHERE symbol = ?` |
| Execution monitor | `src/quantstack/execution/execution_monitor.py` | Trailing stop HWM updates, exit execution |
| Portfolio state (multiple methods) | `src/quantstack/execution/portfolio_state.py` (lines ~310, ~411, ~508, ~573, ~596) | `update_position()`, `refresh_prices()`, `record_partial_exit()`, `update_exit_levels()`, `update_monitor_hwm()` |
| Kill switch position closer | `src/quantstack/execution/kill_switch.py` (lines ~121-128) | Emergency close-all via registered closer callback |

Startup reconciliation (introduced in Section 2) will also write to positions and must use the same pattern.

## Dependency

This section depends on **Section 1 (psycopg3 Migration)**. The locking pattern uses psycopg3's `conn.transaction()` context manager for explicit transaction control. Do not implement this section until Section 1 is complete.

## Tests

Write these tests before implementing the locking pattern. Test file: `tests/integration/test_transaction_isolation.py`.

### Row-level locking behavior

```python
# Test: SELECT FOR UPDATE acquires row lock on position
#   - Open transaction A, SELECT ... FOR UPDATE on symbol "AAPL"
#   - Verify the row is returned and lock is held (transaction A still open)

# Test: second writer blocks until first commits
#   - Transaction A holds FOR UPDATE lock on "AAPL"
#   - Transaction B attempts SELECT ... FOR UPDATE on same row
#   - Verify B blocks (does not return) until A commits
#   - After A commits, B acquires lock and proceeds

# Test: lock_timeout fires after 5s, retry once after 500ms
#   - Set lock_timeout = 5000 on connection
#   - Transaction A holds lock indefinitely
#   - Transaction B attempts lock, times out after 5s
#   - B retries after 500ms delay
#   - If A still holds lock, B times out again -> CRITICAL log emitted
#   - Verify operation continues with stale data (does not crash)

# Test: second retry timeout -> CRITICAL log, operation continues with stale data
#   - Simulate both lock attempts timing out
#   - Verify a CRITICAL-level log message is emitted with the symbol and context
#   - Verify the caller does NOT raise -- it proceeds with the pre-lock state

# Test: reader not blocked by writer (MVCC verification)
#   - Transaction A holds FOR UPDATE lock on "AAPL"
#   - A plain SELECT (no FOR UPDATE) on same row in a separate connection
#   - Verify the read returns immediately (not blocked)
#   - Verify the read returns the pre-update snapshot (MVCC)

# Test: single-row constraint -- transaction only locks one position row at a time
#   - Verify that update_position_with_lock() issues FOR UPDATE on exactly one row
#   - Verify no code path acquires locks on multiple position rows in one transaction
```

### Concurrency correctness

```python
# Test: two concurrent updates on same symbol -> no lost writes
#   - Writer A reads position (qty=100), adds 50
#   - Writer B reads position (qty=100), subtracts 20
#   - Both use locking pattern
#   - Final qty must be 130 (not 150 or 80)

# Test: execution monitor + trading graph race on same position -> both updates applied
#   - Simulate execution monitor updating trailing stop HWM
#   - Simultaneously simulate trading graph updating exit levels
#   - Verify both updates are reflected in final row state

# Test: N concurrent writers stress test -> acceptable latency, no lost updates
#   - Spawn 10 threads, each incrementing a counter column on the same position row
#   - After all complete, verify counter == initial + 10
#   - Verify no thread raised an unhandled exception
#   - Verify total wall-clock time is reasonable (< 60s for 10 sequential locks)
```

### Write path coverage

```python
# Test: alpaca_broker.py fill handler uses locking pattern
#   - Mock a fill event, verify the position update goes through update_position_with_lock()

# Test: trade_service.py metadata update uses locking pattern
#   - Call the metadata update path, verify SELECT FOR UPDATE is issued

# Test: execution_monitor.py trailing stop update uses locking pattern
#   - Trigger a trailing stop HWM update, verify locking pattern used

# Test: startup reconciliation uses locking pattern
#   - Simulate reconciliation writing a stop price, verify locking

# Test: kill_switch position closer uses locking pattern
#   - Trigger kill switch, verify position closer uses locking for state updates
```

## Implementation

### Core locking function

Create a reusable locking function. This is the single entry point for all position mutations. All five write paths must route through it.

**File:** `src/quantstack/execution/portfolio_state.py` (add to existing module)

```python
def update_position_with_lock(conn, symbol: str, updates: dict) -> bool:
    """Update a position row with exclusive row-level lock.

    Uses SELECT FOR UPDATE to prevent concurrent modifications.
    The lock is held only for the duration of the transaction.

    Args:
        conn: PgConnection instance (psycopg3-based, from Section 1)
        symbol: The ticker symbol identifying the position row
        updates: dict of column_name -> new_value to apply

    Returns:
        True if the update succeeded, False if the lock could not be acquired
        after retries (stale data path).

    Lock timeout: 5 seconds. One retry after 500ms. If both attempts fail,
    logs CRITICAL and returns False. Caller must handle the False case
    (typically: proceed with existing state, which is fail-safe because
    the position still has its prior protection levels).
    """
```

The function body follows this pattern:

1. Set `lock_timeout = '5s'` on the connection (session-level setting).
2. Begin an explicit transaction via `conn.transaction()` context manager.
3. Execute `SELECT * FROM positions WHERE symbol = %s FOR UPDATE` to acquire the row lock.
4. If the lock times out (psycopg raises `OperationalError` with `lock_timeout`), wait 500ms, retry once.
5. If the retry also times out, log CRITICAL with the symbol name and return False.
6. If the lock is acquired, apply the updates dict as a parameterized `UPDATE positions SET ... WHERE symbol = %s`.
7. Commit (implicit on context manager exit).

### Single-row constraint

This is a hard invariant: every transaction locks at most one position row. This eliminates deadlock risk entirely because there is no lock ordering problem when each transaction only holds one lock.

Enforce this by design -- the `update_position_with_lock()` function takes a single `symbol` parameter, not a list. There is no batch variant. Any code that needs to update multiple positions must call the function in a loop, with each call being its own transaction.

### Migrating the five write paths

Each write path currently does a bare `UPDATE positions SET ... WHERE symbol = ?` without any locking. Each must be refactored to call `update_position_with_lock()` instead.

**1. `portfolio_state.py` -- `update_position()` (line ~310)**

Currently executes a direct `UPDATE positions SET quantity = ?, avg_cost = ? ...`. Refactor to call `update_position_with_lock()`. The SELECT FOR UPDATE returns the current row state, which the function uses to compute the new values (e.g., averaging cost basis on an add).

**2. `portfolio_state.py` -- `refresh_prices()` (line ~411)**

Updates `current_price` and `unrealized_pnl`. This is a read-heavy path (called on every price tick). Price refreshes are idempotent and not safety-critical -- a stale price display is harmless. However, if the execution monitor is simultaneously updating the same row for a trailing stop, one write can be lost.

Decision: use the locking pattern here too, because the trailing stop HWM lives on the same row and must not be clobbered. The 5s timeout is generous enough that normal contention resolves quickly.

**3. `portfolio_state.py` -- other update methods (lines ~508, ~573, ~596)**

`record_partial_exit()`, `update_exit_levels()`, `update_monitor_hwm()` -- all follow the same pattern. Refactor each to use `update_position_with_lock()`.

**4. `trade_service.py` -- metadata update (line ~253)**

Currently builds a dynamic `UPDATE positions SET {sets} WHERE symbol = ?`. Refactor to pass the updates dict to `update_position_with_lock()`.

**5. `kill_switch.py` -- position closer callback (line ~121-128)**

The kill switch closer calls `broker.close_all_positions()` or similar. The actual position state mutation happens through `portfolio_state.update_position()` downstream. Once `update_position()` uses locking, this path is covered transitively. Verify this by tracing the call chain from the closer callback to the actual SQL statement.

### Timeout and retry logic

The retry is deliberately simple: one retry after a fixed 500ms delay. No exponential backoff, no jitter. Rationale:

- There are only two concurrent writers (execution monitor and trading graph). If one holds a lock, the other waits at most a few hundred milliseconds for a normal update to complete.
- A 5-second timeout that fires indicates something abnormal -- the lock holder is stuck (crashed mid-transaction, long GC pause, etc.). A single retry covers the case where the timeout was marginal.
- Two consecutive 5-second timeouts (10s total) means the system is unhealthy. Logging CRITICAL and continuing with stale data is the right tradeoff: the position still has its existing stop levels, which is safe. Hanging indefinitely waiting for a lock is worse.

### Failure mode: lock holder crashes mid-transaction

If a process crashes while holding a FOR UPDATE lock, PostgreSQL's idle-in-transaction timeout (already set to 30s in `PgConnection._ensure_raw()`) will terminate the session and release the lock. The 5s lock_timeout on the waiter means it will retry twice and fail before the 30s cleanup fires. This is acceptable: the position retains its prior state, and the next cycle will succeed.

After Section 1 (psycopg3 migration), verify that the idle-in-transaction timeout is still set on connections acquired from the new pool.

### What NOT to lock

Do not add locking to:

- **Position reads** for monitoring, dashboards, reporting, or the TUI (`src/quantstack/tui/queries/portfolio.py`). MVCC gives reads a consistent snapshot without blocking.
- **The risk gate's portfolio read** (`src/quantstack/execution/risk_gate.py`). It reads a snapshot for sizing calculations and does not write.
- **Non-position tables** (orders, signals, strategies, trades). These do not have concurrent write contention between the execution monitor and trading graph.
- **Supervisor graph reads**. The supervisor only reads positions for health checks.

### Connection management note

With psycopg3 (from Section 1), explicit transactions use the `conn.transaction()` context manager, which handles BEGIN/COMMIT/ROLLBACK automatically. The `SET lock_timeout` must be issued before entering the transaction block (it is a session-level setting, not a transaction-level one). Reset it after the transaction completes to avoid affecting subsequent queries on the same pooled connection.

```python
# Pseudocode for the locking pattern within psycopg3:
#
# conn.execute("SET lock_timeout = '5s'")
# with conn.transaction():
#     row = conn.execute(
#         "SELECT * FROM positions WHERE symbol = %s FOR UPDATE", [symbol]
#     ).fetchone()
#     if row is None:
#         return False  # position doesn't exist
#     conn.execute(
#         "UPDATE positions SET ... WHERE symbol = %s", [symbol, ...]
#     )
# conn.execute("SET lock_timeout = '0'")  # reset to no timeout
```

### Verification checklist

After implementation, verify the following manually or via integration tests:

1. Run two concurrent position updates on the same symbol -- confirm no lost writes.
2. Run a plain SELECT while a FOR UPDATE lock is held -- confirm it returns immediately.
3. Simulate a lock timeout -- confirm CRITICAL log is emitted and the caller does not crash.
4. Grep the codebase for all `UPDATE positions` statements -- confirm every one routes through `update_position_with_lock()` or is a read-only path.
5. Confirm no transaction acquires FOR UPDATE on more than one position row.
