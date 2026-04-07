# Section 14: Cross-Section Integration Tests

## Overview

This section defines integration tests that verify the wiring between multiple execution layer subsections. Each test exercises a pipeline that spans two or more of the subsections implemented in sections 01-13. These tests are the final validation gate -- they confirm that the independently-built components compose correctly at runtime.

**Dependencies:** Sections 04 (SEC compliance), 05 (audit trail), 08 (TWAP/VWAP), 09 (paper broker enhance), 10 (liquidity model), 11 (slippage enhance), 12 (options monitoring), 13 (funding costs). All must be implemented before these integration tests can pass.

**File location:** `tests/integration/test_execution_layer_integration.py`

**Test runner:** `uv run pytest tests/integration/test_execution_layer_integration.py`

---

## Background

The execution layer is composed of subsystems that interact through shared database tables, fill hooks, and risk gate checks. Unit tests in each section verify that subsystem in isolation. These integration tests verify the data flow across subsystem boundaries:

- A TWAP child fill must propagate through fill_legs recording, TCA EWMA update, and audit trail insertion as a single atomic pipeline.
- A PDT check must count day trades created by TWAP round-trips, not just IMMEDIATE order round-trips.
- A wash sale triggered by a buy must retroactively adjust cost basis in the tax lot system.
- The algo scheduler must cleanly recover from a simulated crash, cancelling orphaned parent and child orders.
- Options monitoring rules must trigger exits through the execution monitor's standard exit machinery.

---

## Test Infrastructure

### Fixtures

The tests reuse the existing project fixtures (`PaperBroker` with in-memory SQLite, `MonitoredPosition` builder, `OrderRequest` helper) and add execution-layer-specific fixtures.

**Shared DB fixture:** All integration tests in this file share a single in-memory SQLite database with all execution layer tables created (fill_legs, tca_parameters, day_trades, pending_wash_losses, wash_sale_flags, tax_lots, algo_parent_orders, algo_child_orders, algo_performance, execution_audit, slippage_accuracy, plus the positions table with margin_used and cumulative_funding_cost columns). This fixture ensures tests exercise real SQL queries, not mocked data stores.

```python
@pytest.fixture
def execution_db():
    """In-memory SQLite with all execution layer tables.

    Creates every table from sections 01-13 so cross-section
    queries work end-to-end. Yields a connection context manager
    compatible with db_conn().
    """
    ...
```

**Paper broker with fill hooks fixture:** A `PaperBroker` instance wired to the shared DB with all post-fill hooks registered: fill_legs recording, TCA EWMA update, wash sale check, tax lot update, and audit trail insertion. This mirrors the production wiring in `order_lifecycle.py`.

```python
@pytest.fixture
def wired_paper_broker(execution_db):
    """PaperBroker with all post-fill hooks registered.

    Hooks: fill_legs → TCA EWMA → wash sale → tax lots → audit trail.
    Uses execution_db for all persistence.
    """
    ...
```

**Algo scheduler fixture:** An `AlgoScheduler` instance backed by the shared DB and wired paper broker, with a controllable clock for deterministic child scheduling.

```python
@pytest.fixture
def algo_scheduler(execution_db, wired_paper_broker):
    """AlgoScheduler with controllable clock and wired broker.

    Allows tests to advance time to trigger child order submission
    without real wall-clock waits.
    """
    ...
```

---

## Test 1: TWAP Child Fill Pipeline (Sections 02, 06, 05, 08)

**What it verifies:** A TWAP parent order produces child fills, each child fill creates a fill_leg row, triggers a TCA EWMA parameter update, and writes an execution_audit row. The full pipeline fires for every child, not just the final fill.

**Pipeline under test:**
1. Submit a TWAP parent order for 600 shares of SPY over 3 slices
2. Scheduler dispatches child 1 (~200 shares) to paper broker
3. Paper broker fills child against historical bar data
4. Post-fill hook chain fires: fill_leg insert -> TCA EWMA upsert -> execution_audit insert
5. Repeat for children 2 and 3
6. After all children complete, parent transitions to COMPLETED

**Assertions:**
- `fill_legs` table contains exactly 3 rows for the parent's order_id, with leg_sequence 1, 2, 3
- `tca_parameters` table has a row for SPY with `sample_count == 3` (one increment per child fill)
- `execution_audit` table contains 3 rows, each with a non-null `fill_leg_id` linking back to the correct leg
- Parent order's `filled_quantity` equals sum of all child `filled_quantity` values
- Parent order's `avg_fill_price` equals the VWAP across all fill legs
- `fills` summary row reflects the cumulative VWAP (backward-compatible view)

```python
@pytest.mark.asyncio
async def test_twap_fill_pipeline_creates_legs_tca_and_audit(
    algo_scheduler, execution_db
):
    """TWAP child fill -> fill_leg -> TCA EWMA update -> audit trail.

    Verifies the full post-fill hook chain fires for each child order,
    not just the final parent completion.
    """
    ...
```

---

## Test 2: PDT Counting with TWAP Round-Trips (Sections 04, 08)

**What it verifies:** When a TWAP buy completes and a subsequent TWAP sell completes on the same trading day, the system records a day trade. The PDT checker counts TWAP-originated day trades the same as IMMEDIATE order day trades. A 4th day trade is blocked when account equity is below $25K.

**Pipeline under test:**
1. Pre-seed `day_trades` table with 3 existing day trades in the rolling 5-business-day window
2. Submit a TWAP buy for 100 shares of AAPL (fills via children)
3. Submit a TWAP sell for 100 shares of AAPL on the same trading day (completing the round-trip)
4. Post-fill hook detects same-day open+close and inserts a `day_trades` record
5. Submit a 5th order (another buy-to-open) -- risk gate's PDT check queries `day_trades`, finds 4 in the window, and rejects the order

**Assertions:**
- After the sell completes, `day_trades` table has 4 rows in the 5-day window
- The 4th day trade record references the TWAP parent's order_ids (not child order_ids)
- A subsequent order submission returns a REJECT result with reason containing "PDT"
- If account equity is set to >= $25K, the same 5th order is APPROVED (PDT exemption)

```python
@pytest.mark.asyncio
async def test_pdt_counts_twap_round_trips(
    algo_scheduler, execution_db
):
    """Intraday TWAP round-trip records a day trade; 4th trade blocked under $25K.

    PDT enforcement must work with algorithmic fills, not just immediate orders.
    """
    ...
```

---

## Test 3: Wash Sale into Tax Lot Cost Basis Adjustment (Sections 04)

**What it verifies:** The two-phase wash sale detection correctly identifies a loss sale followed by a repurchase within 30 days, flags the wash sale, and adjusts the replacement lot's cost basis by the disallowed loss amount.

**Pipeline under test:**
1. Create a tax lot: buy 100 shares of XYZ at $50 (lot created in `tax_lots` table)
2. Sell 100 shares of XYZ at $45 -- realized loss of $500
3. Post-fill hook creates a `pending_wash_losses` record (loss=$500, window_end=sell_date+30 days)
4. Tax lot manager marks the original lot as closed with realized_pnl = -$500
5. Within 30 days, buy 100 shares of XYZ at $48
6. Post-fill hook detects the pending wash loss, creates a `wash_sale_flags` record
7. New tax lot's cost basis adjusted from $48 to $53 ($48 + $5 disallowed loss per share)
8. `pending_wash_losses` record marked as resolved

**Assertions:**
- After the loss sale: `pending_wash_losses` has 1 unresolved row for XYZ with `loss_amount == 500.0`
- After the loss sale: `tax_lots` has a closed lot with `realized_pnl == -500.0`
- After the repurchase: `wash_sale_flags` has 1 row with `disallowed_loss == 500.0`
- After the repurchase: new tax lot has `cost_basis == 53.0` and `wash_sale_adjustment == 5.0`
- After the repurchase: `pending_wash_losses` row is resolved (`resolved == True`)
- If the repurchase happens 31+ days after the sell, no wash sale is flagged and cost basis remains $48

```python
def test_wash_sale_adjusts_replacement_lot_cost_basis(
    wired_paper_broker, execution_db
):
    """Sell at loss -> buy within 30 days -> wash sale flagged -> cost basis adjusted.

    Exercises the two-phase detection: pending_wash_losses created on loss sale,
    resolved when replacement shares purchased within the window.
    """
    ...
```

---

## Test 4: Algo Scheduler Crash Recovery (Sections 07, 08)

**What it verifies:** When the algo scheduler restarts (simulating a crash), it finds ACTIVE parent orders in the database, cancels their open child broker orders, marks parents as CANCELLED with reason "system_restart_recovery", and does NOT attempt to resume execution.

**Pipeline under test:**
1. Submit a TWAP parent order for 1000 shares with 6 children
2. Allow children 1-3 to fill normally
3. Simulate a crash: stop the scheduler's async loop without graceful shutdown
4. Verify DB state: parent is ACTIVE, children 4-6 are PENDING, children 1-3 are FILLED
5. Create a new scheduler instance and call `startup_recovery()`
6. Recovery logic queries for ACTIVE parents, cancels any open broker child orders, marks parent as CANCELLED

**Assertions:**
- After recovery: parent order status is "cancelled" with cancel_reason "system_restart_recovery"
- After recovery: children 4-6 status is "cancelled"
- After recovery: children 1-3 remain "filled" (completed work is preserved)
- Parent's `filled_quantity` reflects only the 3 successful children (partial fill preserved)
- Broker cancel was called for any children that had open `broker_order_id` values
- No new child orders were submitted during recovery (conservative -- no resume)

```python
@pytest.mark.asyncio
async def test_algo_scheduler_crash_recovery(
    execution_db, wired_paper_broker
):
    """Active parents cancelled on restart; completed children preserved.

    Crash recovery must never resume mid-execution. It cancels orphaned
    state and lets the trading graph re-evaluate from scratch.
    """
    ...
```

---

## Test 5: Options Pin Risk Exit Trigger (Sections 12)

**What it verifies:** An options position with DTE < 3 and underlying price within 1% of strike triggers the pin_risk rule in the execution monitor, which calls `_submit_exit()` because pin_risk is configured as `auto_exit`.

**Pipeline under test:**
1. Create an options position: long call, strike $150, DTE = 2, instrument_type = "option"
2. Feed a price tick where underlying = $149.50 (within 1% of $150 strike)
3. Mock `compute_greeks_dispatch()` to return Greeks with the above parameters
4. Execution monitor's `_evaluate_options_rules()` fires, detects pin_risk condition
5. Since pin_risk action is "auto_exit", monitor calls `_submit_exit()`

**Assertions:**
- Broker's `execute()` was called with a sell order for the options position
- The exit reason logged includes "pin_risk"
- If the price is $145 (more than 1% from strike), pin_risk does NOT trigger
- If DTE is 5 (>= 3), pin_risk does NOT trigger even at $149.50
- Equity positions are skipped entirely by `_evaluate_options_rules()`
- If pin_risk is configured as "flag_only", the alert is logged but `_submit_exit()` is NOT called

```python
@pytest.mark.asyncio
async def test_options_pin_risk_triggers_exit(
    execution_db
):
    """Options position near strike at expiry triggers pin_risk auto_exit.

    Verifies that options-specific monitoring rules integrate with the
    standard execution monitor exit machinery.
    """
    ...
```

---

## Test 6: Slippage Accuracy Drift Alert (Sections 06, 11)

**What it verifies:** When the ratio of predicted slippage to realized slippage drifts beyond the 2.0x threshold, the slippage monitoring system raises an alert. This validates that TCA EWMA predictions are compared against actual fills and that accuracy tracking works end-to-end.

**Pipeline under test:**
1. Seed `tca_parameters` for AAPL morning bucket with `ewma_total_bps = 5.0`, `sample_count = 60`
2. Execute a fill where realized slippage is 12 bps (2.4x the predicted 5 bps)
3. Post-fill hook computes `predicted / realized` ratio and stores in `slippage_accuracy`
4. Accuracy monitoring detects ratio 0.42 (5/12), which is below the 0.5x threshold
5. Alert raised

**Assertions:**
- `slippage_accuracy` table has a row with `predicted_bps == 5.0` and `realized_bps == 12.0`
- Alert is logged (check logging output or alert table) indicating slippage model drift
- When predicted/realized ratio is between 0.5x and 2.0x (e.g., predicted=5, realized=6), no alert is raised

```python
def test_slippage_accuracy_drift_raises_alert(
    wired_paper_broker, execution_db
):
    """Slippage model drift beyond 2x threshold triggers alert.

    Validates the feedback loop from TCA EWMA predictions through
    fill recording to accuracy monitoring.
    """
    ...
```

---

## Implementation Notes

### Test Isolation

Each test function gets a fresh database via the `execution_db` fixture (function-scoped). Tests must not depend on execution order or shared mutable state outside the fixture-provided DB.

### Async Tests

Tests involving the algo scheduler or execution monitor use `@pytest.mark.asyncio`. The algo scheduler fixture provides a controllable clock (mock `asyncio.sleep` or time source) so tests do not require real wall-clock waits. Target: each test completes in under 2 seconds.

### Mocking Boundaries

- **Broker calls:** Use `MagicMock` for `broker.execute()` and `broker.cancel()` -- no real Alpaca API calls.
- **Price feed:** Use `PaperPriceFeed` with injected events (same pattern as existing `test_execution_monitor_integration.py`).
- **Greeks computation:** Mock `compute_greeks_dispatch()` to return controlled Greek values for options tests.
- **NBBO quotes:** Mock the Alpaca IEX quote endpoint to return deterministic bid/ask for audit trail tests.
- **Historical bars:** Inject synthetic bar data for paper broker TWAP fill simulation rather than fetching from Alpha Vantage.

### What Is NOT Tested Here

- Individual subsystem logic in isolation (covered by unit tests in sections 02-13)
- Live broker connectivity (out of scope for automated tests)
- Performance under load (separate benchmark suite if needed)
- UI/dashboard rendering of audit data

### File Structure

```
tests/integration/
    test_execution_layer_integration.py   # all 6 tests above
    conftest.py                           # existing; extend with execution_db fixture
```

The `execution_db` fixture can be added to the existing `tests/integration/conftest.py` or placed in `test_execution_layer_integration.py` directly. If other integration test files need the same DB fixture, prefer `conftest.py`.
