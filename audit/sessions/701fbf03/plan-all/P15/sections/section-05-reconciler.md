# Section 05: Reconciler

## Objective

Build position and P&L reconciliation between system state and broker state, ensuring the system never drifts from reality. Broker is always source of truth.

## Files to Create

### `src/quantstack/execution/reconciler.py`

Position and P&L reconciliation logic.

## Files to Modify

### `src/quantstack/execution/__init__.py`

Export `PositionReconciler`.

## Implementation Details

### PositionReconciler Class

```python
class PositionReconciler:
    def __init__(self, broker, tolerance_usd: float = 100.0, pnl_tolerance_pct: float = 0.01): ...

    async def reconcile_positions(self) -> ReconciliationReport: ...
    async def reconcile_pnl(self) -> PnLReconciliationReport: ...
    async def full_reconciliation(self) -> FullReconciliationReport: ...
    def _resolve_mismatch(self, mismatch: PositionMismatch) -> None: ...
```

### Position Reconciliation Flow

1. Fetch system positions from `positions` DB table
2. Fetch broker positions via broker API (`broker.get_positions()`)
3. Build a merged view keyed by symbol
4. For each symbol, compare:
   - Quantity (shares/contracts)
   - Side (long/short)
   - Notional value (price * quantity)
5. Flag mismatches exceeding `tolerance_usd` ($100 default)
6. For each mismatch: **adjust system state to match broker** (broker is source of truth)

### Mismatch Types

```python
class MismatchType(str, Enum):
    QUANTITY_DIFF = "quantity_diff"      # different share counts
    PHANTOM_SYSTEM = "phantom_system"    # system has position, broker doesn't
    PHANTOM_BROKER = "phantom_broker"    # broker has position, system doesn't
    SIDE_MISMATCH = "side_mismatch"      # system says long, broker says short
```

### ReconciliationReport Dataclass

```python
@dataclass
class ReconciliationReport:
    timestamp: datetime
    positions_checked: int
    mismatches: list[PositionMismatch]
    corrections_applied: list[str]
    all_clear: bool  # True if no mismatches found
```

### P&L Reconciliation

- Compare system-calculated daily P&L vs broker-reported P&L
- Alert threshold: >1% discrepancy on daily P&L
- Do NOT auto-correct P&L — log for investigation only (P&L calculation bugs need root-cause analysis)

### Resolution Strategy

For each mismatch type:
- `QUANTITY_DIFF`: update system position to match broker quantity
- `PHANTOM_SYSTEM`: remove phantom position from system DB, log warning
- `PHANTOM_BROKER`: insert broker position into system DB, log warning
- `SIDE_MISMATCH`: update system side to match broker, log ERROR (this is a serious bug)

### Scheduling

- Run on every mode transition (via transition hook from Section 02)
- Run every 4 hours via scheduler job
- Run on-demand via supervisor graph tool

### DB Logging

```sql
CREATE TABLE IF NOT EXISTS reconciliation_log (
    id SERIAL PRIMARY KEY,
    reconciliation_type TEXT NOT NULL,  -- 'position' or 'pnl'
    positions_checked INTEGER,
    mismatches_found INTEGER,
    corrections_applied JSONB,
    all_clear BOOLEAN,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

## Test Requirements

- `tests/unit/execution/test_reconciler.py`:
  - Mock broker and DB with matching positions → verify `all_clear=True`
  - Inject quantity mismatch → verify detection and correction
  - Inject phantom system position → verify removal
  - Inject phantom broker position → verify insertion
  - Inject side mismatch → verify detection and ERROR log
  - Test tolerance: $99 diff not flagged, $101 diff flagged
  - Test P&L reconciliation: 0.5% diff OK, 1.5% diff flagged
  - Test that reconciliation runs without error when no positions exist

## Acceptance Criteria

1. Broker is always source of truth — system state is corrected to match, never the reverse
2. Every mismatch is logged with full details (symbol, system value, broker value, correction)
3. P&L mismatches are logged but NOT auto-corrected
4. Tolerance is configurable via constructor parameter
5. Reconciliation completes without error even when broker API returns empty positions
6. All DB writes use `db_conn()` context manager
7. Side mismatch triggers ERROR-level log (indicates potential serious bug)
