# Section 05: Performance Attribution

## Overview

This section adds an automated P&L attribution engine that decomposes every trading cycle's returns into four explainable components: factor contribution (market/sector/style), timing quality (entry/exit vs VWAP), stock selection alpha (residual), and transaction costs (slippage + commissions). The engine runs as a deterministic graph node after `reflect` in the trading graph, persists results to a `cycle_attribution` DB table, and enforces an accounting identity assertion (all components must sum to total P&L).

**Goal:** Every trading cycle produces a structured attribution breakdown so the learning loop can identify whether returns came from market beta, good timing, genuine stock-picking alpha, or were eroded by costs.

## Dependencies

- **section-01-db-schema** (must be complete): Provides the `cycle_attribution` table and the `cycle_attribution: dict = {}` field on `TradingState`.
- **section-02-system-alerts** (must be complete): The `emit_system_alert()` helper is used when the accounting identity is violated.

## File Inventory

| File | Action | Purpose |
|------|--------|---------|
| `src/quantstack/performance/attribution.py` | **CREATE** | Attribution engine: dataclasses + `compute_cycle_attribution()` |
| `src/quantstack/graphs/trading/nodes.py` | **MODIFY** | Add `make_attribution` node factory |
| `src/quantstack/graphs/trading/graph.py` | **MODIFY** | Wire attribution node after `reflect`, before END |
| `tests/unit/test_attribution.py` | **CREATE** | Unit tests for attribution computation |
| `tests/integration/test_attribution_node.py` | **CREATE** | Integration tests for node wiring and DB persistence |

---

## Tests

Write all tests before implementation. The tests define the contract.

### Unit Tests (`tests/unit/test_attribution.py`)

```python
# Test: compute_cycle_attribution components sum to total_pnl (accounting identity)
#   Given a cycle with known positions, fills, and benchmark returns,
#   assert factor + timing + selection + cost == total_pnl within float tolerance (1e-6).

# Test: compute_cycle_attribution with zero fills returns all-zero components
#   Given an empty fills list, all four components and total_pnl should be 0.0.

# Test: compute_cycle_attribution factor_contribution reflects benchmark correlation
#   Given positions with known weights and a benchmark return of 2%,
#   factor_contribution should approximate sum(weight_i * benchmark_return).

# Test: compute_cycle_attribution timing_contribution reflects entry vs VWAP difference
#   Given a fill at $100 and a cycle VWAP of $102 for a long entry,
#   timing_contribution should be positive (bought below VWAP).
#   Given a fill at $104 and a VWAP of $102, timing_contribution should be negative.

# Test: compute_cycle_attribution cost_contribution equals slippage + commissions
#   Given fills with known slippage (fill_price vs mid_price) and commission amounts,
#   cost_contribution should equal the negative sum of (slippage + commissions).

# Test: compute_cycle_attribution handles cycle with no active positions (empty portfolio)
#   Returns CycleAttribution with all-zero fields and empty per_position list.

# Test: attribution accounting identity assertion fires when components don't sum
#   When factor + timing + selection + cost != total_pnl (e.g., due to a bug in
#   one component), the function should log a warning and place the difference
#   into an "unattributed" bucket rather than raising an exception.
```

### Integration Tests (`tests/integration/test_attribution_node.py`)

```python
# Test: attribution_node reads state, computes, writes to cycle_attribution table
#   Given a TradingState with positions, fills, and benchmark data populated by
#   upstream nodes, the attribution node should insert a row into cycle_attribution
#   with all four components and the cycle_id.

# Test: attribution_node returns dict with cycle_attribution key (TradingState compat)
#   The node's return dict must contain {"cycle_attribution": {...}} so TradingState
#   (which has extra="forbid") accepts the merge without ValidationError.

# Test: attribution_node runs after reflect in trading graph node ordering
#   Compile the trading graph and verify the edge from "reflect" goes to "attribution"
#   and "attribution" goes to END (not reflect -> END).
```

---

## Implementation Details

### 1. Attribution Engine (`src/quantstack/performance/attribution.py`)

This is a new module in the existing `src/quantstack/performance/` package.

**Data model:**

```python
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class PositionAttribution:
    """Per-position P&L decomposition."""
    symbol: str
    weight: float
    total_pnl: float
    factor_pnl: float
    timing_pnl: float
    selection_pnl: float
    cost_pnl: float


@dataclass
class CycleAttribution:
    """Full-cycle P&L decomposition across four components."""
    cycle_id: str
    total_pnl: float
    factor_contribution: float     # market + sector + style beta
    timing_contribution: float     # entry/exit quality vs VWAP
    selection_contribution: float  # stock-specific alpha (residual)
    cost_contribution: float       # slippage + commission drag
    per_position: list[PositionAttribution] = field(default_factory=list)
    computed_at: datetime = field(default_factory=datetime.utcnow)
```

**Core function signature:**

```python
async def compute_cycle_attribution(
    positions: list,
    fills: list,
    benchmark_returns: float,
    sector_returns: dict[str, float],
) -> CycleAttribution:
    """Decompose cycle P&L into four components.

    Factor: sum of (position_weight * benchmark_return) + sector residual.
      For each position, factor_pnl_i = weight_i * benchmark_return.
      Sector residual adds the difference between position's sector return
      and the broad benchmark return, weighted by position size.

    Timing: compare actual entry/exit prices to cycle VWAP.
      For longs: timing = (vwap - fill_price) * quantity (positive = bought cheap).
      For shorts: timing = (fill_price - vwap) * quantity.
      Uses fills from the current cycle only.

    Selection: residual P&L after removing factor and timing.
      selection = total_pnl - factor - timing - cost.
      This is the "alpha" component -- what's left after explaining
      market exposure, entry quality, and costs.

    Cost: realized slippage (fill_price vs mid at time of fill) + commissions.
      Always negative or zero. Pulled from fill metadata.

    Accounting identity: factor + timing + selection + cost == total_pnl.
    If violated beyond 1e-6 tolerance, log warning with the gap amount and
    add the discrepancy to an 'unattributed' field rather than raising.
    """
```

**Key design decisions:**

- **Selection is the residual.** Factor, timing, and cost are computed directly from data. Selection = total - factor - timing - cost. This guarantees the accounting identity holds by construction (selection absorbs any remainder). The assertion check is a defense against bugs in the factor/timing/cost calculations where total_pnl itself might be wrong.
- **VWAP source:** Use the cycle's volume-weighted average price from OHLCV data already fetched by `data_refresh`. If VWAP is unavailable for a symbol, fall back to (high + low) / 2 as an approximation.
- **Benchmark returns:** Read from the state's market context or fetch SPY return for the cycle period. The benchmark symbol is configurable via the `factor_config` table (default: SPY).
- **Sector returns:** Computed from sector ETF returns (XLK, XLF, XLE, etc.) or, if unavailable, use benchmark return for all sectors (degrades factor decomposition to pure beta but doesn't break).

### 2. Trading Graph Node (`src/quantstack/graphs/trading/nodes.py`)

Add a new node factory following the existing pattern in the file.

```python
def make_attribution():
    """Create the attribution node (deterministic, no LLM).

    Runs after reflect. Reads positions and fills from state,
    fetches benchmark data, computes attribution, persists to DB.
    """
    async def attribution_node(state: TradingState) -> dict:
        """Decompose cycle P&L and persist attribution."""
        # 1. Extract positions and fills from state
        # 2. Fetch benchmark returns for cycle period
        # 3. Fetch sector returns (best-effort)
        # 4. Call compute_cycle_attribution()
        # 5. Write to cycle_attribution table via db_conn()
        # 6. Return {"cycle_attribution": attribution.to_dict()}
        ...

    return attribution_node
```

**Important implementation notes:**

- This node is **deterministic** (no LLM call), same as `data_refresh` and `risk_sizing`.
- It must **never raise** -- if attribution computation fails (missing data, division by zero), log the error and return `{"cycle_attribution": {}}` so the graph completes. Attribution is observability, not a trade-critical path.
- Use `db_conn()` context manager for the DB write, consistent with all other nodes.
- The node reads `state.position_reviews`, `state.exit_orders`, `state.entry_orders` for fill data, and `state.data_refresh_summary` for market prices.

### 3. Graph Wiring (`src/quantstack/graphs/trading/graph.py`)

Three changes to the graph builder:

**a) Import the new node factory:**

Add `make_attribution` to the imports from `.nodes`.

**b) Register the node:**

```python
graph.add_node("attribution", make_attribution())
```

No retry policy needed -- deterministic node that handles its own errors.

**c) Update the edge from reflect:**

Change the current edge:
```python
graph.add_edge("reflect", END)
```

To:
```python
graph.add_edge("reflect", "attribution")
graph.add_edge("attribution", END)
```

**d) Update NODE_CLASSIFICATION:**

Add `"attribution": "non_blocking"` to the classification dict. Attribution failure should never halt the pipeline.

### 4. DB Persistence

The `cycle_attribution` table schema (created by section-01-db-schema):

| Column | Type | Description |
|--------|------|-------------|
| cycle_id | TEXT | Unique cycle identifier |
| graph_cycle_number | INTEGER | Cycle number from TradingState |
| total_pnl | FLOAT | Total P&L for the cycle |
| factor_contribution | FLOAT | Market/sector/style component |
| timing_contribution | FLOAT | Entry/exit quality component |
| selection_contribution | FLOAT | Stock-picking alpha component |
| cost_contribution | FLOAT | Transaction cost drag |
| per_position | JSONB | Serialized list of PositionAttribution |
| computed_at | TIMESTAMPTZ | When attribution was computed |

The insert query:

```python
INSERT INTO cycle_attribution (
    cycle_id, graph_cycle_number, total_pnl,
    factor_contribution, timing_contribution,
    selection_contribution, cost_contribution,
    per_position, computed_at
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
```

### 5. TradingState Field

Section-01 adds this field to `TradingState` in `src/quantstack/graphs/state.py`:

```python
cycle_attribution: dict = {}  # Populated by attribution_node after reflect
```

This must exist before the attribution node can return data, because `TradingState` uses `extra="forbid"`. If this field is missing, the graph will raise a `ValidationError` when merging the attribution node's return dict.

---

## Accounting Identity Enforcement

The accounting identity `factor + timing + selection + cost == total_pnl` is the critical invariant for this module.

**How it's maintained:**

1. Factor, timing, and cost are computed independently from data.
2. Selection is derived as `total_pnl - factor - timing - cost` (residual).
3. By construction, the identity always holds.
4. The assertion check validates that `total_pnl` itself is consistent with the position-level P&L sum. If the position-level P&Ls don't sum to the reported total, the discrepancy is logged and an `emit_system_alert()` is called with category `performance_degradation` and severity `warning`.

**What NOT to do:**

- Do not raise an exception on identity violation. The trading graph must complete.
- Do not silently swallow the discrepancy. Log it and alert.
- Do not attempt to "fix" the numbers by redistributing the gap. The unattributed bucket makes the discrepancy visible for debugging.

---

## Edge Cases

| Scenario | Handling |
|----------|----------|
| No fills this cycle | All components = 0.0, per_position = [] |
| No benchmark data available | factor_contribution = 0.0, log warning, selection absorbs everything |
| Position opened and closed in same cycle | Full round-trip attribution computed from entry and exit fills |
| Partial fill (order not fully filled) | Use actual filled quantity, not requested quantity |
| VWAP unavailable for a symbol | Fall back to (high + low) / 2 |
| Division by zero (zero portfolio value) | Return all-zero attribution, log warning |
| DB write fails | Log error, return attribution dict anyway (observability degrades, trading unaffected) |
