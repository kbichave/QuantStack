# Section 08: Performance Benchmarks

## Objective

Track portfolio performance against standard benchmarks and decompose returns into attributable components (signal, execution, timing, risk management).

## Files to Create

### `src/quantstack/autonomous/benchmarks.py`

Benchmark tracking and return attribution.

## Implementation Details

### BenchmarkTracker Class

```python
class BenchmarkTracker:
    def __init__(self, benchmarks: list[str] | None = None): ...

    async def compute_metrics(self, window: str = "1w") -> BenchmarkReport: ...
    async def compute_attribution(self, window: str = "1w") -> AttributionReport: ...
    def _fetch_benchmark_returns(self, symbol: str, window: str) -> pd.Series: ...
    def _fetch_portfolio_returns(self, window: str) -> pd.Series: ...
```

### Default Benchmarks

- `SPY` — S&P 500 (market benchmark)
- `60/40` — 60% SPY + 40% AGG (balanced benchmark)
- Equal-weight universe — equal allocation across all universe symbols

### BenchmarkReport Dataclass

```python
@dataclass
class BenchmarkReport:
    window: str  # "1w", "1m", "3m", "ytd"
    portfolio_return: float
    benchmarks: dict[str, BenchmarkMetrics]

@dataclass
class BenchmarkMetrics:
    benchmark_name: str
    benchmark_return: float
    alpha: float                # portfolio return - benchmark return
    beta: float                 # regression beta
    information_ratio: float    # alpha / tracking_error
    tracking_error: float       # std(portfolio_return - benchmark_return)
```

### Rolling Windows

Compute metrics for: 1w, 1m, 3m, YTD. Each window uses daily return series.

### Return Attribution

Decompose total return into four components:

1. **Signal alpha**: Difference between portfolio return and what a random entry/exit at the same times would yield. Measured by comparing actual signal-driven entries to time-randomized entries on the same symbols.

2. **Execution alpha**: Savings from execution optimization (TWAP/VWAP, smart routing) vs naive market orders. Source: `tca_results` table, `implementation_shortfall` column.

3. **Timing alpha**: Return from entry/exit timing vs holding from market open to close. Difference between actual entry price and daily VWAP.

4. **Risk management**: Return preserved by risk gate rejections. For each rejected trade, compute what the P&L would have been. Sum of avoided losses = risk management alpha.

### AttributionReport Dataclass

```python
@dataclass
class AttributionReport:
    window: str
    total_return: float
    signal_alpha: float
    execution_alpha: float
    timing_alpha: float
    risk_management_alpha: float
    residual: float  # total - sum of components (unexplained)
```

### Data Sources

- Portfolio returns: `portfolio_snapshots` table (daily equity values)
- Benchmark prices: Alpha Vantage daily data (already cached by data refresh job)
- TCA data: `tca_results` table
- Rejected trades: `risk_gate_rejections` table with `would_have_been_pnl` (computed after the fact)

### DB Schema

```sql
CREATE TABLE IF NOT EXISTS benchmark_snapshots (
    id SERIAL PRIMARY KEY,
    snapshot_date DATE NOT NULL,
    benchmark_name TEXT NOT NULL,
    daily_return DOUBLE PRECISION,
    cumulative_return DOUBLE PRECISION,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(snapshot_date, benchmark_name)
);
```

## Test Requirements

- `tests/unit/autonomous/test_benchmarks.py`:
  - Test metric computation with known return series (verify alpha, beta, IR calculations)
  - Test with zero portfolio return (no trades)
  - Test attribution components sum to approximately total return (within residual tolerance)
  - Test rolling window selection (1w = 5 trading days, 1m = ~21 trading days)
  - Test missing benchmark data (graceful handling, not crash)

## Acceptance Criteria

1. Alpha, beta, IR, tracking error computed correctly against standard finance formulas
2. Attribution components sum to total return within 5% residual
3. Works with as few as 5 data points (1 week of trading)
4. Benchmark data uses existing Alpha Vantage cache — no additional API calls
5. All computations are pure functions of return series (testable without DB)
6. Missing data for any single benchmark does not prevent computation of others
