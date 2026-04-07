# Section 04: Factor Exposure Monitor

## Overview

Build a factor exposure computation module that calculates portfolio-level beta, sector weights, style scores, and momentum crowding every supervisor cycle. Persist results to `factor_exposure_history` for trend analysis. Trigger configurable drift alerts (via the system alert infrastructure from section-02) when exposures breach thresholds read from the `factor_config` DB table.

**Dependencies:**

- **section-01-db-schema** must be complete: this section requires the `factor_config` table (with default rows) and `factor_exposure_history` table.
- **section-02-system-alerts** must be complete: drift alerts are emitted via `emit_system_alert()` from `src/quantstack/tools/functions/system_alerts.py`.

**Blocks:** section-08-multi-mode (needs factor exposure for mode-aware computation).

---

## Existing Code Context

There is already a `FactorExposure` dataclass in `src/quantstack/risk/portfolio_risk.py` (line ~141) used by the `PortfolioRiskAnalyzer`. It tracks `market_beta`, `size_tilt`, `momentum_tilt`, `sector_concentration`, and `dominant_sector`. The new module extends this with richer metrics (full sector weight breakdown, style scores, momentum crowding percentage) and adds DB persistence and configurable alerting. The existing class is not replaced -- the new module serves a different purpose (monitoring + alerting vs. single-point risk scoring).

The supervisor graph's `health_check` node in `src/quantstack/graphs/supervisor/nodes.py` is the integration point. It already runs every supervisor cycle (300s) and performs system introspection. Factor exposure computation will be called from here.

Health metrics collection lives in `src/quantstack/graphs/supervisor/health_metrics.py` and follows a pattern of querying PostgreSQL and returning a dict of metrics. The factor exposure monitor should follow this same pattern.

---

## Tests

All tests go in `tests/unit/test_factor_exposure.py` and `tests/integration/test_factor_exposure.py`.

### Unit Tests

```python
# tests/unit/test_factor_exposure.py

# Test: compute_factor_exposure calculates beta correctly for known return series
#   Given a portfolio with returns perfectly correlated to benchmark (r=1.0),
#   beta should equal the ratio of portfolio vol to benchmark vol.
#   Use a synthetic return series where the answer is analytically known.

# Test: compute_factor_exposure returns sector_weights summing to 1.0 (within tolerance)
#   Given 3 positions in different sectors, verify weights sum to 1.0 +/- 1e-9.

# Test: compute_factor_exposure handles single-position portfolio
#   One position should produce sector_weights = {"<sector>": 1.0},
#   beta based on that single stock vs benchmark.

# Test: compute_factor_exposure handles portfolio with no sector data (falls back gracefully)
#   When positions lack sector metadata, sector_weights should be {"unknown": 1.0}
#   and no exception is raised.

# Test: check_factor_drift triggers alert when beta drift exceeds threshold
#   Given exposure with portfolio_beta=1.5 and config beta_drift_threshold=0.3
#   (drift from 1.0 = 0.5 > 0.3), should return one alert with category='factor_drift'.

# Test: check_factor_drift triggers alert when top sector exceeds sector_max_pct
#   Given exposure with top_sector_pct=55 and config sector_max_pct=40,
#   should return alert.

# Test: check_factor_drift triggers alert when momentum crowding exceeds threshold
#   Given exposure with momentum_crowding_pct=80 and config momentum_crowding_pct=70,
#   should return alert.

# Test: check_factor_drift returns empty list when all metrics within thresholds
#   Given exposure well within all thresholds, should return [].

# Test: check_factor_drift reads thresholds from config dict (not hardcoded)
#   Pass a custom config with higher thresholds; verify the same exposure
#   that triggered alerts with default config now passes.

# Test: factor config defaults are correct (beta=0.3, sector=40, momentum=70, benchmark=SPY)
#   Verify FACTOR_CONFIG_DEFAULTS constant matches expected values.
```

### Integration Tests

```python
# tests/integration/test_factor_exposure.py

# Test: factor_exposure_history row written on each computation
#   Call the full compute + persist flow with a test DB.
#   Query factor_exposure_history and verify one row exists with expected fields.

# Test: factor drift alert creates system_alert with category='factor_drift'
#   Trigger a drift condition, verify a row in system_alerts table
#   with category='factor_drift' and appropriate severity.
```

---

## Implementation

### New Module: `src/quantstack/risk/factor_exposure.py`

This is the core computation module. It contains the data model, computation logic, drift checking, and DB persistence.

#### Data Model

```python
@dataclass
class FactorExposureSnapshot:
    """Full factor exposure snapshot for monitoring and persistence."""
    portfolio_beta: float
    sector_weights: dict[str, float]  # sector name -> weight (0.0 to 1.0)
    top_sector: str
    top_sector_pct: float
    style_scores: dict[str, float]    # momentum, value, growth, quality -> score
    momentum_crowding_pct: float      # % of portfolio in top-momentum quintile
    benchmark_symbol: str
    alerts_triggered: int
    computed_at: datetime
```

#### Factor Config Defaults

```python
FACTOR_CONFIG_DEFAULTS: dict[str, str] = {
    "beta_drift_threshold": "0.3",
    "sector_max_pct": "40",
    "momentum_crowding_pct": "70",
    "benchmark_symbol": "SPY",
}
```

These defaults are also inserted as rows in the `factor_config` DB table by `ensure_tables()` (section-01). At runtime, the module reads from the DB table, falling back to these constants only if the DB read fails.

#### `compute_factor_exposure()`

```python
async def compute_factor_exposure(
    positions: list[dict],
    benchmark_symbol: str,
) -> FactorExposureSnapshot:
    """Compute portfolio factor exposure against benchmark.

    Beta: OLS regression of portfolio daily returns against benchmark
    over trailing 60 trading days. Portfolio returns are the
    weighted sum of per-position returns (weights by notional value).

    Sector weights: sum of position notional value by GICS sector,
    divided by total portfolio notional. Positions without sector
    metadata are bucketed under "unknown".

    Style scores (per-position, then portfolio-weighted average):
      - momentum: 12-month return minus 1-month return
      - value: inverse P/E percentile rank
      - growth: revenue growth percentile rank
      - quality: ROA percentile rank

    Momentum crowding: percentage of portfolio notional in positions
    whose momentum score is in the top quintile (>= 80th percentile).

    Positions list format: each dict must have at minimum 'symbol',
    'quantity', 'market_value'. Optional: 'sector', style factor fields.
    Missing optional fields degrade gracefully (sector -> "unknown",
    style -> 0.0, momentum crowding -> 0.0).
    """
```

Implementation notes:

- Benchmark return data: fetch from the existing data layer (`src/quantstack/data/fetcher.py`) using the same patterns as `PortfolioRiskAnalyzer.compute_factor_exposure()` in `portfolio_risk.py` (line ~378). That method already does beta regression over a trailing window.
- For sector weights, use the sector metadata already available on positions from the portfolio state. If missing, query Alpha Vantage OVERVIEW endpoint (already used elsewhere).
- Style scores can start simple -- momentum is computable from price history; value/growth/quality require fundamental data that may not be available for all symbols. Return 0.0 for unavailable scores and log a debug message.

#### `check_factor_drift()`

```python
async def check_factor_drift(
    exposure: FactorExposureSnapshot,
    config: dict[str, str],
) -> list[dict]:
    """Check factor exposure against configurable thresholds.

    Reads thresholds from the config dict (sourced from factor_config table).
    Returns a list of alert dicts ready for emit_system_alert().

    Checks performed:
    1. Beta drift: abs(portfolio_beta - 1.0) > float(config["beta_drift_threshold"])
       Severity: "warning" if drift < 2x threshold, "critical" if >= 2x.
    2. Sector concentration: top_sector_pct > float(config["sector_max_pct"])
       Severity: "warning".
    3. Momentum crowding: momentum_crowding_pct > float(config["momentum_crowding_pct"])
       Severity: "warning".

    Each alert dict has keys: category, severity, title, detail, metadata.
    Category is always 'factor_drift'.
    """
```

#### `load_factor_config()`

```python
async def load_factor_config() -> dict[str, str]:
    """Read factor_config table rows into a dict.

    Returns {config_key: value} for all rows in factor_config.
    Falls back to FACTOR_CONFIG_DEFAULTS if DB read fails (logs warning).
    Uses db_conn() context manager.
    """
```

#### `persist_factor_snapshot()`

```python
async def persist_factor_snapshot(snapshot: FactorExposureSnapshot) -> None:
    """Write a FactorExposureSnapshot row to factor_exposure_history.

    Uses db_conn() context manager. INSERT with all fields from the snapshot.
    sector_weights and style_scores stored as JSONB.
    """
```

#### `run_factor_exposure_check()`

Top-level orchestrator called by the supervisor health_check node:

```python
async def run_factor_exposure_check(positions: list[dict]) -> dict:
    """Full factor exposure cycle: compute, persist, check drift, emit alerts.

    1. Load config from factor_config table
    2. Compute exposure via compute_factor_exposure()
    3. Persist snapshot to factor_exposure_history
    4. Check drift via check_factor_drift()
    5. For each drift alert, call emit_system_alert() (from section-02)
    6. Return summary dict: {portfolio_beta, top_sector, alerts_triggered}

    This function is the single entry point for the supervisor integration.
    """
```

### Supervisor Integration: `src/quantstack/graphs/supervisor/nodes.py`

Modify the `health_check` node (the `make_health_check` factory function, line ~46) to call `run_factor_exposure_check()` after the existing health introspection logic.

The integration pattern:

1. After the existing health check LLM call completes, fetch current positions from the portfolio state (query the DB, same pattern used by `collect_health_metrics()` in `health_metrics.py`).
2. Call `run_factor_exposure_check(positions)`.
3. Include the factor exposure summary in the node's return dict under a `factor_exposure_summary` key.
4. If factor exposure computation fails (no positions, data fetch error, etc.), log the error and continue -- factor exposure is observability, not a gate. The health check must not fail because factor computation failed.

The call should be wrapped in a try/except that logs and continues:

```python
try:
    factor_summary = await run_factor_exposure_check(positions)
except Exception:
    logger.exception("Factor exposure check failed — continuing health check")
    factor_summary = {"error": "computation_failed"}
```

---

## DB Tables Required (from section-01)

For reference, these tables must exist before this section's code runs. They are created in section-01-db-schema.

**`factor_config`** -- configurable thresholds and benchmark:

| Column | Type | Notes |
|--------|------|-------|
| config_key | TEXT PK | e.g., "beta_drift_threshold" |
| value | TEXT | parsed by caller |
| updated_at | TIMESTAMPTZ | |

Default rows inserted by `ensure_tables()`: `beta_drift_threshold=0.3`, `sector_max_pct=40`, `momentum_crowding_pct=70`, `benchmark_symbol=SPY`.

**`factor_exposure_history`** -- per-cycle snapshots:

| Column | Type | Notes |
|--------|------|-------|
| id | BIGSERIAL PK | |
| portfolio_beta | FLOAT | |
| sector_weights | JSONB | |
| style_scores | JSONB | |
| momentum_crowding_pct | FLOAT | |
| benchmark_symbol | TEXT | |
| alerts_triggered | INT | |
| computed_at | TIMESTAMPTZ | |

---

## File Summary

| File | Action | Description |
|------|--------|-------------|
| `src/quantstack/risk/factor_exposure.py` | CREATE | Core module: data model, computation, drift check, persistence, orchestrator |
| `src/quantstack/graphs/supervisor/nodes.py` | MODIFY | Call `run_factor_exposure_check()` from `health_check` node |
| `tests/unit/test_factor_exposure.py` | CREATE | 10 unit tests for computation and drift logic |
| `tests/integration/test_factor_exposure.py` | CREATE | 2 integration tests for DB persistence and alert creation |

---

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Separate module vs. extending `PortfolioRiskAnalyzer` | New module `risk/factor_exposure.py` | Different purpose: monitoring + alerting + DB persistence vs. single-point risk scoring. The existing `FactorExposure` dataclass in `portfolio_risk.py` is simpler and used inline. Keeping them separate avoids bloating the risk analyzer with alert and persistence concerns. |
| Config source | DB table with code defaults as fallback | Thresholds will need tuning as portfolio composition changes. DB-backed config avoids code deploys for threshold adjustments. |
| Style scores starting simple | Return 0.0 for unavailable fundamental data | Momentum is always computable from price history. Value/growth/quality require fundamental data that may not be in the DB for all symbols. Start with what is available; enrich later as data coverage grows. |
| Supervisor integration vs. standalone service | Called from existing health_check node | No new service needed. The supervisor already runs every 300s and introspects system health. Factor exposure is another health dimension. |
| Failure isolation | try/except with continue | Factor exposure is observability. A computation failure must never block the supervisor health check or any downstream graph processing. |
