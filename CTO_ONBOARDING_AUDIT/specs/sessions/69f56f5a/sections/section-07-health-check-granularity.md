# Section 7: Health Check Granularity

## Background

The supervisor graph runs every 5 minutes. Its `health_check` node (in `src/quantstack/graphs/supervisor/nodes.py`) currently uses an LLM call to run system introspection tools and return a JSON health status with an `overall` field. It checks kill switch status and heartbeat freshness for trading-graph (120s max) and research-graph (600s max), classifying each as healthy/degraded/critical.

What it does NOT track today:
- **Cycle success rate** -- how often do graph cycles complete without errors?
- **Error counts** -- how many errors occurred in the most recent cycle?
- **Strategy generation velocity** -- is the research pipeline actually producing strategies?
- **Research queue depth** -- is the backlog growing unboundedly?

These are pure operational metrics derivable from existing tables (`graph_checkpoints`, `strategies`, and the research queue) with simple SQL queries. They do not require ML, statistical modeling, or domain-specific thresholds.

IC (information coefficient) trend monitoring is explicitly deferred to a later phase -- it is a research-grade statistical computation that does not belong in an operational resilience effort.

### Dependencies

- **Section 06 (Monitoring Stack)**: This section requires Prometheus and Grafana to be running in docker-compose. The Prometheus scrape config must include the supervisor-graph's `/metrics` endpoint. The Grafana alerting provisioning directory must exist at `config/grafana/provisioning/alerting/`.
- **Existing Prometheus metrics module**: `src/quantstack/observability/metrics.py` already registers gauges (portfolio NAV, daily P&L, kill switch state, signal staleness) via `prometheus_client`. The new health gauges follow the same pattern.

---

## Tests First

All tests go in `tests/unit/test_health_metrics.py`. These are unit tests -- they do not require a running database or Docker services. Mock the DB queries.

```python
# tests/unit/test_health_metrics.py

import pytest

# --- collect_health_metrics() ---

# Test: computes cycle success rate from graph_checkpoints
#   Setup: mock DB query returning 10 checkpoint rows, 7 with status='success'
#   Assert: returned dict has cycle_success_rate == 0.70 for the given graph

# Test: returns 0.0 success rate when no checkpoints exist
#   Setup: mock DB query returning empty result
#   Assert: cycle_success_rate == 0.0

# Test: computes error count for most recent cycle
#   Setup: mock DB query returning 3 rows with status='error' for latest cycle_id
#   Assert: cycle_error_count == 3

# Test: counts strategies created in last 7 days
#   Setup: mock DB query returning count=5
#   Assert: strategy_generation_7d == 5

# Test: counts pending research queue items
#   Setup: mock DB query returning count=12
#   Assert: research_queue_depth == 12

# --- Prometheus gauge updates ---

# Test: cycle_success_rate gauge updated with computed value
#   Setup: call update function with success_rate=0.85, graph_name='trading'
#   Assert: gauge .labels(graph_name='trading')._value equals 0.85

# Test: cycle_error_count gauge updated with computed value
#   Setup: call update function with error_count=2, graph_name='research'
#   Assert: gauge .labels(graph_name='research')._value equals 2.0

# Test: strategy_generation_7d gauge updated
#   Setup: call update function with count=3
#   Assert: gauge._value equals 3.0

# --- Threshold alerting (Grafana rule expressions) ---
# These validate the PromQL expressions are correct, not that Grafana fires them.

# Test: alert rule YAML contains expression for success rate < 0.70
#   Read the alerts.yaml file content, assert the PromQL expression is present

# Test: alert rule YAML contains expression for error count > 3
#   Same approach

# Test: alert rule YAML contains info-severity rule for 0 strategies in 7 days
#   Same approach
```

---

## Implementation

### 7.1 Health metrics collector function

**File:** `src/quantstack/graphs/supervisor/nodes.py` (extend, do not replace existing health_check node)

Add a new function `collect_health_metrics()` that the existing `health_check` node calls after its current LLM-based check. This function queries the database directly -- it does not use the LLM.

```python
async def collect_health_metrics() -> dict[str, float]:
    """Query operational health metrics from PostgreSQL.

    Returns dict with keys:
        - trading_cycle_success_rate (float, 0.0-1.0)
        - research_cycle_success_rate (float, 0.0-1.0)
        - trading_cycle_error_count (int)
        - research_cycle_error_count (int)
        - strategy_generation_7d (int)
        - research_queue_depth (int)
    """
```

**Cycle success rate query** (per graph, last 10 cycles):
```sql
SELECT
    COUNT(*) FILTER (WHERE status = 'success') AS successes,
    COUNT(*) AS total
FROM graph_checkpoints
WHERE graph_name = %(graph_name)s
ORDER BY started_at DESC
LIMIT 10;
```
Return `successes / total` if total > 0, else 0.0.

**Error count query** (most recent cycle):
```sql
SELECT COUNT(*) AS error_count
FROM graph_checkpoints
WHERE graph_name = %(graph_name)s
  AND status = 'error'
  AND cycle_id = (
      SELECT cycle_id FROM graph_checkpoints
      WHERE graph_name = %(graph_name)s
      ORDER BY started_at DESC LIMIT 1
  );
```

**Strategy generation velocity** (last 7 days):
```sql
SELECT COUNT(*) FROM strategies
WHERE created_at >= NOW() - INTERVAL '7 days';
```

**Research queue depth**:
```sql
SELECT COUNT(*) FROM research_tasks
WHERE status = 'pending';
```

Use `db_conn()` context manager for all queries. Each query is a single short-lived connection -- do not hold a connection across all four queries.

After collecting metrics, store them in `system_state` for non-Prometheus consumers:
- Key pattern: `health_metric_{metric_name}`
- Value: the numeric value as a string
- Update via `INSERT ... ON CONFLICT DO UPDATE`

### 7.2 Prometheus gauge registration

**File:** `src/quantstack/observability/metrics.py` (extend existing module)

Follow the existing pattern in this file: module-level variables initialized to `None`, lazy initialization in `_init_metrics()`, and public setter functions.

New gauges to register inside `_init_metrics()`:

```python
_cycle_success_rate: Gauge | None = None
_cycle_error_count: Gauge | None = None
_strategy_generation_7d: Gauge | None = None
_research_queue_depth: Gauge | None = None

# Inside _init_metrics():
_cycle_success_rate = Gauge(
    "quantstack_cycle_success_rate",
    "Fraction of last 10 graph cycles that completed successfully",
    ["graph_name"],
    registry=REGISTRY,
)
_cycle_error_count = Gauge(
    "quantstack_cycle_error_count",
    "Number of errors in the most recent graph cycle",
    ["graph_name"],
    registry=REGISTRY,
)
_strategy_generation_7d = Gauge(
    "quantstack_strategy_generation_7d",
    "Number of new strategies created in the last 7 days",
    registry=REGISTRY,
)
_research_queue_depth = Gauge(
    "quantstack_research_queue_depth",
    "Number of pending research tasks in the queue",
    registry=REGISTRY,
)
```

Public setter functions:

```python
def record_cycle_success_rate(graph_name: str, rate: float) -> None:
    """Set the cycle success rate gauge for a graph."""

def record_cycle_error_count(graph_name: str, count: int) -> None:
    """Set the cycle error count gauge for a graph."""

def record_strategy_generation(count: int) -> None:
    """Set the 7-day strategy generation count gauge."""

def record_research_queue_depth(count: int) -> None:
    """Set the research queue depth gauge."""
```

Each follows the existing pattern: call `_init_metrics()`, then set the gauge value.

### 7.3 Integration into supervisor health_check node

**File:** `src/quantstack/graphs/supervisor/nodes.py`

Inside `make_health_check`, after the existing LLM-based health check logic, add a call to `collect_health_metrics()` and then update the Prometheus gauges:

```python
# After existing health_status is computed:
try:
    metrics = await collect_health_metrics()
    record_cycle_success_rate("trading", metrics["trading_cycle_success_rate"])
    record_cycle_success_rate("research", metrics["research_cycle_success_rate"])
    record_cycle_error_count("trading", metrics["trading_cycle_error_count"])
    record_cycle_error_count("research", metrics["research_cycle_error_count"])
    record_strategy_generation(metrics["strategy_generation_7d"])
    record_research_queue_depth(metrics["research_queue_depth"])
except Exception as exc:
    logger.warning("Health metrics collection failed: %s", exc)
    # Non-fatal -- the existing health check result is still valid
```

The health metrics collection is wrapped in a try/except because it must not break the existing health check if the queries fail (e.g., table doesn't exist yet, DB connection issue). Log WARNING, not ERROR -- this is a degraded-but-functional state.

### 7.4 Grafana alert rules

**File:** `config/grafana/provisioning/alerting/alerts.yaml` (extend -- this file is created in Section 06)

Add these rules to the existing alert rules file:

| Alert Name | PromQL Expression | For | Severity | Channel |
|------------|------------------|-----|----------|---------|
| Low cycle success rate | `quantstack_cycle_success_rate < 0.70` | 15m | Warning | Discord |
| High cycle error count | `quantstack_cycle_error_count > 3` | 0m | Warning | Discord |
| No strategies generated | `quantstack_strategy_generation_7d == 0` | 0m | Info | Discord |
| Research queue backlog | `quantstack_research_queue_depth > 50` | 5m | Warning | Discord |

The "for" duration on success rate (15m = 3 supervisor cycles) prevents alerting on transient single-cycle failures. Error count fires immediately because a single cycle with >3 errors warrants attention. Strategy generation is informational -- zero strategies in 7 days means the research pipeline is stalled but is not an emergency. Research queue backlog at >50 with 5m persistence means the queue is growing, not just spiking.

Alert rule YAML structure (follows Grafana provisioning format):

```yaml
# Appended to existing alerts.yaml
- orgId: 1
  name: Low cycle success rate
  condition: C
  data:
    - refId: A
      datasourceUid: prometheus
      model:
        expr: quantstack_cycle_success_rate < 0.70
  for: 15m
  labels:
    severity: warning
  annotations:
    summary: "Graph {{ $labels.graph_name }} cycle success rate below 70%"

- orgId: 1
  name: High cycle error count
  condition: C
  data:
    - refId: A
      datasourceUid: prometheus
      model:
        expr: quantstack_cycle_error_count > 3
  for: 0m
  labels:
    severity: warning
  annotations:
    summary: "Graph {{ $labels.graph_name }} has {{ $value }} errors in latest cycle"

- orgId: 1
  name: No strategies generated (7d)
  condition: C
  data:
    - refId: A
      datasourceUid: prometheus
      model:
        expr: quantstack_strategy_generation_7d == 0
  for: 0m
  labels:
    severity: info
  annotations:
    summary: "No new strategies generated in the last 7 days"

- orgId: 1
  name: Research queue backlog
  condition: C
  data:
    - refId: A
      datasourceUid: prometheus
      model:
        expr: quantstack_research_queue_depth > 50
  for: 5m
  labels:
    severity: warning
  annotations:
    summary: "Research queue depth is {{ $value }} (threshold: 50)"
```

---

## Key Files Summary

| File | Action | What |
|------|--------|------|
| `src/quantstack/graphs/supervisor/nodes.py` | Modify | Add `collect_health_metrics()`, call from `health_check` |
| `src/quantstack/observability/metrics.py` | Modify | Add 4 new Prometheus gauges + setter functions |
| `config/grafana/provisioning/alerting/alerts.yaml` | Modify | Add 4 health-based PromQL alert rules |
| `tests/unit/test_health_metrics.py` | Create | Unit tests for metrics collection and gauge updates |

---

## What Is Explicitly Out of Scope

- **IC (information coefficient) trend monitoring** -- deferred. Computing rolling IC requires specifying lookback windows, sample size requirements, and re-enablement criteria. These are research decisions, not ops decisions.
- **Grafana dashboard panels for health metrics** -- the dashboard is defined in Section 06. Health metrics will automatically appear in Prometheus and can be queried in Grafana, but adding specific dashboard panels is a Section 06 concern.
- **Alerting infrastructure (Discord contact points, Grafana provisioning setup)** -- defined in Sections 05 and 06. This section only adds alert rule expressions to the existing provisioned alerting YAML.
