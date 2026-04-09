# Section 06: Health Dashboard

## Objective

Build a unified health dashboard that aggregates status from all subsystems (loops, authority, reconciliation, signals, agents) into a single queryable view for the supervisor graph and status script.

**Depends on:** section-03-loop-verifier, section-04-authority-matrix, section-05-reconciler

## Files to Create

### `src/quantstack/autonomous/health_dashboard.py`

Unified health aggregation.

## Files to Modify

### `status.sh`

Add dashboard output sections for portfolio, signals, agents, research, and system health.

### `src/quantstack/autonomous/__init__.py`

Export `HealthDashboard`, `SystemHealth`.

## Implementation Details

### HealthDashboard Class

```python
class HealthDashboard:
    def __init__(
        self,
        loop_verifier: LoopVerifier,
        authority_matrix: AuthorityMatrix,
        reconciler: PositionReconciler,
    ): ...

    async def get_full_health(self) -> SystemHealth: ...
    async def get_portfolio_health(self) -> PortfolioHealth: ...
    async def get_signal_health(self) -> SignalHealth: ...
    async def get_agent_health(self) -> AgentHealth: ...
    async def get_research_health(self) -> ResearchHealth: ...
    async def get_system_health(self) -> InfraHealth: ...
```

### SystemHealth Dataclass

Top-level container with sub-sections:

```python
@dataclass
class SystemHealth:
    timestamp: datetime
    overall_status: Literal["healthy", "degraded", "critical"]
    portfolio: PortfolioHealth
    signals: SignalHealth
    agents: AgentHealth
    research: ResearchHealth
    infrastructure: InfraHealth
    feedback_loops: dict[str, LoopHealth]
    last_reconciliation: ReconciliationReport | None
```

`overall_status` logic:
- `critical`: any kill switch active, OR reconciliation mismatch unresolved, OR >2 loops broken
- `degraded`: 1-2 loops stale/broken, OR any agent win rate < 30%, OR data staleness detected
- `healthy`: everything else

### PortfolioHealth

- `positions_count: int`
- `unrealized_pnl: float`
- `greeks_exposure: dict` (delta, gamma, vega, theta — from positions table)
- `margin_utilization_pct: float`
- `max_single_position_pct: float` (largest position as % of portfolio)

### SignalHealth

- `per_collector_ic: dict[str, float]` — latest IC per signal collector
- `stale_collectors: list[str]` — collectors with data older than threshold
- `synthesis_weights: dict[str, float]` — current signal weights

### AgentHealth

- `per_agent_quality: dict[str, float]` — quality score per agent
- `per_agent_win_rate: dict[str, float]`
- `per_agent_last_cycle: dict[str, datetime]`
- `underperforming_agents: list[str]` — win rate < 40%

### ResearchHealth

- `pipeline_depth: int` — items in research queue
- `hypothesis_velocity_7d: float` — hypotheses generated per day (7-day avg)
- `strategy_counts: dict[str, int]` — count by lifecycle stage (paper, forward_testing, live, retired)

### InfraHealth

- `db_connected: bool`
- `api_rate_limits: dict[str, dict]` — per-API remaining calls
- `container_status: dict[str, str]` — per-container up/down (from Docker API or health endpoints)
- `disk_usage_pct: float`

### Data Sources

All health metrics come from existing DB tables and APIs — no new data collection needed:
- Portfolio: `positions` table, broker API
- Signals: `signal_ic` table, `signal_weights` table
- Agents: `agent_metrics` table
- Research: `research_queue` table, `strategies` table
- Infrastructure: DB connection test, API calls, Docker socket

### status.sh Integration

Add a `--dashboard` flag to `status.sh` that calls `python -c "from quantstack.autonomous.health_dashboard import ...; ..."` and pretty-prints the result.

## Test Requirements

- `tests/unit/autonomous/test_health_dashboard.py`:
  - Mock all sub-components, verify `get_full_health()` assembles correctly
  - Test `overall_status` classification: healthy, degraded, critical scenarios
  - Test that a single broken component does not crash the full dashboard (graceful degradation)
  - Test with empty/missing data (new system with no trades yet)

## Acceptance Criteria

1. `get_full_health()` returns a complete `SystemHealth` in under 10 seconds
2. Overall status correctly reflects the worst-case across all sub-sections
3. Partial failures in one sub-section do not prevent other sections from reporting
4. All data comes from existing tables — no new data collection infrastructure
5. Dashboard is usable by both supervisor graph (programmatic) and status.sh (human-readable)
