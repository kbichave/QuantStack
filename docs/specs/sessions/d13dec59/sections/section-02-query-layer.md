# Section 2: Query Layer

## Goal

Create a typed query module at `src/quantstack/tui/queries/` that encapsulates all 45 database queries the dashboard needs. Every function returns a dataclass (or list of dataclasses) instead of raw tuples, and every function degrades gracefully on error by returning a sensible default (empty list, None, zero).

## Dependencies

- **Section 1 (Package Scaffolding)** must be complete: the `src/quantstack/tui/` package and `queries/__init__.py` must exist.
- No dependency on Section 3 (Charts) or any widget sections.

## Design Principles

1. **Query functions do not manage connections.** Callers (widget `fetch_data()` methods) acquire connections via `pg_conn()` context managers from `quantstack.db`. Query functions accept a `PgConnection` instance as their first argument.

2. **Use `PgConnection.execute()` / `.fetchall()` / `.fetchone()`.** Never create raw cursors. `PgConnection` provides retry logic for broken connections and automatic rollback on errors.

3. **Every function returns a typed dataclass or list of dataclasses.** Define the dataclass in the same file as the query that returns it.

4. **Every function wraps its body in try/except, logs the error (with the query function name, not raw SQL), and returns the default value.** This preserves the v1 graceful degradation pattern where a missing table or broken connection never crashes the dashboard.

5. **Placeholder parameter style:** use `%s` (psycopg2 native). The `PgConnection._translate()` method converts `?` to `%s`, so either works, but prefer `%s` directly for clarity.

## Connection Pattern (from `quantstack.db`)

The existing `pg_conn()` context manager returns a `PgConnection`:

```python
from quantstack.db import pg_conn

with pg_conn() as conn:
    conn.execute("SELECT * FROM strategies WHERE status = %s", ("live",))
    rows = conn.fetchall()
```

`PgConnection` key methods: `.execute(sql, params)`, `.fetchone()`, `.fetchall()`, `.fetchdf()`. It handles broken-connection retry internally. Autocommit is off; `pg_conn()` commits on clean exit and rolls back on exception.

**Dashboard query functions receive an already-acquired `PgConnection`** — they do NOT call `pg_conn()` themselves. This is because widgets call multiple query functions within a single `with pg_conn() as conn:` block in their `fetch_data()` method.

## Error Handling Pattern

Every query function follows this structure:

```python
def fetch_example(conn: PgConnection) -> list[ExampleRow]:
    """Fetch examples. Returns empty list on any error."""
    try:
        conn.execute("SELECT col1, col2 FROM example ORDER BY col1")
        rows = conn.fetchall()
        return [ExampleRow(col1=r[0], col2=r[1]) for r in rows]
    except Exception:
        logger.warning("fetch_example failed", exc_info=True)
        return []
```

The `logger` is from `loguru` (`from loguru import logger`), consistent with the rest of the codebase.

## Porting from v1

The existing `scripts/dashboard.py` uses a `_query(sql)` helper that opens a fresh psycopg2 connection per call. Many of its SQL statements can be reused. Key queries to port:

- Kill switch: `SELECT value FROM system_state WHERE key = 'kill_switch'`
- AV calls: `SELECT value FROM system_state WHERE key = %s` (today's date key)
- Regime: `SELECT symbol, trend_regime, volatility_regime, confidence FROM regimes ...`
- Positions: `SELECT symbol, quantity, avg_cost, current_price, unrealized_pnl, ... FROM positions`
- Equity: `SELECT total_equity, daily_pnl, daily_return_pct, cash FROM daily_equity ...`
- Closed trades: `SELECT symbol, side, realized_pnl, holding_days, strategy_id, ... FROM closed_trades`
- Strategies: `SELECT name, status, symbol, instrument_type, time_horizon, ... FROM strategies`
- Research queue/WIP, ML experiments, bugs, alpha programs
- Signal state, reflections, decision events, agent events

New queries (not in v1) are listed per file below.

---

## Tests (write BEFORE implementing)

**File:** `tests/unit/test_tui/test_queries.py`

```python
"""Tests for src/quantstack/tui/queries/ — all 45 dashboard query functions.

Each test patches pg_conn() to return a mock PgConnection, verifies the function
returns the correct dataclass type, and verifies graceful degradation (returns
default value) when the connection raises.
"""
import pytest
from unittest.mock import MagicMock
from quantstack.db import PgConnection


def _mock_conn(rows=None, scalar=None):
    """Build a mock PgConnection that returns the given rows on fetchall()
    or scalar on fetchone()."""
    conn = MagicMock(spec=PgConnection)
    conn.execute.return_value = conn
    conn.fetchall.return_value = rows or []
    conn.fetchone.return_value = (scalar,) if scalar is not None else None
    return conn


def _failing_conn():
    """Build a mock PgConnection whose execute() raises."""
    conn = MagicMock(spec=PgConnection)
    conn.execute.side_effect = Exception("connection lost")
    return conn


# -- System queries (queries/system.py) --

# Test: fetch_kill_switch returns bool, default False on error
# Test: fetch_av_calls returns int, default 0 on error
# Test: fetch_regime returns RegimeState dataclass with trend/vol/confidence
# Test: fetch_graph_checkpoints returns list of GraphCheckpoint dataclasses
# Test: fetch_heartbeats returns list of Heartbeat dataclasses
# Test: fetch_agent_events returns list of AgentEvent, LIMIT 60, ordered DESC

# -- Portfolio queries (queries/portfolio.py) --

# Test: fetch_equity_summary returns EquitySummary dataclass
# Test: fetch_positions returns list of Position dataclasses, ordered by unrealized_pnl DESC
# Test: fetch_closed_trades returns list of ClosedTrade, LIMIT 10
# Test: fetch_equity_curve returns list of EquityPoint (30 rows)
# Test: fetch_benchmark returns list of BenchmarkPoint (30 rows for SPY)
# Test: fetch_pnl_by_strategy returns list of StrategyPnl dataclasses
# Test: fetch_pnl_by_symbol returns list of SymbolPnl dataclasses

# -- Strategy queries (queries/strategies.py) --

# Test: fetch_strategy_pipeline returns list of StrategyCard with fwd_trades/fwd_pnl computed
# Test: strategies are ordered by status priority (live > forward > backtested > draft > retired)

# -- Data health queries (queries/data_health.py) --

# Test: fetch_ohlcv_freshness returns dict[symbol, datetime]
# Test: fetch_news_freshness returns dict[symbol, datetime]
# Test: each freshness query returns empty dict on error
# Test: fetch_collector_health returns dict[collector_name, bool]

# -- Signal queries (queries/signals.py) --

# Test: fetch_active_signals returns list of Signal sorted by confidence DESC
# Test: fetch_signal_brief parses brief_json JSONB correctly

# -- Risk queries (queries/risk.py) --

# Test: fetch_risk_snapshot returns RiskSnapshot or None if table empty
# Test: fetch_equity_alerts returns list of EquityAlert with status

# -- All queries: error handling --

# Test: every query function returns its default value when conn.execute() raises
# Test: every query function uses PgConnection.execute() (not raw cursors)
```

The test helpers `_mock_conn` and `_failing_conn` provide the two scenarios every query function must handle. Each individual test should:
1. Call the query function with `_mock_conn(rows=[...])` containing sample data matching the SQL columns.
2. Assert the return is the correct dataclass type with correct field values.
3. Call the query function with `_failing_conn()` and assert it returns the documented default.

---

## File-by-File Specification

### `src/quantstack/tui/queries/__init__.py`

Exports all query functions for convenient imports. May also define shared types used across multiple query files (if any).

### `src/quantstack/tui/queries/system.py`

**Dataclasses:**

```python
@dataclass
class RegimeState:
    symbol: str
    trend: str        # trending_up, trending_down, ranging, unknown
    volatility: str   # low, normal, high
    confidence: float

@dataclass
class GraphCheckpoint:
    graph_name: str
    node_name: str
    cycle_number: int
    started_at: datetime
    duration_seconds: float | None

@dataclass
class Heartbeat:
    loop_name: str
    last_beat: datetime
    status: str

@dataclass
class AgentEvent:
    graph_name: str
    node_name: str
    agent_name: str
    event_type: str
    content: str
    created_at: datetime
```

**Functions:**

- `fetch_kill_switch(conn) -> bool` — Queries `system_state` WHERE key = 'kill_switch'. Returns True if value is 'active', False otherwise. Default: False.
- `fetch_av_calls(conn) -> int` — Queries `system_state` WHERE key = today's AV counter key (format: `av_calls_YYYY-MM-DD`). Returns int count. Default: 0.
- `fetch_regime(conn) -> RegimeState | None` — Queries latest row from `regimes` (or `loop_iteration_context` JSON fallback as v1 does). Default: None.
- `fetch_graph_checkpoints(conn) -> list[GraphCheckpoint]` — Queries `graph_checkpoints` DISTINCT ON (graph_name) ORDER BY started_at DESC. Default: [].
- `fetch_heartbeats(conn) -> list[Heartbeat]` — Queries `heartbeats` or `loop_heartbeats` DISTINCT ON (loop_name) ORDER BY last_beat DESC. Default: [].
- `fetch_agent_events(conn, limit=60) -> list[AgentEvent]` — Queries `agent_events` ORDER BY created_at DESC LIMIT 60. Default: [].

### `src/quantstack/tui/queries/portfolio.py`

**Dataclasses:**

```python
@dataclass
class EquitySummary:
    total_equity: float
    cash: float
    daily_pnl: float
    daily_return_pct: float
    high_water: float
    drawdown_pct: float

@dataclass
class Position:
    symbol: str
    quantity: float
    avg_cost: float
    current_price: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    strategy_id: str
    holding_days: int

@dataclass
class ClosedTrade:
    symbol: str
    side: str
    realized_pnl: float
    holding_days: int
    strategy_id: str
    exit_reason: str
    closed_at: datetime

@dataclass
class EquityPoint:
    date: date
    equity: float

@dataclass
class BenchmarkPoint:
    date: date
    symbol: str
    close: float
    daily_return_pct: float

@dataclass
class StrategyPnl:
    strategy_id: str
    strategy_name: str
    realized_pnl: float
    unrealized_pnl: float
    win_count: int
    loss_count: int
    sharpe: float | None

@dataclass
class SymbolPnl:
    symbol: str
    total_pnl: float
```

**Functions:**

- `fetch_equity_summary(conn) -> EquitySummary | None` — Latest row from `daily_equity`, augmented with cash from `cash_balance`. Computes drawdown from high-water mark. Default: None.
- `fetch_positions(conn) -> list[Position]` — From `positions` WHERE quantity != 0, ORDER BY unrealized_pnl DESC. Default: [].
- `fetch_closed_trades(conn, limit=10) -> list[ClosedTrade]` — From `closed_trades` ORDER BY closed_at DESC LIMIT 10. Default: [].
- `fetch_equity_curve(conn, days=30) -> list[EquityPoint]` — From `daily_equity` last N rows ORDER BY date. Default: [].
- `fetch_benchmark(conn, symbol="SPY", days=30) -> list[BenchmarkPoint]` — From `benchmark_daily` WHERE symbol = %s, last N rows ORDER BY date. Default: [].
- `fetch_pnl_by_strategy(conn) -> list[StrategyPnl]` — Joins `strategies` with aggregated `closed_trades` + `positions`. Default: [].
- `fetch_pnl_by_symbol(conn) -> list[SymbolPnl]` — Aggregated P&L (realized + unrealized) grouped by symbol, ORDER BY total_pnl DESC. Default: [].

### `src/quantstack/tui/queries/strategies.py`

**Dataclasses:**

```python
@dataclass
class StrategyCard:
    strategy_id: str
    name: str
    status: str          # draft, backtested, forward_testing, live, retired
    symbol: str
    instrument_type: str
    time_horizon: str
    sharpe: float | None
    max_drawdown: float | None
    win_rate: float | None
    fwd_trades: int
    fwd_pnl: float
    fwd_days: int
    fwd_required_days: int
```

**Functions:**

- `fetch_strategy_pipeline(conn) -> list[StrategyCard]` — Single query joining `strategies` with aggregated `closed_trades` to compute forward testing stats (trade count, P&L, win rate since `updated_at`). Ordered by status priority: live=0, forward_testing=1, backtested=2, draft=3, retired=4. Default: [].

The status ordering uses a `CASE` expression in the SQL `ORDER BY` clause.

### `src/quantstack/tui/queries/data_health.py`

**Dataclasses:**

```python
@dataclass
class CollectorHealth:
    collector_name: str
    is_healthy: bool
    last_success: datetime | None
    failure_count: int
```

**Functions (7 freshness queries + 1 collector health):**

- `fetch_ohlcv_freshness(conn) -> dict[str, datetime]` — `SELECT symbol, MAX(timestamp) FROM ohlcv WHERE timeframe = '1D' GROUP BY symbol`. Default: {}.
- `fetch_news_freshness(conn) -> dict[str, datetime]` — `SELECT symbol, MAX(published_at) FROM news GROUP BY symbol`. Default: {}.
- `fetch_sentiment_freshness(conn) -> dict[str, datetime]` — From `sentiment_scores` or equivalent table. Default: {}.
- `fetch_fundamentals_freshness(conn) -> dict[str, datetime]` — From `fundamentals`, MAX of fetched_at or updated_at. Default: {}.
- `fetch_options_freshness(conn) -> dict[str, datetime]` — From `options_flow` or `options_chain`. Default: {}.
- `fetch_insider_freshness(conn) -> dict[str, datetime]` — From `insider_trades`. Default: {}.
- `fetch_macro_freshness(conn) -> dict[str, datetime]` — From `macro_indicators`. Default: {}.
- `fetch_collector_health(conn) -> list[CollectorHealth]` — Iterates `signal_state.brief_json` across symbols, counts `collector_failures` per collector name. Default: [].

Each freshness function returns a `dict[str, datetime]` mapping symbol to its most recent timestamp for that data type. Empty dict on error.

### `src/quantstack/tui/queries/signals.py`

**Dataclasses:**

```python
@dataclass
class Signal:
    symbol: str
    action: str          # BUY, SELL, HOLD
    confidence: float
    position_size_pct: float
    generated_at: datetime
    factors: dict        # parsed from brief_json: {ml: float, sentiment: float, ...}

@dataclass
class SignalBrief:
    symbol: str
    action: str
    confidence: float
    ml_score: float | None
    sentiment_score: float | None
    technical_score: float | None
    options_score: float | None
    macro_score: float | None
    risk_flags: list[str]
    collector_failures: list[str]
    generated_at: datetime
```

**Functions:**

- `fetch_active_signals(conn) -> list[Signal]` — From `signal_state` ORDER BY confidence DESC. Parses `brief_json` JSONB for factor breakdown. Default: [].
- `fetch_signal_brief(conn, symbol: str) -> SignalBrief | None` — Full signal brief for a single symbol from `signal_state`. Parses nested JSONB fields for per-factor scores, risk flags, and collector failures. Default: None.

**JSONB parsing note:** `brief_json` is stored as JSONB in PostgreSQL. Due to the `register_default_jsonb(loads=lambda x: x)` override in `db.py`, the column is returned as a raw string. Use `json.loads()` to parse it. Handle malformed JSON by returning the default.

### `src/quantstack/tui/queries/calendar.py`

**Dataclasses:**

```python
@dataclass
class MarketHoliday:
    date: date
    name: str
    market_status: str   # closed, early_close
    close_time: time | None

@dataclass
class EarningsEvent:
    symbol: str
    report_date: date
    report_time: str | None  # BMO, AMC

@dataclass
class MacroEvent:
    date: date
    name: str
    category: str        # FOMC, CPI, PPI, etc.
```

**Functions:**

- `fetch_market_holidays(conn, days_ahead=90) -> list[MarketHoliday]` — From `market_holidays` WHERE date >= CURRENT_DATE AND date <= CURRENT_DATE + interval. ORDER BY date. Default: [].
- `fetch_earnings_calendar(conn, days_ahead=90) -> list[EarningsEvent]` — From `earnings_calendar` for universe symbols, next N days. Default: [].
- `fetch_macro_events(conn, days_ahead=90) -> list[MacroEvent]` — From `macro_indicators` or a dedicated events table. FOMC dates, CPI/PPI releases. Default: [].

Note: `market_holidays` table is created by Section 12 (DB Migrations). If the table does not exist, the query will raise and the function returns [].

### `src/quantstack/tui/queries/agents.py`

**Dataclasses:**

```python
@dataclass
class GraphActivity:
    graph_name: str
    current_node: str
    current_agent: str
    cycle_number: int
    cycle_started: datetime
    event_count: int

@dataclass
class CycleHistory:
    graph_name: str
    cycle_number: int
    duration_seconds: float
    primary_agent: str
    tool_count: int

@dataclass
class AgentSkill:
    agent_name: str
    accuracy: float | None
    win_rate: float | None
    avg_pnl: float | None
    information_coefficient: float | None
    trend: str           # improving, stable, declining

@dataclass
class CalibrationRecord:
    agent_name: str
    stated_confidence: float
    actual_win_rate: float
    is_overconfident: bool

@dataclass
class PromptVersion:
    agent_name: str
    version: int
    optimized_at: datetime
    active_candidates: int
```

**Functions:**

- `fetch_graph_activity(conn) -> list[GraphActivity]` — Current state per graph from `graph_checkpoints` + `agent_events` count. Default: [].
- `fetch_cycle_history(conn, limit=3) -> list[CycleHistory]` — Last N completed cycles per graph from `graph_checkpoints`. Default: [].
- `fetch_agent_skills(conn) -> list[AgentSkill]` — From `agent_skills` table. Degrades to [] if table empty or missing.
- `fetch_calibration(conn) -> list[CalibrationRecord]` — From `calibration_records`. Computes `is_overconfident = stated_confidence > actual_win_rate + 0.1`. Degrades to [].
- `fetch_prompt_versions(conn) -> list[PromptVersion]` — From `prompt_versions`. Degrades to [].

### `src/quantstack/tui/queries/research.py`

**Dataclasses:**

```python
@dataclass
class ResearchWip:
    symbol: str
    domain: str
    agent_id: str
    started_at: datetime
    duration_minutes: float

@dataclass
class ResearchQueueItem:
    task_type: str
    status: str
    topic: str
    priority: int

@dataclass
class MlExperiment:
    id: int
    created_at: datetime
    model_type: str
    symbol: str
    test_auc: float | None
    sharpe: float | None
    feature_count: int
    verdict: str

@dataclass
class AlphaProgram:
    name: str
    status: str
    progress_pct: float
    findings_summary: str

@dataclass
class Breakthrough:
    feature_name: str
    importance: float

@dataclass
class TradeReflection:
    symbol: str
    realized_pnl_pct: float
    lesson: str
    created_at: datetime

@dataclass
class BugRecord:
    id: int
    tool_name: str
    status: str           # open, in_progress, resolved
    error_summary: str
    created_at: datetime

@dataclass
class ConceptDrift:
    symbol: str
    recent_auc: float
    historical_auc: float
    drift_magnitude: float
```

**Functions:**

- `fetch_research_wip(conn) -> list[ResearchWip]` — From `research_wip`. Computes duration from `heartbeat_at` or `started_at`. Default: [].
- `fetch_research_queue(conn) -> list[ResearchQueueItem]` — From `research_queue` WHERE status = 'pending' ORDER BY priority. Default: [].
- `fetch_ml_experiments(conn, limit=10) -> list[MlExperiment]` — From `ml_experiments` ORDER BY created_at DESC LIMIT 10. Default: [].
- `fetch_alpha_programs(conn) -> list[AlphaProgram]` — From `alpha_research_program` WHERE status = 'active'. Default: [].
- `fetch_breakthroughs(conn) -> list[Breakthrough]` — From `breakthrough_features` ORDER BY importance DESC. Degrades to [] if table missing.
- `fetch_reflections(conn, limit=10) -> list[TradeReflection]` — From `trade_reflections` ORDER BY created_at DESC LIMIT 10. Default: [].
- `fetch_bugs(conn) -> list[BugRecord]` — From `bugs` WHERE status IN ('open', 'in_progress'). Default: [].
- `fetch_concept_drift(conn, window_days=14, threshold=0.05) -> list[ConceptDrift]` — Compares recent AUC (last N days) vs historical average per symbol from `ml_experiments`. Returns symbols where drift exceeds threshold. Default: [].

### `src/quantstack/tui/queries/risk.py`

**Dataclasses:**

```python
@dataclass
class RiskSnapshot:
    gross_exposure: float
    net_exposure: float
    concentration: float
    correlation: float
    sector_exposure: float
    var_1d: float
    max_drawdown: float
    snapshot_at: datetime

@dataclass
class RiskEvent:
    event_type: str       # risk_rejection, drawdown_alert, correlation_alert
    symbol: str | None
    details: str
    created_at: datetime

@dataclass
class EquityAlert:
    alert_id: int
    alert_type: str
    status: str           # active, cleared
    message: str
    created_at: datetime
    cleared_at: datetime | None
```

**Functions:**

- `fetch_risk_snapshot(conn) -> RiskSnapshot | None` — Latest row from `risk_snapshots`. Default: None (displayed as "No risk data available" by widget).
- `fetch_risk_events(conn, days=7) -> list[RiskEvent]` — From `decision_events` WHERE event_type IN ('risk_rejection', 'drawdown_alert', 'correlation_alert') AND created_at >= NOW() - interval. Default: [].
- `fetch_equity_alerts(conn) -> list[EquityAlert]` — From `equity_alerts` LEFT JOIN `alert_updates` to get clearance status. Default: [].

---

## Implementation Checklist

1. Create `src/quantstack/tui/queries/__init__.py` with re-exports.
2. Create `tests/unit/test_tui/__init__.py` (if not from Section 1) and `tests/unit/test_tui/test_queries.py` with test stubs per the test spec above.
3. Implement `queries/system.py` — 6 functions, 4 dataclasses.
4. Implement `queries/portfolio.py` — 7 functions, 7 dataclasses.
5. Implement `queries/strategies.py` — 1 function, 1 dataclass.
6. Implement `queries/data_health.py` — 8 functions, 1 dataclass.
7. Implement `queries/signals.py` — 2 functions, 2 dataclasses.
8. Implement `queries/calendar.py` — 3 functions, 3 dataclasses.
9. Implement `queries/agents.py` — 5 functions, 5 dataclasses.
10. Implement `queries/research.py` — 8 functions, 8 dataclasses.
11. Implement `queries/risk.py` — 3 functions, 3 dataclasses.
12. Run tests: `uv run pytest tests/unit/test_tui/test_queries.py -v`.
13. Verify all 45 query functions (6+7+1+8+2+3+5+8+3 = 43 listed, plus `fetch_signal_brief` and `fetch_concept_drift` = 45) return correct types and degrade gracefully.

## Key Invariants

- Query functions NEVER call `pg_conn()`. They receive a `PgConnection` argument.
- Query functions NEVER raise exceptions to callers. They catch everything and return defaults.
- Query functions NEVER mutate the database. This is a read-only dashboard.
- All imports are at module level (no deferred imports).
- Dataclasses use `from __future__ import annotations` for forward-reference support and `float | None` syntax.
