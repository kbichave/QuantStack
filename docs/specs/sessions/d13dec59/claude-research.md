# Research Findings — Dashboard v2

## Part 1: Codebase Analysis

### Current Dashboard Implementation

**Location:** `scripts/dashboard.py` (~796 lines)
**Wrapper:** `status.sh` delegates to `python scripts/dashboard.py --watch`

**Architecture:**
- Rich library (no Textual yet) — `Console`, `Panel`, `Table`, `Text`, `Layout`, `Columns`, `Live`
- Two modes: print-once (`dashboard.py`) and live-refresh (`--watch`, 10s interval via `--interval`)
- Uses `Rich.live.Live` with `screen=True` for full-screen rendering
- All queries wrapped in try-except with graceful degradation (empty lists, "n/a" on failure)

**Current Sections (8):**
1. Header bar (kill switch, trading mode, AV usage, regime, universe)
2. Services panel (Docker health, graph cycles, heartbeats)
3. Portfolio (equity, cash, PnL, positions with unrealized PnL)
4. Recent trades (last 5 closed trades)
5. Strategies (status breakdown, Sharpe ratios, signals)
6. Research queue (WIP, pending, ML experiments, alpha programs, bugs)
7. Data health (fresh/stale symbol counts, upcoming earnings)
8. Decisions & agent activity (decisions, reflections, agent events by team)

### Database Schema (Key Tables)

**Portfolio/Execution:**
- `positions` — open positions (symbol, qty, avg_cost, current_price, unrealized_pnl, side, strategy_id)
- `closed_trades` — history (symbol, side, realized_pnl, holding_days, strategy_id, exit_reason, closed_at)
- `daily_equity` — NAV snapshots (date, cash, positions_value, total_equity, daily_pnl, daily_return_pct)
- `cash_balance` — current cash
- `fills` — execution records (order_id, slippage_bps, commission)

**Strategies & Research:**
- `strategies` — registered strategies (name, status, symbol, instrument_type, time_horizon, backtest_summary JSONB)
- `strategy_outcomes` — per-trade outcomes (strategy_id, entry/exit prices, realized_pnl_pct)
- `strategy_daily_pnl` — daily roll-up (date, strategy_id, realized/unrealized pnl, win/loss counts)
- `research_queue` — pending tasks (type, priority, topic, status, context_json)
- `research_wip` — active work (symbol, domain, agent_id, heartbeat_at)
- `ml_experiments` — training records (experiment_id, symbol, model_type, test_auc, cv_auc, verdict)
- `alpha_research_program` — investigations (thesis, status, experiments_run, best_oos_sharpe)

**Signals & Decisions:**
- `signal_state` — active signals (symbol, action, confidence, position_size_pct, generated_at, expires_at)
- `decision_events` — audit trail (event_type, agent_name, symbol, action, confidence, risk_approved)
- `agent_events` — real-time feed (graph_name, node_name, agent_name, event_type, content, created_at)
- `trade_reflections` — post-trade analysis (symbol, strategy_id, realized_pnl_pct, lesson)

**System State:**
- `system_state` — key-value store (kill_switch, av_daily_calls, etc.)
- `regime_states` — market regime (symbol, trend_regime, volatility_regime, confidence)
- `loop_heartbeats` — loop health (loop_name, iteration, started_at, finished_at, status)
- `graph_checkpoints` — cycle tracking (graph_name, cycle_number, duration_seconds, status)
- `loop_iteration_context` — stateless context (loop_name, context_key, context_json)

**Market Data:**
- `ohlcv` — candles (symbol, timeframe, timestamp, OHLCV)
- `earnings_calendar` — upcoming earnings (symbol, report_date, estimate, reported_eps, surprise_pct)
- `company_overview` — fundamentals (symbol, name, sector, industry, market_cap, last_updated)
- `news_sentiment` — (time_published, title, ticker, overall_sentiment_score)
- `options_chains` — (contract_id, underlying, expiry, strike, bid/ask, iv, greeks)
- `insider_trades` — (ticker, transaction_date, owner_name, shares, price_per_share)
- `macro_indicators` — (indicator, date, value)

**Operations:**
- `bugs` — tool failure tracking (tool_name, error_fingerprint, status, consecutive_errors)
- `agent_skills` — calibration (agent_id, prediction_count, correct_predictions, signal_count, winning_signals)
- `calibration_records` — (agent_name, stated_confidence, was_correct, pnl)
- `prompt_versions` — A/B testing (node_name, version, prompt_text, status, avg_pnl_since)
- `equity_alerts` — (symbol, action, strategy_name, debate_verdict, status)
- `risk_snapshots` — (snapshot_time, total_equity, gross_exposure, net_exposure, largest_position_pct, var_95, cvar_99)

### DB Connection Pattern

**Location:** `src/quantstack/db.py`
- Thread-safe `ThreadedConnectionPool` (2-10 connections, configurable via `PG_POOL_MAX`)
- Context managers: `pg_conn()` / `db_conn()` for auto commit/rollback
- Retry logic for broken connections (idle timeout recovery)
- Session timeout: `idle_in_transaction_session_timeout = 30s`
- Default DSN: `postgresql://localhost/quantstack`
- Docker-aware: converts localhost/docker hostnames automatically

### Testing Infrastructure

- **Framework:** pytest (7.4.0+) with pytest-asyncio (auto mode), pytest-cov
- **Config:** `pyproject.toml` — `testpaths = ["tests", "src/quantstack/core/tests"]`
- **Fixtures:** `tests/unit/conftest.py` — OHLCV generators (uptrend, downtrend, V/W shapes), mock settings
- **Pattern:** DataFrame assertions, mock settings for env isolation, no live API calls in unit tests
- **Coverage omissions:** External APIs, WebSocket, GPU, live execution, visualization

### Rich/Textual Status

- Rich is used extensively (`Console`, `Panel`, `Table`, `Text`, `Layout`, `Columns`, `Live`)
- **No Textual dependency** yet — migration to Textual is part of this spec
- No existing Textual usage anywhere in the codebase

---

## Part 2: Web Research — Textual TUI Framework

### Tabbed Applications

Use `TabbedContent` widget — the primary mechanism for tab-based navigation:

```python
# Method 1: Positional args
TabbedContent(
    ("Tab 1", content_widget_1),
    ("Tab 2", content_widget_2),
)

# Method 2: TabPane wrappers (more control)
TabbedContent(
    TabPane("Title", id="pane_id", children=[...]),
    TabPane("Other", id="other_id", children=[...]),
)
```

**Key features:**
- `TabbedContent.active` reactive attribute for programmatic switching
- `initial` parameter for default tab
- Dynamic tab management: `add_pane()`, `remove_pane()`, `enable_tab()`, `disable_tab()`
- CSS targeting via `--content-tab-{pane_id}`

### App Structure Best Practice

**Recommended hierarchy:**
```
App (root)
├─ Screen: Main View
│  ├─ Header (docked top)
│  ├─ TabbedContent
│  │  ├─ TabPane: Overview
│  │  ├─ TabPane: Portfolio
│  │  └─ TabPane: Research
│  └─ Footer (docked bottom)
├─ ModalScreen: Detail overlays (drill-down)
└─ Modes: for independent screen stacks if needed
```

- **Screens** for distinct app states with push/pop navigation
- **ModalScreen** for drill-down overlays (prevents app keybindings, shows through previous screen)
- **TabbedContent** for peer-level content within a single screen
- **messages-up, attributes-down** pattern for widget communication

### Periodic Refresh

Use `set_interval()` + `@work` decorated async methods:

```python
def on_mount(self) -> None:
    self.set_interval(self._refresh_services, interval=5.0)
    self.set_interval(self._refresh_portfolio, interval=15.0)
    self.set_interval(self._refresh_data_health, interval=60.0)
    self.set_interval(self._refresh_research, interval=120.0)

@work(exclusive=True)
async def _refresh_services(self) -> None:
    data = await self.fetch_service_health()
    self.update_services_widget(data)
```

**Key worker options:**
- `exclusive=True` — cancels previous workers (prevents out-of-order)
- `exit_on_error=False` — don't crash app on exception
- `thread=True` — for blocking operations (call `App.call_from_thread()` for UI)

### Tab-Specific Refresh

Only refresh visible tab's data:

```python
async def on_tabbed_content_active_changed(self, event) -> None:
    if event.pane.id == "portfolio":
        await self._refresh_portfolio()
```

### Keybindings

```python
BINDINGS = [
    ("1", "switch_tab('overview')", "Overview"),
    ("q", "quit", "Quit"),
    ("r", "refresh_all", "Refresh"),
    ("?", "show_help", "Help"),
    ("/", "search", "Search"),
    ("j", "scroll_down", "Down"),
    ("k", "scroll_up", "Up"),
    ("enter", "expand_detail", "Detail"),
]
```

Actions can be async, dynamically enabled/disabled via `check_action()`.

### Modal Overlays (Drill-down)

```python
class DetailScreen(ModalScreen):
    BINDINGS = [("escape", "cancel", "Close")]

    def compose(self):
        with Container(id="dialog"):
            yield Static("Detail view content")

    def action_cancel(self) -> None:
        self.dismiss()
```

### Reactive Data Binding

```python
class MarketWidget(Static):
    price = reactive(0.0)

    def watch_price(self, old: float, new: float) -> None:
        self.styles.color = "green" if new > old else "red"

    def render(self) -> str:
        return f"${self.price:.2f}"
```

---

## Part 3: Web Research — ASCII Chart Rendering

### Library Options

| Library | Strength | Use Case |
|---------|----------|----------|
| **Plotext** | Full charts (line, bar, candlestick, heatmap) | Equity curves, detailed charts |
| **Rich** | Tables, progress bars, panels | Structural elements (already used) |
| **asciichartpy** | Lightweight sparklines | Inline mini-charts |

**Recommendation:** Custom Unicode rendering for sparklines/heatmaps (zero dependency), Plotext only if candlestick charts needed.

### Unicode Block Sparklines

```python
def sparkline(data: list[float], width: int = 20) -> str:
    blocks = "▁▂▃▄▅▆▇█"
    if not data:
        return ""
    mn, mx = min(data), max(data)
    rng = mx - mn or 1
    if len(data) > width:
        step = len(data) / width
        data = [data[int(i * step)] for i in range(width)]
    return "".join(blocks[min(7, int(((v - mn) / rng) * 7))] for v in data)
```

### Braille Characters (2x4 dot grid, 256 patterns)

Higher resolution than block characters. Good for correlation matrices, heatmaps, volatility surfaces. Each char = 2 columns x 4 rows of dots.

### Integration with Textual

Rich renderables (Table, Panel, Text) work directly inside Textual's `Static.render()`. Custom sparklines and charts can be returned as `Text` objects with color markup.

---

## Part 4: Web Research — Tiered Async Data Refresh

### Recommended Tier Structure

| Tier | Interval | Sections | Queries |
|------|----------|----------|---------|
| T1 (5s) | 5s | Services, agent events | ~5 |
| T2 (15s) | 15s | Portfolio, signals, positions | ~8 |
| T3 (60s) | 60s | Data health, strategies, risk | ~20 |
| T4 (120s) | 120s | Research, ML, calibration | ~12 |

### Staggered Start (Anti-Thundering-Herd)

```python
def on_mount(self) -> None:
    self.set_interval(self._refresh_t1, interval=5.0)
    self.set_timer(0.3, lambda: self.set_interval(self._refresh_t2, interval=15.0))
    self.set_timer(0.6, lambda: self.set_interval(self._refresh_t3, interval=60.0))
    self.set_timer(0.9, lambda: self.set_interval(self._refresh_t4, interval=120.0))
```

### Connection Pool Sizing

Current pool: 2-10 (`PG_POOL_MAX`). With 45 queries staggered across tiers:
- At most ~8 concurrent queries (T1 + T2 overlap worst case)
- Pool size of 5-8 should suffice
- Use `asyncio.Semaphore(5)` as safety valve

### Visibility-Aware Refresh

- Always-on queries: header bar (kill switch, regime, AV count) — run on every tier
- Per-tab queries: only run when tab is active
- On tab switch: immediate refresh of new tab's data
- Cache per-tab data with staleness tracking

### DB Pattern for Dashboard

Use existing `pg_conn()` context manager from `db.py`. Run queries in `@work(thread=True)` since psycopg2 is blocking:

```python
@work(thread=True, exclusive=True)
def fetch_portfolio(self) -> dict:
    with pg_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT * FROM positions ORDER BY unrealized_pnl DESC")
        return cur.fetchall()
```

---

## Testing Considerations

### Existing Setup
- pytest with asyncio, conftest fixtures for synthetic data
- No existing dashboard tests (dashboard is in `scripts/`, outside test coverage)

### Recommended Approach for Dashboard v2
- **Unit test widgets** with Textual's `pilot` testing framework (`async with app.run_test() as pilot`)
- **Mock DB layer** — inject query results, don't require live PostgreSQL
- **Test refresh logic** — verify correct tier intervals, tab-specific queries
- **Test keybindings** — `pilot.press("1")` to verify tab switching
- **Snapshot testing** — Textual supports SVG/text snapshots for visual regression
