# QuantStack Dashboard v2 — Complete Specification

## Overview

Replace the existing Rich-based terminal dashboard (`scripts/dashboard.py`, 796 lines) with a Textual TUI application. The new dashboard lives in `src/quantstack/dashboard/` as a proper Python package. It provides 6 tabs, 13 sections, 45 queries (tiered refresh), tabbed navigation, drill-down modals, and curated data visualizations using custom Unicode rendering.

This is a **read-only observation tool**. No write capabilities. The old v1 dashboard is fully replaced (no --simple fallback).

---

## Key Decisions (from interview)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Code location | `src/quantstack/dashboard/` package | Testable, importable, proper structure |
| Rollout | All 6 tabs at once | Owner wants complete replacement |
| v1 compat | Replace v1 entirely | No --simple flag, no backward compat |
| Charting | Custom Unicode (block chars, braille) | Zero dependency, full Textual compat |
| DB access | `@work(thread=True)` with psycopg2 | Minimal migration, reuse existing `pg_conn()` |
| Missing tables | Create trivial ones, degrade for complex | `market_holidays` created; `risk_snapshots` degrades |
| Drill-downs | Curated summaries | Concise, not full data dumps |
| Interactivity | Strictly read-only | Observation only |

---

## Technology Stack

- **Textual** (>=0.50) — TUI framework (tabs, scrolling, keybindings, modals)
- **Rich** (>=13.0) — Rendering engine (already installed)
- **psycopg2** (>=2.9) — PostgreSQL (already installed, used via thread workers)
- **Custom Unicode** — Sparklines (▁▂▃▄▅▆▇█), heatmaps, bar charts

No new external services. No new Docker containers. No web server.

---

## Package Structure

```
src/quantstack/dashboard/
├── __init__.py
├── app.py              # TextualApp subclass, tab definitions, keybindings
├── __main__.py         # CLI entry point (python -m quantstack.dashboard)
├── widgets/
│   ├── __init__.py
│   ├── header.py       # Header bar widget
│   ├── services.py     # Services & graph health
│   ├── portfolio.py    # Portfolio summary + equity curve + positions
│   ├── trades.py       # Recent/closed trades table
│   ├── strategies.py   # Strategy pipeline kanban
│   ├── data_health.py  # Data health matrix
│   ├── calendar.py     # Market calendar & events
│   ├── signals.py      # Signal engine dashboard
│   ├── agents.py       # Graph activity feeds + agent scorecard
│   ├── research.py     # Research queue + ML lab + discoveries
│   ├── risk.py         # Risk console
│   ├── decisions.py    # Decision log / audit trail
│   ├── digest.py       # Daily digest summary
│   └── charts.py       # Unicode sparkline, heatmap, bar chart renderers
├── queries/
│   ├── __init__.py
│   ├── portfolio.py    # Portfolio/equity/trades queries
│   ├── strategies.py   # Strategy pipeline queries
│   ├── signals.py      # Signal state queries
│   ├── data_health.py  # Per-data-type freshness queries
│   ├── agents.py       # Agent events, skills, calibration queries
│   ├── research.py     # Research queue, ML, alpha program queries
│   ├── risk.py         # Risk snapshots, alerts queries
│   ├── system.py       # Kill switch, regime, AV calls, services
│   └── calendar.py     # Earnings, holidays, FOMC, macro events
├── screens/
│   ├── __init__.py
│   └── detail.py       # ModalScreen for drill-down views
└── refresh.py          # Tiered refresh scheduler
```

---

## Tab Layout

```
[Overview] [Portfolio] [Strategies] [Data & Signals] [Agents] [Research]
```

### Tab 1: Overview
Compact summaries from all other tabs on one screen. Fits 24-line terminal.
- Header bar (1 line)
- Services compact + Risk compact (3 lines each, side-by-side)
- Portfolio compact + Recent trades (side-by-side)
- Strategy pipeline counts + Top signals (side-by-side)
- Data health bars + Research compact (side-by-side)
- Agent activity (1 line per graph)
- Daily digest (after 17:00 ET)

### Tab 2: Portfolio
- Equity summary (equity, cash, exposure, drawdown)
- ASCII equity curve (30d, Unicode block chars)
- Benchmark comparison (vs SPY)
- Open positions table (scrollable)
- Closed trades (last 10)
- P&L attribution by strategy
- P&L attribution by symbol (horizontal bars)
- Daily P&L heatmap (weekday grid, color-coded)

### Tab 3: Strategies
- Kanban pipeline: Draft → Backtested → Forward Testing → Live → Retired
- Per-strategy cards (Sharpe, MaxDD, win rate)
- Forward testing progress bars
- Promotion gate criteria
- Strategy detail drill-down (Enter key)

### Tab 4: Data & Signals
- Market calendar (holidays, earnings 90d, FOMC, macro releases)
- Data health matrix (symbol x data-type, staleness thresholds)
- Coverage summary bars
- Signal engine (top signals ranked by confidence, multi-factor breakdown)
- Collector health status
- Expandable signal brief per symbol

### Tab 5: Agents
- Graph activity feeds (3 panels, one per graph)
- Current agent + node per graph
- Cycle progress bars
- Cycle history (last 3)
- Agent performance scorecard (accuracy, win rate, IC, calibration)

### Tab 6: Research
- Research WIP + queue with priorities
- ML experiment table with verdicts
- Concept drift alerts
- Alpha program progress
- Breakthrough features
- Trade reflections (lessons)
- Bug/self-healing status

---

## Keybindings

| Key | Action |
|-----|--------|
| `1-6` | Switch tabs |
| `Tab/Shift+Tab` | Next/prev tab |
| `q` | Quit |
| `r` | Force refresh all |
| `?` | Help overlay |
| `/` | Search/filter in current panel |
| `j/k` | Scroll up/down in focused panel |
| `Enter` | Expand selected item (drill-down modal) |

---

## Tiered Refresh Strategy

| Tier | Interval | Sections | Est. Queries |
|------|----------|----------|-------------|
| T1 | 5s | Services, agent events, header | ~5 |
| T2 | 15s | Portfolio, positions, signals | ~8 |
| T3 | 60s | Data health, strategies, risk | ~20 |
| T4 | 120s | Research, ML, calibration | ~12 |

**Rules:**
- Header queries (kill switch, regime, AV count) run on T1 always
- Per-tab queries only run when tab is active
- On tab switch: immediate refresh of new tab's data
- Stagger start times (0.0s, 0.3s, 0.6s, 0.9s offsets) to avoid query storms
- All queries via `@work(thread=True)` using existing `pg_conn()` context managers
- Connection pool: existing 2-10 pool in `db.py` + `asyncio.Semaphore(5)` safety

---

## New Database Tables

### market_holidays (create)
```sql
CREATE TABLE IF NOT EXISTS market_holidays (
    date        DATE PRIMARY KEY,
    name        TEXT NOT NULL,
    market_status TEXT NOT NULL CHECK (market_status IN ('closed', 'early_close')),
    close_time  TIME,
    exchange    TEXT NOT NULL DEFAULT 'NYSE'
);
```
Seeded with US market holidays. No API dependency.

### benchmark_daily (create)
```sql
CREATE TABLE IF NOT EXISTS benchmark_daily (
    date        DATE NOT NULL,
    symbol      TEXT NOT NULL DEFAULT 'SPY',
    close       NUMERIC,
    daily_return_pct NUMERIC,
    PRIMARY KEY (date, symbol)
);
```
Populated by data acquisition pipeline from existing OHLCV data for SPY.

### Tables that degrade gracefully if empty
- `risk_snapshots` — Risk console shows "No risk data"
- `breakthrough_features` — Research tab omits section
- `prompt_versions`, `prompt_candidates` — Agent scorecard omits prompt evolution

---

## Drill-Down Modal Behavior

On `Enter` for selected row, a `ModalScreen` overlay shows curated summary:

- **Position** → Symbol, qty, entry date/price, current price, P&L, strategy name, regime at entry, stop/target
- **Strategy** → Name, status, type, symbol, entry/exit rules, backtest metrics, forward test stats
- **Signal** → Symbol, action, confidence, top 5 contributing factors, risk flags, expiry
- **Trade** → Entry/exit details, P&L, strategy, decision reasoning, reflection (if exists)
- **Agent event** → Full event content, tool name, agent name, timestamp

Dismiss with `Esc`.

---

## Query Inventory (45 total)

### Always-on (header, T1)
1. Kill switch status (`system_state`)
2. AV daily calls (`system_state`)
3. Regime (`regime_states` or `loop_iteration_context`)
4. Docker service health (`docker compose ps`)
5. Graph checkpoints (`graph_checkpoints`)
6. Loop heartbeats (`loop_heartbeats`)
7. Agent events (`agent_events` LIMIT 60)

### Portfolio tab (T2)
8. Current equity (`daily_equity` latest)
9. Cash balance (`cash_balance`)
10. Open positions (`positions`)
11. Closed trades (`closed_trades` LIMIT 10)
12. Equity curve 30d (`daily_equity` last 30)
13. Benchmark 30d (`benchmark_daily` last 30)
14. P&L by strategy (`strategy_daily_pnl` aggregated)
15. P&L by symbol (`closed_trades` GROUP BY symbol)

### Strategies tab (T3)
16. Strategy pipeline (`strategies` with forward test stats from `closed_trades`)

### Data & Signals tab (T3/T2)
17. OHLCV freshness (`ohlcv` MAX timestamp per symbol)
18. News freshness (`news_sentiment` MAX time_published per symbol)
19. Sentiment freshness (`signal_state` brief_json parse)
20. Fundamentals freshness (`company_overview` last_updated)
21. Options freshness (`options_chains` MAX timestamp)
22. Insider freshness (`insider_trades` MAX filing_date)
23. Macro freshness (`macro_indicators` MAX released_at)
24. Collector health (`signal_state` collector_failures parse)
25. Active signals (`signal_state` ORDER BY confidence)
26. Signal brief detail (`signal_state` full brief_json per symbol)
27. Earnings calendar (`earnings_calendar` next 90d)
28. Market holidays (`market_holidays` next 90d)
29. Macro events (`macro_indicators` upcoming)

### Agents tab (T1/T4)
30. Agent events by graph (`agent_events` grouped)
31. Graph cycle history (`graph_checkpoints` last 3 per graph)
32. Agent skills (`agent_skills`)
33. Calibration records (`calibration_records` aggregated)
34. Prompt versions (`prompt_versions`)

### Research tab (T4)
35. Research WIP (`research_wip`)
36. Research queue (`research_queue` ORDER BY priority)
37. ML experiments (`ml_experiments` last 10)
38. Alpha programs (`alpha_research_program`)
39. Breakthrough features (`breakthrough_features`)
40. Trade reflections (`trade_reflections` last 10)
41. Bugs (`bugs`)
42. Concept drift (`ml_experiments` AUC comparison)

### Risk (T3)
43. Risk snapshot (`risk_snapshots` latest)
44. Equity alerts (`equity_alerts` + `alert_updates`)
45. Risk rejections (`decision_events` WHERE event_type = 'risk_rejection')

---

## Visual Design

### Color Scheme
- **Green:** Positive P&L, bullish signals, healthy services, within risk limits
- **Red:** Negative P&L, bearish signals, errors, risk breaches
- **Yellow:** Neutral, borderline, warnings, stale data
- **Cyan:** Informational, earnings events, agent activity
- **Dim:** Inactive, retired, empty states
- **Bold:** Headings, critical values

### Unicode Rendering
- **Sparklines:** ▁▂▃▄▅▆▇█ (8-level block chars, normalize data to 0-7 range)
- **Progress bars:** ████████░░ (filled + empty blocks)
- **Heatmap cells:** Color-coded cells with value text (Rich markup)
- **Horizontal bars:** █ repeated proportional to value, colored green/red

---

## Entry Points

```bash
# Primary (replaces status.sh)
python -m quantstack.dashboard

# Or via script wrapper
python scripts/dashboard.py  # Thin wrapper calling dashboard package
```

`scripts/dashboard.py` becomes a thin wrapper that imports and runs the Textual app. The old v1 code is removed.

---

## Error Handling

- All queries return fallback values on failure (empty list, None, 0)
- Dashboard never crashes on DB errors — shows "No data" or "Error" in affected widget
- Docker health check falls back gracefully if docker CLI unavailable
- Connection pool exhaustion: semaphore prevents queue buildup, queries wait or timeout
- Widget-level error boundaries: one widget failing doesn't affect others
