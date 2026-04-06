# QuantStack Dashboard v2 — Specification

> **Status:** Draft | **Author:** Claude | **Date:** 2026-04-03
> **Current:** Rich TUI, 795 lines, 17 queries, 8 sections, 10s refresh
> **Target:** World-class autonomous trading command center

---

## Table of Contents

1. [Design Philosophy](#1-design-philosophy)
2. [Technology Decision](#2-technology-decision)
3. [Layout Architecture](#3-layout-architecture)
4. [Section Specifications](#4-section-specifications)
   - 4.1 [Header Bar](#41-header-bar-unchanged)
   - 4.2 [Services & Graph Health](#42-services--graph-health-enhanced)
   - 4.3 [Market Calendar & Events](#43-market-calendar--events-new)
   - 4.4 [Data Health Matrix](#44-data-health-matrix-redesigned)
   - 4.5 [Strategy Pipeline](#45-strategy-pipeline-new)
   - 4.6 [Portfolio & P&L](#46-portfolio--pl-redesigned)
   - 4.7 [Graph Activity Feeds](#47-graph-activity-feeds-redesigned)
   - 4.8 [Signal Engine Dashboard](#48-signal-engine-dashboard-new)
   - 4.9 [Agent Performance Scorecard](#49-agent-performance-scorecard-new)
   - 4.10 [Research & ML Lab](#410-research--ml-lab-enhanced)
   - 4.11 [Risk Console](#411-risk-console-new)
   - 4.12 [Decisions & Audit Trail](#412-decisions--audit-trail-enhanced)
   - 4.13 [Daily Digest Summary](#413-daily-digest-summary-new)
5. [Navigation & Interaction](#5-navigation--interaction)
6. [Visual Placement Map](#6-visual-placement-map)
7. [Query Inventory](#7-query-inventory)
8. [Implementation Plan](#8-implementation-plan)

---

## 1. Design Philosophy

**Principle: Information hierarchy follows decision urgency.**

The dashboard serves one user — the owner — who needs to answer these questions in order:

1. **Is anything on fire?** (kill switch, risk alerts, service health) — always visible
2. **Am I making money?** (P&L, equity curve, positions) — top of viewport
3. **What's the system doing right now?** (agent activity, graph cycles) — middle
4. **Is the data pipeline healthy?** (freshness, gaps, calendar) — accessible
5. **How are strategies evolving?** (pipeline, performance, promotions) — one keypress away
6. **What did I learn?** (reflections, decisions, ML experiments) — deep dive

Every section earns its screen real estate by answering one of these questions.

---

## 2. Technology Decision

### Remain on Rich TUI (enhanced)

**Why not a web dashboard (Grafana, Streamlit, custom React)?**

| Factor | Rich TUI | Web UI |
|--------|----------|--------|
| Startup time | 0ms (already in terminal) | 5-30s (browser, server) |
| Resource overhead | ~20MB RSS | 200MB+ (Node/Python server + browser) |
| Deployment complexity | `pip install rich` | Docker service, port mapping, auth |
| Fits autonomous ethos | Yes — no human UI dependency | Adds a service to maintain |
| SSH-friendly | Perfect | Requires tunneling |
| Latency to data | Direct psycopg2 | HTTP round-trip |

**Enhancements over v1:**
- **Textual framework** (built on Rich) for tabbed navigation, scrollable panels, keybindings
- Tabs let us show 3x more data without cramming one screen
- Mouse support, focus navigation, searchable panels
- Still pure terminal — runs over SSH, no browser needed

**Tradeoff:** Textual adds ~5MB dependency and requires learning its widget API. Worth it for tabbed navigation which unlocks the expanded dashboard without overwhelming a single screen.

### Fallback: `--simple` flag

Retain current Rich single-screen mode as `--simple` for quick glances and CI/monitoring.

---

## 3. Layout Architecture

### Tab Structure

```
[Overview] [Portfolio] [Strategies] [Data & Signals] [Agents] [Research]
```

**Tab 1: Overview** — The "glance" tab. Everything critical on one screen.
**Tab 2: Portfolio** — Deep P&L, positions, trades, equity curve.
**Tab 3: Strategies** — Full pipeline view, per-strategy performance.
**Tab 4: Data & Signals** — Data health matrix, signal engine status, calendar.
**Tab 5: Agents** — Per-graph activity feeds, agent scorecards.
**Tab 6: Research** — ML lab, research queue, discoveries, reflections.

### Keybindings

| Key | Action |
|-----|--------|
| `1-6` | Switch tabs |
| `Tab` / `Shift+Tab` | Next/previous tab |
| `q` | Quit |
| `r` | Force refresh |
| `?` | Help overlay |
| `/` | Search/filter within current panel |
| `j/k` | Scroll up/down in focused panel |
| `Enter` | Expand selected item (strategy detail, trade detail, etc.) |
| `p` | Toggle paper/live indicator highlight |

### Refresh Strategy

| Data type | Refresh interval | Rationale |
|-----------|-----------------|-----------|
| Services/health | 5s | Detect outages fast |
| Agent events | 5s | Real-time activity feel |
| Portfolio/positions | 15s | Positions don't change sub-second |
| Data health | 60s | Data freshness is hourly concern |
| Strategy pipeline | 60s | Lifecycle changes are rare |
| ML/research | 120s | Experiments take minutes-hours |

---

## 4. Section Specifications

### 4.1 Header Bar (unchanged)

```
QUANTSTACK  16:48:37  | LIVE | Kill: ok | Regime: trending_up (82%) | AV: 393/25000 | Universe: 20 | Data: 20 syms
```

**Data sources:** `system_state`, `regime_states`, env vars
**Queries:** 3 (kill switch, AV calls, regime)

---

### 4.2 Services & Graph Health (enhanced)

```
┌─────────────────────────────── Services ───────────────────────────────┐
│ postgres UP  langfuse UP  langfuse-db UP  ollama UP                    │
│                                                                        │
│  Research Graph    UP  c#9  170s   ████████░░ 80%  [quant_researcher]  │
│  Trading Graph     UP  c#5   28s   ██░░░░░░░░ 20%  [daily_planner]    │
│  Supervisor Graph  UP  c#3   45s   ████░░░░░░ 40%  [health_monitor]   │
│                                                                        │
│  Errors (last 1h): 0  │  Avg cycle: R=165s T=32s S=48s                │
└────────────────────────────────────────────────────────────────────────┘
```

**New additions vs v1:**
- Progress bar per graph showing estimated cycle completion (based on avg cycle time)
- Current active agent per graph (from latest `agent_events`)
- Error count in last hour (from `loop_events` where event_type = 'error')
- Average cycle duration per graph (from `graph_checkpoints`)

**Data sources:** `docker compose ps`, `graph_checkpoints`, `loop_heartbeats`, `agent_events`

---

### 4.3 Market Calendar & Events (NEW)

**Tab: Data & Signals**

```
┌──────────────────────────── Market Calendar ────────────────────────────┐
│                                                                         │
│  Today: Thu Apr 03 2026         Market: OPEN (closes 16:00 ET)         │
│                                                                         │
│  ── Upcoming Events ──────────────────────────────────────────────────  │
│  04/03  FOMC Minutes Release    14:00 ET   [macro]                     │
│  04/04  Good Friday             MARKET CLOSED                          │
│  04/07  Easter Monday           MARKET OPEN (normal hours)             │
│  04/14  JNJ Earnings (BMO)      Q1 est: $2.57  [earnings]             │
│  04/14  JPM Earnings (BMO)      Q1 est: $4.11  [earnings]             │
│  04/16  NFLX Earnings (AMC)     Q1 est: $5.67  [earnings]             │
│                                                                         │
│  ── Holiday Calendar (next 90 days) ──────────────────────────────────  │
│  04/04  Good Friday          CLOSED                                    │
│  05/25  Memorial Day         CLOSED                                    │
│  06/19  Juneteenth           CLOSED                                    │
│  07/03  Independence Day     EARLY CLOSE 13:00 ET                      │
│                                                                         │
│  ── FOMC / Macro ─────────────────────────────────────────────────────  │
│  05/06-07  FOMC Meeting      Rate decision 05/07 14:00 ET             │
│  06/17-18  FOMC Meeting      Rate decision 06/18 14:00 ET             │
│  04/10     CPI Release       08:30 ET                                  │
│  04/11     PPI Release       08:30 ET                                  │
└─────────────────────────────────────────────────────────────────────────┘
```

**Features:**
- US market holiday calendar (hardcoded + fetched annually from exchange calendar)
- Earnings dates for all universe symbols (from `earnings_calendar`)
- FOMC meeting dates (from `macro_indicators` or hardcoded schedule)
- Major macro releases (CPI, PPI, NFP, GDP — from `macro_indicators`)
- Color coding: red = market closed, yellow = early close, cyan = earnings, white = macro
- Days until next market closure shown in header bar when < 3 trading days away

**Data sources:** `earnings_calendar`, `macro_indicators`, hardcoded US holiday schedule

**Implementation:** New table `market_holidays` with columns: `date`, `name`, `market_status` (closed/early_close/normal), `close_time`. Seeded annually. FOMC dates from `macro_indicators` where indicator = 'FOMC_RATE'.

---

### 4.4 Data Health Matrix (REDESIGNED)

**Tab: Data & Signals** (also summary on Overview tab)

```
┌───────────────────────────── Data Health Matrix ─────────────────────────────┐
│                                                                              │
│  Symbol   OHLCV   News    Sentiment  Fundamentals  Options  Insider  Macro   │
│  ──────   ─────   ────    ─────────  ────────────  ───────  ───────  ─────   │
│  AAPL     ✓ 0d    ✓ 2h   ✓ 2h       ✓ 14d         ✓ 1d     ✓ 7d    ✓ 1d   │
│  TSLA     ✓ 0d    ✓ 3h   ✓ 3h       ✓ 14d         ✓ 1d     ✓ 7d    ✓ 1d   │
│  JPM      ✓ 0d    ✗ 26h  ✓ 4h       ✓ 30d         ✓ 1d     ✗ 45d   ✓ 1d   │
│  NVDA     ✓ 0d    ✓ 1h   ✓ 1h       ✓ 7d          ✓ 1d     ✓ 3d    ✓ 1d   │
│  QQQ      ✓ 0d    ✓ 5h   ✗ 48h      ✓ N/A (ETF)   ✓ 1d     N/A     ✓ 1d   │
│  ...                                                                         │
│                                                                              │
│  ── Coverage Summary ────────────────────────────────────────────────────    │
│  OHLCV:        20/20  ████████████████████ 100%  (all fresh)                │
│  News:         18/20  ██████████████████░░  90%  (JPM, ACHR stale)          │
│  Sentiment:    17/20  █████████████████░░░  85%  (QQQ, MCP, ALAB stale)    │
│  Fundamentals: 20/20  ████████████████████ 100%  (threshold: 90d)           │
│  Options:      15/20  ███████████████░░░░░  75%  (5 no chain data)          │
│  Insider:      14/20  ██████████████░░░░░░  70%  (ETFs excluded)            │
│  Macro:         1/1   ████████████████████ 100%  (shared across symbols)    │
│                                                                              │
│  Collector Status: 22/22 healthy  │  Last full sweep: 14:30 ET (2h18m ago) │
│  Signal Briefs:    18/20 fresh    │  Avg collection: 4.2s/symbol            │
└──────────────────────────────────────────────────────────────────────────────┘
```

**What changed from v1:**
- v1 only showed OHLCV freshness count. v2 shows a per-symbol x per-data-type matrix
- Per-collector health status (healthy / stale / error)
- Coverage bars with percentages per data type
- Staleness thresholds are data-type-aware:
  - OHLCV: stale after 2 trading days
  - News/Sentiment: stale after 24h
  - Fundamentals: stale after 90d (quarterly)
  - Options: stale after 1 trading day
  - Insider: stale after 30d
  - Macro: stale after 7d

**Freshness detection queries** (new — one per data type):

| Data Type | Source Table | Freshness Column |
|-----------|-------------|-----------------|
| OHLCV | `ohlcv` | `MAX(timestamp) WHERE timeframe='1D'` per symbol |
| News | `news_sentiment` | `MAX(time_published)` per symbol |
| Sentiment | `signal_snapshots` or `signal_state` | check `brief_json->sentiment_score IS NOT NULL` + `expiry` |
| Fundamentals | `company_overview` | `last_updated` per symbol |
| Options | `options_chains` | `MAX(timestamp)` per symbol |
| Insider | `insider_trades` | `MAX(filing_date)` per symbol |
| Macro | `macro_indicators` | `MAX(released_at)` |

**Collector health:** Query `signal_state` per symbol, parse `brief_json` for `collector_failures` list. If a collector name appears in >50% of symbols, mark it unhealthy.

---

### 4.5 Strategy Pipeline (NEW)

**Tab: Strategies**

```
┌────────────────────────── Strategy Pipeline ──────────────────────────────┐
│                                                                           │
│   DRAFT (3)          BACKTESTED (6)      FORWARD TEST (2)    LIVE (0)    │
│   ┌─────────┐        ┌─────────────┐     ┌──────────────┐   ┌────────┐  │
│   │ AMD     │  ───►  │ TSLA swing  │ ──► │ AAPL inv     │   │        │  │
│   │ momentum│        │ Sharpe: 1.4 │     │ 12d / 30d    │   │ (none) │  │
│   │         │        │ MaxDD: -8%  │     │ P&L: +$142   │   │        │  │
│   ├─────────┤        ├─────────────┤     │ Win: 3/4     │   │        │  │
│   │ ALAB    │        │ JPM mr      │     ├──────────────┤   │        │  │
│   │ swing   │        │ Sharpe: 0.9 │     │ NVDA opt     │   │        │  │
│   │         │        │ MaxDD: -12% │     │ 5d / 30d     │   │        │  │
│   ├─────────┤        ├─────────────┤     │ P&L: +$87    │   │        │  │
│   │ ACHR    │        │ NVDA swing  │     │ Win: 2/2     │   │        │  │
│   │ earnings│        │ Sharpe: 1.1 │     └──────────────┘   │        │  │
│   └─────────┘        │ MaxDD: -15% │                        └────────┘  │
│                      ├─────────────┤                                     │
│     RETIRED (4)      │ TSLA mr     │                                     │
│   ┌─────────────┐    │ Sharpe: 0.7 │                                     │
│   │ QQQ_swing_1 │    │ MaxDD: -18% │                                     │
│   │ reason: dd  │    ├─────────────┤                                     │
│   │ lived: 14d  │    │ ABBV inv    │                                     │
│   ├─────────────┤    │ Sharpe: 1.8 │                                     │
│   │ QQQ_opts_2  │    │ MaxDD: -5%  │                                     │
│   │ reason: vol │    ├─────────────┤                                     │
│   │ lived: 7d   │    │ JPM mr      │                                     │
│   └─────────────┘    │ Sharpe: 0.6 │                                     │
│                      └─────────────┘                                     │
│                                                                           │
│  ── Promotion Gates ─────────────────────────────────────────────────    │
│  Forward → Live requires: 30+ days, Sharpe > 0.5, MaxDD < 20%,          │
│                           win_rate > 40%, 10+ trades, regime match       │
│  Auto-retire triggers:   IS/OOS divergence > 4x, win_rate drop > 20pts  │
│                                                                           │
│  ── Strategy Detail (Enter to expand) ───────────────────────────────    │
│  Selected: AAPL_investment_quality_1                                     │
│  Type: equity investment | Horizon: weeks-months | Regime: trending_up   │
│  Entry: quality_score > 0.7 AND insider_cluster_buy                      │
│  Exit: stop -8% OR target +20% OR regime_flip                            │
│  Backtest: Sharpe 1.6 | MaxDD -6.2% | Trades 23 | Win 65%              │
│  Forward:  Sharpe 1.2 | MaxDD -3.1% | Trades  4 | Win 75%   12d/30d   │
└───────────────────────────────────────────────────────────────────────────┘
```

**Features:**
- Visual Kanban-style pipeline: Draft → Backtested → Forward Testing → Live → Retired
- Per-strategy cards with key metrics (Sharpe, MaxDD, win rate)
- Forward testing progress bar (days elapsed / required)
- Promotion gate criteria displayed
- Expandable detail view per strategy (Enter key)
- Retirement reasons shown
- Color coding: green = meeting gates, yellow = borderline, red = failing gates

**Data sources:** `strategies` (all fields), `strategy_daily_pnl`, `closed_trades` (per strategy), `regime_strategy_matrix`

**Key query:**
```sql
SELECT s.strategy_id, s.name, s.status, s.symbol, s.instrument_type,
       s.time_horizon, s.regime_affinity,
       s.backtest_summary::jsonb->>'sharpe' as sharpe,
       s.backtest_summary::jsonb->>'max_drawdown' as max_dd,
       s.backtest_summary::jsonb->>'total_trades' as total_trades,
       s.backtest_summary::jsonb->>'win_rate' as win_rate,
       s.created_at, s.updated_at,
       -- forward testing stats
       (SELECT COUNT(*) FROM closed_trades ct WHERE ct.strategy_id = s.strategy_id
        AND ct.closed_at >= s.updated_at) as fwd_trades,
       (SELECT SUM(realized_pnl) FROM closed_trades ct WHERE ct.strategy_id = s.strategy_id
        AND ct.closed_at >= s.updated_at) as fwd_pnl
FROM strategies s
ORDER BY
  CASE s.status
    WHEN 'live' THEN 1
    WHEN 'forward_testing' THEN 2
    WHEN 'backtested' THEN 3
    WHEN 'draft' THEN 4
    WHEN 'retired' THEN 5
  END,
  s.updated_at DESC;
```

---

### 4.6 Portfolio & P&L (REDESIGNED)

**Tab: Portfolio**

```
┌────────────────────────────── Portfolio ──────────────────────────────────┐
│                                                                           │
│  Equity: $10,234   Cash: $6,500 (63%)   Today: +$127.50 (+1.26%)        │
│  High Water: $10,500   Drawdown: -2.5%   Gross Exposure: $3,734 (37%)   │
│                                                                           │
│  ── Equity Curve (30d) ──────────────────────────────────────────────    │
│  $10.5k │        ╭──╮                                                    │
│  $10.0k │  ╭─────╯  ╰──╮    ╭────╮     ╭───────────╮                   │
│  $ 9.5k │──╯           ╰────╯    ╰─────╯           ╰──── ●             │
│  $ 9.0k │                                                                │
│         └──────────────────────────────────────────────────────          │
│          03/04                    03/18                   04/03           │
│                                                                           │
│  vs SPY: +2.3% alpha (30d)  │  Sharpe: 1.4  │  Sortino: 1.8            │
│                                                                           │
│  ── Open Positions ──────────────────────────────────────────────────    │
│  Symbol  Qty   Entry    Current   P&L         %      Strategy    Days   │
│  AAPL    10    $178.50  $182.30   +$38.00    +2.1%  inv_qual_1   12    │
│  NVDA     5    $890.00  $905.00   +$75.00    +1.7%  swing_mom_2   3    │
│  TSLA    -2    $245.00  $240.00   +$10.00    +2.0%  short_mr_1    5    │
│                                                                           │
│  ── Closed Trades (last 10) ────────────────────────────────────────    │
│  Symbol  Side   P&L        Held   Strategy       Exit Reason    Date    │
│  JPM     LONG   +$89.00    8d     swing_mom_1    take_profit    04/01   │
│  ABBV    LONG   -$23.00    3d     inv_qual_2     stop_loss      03/29   │
│  QQQ     LONG   +$156.00   12d    swing_mr_1     signal_flip    03/25   │
│                                                                           │
│  ── P&L Attribution by Strategy ─────────────────────────────────────   │
│  Strategy          Realized    Unrealized   Win/Loss   Sharpe           │
│  inv_qual_1        +$245.00    +$38.00      5/2        1.6              │
│  swing_mom_2       +$89.00     +$75.00      3/1        1.2              │
│  short_mr_1        -$12.00     +$10.00      1/2        0.3              │
│                                                                           │
│  ── P&L Attribution by Symbol ───────────────────────────────────────   │
│  AAPL   +$283.00  ████████████████████                                   │
│  JPM    +$89.00   █████████                                              │
│  NVDA   +$75.00   ███████                                                │
│  ABBV   -$23.00   ██ (red)                                               │
│                                                                           │
│  ── Daily P&L Heatmap (last 30 days) ────────────────────────────────   │
│  Mon  Tue  Wed  Thu  Fri                                                 │
│  +1.2 -0.3 +0.8 +0.1 -0.5    ← week of 03/02                          │
│  +0.4 +1.5 -0.2 +0.9 +0.3    ← week of 03/09                          │
│  -1.1 +0.6 +0.2 -0.4 +1.8    ← week of 03/16                          │
│  +0.3 -0.1 +0.7 +1.3 +0.5    ← week of 03/23                          │
│  +0.2 +0.8 +1.1 ●            ← current week                            │
│  (green = positive, red = negative, intensity = magnitude)               │
└───────────────────────────────────────────────────────────────────────────┘
```

**New additions vs v1:**
- ASCII equity curve (30-day sparkline using braille/box-drawing characters)
- Benchmark comparison (portfolio return vs SPY)
- Portfolio-level Sharpe, Sortino, max drawdown
- P&L attribution by strategy AND by symbol (horizontal bars)
- Daily P&L calendar heatmap (weekday grid, color-coded)
- Exposure breakdown (long/short/net)
- Expanded closed trades (10 instead of 5)

**Data sources:** `daily_equity`, `positions`, `closed_trades`, `strategy_daily_pnl`, `benchmark_daily`

**Equity curve rendering:** Query last 30 rows from `daily_equity`, normalize to terminal width, render using Unicode block characters (▁▂▃▄▅▆▇█) or braille patterns.

---

### 4.7 Graph Activity Feeds (REDESIGNED)

**Tab: Agents**

```
┌──────── Research Graph (c#9, 170s) ────────┬──────── Trading Graph (c#5, 28s) ────────┐
│                                             │                                           │
│  Active: quant_researcher                   │  Active: daily_planner                    │
│  Node:   hypothesis_generation              │  Node:   plan_day                         │
│  Phase:  ████████░░░░░░░░░░░░ 40%           │  Phase:  ██░░░░░░░░░░░░░░░░░░ 10%        │
│                                             │                                           │
│  14m  * quant_research register_strategy()  │  No recent activity                       │
│  14m  * quant_research run_backtest()       │                                           │
│  14m  * quant_research run_backtest()       │                                           │
│  14m  * quant_research load_market_data()   │                                           │
│  14m  * quant_research load_market_data()   │                                           │
│  14m  * quant_research load_market_data()   │                                           │
│   4m  > quant_research Cycle 1, regime:     │                                           │
│         unknown.                            │                                           │
│                                             │                                           │
│  Cycle history:                             │  Cycle history:                           │
│  c#9: 170s (quant_researcher 6 tools)       │  c#5: 28s (data_refresh only)             │
│  c#8: 145s (ml_scientist 4 tools)           │  c#4: 95s (position_monitor 3 tools)      │
│  c#7: 200s (strategy_rd 8 tools)            │  c#3: 120s (full cycle, 2 entries)        │
│                                             │                                           │
├──────── Supervisor Graph (c#3, 45s) ────────┤                                           │
│                                             │                                           │
│  Active: health_monitor                     │                                           │
│   4m  * health_monitor check_system_status()│                                           │
│   4m  * health_monitor check_heartbeat()    │                                           │
│   4m  * health_monitor check_heartbeat()    │                                           │
│   4m  < health_monitor Now let me analyze   │                                           │
│   4m  > self_healer    System health:       │                                           │
│                                             │                                           │
└─────────────────────────────────────────────┴───────────────────────────────────────────┘
```

**New additions vs v1:**
- Current active agent and node per graph (not just event list)
- Graph phase progress bar (based on node index / total nodes)
- Cycle history (last 3 cycles with duration, primary agent, tool count)
- Better visual separation — each graph gets its own bordered panel
- Scrollable within each panel (j/k keys when focused)

**Data sources:** `agent_events`, `graph_checkpoints`, `loop_heartbeats`

---

### 4.8 Signal Engine Dashboard (NEW)

**Tab: Data & Signals**

```
┌───────────────────────── Signal Engine Status ───────────────────────────┐
│                                                                          │
│  ── Active Signals (sorted by confidence) ───────────────────────────   │
│  Symbol  Action   Conf   ML     Sentiment  Technical  Options  Macro    │
│  NVDA    BUY      87%    bull   positive   bullish    +GEX     risk-on  │
│  AAPL    HOLD     72%    bull   neutral    bullish    neutral  risk-on  │
│  TSLA    SELL     68%    bear   negative   bearish    -GEX     neutral  │
│  JPM     BUY      61%    bull   positive   neutral    +GEX     risk-on  │
│                                                                          │
│  ── Signal Brief: NVDA (Enter to expand) ────────────────────────────   │
│  Regime: trending_up | Vol: normal | HMM: stable (p=0.85)               │
│  Technical: RSI 62, ADX 28, above 20/50/200 SMA                        │
│  ML: 78% prob positive (top features: momentum_10d, vol_ratio, RSI)    │
│  Options: GEX +$2.1B, gamma_flip $890, IV_skew -0.3, VRP +2.1%        │
│  Sentiment: 0.72 (positive), 15 articles/24h, Reddit mentions +40%     │
│  Insider: cluster_buy (3 insiders in 14d), $2.3M total                  │
│  Quality: 0.81 (ROA 18%, revenue_accel +12%, Piotroski 8/9)            │
│  Risk: key_risks = [earnings_in_11d, FOMC_in_4d]                        │
│                                                                          │
│  ── Collector Health ────────────────────────────────────────────────   │
│  ✓ technical  ✓ regime  ✓ fundamentals  ✓ options_flow  ✓ ml_signal    │
│  ✓ sentiment  ✓ social  ✓ macro  ✓ sector  ✓ flow  ✓ insider          │
│  ✓ quality  ✓ events  ✓ statarb  ✓ volume  ✓ risk  ✓ cross_asset     │
│  ✗ short_interest (timeout 3x)  ✓ l2_microstructure  ✓ enhanced_sent  │
│                                                                          │
│  Avg collection time: 4.2s/symbol  │  Last full sweep: 14:30 ET        │
│  Briefs cached: 18/20  │  Failures in last 1h: 2 (short_interest)      │
└──────────────────────────────────────────────────────────────────────────┘
```

**Features:**
- Top signals ranked by confidence with multi-factor breakdown
- Expandable signal brief per symbol (full SignalBrief rendered)
- Collector health status (checkmark/cross per collector)
- Collection timing stats
- Color coding: green = bullish signals, red = bearish, yellow = neutral

**Data sources:** `signal_state` (parse `brief_json`), `signal_snapshots`

---

### 4.9 Agent Performance Scorecard (NEW)

**Tab: Agents**

```
┌───────────────────── Agent Performance Scorecard ────────────────────────┐
│                                                                          │
│  Agent               Accuracy  Win Rate  Avg P&L    IC     Trend        │
│  ─────               ────────  ────────  ───────    ──     ─────        │
│  quant_researcher     72%       --        --        0.08   IMPROVING    │
│  ml_scientist         68%       --        --        0.12   STABLE       │
│  daily_planner        --        65%      +$42      0.15   IMPROVING    │
│  position_monitor     --        71%      +$18      0.09   STABLE       │
│  trade_debater        81%       73%      +$55      0.18   IMPROVING    │
│  risk_analyst         --        --        --        --     --           │
│  options_analyst      64%       58%      +$23      0.06   DECAYING     │
│  trade_reflector      --        --        --        --     --           │
│                                                                          │
│  ── Calibration ─────────────────────────────────────────────────────   │
│  trade_debater:  says 80% conf → actual 73% win rate (well-calibrated) │
│  daily_planner:  says 75% conf → actual 65% win rate (overconfident)   │
│  options_analyst: says 70% conf → actual 58% win rate (overconfident)  │
│                                                                          │
│  ── Prompt Evolution ────────────────────────────────────────────────   │
│  Last optimization: 2d ago  │  Active candidates: 3  │  Critiques: 7   │
└──────────────────────────────────────────────────────────────────────────┘
```

**Features:**
- Per-agent accuracy, win rate, average P&L, Information Coefficient (IC), IC trend
- Calibration check: stated confidence vs actual outcomes
- Prompt evolution status (from `prompt_versions`, `prompt_candidates`, `prompt_critiques`)
- Color: green = improving, yellow = stable, red = decaying

**Data sources:** `agent_skills`, `calibration_records`, `strategy_outcomes`, `prompt_versions`, `prompt_candidates`

---

### 4.10 Research & ML Lab (ENHANCED)

**Tab: Research**

```
┌────────────────────────── Research Queue ─────────────────────────────────┐
│                                                                           │
│  ── Work In Progress ────────────────────────────────────────────────    │
│  RUNNING  quant_researcher  NVDA momentum hypothesis   (45m)             │
│  RUNNING  ml_scientist      XGBoost feature drift check (12m)            │
│                                                                           │
│  ── Queue (12 pending) ──────────────────────────────────────────────    │
│  1. [P1] backtest   AAPL earnings_drift strategy                         │
│  2. [P1] hypothesis TSLA vol_compression regime play                     │
│  3. [P2] bug_fix    options_flow collector timeout                       │
│  4. [P2] backtest   JPM mean_reversion_enhanced                          │
│  5. [P3] hypothesis AMD sector_rotation momentum                         │
│  ... +7 more                                                              │
│                                                                           │
├────────────────────────── ML Experiments ─────────────────────────────────┤
│                                                                           │
│  Total: 47  │  Best AUC: 0.73 (XGBoost, NVDA, 2026-03-28)              │
│                                                                           │
│  Recent experiments:                                                      │
│  Date     Model      Symbol  AUC    Sharpe  Features  Verdict            │
│  04/02    LightGBM   AAPL    0.68   1.2     42        passed             │
│  04/01    XGBoost    NVDA    0.73   1.5     38        champion           │
│  03/30    LightGBM   TSLA    0.61   0.8     42        failed (overfit)   │
│  03/28    RandomFor  JPM     0.65   1.0     35        passed             │
│                                                                           │
│  Concept drift alerts: TSLA model drifting (AUC -0.08 in 14d)           │
│                                                                           │
├────────────────────────── Discoveries ────────────────────────────────────┤
│                                                                           │
│  Alpha Programs: 3 active                                                 │
│  1. momentum_enhanced  (60% complete)  findings: vol_cluster works       │
│  2. earnings_drift     (30% complete)  findings: SUE > 2 predicts +3d   │
│  3. options_flow       (10% complete)  findings: GEX flip is signal      │
│                                                                           │
│  Breakthrough Features: insider_cluster_buy (imp: 0.89), gex_flip (0.74)│
│                                                                           │
│  ── Trade Reflections (Lessons Learned) ─────────────────────────────   │
│  04/01  ABBV -3.2%  "Late entry after gap up; wait for pullback"         │
│  03/28  QQQ  +8.1%  "Vol compression → breakout pattern reliable"        │
│  03/25  TSLA -5.0%  "Ignored regime flip signal; respect the regime"     │
│                                                                           │
│  ── Bugs (self-healing) ─────────────────────────────────────────────   │
│  0 open  │  3 resolved this week  │  AutoResearchClaw: idle              │
└───────────────────────────────────────────────────────────────────────────┘
```

**New additions vs v1:**
- Research WIP with duration
- Full queue with priority levels
- ML experiment table with verdicts
- Concept drift alerts
- Alpha research program progress
- Breakthrough features
- Expanded trade reflections with quotes
- Bug/self-healing status

**Data sources:** `research_queue`, `research_wip`, `ml_experiments`, `alpha_research_program`, `breakthrough_features`, `trade_reflections`, `bugs`

---

### 4.11 Risk Console (NEW)

**Tab: Overview (compact) + Portfolio (expanded)**

```
┌──────────────────────────── Risk Console ────────────────────────────────┐
│                                                                          │
│  ── Portfolio Risk ──────────────────────────────────────────────────   │
│  Gross Exposure:  $3,734 (37% of equity)    Limit: 80%    ✓ OK         │
│  Net Exposure:    $2,534 (25% of equity)    Long: 35% Short: 10%       │
│  Concentration:   NVDA 20%, AAPL 15%        Max single: 25%  ✓ OK      │
│  Correlation:     avg pairwise 0.32         Limit: 0.70    ✓ OK        │
│  Sector:          Tech 45%, Finance 25%     Max sector: 50%  ✓ OK      │
│  Daily VaR (99%): -$245                     Limit: -$500   ✓ OK        │
│  Max Drawdown:    -2.5% (from $10,500)      Limit: -15%    ✓ OK        │
│                                                                          │
│  ── Risk Events (last 7d) ───────────────────────────────────────────   │
│  04/02  risk_rejection  TSLA entry: position_too_large (wanted 30%)     │
│  03/31  drawdown_alert  portfolio drawdown hit -5% (recovered to -2.5%) │
│  03/29  correlation     NVDA+AMD correlation 0.85 > 0.70 (flagged)      │
│                                                                          │
│  ── Equity Alerts ───────────────────────────────────────────────────   │
│  ACTIVE  drawdown_warning   DD > -3% for 2+ days   triggered 04/01     │
│  CLEARED equity_min         equity < $9,000         cleared 03/30       │
│                                                                          │
│  Kill Switch: INACTIVE  │  Daily Halt: not triggered  │  Breakers: 0   │
└──────────────────────────────────────────────────────────────────────────┘
```

**Features:**
- Real-time portfolio risk metrics vs limits
- Risk event history (rejections, alerts, breaches)
- Active equity alerts with lifecycle
- Kill switch / daily halt / circuit breaker status
- Color coding: green = within limits, yellow = >75% of limit, red = breached

**Data sources:** `risk_snapshots`, `equity_alerts`, `alert_updates`, `system_state`, `positions` (for concentration calc)

---

### 4.12 Decisions & Audit Trail (ENHANCED)

**Tab: Overview (compact summary) + Research (full)**

```
┌────────────────────────── Decision Log ──────────────────────────────────┐
│                                                                          │
│  04/03 16:45  daily_planner   ENTER NVDA  conf:87%                      │
│               "Strong momentum + insider cluster buy + GEX support"      │
│                                                                          │
│  04/03 16:30  trade_debater   SKIP TSLA   conf:68%                      │
│               "Bear case: regime uncertainty, IV elevated pre-earnings"  │
│                                                                          │
│  04/03 16:20  position_monitor TIGHTEN AAPL  conf:72%                   │
│               "Approaching resistance at $185, tighten stop to $178"     │
│                                                                          │
│  04/03 16:15  risk_analyst    REJECT AMD entry  violation: concentration │
│               "Would exceed 25% single-name limit (current 22% + 8%)"   │
│                                                                          │
│  04/03 09:30  fund_manager    APPROVE 2 entries  capital: $2,400        │
│               "NVDA $1,800 (full), JPM $600 (quarter) — regime fit"     │
│                                                                          │
│  [Enter to expand any decision — shows full reasoning chain]             │
└──────────────────────────────────────────────────────────────────────────┘
```

**Data sources:** `decision_events` (with full `decision_json` for drill-down)

---

### 4.13 Daily Digest Summary (NEW)

**Tab: Overview (bottom panel, auto-populated after 17:00 ET)**

```
┌───────────────────── Daily Digest (Apr 03 2026) ─────────────────────────┐
│                                                                          │
│  Portfolio:  +$127.50 (+1.26%)  │  2 trades  │  5 open positions        │
│  Strategies: 0 promoted, 1 retired (QQQ_opts_2: vol_crush)              │
│  Research:   3 hypotheses tested, 1 passed (NVDA_momentum)              │
│  ML:         1 experiment (LightGBM AAPL, AUC 0.68)                    │
│  Loops:      R=9 cycles  T=5 cycles  S=3 cycles  │  Errors: 0          │
│  Risk:       No breakers  │  Kill switch: inactive                      │
│  Data:       20/20 OHLCV fresh  │  2 stale news sources                 │
│                                                                          │
│  Top lesson: "Insider cluster buys within trending_up regime are the     │
│  strongest entry signal — NVDA +$75 in 3 days."                         │
└──────────────────────────────────────────────────────────────────────────┘
```

**Data source:** `daily_digest` table (written by supervisor's `DigestReport` at 17:00 ET), or synthesized from component queries if digest hasn't run.

---

## 5. Navigation & Interaction

### Overview Tab — The "Glance" Screen

The Overview tab assembles compact versions of the most critical panels:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ QUANTSTACK 16:48:37 | LIVE | Kill: ok | Regime: trending_up | AV: 393/25k │
├─────────────────────────────────────────────────────────────────────────────┤
│ [Overview] [Portfolio] [Strategies] [Data & Signals] [Agents] [Research]   │
├─────────────────────────────────────┬───────────────────────────────────────┤
│         Services (compact)          │         Risk Console (compact)        │
│  R:UP c#9 170s  T:UP c#5 28s       │  Exposure: 37%  DD: -2.5%  VaR: $245│
│  S:UP c#3 45s   Errors: 0          │  Alerts: 1 active  Kill: ok          │
├─────────────────────────────────────┼───────────────────────────────────────┤
│          Portfolio (compact)        │         Recent Trades                 │
│  Equity: $10,234  Today: +$127.50  │  JPM  +$89   8d  take_profit  04/01  │
│  Open: AAPL +2.1%, NVDA +1.7%     │  ABBV -$23   3d  stop_loss    03/29  │
│  Exposure: L:35% S:10% N:25%      │  QQQ  +$156  12d signal_flip  03/25  │
├─────────────────────────────────────┼───────────────────────────────────────┤
│      Strategies (pipeline counts)   │      Signals (top 3)                 │
│  Draft:3 BT:6 FT:2 Live:0 Ret:4   │  NVDA BUY 87%  AAPL HOLD 72%       │
│  Next promotion: AAPL inv (18d)    │  TSLA SELL 68%                       │
├─────────────────────────────────────┼───────────────────────────────────────┤
│      Data Health (summary bars)     │      Research (compact)              │
│  OHLCV ████████████████████ 100%   │  WIP: 2  Queue: 12  ML exp: 47     │
│  News  ██████████████████░░  90%   │  Bugs: 0  Breakthroughs: 2          │
│  Opts  ███████████████░░░░░  75%   │  Last lesson: "insider + regime..."  │
├─────────────────────────────────────┴───────────────────────────────────────┤
│                          Agent Activity (all 3 graphs)                      │
│  R: quant_researcher register_strategy() 14m | T: idle | S: health_check  │
├─────────────────────────────────────────────────────────────────────────────┤
│                         Daily Digest (after 17:00 ET)                       │
│  +$127.50 | 2 trades | 0 promoted | 1 retired | 9 research cycles | 0 err │
├─────────────────────────────────────────────────────────────────────────────┤
│  ./stop.sh | ./start.sh | --watch | http://localhost:8421 | ? for help     │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Drill-down Model

Every panel supports `Enter` to expand a selected row:
- **Position** → full position detail (entry date, strategy, regime at entry, stop/target levels)
- **Strategy** → full strategy card (rules, backtest metrics, forward testing stats)
- **Signal** → full SignalBrief (all 66+ fields, organized by category)
- **Trade** → full trade detail (entry/exit, reasoning from decision_events, reflection)
- **Agent event** → full event payload (tool arguments, LLM response)

Drill-down opens as a modal overlay, dismissed with `Esc`.

---

## 6. Visual Placement Map

### Screen Real Estate Budget (per tab)

```
Tab: Overview (single-screen summary)
┌─────────────────────────────────────────────────────────┐
│ Header (1 line)                                         │  1 line
│ Tab bar (1 line)                                        │  1 line
│ Services (3 lines)      │ Risk Console (3 lines)        │  3 lines
│ Portfolio (3 lines)     │ Recent Trades (4 lines)       │  4 lines
│ Strategies (3 lines)    │ Signals (3 lines)             │  3 lines
│ Data Health (4 lines)   │ Research (4 lines)            │  4 lines
│ Agent Activity (2 lines, scrollable)                    │  2 lines
│ Daily Digest (2 lines)                                  │  2 lines
│ Footer (1 line)                                         │  1 line
└─────────────────────────────────────────────────────────┘
                                                Total: ~21 lines (fits 24-line terminal)

Tab: Portfolio (scrollable, ~60 lines)
┌─────────────────────────────────────────────────────────┐
│ Equity summary (2 lines)                                │
│ Equity curve ASCII chart (6 lines)                      │
│ Benchmark comparison (1 line)                           │
│ Open Positions table (variable, up to 10)               │
│ Closed Trades table (10 rows)                           │
│ P&L by Strategy table (variable)                        │
│ P&L by Symbol bars (variable)                           │
│ Daily P&L Heatmap (6 lines)                             │
└─────────────────────────────────────────────────────────┘

Tab: Strategies (scrollable, ~50 lines)
┌─────────────────────────────────────────────────────────┐
│ Pipeline Kanban (15 lines)                              │
│ Promotion Gates (3 lines)                               │
│ Strategy Detail panel (15 lines, expandable)            │
└─────────────────────────────────────────────────────────┘

Tab: Data & Signals (scrollable, ~50 lines)
┌─────────────────────────────────────────────────────────┐
│ Market Calendar (15 lines)                              │
│ Data Health Matrix (variable, one row per symbol)       │
│ Signal Engine Status (15 lines)                         │
└─────────────────────────────────────────────────────────┘

Tab: Agents (scrollable, ~40 lines)
┌─────────────────────────────────────────────────────────┐
│ Graph Activity (3 panels side-by-side, 15 lines each)   │
│ Agent Performance Scorecard (12 lines)                  │
└─────────────────────────────────────────────────────────┘

Tab: Research (scrollable, ~50 lines)
┌─────────────────────────────────────────────────────────┐
│ Research Queue & WIP (10 lines)                         │
│ ML Experiments table (10 lines)                         │
│ Alpha Programs & Discoveries (8 lines)                  │
│ Trade Reflections (6 lines)                             │
│ Bugs / Self-healing (3 lines)                           │
└─────────────────────────────────────────────────────────┘
```

---

## 7. Query Inventory

### New Queries Required (beyond existing 17)

| # | Section | Query | Table(s) |
|---|---------|-------|----------|
| 1 | Data Health | News freshness per symbol | `news_sentiment` |
| 2 | Data Health | Sentiment freshness per symbol | `signal_state` (parse brief_json) |
| 3 | Data Health | Fundamentals freshness per symbol | `company_overview` |
| 4 | Data Health | Options freshness per symbol | `options_chains` |
| 5 | Data Health | Insider freshness per symbol | `insider_trades` |
| 6 | Data Health | Macro freshness | `macro_indicators` |
| 7 | Data Health | Collector health | `signal_state` (parse collector_failures) |
| 8 | Calendar | Earnings (expanded to 90d) | `earnings_calendar` |
| 9 | Calendar | Macro events | `macro_indicators` |
| 10 | Strategies | Full pipeline with forward testing stats | `strategies` + `closed_trades` |
| 11 | Portfolio | Equity curve (30d) | `daily_equity` (last 30 rows) |
| 12 | Portfolio | Benchmark comparison | `benchmark_daily` |
| 13 | Portfolio | P&L by strategy | `strategy_daily_pnl` |
| 14 | Portfolio | P&L by symbol | `closed_trades` GROUP BY symbol |
| 15 | Portfolio | Daily P&L heatmap | `daily_equity` (last 30 rows) |
| 16 | Risk | Portfolio risk metrics | `risk_snapshots` (latest) |
| 17 | Risk | Risk events | `equity_alerts` + `alert_updates` |
| 18 | Risk | Risk rejections | `decision_events` WHERE event_type = 'risk_rejection' |
| 19 | Agents | Agent skills | `agent_skills` |
| 20 | Agents | Calibration | `calibration_records` |
| 21 | Agents | Prompt evolution | `prompt_versions` + `prompt_candidates` |
| 22 | Research | ML experiments (expanded) | `ml_experiments` (last 10) |
| 23 | Research | Alpha programs | `alpha_research_program` |
| 24 | Research | Breakthrough features | `breakthrough_features` |
| 25 | Research | Concept drift | `ml_experiments` (compare recent vs prior AUC) |
| 26 | Signals | Top signals with breakdown | `signal_state` (parse brief_json) |
| 27 | Signals | Signal brief detail | `signal_state` (full brief_json per symbol) |
| 28 | Digest | Daily digest | `loop_iteration_context` WHERE context_key = 'daily_digest' |

**Total queries: 17 (existing) + 28 (new) = 45 queries per full refresh.**

### Performance Mitigation

Not all queries run on every refresh. Stagger by section:
- **5s tier** (services, agent events): 5 queries
- **15s tier** (portfolio, signals): 8 queries
- **60s tier** (data health, strategies, risk): 20 queries
- **120s tier** (research, ML, calibration): 12 queries

Each tab only runs queries for its visible sections + always-on header queries.

---

## 8. Implementation Plan

### Phase 1: Framework Migration (Textual)
- Migrate from Rich Live to Textual App with tab structure
- Preserve `--simple` flag for backward-compatible Rich single-screen mode
- Implement tab navigation, keybindings, scrollable panels
- Port existing 8 sections into Overview tab

### Phase 2: Data Health Matrix + Calendar
- Add per-data-type freshness queries
- Build the symbol x data-type matrix renderer
- Add market holiday calendar (hardcoded US schedule + earnings from DB)
- Add coverage summary bars

### Phase 3: Strategy Pipeline
- Build Kanban-style pipeline visualization
- Add forward testing progress tracking
- Add promotion gate criteria display
- Implement strategy detail drill-down

### Phase 4: Portfolio & P&L
- ASCII equity curve renderer (Unicode block characters)
- Benchmark comparison
- P&L attribution tables (by strategy, by symbol)
- Daily P&L heatmap (weekday grid)

### Phase 5: Signal Engine + Risk Console
- Parse `signal_state.brief_json` for multi-factor signal display
- Collector health status
- Risk metrics vs limits display
- Risk event history

### Phase 6: Agent Scorecard + Research Lab
- Agent performance table from `agent_skills`
- Calibration analysis
- ML experiment table with drift detection
- Alpha program progress

### Phase 7: Polish
- Drill-down modal overlays
- Search/filter within panels
- Color theme consistency
- Performance optimization (query staggering)
- Documentation

---

## Appendix A: New Table Required

### `market_holidays`

```sql
CREATE TABLE IF NOT EXISTS market_holidays (
    date        DATE PRIMARY KEY,
    name        TEXT NOT NULL,
    market_status TEXT NOT NULL CHECK (market_status IN ('closed', 'early_close')),
    close_time  TIME,  -- NULL for full close, e.g., '13:00' for early close
    exchange    TEXT NOT NULL DEFAULT 'NYSE'
);
```

Seeded annually with US market holidays. No external API dependency — holidays are published years in advance.

---

## Appendix B: Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `textual` | >=0.50 | TUI framework (tabs, scrolling, keybindings) |
| `rich` | >=13.0 | Rendering engine (already installed) |
| `psycopg2` | >=2.9 | PostgreSQL (already installed) |

No new external services. No new Docker containers. No web server.

---

## Appendix C: Feature Comparison (v1 vs v2)

| Feature | v1 (current) | v2 (spec) |
|---------|-------------|-----------|
| Sections | 8 | 13 |
| Tabs | 1 (single screen) | 6 |
| Data types monitored | 1 (OHLCV) | 7 (OHLCV, News, Sentiment, Fundamentals, Options, Insider, Macro) |
| Strategy lifecycle view | List | Kanban pipeline with gates |
| P&L visualization | Number only | Equity curve + heatmap + attribution |
| Graph activity | Event log | Event log + progress + cycle history |
| Signal engine | Hidden | Full multi-factor display |
| Agent performance | None | Scorecard with IC, calibration, trends |
| Risk console | None | Full risk metrics vs limits |
| Market calendar | Earnings only (14d) | Holidays + earnings + FOMC + macro (90d) |
| Navigation | Scroll only | Tabs + drill-down + search |
| Queries | 17 | 45 (staggered refresh) |
| Refresh | Fixed 10s | Tiered (5s/15s/60s/120s by section) |
| Backward compat | N/A | `--simple` flag preserves v1 |
