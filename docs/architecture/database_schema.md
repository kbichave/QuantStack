# Database Schema

All state lives in PostgreSQL. 60+ tables, idempotent migrations, single source of truth.

---

## Overview

- **Connection:** `TRADER_PG_URL` env var (default: `postgresql://localhost/quantstack`)
- **Pool:** `psycopg2.ThreadedConnectionPool`, size 20 (override with `PG_POOL_MAX`)
- **Access:** Always use `db_conn()` context manager from `src/quantstack/db.py`
- **Migrations:** `run_migrations(conn)` runs at startup, idempotent (`CREATE TABLE IF NOT EXISTS`, `ADD COLUMN IF NOT EXISTS`)
- **Timestamps:** All `TIMESTAMPTZ` (timezone-aware)
- **Concurrency:** Advisory locks prevent thundering-herd on simultaneous startup

---

## Tables by subsystem

### Trading

| Table | Key Columns | Purpose |
|-------|-------------|---------|
| `positions` | position_id, symbol, qty, avg_entry_price, status, strategy_id | Active and closed position state |
| `fills` | fill_id, order_id, symbol, side, qty, fill_price, realized_pnl | Individual order fills |
| `closed_trades` | trade_id, symbol, entry_price, exit_price, pnl, strategy_id | Completed round-trip trades |
| `cash_balance` | balance, updated_at | Current cash (single row) |
| `decision_events` | session_id, event_type, decision_json, timestamp | Audit trail of all decisions |

### Strategy Lifecycle

| Table | Key Columns | Purpose |
|-------|-------------|---------|
| `strategies` | strategy_id, name, status, regime_affinity, params, backtest_summary | Strategy registry (draft -> backtested -> forward_testing -> live -> retired) |
| `strategy_outcomes` | strategy_id, trade_id, pnl, confidence | Per-trade strategy attribution |
| `strategy_daily_pnl` | strategy_id, date, pnl, cumulative_pnl | Daily P&L tracking per strategy |
| `regime_strategy_matrix` | regime, strategy_type, score | Regime-strategy fitness mapping |

### Research Pipeline

| Table | Key Columns | Purpose |
|-------|-------------|---------|
| `research_queue` | task_id, task_type, priority, context_json, status | Research task queue (hypothesis, bug_fix, etc.) |
| `research_wip` | research_id, hypothesis, symbols, status | Work-in-progress research with locking |
| `research_plans` | plan_id, domain, objectives, status | Multi-week research programs |
| `research_trajectories` | trajectory_id, domain, hypothesis, outcome | Research path tracking |
| `ml_experiments` | experiment_id, model_type, features, metrics | ML training runs and results |
| `alpha_research_program` | program_id, domain, investigations | Active alpha research programs |
| `ml_research_program` | program_id, focus_area, status | ML research roadmap tracking |
| `breakthrough_features` | feature_name, importance, discovery_date | Notable feature discoveries |

### Market Data

| Table | Key Columns | Purpose |
|-------|-------------|---------|
| `ohlcv` | symbol, date, open, high, low, close, volume | Daily OHLCV (Alpha Vantage / Alpaca) |
| `ohlcv_1m` | symbol, timestamp, open, high, low, close, volume | 1-minute intraday bars |
| `options_chains` | symbol, expiration, strike, type, bid, ask, iv, greeks | Options chain snapshots |
| `data_metadata` | symbol, data_source, last_updated | Data freshness tracking |
| `company_overview` | symbol, sector, market_cap, pe_ratio, ... | Fundamental company data |
| `financial_statements` | symbol, period, revenue, net_income, ... | Quarterly/annual financials |
| `financial_metrics` | symbol, metric_name, value | Derived financial ratios |
| `earnings_calendar` | symbol, report_date, estimate, actual | Earnings dates and results |
| `analyst_estimates` | symbol, metric, estimate, period | Consensus analyst estimates |
| `insider_trades` | symbol, insider_name, transaction_type, shares | Insider trading activity |
| `institutional_ownership` | symbol, institution, shares, change | 13F institutional holdings |
| `macro_indicators` | indicator, value, date | GDP, CPI, unemployment, etc. |
| `corporate_actions` | symbol, action_type, ex_date, details | Splits, dividends, M&A |
| `sec_filings` | symbol, filing_type, date, url | SEC filing metadata |
| `news_sentiment` | symbol, headline, sentiment_score, published_at | News sentiment analysis |

### Signals & Screening

| Table | Key Columns | Purpose |
|-------|-------------|---------|
| `signal_state` | symbol, brief_json, expiry | Cached SignalBrief per symbol |
| `signal_snapshots` | symbol, snapshot_date, brief_json | Historical signal snapshots |
| `screener_results` | symbol, score, timestamp | Stock screener output |

### Loop Coordination

| Table | Key Columns | Purpose |
|-------|-------------|---------|
| `loop_events` | loop_name, event_type, details, timestamp | Loop lifecycle events |
| `loop_cursors` | loop_name, cursor_key, cursor_value | Resumption cursors |
| `loop_heartbeats` | loop_name, iteration, started_at, finished_at, status | Liveness tracking (Docker health checks) |
| `loop_iteration_context` | loop_name, context_key, context_json, updated_at | Per-loop state persistence |

### System

| Table | Key Columns | Purpose |
|-------|-------------|---------|
| `system_state` | key, value, updated_at | Global state (kill_switch, credit_regime, av_daily_calls) |
| `bugs` | bug_id, tool_name, error_fingerprint, consecutive_errors, arc_task_id | Tool failure tracking for self-healing |
| `universe` | ticker, sector, market_cap, volatility, status | Active trading universe |

### Agent & Learning

| Table | Key Columns | Purpose |
|-------|-------------|---------|
| `agent_conversations` | crew_name, messages_json, context | Conversation history |
| `agent_memory` | agent_id, memory_type, content | Agent-specific memory |
| `agent_skills` | agent_id, skill_data | Agent skill profiles |
| `calibration_records` | agent_id, metric, value, date | Agent calibration data |
| `trade_reflections` | trade_id, classification, lessons | Post-trade analysis |
| `reflexion_episodes` | episode_id, agent, prompt, outcome | Reflexion learning episodes |

### Equity Tracking

| Table | Key Columns | Purpose |
|-------|-------------|---------|
| `daily_equity` | date, equity, cash, positions_value | Daily portfolio valuation |
| `benchmark_daily` | benchmark, date, value | Benchmark (SPY, etc.) tracking |
| `benchmark_comparison` | date, portfolio_return, benchmark_return | Relative performance |
| `equity_alerts` | alert_type, condition, triggered_at | Drawdown / equity alerts |
| `alert_exit_signals` | alert_id, symbol, signal_type | Alert-triggered exit signals |
| `alert_updates` | alert_id, update_type, details | Alert lifecycle tracking |

### Prompt Evolution

| Table | Key Columns | Purpose |
|-------|-------------|---------|
| `prompt_versions` | prompt_id, agent, version, content | Prompt version history |
| `prompt_candidates` | candidate_id, agent, content, score | Prompt optimization candidates |
| `prompt_critiques` | critique_id, candidate_id, feedback | Review feedback on prompts |
| `judge_verdicts` | verdict_id, candidate_id, verdict | Final prompt selection verdicts |
| `step_credits` | step_id, agent, credit | Step-level credit assignment |

---

## Key relationships

- `strategies.strategy_id` -> referenced by `positions`, `fills`, `strategy_outcomes`, `strategy_daily_pnl`
- `research_queue.task_id` -> referenced by `bugs.arc_task_id` (self-healing pipeline)
- `loop_heartbeats` keyed by `loop_name` (trading-graph, research-graph, supervisor-graph)
- `system_state` keyed by `key` (kill_switch, credit_regime, av_daily_calls_{date})

---

## Adding a table

1. Add `CREATE TABLE IF NOT EXISTS` to the appropriate `_migrate_*_pg()` function in `src/quantstack/db.py`
2. Use `ADD COLUMN IF NOT EXISTS` for schema evolution on existing tables
3. Never use `DROP TABLE` or `DROP COLUMN` — data is the system of record
4. All timestamps should be `TIMESTAMPTZ`
5. Add indexes for columns used in WHERE clauses
6. Update this doc when adding tables

---

## Key files

| File | Purpose |
|------|---------|
| `src/quantstack/db.py` | All migrations, connection pool, `db_conn()` context manager |
| `docs/ops-runbook.md` | Diagnostic queries for common operational scenarios |
