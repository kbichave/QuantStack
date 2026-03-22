# MCP Tool Consolidation Plan

## Context
120+ MCP tools is organizational debt. For an autonomous system with zero humans,
only ~20 tools are needed in the MCP layer. Everything else should be a direct
Python function call (faster, no serialization overhead, easier to test).

## Core Tools (KEEP as MCP — used by autonomous runner + interactive sessions)

### Analysis (4)
- `get_signal_brief` — SignalEngine entry point
- `run_multi_signal_brief` — batch analysis
- `get_regime` — regime classification
- `check_strategy_rules` — live rule evaluation

### Execution (4)
- `execute_trade` — order submission through risk gate
- `close_position` — position exit
- `cancel_order` — order cancellation
- `get_fills` — fill history

### Portfolio (3)
- `get_portfolio_state` — positions + equity snapshot
- `get_risk_metrics` — exposure + limit headroom
- `get_system_status` — kill switch + health

### Strategy (4)
- `register_strategy` — strategy registration
- `list_strategies` — strategy catalog
- `run_backtest` — single-period backtest
- `run_walkforward` — walk-forward validation

### ML (3)
- `train_ml_model` — model training
- `predict_ml_signal` — inference
- `check_concept_drift` — drift detection (from quantcore MCP)

### Attribution (2 — NEW, to be built in Phase A)
- `get_daily_equity` — daily NAV/equity curve
- `get_strategy_pnl` — per-strategy P&L attribution

## Demote to Python-only (REMOVE from MCP registration)

### Backtesting extras
- `run_backtest_mtf` → direct Python: `quant_pod.mcp.tools.backtesting.run_backtest_mtf`
- `run_walkforward_mtf` → direct Python
- `walk_forward_sparse_signal` → direct Python
- `run_backtest_options` → direct Python

### Decoder
- `decode_strategy` → direct Python
- `decode_from_trades` → direct Python

### Meta orchestration
- `get_regime_strategies` → direct Python
- `set_regime_allocation` → direct Python
- `resolve_portfolio_conflicts` → direct Python
- `get_strategy_gaps` → direct Python
- `promote_draft_strategies` → direct Python

### Learning loop
- `get_rl_status` → direct Python
- `get_rl_recommendation` → direct Python
- `promote_strategy` → direct Python
- `retire_strategy` → direct Python
- `get_strategy_performance` → direct Python
- `validate_strategy` → direct Python
- `update_regime_matrix_from_performance` → direct Python

### Feedback
- `get_fill_quality` → direct Python
- `get_position_monitor` → direct Python

### Intraday
- `get_intraday_status` → direct Python
- `get_tca_report` → direct Python
- `get_algo_recommendation` → direct Python

### Portfolio optimization
- `optimize_portfolio` → direct Python
- `compute_hrp_weights` → direct Python

### NLP
- `analyze_text_sentiment` → direct Python

### Strategy extras
- `get_strategy` → direct Python
- `update_strategy` → direct Python

### Audit
- `get_recent_decisions` → direct Python
- `get_audit_trail` → direct Python

## Implementation
Demote tools by removing their imports from `server.py` and their `@mcp.tool()`
decorators. The underlying functions remain — they just stop being MCP-callable.

This is a separate PR from the Day 1 cuts to avoid breaking the MCP server
during the attribution system build.
