# Python Toolkit -- Direct Imports

All computation uses Python imports via Bash. No MCP servers — every function is called directly.
Functions are async -- wrap with `asyncio.run()` when calling from Bash.

## Quick Pattern

```bash
python3 -c "
import asyncio
from quantstack.mcp.tools.qc_data import fetch_market_data
result = asyncio.run(fetch_market_data('SPY', 'daily', days=252))
print(result)
"
```

---

## Data & Market

| Function | Import | Use |
|----------|--------|-----|
| `fetch_market_data(symbol, tf, days)` | `quantstack.mcp.tools.qc_data` | OHLCV data |
| `compute_technical_indicators(symbol, tf)` | `quantstack.mcp.tools.qc_indicators` | All technicals |
| `compute_all_features(symbol, tf)` | `quantstack.mcp.tools.qc_data` | Full feature matrix |
| `get_company_facts(symbol)` | `quantstack.mcp.tools.qc_fundamentals` | Fundamentals |
| `get_financial_statements(symbol, type)` | `quantstack.mcp.tools.qc_fundamentals` | Income/balance/cashflow |
| `get_earnings_data(symbol)` | `quantstack.mcp.tools.qc_data` | Earnings history |
| `get_insider_trades(symbol)` | `quantstack.mcp.tools.qc_data` | Insider transactions |
| `analyze_volume_profile(symbol)` | `quantstack.mcp.tools.qc_data` | Volume/liquidity |
| `get_company_news(symbol)` | `quantstack.mcp.tools.qc_data` | News sentiment |
| `run_screener(criteria)` | `quantstack.mcp.tools.qc_data` | Stock screening |
| `get_event_calendar(symbol)` | `quantstack.mcp.tools.qc_data` | Earnings/events calendar |
| `get_price_snapshot(symbol)` | `quantstack.mcp.tools.qc_data` | Real-time price |

## Signals & Intel

| Function | Import | Use |
|----------|--------|-----|
| `run_multi_signal_brief(symbols)` | `quantstack.mcp.tools.signal` | Batch signal briefs |
| `get_capitulation_score(symbol)` | `quantstack.mcp.tools.capitulation` | Bottom detection |
| `get_institutional_accumulation(symbol)` | `quantstack.mcp.tools.institutional_accumulation` | Smart money |
| `get_cross_domain_intel(symbol, domain)` | `quantstack.mcp.tools.cross_domain` | Cross-domain signals |
| `get_credit_market_signals()` | `quantstack.mcp.tools.macro_signals` | Credit/macro |
| `get_market_breadth()` | `quantstack.mcp.tools.cross_domain` | Breadth indicators |
| `analyze_text_sentiment(text)` | `quantstack.mcp.tools.nlp` | NLP sentiment |

## Strategy & Backtesting

| Function | Import | Use |
|----------|--------|-----|
| `register_strategy(...)` | `quantstack.mcp.tools._impl` | Register strategy |
| `run_backtest(strategy_id, ...)` | `quantstack.mcp.tools._impl` | Single backtest |
| `run_walkforward_mtf(...)` | `quantstack.mcp.tools.qc_backtesting` | Walk-forward |
| `run_combinatorial_cv(...)` | `quantstack.mcp.tools.qc_backtesting` | PBO computation |
| `compute_information_coefficient(...)` | `quantstack.mcp.tools.qc_research` | IC analysis |
| `compute_alpha_decay(strategy_id)` | `quantstack.mcp.tools.qc_research` | Alpha half-life |
| `compute_deflated_sharpe_ratio(...)` | `quantstack.mcp.tools.qc_research` | DSR adjustment |
| `run_monte_carlo(...)` | `quantstack.mcp.tools.qc_research` | Monte Carlo sim |
| `check_strategy_rules(symbol, strategy_id)` | `quantstack.mcp.tools.qc_research` | Rule evaluation |
| `list_strategies(...)` | `quantstack.mcp.tools._impl` | Strategy listing |
| `promote_strategy(strategy_id)` | `quantstack.mcp.tools._impl` | Promote to live |

## ML & RL

| Function | Import | Use |
|----------|--------|-----|
| `train_ml_model(...)` | `quantstack.mcp.tools.ml` | Train LightGBM/XGB/CatBoost |
| `train_stacking_ensemble(...)` | `quantstack.mcp.tools.ml` | Stacking ensemble |
| `predict_ml_signal(model_id, symbol)` | `quantstack.mcp.tools.ml` | Model prediction |
| `analyze_model_shap(model_id)` | `quantstack.mcp.tools.ml` | SHAP analysis |
| `check_concept_drift(model_id)` | `quantstack.mcp.tools.ml` | Drift detection |
| `finrl_train_model(...)` | `quantstack.mcp.tools.finrl_tools` | RL agent training |
| `finrl_predict(model_id, symbol)` | `quantstack.mcp.tools.finrl_tools` | RL prediction |
| `finrl_evaluate_model(model_id)` | `quantstack.mcp.tools.finrl_tools` | RL evaluation |

## Options

| Function | Import | Use |
|----------|--------|-----|
| `get_options_chain(symbol)` | `quantstack.mcp.tools.qc_options` | Options chain |
| `compute_greeks(...)` | `quantstack.mcp.tools.qc_options` | Greeks |
| `get_iv_surface(symbol)` | `quantstack.mcp.tools.qc_options` | IV surface |
| `price_option(...)` | `quantstack.mcp.tools.qc_options` | Option pricing |
| `score_trade_structure(...)` | `quantstack.mcp.tools.qc_options` | Structure scoring |
| `simulate_trade_outcome(...)` | `quantstack.mcp.tools.qc_options` | Outcome simulation |

## Risk

| Function | Import | Use |
|----------|--------|-----|
| `compute_var(...)` | `quantstack.mcp.tools.qc_risk` | VaR/CVaR |
| `stress_test_portfolio(...)` | `quantstack.mcp.tools.qc_risk` | Monte Carlo stress test |
| `compute_position_size(...)` | `quantstack.mcp.tools.qc_risk` | Kelly/risk-based sizing |
| `compute_max_drawdown(...)` | `quantstack.mcp.tools.qc_risk` | Max drawdown |
| `check_risk_limits(...)` | `quantstack.mcp.tools.qc_risk` | Risk limit checks |

## Portfolio

| Function | Import | Use |
|----------|--------|-----|
| `compute_hrp_weights(...)` | `quantstack.mcp.tools.portfolio` | HRP optimization |
| `optimize_portfolio(...)` | `quantstack.mcp.tools.portfolio` | MVO optimization |
| `get_fill_quality(...)` | `quantstack.mcp.tools.feedback` | Fill quality TCA |
| `get_strategy_pnl(...)` | `quantstack.mcp.tools.attribution` | P&L attribution |

## Core Modules (lower-level)

| Module | Import | Use |
|--------|--------|-----|
| `SignalEngine` | `quantstack.signal_engine` | Raw signal collection |
| `RuleEngine` | `quantstack.strategies.rule_engine` | Strategy rule evaluation |
| `BacktestEngine` | `quantstack.core.backtesting.engine` | Direct backtest |
| `CausalFilter` | `quantstack.core.validation.causal_filter` | Granger causality |
| `pg_conn()` | `quantstack.db` | Database context manager |
| `DataProviderRegistry` | `quantstack.data.registry` | Data fetching |
| `MultiTimeframeFeatureFactory` | `quantstack.core.features.factory` | Feature computation |
