# Session Handoffs

> Cross-session context + self-modification log
> Read at start of: every session
> Update: when context transfers needed, when any config/skill file modified

## Handoff Log

| Date | From Session | To Session | Key Context |
|------|-------------|------------|-------------|
| 2026-03-20 | research_iter2 | next_workshop | 2 strategies promoted to forward_testing: regime_momentum_v1 (SPY, OOS Sharpe 0.777) and vol_compress_xle_v1 (XLE, OOS Sharpe 0.219). Fixed MCP backtest bug (stop_loss_atr_multiple). 3 regime gaps identified. TSLA vol at 5th percentile -- watch for explosion. GLD/TLT divergence from risk-off is anomalous. |
| 2026-03-20 | research_iter2 | next_workshop | Files modified: strategy_registry.md (full rewrite), workshop_lessons.md (full rewrite), quantcore/mcp/tools/backtesting.py (bug fix line 89-94). |
| 2026-03-20 | ml_scientist_batch1 | next_ml_cycle | Initial ML model training complete: 6 symbols x 2 model types = 12 experiments. Champions: SPY (XGB 0.74), QQQ (XGB 0.68), NVDA (LGB 0.65), AAPL (LGB 0.60), TSLA (LGB 0.58). IWM rejected (AUC < 0.50). 6 critical bugs fixed in training pipeline. DuckDB persistence blocked by MCP lock -- needs INSERT on next restart. |
| 2026-03-20 | ml_scientist_batch1 | next_ml_cycle | Files modified: ml.py (4 bugs), trainer.py (2 bugs), ml_model_registry.md (rewrite), ml_experiment_log.md (rewrite), ml_research_program.md (rewrite), training_results.json (added AAPL). New files: AAPL_*.joblib/json models. |
