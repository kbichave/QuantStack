# QuantPod Research Loop

This prompt runs repeatedly in a loop. You have NO memory between iterations.
Your ONLY context is what's in DuckDB and `.claude/memory/` files.
Read state first. Then act. Then write state back. Every iteration.

## STEP 1: READ STATE (do this FIRST, every single iteration)

Run this code to know where you are:

```python
from quant_pod.db import open_db, run_migrations
conn = open_db()
run_migrations(conn)

exp_count = conn.execute("SELECT COUNT(*) FROM ml_experiments").fetchone()[0]
strat_count = conn.execute("SELECT COUNT(*) FROM strategies").fetchone()[0]
feat_count = conn.execute("SELECT COUNT(*) FROM breakthrough_features").fetchone()[0]
model_count = conn.execute("SELECT COUNT(*) FROM ml_experiments WHERE verdict='champion'").fetchone()[0]

print(f"Experiments: {exp_count}, Strategies: {strat_count}, Features: {feat_count}, Models: {model_count}")

iteration = ((exp_count + strat_count) % 15) + 1
print(f"This is iteration {iteration}")

# Read what exists
strategies = conn.execute("SELECT strategy_id, name, status, regime_affinity FROM strategies ORDER BY created_at DESC LIMIT 10").fetchall()
experiments = conn.execute("SELECT experiment_id, symbol, test_auc, verdict, notes FROM ml_experiments ORDER BY created_at DESC LIMIT 5").fetchall()
features = conn.execute("SELECT feature_name, occurrence_count, avg_shap_importance FROM breakthrough_features ORDER BY avg_shap_importance DESC LIMIT 5").fetchall()

print(f"Strategies: {strategies}")
print(f"Recent experiments: {experiments}")
print(f"Breakthrough features: {features}")
```

Also read `.claude/memory/workshop_lessons.md` for what previous iterations learned.

## STEP 2: DETERMINE WHAT TO DO

| Iteration | Task | Who Does It |
|-----------|------|-------------|
| 1 | Discover trending_up strategies for SPY, QQQ | Spawn `quant-researcher` |
| 2 | Discover ranging strategies for IWM, NVDA | Spawn `quant-researcher` |
| 3 | Discover trending_down strategies for TSLA, SPY | Spawn `quant-researcher` |
| 4 | Review all strategies — check overfitting, leakage | YOU do it |
| 5 | Train LightGBM for SPY, QQQ with CausalFilter | Spawn `ml-scientist` |
| 6 | Train XGBoost for TSLA, NVDA. Run SHAP. | Spawn `ml-scientist` |
| 7 | Train for IWM. Feature ablation on best model. | Spawn `ml-scientist` |
| 8 | Review all models — check AUC, CV stability, SHAP | YOU do it |
| 9 | Options strategies for SPY, TSLA using IV/Greeks | Spawn `quant-researcher` |
| 10 | Portfolio correlations, factor exposure, allocation | Spawn `execution-researcher` |
| 11 | Cross-pollinate: feed SHAP to researcher, failures to ML | YOU do it |
| 12 | Interaction features, regime-conditional models | Spawn `ml-scientist` |
| 13 | Combine best strategies into portfolio | Spawn `quant-researcher` |
| 14 | Audit: leakage detection, overfitting check, deflated Sharpe | YOU do it |
| 15 | Generate trading sheets, update memory, git commit | YOU do it |

## STEP 3: EXECUTE

### For STRATEGY iterations (1, 2, 3, 9, 13):

Spawn Agent with `subagent_type="alpha-research"` and a prompt that includes:
- The SPECIFIC symbols and regime to target
- What strategies already exist (from Step 1 state read)
- What SHAP features were found (from breakthrough_features)
- What previous iterations learned (from workshop_lessons.md)

Tell the agent it MUST call these tools (not discuss them — CALL them):
1. `mcp__quantpod__get_regime(symbol)` — confirm regime
2. `mcp__quantcore__get_options_chain(symbol)` — get GEX, IV, dealer positioning
3. `mcp__quantpod__register_strategy(name, entry_rules, exit_rules, parameters, regime_affinity)` — register
4. `mcp__quantpod__run_backtest(strategy_id, symbol)` — backtest
5. `mcp__quantpod__run_walkforward(strategy_id, symbol, use_purged_cv=True)` — validate OOS

Tell the agent: "Combine options flow + technicals + macro. Not just RSI."

After agent returns: log result to ml_experiments table.

### For ML iterations (5, 6, 7, 12):

Spawn Agent with `subagent_type="data-scientist"` and a prompt that includes:
- Which symbols and model type
- What breakthrough features exist
- What strategies need ML signal support

Tell the agent it MUST call:
1. `mcp__quantpod__train_ml_model(symbol, model_type, feature_tiers, apply_causal_filter=True)`
2. Report AUC, CV scores, top SHAP features
3. Write SHAP features to breakthrough_features table

After agent returns: log to ml_experiments, update breakthrough_features.

### For REVIEW iterations (4, 8, 11, 14):

YOU do this directly (no agents). Query DB:
- Strategies with OOS Sharpe > 3.0 → delete (fake)
- Models with AUC > 0.75 → investigate (leakage)
- IS/OOS ratio > 2.0 → reject (overfitting)
- Write findings to `.claude/memory/workshop_lessons.md`
- Note what to tell researcher next (from SHAP) and ML next (from failures)

### For OUTPUT iteration (15):

```python
from quant_pod.performance.trading_sheet import TradingSheetGenerator
import asyncio
sheets = asyncio.run(TradingSheetGenerator().generate_all(["SPY","QQQ","IWM","TSLA","NVDA"]))
with open("trading_sheets_monday.md", "w") as f:
    for sheet in sheets:
        f.write(sheet.to_markdown() + "\n\n---\n\n")
```

Update memory files. Git commit: `research: cycle complete`

Check completion: if strategies ≥ 3 regimes AND models ≥ 5 symbols AND experiments ≥ 15 AND features ≥ 3:
Output <promise>TRADING_READY</promise>

## STEP 4: WRITE STATE BACK

Before exiting, ALWAYS:
1. Append to `.claude/memory/workshop_lessons.md`: what this iteration did and learned
2. Update `.claude/memory/strategy_registry.md` if strategies changed
3. Update `.claude/memory/ml_experiment_log.md` if experiments ran

This is how the NEXT iteration knows what happened.

## DATA AVAILABLE

- SPY/QQQ/IWM/TSLA/NVDA: 1,057 D1 + ~30K M15 bars (2022-2026)
- Options: 12K+ contracts per symbol, full Greeks (Alpha Vantage)
- Economic: CPI, Fed Funds, NFP, GDP (3,725 points)
- Macro calendar: 220 events (FOMC/CPI/NFP/OPEX)
- 200+ technical features via `mcp__quantcore__compute_all_features()`
- Fundamentals: `mcp__quantcore__get_financial_metrics()`
- Insider trades: `mcp__quantcore__get_insider_trades()`
- News sentiment: `mcp__quantcore__get_company_news()`
- Volatility: `mcp__quantcore__fit_garch_model()`, `forecast_volatility()`

## RULES

- Every strategy iteration: register → backtest → walkforward. NO SHORTCUTS.
- Every ML iteration: train_ml_model → AUC check → SHAP → log. NO DESCRIPTIONS WITHOUT TOOL CALLS.
- Don't build "RSI < 30 → buy." Combine options flow + technicals + macro + ML + fundamentals.
- Write state back every iteration. Next iteration has no memory — only files and DB.
- After 45 iterations output <promise>TRADING_READY</promise> with whatever you have.

## OBJECTIVE

Discover ALL profitable strategies — not just one. Multiple strategies across
multiple regimes (trending_up, trending_down, ranging) for 5 symbols. Train ML
models for each. Produce a Monday trading playbook with specific executable trades.
This company makes money starting Monday. Your job is to make that possible.
