# Research Shared — Hard Rules, Data, State, Write Procedures

**This file is referenced by all research prompts. Read it first before executing any research iteration.**

---

## HARD RULES (always enforced)

| # | Rule | Kill Threshold |
|---|------|---------------|
| 1 | Kill overfitting | OOS Sharpe > 3.0 = fake. DELETE. |
| 2 | Kill leakage | AUC > 0.75 = leakage. INVESTIGATE then DELETE. |
| 3 | Kill fragility | IS/OOS ratio > 2.5, PBO > 0.5 = overfit. DELETE. |
| 4 | Kill instability | cv_auc_std > 0.1 = unstable. Do NOT promote. |
| 5 | 4+ signal sources per strategy | Microstructure + statistical + flow + macro/fundamentals minimum. Single-indicator = banned. |
| 6 | Regime is ONE input, not a filter | Bear markets bounce. Bull markets pull back. Trade signals, not labels. |
| 7 | Multi-timeframe by default | Use `run_backtest_mtf` / `run_walkforward_mtf`. Daily-only misses intraday edge. |
| 8 | Benchmark vs SPY | If portfolio doesn't beat buy-and-hold, we're adding complexity for nothing. |
| 9 | One variable at a time in ML | Change one thing per experiment. Log SHAP to `breakthrough_features`. |
| 10 | Write state every iteration | Your future self has ZERO memory. Files ARE your memory. |

---

## SYMBOLS

Discover from cache, never hardcode:
```python
symbols = conn.execute("SELECT DISTINCT symbol FROM ohlcv ORDER BY symbol").fetchall()
```

## DATA INVENTORY

**Source: Alpha Vantage (premium, 75 calls/min).** Alpaca = paper execution only.

| Data | Coverage |
|------|----------|
| OHLCV | Daily/Weekly (~20yr). Intraday 5-min available if fetched via `acquire_historical_data.py --phases ohlcv_5min` |
| Options | 12K+ contracts/symbol, full Greeks (HISTORICAL_OPTIONS) |
| Fundamentals | Income stmt, balance sheet, cash flow, overview |
| Valuation | P/E, P/B, EV/EBITDA, FCF yield, dividend yield (from fundamentals) |
| Quality Factors | Piotroski F-Score, Novy-Marx GP, Sloan Accruals, Beneish M-Score |
| Growth Metrics | Revenue acceleration, operating leverage, earnings momentum (SUE) |
| Ownership | Insider cluster buys, institutional herding (LSV), analyst revision momentum |
| Earnings | History, estimates, call transcripts + LLM sentiment |
| Macro | CPI, Fed Funds, GDP, NFP, unemployment, treasury yield curve |
| Flow | Insider txns, institutional holdings, news sentiment |

---

## STEP 0: HEARTBEAT
```python
record_heartbeat(loop_name="research_loop", iteration=N, status="running")
```

## STEP 1: READ STATE

```python
from quantstack.db import open_db, run_migrations
import json, os

conn = open_db()
run_migrations(conn)

# --- Counts ---
counts = {}
for label, q in [
    ("strats", "SELECT COUNT(*) FROM strategies"),
    ("exps", "SELECT COUNT(*) FROM ml_experiments"),
    ("feats", "SELECT COUNT(*) FROM breakthrough_features"),
    ("champs", "SELECT COUNT(*) FROM ml_experiments WHERE verdict='champion'"),
]:
    counts[label] = conn.execute(q).fetchone()[0]

# --- State file (per-mode to allow parallel runs) ---
_mode_suffix = os.environ.get("RESEARCH_MODE", "all").lower()
STATE_FILE = os.path.expanduser(f"~/.quant_pod/ralph_state_{_mode_suffix}.json")
os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
state = json.loads(open(STATE_FILE).read()) if os.path.exists(STATE_FILE) else {
    "iteration": 0, "research_programs": [], "errors": [], "cross_pollination": {}
}
state["iteration"] += 1

print(f"ITERATION {state['iteration']} | MODE: {_mode_suffix} | {counts}")

# --- What exists ---
strategies = conn.execute(
    "SELECT strategy_id, name, status, regime_affinity, oos_sharpe "
    "FROM strategies ORDER BY created_at DESC LIMIT 20"
).fetchall()

experiments = conn.execute(
    "SELECT experiment_id, symbol, test_auc, verdict, notes "
    "FROM ml_experiments ORDER BY created_at DESC LIMIT 10"
).fetchall()

features = conn.execute(
    "SELECT feature_name, occurrence_count, avg_shap_importance "
    "FROM breakthrough_features ORDER BY avg_shap_importance DESC LIMIT 10"
).fetchall()

programs = conn.execute(
    "SELECT * FROM alpha_research_program WHERE status='active' ORDER BY priority DESC"
).fetchall()

# --- Optimization feedback ---
loss_patterns = conn.execute("""
    SELECT root_cause, COUNT(*) as cnt, ROUND(AVG(pnl_pct), 1) as avg_loss
    FROM reflexion_episodes GROUP BY root_cause ORDER BY cnt DESC LIMIT 5
""").fetchall()

recent_episodes = conn.execute("""
    SELECT symbol, strategy_id, root_cause, pnl_pct, verbal_reinforcement, counterfactual
    FROM reflexion_episodes ORDER BY created_at DESC LIMIT 10
""").fetchall()

judge_rejections = conn.execute("""
    SELECT flags, reasoning FROM judge_verdicts
    WHERE approved = false ORDER BY created_at DESC LIMIT 5
""").fetchall()

textgrad_critiques = conn.execute("""
    SELECT node_name, critique FROM prompt_critiques ORDER BY created_at DESC LIMIT 5
""").fetchall()

# --- P&L attribution ---
strategy_pnl = conn.execute("""
    SELECT strategy_id, SUM(realized_pnl) as total_pnl, SUM(num_trades) as trades,
           SUM(win_count) as wins, SUM(loss_count) as losses
    FROM strategy_daily_pnl
    WHERE date >= CURRENT_DATE - INTERVAL '30' DAY
    GROUP BY strategy_id ORDER BY total_pnl ASC LIMIT 10
""").fetchall()

step_blame = conn.execute("""
    SELECT step_type, ROUND(AVG(credit_score), 2) as avg_credit,
           COUNT(*) as observations
    FROM step_credits WHERE credit_score < 0
    GROUP BY step_type ORDER BY avg_credit ASC
""").fetchall()

benchmark = conn.execute("""
    SELECT window_days, portfolio_sharpe, benchmark_sharpe, alpha
    FROM benchmark_comparison
    WHERE benchmark = 'SPY'
    ORDER BY date DESC, window_days LIMIT 3
""").fetchall()

print(f"Programs: {len(programs)} active | Losses: {loss_patterns}")
print(f"Strategy P&L (30d): {strategy_pnl}")
print(f"Step blame: {step_blame} | Benchmark: {benchmark}")
```

**Then read:** `.claude/memory/workshop_lessons.md` (prior iterations' memory to you).

### Convert Loss Episodes to Research Tasks

For each `recent_episode`, map root cause to action:

| Root Cause | Action |
|------------|--------|
| `regime_shift` | Add HMM stability > 0.7 entry filter. Test 1-bar regime confirmation delay. Verify regime classifier accuracy. |
| `sizing_error` | Audit Kelly inputs (stale win_rate?). Retrain ML if >30d old. Test half-Kelly cap. |
| `entry_timing` | Add confirmation bar (close above/below, not just touch). Test volume spike filter. |
| `strategy_mismatch` | Set regime_affinity to 0.0 for that regime. Check coverage gap. |
| `stop_loss_width` | Compute ATR-based stop at 1.5x. Test trailing vs fixed. Reduce max hold. |
| `data_gap` | Identify failed collectors. Add fallback or skip symbol when coverage < 80%. |

### Step Credit Attribution → Research Direction

| Worst Step | Research Action |
|-----------|----------------|
| `signal` | Improve collectors, add fallback data sources, check IC attribution per collector. |
| `regime` | Improve regime classifier, add 1-bar confirmation delay, test HMM stability filter. |
| `strategy_selection` | Audit regime_affinity weights, retrain on recent data, check strategy-regime matrix. |
| `sizing` | Audit Kelly inputs (stale win_rate?), cap size at conviction < 0.6, test half-Kelly. |
| `debate` | Review bear case weighting, check if reflexion episodes are injected into debate. |

If `strategy_pnl` shows a strategy with negative total P&L over 30d AND 10+ trades: flag for retirement review.
If `benchmark` shows portfolio Sharpe < benchmark Sharpe across all windows: bias research toward new alpha sources, not parameter tuning.
If all P&L tables are empty: system hasn't traded enough. Focus on getting strategies to paper trading.

**Feedback triage:**
- Repeated judge rejections on same flag? Tighten hypothesis criteria before submitting.
- TextGrad critiques concentrated on one node? That node is the weakest link; prioritize it.
- All tables empty? System hasn't traded enough. Focus on getting strategies to paper trading.

---

## WRITE STATE + HEARTBEAT (end of every iteration)

**Mandatory before exit (no exceptions):**

1. **State file**: what you did, which programs advanced, what you learned, what to do next iteration
2. **`alpha_research_program` table**: experiment_count, last_result, next_step, status
3. **`.claude/memory/workshop_lessons.md`**: what worked, what failed, why, feature discoveries, recommendations
4. **`.claude/memory/strategy_registry.md`**: if strategies changed
5. **`.claude/memory/ml_experiment_log.md`**: if experiments ran

**CTO Verification (before committing):**
```python
# Leakage
suspect = conn.execute(
    "SELECT experiment_id, symbol, test_auc FROM ml_experiments "
    "WHERE test_auc > 0.75 AND created_at >= CURRENT_TIMESTAMP - INTERVAL '1' DAY"
).fetchall()
if suspect: print(f"LEAKAGE WARNING: {suspect}")

# Overfitting
overfit = conn.execute(
    "SELECT strategy_id, oos_sharpe FROM strategies "
    "WHERE oos_sharpe > 3.0 AND created_at >= CURRENT_TIMESTAMP - INTERVAL '1' DAY"
).fetchall()
if overfit: print(f"OVERFITTING: {overfit}")

# Instability
unstable = conn.execute(
    "SELECT experiment_id, symbol, cv_auc_mean FROM ml_experiments "
    "WHERE cv_auc_mean IS NOT NULL AND created_at >= CURRENT_TIMESTAMP - INTERVAL '1' DAY"
).fetchall()
# cv_auc_std > 0.1 = unstable, don't promote
```

**Final heartbeat:**
```python
record_heartbeat(loop_name="research_loop", iteration=N, status="iteration_complete")
```

---

## ERROR HANDLING

| Failure | Response |
|---------|----------|
| Tool call returns error | Log error to `state["errors"]`. Retry once with different params. If still fails, skip and note in workshop_lessons. |
| Backtest crashes | Check data completeness for symbol. If <80% coverage, skip symbol. Log to workshop_lessons. |
| MCP server unreachable | Use cached data. Note staleness in signal brief. Do not trade on stale data. |
| API rate limit hit | Back off 60s. Reduce batch size. Prioritize highest-value symbols. |
| Agent spawn fails | Do the work yourself (reduced scope). Log failure mode. |
| State file corrupted | Rebuild from DB tables (strategies, ml_experiments, alpha_research_program). |

---

## SHARED EXECUTION PATHS

These paths are available to all research modes:

### ML RESEARCH (spawn `ml-scientist`)

**Delegation template:**
```
ML program: {thesis}
Symbol(s): {symbols}
Last experiment: {model_type, features, result}
Result: {AUC, SHAP findings, what worked/failed}
This iteration: {specific_next_step}

REQUIREMENTS:
- ONE variable change per experiment
- Use all 5 feature tiers. Apply CausalFilter.
- Log SHAP to breakthrough_features. Log everything to ml_experiments.
```

**After return:** AUC improved? Try more symbols or build ensemble. Degraded? Analyze what changed. Cross-pollinate SHAP findings to strategy researchers.

### RL RESEARCH (spawn `ml-scientist`)

```
Check get_rl_status(). Train RL for execution timing (DQN), sizing (PPO), alpha selection (Thompson Sampling).
< 100 trades: configure shadow recording, move on.
Enough data: train, evaluate, compare to heuristic baseline.
```

### REVIEW + CROSS-POLLINATE (you, no agents)

Every 5-6 iterations or when results accumulate.
```sql
DELETE FROM strategies WHERE oos_sharpe > 3.0;  -- kill fakes
SELECT experiment_id, symbol, test_auc FROM ml_experiments WHERE test_auc > 0.75;  -- flag leakage
```
Run `check_concept_drift(symbol)` for champions. Run `compute_alpha_decay(strategy_id)` for top strategies. Update state file `cross_pollination` and `workshop_lessons.md`.

### PARAMETER OPTIMIZATION (spawn `strategy-rd`)

When 3+ strategies passed walk-forward. Bayesian search (Optuna TPE), 50-100 trials, objective = mean OOS Sharpe across walk-forward folds. Reject if fragile (small param change -> big Sharpe change).

### PORTFOLIO + OUTPUT (spawn `execution-researcher` or you)

When 10+ validated strategies AND 3+ ML champions.
- Run HRP, min-variance, risk parity, max Sharpe. Compare.
- Correlation > 0.7: cap. Single strategy > 30%: redistribute. Fractional Kelly (f*/3).
- Stress test: COVID, rate hike, flash crash.
- Full audit: `detect_leakage`, `compute_probability_of_overfitting`, `compute_deflated_sharpe_ratio`, `check_lookahead_bias`, `compute_alpha_decay`, `run_monte_carlo`.
- Kill: PBO > 0.5, deflated Sharpe < 0, IS/OOS > 3.0, alpha half-life < 20d.

### STRATEGY DEPLOYMENT (you)

After portfolio construction + audit pass:
1. `promote_draft_strategies()` -> forward_testing
2. `set_regime_allocation()` -> update runner
3. Verify ML models: `predict_ml_signal(symbol)` for each
4. Benchmark vs SPY

---

## MANDATORY CHECKS

- **Every iteration** (<1 min): `get_system_status()`. Kill switch/halt? STOP.
- **Every 5 iterations**: full review (kill fakes, flag leakage, concept drift, alpha decay, cross-pollinate, update `workshop_lessons.md`).

**Write decision + reasoning to state file BEFORE acting.**
