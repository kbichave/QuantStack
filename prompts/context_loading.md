# Context Loading Procedure

**Both research and trading loops execute Steps 0, 1, and 1b before acting.**
The trading loop skips Step 1c (cross-domain intel — research loops only).

Skipping context loading causes duplicate alerts, repeated failed experiments, contradictory positions, and wasted compute. Your memory resets every session — files ARE your memory.

---

## Step 0: Heartbeat

**Use `record_heartbeat` via Python import — do NOT write raw SQL against `loop_heartbeats`.**

```
record_heartbeat(loop_name="research_loop", iteration=N, status="running")
# trading loop: loop_name="trading_loop"
```

---

## Step 1: Read DB State

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

# --- State file (per-mode + per-symbol to allow parallel runs) ---
_mode_suffix = os.environ.get("RESEARCH_MODE", "all").lower()
_sym_suffix = f"_{os.environ.get('TARGET_SYMBOL', '').upper()}" if os.environ.get("TARGET_SYMBOL") else ""
STATE_FILE = os.path.expanduser(f"~/.quant_pod/ralph_state_{_mode_suffix}{_sym_suffix}.json")
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

# --- Active equity alerts (avoid duplicates, write updates for existing) ---
active_alerts = conn.execute("""
    SELECT id, symbol, time_horizon, status, confidence, regime, created_at
    FROM equity_alerts
    WHERE status IN ('pending', 'watching', 'acted')
    ORDER BY created_at DESC LIMIT 20
""").fetchall()
print(f"Active alerts: {len(active_alerts)} (avoid creating duplicates for these symbols)")
```

---

## Step 1b: Load Context (MANDATORY)

**Load in this order** (later items depend on earlier ones):

### 1. Prompt Parameters

Load `~/.quant_pod/prompt_params.json`. All downstream thresholds, tier assignments, and conviction caps use these values.

```python
import json, os
PARAMS_FILE = os.path.expanduser("~/.quant_pod/prompt_params.json")
params = json.loads(open(PARAMS_FILE).read()) if os.path.exists(PARAMS_FILE) else {}
if not params:
    print("WARNING: prompt_params.json missing — using hardcoded defaults from research_shared.md")
```

### 2. Structural Memory (cross-ticker learnings)

```python
memory_files = {
    "workshop_lessons":  ".claude/memory/workshop_lessons.md",
    "strategy_registry": ".claude/memory/strategy_registry.md",
    "ml_experiment_log": ".claude/memory/ml_experiment_log.md",
    "session_handoffs":  ".claude/memory/session_handoffs.md",
    "trade_journal":     ".claude/memory/trade_journal.md",
}
for name, path in memory_files.items():
    if os.path.exists(path):
        content = open(path).read()
        print(f"Loaded {name}: {len(content)} chars")
    else:
        print(f"MISSING: {name} ({path}) -- will create if needed")
```

| File | Extract | Why It Matters |
|------|---------|---------------|
| `workshop_lessons.md` | Failed hypotheses ledger, engine bugs, signal hierarchy discoveries, anti-patterns | Prevents re-testing dead ends |
| `strategy_registry.md` | Active strategies, regime affinities, recent performance, retirement candidates | Prevents duplicate strategies; shows coverage gaps |
| `ml_experiment_log.md` | Champion models, feature sets tried, what improved/degraded OOS | Prevents re-running identical experiments |
| `session_handoffs.md` | Unfinished work, next-step priorities, tool gaps, blockers | Continues where last session left off |
| `trade_journal.md` | Recent trade outcomes, root causes of losses, sizing lessons | Feeds into research direction |

### 3. Per-Ticker Memory

```python
ticker_dir = ".claude/memory/tickers/"
template = ".claude/memory/templates/ticker_template.md"

for sym_tuple in symbols:
    sym = sym_tuple[0]
    ticker_file = f"{ticker_dir}{sym}.md"
    if os.path.exists(ticker_file):
        content = open(ticker_file).read()
        print(f"Loaded ticker memory: {sym} ({len(content)} chars)")
    else:
        if os.path.exists(template):
            import shutil
            os.makedirs(ticker_dir, exist_ok=True)
            shutil.copy(template, ticker_file)
            print(f"Created ticker memory from template: {sym}")
        else:
            print(f"WARNING: no template at {template}, skipping {sym}")
```

| Section | Extract | Why It Matters |
|---------|---------|---------------|
| Evidence map | Last known signals, tiers, directional readings | Avoids re-running tools that haven't changed |
| Active strategies | Strategy IDs, parameters, last backtest results | Prevents duplicate registration |
| ML models | Champion model type, features, AUC, last training date | Prevents re-training identical models |
| Research log | What was tried, what failed, what's next | Continues per-symbol research |
| Lessons | Symbol-specific quirks | Informs strategy design |

### 4. Active Alerts

```python
# active_alerts was loaded in the DB query block (Step 1)
for alert in active_alerts:
    print(f"  Alert {alert['id']}: {alert['symbol']} {alert['time_horizon']} "
          f"status={alert['status']} confidence={alert['confidence']} "
          f"regime={alert['regime']} age={alert['created_at']}")
```

**Rules:**
- Do NOT create a new alert for a symbol that already has an active alert in the same time_horizon. Update the existing one instead.
- If an alert's `thesis_status` has been updated to `broken`, skip that symbol for new longs.
- Old alerts (> 14 days for swing, > 30 days for investment) need a thesis check update, not a new alert.

### 5. Reflexion + Feedback Context

```python
context_brief = {
    "top_loss_pattern": loss_patterns[0] if loss_patterns else None,
    "judge_flags_to_avoid": [r["flags"] for r in judge_rejections] if judge_rejections else [],
    "weakest_pipeline_step": step_blame[0]["step_type"] if step_blame else None,
    "beating_spy": benchmark[0]["alpha"] > 0 if benchmark else None,
    "losing_strategies": [s["strategy_id"] for s in strategy_pnl if s["total_pnl"] < 0 and s["trades"] > 10],
}
print(f"Context brief: {context_brief}")
```

### 6. Context Validation Checklist

Before proceeding to any research or trading step, verify:

- [ ] `params` loaded (prompt_params.json)
- [ ] `workshop_lessons.md` read (or noted as missing)
- [ ] `strategy_registry.md` read (or noted as missing)
- [ ] `session_handoffs.md` read (or noted as missing) — check for unfinished priorities
- [ ] Ticker files read for ALL watchlist symbols
- [ ] Active alerts loaded and reviewed for duplicates
- [ ] Loss patterns and judge rejections reviewed
- [ ] `context_brief` synthesized and printed

**If any critical file is missing:** log it in `state["errors"]` and proceed with reduced context. Do NOT block on missing files.

---

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

## Step 1c: Cross-Domain Intelligence (research loops only)

*Trading loop skips this step — it gets cross-domain context via signal brief.*

Query what OTHER research domains have discovered — their alerts, thesis statuses, and technical levels provide context that improves your domain's decisions:

```python
intel = get_cross_domain_intel(
    symbol=_target or "",
    requesting_domain=_mode_suffix,  # "investment", "swing", "options", or "all"
)

if intel.get("success"):
    cross_intel = intel.get("intel_items", [])
    convergence = intel.get("symbol_convergence", [])

    actionable = [i for i in cross_intel if i.get("relevance", 0) >= 0.7]
    converging = [c for c in convergence if len(c.get("domains_active", [])) >= 2]

    print(f"Cross-domain: {len(cross_intel)} items, {len(actionable)} actionable, "
          f"{len(converging)} converging symbols")

    state["cross_domain_intel"] = {
        "actionable_count": len(actionable),
        "converging_symbols": [c["symbol"] for c in converging],
        "top_items": [
            {"symbol": i["symbol"], "type": i["intel_type"], "headline": i["headline"]}
            for i in actionable[:5]
        ],
    }
```

#### Cross-Domain Intelligence Mapping

| Intel Type | If You Are... | Action |
|---|---|---|
| `fundamental_floor` | Swing | Use book/intrinsic value as stop floor when it's between 1-2.5x ATR from price. Free support from fundamental buyers. |
| `thesis_status=weakening` | Swing | Avoid new longs. Tighten trailing stop to 5% on existing. |
| `thesis_status=broken` | Swing, Options | **HARD RULE:** do NOT enter longs. Exit existing positions. |
| `thesis_status=intact` | Options | High-conviction directional — consider long calls/puts aligned with thesis. |
| `fundamental_event` | Options | Catalyst = IV inflation window. Check IV rank before entry. |
| `fundamental_event` | Swing | Position BEFORE event if thesis supports. Tighten stop through event. |
| `technical_levels` | Investment | Use breakout/support for entry timing. Wait for pullback to support. |
| `technical_levels` | Options | Strike selection — sell premium at support/resistance levels. |
| `momentum_signal` (bullish) | Investment | Price confirms thesis → favorable entry timing. Size up. |
| `momentum_signal` (bearish) | Investment | Price contradicts thesis → delay entry or reduce size by 50%. |
| `options_strategies_active` | Investment, Swing | Check IV rank via `get_iv_surface` before sizing equity positions. |
| `convergence` (aligned) | Any | High-conviction: 2+ domains agree. Size up within risk limits. |
| `convergence` (conflicting) | Any | Caution: reduce size by 50%. Document conflict in trade journal. |
