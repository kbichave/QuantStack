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
| 11 | Multiple testing correction | "Given that I've tested N hypotheses this cycle, is this Sharpe still statistically significant?" After 10+ hypotheses, the observed best Sharpe is inflated by selection bias. Adjust for the number of trials. If the Sharpe is no longer significant after deflation, DELETE. |
| 12 | Economic mechanism required | "Can I explain WHO is on the other side of this trade and WHY the edge persists?" Every `register_strategy` MUST include an `economic_mechanism` field. Without one → draft-only, cannot promote. |
| 13 | Minimum statistical sample | "Do I have enough trades for the Sharpe to be statistically distinguishable from zero?" Formula: `N >= (1.96/target_SR)^2 * 252/hold_days`. Below minimum → exploratory only, not promotable. |

---

## HYPOTHESIS PRE-REGISTRATION (MANDATORY before any backtest)

Before calling `register_strategy` or `run_backtest`, document in the state file:

**Pre-Registration Checklist — answer ALL before testing:**

1. **What is the prediction?** State the directional hypothesis with expected sign. "Long when X, because Y should drive returns positive."
2. **What is the economic mechanism?** WHO is on the other side of this trade? WHY does this edge exist — is it a behavioral bias (overreaction, anchoring, loss aversion), a structural force (index rebalancing, dealer hedging, tax-loss selling), or a risk premium (carry, volatility, liquidity)? WHY hasn't it been arbitraged away (capacity constraints, behavioral persistence, structural friction)?
3. **What effect size do you expect?** "Sharpe ~0.5-1.0" or "IC ~0.03-0.06". This anchors expectations before you see results. If the backtest returns Sharpe 3.0 and you expected 0.7, something is wrong — investigate before celebrating.
4. **How many trades do you need?** For a Sharpe ratio to be statistically distinguishable from zero, you need roughly `N >= (1.96/target_SR)^2 * 252 / holding_period_days` trades. Calculate this number. If your backtest can't produce enough trades, the result is not statistically reliable regardless of the Sharpe.
5. **What would falsify this?** "If OOS Sharpe < 0.3 across 3+ symbols, the hypothesis is false." Define failure BEFORE testing so you can't move the goalposts after seeing results.
6. **How many hypotheses have you tested?** Track the cumulative count in `state["hypotheses_tested_total"]`. After 10+ hypotheses, selection bias becomes material — your observed best Sharpe is inflated by how many things you tried. After 20+, require deflated Sharpe adjustment.

**Hypotheses without an economic mechanism** are exploratory fishing expeditions. They get ONE backtest and must meet a higher bar (Sharpe > 1.5 IS) to proceed. Hypotheses WITH a mechanism from published research or clear structural reasoning get the standard pipeline.

Reference: Harvey, Liu, Zhu (2016) "...and the Cross-Section of Expected Returns"; Ioannidis (2005) "Why Most Published Research Findings Are False"

---

## SIGNAL HIERARCHY — Preferred for All Strategies

Signals are ranked by institutional predictive value. Prefer higher-tier signals as entry gates.
The workshop_lessons.md (iteration 3) documents why tier_1 as a *standalone* entry is a failure mode:
RSI/Stoch rules fire ~90% of the time, creating an always-on signal with no real edge.
The issue is using tier_1 *alone* in the middle of a range — not the indicators themselves.

| Tier | Name | Role in Strategy | Preferred count | Examples |
|------|------|-----------------|----------------|---------|
| **tier_3_institutional** | Institutional | PRIMARY entry gate | ≥ 1 non-neutral | GEX, IV skew z-score, LSV herding, insider cluster, capitulation_score |
| **tier_2_smart_money** | Smart Money | SECONDARY confirmation | ≥ 1 | FVG, CVD divergence, WVF extreme, Order Blocks, exhaustion_bottom |
| **tier_4_regime_macro** | Regime/Macro | CONTEXT gate | ≥ 1 | HMM state, credit_regime, breadth_score, piotroski_f_score |
| **tier_1_retail** | Retail Noise | Confirmation / exit timing | Avoid as standalone entry | RSI, MACD, Bollinger Bands, Stochastic, SMA crossover, ADX |

**Conviction guidance based on signal mix:**
- ≥1 tier_3 + ≥1 tier_2 → full conviction allowed
- ≥1 tier_3, no tier_2 → 75% max conviction
- Only tier_2 signals → 60% max conviction, document why no tier_3 available
- Only tier_1 signals as the thesis driver → 40% max conviction; treat as exploratory hypothesis, not a deployable strategy
- **Exception:** RSI < 20 or RSI > 80 at a thesis-defined price extreme is valid washout confirmation — adds up to +5% conviction on top of tier_2/3 signals. Oscillator extremes at known support/resistance are institutionally meaningful as sentiment gauges.

### Constraints
- **Avoid** registering a strategy where RSI/MACD/Stoch/BB is the sole entry trigger — document why if you do
- **Check tier_4_regime_macro first for bottom plays:** if `credit_regime == "widening"` AND `breadth_score < 0.15`, entry conviction is capped at 50% regardless of technical signals
- **ATR** is tier_1 as entry signal, tier_2 as risk-sizing tool (stops, position size)
- **BB width compression** is a valid tier_2 SETUP FILTER but entry still needs tier_3 confirmation
- **Regime label alone** is not an entry gate (Rule #6 above) — use HMM stability + tier_3 confirmation

### Bottom-Detection Tools
- `get_capitulation_score(symbol)` → composite washout score [tier_3], score > 0.65 = high conviction
- `get_institutional_accumulation(symbol)` → GEX + insider cluster + IV skew [tier_3], score > 0.55
- `get_credit_market_signals()` → HYG/LQD spread, yield curve, dollar [tier_4], credit_regime context
- `get_market_breadth()` → 15 sector ETFs breadth cascade [tier_4], breadth_divergence = hidden accumulation

---

## TOOL DISCOVERY FIRST

**RSI, MACD, and Bollinger Bands are priced in. Every retail algo runs them. The edge is in
the signals your competitors haven't found yet — and those signals are in tools you haven't
used yet. The depth of your alpha is bounded by the depth of your toolbox exploration.**

You have 160+ MCP tools. Before gathering evidence for any symbol, survey the catalog for
tools relevant to the current thesis. The tools you have never called are where uncrowded
signals live.

| Evidence category | What you're trying to learn | Example tools — not exhaustive, search for more |
|---|---|---|
| **Regime / macro context** | What phase is the market in? What's the macro backdrop? | `get_regime`, `get_credit_market_signals`, `get_market_breadth`, `get_macro_indicator`, HMM state |
| **Price structure** | Where is price relative to history? What are the structural levels? | `compute_technical_indicators`, `analyze_volume_profile`, `get_signal_brief`, multi-timeframe features |
| **Institutional positioning** | What are smart money / insiders doing? | `get_institutional_accumulation`, `get_av_insider_transactions`, `get_av_institutional_holdings`, `get_capitulation_score` |
| **Fundamentals / quality** | Is the business improving or deteriorating? | `get_financial_statements`, Piotroski F-Score, Beneish M-Score, Novy-Marx GP, earnings momentum (SUE) |
| **Flow / sentiment** | What does options flow, news, and sentiment say? | `get_signal_brief` (sentiment collector), `get_av_news_sentiment`, unusual options flow, GEX |
| **Volatility** | Is vol elevated or compressed? What's the vol forecast? | `get_iv_surface`, `fit_garch_model`, `forecast_volatility`, IV rank, term structure |
| **Microstructure** | What does order flow and liquidity reveal? | `analyze_liquidity`, `get_tca_report`, CVD divergence, FVG, order blocks |
| **Cross-domain intel** | What do the other research domains already know? | `get_cross_domain_intel` — always run this; free context |
| **ML / RL signals** | What do trained models predict for this symbol? | `finrl_screen_stocks`, `finrl_predict`, existing champion models, `get_ml_signal` |

**Before starting evidence gathering:** list which tools in each category you will use AND which you considered but skipped. If you're skipping a whole category, say why. Skipping = choosing ignorance.

---

## MANDATORY RESEARCH PIPELINE (per symbol, before ANY hypothesis)

Strategies built from a single data type (RSI-only, SMA-only) violate Rule #5 and will be
rejected by the judge. This pipeline forces breadth-first discovery for EVERY symbol.

### Step A — Gather Multi-Domain Evidence

**Run ALL categories in parallel for each symbol.** Use the best available tool per category —
not necessarily the first one that comes to mind. If a tool fails, log the failure and try
an alternative in the same category. Partial evidence beats single-category evidence.

Run in parallel batches, grouped by dependency:

```
Batch 1 (no dependencies): regime context, price structure, fundamentals, macro
Batch 2 (after batch 1):   institutional positioning, flow/sentiment, vol surface, microstructure
Batch 3 (after batch 2):   cross-domain intel, ML/RL signals, bottom detection if thesis warrants
```

**Extract key facts into a structured evidence map:**

| Category | Tools Used | Key Findings | Tier | Direction |
|----------|-----------|-------------|------|-----------|
| Regime/Macro | {tools called} | {findings} | tier_4 | bullish/bearish/neutral |
| Price Structure | {tools called} | {findings} | tier_1 | ... |
| Fundamentals/Quality | {tools called} | {findings} | tier_4 | ... |
| Institutional Positioning | {tools called} | {findings} | tier_3 | ... |
| Flow/Sentiment | {tools called} | {findings} | tier_2 | ... |
| Volatility | {tools called} | {findings} | tier_2/3 | ... |
| Microstructure | {tools called} | {findings} | tier_2 | ... |
| Cross-domain | {tools called} | {findings} | context | ... |
| ML/RL Signals | {tools called} | {findings} | varies | ... |
| **Tools Explored** | **{all tools considered this iteration, not just called}** | — | — | — |

### Step B — Synthesize: Convergence & Conflict Analysis

Score the evidence map. **Do not skip this step.**

```
CONVERGENCE SCORE:
  Categories pointing BULLISH: {list with tier}
  Categories pointing BEARISH: {list with tier}
  Categories NEUTRAL/MIXED:    {list}

  Bullish count: X/8  |  Bearish count: Y/8
  Tier-3+ bullish signals: {count}  ← need ≥1 for full conviction
  Tier-3+ bearish signals: {count}

CONFLICTS TO RESOLVE:
  - {e.g., "Fundamentals bullish (revenue +70%) but technicals bearish (below all MAs)"}
  - {e.g., "Insiders selling but institutions accumulating"}

THESIS DIRECTION: {BULLISH | BEARISH | NEUTRAL — skip if neutral}
CONVICTION: {0-100%} (apply signal hierarchy caps from SIGNAL HIERARCHY section)
```

If convergence is < 3 categories in the same direction AND no tier_3+ signals align,
**do not form a strategy hypothesis**. Log "insufficient convergence" and move to next symbol.

### Step C — Design Composite Strategy

Every strategy must combine signals from **3+ categories** in its entry rules.
The rule engine supports indicator-vs-indicator comparisons (e.g., `close > sma_200`).

**Template for composite entry rules:**

```json
{
  "entry_rules": [
    {"indicator": "close", "condition": "above", "value": "sma_50", "tier": "tier_1_trend"},
    {"indicator": "rsi", "condition": "below", "value": 40, "tier": "tier_1_washout"},
    {"indicator": "piotroski_f_score", "condition": "above", "value": 6, "tier": "tier_4_quality"},
    {"indicator": "macd_histogram", "condition": "above", "value": 0, "tier": "tier_1_momentum"}
  ],
  "exit_rules": [
    {"type": "time_stop", "hold_days": 45},
    {"type": "trailing_stop", "trailing_pct": 15},
    {"indicator": "close", "condition": "below", "value": "sma_200", "tier": "trend_break"}
  ]
}
```

**Minimum rule complexity:**
- Entry: ≥ 3 rules from ≥ 3 different categories (technicals, fundamentals, flow, macro, vol)
- Exit: ≥ 2 rules (time-based + price-based). Never rely on RSI crossover alone for exits.
- At least 1 entry rule must use a tier_2+ signal
- Regime should be ONE input rule, not the entire strategy

**Anti-patterns (will be rejected):**
- ❌ Single RSI/MACD/SMA threshold as sole entry
- ❌ Entry rules all from the same category (e.g., RSI + MACD + Stoch = all tier_1 technicals)
- ❌ RSI crossover as primary exit (fires same-day, creates 1-day holds)
- ❌ Hardcoded price levels that change over time (use indicator references: `close > sma_200`)

**Multi-Timeframe Strategy Design:**

When 1D signals fire but trade count or precision is insufficient, use a higher-timeframe setup + lower-timeframe trigger:

| Pattern | When to use | Tools |
|---------|-------------|-------|
| 1D setup + 1H trigger | Most common. Daily oversold/overbought sets up; 1H momentum times entry. Tighter stops. | `run_backtest_mtf`, `run_walkforward_mtf` |
| 4H setup + 15min trigger | More trade count than 1D. Good for strategies needing >200 trades/year. | same |
| 1W setup + 1D trigger | Macro regime as setup, daily as entry. Very low count, high conviction. | same |

MTF rules:
- Setup timeframe defines the edge (validate it at that TF first in isolation)
- Trigger timeframe refines entry timing only — do NOT redefine the edge from it
- Use trigger-TF ATR for stops (tighter, matches entry precision)
- Time stop runs from the SETUP date, not the trigger date
- Add `max_trigger_wait_days` param to avoid entering stale setups

### Step D — Validation Gates

**You are proving claims, not executing a checklist. Each gate states what must be true.
Pick the tools that best answer the question — you have 160+ available. Search them.
`run_backtest` is one option; `run_backtest_mtf`, `run_backtest_options`, `run_monte_carlo`,
`compute_information_coefficient`, `run_walkforward_mtf`, `finrl_evaluate_model`, and
`run_combinatorial_cv` are others. Use the right tool for the question.**

---

**Gate 0 — Register:** `register_strategy(...)` with composite rules from Step C.
MUST include `economic_mechanism`. Without it → draft-only, cannot promote.

---

**Gate 1 — Signal Validity** *(before spending compute on backtests)*

If ANY answer is "no" or "unknown", log as failed hypothesis and stop:
- **Does the signal predict returns?** IC between signal and forward returns at your intended horizon must be positive. IC < 0.02 = noise.
- **Is alpha decay consistent with holding period?** Half-life of signal must exceed holding_period_days. If peak IC is at a different horizon than designed, adjust the holding period to match.
- **Are features stationary?** Raw price levels create spurious correlations. Use returns, rolling z-scores, or spreads-from-mean. Non-stationary features → transform, don't proceed.
- **Do you have enough expected trades?** N >= (1.96/target_SR)² × 252/hold_days. Below minimum → exploratory only.

---

**Gate 2 — In-Sample Performance**

Prove IS performance using whatever backtest tools fit the strategy type:
- IS Sharpe meets domain threshold (see domain prompt)
- Trade count meets statistical minimum from Gate 1
- P&L attribution: which rules/signals contribute? Which are noise? Drop noise rules.
- Multi-timeframe consistency: does the edge hold across timeframes if applicable?

---

**Gate 3 — Out-of-Sample Consistency** *(mandatory before promotion)*

If ANY triggers a red flag, log as negative result and stop:
- **OOS Sharpe** meets domain threshold across folds and across 3+ symbols
- **IS/OOS ratio < 2.5** — more than 2.5x degradation = fragility, not alpha
- **PBO < 0.40** — use combinatorial cross-validation. PBO > 0.40 = more likely overfit than real. DELETE.
- **Deflated Sharpe > 0** — account for N hypotheses tested this cycle. DSR ≤ 0 = selection bias explains the Sharpe. DELETE.
- **No data leakage** — any feature using future information produces spectacular backtests and zero live performance. If detected, INVESTIGATE then DELETE.

---

**Gate 4 — Robustness** *(mandatory for promotion)*

- **Cost sensitivity:** Sharpe still meets threshold at 2x assumed slippage. If not, the strategy has no real edge after execution costs.
- **Stress test:** Max drawdown within domain limits during worst 5% of historical periods.
- **Regime stability:** Does it hold across the regimes it claims to target?
- *(Options only)* Greeks under worst-case: delta/gamma/vega/theta at entry AND at underlying ±2 ATR, VIX +50%.

---

**Gate 5 — ML/RL Lift** *(strongly preferred; not a hard block)*

Use whatever ML and RL tools best answer these questions:
- Does a supervised model improve on the rule-based signal? Search available tools — classification, regression, ensemble, stacking, RL sizing/execution agents are all options.
- Log SHAP/feature importance to `breakthrough_features`. One variable at a time (Rule 9).
- Champion vs challenger: if a model already exists for this symbol, beat it or retire it.
- RL agents for execution timing or position sizing if sufficient trade history exists.

**CausalFilter (MANDATORY for ML-backed strategies before walk-forward):**

Validate that features Granger-cause returns before spending compute on full validation:

```python
from quantstack.core.validation.causal_filter import CausalFilter

causal = CausalFilter(max_lag=5, significance_level=0.05)
X_filtered = causal.fit_transform(features_df, forward_returns)
result = causal.get_result()

drop_rate = len(result.dropped_features) / (len(result.surviving_features) + len(result.dropped_features))

if drop_rate > 0.30:
    # >30% of features are non-causal — hypothesis needs rework
    # Log in workshop_lessons.md and return to Step C
    pass
```

- drop_rate > 0.30: re-evaluate the feature set before proceeding. Log surviving vs dropped features.
- Lagged-price-only top SHAP features after filtering: autocorrelation artefact — discard.
- **Do not register an ML model directly as a strategy.** Convert top SHAP features into auditable entry rules, then proceed through Gates 2-4.
- Skipped for pure rule-based strategies (RSI/SMA-only with no ML component).

---

**Gate 6 — Update**

- Update strategy status in DB
- Write findings to memory files (strategy_registry, ml_model_registry, ml_experiment_log, workshop_lessons)

---

---

**Pre-Promotion Checklist** — answer all before calling `promote_strategy` or setting `status="forward_testing"`:

- [ ] Total trades > 50 in backtest
- [ ] Walk-forward OOS Sharpe > 0 in majority of folds
- [ ] No parameter was tuned to a single data point
- [ ] Strategy logic is explainable in one sentence
- [ ] Entry/exit rules use different indicators (not the same twice)
- [ ] Risk params include stop loss (no open-ended risk)
- [ ] If ML-backed: features passed CausalFilter (Granger causality at p<0.05 after Bonferroni)

---

**On failure at any gate:** Diagnose the specific failure mode before moving on.
"OOS Sharpe low" is not a diagnosis. "OOS Sharpe low because signal decays by day 8 but holding period is 20 days" is. Log the specific root cause.

**Short-history symbols** (< 504 bars, e.g., recent IPOs): the validation tools auto-adjust parameters when data is insufficient — proceed with their suggested params and document the wider confidence intervals as a limitation.

---

## SYMBOLS

```python
# TARGET_SYMBOL narrows research to one stock. Unset = full watchlist.
_target = os.environ.get("TARGET_SYMBOL", "").upper()
if _target:
    symbols = [(s,) for s in _target.split(",")]
    print(f"TARGET_SYMBOL: {[s[0] for s in symbols]}")
else:
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

**Institutional-grade bottom detection tools:**

| Tool | Tier | What It Measures |
|------|------|-----------------|
| `get_capitulation_score(symbol)` | tier_3 | Vol exhaustion + support integrity + WVF + PercentR dual exhaustion. Score >0.65 = washout |
| `get_institutional_accumulation(symbol)` | tier_3 | GEX dealer positioning + IV skew extreme + insider cluster (CEO-weighted) + institutional direction. Score >0.55 = accumulating |
| `get_credit_market_signals()` | tier_4 | HYG/LQD spread, TLT/SHY yield curve, UUP dollar, GLD/TLT divergence. credit_regime gate |
| `get_market_breadth()` | tier_4 | 15 sector ETF % above 50d SMA. breadth_divergence = hidden accumulation signal |

---

## TRANSACTION COST REALISM

**Flat 0.1% slippage for all instruments is unrealistic.** Use tiered costs in backtests:

| Instrument | Slippage | Rationale |
|------------|----------|-----------|
| Large-cap ETFs (SPY, QQQ) | 0.05% | Tight spreads, deep liquidity |
| Large-cap stocks (AAPL, MSFT, NVDA) | 0.05% | Sub-penny spreads |
| Mid-cap stocks | 0.10% | Wider spreads, less depth |
| Small-cap / low-volume | 0.15% | Significant market impact |
| Options | bid-ask spread / 2 (or 0.15% if unknown) | Options have wide spreads |

**Cost sensitivity test (MANDATORY for promotion):** After any successful backtest, re-run at 2x your assumed slippage. If Sharpe drops below 0.5, the strategy is cost-fragile — it doesn't have enough edge to survive real execution costs. Document this in the strategy evaluation.

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

**Then read memory files:**
1. `.claude/memory/workshop_lessons.md` — cross-ticker structural lessons (engine behavior, pitfalls)
2. `.claude/memory/tickers/{SYMBOL}.md` — per-ticker research state, strategies, evidence maps, lessons
   - If the file doesn't exist, create it from `.claude/memory/templates/ticker_template.md`
   - Read ALL ticker files for symbols in your watchlist, not just the target
3. `.claude/memory/strategy_registry.md` — strategy status overview

### Step 1b: Cross-Domain Intelligence

Query what OTHER research domains have discovered — their alerts, thesis statuses, and
technical levels provide context that improves your domain's decisions:

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

Use this table to interpret intel items from other domains:

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
3. **`.claude/memory/tickers/{SYMBOL}.md`**: update evidence map, strategies, ML models, research log, lessons for EACH symbol touched this iteration. This is the PRIMARY memory file per ticker.
4. **`.claude/memory/workshop_lessons.md`**: ONLY cross-ticker structural learnings (engine bugs, backtesting pitfalls, signal hierarchy discoveries). Do NOT put ticker-specific findings here.
5. **`.claude/memory/strategy_registry.md`**: if strategies changed
6. **`.claude/memory/ml_experiment_log.md`**: if experiments ran
7. **Negative result ledger** (append to `workshop_lessons.md` under "## Failed Hypotheses"):

   For EVERY hypothesis that failed at any pipeline stage this iteration:
   - Hypothesis ID + pre-registration summary (prediction, mechanism, expected effect size)
   - Stage where it failed (backtest / IC gate / walkforward / PBO gate / leakage / stress test)
   - Root cause analysis: NOT "Sharpe was low" but specifically WHY — e.g., "IC was 0.001 at intended horizon, indistinguishable from noise" or "PBO = 0.62, strategy is more likely overfit than real" or "alpha half-life was 3 days for a 30-day holding strategy"
   - What this rules OUT for future research (e.g., "Simple RSI divergence on AAPL has no edge in trending regimes — don't revisit")

   The negative result ledger is one of the most valuable research assets. It prevents revisiting dead ends session after session. A research program with zero failed hypotheses documented is a research program that isn't testing enough ideas.

8. **Research velocity metrics** (update in state file):
   - `hypotheses_tested_total`: cumulative since research start
   - `hypotheses_tested_this_session`: count this session
   - `hit_rate`: strategies reaching forward_testing / total tested
   - `avg_pipeline_stages`: mean number of stages a hypothesis survives before rejection

   Target velocity: 3-5 hypotheses tested per session. Hit rate 10-20% is healthy.
   - Hit rate > 50% = not testing enough hypotheses (survivorship bias — only testing "sure things")
   - Hit rate < 5% = hypothesis generation is too noisy, needs tighter pre-registration

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

**Tool Gap Protocol** — when a needed capability doesn't exist in MCP tools:

1. Use the best available workaround (direct Python, manual computation, approximation)
2. Do NOT halt or skip the research step — work around it and continue
3. Document the gap in `workshop_lessons.md` under `## Missing Tools`:
   ```
   Missing: <tool_name>
   Input:   <schema>
   Output:  <schema>
   Priority: HIGH (unlocks a strategy class) | MEDIUM (improves existing) | LOW (nice to have)
   ```
4. HIGH-priority gaps get flagged in `session_handoffs.md` for the next build session

---

## SHARED EXECUTION PATHS

These paths are available to all research modes:

### ML RESEARCH (spawn `ml-scientist`)

**Delegation template:**
```
ML program: {thesis}
Symbol(s): {symbols}
Last experiment: {model_type, features, AUC, SHAP findings, what worked/failed}
This iteration goal: {what you want to learn or improve — e.g. "improve OOS AUC", "test if macro features add lift", "reduce feature redundancy"}

Context:
- Breakthrough features from DB: {top features from breakthrough_features table}
- Current champion: {model_type, AUC, feature count}
- Regime this strategy targets: {regime}
```

**After return:** AUC improved? Try more symbols or build ensemble. Degraded? Analyze what changed. Cross-pollinate SHAP findings to strategy researchers via `breakthrough_features` table.

### RL RESEARCH (spawn `ml-scientist`)

```
Goal: train a DRL agent for {execution timing | position sizing | alpha selection} on {symbol(s)}.

Context:
- Available trade history: check finrl_list_models() and query fills table for trade count
- If < 100 trades: configure shadow recording via finrl_create_environment + finrl_train_model
  with env_type matching the goal, then move on — don't waste cycles training on thin data
- If enough data: train, evaluate OOS, compare to heuristic baseline (TWAP / fixed sizing / equal weight)
- Shadow mode is automatic for new models — they need 63 trading days before promotion eligibility
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
- **Full audit — answer these questions for the portfolio:**
  - Is there data leakage in any component strategy?
  - What is the probability the portfolio Sharpe is overfit (PBO)?
  - Is the portfolio Sharpe still significant after deflating for all strategies tested this cycle?
  - Are any features using future information?
  - How fast is alpha decaying? What's the half-life?
  - What does Monte Carlo simulation say about confidence intervals on returns?
  - Is the portfolio generating alpha, or just factor beta? Decompose into market/value/momentum/quality/vol exposures. If residual alpha < 30% of total return, the "alpha" is really factor exposure.
- Kill: PBO > 0.40, deflated Sharpe < 0, IS/OOS > 3.0, alpha half-life < 20d, cost-fragile (Sharpe < 0.5 at 2x slippage).

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
