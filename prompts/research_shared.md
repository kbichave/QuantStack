# Research Shared --Hard Rules, Data, State, Write Procedures

**This file is referenced by all research prompts. Read it first before executing any research iteration.**

---

## TUNABLE PARAMETERS (load before applying any rule)

All numeric thresholds, tier assignments, and conviction caps live in a parameters file.
The prompt provides **defaults** for cold start. Once empirical data exists, the parameters
file overrides the defaults. This prevents hardcoded priors from overriding what the system
has actually learned.

```python
import json, os

PARAMS_FILE = os.path.expanduser("~/.quant_pod/prompt_params.json")

DEFAULT_PARAMS = {
    "kill_thresholds": {
        "equity_daily":  {"max_oos_sharpe": 3.0, "max_auc": 0.75, "max_is_oos_ratio": 2.5, "max_pbo": 0.50, "max_cv_std": 0.10},
        "equity_weekly": {"max_oos_sharpe": 2.5, "max_auc": 0.72, "max_is_oos_ratio": 2.5, "max_pbo": 0.50, "max_cv_std": 0.12},
        "options_daily": {"max_oos_sharpe": 4.0, "max_auc": 0.80, "max_is_oos_ratio": 3.0, "max_pbo": 0.50, "max_cv_std": 0.12},
    },
    "signal_tiers": {
        # Defaults -- overridden by signal_tier_performance table when empirical data exists
        "tier_3_institutional": {"default_signals": ["GEX", "IV_skew_zscore", "LSV_herding", "insider_cluster", "capitulation_score"], "conviction_cap": 1.0},
        "tier_2_smart_money":   {"default_signals": ["FVG", "CVD_divergence", "WVF_extreme", "order_blocks", "exhaustion_bottom"], "conviction_cap": 0.75},
        "tier_4_regime_macro":  {"default_signals": ["HMM_state", "credit_regime", "breadth_score", "piotroski_f_score"], "conviction_cap": 1.0},
        "tier_1_retail":        {"default_signals": ["RSI", "MACD", "BB", "Stochastic", "SMA_crossover", "ADX"], "conviction_cap": 0.40},
    },
    "conviction_caps": {
        # These are starting defaults. Recalibrate every 5 iterations from step_credits + strategy_daily_pnl.
        "tier3_plus_tier2": 1.0,
        "tier3_no_tier2":   0.75,
        "tier2_only":       0.60,
        "tier1_only":       0.40,
    },
    "min_signal_categories": 2,       # Minimum orthogonal categories. Default 2, prefer 3-4.
    "min_signal_categories_preferred": 3,  # Below this, require written justification.
    "slippage_defaults": {
        "large_cap_etf":   0.05,
        "large_cap_stock": 0.05,
        "mid_cap":         0.10,
        "small_cap":       0.15,
        "options":         0.15,
    },
    "bottom_detection_thresholds": {
        # Per-tool thresholds. Recalibrate from actual alert hit rates.
        "capitulation_score": 0.65,
        "institutional_accumulation": 0.55,
    },
    "last_calibrated": None,
    "calibration_iteration": 0,
}

if os.path.exists(PARAMS_FILE):
    params = {**DEFAULT_PARAMS, **json.loads(open(PARAMS_FILE).read())}
    print(f"Loaded prompt_params.json (calibrated iter {params.get('calibration_iteration', '?')})")
else:
    params = DEFAULT_PARAMS
    print("Using DEFAULT prompt params (no calibration yet)")
```

**Calibration cycle (every 5 iterations during REVIEW + CROSS-POLLINATE):**

1. Query `signal_tier_performance` table (IC, PnL contribution, win rate by signal). Update `signal_tiers` if empirical rankings diverge from defaults.
2. Query `strategy_daily_pnl` + `step_credits` to recalibrate `conviction_caps`. If tier_2-only strategies outperform tier_3+tier_2, adjust caps.
3. Query `get_tca_report` for symbols with 20+ fills. Override `slippage_defaults` with empirical medians.
4. Check false positive rate on kill thresholds (strategies killed that later showed edge in paper trading). Widen thresholds where false positive rate > 20%.
5. Write updated params to `PARAMS_FILE` with `last_calibrated` and `calibration_iteration`.

---

## HARD RULES (always enforced)

| # | Rule | Kill Threshold |
|---|------|---------------|
| 1 | Kill overfitting | OOS Sharpe > `params["kill_thresholds"][instrument_type]["max_oos_sharpe"]` = suspicious. INVESTIGATE. If no explanation, DELETE. (Default: 3.0 equity daily, 4.0 options) |
| 2 | Kill leakage | AUC > `params["kill_thresholds"][instrument_type]["max_auc"]` = leakage. INVESTIGATE then DELETE. (Default: 0.75 equity, 0.80 options. Scale with prediction horizon: shorter horizon tolerates higher AUC.) |
| 3 | Kill fragility | IS/OOS ratio > `max_is_oos_ratio`, PBO > `max_pbo` = overfit. DELETE. (Defaults: 2.5, 0.50. Short-history symbols get wider tolerance per params.) |
| 4 | Kill instability | cv_auc_std > `max_cv_std` = unstable. Do NOT promote. (Default: 0.10. Evaluate relative to cv_auc_mean: std/mean > 0.18 is a stronger signal than raw std alone.) |
| 5 | Multi-source signal requirement | Strategies must use signals from `params["min_signal_categories"]`+ orthogonal categories (default: 2, preferred: 3+). Below preferred count requires written justification of why each omitted category would add noise. Single-indicator = banned. |
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

**Pre-Registration Checklist --answer ALL before testing:**

1. **What is the prediction?** State the directional hypothesis with expected sign. "Long when X, because Y should drive returns positive."
2. **What is the economic mechanism?** WHO is on the other side of this trade? WHY does this edge exist --is it a behavioral bias (overreaction, anchoring, loss aversion), a structural force (index rebalancing, dealer hedging, tax-loss selling), or a risk premium (carry, volatility, liquidity)? WHY hasn't it been arbitraged away (capacity constraints, behavioral persistence, structural friction)?
3. **What effect size do you expect?** "Sharpe ~0.5-1.0" or "IC ~0.03-0.06". This anchors expectations before you see results. If the backtest returns Sharpe 3.0 and you expected 0.7, something is wrong --investigate before celebrating.
4. **How many trades do you need?** For a Sharpe ratio to be statistically distinguishable from zero, you need roughly `N >= (1.96/target_SR)^2 * 252 / holding_period_days` trades. Calculate this number. If your backtest can't produce enough trades, the result is not statistically reliable regardless of the Sharpe.
5. **What would falsify this?** "If OOS Sharpe < 0.3 across 3+ symbols, the hypothesis is false." Define failure BEFORE testing so you can't move the goalposts after seeing results.
6. **How many hypotheses have you tested?** Track the cumulative count in `state["hypotheses_tested_total"]`. After 10+ hypotheses, selection bias becomes material --your observed best Sharpe is inflated by how many things you tried. After 20+, require deflated Sharpe adjustment.

**Hypotheses without an economic mechanism** are exploratory fishing expeditions. They get ONE backtest and must meet a higher bar (Sharpe > 1.5 IS) to proceed. Hypotheses WITH a mechanism from published research or clear structural reasoning get the standard pipeline.

Reference: Harvey, Liu, Zhu (2016) "...and the Cross-Section of Expected Returns"; Ioannidis (2005) "Why Most Published Research Findings Are False"

---

## SIGNAL HIERARCHY --Data-Driven with Defaults

Signals are ranked by predictive value. The **default** ranking below is the cold-start prior.
Once `signal_tier_performance` has empirical data (IC, PnL contribution, win rate per signal),
use empirical rankings instead. Recalibrate every 5 iterations.

The workshop_lessons.md (iteration 3) documents why tier_1 as a *standalone* entry is a failure mode:
RSI/Stoch rules fire ~90% of the time, creating an always-on signal with no real edge.
The issue is using tier_1 *alone* in the middle of a range -- not the indicators themselves.

**Default tier assignments** (override with `params["signal_tiers"]` when calibrated):

| Tier | Name | Default Role | Examples |
|------|------|-------------|---------|
| **tier_3_institutional** | Institutional | PRIMARY entry gate | GEX, IV skew z-score, LSV herding, insider cluster, capitulation_score |
| **tier_2_smart_money** | Smart Money | SECONDARY confirmation | FVG, CVD divergence, WVF extreme, Order Blocks, exhaustion_bottom |
| **tier_4_regime_macro** | Regime/Macro | CONTEXT gate | HMM state, credit_regime, breadth_score, piotroski_f_score |
| **tier_1_retail** | Retail/Technical | Confirmation / exit timing | RSI, MACD, Bollinger Bands, Stochastic, SMA crossover, ADX |

**Tier promotion/demotion:** If a signal's empirical IC and PnL contribution consistently
outperform its tier peers (measured over 50+ trades), promote it. If it underperforms, demote it.
Log tier changes in `workshop_lessons.md` with evidence.

**Conviction guidance** (read from `params["conviction_caps"]`, defaults below):
- tier_3 + tier_2 present: `params["conviction_caps"]["tier3_plus_tier2"]` (default 1.0)
- tier_3 only, no tier_2: `params["conviction_caps"]["tier3_no_tier2"]` (default 0.75)
- tier_2 only: `params["conviction_caps"]["tier2_only"]` (default 0.60), document why no tier_3 available
- tier_1 only as thesis driver: `params["conviction_caps"]["tier1_only"]` (default 0.40), treat as exploratory
- **Exception:** Oscillator extremes (RSI < 20 / > 80) at thesis-defined price extremes add up to +5% conviction on top of tier_2/3 signals. This is a sentiment gauge, not a standalone entry.

### Constraints
- **Avoid** registering a strategy where any single tier_1 indicator is the sole entry trigger -- document why if you do
- **Check regime/macro context first for bottom plays:** if credit conditions are tightening AND breadth is weak, entry conviction is capped at 50% regardless of technical signals. Use whatever regime/macro tools are available to assess this.
- **ATR** is tier_1 as entry signal, tier_2 as risk-sizing tool (stops, position size)
- **Volatility compression** (BB width, ATR contraction, etc.) is a valid tier_2 SETUP FILTER but entry still needs higher-tier confirmation
- **Regime label alone** is not an entry gate (Rule #6 above) -- use regime stability + higher-tier confirmation

### Bottom-Detection Tools (capability-based, not tool-name-specific)

Search available tools for these capabilities. The specific tool names below are current defaults;
if better tools exist or tools are renamed, use the best available. Thresholds come from
`params["bottom_detection_thresholds"]` and should be recalibrated from actual alert hit rates.

| Capability | Current Tool | Tier | Default Threshold |
|-----------|-------------|------|-------------------|
| Composite washout / capitulation scoring | `get_capitulation_score(symbol)` | tier_3 | `params["bottom_detection_thresholds"]["capitulation_score"]` (default 0.65) |
| Institutional accumulation detection | `get_institutional_accumulation(symbol)` | tier_3 | `params["bottom_detection_thresholds"]["institutional_accumulation"]` (default 0.55) |
| Credit market / macro stress signals | `get_credit_market_signals()` | tier_4 | credit_regime context (no single threshold) |
| Market breadth / sector health | `get_market_breadth()` | tier_4 | breadth_divergence = hidden accumulation signal |

---

## EVIDENCE BREADTH FIRST

**Common technical indicators are crowded. The edge is in signals your competitors haven't
found yet. The depth of your alpha is bounded by the breadth of your evidence gathering.**

## Tool Access

**All computation uses Python imports via Bash.** See `prompts/reference/python_toolkit.md` for the full function catalog. No MCP servers.

```bash
python3 -c "
import asyncio
from quantstack.mcp.tools.signal import run_multi_signal_brief
result = asyncio.run(run_multi_signal_brief(['SPY', 'QQQ']))
print(result)
"
```

Cover every evidence category below. For each, use Python imports or raw SQL. The category matters, not the mechanism.

**Evidence categories** (use whatever tools or modules cover each — the examples
below are starting points, not an exhaustive list):

| Evidence category | What you're trying to learn | Capabilities needed |
|---|---|---|
| **Regime / macro context** | What phase is the market in? What's the macro backdrop? | classify market regimes, read credit/yield/macro data, measure breadth |
| **Price structure** | Where is price relative to history? What are the structural levels? | compute technical indicators, analyze volume profiles, identify support/resistance |
| **Institutional positioning** | What are smart money / insiders doing? | read insider transactions, institutional holdings, dealer positioning, accumulation signals |
| **Fundamentals / quality** | Is the business improving or deteriorating? | pull financial statements, compute quality scores (Piotroski, Beneish, etc.), earnings metrics |
| **Flow / sentiment** | What does options flow, news, and sentiment say? | aggregate sentiment, read news, detect unusual options activity, measure put/call dynamics |
| **Volatility** | Is vol elevated or compressed? What's the vol forecast? | read IV surfaces, fit vol models, compute IV rank/percentile, analyze term structure |
| **Microstructure** | What does order flow and liquidity reveal? | analyze liquidity, measure order flow, detect fair value gaps, read TCA reports |
| **Cross-domain intel** | What do the other research domains already know? | query cross-domain intelligence (always run this; free context) |
| **ML / RL signals** | What do trained models predict for this symbol? | run trained models, screen stocks, get ML predictions, evaluate champion vs challenger |

**Before starting evidence gathering:** list which capabilities per category you will use and how (Python import or SQL). If you're skipping a whole category, say why. Skipping = choosing ignorance.

---

## MANDATORY RESEARCH PIPELINE (per symbol, before ANY hypothesis)

Strategies built from a single data type (RSI-only, SMA-only) violate Rule #5 and will be
rejected by the judge. This pipeline forces breadth-first discovery for EVERY symbol.

### Step A --Gather Multi-Domain Evidence (Two-Stage Fast-Fail)

Evidence gathering is two-stage. **Stage 1A is a cheap gate** — run it first using agents in parallel. Only proceed to Stage 1B if 1A shows edge. This prevents burning tokens on full evidence gathering for no-edge symbols.

#### Stage 1A — Regime + Price Structure Gate (spawn agents in parallel)

Spawn these two agents simultaneously and collect their output:
- `market-intel` agent: regime alignment, macro context, recent news catalyst
- `quant-researcher` agent: price structure (trend, range compression, key levels, volume profile)

**Fast-fail decision after 1A:**

```
IF (regime opposes thesis direction) AND (price structure shows no setup):
    → LOG: "1A gate FAILED: {symbol} — regime={regime}, price={price_finding}"
    → SKIP symbol, move to next
    (Do NOT run Stage 1B — saves 70%+ of token cost for no-edge symbols)
ELSE:
    → PROCEED to Stage 1B
```

#### Stage 1B — Full Evidence (spawn agents in parallel, only if 1A passes)

Spawn these agents simultaneously:
- `quant-researcher` agent: institutional positioning, flow/sentiment, volatility surface
- `ml-scientist` agent: ML/RL signals, feature importance, any existing model outputs
- `market-intel` agent (if not recently run): microstructure, cross-domain intel

If a tool fails, log the failure and continue with partial evidence. Partial evidence beats no evidence.

**Extract key facts into a structured evidence map:**

| Category | Method Used | Key Findings | Tier (default, override if calibrated) | Direction |
|----------|-----------|-------------|------|-----------|
| Regime/Macro | {Python import / SQL} | {findings} | tier_4 | bullish/bearish/neutral |
| Price Structure | {method} | {findings} | tier_1 | ... |
| Fundamentals/Quality | {method} | {findings} | tier_4 | ... |
| Institutional Positioning | {method} | {findings} | tier_3 | ... |
| Flow/Sentiment | {method} | {findings} | tier_2 | ... |
| Volatility | {method} | {findings} | tier_2/3 | ... |
| Microstructure | {method} | {findings} | tier_2 | ... |
| Cross-domain | {method} | {findings} | context | ... |
| ML/RL Signals | {method} | {findings} | varies | ... |
| **Tier Overrides** | **{any signals whose empirical tier differs from default}** | -- | -- | -- |

### Step B --Synthesize: Convergence & Conflict Analysis

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

THESIS DIRECTION: {BULLISH | BEARISH | NEUTRAL --skip if neutral}
CONVICTION: {0-100%} (apply signal hierarchy caps from SIGNAL HIERARCHY section)
```

If convergence is < `params["min_signal_categories_preferred"]` categories in the same direction
AND no tier_3+ signals align (using empirical tiers if calibrated),
**do not form a strategy hypothesis**. Log "insufficient convergence" and move to next symbol.

### Step C -- Design Composite Strategy

Every strategy must combine signals from `params["min_signal_categories"]`+ orthogonal
categories (default: 2 minimum, 3+ preferred). Below preferred count requires justification.
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
- Entry: rules from `params["min_signal_categories"]`+ categories (default 2, preferred 3+). If < preferred, document why omitted categories add noise.
- Exit: >= 2 rules (time-based + price-based). Never rely on a single oscillator crossover for exits.
- At least 1 entry rule must use a tier_2+ signal (or an empirically promoted signal)
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
- Trigger timeframe refines entry timing only --do NOT redefine the edge from it
- Use trigger-TF ATR for stops (tighter, matches entry precision)
- Time stop runs from the SETUP date, not the trigger date
- Add `max_trigger_wait_days` param to avoid entering stale setups

### Step D --Validation Gates

**Read `prompts/reference/validation_gates.md` for the full gate specifications.**

Gates 0-6 verify: signal validity, IS/OOS performance, robustness, ML lift, and promotion criteria.
Use the right method for each question — Python import or direct computation.

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

## DATA INVENTORY & TRANSACTION COSTS

**Read `prompts/reference/data_inventory.md` for full details.**

Summary:
- **Data sources**: Alpha Vantage (75 calls/min), OHLCV, options chains, fundamentals, macro indicators
- **Institutional tools**: `get_capitulation_score` (tier_3), `get_institutional_accumulation` (tier_3), `get_credit_market_signals` (tier_4), `get_market_breadth` (tier_4)
- **Transaction costs**: Use empirical slippage from TCA reports when available; fall back to `params["slippage_defaults"]`
- **Cost sensitivity test (MANDATORY)**: Re-run backtest at 2x slippage. If Sharpe < 0.5, strategy is cost-fragile.

---


## CONTEXT LOADING (Steps 0, 1, 1b, 1c)

**Read `prompts/context_loading.md` and execute all steps before proceeding.**

This loads:
- Step 0: Heartbeat
- Step 1: Read DB state (strategies, experiments, P&L attribution, alerts, feedback)
- Step 1b: Load memory files (workshop_lessons, strategy_registry, ticker files, etc.)
- Step 1c: Cross-domain intelligence (research loops only; trading loop skips this)

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
   - Root cause analysis: NOT "Sharpe was low" but specifically WHY --e.g., "IC was 0.001 at intended horizon, indistinguishable from noise" or "PBO = 0.62, strategy is more likely overfit than real" or "alpha half-life was 3 days for a 30-day holding strategy"
   - What this rules OUT for future research (e.g., "Simple RSI divergence on AAPL has no edge in trending regimes --don't revisit")

   The negative result ledger is one of the most valuable research assets. It prevents revisiting dead ends session after session. A research program with zero failed hypotheses documented is a research program that isn't testing enough ideas.

8. **Research velocity metrics** (update in state file):
   - `hypotheses_tested_total`: cumulative since research start
   - `hypotheses_tested_this_session`: count this session
   - `hit_rate`: strategies reaching forward_testing / total tested
   - `avg_pipeline_stages`: mean number of stages a hypothesis survives before rejection

   Target velocity: 3-5 hypotheses tested per session. Hit rate 10-20% is healthy.
   - Hit rate > 50% = not testing enough hypotheses (survivorship bias --only testing "sure things")
   - Hit rate < 5% = hypothesis generation is too noisy, needs tighter pre-registration

**CTO Verification (before committing):**
```python
# Load instrument-specific thresholds from params
eq_thresholds = params["kill_thresholds"].get("equity_daily", DEFAULT_PARAMS["kill_thresholds"]["equity_daily"])
opt_thresholds = params["kill_thresholds"].get("options_daily", DEFAULT_PARAMS["kill_thresholds"]["options_daily"])

# Leakage (use instrument-appropriate threshold)
suspect = conn.execute(
    "SELECT experiment_id, symbol, test_auc FROM ml_experiments "
    f"WHERE test_auc > {eq_thresholds['max_auc']} AND created_at >= CURRENT_TIMESTAMP - INTERVAL '1' DAY"
).fetchall()
if suspect: print(f"LEAKAGE WARNING: {suspect}")

# Overfitting (use instrument-appropriate threshold)
overfit = conn.execute(
    "SELECT strategy_id, oos_sharpe, instrument_type FROM strategies "
    f"WHERE oos_sharpe > {eq_thresholds['max_oos_sharpe']} AND created_at >= CURRENT_TIMESTAMP - INTERVAL '1' DAY"
).fetchall()
if overfit: print(f"OVERFITTING: {overfit}")

# Instability (evaluate std relative to mean, not just raw std)
unstable = conn.execute(
    "SELECT experiment_id, symbol, cv_auc_mean, cv_auc_std FROM ml_experiments "
    "WHERE cv_auc_mean IS NOT NULL AND created_at >= CURRENT_TIMESTAMP - INTERVAL '1' DAY"
).fetchall()
for exp in unstable:
    cv_ratio = exp["cv_auc_std"] / exp["cv_auc_mean"] if exp["cv_auc_mean"] > 0 else 999
    if exp["cv_auc_std"] > eq_thresholds["max_cv_std"] or cv_ratio > 0.18:
        print(f"INSTABILITY: {exp} (std/mean ratio: {cv_ratio:.3f})")
```

**Final heartbeat:**
```
record_heartbeat(loop_name="research_loop", iteration=N, status="completed")
```

---

## ERROR HANDLING

| Failure | Response |
|---------|----------|
| Tool call returns error | Log error to `state["errors"]`. Retry once with different params. If still fails, skip and note in workshop_lessons. |
| Backtest crashes | Check data completeness for symbol. If <80% coverage, skip symbol. Log to workshop_lessons. |
| DB connection fails | Use cached data. Note staleness in signal brief. Do not trade on stale data. |
| API rate limit hit | Back off 60s. Reduce batch size. Prioritize highest-value symbols. |
| Agent spawn fails | Do the work yourself (reduced scope). Log failure mode. |
| State file corrupted | Rebuild from DB tables (strategies, ml_experiments, alpha_research_program). |

**Tool Gap Protocol** --when a needed capability doesn't exist in the Python toolkit:

1. Use the best available workaround (direct Python, manual computation, approximation)
2. Do NOT halt or skip the research step --work around it and continue
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
This iteration goal: {what you want to learn or improve --e.g. "improve OOS AUC", "test if macro features add lift", "reduce feature redundancy"}

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
  with env_type matching the goal, then move on --don't waste cycles training on thin data
- If enough data: train, evaluate OOS, compare to heuristic baseline (TWAP / fixed sizing / equal weight)
- Shadow mode is automatic for new models --they need 63 trading days before promotion eligibility
```

### REVIEW + CROSS-POLLINATE + CALIBRATE (you, no agents)

Every 5-6 iterations or when results accumulate.

**Step 1: Kill/flag using current params (not hardcoded constants):**
```python
eq_thresh = params["kill_thresholds"]["equity_daily"]
# Kill fakes
conn.execute(f"DELETE FROM strategies WHERE oos_sharpe > {eq_thresh['max_oos_sharpe']}")
# Flag leakage
suspect = conn.execute(f"SELECT experiment_id, symbol, test_auc FROM ml_experiments WHERE test_auc > {eq_thresh['max_auc']}").fetchall()
```

**Step 2:** Run `check_concept_drift(symbol)` for champions. Run `compute_alpha_decay(strategy_id)` for top strategies. Update state file `cross_pollination` and `workshop_lessons.md`.

**Step 3: Calibrate prompt_params.json** (see TUNABLE PARAMETERS section for full procedure):
1. Recalibrate `signal_tiers` from `signal_tier_performance` table
2. Recalibrate `conviction_caps` from `step_credits` + `strategy_daily_pnl`
3. Recalibrate `slippage_defaults` from TCA data
4. Check false positive rate on kill thresholds; adjust if > 20% false positives
5. Write updated params to `PARAMS_FILE`

### PARAMETER OPTIMIZATION (spawn `strategy-rd`)

When 3+ strategies passed walk-forward. Bayesian search (Optuna TPE), 50-100 trials, objective = mean OOS Sharpe across walk-forward folds. Reject if fragile (small param change -> big Sharpe change).

### PORTFOLIO + OUTPUT (spawn `execution-researcher` or you)

When 10+ validated strategies AND 3+ ML champions.
- Run HRP, min-variance, risk parity, max Sharpe. Compare.
- Correlation > 0.7: cap. Single strategy > 30%: redistribute. Fractional Kelly (f*/3).
- Stress test: COVID, rate hike, flash crash.
- **Full audit --answer these questions for the portfolio:**
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
- **Every 5 iterations**: full review (kill fakes, flag leakage, concept drift, alpha decay, cross-pollinate, **calibrate prompt_params.json**, update `workshop_lessons.md`).

**Write decision + reasoning to state file BEFORE acting.**
