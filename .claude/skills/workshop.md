---
name: workshop
description: Strategy R&D session — hypothesize, backtest, validate, and register trading strategies.
user_invocable: true
---

# /workshop — Strategy Research & Development Session

## Purpose

Research, backtest, and validate new trading strategies.  Every strategy goes
through the same lifecycle: hypothesis → backtest → walk-forward → register
(or reject).  This skill enforces that discipline.

## Workflow

### Step 0: Read Context + Infrastructure Check
Before any tool calls:
- Run `python scripts/check_ollama_health.py` — abort if models not loaded.
  The trading crew used in backtesting depends on local models being resident.
- Verify AWS credentials for Bedrock (used for deep workshop reasoning):
  `aws sts get-caller-identity` — confirm it returns an ARN without error.
  If credentials are expired, deep hypothesis reasoning will fall back to
  Ollama (weaker). Note this in the session if degraded mode is active.
- Read `.claude/memory/workshop_lessons.md` — don't repeat failed hypotheses
- Read `.claude/memory/strategy_registry.md` — what strategies exist, what gaps remain
- Read `.claude/memory/regime_history.md` — what's the current regime?

### Step 1: Regime Check
Call `get_regime(symbol)` to confirm the current market regime.
- Record: trend_regime, volatility_regime, confidence.
- Cross-reference with the regime-strategy matrix in CLAUDE.md.

### Step 2: Identify Gap
Compare active strategies in the registry against the current regime.
- Which regime-strategy slots are empty?
- Which existing strategies are underperforming (check backtest_summary)?
- Is there a new hypothesis worth testing?

### Step 3: Hypothesize
Define entry and exit rules as structured dicts:
```json
{
  "entry_rules": [
    {"indicator": "rsi", "condition": "crosses_below", "value": 30, "direction": "long"},
    {"indicator": "sma_crossover", "condition": "crosses_above", "direction": "long"}
  ],
  "exit_rules": [
    {"indicator": "rsi", "condition": "crosses_above", "value": 70}
  ],
  "parameters": {
    "rsi_period": 14,
    "sma_fast": 10,
    "sma_slow": 50,
    "atr_period": 14
  },
  "risk_params": {
    "stop_loss_atr": 2.0,
    "take_profit_atr": 3.0,
    "position_pct": 0.05
  }
}
```

Supported indicators: `rsi`, `sma_crossover`, `sma_fast`, `sma_slow`, `atr`,
`bb_pct`, `bb_upper`, `bb_lower`, `zscore`, `breakout`, `close`, `high_n`, `low_n`.

Supported conditions: `above`, `below`, `crosses_above`, `crosses_below`,
`greater_than`, `less_than`, `between`.

### Step 4: Register Draft
Call `register_strategy` with the hypothesis.
- Status starts as "draft".
- Set `regime_affinity` to indicate which regimes this strategy targets.
- Set `source` to "workshop".

### Step 5: Backtest
Call `run_backtest(strategy_id, symbol)`.

**Evaluate against thresholds:**
- Sharpe ratio > 1.0?
- Max drawdown < 15%?
- Total trades > 50? (statistical significance)
- Profit factor > 1.3?

**Split-half stability check** (manual):
- Run backtest on first half of date range
- Run backtest on second half
- If Sharpe differs by more than 50%, the strategy may be regime-dependent
  (which is fine if regime_affinity reflects this)

### Step 6: Walk-Forward Validation (if backtest passes)
Call `run_walkforward(strategy_id, symbol)`.

**Evaluate:**
- OOS Sharpe > 0.5?
- OOS degradation < 50% from IS?
- OOS positive folds >= 60% of total folds?
- Overfit ratio < 2.0?

### Step 7: Outcome

**If passes walk-forward:**
- Call `update_strategy(strategy_id, status="forward_testing")`
- This strategy is now eligible for paper trading in /trade sessions

**If fails:**
- Call `update_strategy(strategy_id, status="failed")`
- Log the failure in `.claude/memory/workshop_lessons.md` under "Failed Hypotheses"
- Record: what was tried, why it failed, the regime, the key metric that missed

### Step 8: Iterate
If the first hypothesis failed:
- Review what went wrong (check workshop_lessons.md)
- Adjust parameters or rules
- Go back to Step 3 with a refined hypothesis
- Maximum 3 iterations per session to avoid overfitting

### Step 9: Update Memory
- `.claude/memory/strategy_registry.md` — add/update the strategy entry
- `.claude/memory/workshop_lessons.md` — record findings:
  - What worked and why
  - What failed and why
  - Parameter sensitivity observations
  - Regime-specific notes
- `.claude/memory/session_handoffs.md` — if /meta or /trade needs to know about the new strategy

## ML-Backed Strategy Path (alternative to rule-based)

Use when rule-based hypotheses have failed 2+ iterations with Sharpe < 0.5.

**Step 3 (alternate): Train an ML model to discover entry conditions**

```python
from quantcore.labeling.event_labeler import EventLabeler
from quantcore.models.trainer import ModelTrainer
from quantcore.models.explainer import SHAPExplainer
from quantcore.data.storage import DataStore

store = DataStore()
df = store.load_ohlcv(symbol, timeframe)

labeler = EventLabeler(tp_atr_multiple=2.5, sl_atr_multiple=1.5, max_hold_bars=10)
labeled = labeler.label(df)

trainer = ModelTrainer(model_type="lightgbm")
model, metrics = trainer.train(labeled, target_col="label", cv_folds=5)

explainer = SHAPExplainer(model)
importance = explainer.get_global_importance(labeled)
# top SHAP features reveal which conditions actually predict outcomes
```

**Decision rule after training:**
- OOS accuracy > 55% AND top SHAP features are interpretable →
  convert the top 3 features into explicit rule-based entry conditions,
  then proceed with register_strategy as normal
- OOS accuracy < 52% → signal is noise, document in workshop_lessons.md
- Top SHAP features are lagged price only → autocorrelation artefact, discard

**Do not register an ML model directly as a strategy.** Convert its insights
into auditable rules that can be regime-gated and inspected.
Use `HMMRegimeModel` or `TFTRegimeModel` for regime classification if
`get_regime()` confidence < 0.6.

---

## Tool Gap Protocol

If a capability needed for a workshop step doesn't exist in the MCP tools:
1. Use the best available workaround (direct Python, manual computation, etc.)
2. Document the gap in the session's Final Report under "Missing Tools":
   ```
   Missing: <tool_name>
   Input: <schema>
   Output: <schema>
   Package: quantcore or quant_pod
   Priority: HIGH/MEDIUM (HIGH if it unlocks a whole strategy class)
   ```
3. Do NOT halt or skip steps — work around it, document it, and continue.
These specs are reviewed after workshop sessions and built into QuantCore or QuantPod
before the next session.


## Multi-Timeframe Strategy Design

When 1D signals fire but trade count or precision is insufficient, consider a
cross-timeframe approach: **higher-timeframe setup + lower-timeframe trigger**.

Standard MTF patterns:
- **1D setup + 1H trigger**: Most common. Daily oversold/overbought sets up the trade;
  1H momentum reversal times the entry. Reduces false starts, tighter stops.
- **4H setup + 15min trigger**: More setups than 1D. Good for strategies needing
  higher trade count (>200/year).
- **1W setup + 1D trigger**: Macro regime as setup, daily signal as entry.
  Very low trade count but high conviction.

MTF design rules:
- Setup timeframe defines the edge (tested and validated at that TF first)
- Trigger timeframe refines entry timing only — do NOT use it to redefine the edge
- Use trigger-TF ATR for stops (tighter, appropriate for the entry precision)
- Time stop runs from the SETUP date (not trigger date) to bound total holding time
- Add `max_trigger_wait_days` param to avoid entering stale setups

MTF backtesting, MTF walk-forward, sparse-signal walk-forward, and options convexity
backtesting are all available via QuantPod MCP — use them directly.

## Anti-Overfitting Checklist

Before promoting any strategy to forward_testing:
- [ ] Total trades > 50 in backtest
- [ ] Walk-forward OOS Sharpe > 0 in majority of folds
- [ ] No parameter was tuned to a single data point
- [ ] Strategy logic is explainable in one sentence
- [ ] Entry/exit rules use different indicators (not the same twice)
- [ ] Risk params include stop loss (no open-ended risk)

## Notes

- Never optimize parameters to maximize backtest Sharpe — that's overfitting.
  Instead, pick parameters from domain knowledge, then validate.
- The backtest engine uses the existing `BacktestEngine` from quantcore.
  It's a single-instrument, signal-based engine with slippage + commission.
- Workshop sessions should produce either a registered strategy or a documented
  lesson. Never walk away with neither.
