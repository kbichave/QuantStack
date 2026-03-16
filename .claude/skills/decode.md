---
name: decode
description: Reverse-engineer trading strategies from external signal history. Decode, formalize, backtest, and register.
user_invocable: true
---

# /decode — Strategy Decoder Session

## Purpose

Reverse-engineer a trading strategy from raw trade signals. Turn someone
else's track record into a systematic, backtestable strategy definition.

## Workflow

### Step 0: Read Context
- Read `.claude/memory/strategy_registry.md` — avoid decoding what we already have
- Read `.claude/memory/workshop_lessons.md` — known pitfalls apply here too

### Step 1: Ingest Raw Signals
Accept signals from one of:
- **Manual paste:** user provides a list of trades
- **File upload:** CSV/JSON with trade history
- **`decode_from_trades`:** analyze our own closed_trades or fills

Each signal must have: `symbol`, `direction`, `entry_time`, `entry_price`,
`exit_time`, `exit_price`. Optional: `size`, `notes`.

### Step 2: Clean and Normalize
- Validate required fields are present
- Parse timestamps (supports multiple formats)
- Flag and report missing/invalid data
- Minimum 20 signals recommended; warn if fewer

### Step 3: Decode via MCP
Call `decode_strategy(signals, source_name)`.

Review the output:
- **entry_trigger**: What timing and conditions drive entries?
- **exit_trigger**: Time-based or target-based exits?
- **timing_pattern**: Morning, midday, afternoon, or all-day?
- **style**: Scalper, intraday, swing, or position?
- **regime_affinity**: Which regimes favor this strategy?
- **edge_hypothesis**: The one-sentence thesis

### Step 4: Evaluate the Decoded Strategy
Ask yourself:
- Does the edge hypothesis make sense from a market microstructure perspective?
- Is the timing pattern exploitable systematically (or just coincidence)?
- Is the win rate sustainable, or is it likely overstated by survivorship bias?
- Is the sample size sufficient (>50 signals ideal)?

### Step 5: Formalize into StrategyDefinition
Translate the decoded patterns into formal entry/exit rules:
```json
{
  "entry_rules": [
    {"indicator": "<indicator from decoded entry_trigger>",
     "condition": "<condition>", "value": <threshold>, "direction": "long"}
  ],
  "exit_rules": [
    {"indicator": "<exit trigger indicator>",
     "condition": "<condition>", "value": <threshold>}
  ],
  "parameters": {"<relevant indicator params>": <values>},
  "risk_params": {"stop_loss_atr": 2.0, "position_pct": 0.03}
}
```
Set conservative risk params initially — decoded strategies have
higher uncertainty than workshop strategies.

### Step 6: Register
Call `register_strategy` with `source="decoded"`, `status="draft"`.

### Step 7: Backtest
Call `run_backtest(strategy_id, symbol)`.
Apply the same criteria as /workshop:
- Sharpe > 1.0?
- Max drawdown < 15%?
- Total trades > 50?

### Step 8: Validate or Reject
If backtest passes → `run_walkforward` → if passes:
  `update_strategy(status="forward_testing")`
If fails → `update_strategy(status="failed")`

### Step 9: Update Memory
- `.claude/memory/strategy_registry.md` — add decoded strategy
- `.claude/memory/workshop_lessons.md` — new patterns discovered
- `.claude/memory/session_handoffs.md` — notify /meta and /trade

## Notes
- Decoded strategies should be treated with MORE skepticism than workshop
  strategies. The original trader may have had information we don't have.
- Always check: is the edge in the timing, the signal, or the risk management?
  Timing edges are hardest to systematize.
- Sample size matters enormously. 20 trades is the absolute minimum.
  50+ is preferred. 100+ is high confidence.
