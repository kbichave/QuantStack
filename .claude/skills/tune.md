---
name: tune
description: Improve IC and pod manager prompts based on accumulated accuracy data from reflect sessions. Run after 3+ reflect sessions or when agent_performance.md shows a persistently weak IC.
user_invocable: true
---

# /tune — IC & Pod Manager Prompt Improvement Session

## Purpose

The CrewAI agents that run inside `run_analysis`, `run_ic`, and `run_pod` are only
as good as their prompts. This session reads accumulated IC accuracy data from
`agent_performance.md` and makes targeted edits to the JSON agent definitions and
`tasks.yaml` task descriptions.

**When to run:**
- After 3+ /reflect sessions have accumulated IC accuracy data
- When `agent_performance.md` shows an IC with accuracy < 50% for 2+ sessions
- When a specific IC's Known Biases list has grown to 3+ items
- When you observe the crew producing low-quality DailyBriefs (weak conviction, vague levels)

**What you edit:**
| File | Controls |
|------|----------|
| `packages/quant_pod/prompts/ics/<pod>/<ic_name>.json` | Agent `role`, `goal`, `backstory`, `tools`, `max_iter` |
| `packages/quant_pod/prompts/pod_managers/<name>.json` | Pod manager synthesis focus, delegation behavior |
| `packages/quant_pod/prompts/assistant/trading_assistant.json` | How the assistant weights and synthesizes pod outputs |
| `packages/quant_pod/crews/config/tasks.yaml` | Task `description` (what the IC is asked per run) and `expected_output` (format contract) |

**What you never edit in tune:**
- `settings.llm` — LLM assignment lives in `llm_config.py`, not per-JSON
- `settings.max_iter` beyond 30 — runaway loops
- `allow_delegation` on ICs (must be False — ICs are leaf nodes)
- Tool lists on ICs without verifying the tool exists in `mcp_bridge.py`

---

## Workflow

### Step 1: Read Performance Data
- Read `.claude/memory/agent_performance.md` — IC accuracy, ICIR, Known Biases
- Read `.claude/memory/session_handoffs.md` — any IC issues flagged in prior sessions
- Read `.claude/memory/workshop_lessons.md` — did any IC output contribute to a strategy failure?

Identify the **target ICs**: those with at least one of:
- Rolling accuracy < 50% over last 3 sessions
- Known Biases list ≥ 3 items
- Explicitly flagged in session_handoffs as "unreliable" or "missing X"
- Output described as "vague" or "unhelpful" in trade_journal

If no ICs meet any threshold: **STOP**. No evidence = no edit.
Log "No ICs require tuning" in session_handoffs.md and exit.

### Step 2: Read the Current Prompt

For each target IC, read both files:

```
packages/quant_pod/prompts/ics/<pod>/<ic_name>.json   ← agent definition
packages/quant_pod/crews/config/tasks.yaml             ← task instructions
```

The agent JSON controls *who the agent is* (persona, tools available).
The task YAML controls *what it's asked to do this run* (instructions + output format).

Most accuracy problems come from the task YAML — vague `description` or
missing fields in `expected_output`. Fix the task first before touching the agent.

### Step 3: Diagnose the Root Cause

For each weak IC, determine which of these is the problem:

| Symptom | Root cause | Fix location |
|---------|-----------|-------------|
| Output is vague / missing key metrics | `expected_output` in tasks.yaml doesn't specify format | tasks.yaml |
| IC returns interpretation instead of raw data | `backstory` says "analyst" instead of "return raw values" | JSON backstory |
| IC misses a signal type we care about | `description` in tasks.yaml doesn't ask for it | tasks.yaml description |
| IC uses wrong tool for the job | `tools` list in JSON missing the right QuantCore tool | JSON tools array |
| IC output is correct but pod manager ignores it | Pod manager `goal`/`backstory` doesn't emphasize it | Pod manager JSON |
| Trading assistant underweights a signal | `trading_assistant.json` backstory weights wrong | assistant JSON |

### Step 4: Edit Prompts (One IC at a Time)

**Rules for editing:**
1. Change ONE thing per IC per session — don't rewrite the whole prompt
2. The change must be motivated by specific evidence from agent_performance.md
3. Keep `backstory` factual and constraint-focused (what to return, what NOT to do)
4. Add Known Biases as explicit constraints in the backstory:
   - Example: known bias "overstates RSI significance in trending regimes" →
     add to backstory: "In trending regimes (ADX > 25), RSI is less reliable —
     note this explicitly when ADX > 25."
5. `expected_output` in tasks.yaml is a FORMAT CONTRACT — be specific:
   - Bad:  "Return regime analysis."
   - Good: "Return: Regime: [trending/ranging/unknown], ADX: [value],
     Confidence: [0-1 float], supporting: [2 bullet metrics]"

**Typical edits:**

*Tighten expected_output format:*
```yaml
# Before
expected_output: >
  Raw regime classification with numeric metrics.

# After
expected_output: >
  Exactly 4 lines:
  Regime: [trending_up | trending_down | ranging | unknown]
  ADX: [float, 2 decimal places]
  ATR_pct: [integer, 0-100]
  Confidence: [float 0.0-1.0]
  Do not add interpretation or trade recommendations.
```

*Add a known bias constraint to backstory:*
```json
// Before
"backstory": "You classify market regimes using ADX and ATR percentiles. Return raw labels."

// After
"backstory": "You classify market regimes using ADX and ATR percentiles. Return raw labels.
  CONSTRAINT: When RSI < 35 is present, do not assume mean-reversion regime —
  oversold conditions occur most frequently DURING trending_down, not ranging.
  Always report ADX value alongside regime label to let the caller judge."
```

*Add a missing tool:*
```json
// If regime_detector_ic should also check HMM state transitions:
"tools": ["get_market_regime_snapshot", "compute_all_features", "compute_indicators",
          "get_symbol_snapshot", "run_adf_test"]
```

### Step 5: Validate Tool References

After any JSON edit that modifies `tools`, verify the tool name exists:

```bash
grep -n "def <tool_name>\|name.*=.*\"<tool_name>\"" packages/quant_pod/tools/mcp_bridge.py
```

If the tool doesn't exist in mcp_bridge.py, do NOT add it to the IC's tools list.
Add a note in session_handoffs.md: "IC wants X tool but it's not bridged yet."

### Step 6: Test the Edit

Run the edited IC in isolation to confirm the output format improved:

```
run_ic("<ic_name>", "SPY")
```

Compare the output against the new `expected_output` format you defined.
- If output matches the new format: edit is good, proceed
- If output still doesn't match: the task description needs to be more explicit,
  or the issue is in the agent backstory — make one more targeted edit

### Step 7: Update Memory and Commit

**`.claude/memory/agent_performance.md`:**
- Under the IC's entry, add: `Last tuned: [date], Change: [one sentence what changed]`
- Clear items from Known Biases that the edit addresses

**`.claude/memory/session_handoffs.md`:**
Log every file changed with the same format as other self-modification entries:
```
| [date] | packages/quant_pod/prompts/ics/.../X.json | [what changed] | [why — evidence from agent_performance.md] | [test result from run_ic] |
```

**Commit:**
```bash
git add packages/quant_pod/prompts/ packages/quant_pod/crews/config/tasks.yaml
git add .claude/memory/agent_performance.md .claude/memory/session_handoffs.md
git commit -m "skill: tune [ic_name] — [one sentence what changed and why]"
```

---

## IC Prompt Reference

| IC | JSON location | Pod | Primary job |
|----|--------------|-----|-------------|
| `data_ingestion_ic` | ics/data/ | data | Fetch OHLCV, report data quality |
| `market_snapshot_ic` | ics/market_monitor/ | market_monitor | Current price/volume snapshot |
| `regime_detector_ic` | ics/market_monitor/ | market_monitor | ADX/ATR regime classification |
| `trend_momentum_ic` | ics/technicals/ | technicals | RSI, MACD, ADX, MA values |
| `volatility_ic` | ics/technicals/ | technicals | ATR, Bollinger, VaR |
| `structure_levels_ic` | ics/technicals/ | technicals | Support/resistance levels |
| `statarb_ic` | ics/quant/ | quant | ADF test, IC/ICIR |
| `options_vol_ic` | ics/quant/ | quant | IV, Greeks, skew |
| `fundamentals_ic` | ics/quant/ | quant | Earnings, valuation |
| `news_sentiment_ic` | ics/quant/ | alpha_signals | News sentiment scoring |
| `options_flow_ic` | ics/quant/ | alpha_signals | Unusual options activity |
| `risk_limits_ic` | ics/risk/ | risk | VaR, stress tests, limits |
| `calendar_events_ic` | ics/risk/ | risk | FOMC, earnings, events |

## Evolution Rules

- Never tune an IC without evidence from agent_performance.md (3+ sessions minimum)
- One change per IC per session — multiple simultaneous changes make it impossible
  to know which one improved (or broke) the output
- After tuning, the IC needs 3+ real run_ic calls before accuracy can be re-assessed
- If an edit makes output WORSE (measured in next /reflect): revert it and log why
