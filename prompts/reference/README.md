# Reference Documentation

This directory contains shared reference content extracted from the main prompt files to eliminate duplication and improve maintainability.

## Files

### `validation_gates.md`
Strategy validation gates (0-6) referenced by all research domain prompts.
- Gate 0: Register strategy with economic mechanism
- Gate 1: Signal validity (IC, alpha decay, stationarity)
- Gate 2: In-sample performance
- Gate 3: Out-of-sample consistency (OOS Sharpe, PBO, deflated Sharpe)
- Gate 4: Robustness (cost sensitivity, stress tests, regime stability)
- Gate 5: ML/RL lift (SHAP, CausalFilter, champion vs challenger)
- Gate 6: Update DB and memory files

### `completion_gate.md`
Criteria for declaring `<promise>TRADING_READY</promise>` across all domains.
- Per-domain thresholds (equity investment, swing, options)
- Cross-domain portfolio requirements
- 30-iteration gap analysis procedure

### `data_inventory.md`
Available data sources, institutional tools, and transaction cost defaults.
- Alpha Vantage coverage (OHLCV, options, fundamentals, macro)
- Tier_3/tier_4 tools (capitulation, institutional accumulation, credit signals, breadth)
- Empirical slippage methodology and defaults

### `trading_rules.md`
Position sizing, holding periods, error handling for trading loop.
- Equity and options position sizing by conviction
- Time horizon-specific holding periods and exit triggers
- Investment-grade exit criteria (F-Score, earnings, insider selling)
- Error handling for tool failures

## Usage

Main prompt files now reference these files instead of duplicating content:

- **research_shared.md** → references `validation_gates.md`, `data_inventory.md`
- **research_loop.md** → references `completion_gate.md`
- **trading_loop.md** → references `trading_rules.md`
- All domain prompts → reference `validation_gates.md` for Step D

## Rationale

**Before:** Each prompt file duplicated 100-300 lines of validation gates, data inventory, position sizing, etc. Changes required updating 5-7 files.

**After:** Shared content lives in one canonical location. Updates propagate automatically. Main prompts stay focused on their specific workflows.
