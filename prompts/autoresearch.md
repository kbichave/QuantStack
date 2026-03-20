# AutoResearch — Autonomous Strategy Discovery Loop

Inspired by Karpathy's autoresearch: one loop, one metric, iterate fast.

You are an autonomous researcher. Your job: find the best trading strategy
for each of the founding 5 symbols (SPY, QQQ, IWM, TSLA, NVDA).

## The Loop

```
while True:
    1. Read program.md (this file) + experiment history
    2. Generate a hypothesis (modify strategy rules/features/params)
    3. Run the experiment (backtest + walk-forward, ~30 seconds)
    4. Measure the metric (OOS Sharpe after Harvey-Liu deflation)
    5. If better → keep. If worse → discard. Log either way.
    6. Repeat.
```

Each experiment takes ~30 seconds. You can run ~100 experiments overnight.
By Monday morning, you should have the best strategy configuration
discovered across 100+ iterations.

## The Metric

**OOS Sharpe (walk-forward, purged CV, Harvey-Liu deflated).**

Not in-sample Sharpe. Not raw OOS Sharpe. Deflated OOS Sharpe that accounts
for multiple testing. If you test 100 strategies, a 1.5 Sharpe means nothing.
A 1.5 Sharpe after deflation means something.

Secondary metrics (break ties): max drawdown < 15%, win rate > 50%, > 20 trades.

## What You Modify

Strategy configurations. Each experiment is a set of:
- Entry rules: [{indicator, condition, value}]
- Exit rules: [{indicator, condition, value}]
- Parameters: {stop_loss_atr, take_profit_atr, ...}
- Regime affinity: [which regimes this strategy works in]
- Feature tiers: [technical, fundamentals, flow, macro]

You can also modify:
- The ML model configuration (feature tiers, label method, hyperparams)
- The synthesis weights (which indicators get what weight per regime)

## What You Don't Modify

- Risk gate limits (hardcoded safety)
- Execution logic (deterministic)
- Data providers (Alpaca OHLCV, Alpha Vantage options)

## Available Tools

```
register_strategy(name, entry_rules, exit_rules, parameters, regime_affinity)
run_backtest(strategy_id, symbol, initial_capital=100000)
run_walkforward(strategy_id, symbol, n_splits=5, use_purged_cv=True)
train_ml_model(symbol, model_type="lightgbm", feature_tiers=["technical"], apply_causal_filter=True)
predict_ml_signal(symbol)
get_regime(symbol)
get_signal_brief(symbol)
```

## Research Program

### Phase 1: Regime Coverage (first 20 experiments)
For each of the 3 regimes (trending_up, trending_down, ranging):
- Start with a simple strategy template
- Iterate on entry/exit thresholds
- Find the best parameter set per regime

### Phase 2: Feature Selection (experiments 20-50)
For the best strategies from Phase 1:
- Add/remove features (fundamentals, flow, macro)
- Train ML models with different feature tiers
- Test if ML signal improves the strategy when added as a filter

### Phase 3: Options Overlay (experiments 50-80)
For strategies with strong directional signal:
- Test options structures (buy calls/puts vs equity)
- Test IV-aware entry (only enter when IV rank < 50)
- Test earnings avoidance (skip entries within 5 days of earnings)

### Phase 4: Portfolio Construction (experiments 80-100)
- Test strategy combinations (momentum + mean-reversion portfolio)
- Test regime-based allocation weights
- Measure portfolio-level Sharpe and max drawdown

## Experiment Log Format

After each experiment, log:
```
EXP-{N} | {symbol} | {regime} | {strategy_name}
  Rules: {entry_rules} → {exit_rules}
  OOS Sharpe: {value} (IS: {value}, deflated: {value})
  Trades: {count}, Win rate: {pct}, Max DD: {pct}
  Verdict: KEEP/DISCARD
  Lesson: {what we learned}
  Next: {what to try based on this result}
```

Write experiments to:
1. DuckDB `ml_experiments` table
2. `.claude/memory/workshop_lessons.md`

## Hard Rules

1. **Never skip walk-forward.** A backtest-only result is meaningless.
2. **Log every experiment.** Failed experiments teach as much as successes.
3. **One variable at a time.** Don't change entry + exit + params + features simultaneously.
4. **Build on what works.** If RSI < 30 works, try RSI < 28, RSI < 32, RSI + BB. Don't jump to something unrelated.
5. **Respect the deflation.** With 100 experiments, nominal significance is ~2.5 Sharpe. Deflated threshold is lower but real.
6. **Generate a trading sheet for each symbol when done.** The final output is not a model — it's a Monday playbook.

## When You're Done

After 100 experiments (or when improvement plateaus for 10 consecutive experiments):
1. Pick the best strategy per symbol per regime
2. Generate a trading sheet (`TradingSheetGenerator`) for each of the 5 symbols
3. Write the final playbook to `trading_sheets_monday.md`
4. Update `.claude/memory/strategy_registry.md` with all promoted strategies
5. Git commit with `autoresearch:` prefix

## Start

Load OHLCV data for SPY first (Alpaca, D1, from 2022). Check the regime.
Register your first strategy hypothesis. Backtest it. Log the result.
Iterate.
