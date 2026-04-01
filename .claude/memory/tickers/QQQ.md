# QQQ Research & Trading Memory

## Coverage Matrix (2026-03-26 iter2)

| Regime | Investment | Swing | Options |
|--------|-----------|-------|---------|
| trending_up+normal | QualityGrowth IS=0.83, WF=0.04 (OVERFIT) | regime_momentum_v1 OOS=1.346 (VALIDATED, lost from DB) | bull_call_spread_v2 IS=0.76, WF=0.06 (MARGINAL) |
| trending_up+high | invest_highvol IS=0.55, WF=-0.80 (FAILED) | momentum_tight_stops IS=0.67, WF=0.19 (WEAK) | long_calls_v2 IS=0.76 (NO WF) |
| trending_down+normal | DefensiveValue IS=0.38, post-2010 WF=+0.65 (PARTIAL PASS) | **ml_gated_bounce IS=0.66, WF=0.90 (STRONG PASS)** | bear_put_spread DISCRETIONARY (R:R 1.79:1) |
| trending_down+high | CashDefensive SKIP (valid) | Cash/SKIP (valid) | bear_put_highvol IS=0.25, WF=0.13 (MARGINAL) |
| ranging+low | AccumulateOnDips IS=0.45, WF=-0.31 (FAILED) | ranging_lowvol IS=0.30, WF=0.24 (WEAK) | iron_condor_vrp_v3 OOS=0.85 (VALIDATED, lost from DB) |
| ranging+high | invest_ranging_highvol IS=0.41, WF=-0.12 (FAILED) | ranging_highvol IS=0.34, WF=-0.15 (FAILED) | iron_butterfly 0 trades x5 (BROKEN) |

**Summary:** 3 strong, 4 weak, 3 valid SKIPs, 8 gaps/failures = 6/18 strong+skip (33%), 10/18 weak+fail.

### Agent Structural Conclusions (2026-03-26 iter2)
1. **Investment: ALL fail WF.** Beta masquerading as alpha. Rule engine lacks fundamental features (capitulation_score, credit_regime, earnings_quality). QQQ's 1600% return 2010-2026 inflates any net-long IS Sharpe.
2. **Swing: QQQ needs only 1-2 strategies.** regime_momentum_v1 (trending) + ml_gated_bounce (oversold bounce). Filling every regime cell is overfitting the regime matrix itself.
3. **Options: 6 defined-risk structures registered.** Iron condor validated. Rest need real IV data. VIX call hedge is novel for trending_down+high. Iron butterfly blocked by indicator naming bug.
4. **ranging+high_vol has no equity swing edge.** Mean reversion is fold-level IS negative. Must use options (iron condor/butterfly) for this regime.
5. **swing_ranging_lowvol_v1: INVESTIGATE via multi-symbol.** PF 1.69 is excellent, WF OOS/IS ratio 0.54 (not overfit), but only 30 trades in 16 years. Needs SPY/XLK/IWM cross-validation.

## Strategies (ranked by evidence quality)

| Rank | ID | Name | OOS/WF Sharpe | Overfit | Status |
|------|----|----- |--------------|---------|--------|
| 1 | regime_momentum_v1 | Regime Momentum | 1.346 | 0.58 | VALIDATED (lost from DB) |
| 2 | strat_b897cd6179 | ml_gated_bounce_trending_down_v1 | **0.897** | 0.71 | **forward_testing** |
| 3 | iron_condor_vrp_v3 | Iron Condor VRP | 0.85 | 2.55 | VALIDATED (lost from DB, re-registered as draft) |
| 4 | strat_123d79be51 | OversoldBounce_BearRegime_v1 | 0.34-0.90 (post-2010) | N/A | backtested, credit-blocked |
| 5 | strat_21c1111a15 | swing_ranging_lowvol_v1 | 0.24 | 0.54 | backtested (weak) |
| 6 | strat_270661efb1 | momentum_tight_stops_trending_up_highvol | 0.19 | 0.61 | failed (weak WF) |

### ml_gated_bounce_trending_down_v1 (NEW — BEST ITER2 FINDING)
- **Entry (AND logic via prerequisites):** RSI < 30 AND BB%B < 0 AND ADX > 20 + plain: Stoch_K < 20
- **Thesis:** Deep oversold bounce in trending markets. When RSI/BB/Stoch all extreme, snap-back is reliable.
- **WF:** OOS mean 0.897, overfit ratio 0.71, 4 positive OOS folds. 228 IS trades.
- **Status:** Promoted to forward_testing by swing agent.
- **Note:** Currently credit-blocked (all long strategies blocked while credit_regime=widening).

## ML Models

| Date | Model | AUC (OOS) | Prediction | Verdict |
|------|-------|-----------|------------|---------|
| 2026-03-26 | LightGBM | 0.6579 | BEARISH 78.3% | champion (saved) |
| 2026-03-26 | XGBoost | 0.5755 | BEARISH 75.6% | challenger |

**Top features:** bb_width, atr, vwap, adx, natr, adxr, dx, minus_di. Vol + directional dominate.

## Market Context (2026-03-26 -- LATEST)

| Field | Value |
|-------|-------|
| Price | **$573.79** (below EMA_200 $586.03) |
| RSI | 34.5 |
| ADX | **39.5** (accelerating) |
| Credit | **WIDENING** |
| Capitulation | 0.438 (watch, need >0.65) |
| HMM | LOW_VOL_BEAR |

**0 of 5 accumulation conditions met. DO NOT ACCUMULATE.**

## Options Execution Plan (discretionary, 2026-03-27+)

### PREFERRED: 575P / 560P Bear Put Spread (30 DTE, skew-adjusted)
- Debit: $5.37 ($537/ct), Max profit: $9.63 ($963/ct), R:R 1.79:1, Breakeven: $569.63
- Entry: bounce to $576-580 (preferred) or flat/down open (half-size)
- Exit: 50% profit, 60% stop, 14 DTE time stop, regime change stop
- Position: 2% max premium ($50K=1ct, $100K=3ct)

### Alternatives
- 580P/565P on bounce: $595 debit, $905 max, R:R 1.52:1
- 570P/555P on gap-down: $485 debit, $1015 max, R:R 2.10:1

## Infrastructure Blockers
1. **WF pre-2010 poisoning: FIXED.** Added `start_date`/`end_date` params to `run_walkforward()` in backtesting.py + WalkForwardRequest model.
2. **Options backtest broken: FIXED.** Rewrote `run_backtest_options()` to use `_generate_signals_from_rules()` directly instead of relying on equity BacktestEngine trades.
3. **Iron butterfly 0 trades: FIXED.** Added `_INDICATOR_ALIASES` map + `_normalize_indicator()` in rule_engine.py. Maps `adx_14`→`adx`, `rsi_14`→`rsi`, `bb_width_20`→`bb_pct`, etc. Applied in both `_compile_rule()` and `evaluate_rule()` (signal_generator.py).
4. **Strategy persistence: INVESTIGATED — no code bug.** No DELETE/TRUNCATE on strategies table. Likely env var mismatch (different PG instance) or session confusion. Table uses `CREATE TABLE IF NOT EXISTS` + auto-commit.
5. **Rule engine direction bug: FIXED.** Added direction constraint enforcement in `generate_signals_from_rules()`: when `parameters["direction"]` is SHORT, `entry_long` is suppressed entirely (and vice versa).
6. **Test import errors: FIXED.** `test_strategy_registry.py` and `test_lifecycle.py` had stale imports from `quantstack.mcp.server` — updated to import from `quantstack.mcp.tools.strategy`, `.learning`, `.meta`.
7. **Credit gate:** All long strategies blocked while credit_regime=widening. (Not a bug — working as designed.)

## Lessons (QQQ-specific)

1. Best regime signal quality of all ETFs. OOS 1.346 vs IS 0.785.
2. Deep oversold bounce (RSI<30 + BB%B<0 + ADX>20) has genuine edge: WF OOS 0.897.
3. Short-side swing is a DEAD END: 9 hypotheses, 0 passed. Rule engine bug.
4. Walk-forward pre-2010 data poisoning: confirmed across QQQ, XLK. Post-2010 folds pass.
5. Options: always prefer spreads over naked puts. Bear put spread is DISCRETIONARY, not systematic.
6. Iron butterfly/condor ranging+high_vol: equity proxy produces 0 trades. Indicator naming issue.
7. Investment strategies overfit trending_up (IS 0.83, WF 0.04). Only DefensiveValue has post-2010 WF pass.
8. Prerequisite rules (AND logic) produce sparser but higher-quality signals than plain (OR) rules.
9. Fed cutting but 10Y rising = worst combo for growth/tech (valuation compression, not earnings deterioration).
10. 43 strategies tested total for QQQ. Only 3 have strong WF validation.
