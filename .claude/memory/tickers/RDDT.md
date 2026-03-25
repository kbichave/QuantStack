# RDDT Research & Trading Memory

> Per-ticker memory file. Updated by research and trading loops.
> Read at START of any session involving this symbol.

## Fundamental Snapshot

_Last updated: 2026-03-25_

| Metric | Value | Notes |
|--------|-------|-------|
| Revenue (TTM) | $2.2B | Q4'25: $726M (+70% YoY) |
| Revenue Growth YoY | +70% | Accelerating |
| Gross Margin | 91.9% | Best-in-class ad platform |
| Operating Margin | 31.9% | Was 12.4% Q4'24 — massive operating leverage |
| FCF (TTM) | ~$684M | CapEx negligible ($3-5M/qtr) |
| FCF Yield | 2.47% | Low — growth premium |
| P/E | N/A | |
| P/S | 12.5x TTM | |
| Net Cash/Debt | +$2.17B | $2.48B liquid - $310M debt |
| Shares Outstanding | 202.9M | |
| SBC (% of revenue) | ~15.6% | ~$343M TTM, ~1.2% annual dilution |

## Price Action

_Last updated: 2026-03-24_

| Level | Value |
|-------|-------|
| Current Price | $136.12 |
| SMA 20 | $141.37 |
| SMA 50 | $164.80 |
| SMA 200 | $191.78 |
| Key Support | $127.70 (Feb 12 low) |
| Key Resistance | $164.80 (SMA 50) |
| 52w High | $263 (Jan 2026) |
| 52w Low | $127.70 (Feb 2026) |

**Price is DOWN 48% from $263 Jan 2026 high. Below all major MAs.**

## Investment Thesis

| Thesis Type | Fit | Evidence |
|-------------|-----|----------|
| Value | WEAK | FCF yield 2.5%, SBC-adjusted 1.2%. P/S 12.5x. Not a value play. |
| Quality-Growth | STRONGEST | 70%+ rev growth + 90%+ GM + 3x OpInc improvement YoY |
| Dividend | N/A | No dividend |
| Sector Rotation | POSSIBLE | Benefits if rates fall (growth re-rating) |
| Earnings Catalyst | STRONG | Every quarter since IPO beaten. Q1'26 est. May 2026 |

**Primary thesis**: Quality-Growth — exceptional revenue growth with expanding margins at a 48% drawdown from highs
**Conviction**: 72% (reduced: no technical trend reversal confirmation yet)
**Catalyst timeline**: Q1 2026 earnings (est. May 2026)

## Evidence Map (from breadth-first scan)

_Needs re-scan with full pipeline — previous research used only tier-1 technicals._

| Category | Key Findings | Tier | Direction |
|----------|-------------|------|-----------|
| Regime | ADX low (10.77), downtrend exhausted | tier_4 | neutral (no trend) |
| Technicals | RSI 38, MACD hist +1.85 (turning), below all MAs | tier_1 | mixed |
| Fundamentals | Revenue +70%, GM 91.9%, OpMargin 31.9%, FCF $684M | tier_4 | bullish |
| Insider/Institutional | Not yet scanned | tier_3 | unknown |
| Flow/Sentiment | Not yet scanned | tier_2 | unknown |
| Macro | Fed rate path — relevant for growth stock re-rating | tier_4 | unknown |
| Volume/Liquidity | Not yet scanned | tier_2 | unknown |
| Cross-domain | Not yet scanned | context | unknown |

**Convergence**: Incomplete — need full scan. Fundamentals strongly bullish, technicals mixed.

## Strategies

| ID | Name | Type | Backtest | WF OOS | ML AUC | Status |
|----|------|------|---------|--------|--------|--------|
| strat_c6af8e64b7ea | RSI Oversold (<38) | tier_1 only | 18 trades, 50% WR, Sharpe -0.08 | -0.07 | N/A | forward_testing |
| strat_848218beaee1 | RSI Washout (<30) | tier_1 only | 10 trades, 30% WR, Sharpe -0.65 | not run | N/A | backtested |
| strat_0a606eb73eec | MACD Histogram crossover | tier_1 only | 0 trades (bug) | not run | N/A | draft |
| strat_cee4668f10f9 | Multi-signal (MACD+Aroon+RSI) | tier_1+tier_2 | 66.67% WR | OOS Sharpe 0.93 | N/A | BEST result |

**NOTE**: All strategies so far are tier_1/tier_2 only. Need composite strategies with tier_3+ signals.

### Swing Strategies (added 2026-03-25, iteration 2-3)

| ID | Name | Type | Backtest | WF OOS Sharpe | Status | Notes |
|----|------|------|---------|---------------|--------|-------|
| strat_154ea6c7be43 | Aroon Breakout Regime-Gated [swing v2] | breakout/momentum | 24T, 62.5% WR, Sharpe 0.794, MaxDD 0.67% | **2.01** (2 folds, both pos) | forward_testing* | *promote blocked by DB tx abort |
| mean_rev_v1 | RSI Mean-Rev Bounce [swing v1] | mean_reversion | not yet registered (DB blocked) | — | draft | RSI<42+MACD hist>0+Aroon>-50 |

**DB Write Block (2026-03-24)**: `compute_and_store_features` caused `INSERT OR REPLACE` PostgreSQL syntax error → all writes aborted. Need MCP server restart to recover. Strategy `strat_154ea6c7be43` validated but status stuck at `backtested` (should be `forward_testing`/promoted).

## Alerts

| Date | Action | Entry | Stop | Target | Confidence | Status |
|------|--------|-------|------|--------|------------|--------|
| 2026-03-25 | BUY | $135 limit | $115 | $200 | 72% | pending |

## ML Models

| Date | Model | Features | AUC (OOS) | IS-OOS Gap | Verdict |
|------|-------|----------|-----------|------------|---------|
| — | Not trained yet | — | — | — | Pipeline verified working (2026-03-24) |

**ML pipeline status**: WORKING. 503 bars → 89 TI features → 484 labeled samples → LightGBM trains successfully.

## Research Log

### Iteration 3 (2026-03-25) — Swing Research

**What was tried:**
- Aroon Breakout [swing v1] — 14 trades, Sharpe 0.457, OK but low conviction in tariff period
- Aroon Breakout Regime-Gated [swing v2] — 24 trades, 62.5% WR, Sharpe 0.794, MaxDD 0.67%
- Walk-forward on v2 — OOS Sharpe 2.01, both folds positive, overfit ratio 0.77. BEST result.
- Mean-reversion strategy designed (RSI<42 + MACD hist>0 + Aroon>-50) — cannot register (DB blocked)
- `compute_and_store_features` crashed with INSERT OR REPLACE → blocked all DB writes

**Current setup (2026-03-24)**:
- Price: $136.12, in HVN support zone $127.70-$154.86
- RSI: 38.11 (oversold), MACD hist: +1.85 (positive), ADX: 10.17 (ranging), Stoch K: 21.41
- Regime: ranging + low vol, ATR at 5th percentile ($8.08)
- 25.63% below 90-day VWAP ($183.03) — deeply oversold relative to recent history
- Mean-reversion signals ACTIVE right now, but no strategy registered yet

**Blocking issues:**
- DB transaction abort — all strategy writes/updates blocked
- ML training still blocked (0-bars, PgDataStore fix not deployed in running server)
- Walk_forward_sparse FunctionTool bug persists

**What to do next iteration:**
1. Restart MCP server (or wait for connection recovery) → register mean-reversion strategy → backtest → WF
2. Train ML model (try `compute_and_store_features` with tiers=technical only first)
3. Build event-driven earnings drift strategy (Q1'26 earnings est. May 2026)
4. Build RDDT/XLC stat-arb pairs trade
5. Attempt stacking ensemble once ML pipeline fixed
6. Complete trading_sheets_monday.md ← **REQUIRED for TRADING_READY gate**

### Iteration 1 (2026-03-25)

**What was tried:**
- RSI oversold strategies (tier_1 only) — poor results
- MACD histogram crossover — 0 trades due to indicator mapping bug
- Multi-signal MACD+Aroon+RSI (tier_1+tier_2) — BEST result: 66.67% WR, OOS Sharpe 0.93
- Fundamental analysis — strong quality-growth case established
- Elliott Wave read — A-wave $263→$127, B-wave targets $179-$211

**What worked:**
- Multi-signal OR-logic (MACD+Aroon+RSI) produced 67% WR vs 50% RSI-only
- OOS Sharpe 0.93 (anti-overfitting — OOS > IS)

**What failed:**
- Single RSI strategies — always-on signal, no edge
- RSI crossover exits — fire same-day, creating 1-day holds
- ML training — DataStore bug (FIXED: switched to PgDataStore)

**What to do next iteration:**
1. Full breadth-first scan (ALL 8 categories) — insider/institutional/flow/macro missing
2. Register composite strategy: fundamentals (revenue growth) + technical (MACD/Aroon) + institutional signals
3. Use time-based exits (hold_days=60) not RSI crossover
4. Train ML model with technical+fundamentals+macro features
5. Walk-forward with small params (min_train=167, test=83, splits=3)

## Lessons (RDDT-specific)

1. **Short history**: IPO March 2024, only ~503 bars. All strategies should be FORWARD_TESTING until 2029.
2. **Exit rule gotcha**: RSI crossover exits fire same-day → 1-day holds. Use time-based exits (hold_days=45-60) for investment strategies.
3. **Tier-2 signals matter**: Adding Aroon+MACD to RSI improved win rate from 50% to 67%.
4. **Elliott Wave**: A-wave $263→$127 (-52%). B-wave targets: $179 (38.2%), $195 (50%), $211 (61.8%). $200 target aligns with 50% Fib.
5. **Not a value play**: FCF yield 2.5%, SBC-adjusted 1.2%. The thesis is quality-growth, not value.
6. **MCP indicator names**: Use `rsi` not `rsi_14`, `macd_histogram` for MACD hist. `sma_50`, `sma_200`, `bb_lower` confirmed working.
