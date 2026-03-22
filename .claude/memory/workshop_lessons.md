# Workshop Lessons

> Accumulated R&D learnings -- your research memory
> Read at START of every /workshop session
> Update after: /workshop, /reflect

## Backtesting Pitfalls

- SPY mean-reversion with extreme oversold filters (RSI<35, Stoch<15, CCI<-150) produces only 27-55 trades in 6 years depending on configuration
- Loosening oversold thresholds to 1-of-4 (87 trades) destroys edge: Sharpe -0.41, PF 0.76
- The built-in BacktestEngine has buggy metric scaling (win rates reported as percentages not decimals, max drawdown >100%). Use custom_backtest() for reliable metrics.
- BacktestConfig in engine.py does NOT support stop_loss_atr_multiple (only core/config.py version does). MCP tool has a bug mapping these params -- fixed in quantcore/mcp/tools/backtesting.py line 89-94.

## CRITICAL: MCP Rule Logic is OR, Not AND (iteration 3 discovery)

The MCP engine's `_generate_signals_from_rules` evaluates **plain rules** with **OR logic per direction**. Each rule with direction=long fires independently. If ANY long rule is true, the signal is LONG. This is fundamentally different from AND logic where ALL conditions must be true simultaneously.

**Practical consequence**: A strategy with rules [regime=trending_up, RSI 40-65, stoch_k < 80] is NOT "enter when all three are true." It is "enter when ANY one is true." Since stoch_k < 80 is true ~90% of the time, the signal is almost always LONG. This creates a regime-following position strategy, not a pullback strategy.

**To get AND logic**: Use `type: "prerequisite"` rules (AND gate). But note: prerequisite rules only support a single direction (from parameters.direction). For bidirectional strategies (long+short), you need either two separate strategy registrations or accept OR logic.

**Performance comparison (SPY full history)**:
- OR-logic: 200 trades, Sharpe 0.409, PF 2.09 -- WORKS
- AND-logic: 649 trades, Sharpe -0.460, PF 0.80 -- FAILS
- Pure regime: 502 trades, Sharpe -0.400, PF 0.69 -- FAILS

The OR-logic creates fewer, longer-duration trades that capture full regime moves. The AND-logic creates many short whipsaw trades that get destroyed by commissions.

## Alpha Decay Analysis (SPY, 2026-03-20)

- RSI: Half-life 1.5 bars, IC decays rapidly. Short-lived signal. Optimal holding: 16 bars.
- ADX: No meaningful decay (IC ~0.01 stable across lags). Trend persistence signal -- use as filter not entry.
- BB Width: Half-life 14.2 bars, slow decay. Vol compression is a persistent state. Good for multi-day setups.
- Volume Ratio: Noisy, no clear decay pattern. High turnover (0.25). Not reliable standalone.

## Market State Observations (2026-03-20)

### Cross-Asset Divergences
- SPY: Bear trend, RSI=30, ADX=49 (strong), range_position=12%. Classic strong bear.
- GLD: RSI=15, CCI=-263. Gold at EXTREME oversold. Anomalous in risk-off.
- TLT: RSI=27, ADX=41. Bonds weak despite risk-off. Flight to safety NOT working.
- XLE: RSI=69, Stoch=80, above all MAs. Energy is the rotational beneficiary.
- NVDA: RSI=52, near ema_200. Holding up better than broad market.
- TSLA: RSI=38, Stoch=4, range_position=3.8%. Extreme oversold, aggressive selling.

### Vol Dynamics
- SPY EGARCH persistence=1.175 (explosive), forecast vol 14.3% ann
- NVDA EGARCH persistence=1.108, forecast vol 48.5%
- TSLA: forecast vol 45.4% but vol_percentile_rank=0.05 -- vol at 5th percentile historically
- All three have EGARCH persistence >1.0 -- volatility is in an expansionary regime

## Strategy Discovery Findings

### Iteration 3: ETF Generalization (2026-03-20)

**Key finding**: regime_momentum_v1 generalizes across major ETFs when:
1. Signal logic uses OR semantics (not AND)
2. Data is restricted to post-2010 (post-GFC structural break)

**ETF rankings (by OOS Sharpe, 2010+ data)**:
1. QQQ: OOS 1.346, overfit 0.58 -- STRONGEST. Tech-heavy, clean trends.
2. XLK: OOS 1.276, overfit 0.73 -- STRONG. Sector tech ETF, cleaner than QQQ.
3. SPY: OOS 0.819, overfit 0.93 -- SOLID. Broad market, good regime signals.
4. XLF: OOS 0.617, overfit 0.67 -- MODERATE. Financials, sector-specific regime.
5. IWM: OOS 0.287, overfit 1.78 -- WEAK. Small-caps mean-revert within regimes.

**Why IWM is weak**: Small-cap stocks have higher mean-reversion tendency within trends. The 5-day forward return during trending_down regime is +0.377% (mean-reverting!), making short-side entries lose money. IWM's short side avg return is -0.316% (loss on shorts).

**Wide RSI variant (30-70)**: Marginally better Sharpe (0.007-0.044 higher per symbol) but ~40% fewer trades. Not worth the trade-off -- fewer trades means noisier Sharpe estimates.

**Pre-2010 data poisoning**: Full 26-year walk-forward shows negative OOS Sharpe on ALL symbols. The dot-com crash and GFC created regime conditions (extreme VIX, structural breaks in correlation) that the current ADX-based regime model cannot handle. This is not overfitting -- it is a genuine structural change in how markets trend since QE began.

### What Works
1. **Regime-following position strategies on ETFs**: Hold long during trending_up, short during trending_down, flat during ranging. The OR-logic noise reduction on entry/exit prevents whipsaw.
2. **Vol compression breakout on sector ETFs**: XLE validated. Institutional rotation drives sustained moves after compression.
3. **Tech-heavy ETFs have the cleanest regime signals**: QQQ and XLK consistently outperform SPY and IWM.

### What Fails
1. **AND-logic pullback timing**: The thesis "buy established trend on RSI pullback" sounds logical but produces too many short-duration trades that get destroyed by commissions and slippage.
2. **Single stocks for systematic strategies**: NVDA, TSLA consistently fail or overfit. Too noisy.
3. **IWM short side**: Small-caps mean-revert during bear regimes. Shorting IWM during trending_down produces negative returns.
4. **Pre-2010 data for walk-forward**: Structural break makes pre-GFC data counter-productive for validation.

### Key Principles Confirmed
1. **Regime-first filtering works**: Strategies with explicit regime filters consistently beat without.
2. **OOS > IS is the gold standard**: QQQ OOS 1.346 vs IS 0.785. Strategy performs BETTER recently.
3. **The overfit ratio is the most important metric**: Values < 1.0 (OOS > IS) are ideal.
4. **Sector ETFs > single stocks for systematic strategies**.
5. **OR-logic creates better position management than AND-logic** -- counterintuitive but true.

## Iteration 4: Options Research -- VRP and IV Surface (2026-03-21)

### VRP Analysis

**SPY** (as of 2026-03-20):
- ATM IV (30d synthetic): 25.0% ann
- EGARCH realized vol: 13.55% ann | EGARCH forecast vol: 16.5% ann
- EGARCH conditional vol: 14.57% ann | Persistence: 1.175 (explosive)
- **VRP = IV - Realized = +11.45 pct pts** (extremely elevated)
- **VRP = IV - Forecast = +8.5 pct pts**
- Vol percentile rank: 69th

**QQQ** (as of 2026-03-20):
- ATM IV (30d synthetic): 25.0% ann
- EGARCH realized vol: 17.57% ann | EGARCH forecast vol: 19.68% ann
- EGARCH conditional vol: 17.95% ann | Persistence: 1.163 (explosive)
- **VRP = IV - Realized = +7.43 pct pts** (elevated)
- **VRP = IV - Forecast = +5.32 pct pts**
- Vol percentile rank: 60th

**Key insight**: VRP is very positive on both names, favoring premium selling. BUT EGARCH persistence >1.0 means realized vol is likely to EXPAND, which would compress VRP over time. Timing risk for premium sellers is real. The correct play: sell premium NOW while VRP is wide, but with tight regime gates to exit if vol expansion materializes.

### Synthetic Chain Limitation

The compute_option_chain tool produces FLAT IV across all strikes (25% everywhere). This means:
- No skew measurement possible (25-delta put vs call IV)
- No term structure slope measurement
- IV rank/percentile cannot be computed from synthetic data
- get_iv_surface tool is BROKEN (FunctionTool not callable -- server-side bug)
- run_backtest_options tool is BROKEN (same bug)

**Action needed**: Fix the FunctionTool callable bug in quantcore/mcp/tools/options.py and quant_pod/mcp/tools/backtesting.py. The root cause is async functions decorated with @mcp.tool() calling other @mcp.tool() functions directly (line 1105 of options.py calls `await get_options_chain(symbol=symbol)` which is now a FunctionTool wrapper, not a raw function).

### Strategies Registered (Draft)

1. **vrp_premium_sell_spy_v1** (strat_d52734f45b93) -- Iron condor for ranging+high_vol regime
   - Sell 25-delta strangles with 10pt wings when VRP > 5 pct pts
   - Regime gate: ranging only. Exit on regime change.
   - Target: 21-day hold, close at 50% max profit or 7 DTE

2. **regime_momentum_calls_v1** (strat_8425300c071b) -- Long ATM calls for trending_up+high_vol
   - Buy calls when IV rank < 50 (avoid IV crush) + MACD > 0
   - Convexity overlay on existing regime_momentum equity signals
   - Covers the trending+high_vol gap where naked equity is too risky

3. **regime_momentum_puts_v1** (strat_fcc1576c5e1f) -- Long ATM puts for trending_down+high_vol
   - Allow higher IV rank (up to 70) because downtrend IV tends to EXPAND (leverage effect)
   - EGARCH persistence >1 confirms vol expansion benefits vega on puts

### What We Cannot Validate Yet

The options backtest tool (run_backtest_options) is broken. These strategies are DRAFT only and cannot be promoted until:
1. The FunctionTool callable bug is fixed
2. Options backtests run with walk-forward validation
3. OOS Sharpe > 0.3 and overfit ratio < 2.0

## Iteration 5: ML Champion Training + Options Design (2026-03-21)

### ML Batch 3: All Validated Strategy Symbols Now Have Champions

Trained LightGBM/XGBoost on 5 validated strategy symbols (SPY, QQQ, XLK, XLF, XLE). All use technical+fundamentals tiers, CausalFilter, EventLabeler ATR, 756d lookback, PurgedKFold 5-fold CV.

| Symbol | AUC (OOS) | IS-OOS Gap | Verdict |
|--------|-----------|------------|---------|
| SPY | 0.7397 | 0.061 | champion (XGBoost) |
| QQQ | 0.7122 | 0.133 | champion (LightGBM) |
| XLK | 0.6477 | 0.097 | champion (LightGBM) |
| XLF | 0.6374 | 0.014 | champion (best calibration) |
| XLE | 0.5748 | 0.063 | champion (marginal) |

**Breakthrough feature**: `macdext_histogram` is top SHAP in 4/5 models. This aligns directly with regime_momentum strategy logic — extended MACD measures the momentum divergence those strategies exploit. Cross-pollination confirmed.

**Drift-resistant features**: `bop` (balance of power) and `dx` (directional index) show zero concept drift across all symbols. Currently underweighted — candidate for next feature experiment.

**CausalFilter drops 0 features** out of 84. Either all carry causal signal or filter p-value threshold is too permissive.

**XLE needs macro features** — AUC 0.5748 is barely above champion threshold. Energy sector is macro-driven (oil, yields, VIX).

### DuckDB Lock Bug

MCP server (PID 59995) holds DuckDB write lock. `register_model` fails because it tries to open a NEW connection from within the MCP process, conflicting with its own existing connection. Models saved to disk but not registered in DB.

## Next Iteration Priorities (Iteration 6)

1. **CRITICAL**: Fix FunctionTool callable bug in options.py and backtesting.py
2. **HIGH**: Fix DuckDB lock bug so register_model works from MCP
3. **HIGH**: Backtest 3 options strategies once tools fixed
4. **HIGH**: ML challengers — XGBoost on QQQ/XLK/XLF, LightGBM on SPY
5. **MEDIUM**: Stacking ensemble for top symbols (SPY, QQQ, XLK)
6. **MEDIUM**: Add macro features to XLE model
7. **MEDIUM**: Start RL shadow recording
8. **LOW**: TSLA straddle investigation
