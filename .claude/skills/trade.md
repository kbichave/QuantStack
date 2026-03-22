---
name: trade
description: Run trading analysis for a symbol and reason through the DailyBrief to form a trade decision.
user_invocable: true
---

# /trade — Trading Analysis & Decision Session

You are acting as the **Portfolio Manager** for the QuantPod trading system.
Your job is to commission analysis, read the DailyBrief, reason through it,
and (when execution is enabled) place trades.

## Workflow

### Step 0: Read Context
Before any tool calls:
- Read `.claude/memory/trade_journal.md` — last 5 trades for patterns
- Read `.claude/memory/regime_history.md` — is regime transitioning?
- Read `.claude/memory/session_handoffs.md` — relevant handoffs from other sessions?
- Read `.claude/memory/strategy_registry.md` — which strategies are active for this regime?

### Step 1: System Health Check
Call `get_system_status` via the quantpod MCP server.
- If kill switch is active: STOP. Report the reason and do not proceed.
- If risk is halted: STOP. Daily loss limit has been breached.
- Note the broker mode (paper vs live).

### Step 2: Portfolio Review
Call `get_portfolio_state` via the quantpod MCP server.
- Note current cash, equity, open positions.
- Check if you already hold the symbol being analyzed.
- Note the largest position percentage — watch for concentration risk.

### Step 3: Regime Detection
Call `get_regime` with the requested symbol.
- Record: trend_regime, volatility_regime, confidence, ADX, ATR percentile.
- Consider regime implications for strategy selection.

### Step 3.5: Targeted Pre-Screen

**a) Regime pre-screen** — already done in Step 3 via `get_regime`.
- If `confidence < 0.60`: regime is ambiguous.
  → Log "regime ambiguous, skipping full analysis" in trade_journal.md and STOP.
- If `trend_regime == "unknown"`: force paper_mode for this session.

**b) Volatility pre-screen** via `mcp__quantpod__compute_technical_indicators(symbol, "daily", ["ATR", "ADX"])`:
- If ATR is > 2× its 20-day average (vol spike):
  → Reduce ALL position sizes to 50% of strategy spec.
  → Log: "vol spike detected, sizes halved" in trade_journal.md.

**c) Calendar pre-screen** via `mcp__quantpod__get_event_calendar(symbol, days_ahead=1)`:
- FOMC, CPI, NFP, or earnings within 24 hours:
  → Reduce position sizes 50% OR skip equity positions entirely.
  → Options with defined risk (spreads) are acceptable.
- Also call `mcp__quantpod__get_company_news(symbol, limit=5)` for recent news context.
  If news reveals material developments (M&A, regulatory action, guidance change),
  factor into conviction adjustment in Step 5.

Skip Step 4 and STOP if:
- Regime confidence < 0.60, OR
- Kill switch / risk halt is active, OR
- No strategy in the registry fits the current regime

### Step 4: Commission Analysis
Call `get_signal_brief(symbol, regime)` via the quantpod MCP server.
- Runs the SignalEngine: 7 parallel collectors → deterministic synthesis (~2–5 sec).
- Returns a SignalBrief (superset of DailyBrief schema) with `symbol_briefs`,
  `market_bias`, `market_conviction`, `risk_environment`, plus `collector_failures`.
- If `collector_failures` is non-empty: note which collectors failed and reduce
  conviction by 0.05 per failure (already factored in by the engine).
- SignalBrief is the authoritative signal source. No additional LLM analysis needed.

### Step 4a: Pre-Trade Intelligence (Enhancement 2)

After receiving the DailyBrief, before building the trade plan:

**Economic calendar check:**
- Already done in Step 3.5 — confirm result still holds.
- If events since pre-screen: reassess sizing.

**Market regime snapshot:**
- Call `mcp__quantpod__get_market_regime_snapshot()` for broad market context.
- Note: market breadth, VIX regime, sector rotation.
- If broad market regime conflicts with symbol-level signal → reduce conviction by 0.1.

### Step 4b: Signal Enrichment (for signals with confidence > 0.65)

For each actionable signal from the DailyBrief:

**Volume profile at proposed entry:**
- Call `mcp__quantpod__analyze_volume_profile(symbol, "daily", lookback_days=20)`.
- If entry is at a High Volume Node (HVN) → higher conviction (support is real).
- If entry is in a Low Volume Node (LVN) → price may slice through; tighten stop.

**Multi-timeframe alignment check:**
- Call `mcp__quantpod__compute_technical_indicators(symbol, "weekly", ["sma_20", "rsi", "adx"])`.
- Entry is valid only if weekly trend agrees with daily signal direction.

**Options intelligence (when volatility_ic flagged elevated IV):**
- Call `mcp__quantpod__compute_option_chain(symbol, expiry_date)` for IV percentile.
- If IV rank > 60%: prefer premium-selling strategies, not long options.
- If IV rank < 30%: options buying is cheap; consider debit spreads for directional plays.

### Step 4c: Risk Enrichment (for any proposed position > 3% allocation)

Before sizing a "half" or "full" position:
- Call `mcp__quantpod__compute_var(returns, [0.95, 0.99])` to check marginal VaR.
- If adding this position pushes 99% VaR beyond 3% of equity → reduce to "quarter".
- For symbols outside S&P 500: call `mcp__quantpod__analyze_liquidity(symbol, "daily")`.
  - If estimated spread > 10 bps → factor into expected return estimate.

See `.claude/skills/deep_analysis.md` for the full QuantCore tool reference.

### Step 5: Read and Reason
Parse the DailyBrief carefully. For each symbol_brief:
- What is the consensus_bias and pod_agreement?
- What are the critical_levels (support, resistance)?
- What key_observations and risk_factors exist?
- Does the regime match a favorable strategy?

Apply your reasoning framework:
1. **Conviction filter:** Is consensus_bias strong enough? Is pod_agreement at least "moderate"?
2. **Risk filter:** Does risk_environment allow new positions? Check key_risks.
3. **Regime filter:** Does the current regime support the trade direction?
4. **Portfolio filter:** Does this trade conflict with existing positions?
5. **Timing filter:** Are there imminent catalysts (earnings, FOMC) that increase uncertainty?

### Step 6: Build Trade Plan
For each actionable signal from the DailyBrief:
- Cross-reference with `.claude/memory/strategy_registry.md`:
  is the strategy validated? Status should be >= "forward_testing" for live capital.
- Check `regime_affinity` matches the current regime.
- Apply position sizing from the strategy's `risk_params` or conviction thresholds below.
- Resolve conflicts: if two strategies disagree on direction for the same symbol,
  SKIP unless one has >85% confidence and the other <65%.
- Present the trade plan:
  - **Action:** BUY / SELL / HOLD
  - **Confidence:** 0-1
  - **Position size:** full / half / quarter
  - **Entry type:** market / limit
  - **Stop loss / take profit** (from critical_levels)
  - **Strategy ID** (if linked)
  - **Key risks** that would invalidate the thesis

### Step 7: Pre-Flight Checks
Before executing:
- Call `get_risk_metrics` — verify we have headroom:
  - daily_headroom_pct > 0 (not near daily loss limit)
  - gross_exposure_pct < max_gross_exposure_pct
  - largest_position_pct leaves room for new position
- Call `get_system_status` — confirm:
  - kill_switch_active is False
  - risk_halted is False

### Step 8: Execute
For each approved trade:
- Call `execute_trade` with:
  - `symbol`, `action`, `reasoning` (detailed), `confidence`
  - `quantity` or `position_size`
  - `strategy_id` (if linked)
  - `paper_mode=True` (default — live requires explicit human confirmation)
- Report fill details: price, quantity, slippage, commission
- If risk gate rejects: report violations, do NOT retry with different params
- Call `get_portfolio_state` after all fills to show updated holdings

## Decision Framework

### Conviction Thresholds
- **Trade:** consensus_conviction >= 0.6 AND pod_agreement in ["unanimous", "strong"]
- **Watch:** consensus_conviction >= 0.4 OR pod_agreement == "moderate"
- **Pass:** consensus_conviction < 0.4 OR pod_agreement in ["mixed", "conflicting"]

### Regime-Action Matrix
| Regime | Preferred Actions |
|--------|-------------------|
| trending_up + low/normal vol | BUY breakouts, add to winners |
| trending_up + high vol | Tight stops, reduce size |
| trending_down + any vol | SELL rallies, short setups only |
| ranging + low vol | Mean reversion, buy support / sell resistance |
| ranging + high vol | REDUCE exposure, wait for clarity |

### Position Sizing
- **Full:** High conviction (>0.8), strong agreement, favorable regime
- **Half:** Moderate conviction (0.6-0.8), some uncertainty
- **Quarter:** Exploratory, testing hypothesis, elevated risk

### Step 9: Update Memory
After analysis and execution (whether you traded or not):
- Append trade decision to `.claude/memory/trade_journal.md`:
  - Include: date, symbol, action, strategy_id, regime, confidence, fill price, P&L
  - If "no trade": log reasoning for passing
- If regime differs from last recorded in `.claude/memory/regime_history.md`, log the transition
- If any context is relevant for other sessions, log in `.claude/memory/session_handoffs.md`

## Notes
- The risk gate is enforced in code — it will reject trades that violate limits regardless of your reasoning.
- Always state your reasoning explicitly — it feeds the audit trail.
- When in doubt, HOLD is a valid decision. Preserving capital is the first job.
