# P06 Self-Interview: Options Desk Upgrade

## Q1: Tool stubs — wiring vs implementing
**Q:** The 6 stubbed tools need implementation. How much new logic is needed vs wiring to existing core functions?
**A:** Mostly wiring. `price_option` → `price_option_dispatch()` (already exists in engine.py). `compute_implied_vol` → `implied_volatility()` (exists in pricing.py) or `implied_vol_vollib()`. `get_iv_surface` → `IVSurface` class (exists in iv_surface.py). `analyze_option_structure` → combine IVSurface metrics + skew analysis. `score_trade_structure` → new logic needed for multi-leg P&L analysis. `simulate_trade_outcome` → new logic for scenario stress testing.

## Q2: Which backend for IV surface fitting?
**Q:** IVSurface already exists with bilinear interpolation. Does the spec require SABR or SVI fitting?
**A:** No. The existing IVSurface with bilinear interpolation on log-moneyness × sqrt(T) is sufficient for now. SABR/SVI fitting is a nice-to-have but not blocking. The pysabr_adapter exists but doesn't need to be wired into the surface for P06. Keep it simple.

## Q3: Auto-hedging — how aggressive?
**Q:** The spec asks for delta hedging, gamma scalping, theta harvesting. How should this integrate with existing execution?
**A:** Delta hedging is the priority. Build a `HedgingEngine` that computes hedge orders (buy/sell underlying shares to neutralize portfolio delta). Trigger: threshold-based (when |portfolio delta| > configurable $X). Execution: through existing trade_service.py. Gamma scalping and theta harvesting are P08 territory — don't implement here, just ensure the hedging engine interface supports them.

## Q4: Pin risk — what action?
**Q:** When pin risk is detected (DTE < 3, within 2% of strike), what should happen?
**A:** Alert via EventBus + auto-close if risk exceeds threshold. Short options with DTE < 3 and spot within 2% of strike have assignment risk. The system should: (1) flag in execution_monitor, (2) send system alert, (3) auto-close the position unless overridden by flag. Never auto-roll — rolling requires selecting new strikes/expiries which is a strategy decision.

## Q5: Complex structures — how to represent?
**Q:** Iron condors, butterflies, etc. need structured representation. What's the data model?
**A:** Extend `OptionsPosition` in models.py with a `structure_type` enum (VERTICAL_SPREAD, IRON_CONDOR, BUTTERFLY, CALENDAR, DIAGONAL, STRADDLE, STRANGLE, RATIO_SPREAD). Each structure is a list of `OptionLeg` objects with named roles. The `contract_selector.py` already has some of this. Add a `StructureBuilder` that constructs multi-leg positions from strategy intent.

## Q6: Greeks aggregation scope?
**Q:** Should Greeks aggregation be per-position, per-strategy, or portfolio-wide?
**A:** All three. Per-position is trivial (already in the option model). Per-strategy: sum Greeks across positions for a given strategy. Portfolio-wide: sum across all options positions. Store snapshots in a `portfolio_greeks_history` table for time-series analysis. P&L attribution by Greek is a separate calculation that runs daily.

## Q7: QuantLib dependency?
**Q:** The spec mentions QuantLib-Python for surface fitting. Is it needed?
**A:** No. The existing vollib + financepy + scipy stack covers everything P06 needs. QuantLib is heavy (large C++ build) and adds deployment complexity for marginal benefit. Defer to P08 if needed.

## Q8: What enables P08?
**Q:** P06 enables P08 (Options Market-Making). What's the interface contract?
**A:** P08 needs: (1) functioning IV surface for mispricing detection, (2) Greeks aggregation for portfolio-level risk, (3) hedging engine for delta neutralization, (4) complex structure support for multi-leg strategies. All four are P06 deliverables. The hedging engine interface should support pluggable strategies so P08 can add gamma scalping and theta harvesting without P06 rework.
