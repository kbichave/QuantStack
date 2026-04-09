<!-- PROJECT_CONFIG
runtime: python-uv
test_command: uv run pytest
END_PROJECT_CONFIG -->

<!-- SECTION_MANIFEST
section-01-schema-migrations
section-02-wire-pricing-tools  depends_on:section-01-schema-migrations
section-03-wire-analysis-tools  depends_on:section-01-schema-migrations
section-04-complex-structures  depends_on:section-01-schema-migrations
section-05-portfolio-greeks  depends_on:section-01-schema-migrations
section-06-hedging-engine  depends_on:section-05-portfolio-greeks
section-07-pin-risk-expiration  depends_on:section-01-schema-migrations
section-08-pnl-attribution  depends_on:section-05-portfolio-greeks
section-09-unit-tests  depends_on:section-02-wire-pricing-tools,section-03-wire-analysis-tools,section-04-complex-structures,section-05-portfolio-greeks,section-06-hedging-engine,section-07-pin-risk-expiration,section-08-pnl-attribution
END_MANIFEST -->

# P06 Implementation Sections Index

## Dependency Graph

| Section | Depends On | Blocks | Parallelizable |
|---------|------------|--------|----------------|
| section-01-schema-migrations | - | 02-08 | Yes |
| section-02-wire-pricing-tools | 01 | 09 | Yes (with 03-07) |
| section-03-wire-analysis-tools | 01 | 09 | Yes (with 02,04-07) |
| section-04-complex-structures | 01 | 09 | Yes (with 02,03,05-07) |
| section-05-portfolio-greeks | 01 | 06, 08, 09 | Yes (with 02-04,07) |
| section-06-hedging-engine | 05 | 09 | Yes (with 08) |
| section-07-pin-risk-expiration | 01 | 09 | Yes (with 02-06) |
| section-08-pnl-attribution | 05 | 09 | Yes (with 06) |
| section-09-unit-tests | 02-08 | - | No |

## Execution Order

1. section-01-schema-migrations (no dependencies)
2. section-02 through section-05, section-07 (parallel after 01)
3. section-06-hedging-engine, section-08-pnl-attribution (after 05)
4. section-09-unit-tests (final)

## Section Summaries

### section-01-schema-migrations
Add portfolio_greeks_history, options_pnl_attribution tables. Add structure_type column to positions.

### section-02-wire-pricing-tools
Wire price_option, compute_implied_vol tools to existing core/options/engine.py and pricing.py functions.

### section-03-wire-analysis-tools
Wire get_iv_surface, analyze_option_structure, score_trade_structure, simulate_trade_outcome tools. New logic for scoring and simulation.

### section-04-complex-structures
StructureType enum, StructureBuilder class for iron condors, butterflies, calendars, straddles, strangles. Payoff profiles.

### section-05-portfolio-greeks
PortfolioGreeks model, compute_portfolio_greeks() aggregation, Greeks history snapshots.

### section-06-hedging-engine
HedgingStrategy ABC, DeltaHedgingStrategy, HedgingEngine orchestrator. Threshold-based delta hedging.

### section-07-pin-risk-expiration
Pin risk detection in execution_monitor. Alert + auto-close for DTE<3 near strike. Feature flag.

### section-08-pnl-attribution
Daily Greek P&L decomposition (delta/gamma/theta/vega/unexplained). options_pnl_attribution table.

### section-09-unit-tests
Full test suite for all new modules: tools, hedging, pin risk, structures, Greeks, P&L attribution.
