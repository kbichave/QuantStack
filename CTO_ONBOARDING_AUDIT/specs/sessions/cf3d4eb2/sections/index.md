<!-- PROJECT_CONFIG
runtime: python-uv
test_command: uv run pytest
END_PROJECT_CONFIG -->

<!-- SECTION_MANIFEST
section-01-schema-foundation
section-02-fill-legs
section-03-business-calendar
section-04-sec-compliance
section-05-audit-trail
section-06-tca-ewma
section-07-algo-scheduler-core
section-08-twap-vwap
section-09-paper-broker-enhance
section-10-liquidity-model
section-11-slippage-enhance
section-12-options-monitoring
section-13-funding-costs
section-14-integration-tests
END_MANIFEST -->

# Implementation Sections Index

## Dependency Graph

| Section | Depends On | Blocks | Parallelizable |
|---------|------------|--------|----------------|
| section-01-schema-foundation | - | all | Yes |
| section-02-fill-legs | 01 | 06, 07, 08, 05 | No |
| section-03-business-calendar | 01 | 04 | Yes (with 02) |
| section-04-sec-compliance | 02, 03 | 14 | No |
| section-05-audit-trail | 02 | 14 | Yes (with 04) |
| section-06-tca-ewma | 02 | 08, 11 | No |
| section-07-algo-scheduler-core | 02 | 08 | Yes (with 06) |
| section-08-twap-vwap | 06, 07 | 09 | No |
| section-09-paper-broker-enhance | 08 | 14 | No |
| section-10-liquidity-model | 01 | 14 | Yes (with 08, 09) |
| section-11-slippage-enhance | 06 | 14 | Yes (with 08, 09, 10) |
| section-12-options-monitoring | 01 | 14 | Yes (with 08-11) |
| section-13-funding-costs | 01 | 14 | Yes (with 08-12) |
| section-14-integration-tests | 04, 05, 08, 09, 10, 11, 12, 13 | - | No |

## Execution Order (Batches)

1. **Batch 1:** section-01-schema-foundation (no dependencies)
2. **Batch 2:** section-02-fill-legs, section-03-business-calendar (parallel after 01)
3. **Batch 3:** section-04-sec-compliance, section-05-audit-trail (parallel after 02+03)
4. **Batch 4:** section-06-tca-ewma, section-07-algo-scheduler-core (parallel after 02)
5. **Batch 5:** section-08-twap-vwap (after 06+07)
6. **Batch 6:** section-09-paper-broker-enhance, section-10-liquidity-model, section-11-slippage-enhance, section-12-options-monitoring, section-13-funding-costs (parallel after respective deps)
7. **Batch 7:** section-14-integration-tests (after all)

## Section Summaries

### section-01-schema-foundation
Database migration infrastructure: all new tables (fill_legs, tca_parameters, day_trades, pending_wash_losses, wash_sale_flags, tax_lots, algo_parent_orders, algo_child_orders, algo_performance, execution_audit, slippage_accuracy). Add columns to positions table (margin_used, cumulative_funding_cost).

### section-02-fill-legs
Fill legs recording in paper_broker and alpaca_broker. VWAP computation helper. Dual-write to fill_legs + fills summary. Tests for fill recording and VWAP calculation.

### section-03-business-calendar
Exchange calendar utility wrapping `exchange_calendars`. Business day arithmetic for PDT (5 business days), TWAP scheduling (market hours), wash sale (30 calendar days). Tests for weekends, holidays, cross-month.

### section-04-sec-compliance
`execution/compliance/` package with pretrade.py (PDTChecker, MarginCalculator) and posttrade.py (WashSaleTracker with two-phase detection, TaxLotManager with FIFO). Risk gate integration for PDT hard block and margin check. Fill hook integration for wash sale and tax lots.

### section-05-audit-trail
`execution_audit` table population on every fill (IMMEDIATE and future child fills). NBBO capture from Alpaca IEX. Price improvement calculation. Algo rationale logging. Audit queries.

### section-06-tca-ewma
EWMA update after every fill. Per-symbol, per-time-bucket parameter storage. Conservative 2x multiplier decaying to 1.0 at 50 fills. Integration with pre_trade_forecast(). Position sizing feedback loop.

### section-07-algo-scheduler-core
New `algo_scheduler.py` (EMS). AlgoParentOrder and ChildOrder types. Parent/child state machines. Async execution loop with run_in_executor for broker calls. Cancellation triggers (kill switch, risk halt, exit signal). Crash recovery via startup_recovery(). POV → VWAP fallback.

### section-08-twap-vwap
TWAP scheduling: equal time slices with jitter and size variation. VWAP scheduling: volume-weighted from historical intraday bars. Volume profile builder with daily caching and synthetic fallback. Child order submission, failure handling, redistribution.

### section-09-paper-broker-enhance
Paper broker fills TWAP/VWAP children against historical bars. Participation cap per bar. Fill price from bar VWAP + directional noise. Partial fill when child exceeds participation. IMMEDIATE orders unchanged.

### section-10-liquidity-model
LiquidityModel class with spread estimation, depth estimation, time-of-day adjustment. Pre-trade check in risk gate (PASS/SCALE_DOWN/REJECT). Stressed exit scenario in continuous risk monitor.

### section-11-slippage-enhance
Paper broker uses EWMA-calibrated spread and impact from tca_parameters. Time-of-day slippage profiles. Slippage accuracy tracking (predicted vs realized). Alert on drift.

### section-12-options-monitoring
Options-specific exit rules in execution_monitor.py. Configurable per-rule action (auto_exit/flag_only). Theta acceleration, pin risk, assignment risk, IV crush, max theta loss. Greeks integration from options engine.

### section-13-funding-costs
Margin interest calculation for leveraged positions. Daily interest accrual. cumulative_funding_cost on positions. Strategy performance metrics integration.

### section-14-integration-tests
Cross-section integration tests: TWAP → fill_legs → TCA → audit trail pipeline. PDT counting with TWAP round-trips. Wash sale → tax lot cost basis adjustment. Algo scheduler crash recovery. Options monitoring exit triggers.
