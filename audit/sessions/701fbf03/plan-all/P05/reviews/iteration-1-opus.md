# Opus Review — P05

**Model:** claude-opus-4-6
**Generated:** 2026-04-08T03:15:00Z

## Key Issues Identified

1. **A/B test design flaw**: Hash-based assignment with ~50 symbols gives N<17 per method — t-test unreliable. Recommended: compute all three methods offline for every symbol, compare against realized returns.
2. **Silent catch-and-pass violations**: Existing synthesis.py has bare `except: pass` blocks that violate project rules.
3. **Module-level cache invalidation undefined**: Conviction param cache has no refresh mechanism specified.
4. **Survivorship bias in calibration**: Only using closed_trades biases the regression toward traded signals.
5. **Sanity bounds on precomputed weights**: No protection against garbage weights (e.g., single collector at 100%).
6. **Staleness window gap**: 7-day staleness with weekly compute = zero tolerance for missed runs.
7. **No rollback for batch jobs**: Upsert overwrites previous values, no way to detect/revert bad data.
8. **Single-row config table smell**: ensemble_config duplicates feature flag pattern.
9. **Deferred imports**: synthesis.py lines 532-534, 548-551 use deferred imports — must be fixed.
10. **Forward return backfill unspecified**: How/when 5-day forward returns are filled in ensemble_ab_results.
11. **Position sizing stacking risk**: Multiple 0.5x scalars can combine to microscopic positions.
