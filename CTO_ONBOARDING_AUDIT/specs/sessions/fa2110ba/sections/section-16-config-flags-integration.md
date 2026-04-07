# Section 16: Config Flags & Integration Testing

## Problem

Phase 7 introduces six independent feedback loops that modify sizing, signal weights, and conviction. Each loop has the potential to reduce positions to near-zero or interact unexpectedly with other loops. Deploying all six simultaneously with no kill-switch would make it impossible to isolate regressions. Each feedback loop needs an independent environment variable that defaults to OFF, allowing safe one-at-a-time rollout. Additionally, every feedback mechanism must behave correctly during cold-start (insufficient data) and the combined multiplicative effect of all loops must be bounded.

## Dependencies

This section is the final integration layer. It depends on:

- **section-07-ic-weight-adjustment** -- defines the `FEEDBACK_IC_WEIGHT_ADJUSTMENT` flag and the IC-based weight adjustment logic in `signal_engine/synthesis.py`.
- **section-08-signal-correlation** -- defines the `FEEDBACK_CORRELATION_PENALTY` flag and the correlation penalty logic in the signal correlation supervisor node.
- **section-10-conviction-multiplicative** -- defines the `FEEDBACK_CONVICTION_MULTIPLICATIVE` flag and the multiplicative conviction calibration in `signal_engine/synthesis.py`.
- **section-12-sharpe-demotion** -- defines the `FEEDBACK_SHARPE_DEMOTION` flag and the live vs. backtest Sharpe demotion gate in `src/quantstack/learning/sharpe_demotion.py`.
- **section-13-drift-detection** -- defines the `FEEDBACK_DRIFT_DETECTION` flag and the concept drift detection layers in `src/quantstack/learning/drift_detector.py`.
- **section-15-regime-transitions** -- defines the `FEEDBACK_TRANSITION_SIZING` flag and the regime transition sizing adjustment in `signal_engine/collectors/regime.py`.

Each of those sections is responsible for implementing the flag check within its own code path. This section defines the centralized flag registry, the integration tests that verify each flag works in isolation, and the compound sizing floor that prevents micro-orders when multiple loops stack.

## Scope

1. A centralized config module that reads and exposes all six feedback flags as typed booleans with `false` defaults.
2. Integration tests verifying each flag independently disables its corresponding adjustment (factor returns to 1.0 when flag is off).
3. Integration test verifying cold-start behavior for each feedback loop (insufficient data produces no adjustment).
4. Integration test verifying compound sizing floor: when multiple loops stack multiplicatively, positions below $100 (or 1 share) are skipped entirely.
5. Documentation of the `.env.example` additions for all six flags.

## The Six Kill-Switch Flags

| Environment Variable | Controls | Default | When OFF (false) | When ON (true) |
|---------------------|----------|---------|------------------|----------------|
| `FEEDBACK_IC_WEIGHT_ADJUSTMENT` | IC-based signal weight adjustment (section-07) | `false` | `ic_factor` always 1.0; static weights used as-is | IC factors applied to collector weights in synthesis |
| `FEEDBACK_CORRELATION_PENALTY` | Signal correlation penalties (section-08) | `false` | `correlation_penalty` always 1.0; no signal downweighting for correlated collectors | Pairwise Spearman penalties applied to weaker signals |
| `FEEDBACK_CONVICTION_MULTIPLICATIVE` | Multiplicative conviction calibration (section-10) | `false` | Reverts to existing additive conviction adjustments | Six multiplicative factors (ADX, stability, timeframe, regime, ML, data quality) replace additive |
| `FEEDBACK_SHARPE_DEMOTION` | Live vs. backtest Sharpe demotion (section-12) | `false` | No demotion checks run; `sharpe_demotion_factor` always 1.0 | Daily supervisor check; auto-demotes strategies with live Sharpe < 50% of backtest for 21+ days |
| `FEEDBACK_DRIFT_DETECTION` | Concept drift detection and auto-retrain (section-13) | `false` | All drift checks skipped; no alerts, no auto-retrain | IC drift (daily), label drift (weekly), interaction drift (monthly) all active |
| `FEEDBACK_TRANSITION_SIZING` | Regime transition probability sizing (section-15) | `false` | `transition_factor` always 1.0; no sizing reduction during regime transitions | Sizing reduced proportionally to transition probability from HMM filtered probabilities |

**Key design principle:** When a flag is OFF, data collection continues. Agent skill tracking, IC attribution recording, signal correlation computation, and model shadow predictions all keep running. Only the sizing/weight adjustments are disabled. This means enabling a flag later does not require a cold-start wait -- historical data is already available.

## Files to Create or Modify

- **New file:** `src/quantstack/config/feedback_flags.py` -- centralized flag registry.
- **Modify:** `.env.example` -- add all six flags with `false` defaults and documentation comments.
- **New test file:** `tests/integration/test_feedback_flags.py` -- integration tests for flag isolation and compound behavior.
- **Verify (no changes):** Each section's implementation file already checks its own flag. This section confirms that wiring via integration tests.

## Tests (Write First)

All tests go in `tests/integration/test_feedback_flags.py`. These are integration-level tests that verify the flag wiring across module boundaries. Use `monkeypatch` to set environment variables. Mock DB calls via the existing `mock_settings` fixture pattern.

### Flag isolation tests

One test per flag. Each test verifies that setting the flag to `false` causes the corresponding adjustment to return its neutral value (1.0 or equivalent), and that no downstream side effects occur.

- **Test: `FEEDBACK_IC_WEIGHT_ADJUSTMENT=false` disables IC weight adjustment.** Set the flag to `false` via `monkeypatch.setenv`. Call the synthesis weight computation with IC data that would normally produce ic_factor < 1.0. Verify that all collectors retain their static weights unmodified. Verify no `SIGNAL_DEGRADATION` events are published.

- **Test: `FEEDBACK_IC_WEIGHT_ADJUSTMENT=true` enables IC weight adjustment.** Set the flag to `true`. Provide IC data where one collector has IC below 0.02. Verify that collector's effective weight is reduced by the sigmoid factor. Verify a `SIGNAL_DEGRADATION` event is published if the collector was previously healthy.

- **Test: `FEEDBACK_CORRELATION_PENALTY=false` disables correlation penalty.** Set the flag to `false`. Provide two collectors with Spearman correlation > 0.7. Verify both retain full weight (penalty = 1.0). Verify no penalty metadata is logged.

- **Test: `FEEDBACK_CORRELATION_PENALTY=true` enables correlation penalty.** Set the flag to `true`. Provide two collectors with correlation = 0.8 where one has lower IC. Verify the weaker collector's weight is penalized (penalty approx 0.2). Verify the stronger collector's weight is unchanged.

- **Test: `FEEDBACK_CONVICTION_MULTIPLICATIVE=false` reverts to additive.** Set the flag to `false`. Run conviction calibration with inputs that would produce different results under multiplicative vs. additive (e.g., low ADX + contradicting timeframe + data failure). Verify the output matches the legacy additive calculation, not the multiplicative one.

- **Test: `FEEDBACK_CONVICTION_MULTIPLICATIVE=true` uses multiplicative factors.** Set the flag to `true`. Provide the same inputs. Verify the output matches the multiplicative formula: `base * f1 * f2 * ... * f6`, clipped to [0.05, 0.95].

- **Test: `FEEDBACK_SHARPE_DEMOTION=false` skips demotion check entirely.** Set the flag to `false`. Call `run_sharpe_demotion_check()`. Verify no DB queries are executed, no events published, and the function returns immediately. Use a mock that raises an exception on any DB call to confirm no queries are attempted.

- **Test: `FEEDBACK_SHARPE_DEMOTION=true` runs demotion logic.** Set the flag to `true`. Provide a strategy with 21+ days of live Sharpe below 50% of backtest. Verify the demotion actions fire: status change, `force_scale(0.25)`, `STRATEGY_DEMOTED` event.

- **Test: `FEEDBACK_DRIFT_DETECTION=false` skips all drift checks.** Set the flag to `false`. Call `run_drift_detection()`. Verify no IC drift, label drift, or interaction drift checks are performed. No alerts, no auto-retrain decisions.

- **Test: `FEEDBACK_DRIFT_DETECTION=true` runs drift detection layers.** Set the flag to `true`. Provide feature data with IC dropped > 2 std from baseline. Verify an alert is raised and the auto-retrain decision tree is evaluated.

- **Test: `FEEDBACK_TRANSITION_SIZING=false` disables transition sizing.** Set the flag to `false`. In the `risk_sizing` node, provide a transition_probability of 0.5 (which would normally halve sizing). Verify the `transition_factor` is 1.0 and sizing is unaffected.

- **Test: `FEEDBACK_TRANSITION_SIZING=true` applies transition sizing.** Set the flag to `true`. Provide transition_probability = 0.5. Verify `transition_factor` = 0.50 and the final position size is halved.

### Cold-start tests

One test per feedback loop verifying that insufficient data produces no adjustment regardless of flag state.

- **Test: IC weight adjustment with < 21 days of IC data.** Flag is `true`, but only 10 days of IC data exist for each collector. Verify `ic_factor` = 1.0 for all collectors (no adjustment). The function should detect insufficient data and return neutral factors.

- **Test: Correlation penalty with < 63 days of signal data.** Flag is `true`, but only 30 days of signal data. Verify no correlation penalties are computed. The correlation matrix computation should return early.

- **Test: Conviction multiplicative with missing inputs.** Flag is `true`, but ADX data is unavailable (None). Verify the ADX factor defaults to 1.0 and all other factors are still computed. Missing inputs never cause exceptions.

- **Test: Sharpe demotion with < 21 days of live returns.** Flag is `true`, but strategy has only 15 days of return data. Verify `compute_live_sharpe()` returns None and no demotion action is taken.

- **Test: Drift detection with < 63 days of feature data.** Flag is `true`, but only 40 days of feature history. Verify all three drift layers are skipped with a log message.

- **Test: Transition sizing with HMM not fit.** Flag is `true`, but HMM failed to converge (transition_probability is None or 0.0). Verify `transition_factor` = 1.0.

- **Test: Agent quality with < 30 trades.** Flag is not applicable (agent quality does not have a kill-switch flag), but verify cold-start: agent with 15 trades has confidence = 1.0 and no `AGENT_DEGRADATION` alert.

### Compound sizing floor test

- **Test: compound multiplicative factors below $100 threshold skip the trade.** Set all six flags to `true`. Provide a scenario where: Kelly sizing = $1000, breaker_factor = 0.5 (SCALED), transition_factor = 0.5 (moderate transition risk), sharpe_demotion_factor = 0.25 (demoted). The compound result is $1000 * 0.5 * 0.5 * 0.25 = $62.50. Verify the trade is skipped (not placed). Verify a log message records the skip with all individual factors.

- **Test: compound multiplicative factors at exactly $100 threshold place the trade.** Same setup but Kelly sizing = $1600, producing $1600 * 0.5 * 0.5 * 0.25 = $100.00. Verify the trade IS placed.

- **Test: compound with all flags OFF produces no reduction.** All flags `false`. Verify Kelly sizing passes through to the risk gate without any multiplicative reduction from feedback loops. `breaker_factor` (from section-03 wiring, always active) is the only external multiplier.

### Flag independence test

- **Test: enabling one flag does not affect others.** Enable only `FEEDBACK_IC_WEIGHT_ADJUSTMENT=true`, all others `false`. Run a full synthesis cycle. Verify IC weight adjustment is applied but correlation penalty is 1.0, conviction is additive, and transition factor is 1.0. Repeat for each flag individually (parameterized test with 6 cases).

## Implementation Details

### Centralized flag registry

Create `src/quantstack/config/feedback_flags.py` with this structure:

```python
import os

def _flag(name: str) -> bool:
    """Read a boolean env var. Defaults to False (safe-off)."""
    return os.environ.get(name, "false").lower() in ("true", "1", "yes")

def ic_weight_adjustment_enabled() -> bool:
    """Section 07: IC-based signal weight adjustment."""
    return _flag("FEEDBACK_IC_WEIGHT_ADJUSTMENT")

def correlation_penalty_enabled() -> bool:
    """Section 08: Signal correlation penalties."""
    return _flag("FEEDBACK_CORRELATION_PENALTY")

def conviction_multiplicative_enabled() -> bool:
    """Section 10: Multiplicative conviction calibration."""
    return _flag("FEEDBACK_CONVICTION_MULTIPLICATIVE")

def sharpe_demotion_enabled() -> bool:
    """Section 12: Live vs. backtest Sharpe demotion."""
    return _flag("FEEDBACK_SHARPE_DEMOTION")

def drift_detection_enabled() -> bool:
    """Section 13: Concept drift detection and auto-retrain."""
    return _flag("FEEDBACK_DRIFT_DETECTION")

def transition_sizing_enabled() -> bool:
    """Section 15: Regime transition probability sizing."""
    return _flag("FEEDBACK_TRANSITION_SIZING")
```

Each section's implementation should import from this module rather than reading `os.environ` directly. This centralizes the flag naming convention, prevents typos across files, and provides a single place to change the parsing logic (e.g., if flags later move from env vars to a DB config table).

**Why functions instead of module-level constants:** Environment variables may be changed at runtime (e.g., via a supervisor command to enable a flag without restart). Functions re-read on each call. The overhead of `os.environ.get()` is negligible compared to any DB or LLM call.

### .env.example additions

Add the following block to `.env.example`, grouped together with a comment header:

```bash
# --- Feedback Loop Kill Switches (Phase 7) ---
# Each flag independently enables/disables one feedback adjustment.
# Default: false (safe). Enable one at a time after verifying data accumulation.
# Data collection (IC tracking, agent skills, etc.) runs regardless of these flags.
FEEDBACK_IC_WEIGHT_ADJUSTMENT=false
FEEDBACK_CORRELATION_PENALTY=false
FEEDBACK_CONVICTION_MULTIPLICATIVE=false
FEEDBACK_SHARPE_DEMOTION=false
FEEDBACK_DRIFT_DETECTION=false
FEEDBACK_TRANSITION_SIZING=false
```

### Wiring verification

Each dependent section already implements its own flag check. This section does not re-implement that logic -- it verifies it via integration tests. The expected flag check locations are:

| Flag | Check Location | Expected Pattern |
|------|---------------|-----------------|
| `FEEDBACK_IC_WEIGHT_ADJUSTMENT` | `signal_engine/synthesis.py`, inside weight computation | If flag is false, skip IC factor computation, use `ic_factor = 1.0` |
| `FEEDBACK_CORRELATION_PENALTY` | `signal_engine/synthesis.py` or `run_signal_correlation()` | If flag is false, skip penalty application, use `correlation_penalty = 1.0` |
| `FEEDBACK_CONVICTION_MULTIPLICATIVE` | `signal_engine/synthesis.py`, conviction adjustment block | If flag is false, run the legacy additive logic instead of multiplicative |
| `FEEDBACK_SHARPE_DEMOTION` | `run_sharpe_demotion_check()` in supervisor nodes | If flag is false, return immediately (no-op) |
| `FEEDBACK_DRIFT_DETECTION` | `run_drift_detection()` in supervisor nodes | If flag is false, return immediately (no-op) |
| `FEEDBACK_TRANSITION_SIZING` | `risk_sizing` node in trading graph | If flag is false, set `transition_factor = 1.0` |

### Compound sizing floor

The minimum tradeable size floor is defined in section-15-regime-transitions. The formula in `risk_sizing` is:

```
final_size = kelly_size * breaker_factor * transition_factor * sharpe_demotion_factor
```

After computing `final_size`, check:

```python
MINIMUM_TRADEABLE_VALUE = 100.0  # dollars

if final_size < MINIMUM_TRADEABLE_VALUE:
    logger.info(
        "Trade skipped: compound sizing below floor",
        strategy_id=strategy_id,
        kelly_size=kelly_size,
        breaker_factor=breaker_factor,
        transition_factor=transition_factor,
        sharpe_demotion_factor=sharpe_demotion_factor,
        final_size=final_size,
        floor=MINIMUM_TRADEABLE_VALUE,
    )
    return None  # skip the trade
```

This floor check happens BEFORE the risk gate. The risk gate remains the final authority for trades that pass the floor.

### Cold-start behavior summary

Every feedback mechanism must produce a neutral output (no adjustment) when insufficient data exists. The cold-start thresholds are:

| Feedback Loop | Cold-Start Condition | Neutral Output |
|--------------|---------------------|----------------|
| IC weight adjustment | < 21 days of per-collector IC data | `ic_factor = 1.0` |
| Correlation penalty | < 63 days of signal data | `correlation_penalty = 1.0` |
| Conviction multiplicative | Any factor input is None | That factor = 1.0 |
| Sharpe demotion | < 21 trading days of live returns | No demotion check |
| Drift detection | < 63 days of feature data | Skip all drift layers |
| Transition sizing | HMM not fit / < 120 bars | `transition_factor = 1.0` |
| Agent quality (always on) | < 30 trades per agent | `confidence = 1.0`, no alert |

Each section's implementation is responsible for enforcing its own cold-start check. The integration tests in this section verify that enforcement.

### Rollout procedure

The recommended rollout order (enable one flag at a time, observe for 1-2 weeks):

1. `FEEDBACK_CONVICTION_MULTIPLICATIVE=true` -- lowest risk, most observable (conviction values change but sizing pipeline unchanged). Compare trade conviction distributions before/after.
2. `FEEDBACK_IC_WEIGHT_ADJUSTMENT=true` -- moderate risk. Watch for SIGNAL_DEGRADATION events. Verify no collector drops to zero weight unexpectedly.
3. `FEEDBACK_CORRELATION_PENALTY=true` -- moderate risk. Watch effective independent signal count. Should be ~10-12 out of 22 collectors.
4. `FEEDBACK_TRANSITION_SIZING=true` -- moderate risk. Watch for excessive trade skipping during volatile periods. Monitor the compound floor trigger rate.
5. `FEEDBACK_SHARPE_DEMOTION=true` -- higher risk (changes strategy status). Verify backtest_sharpe is populated for all active strategies first.
6. `FEEDBACK_DRIFT_DETECTION=true` -- highest risk (triggers auto-retrain). Monitor MODEL_DEGRADATION events and retrain cooldown enforcement.

### Rollback

Any flag can be set back to `false` to immediately disable its adjustment. No code changes required. For flags that modify persistent state when ON (Sharpe demotion changes strategy status, drift detection may trigger retrains), manual cleanup of the state change is needed after rollback:

- Sharpe demotion rollback: set flag to `false` + update affected strategies back to `active` status + call `force_scale(strategy_id, 1.0)`.
- Drift detection rollback: set flag to `false`. Already-triggered retrains cannot be undone, but no new retrains will fire.
- All other flags: setting to `false` is a complete rollback with no residual state changes.

## Checklist

1. Write all tests in `tests/integration/test_feedback_flags.py` (they should fail initially).
2. Create `src/quantstack/config/feedback_flags.py` with the centralized flag registry.
3. Update `.env.example` with the six flag definitions.
4. Verify each dependent section imports from `feedback_flags.py` rather than reading env vars directly (may require minor refactors in sections 07, 08, 10, 12, 13, 15).
5. Implement the compound sizing floor check in `risk_sizing` if not already present from section-15.
6. Run all integration tests. Verify each flag independently disables its adjustment.
7. Run all cold-start tests. Verify neutral outputs with insufficient data.
8. Run the compound floor test. Verify micro-orders are skipped.
9. Run the flag independence test. Verify no cross-contamination between flags.
