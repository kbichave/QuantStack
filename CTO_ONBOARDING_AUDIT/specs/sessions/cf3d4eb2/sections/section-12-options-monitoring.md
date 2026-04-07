# Section 12: Options Monitoring Rules

## Overview

The execution monitor (`src/quantstack/execution/execution_monitor.py`) enforces deterministic exit rules on all positions: kill switch, hard stop-loss, take profit, trailing stop, time stop, and intraday flatten. All six rules are equity-centric. Options positions pass through the same rules, but none account for options-specific risks: accelerating theta decay near expiry, pin risk at the strike, early assignment around ex-dividend dates, IV crush after earnings, or cumulative theta erosion.

This section adds five options-specific monitoring rules to the execution monitor. Each rule has a configurable action (`auto_exit` or `flag_only`), and equity positions skip evaluation entirely.

## Dependencies

- **Section 01 (Schema Foundation):** The schema migration infrastructure must be in place if any new tables are needed for IV snapshots or earnings tracking.
- **Options engine Greeks:** `compute_greeks_dispatch()` in `src/quantstack/core/options/engine.py` (lines 293-364) already computes delta, gamma, theta, vega, and rho. This section calls it to get live Greeks for monitored options positions.
- **MonitoredPosition:** Already has `instrument_type`, `option_contract`, `entry_price`, and `underlying_symbol` fields (lines 88-104 of `execution_monitor.py`).
- **`_submit_exit()`:** Existing method (line 354) already dispatches to `_submit_options_exit()` for options positions. No changes needed to exit submission.

## Files to Create or Modify

| File | Action |
|------|--------|
| `src/quantstack/execution/execution_monitor.py` | Add `_evaluate_options_rules()` method, `OptionsMonitorRule` dataclass, default rule config, call from `_on_price_update` |
| `tests/unit/execution/test_options_monitoring.py` | New test file for all options monitoring rules |

No new database tables are required for the initial implementation. IV snapshots and earnings calendar support (needed for the `iv_crush` and `assignment_risk` rules) are deferred until those data sources are available. The three rules that depend only on Greeks and position data (`theta_acceleration`, `pin_risk`, `max_theta_loss`) ship first.

## Tests

All tests go in `tests/unit/execution/test_options_monitoring.py`. Write these before implementation.

```python
# --- theta_acceleration ---

# Test: theta_acceleration triggers when DTE < 7 AND theta/premium > 5%/day
#   Setup: MonitoredPosition with instrument_type="options", DTE=5, current_premium=2.00,
#          theta=-0.12 (6% of premium per day).
#   Expected: rule triggers, action is auto_exit.

# Test: theta_acceleration does NOT trigger when DTE = 8
#   Setup: same theta/premium ratio but DTE=8.
#   Expected: rule does not trigger.

# Test: theta_acceleration does NOT trigger when theta/premium < 5%
#   Setup: DTE=5, current_premium=4.00, theta=-0.10 (2.5%/day).
#   Expected: rule does not trigger.

# --- pin_risk ---

# Test: pin_risk triggers when DTE < 3 AND price within 1% of strike
#   Setup: DTE=2, strike=100, underlying_price=100.50 (0.5% from strike).
#   Expected: rule triggers, action is auto_exit.

# Test: pin_risk does NOT trigger when DTE = 4
#   Setup: DTE=4, price within 1% of strike.
#   Expected: rule does not trigger.

# Test: pin_risk does NOT trigger when price > 1% from strike
#   Setup: DTE=2, strike=100, underlying_price=105.
#   Expected: rule does not trigger.

# --- assignment_risk ---

# Test: assignment_risk triggers when short call ITM AND ex-div within 2 days
#   Setup: short call, strike=95, underlying=100, ex_div_date in 1 day.
#   Expected: rule triggers, action is flag_only.

# Test: assignment_risk does NOT trigger for long call (only short calls face early assignment)

# --- iv_crush ---

# Test: iv_crush triggers when post-earnings AND IV dropped > 30%
#   Setup: earnings 1 day ago, pre-earnings IV=0.60, current IV=0.35 (42% drop).
#   Expected: rule triggers, action is flag_only.

# Test: iv_crush does NOT trigger when IV drop < 30%
#   Setup: earnings 1 day ago, pre-earnings IV=0.60, current IV=0.50 (17% drop).
#   Expected: rule does not trigger.

# --- max_theta_loss ---

# Test: max_theta_loss triggers when cumulative decay > 40% of entry premium
#   Setup: entry_premium=5.00, current_premium=2.80 (decay = 2.20 = 44% of entry).
#   Expected: rule triggers, action is auto_exit.

# Test: max_theta_loss does NOT trigger when decay < 40%
#   Setup: entry_premium=5.00, current_premium=3.50 (decay = 30%).
#   Expected: rule does not trigger.

# --- action configuration ---

# Test: auto_exit rules call _submit_exit()
#   Setup: theta_acceleration triggers with action="auto_exit".
#   Expected: _submit_exit called with reason containing rule name.

# Test: flag_only rules log alert but do NOT trigger exit
#   Setup: assignment_risk triggers with action="flag_only".
#   Expected: alert logged, _submit_exit NOT called.

# Test: rule configuration overrides default action
#   Setup: override pin_risk to action="flag_only" (default is "auto_exit").
#   Expected: pin_risk triggers but does not call _submit_exit.

# --- equity skip ---

# Test: equity positions skip options rule evaluation
#   Setup: MonitoredPosition with instrument_type="equity".
#   Expected: _evaluate_options_rules returns (False, "") immediately.
```

## Rule Configuration

Define a dataclass for rule configuration and a default set of rules. Place these at module level in `execution_monitor.py`, near the existing `MonitoredPosition` and `HoldingType` definitions.

```python
@dataclass
class OptionsMonitorRule:
    name: str
    enabled: bool
    action: str  # "auto_exit" | "flag_only"
```

Default rule configuration:

| Rule | Default Action | Rationale |
|------|---------------|-----------|
| `theta_acceleration` | `auto_exit` | Theta decay is exponential near expiry; holding is almost always wrong |
| `pin_risk` | `auto_exit` | Unpredictable delta and settlement risk at the strike near expiry |
| `assignment_risk` | `flag_only` | Requires human judgment on dividend capture vs assignment cost |
| `iv_crush` | `flag_only` | Post-earnings IV collapse may already be priced into position thesis |
| `max_theta_loss` | `auto_exit` | Cumulative erosion past 40% of entry premium signals thesis failure |

Store defaults as a module-level dict. Allow override via environment variable or config file (implementation detail left to implementer's judgment -- the key contract is that rules are configurable at startup).

## Rule Definitions

### Theta Acceleration

**Trigger condition:** `DTE < 7 AND abs(theta) / current_premium > 0.05`

Theta decay accelerates exponentially in the final week before expiry. When daily theta loss exceeds 5% of the option's current premium, the position is burning value faster than most directional moves can compensate.

**Inputs needed:**
- DTE: computed from `option_expiry` on the position (already available via `MonitoredPosition`)
- Theta: from `compute_greeks_dispatch()` using current spot, strike, time-to-expiry, and IV
- Current premium: the option's current market price (from price feed)

### Pin Risk

**Trigger condition:** `DTE < 3 AND abs(underlying_price - strike) / strike < 0.01`

Options near the strike at expiry have highly unstable delta. Settlement outcome (ITM vs OTM) is uncertain, and gamma exposure is extreme. Auto-exit avoids the coin-flip.

**Inputs needed:**
- DTE: from position expiry
- Underlying price: from price feed (use `underlying_symbol` on `MonitoredPosition`)
- Strike: from position data

### Assignment Risk

**Trigger condition:** `position is short call AND call is ITM AND ex_dividend_date within 2 trading days`

Early assignment risk on short calls spikes when time value drops below the dividend amount. The option holder exercises to capture the dividend.

**Inputs needed:**
- Position side and option type (short call)
- Moneyness: underlying price vs strike
- Ex-dividend calendar data: **not currently available**. This rule ships as `enabled=False` by default until an ex-dividend data source is integrated. The rule logic and tests should be complete; only the data feed is deferred.

### IV Crush

**Trigger condition:** `earnings_event within 2 trading days AND (pre_earnings_iv - current_iv) / pre_earnings_iv > 0.30`

After earnings, implied volatility collapses as uncertainty resolves. If IV drops more than 30%, long options lose significant extrinsic value regardless of direction.

**Inputs needed:**
- Earnings date: derivable from Alpha Vantage earnings calendar
- Pre-earnings IV snapshot: requires capturing IV before the earnings event
- Current IV: from options pricing

**Deferred data:** Like assignment risk, this rule ships as `enabled=False` until earnings calendar integration and IV snapshot capture are implemented. Write the rule logic and tests with mock data so the rule is ready to activate.

### Max Theta Loss

**Trigger condition:** `(entry_premium - current_premium) / entry_premium > 0.40`

When cumulative time decay erodes more than 40% of the entry premium, the original thesis has likely failed. This is a position-level stop on theta erosion, independent of P&L (which may be masked by delta gains on a moving underlying).

**Inputs needed:**
- Entry premium: already stored on the position (`entry_price` for options is the premium paid)
- Current premium: from price feed

## Integration into Execution Monitor

### Where to Add

The options rule evaluation happens inside the `_on_price_update` callback in `ExecutionMonitor`. After the existing `evaluate_rules()` call on `MonitoredPosition` (which handles kill switch, stop-loss, take profit, trailing stop, time stop, and intraday flatten), add a second evaluation pass for options-specific rules.

The flow becomes:

1. Existing `position.evaluate_rules(price, timestamp, kill_active)` -- if this returns `should_exit=True`, exit immediately (equity rules take priority).
2. If equity rules did NOT trigger exit, AND `position.instrument_type == "options"`, call `self._evaluate_options_rules(position, price, timestamp)`.
3. `_evaluate_options_rules` returns `(should_exit: bool, reason: str)`. If `should_exit`, call `_submit_exit`. If a `flag_only` rule triggered, log the alert but return `(False, "")`.

### Method Signature

Add to `ExecutionMonitor`:

```python
async def _evaluate_options_rules(
    self,
    position: MonitoredPosition,
    current_price: float,
    current_time: datetime,
) -> tuple[bool, str]:
    """Evaluate options-specific exit rules.

    Skips non-options positions. For each enabled rule, checks the trigger
    condition. auto_exit rules return (True, reason). flag_only rules log
    an alert and return (False, "").

    Returns:
        (should_exit, reason) -- reason includes rule name for audit trail.
    """
    ...
```

### Greeks Fetching

Inside `_evaluate_options_rules`, fetch Greeks once per evaluation cycle (not per rule):

1. Extract strike, expiry, option_type from position data.
2. Compute DTE from expiry and current_time.
3. Call `compute_greeks_dispatch(spot=current_price, strike=strike, time_to_expiry=dte_years, vol=iv, option_type=option_type)`.
4. Extract theta from the result dict at `result["greeks"]["theta"]`.

**IV source:** The current options engine requires IV as an input. For monitoring purposes, use the IV implied from the option's current market price via a root-finding approach, or use the last known IV from the position's entry data as a fallback. The exact IV source is an implementation detail -- the key requirement is that `compute_greeks_dispatch` receives a reasonable volatility estimate.

**Error handling:** If Greeks computation fails (missing data, numerical issues), skip options rule evaluation for this tick and log a warning. Do not block equity rule evaluation. Do not crash the monitor loop.

### Logging and Alerting

For `flag_only` rules that trigger:
- Log at WARNING level: `[ExecMonitor] OPTIONS_ALERT {rule_name}: {symbol} — {details}`
- Optionally write to an alerts table in the database (reuse existing alert infrastructure if available)
- The alert should include: rule name, symbol, contract, trigger values (e.g., "DTE=2, distance_to_strike=0.5%"), and the configured action

For `auto_exit` rules that trigger:
- Log at INFO level with the same detail as equity exits
- The `reason` string passed to `_submit_exit` should be prefixed with `options_` (e.g., `options_theta_acceleration`, `options_pin_risk`) so it is distinguishable in the `closed_trades` table's `exit_reason` column

### MonitoredPosition Additions

The `MonitoredPosition` dataclass may need additional fields to support these rules:

- `option_strike: float | None` -- already derivable from `option_contract` but worth storing explicitly for rule evaluation
- `option_expiry: date | None` -- same
- `option_type: str | None` -- "call" or "put"
- `entry_premium: float | None` -- the premium at entry (for max_theta_loss)

Check whether the existing `from_portfolio_position` classmethod already populates these from the `Position` object. If not, extend it. The underlying `Position` dataclass in `portfolio_state.py` already has `option_strike`, `option_expiry`, `option_type`, and `entry_price` fields.

## Implementation Notes

- **Rule evaluation order:** Evaluate all enabled rules. If multiple `auto_exit` rules trigger simultaneously, use the first one in priority order (theta_acceleration > pin_risk > max_theta_loss). Only one exit is submitted.
- **Performance:** Greeks computation involves numerical methods (vollib backend). For a small number of options positions (typical for this account size), per-tick computation is acceptable. If position count grows, consider caching Greeks with a short TTL (e.g., 30 seconds).
- **Testing strategy:** Mock `compute_greeks_dispatch` in tests. Build `MonitoredPosition` instances with controlled parameters. Do not require a live options engine or price feed for unit tests.
- **Deferred rules:** `assignment_risk` and `iv_crush` ship with `enabled=False`. Their test cases use mock data for ex-dividend dates and IV snapshots. When the data sources become available, flip `enabled=True` and the rules activate without code changes.
