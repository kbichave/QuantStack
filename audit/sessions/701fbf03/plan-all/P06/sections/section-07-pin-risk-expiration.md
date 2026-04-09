# Section 07 -- Pin Risk and Expiration Management

## Description

Add pin risk detection and automatic close logic for short options positions approaching expiration near the strike price. "Pin risk" occurs when a short option is near ATM close to expiry -- assignment becomes unpredictable and the position can flip between worthless and full assignment on small price moves. This section adds detection, alerting, and optional auto-close to the execution monitor.

## Files to Modify

| File | Change |
|------|--------|
| `src/quantstack/execution/execution_monitor.py` | Add `check_pin_risk()` method |
| `src/quantstack/config/feedback_flags.py` | Add `pin_risk_auto_close_enabled()` flag |

## What to Implement

### 1. `check_pin_risk()` in `execution_monitor.py`

Add this method to the `ExecutionMonitor` class (or as a standalone function called from the periodic check):

```python
from quantstack.config.feedback_flags import pin_risk_auto_close_enabled
from quantstack.db import db_conn

PIN_RISK_DTE_THRESHOLD = 3       # flag when DTE < 3 days
PIN_RISK_DISTANCE_PCT = 0.02     # flag when |spot - strike| / spot < 2%
PIN_RISK_AUTO_CLOSE_DTE = 1      # auto-close when DTE < 1 day

async def check_pin_risk(self) -> list[dict]:
    """
    Check all short option positions for pin risk.

    Pin risk criteria:
    - Position has short options (quantity < 0)
    - DTE < PIN_RISK_DTE_THRESHOLD (3 days)
    - |spot - strike| / spot < PIN_RISK_DISTANCE_PCT (2%)

    Actions:
    - Always: insert system_alerts row with severity=HIGH
    - If pin_risk_auto_close_enabled() AND DTE < PIN_RISK_AUTO_CLOSE_DTE:
      submit close order via trade_service

    Returns list of flagged positions for logging/tracing.
    """
    flagged = []

    with db_conn() as conn:
        short_options = conn.execute("""
            SELECT p.id, p.symbol, p.strike, p.expiry, p.quantity,
                   p.option_type, p.strategy_name, p.position_id
            FROM positions p
            WHERE p.asset_type = 'option'
              AND p.status = 'open'
              AND p.quantity < 0
        """).fetchall()

    if not short_options:
        return flagged

    for pos in short_options:
        symbol = pos["symbol"]
        strike = pos["strike"]
        expiry = pos["expiry"]
        quantity = pos["quantity"]

        # Compute DTE
        from datetime import date
        dte = (expiry - date.today()).days

        if dte >= PIN_RISK_DTE_THRESHOLD:
            continue

        # Get current spot
        spot = await self._get_spot_price(symbol)
        if spot <= 0:
            continue

        distance_pct = abs(spot - strike) / spot

        if distance_pct >= PIN_RISK_DISTANCE_PCT:
            continue

        # PIN RISK DETECTED
        alert = {
            "position_id": pos["position_id"],
            "symbol": symbol,
            "strike": strike,
            "expiry": str(expiry),
            "dte": dte,
            "spot": spot,
            "distance_pct": round(distance_pct, 4),
            "quantity": quantity,
            "option_type": pos["option_type"],
        }
        flagged.append(alert)

        # Insert system alert
        self._insert_pin_risk_alert(alert)

        # Auto-close if enabled and DTE < 1
        if pin_risk_auto_close_enabled() and dte < PIN_RISK_AUTO_CLOSE_DTE:
            logger.warning(
                f"PIN RISK AUTO-CLOSE: {symbol} {pos['option_type']} "
                f"strike={strike} expiry={expiry} DTE={dte} "
                f"distance={distance_pct:.2%} qty={quantity}"
            )
            await self._close_pin_risk_position(pos)
        else:
            logger.warning(
                f"PIN RISK FLAGGED: {symbol} {pos['option_type']} "
                f"strike={strike} expiry={expiry} DTE={dte} "
                f"distance={distance_pct:.2%} qty={quantity}"
            )

    return flagged
```

### 2. Alert insertion

```python
def _insert_pin_risk_alert(self, alert: dict) -> None:
    """Insert pin risk alert into system_alerts table."""
    import json
    with db_conn() as conn:
        conn.execute("""
            INSERT INTO system_alerts (severity, alert_type, message, metadata, created_at)
            VALUES ('HIGH', 'pin_risk', %s, %s, now())
        """, [
            f"Pin risk: {alert['symbol']} {alert['option_type']} "
            f"K={alert['strike']} exp={alert['expiry']} DTE={alert['dte']} "
            f"distance={alert['distance_pct']:.2%}",
            json.dumps(alert),
        ])
```

### 3. Auto-close execution

```python
async def _close_pin_risk_position(self, pos: dict) -> None:
    """Submit a closing order for a pin risk position."""
    try:
        from quantstack.execution.trade_service import submit_order

        # Close a short position by buying back
        close_qty = abs(pos["quantity"])  # buy back the short contracts

        submit_order(
            symbol=pos["symbol"],
            quantity=close_qty,
            order_type="market",
            tag="pin_risk_close",
            reason=f"Auto-close pin risk: DTE={pos['dte']} distance from strike < 2%",
            option_type=pos["option_type"],
            strike=pos["strike"],
            expiry=pos["expiry"],
        )

        # Update position status
        with db_conn() as conn:
            conn.execute("""
                UPDATE positions SET status = 'closing', notes = notes || ' [pin_risk_auto_close]'
                WHERE id = %s
            """, [pos["id"]])

    except Exception as e:
        logger.error(f"Pin risk auto-close failed for position {pos['id']}: {e}")
        # Insert a second alert for the failed close
        self._insert_pin_risk_alert({
            **pos,
            "auto_close_failed": True,
            "error": str(e),
        })
```

### 4. Feedback flag in `feedback_flags.py`

```python
def pin_risk_auto_close_enabled() -> bool:
    """P06: Automatically close short options with pin risk near expiry."""
    return _flag("FEEDBACK_PIN_RISK_AUTO_CLOSE")
```

Default: `True` via environment variable. Set `FEEDBACK_PIN_RISK_AUTO_CLOSE=true` in `.env`. The plan says default True, but the flag system defaults to False for safety. To make this effectively default-on, either:
- Set it in `.env` / Docker Compose env
- Or change the default in the flag function: `return os.environ.get("FEEDBACK_PIN_RISK_AUTO_CLOSE", "true").lower() in ("true", "1", "yes")`

Recommendation: Use the second approach (default True) since pin risk auto-close is a safety mechanism, not a performance optimization. Opting out should require explicit action.

### 5. Integration with execution monitor periodic check

In the existing periodic monitoring loop, add after existing checks:

```python
# Pin risk check (runs every cycle)
pin_risk_flags = await self.check_pin_risk()
if pin_risk_flags:
    logger.warning(f"Pin risk detected: {len(pin_risk_flags)} positions flagged")
```

## Tests to Write

File: `tests/unit/options/test_pin_risk.py`

1. **test_no_short_options_no_flags** -- Only long options in DB. Verify empty flagged list.
2. **test_dte_above_threshold_no_flag** -- Short put, DTE=5, spot near strike. Not flagged (DTE >= 3).
3. **test_dte_below_threshold_distance_above_no_flag** -- Short put, DTE=2, |spot-strike|/spot = 5%. Not flagged (distance too large).
4. **test_pin_risk_detected** -- Short call, DTE=2, spot=100, strike=101 (distance=1%). Verify flagged.
5. **test_pin_risk_detected_put** -- Short put, DTE=1, spot=100, strike=99 (distance=1%). Verify flagged.
6. **test_alert_inserted** -- Trigger pin risk. Verify row in `system_alerts` with `severity='HIGH'` and `alert_type='pin_risk'`.
7. **test_auto_close_when_enabled_and_dte_zero** -- DTE=0, auto_close enabled. Verify close order submitted via trade_service mock.
8. **test_no_auto_close_when_disabled** -- DTE=0, `FEEDBACK_PIN_RISK_AUTO_CLOSE=false`. Verify alert inserted but no close order.
9. **test_no_auto_close_when_dte_above_one** -- DTE=2, auto_close enabled. Verify alert only (DTE >= 1, auto-close requires DTE < 1).
10. **test_auto_close_failure_inserts_second_alert** -- Mock trade_service to raise. Verify second alert with `auto_close_failed=True`.
11. **test_distance_at_exact_threshold** -- |spot-strike|/spot = 0.02 exactly. This is `>=` threshold, so NOT flagged (boundary: strictly less than 2%).
12. **test_multiple_positions_independently_checked** -- 3 short options, 1 has pin risk. Verify only 1 flagged.

## Edge Cases

- **DTE = 0 (expiration day)**: The most critical case. `(expiry - today).days = 0`. Auto-close should fire immediately. If market is already closed on expiration Friday, the order queues for next session (Saturday/Monday depending on broker). This may be too late. Consider adding a `pre_expiry_close_hour` check: if DTE=0 and current time < 15:30 ET, submit close.
- **After-hours assignment**: Options can be exercised after market close on expiration day. Auto-close during market hours reduces but does not eliminate this risk.
- **Spread positions**: A short leg in a spread is hedged by the long leg. Flagging pin risk on the short leg alone may be overly aggressive. Enhancement: check if the position is part of a defined-risk structure (via `structure_type`) and suppress the alert if the long wing covers the short leg.
- **Zero spot price**: Skip the position with a warning. Do not divide by zero.
- **Expiry on weekend/holiday**: `(expiry - today).days` may be 0 on a Friday for Saturday expiry. Most equity options expire Friday close. This is correct behavior.
- **Position already in `closing` status**: Query filters for `status = 'open'` only. A position already being closed will not be double-flagged.
- **Multiple cycles flagging the same position**: The alert table gets a new row each cycle. Consider adding a deduplication check: do not insert if an alert for the same `position_id` and `alert_type='pin_risk'` exists within the last 15 minutes. Or use `ON CONFLICT DO NOTHING` with a unique index on `(position_id, alert_type, DATE(created_at))`.
