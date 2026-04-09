# Section 08 -- Greek P&L Attribution

## Description

Implement daily Greek-based P&L attribution for options positions. Decomposes the actual realized P&L into delta, gamma, theta, and vega components, with an unexplained residual. This enables the system to understand which Greeks are driving returns (and losses), informing strategy selection and hedging decisions.

## Files to Create

| File | Purpose |
|------|---------|
| (none -- all code goes in existing files) | |

## Files to Modify

| File | Change |
|------|--------|
| `src/quantstack/core/options/engine.py` | Add `compute_greek_pnl_attribution()` function |
| `scripts/scheduler.py` | Add daily EOD schedule entry for attribution |

## What to Implement

### 1. `compute_greek_pnl_attribution()` in `engine.py`

```python
from datetime import date, timedelta
from quantstack.db import db_conn

def compute_greek_pnl_attribution(
    attribution_date: date | None = None,
) -> list[dict]:
    """
    Compute Greek-based P&L attribution for all symbols with options positions.

    For each symbol on the given date:
      delta_pnl  = delta * price_change
      gamma_pnl  = 0.5 * gamma * price_change^2
      theta_pnl  = theta * 1  (one trading day)
      vega_pnl   = vega * iv_change
      unexplained = actual_pnl - (delta + gamma + theta + vega)

    Greeks used are the START-OF-DAY values (from the most recent portfolio_greeks_history
    snapshot before market open). Price and IV changes are intraday (open to close).

    Args:
        attribution_date: Date to compute attribution for. Defaults to today.

    Returns:
        List of per-symbol attribution dicts.
    """
    if attribution_date is None:
        attribution_date = date.today()

    prev_date = attribution_date - timedelta(days=1)
    # Adjust for weekends: if prev_date is Sunday, use Friday
    while prev_date.weekday() >= 5:
        prev_date -= timedelta(days=1)

    results = []

    # Get symbols with open options positions
    with db_conn() as conn:
        symbols_rows = conn.execute("""
            SELECT DISTINCT symbol FROM positions
            WHERE asset_type = 'option' AND status IN ('open', 'closed')
              AND (opened_at::date <= %s OR closed_at::date = %s)
        """, [attribution_date, attribution_date]).fetchall()

    symbols = [r["symbol"] for r in symbols_rows]
    if not symbols:
        return results

    for symbol in symbols:
        attribution = _compute_symbol_attribution(symbol, attribution_date, prev_date)
        if attribution:
            results.append(attribution)
            _store_attribution(attribution, attribution_date)

    return results


def _compute_symbol_attribution(
    symbol: str, attr_date: date, prev_date: date
) -> dict | None:
    """
    Compute attribution for a single symbol.

    Data sources:
    - Start-of-day Greeks: latest portfolio_greeks_history snapshot before 09:30 ET on attr_date
    - Price change: close(attr_date) - close(prev_date)
    - IV change: atm_iv(attr_date) - atm_iv(prev_date)  (from options_chain or IV surface snapshots)
    - Actual P&L: sum of position P&L changes for this symbol on attr_date
    """
    with db_conn() as conn:
        # Get start-of-day Greeks from portfolio_greeks_history
        greeks_row = conn.execute("""
            SELECT symbol_greeks FROM portfolio_greeks_history
            WHERE snapshot_time::date = %s
            ORDER BY snapshot_time ASC
            LIMIT 1
        """, [attr_date]).fetchone()

        if not greeks_row:
            logger.warning(f"No Greeks snapshot for {attr_date}, skipping attribution")
            return None

    import json
    symbol_greeks = json.loads(greeks_row["symbol_greeks"] or "{}")
    greeks = symbol_greeks.get(symbol)
    if not greeks:
        return None

    delta = greeks.get("delta", 0)  # dollar delta
    gamma = greeks.get("gamma", 0)  # dollar gamma (for 1% move)
    theta = greeks.get("theta", 0)  # dollar theta per day
    vega = greeks.get("vega", 0)    # dollar vega per vol point

    # Get price change
    price_change = _get_price_change(symbol, prev_date, attr_date)
    spot_prev = _get_close_price(symbol, prev_date)

    # Get IV change (ATM IV)
    iv_change = _get_iv_change(symbol, prev_date, attr_date)

    # Compute attributed P&L components
    # delta_pnl: delta_dollars already incorporates spot and multiplier
    # For a $1 move in underlying: pnl_change = delta_dollars * (price_change / spot)
    if spot_prev and spot_prev > 0:
        pct_move = price_change / spot_prev
        delta_pnl = delta * pct_move
        gamma_pnl = gamma * (pct_move / 0.01) ** 2 * 0.01  # gamma_dollars is for 1% move
    else:
        delta_pnl = 0
        gamma_pnl = 0

    theta_pnl = theta  # already $/day
    vega_pnl = vega * iv_change  # vega is $/vol-point, iv_change in vol-points

    attributed_total = delta_pnl + gamma_pnl + theta_pnl + vega_pnl

    # Get actual P&L
    actual_pnl = _get_actual_options_pnl(symbol, attr_date)
    unexplained = actual_pnl - attributed_total

    return {
        "symbol": symbol,
        "date": attr_date,
        "delta_pnl": round(delta_pnl, 2),
        "gamma_pnl": round(gamma_pnl, 2),
        "theta_pnl": round(theta_pnl, 2),
        "vega_pnl": round(vega_pnl, 2),
        "unexplained_pnl": round(unexplained, 2),
        "total_pnl": round(actual_pnl, 2),
        "inputs": {
            "price_change": price_change,
            "iv_change": iv_change,
            "delta": delta,
            "gamma": gamma,
            "theta": theta,
            "vega": vega,
        },
    }
```

### 2. Helper functions

```python
def _get_price_change(symbol: str, prev_date: date, attr_date: date) -> float:
    """Get close-to-close price change."""
    with db_conn() as conn:
        rows = conn.execute("""
            SELECT date, close FROM daily_prices
            WHERE symbol = %s AND date IN (%s, %s)
            ORDER BY date
        """, [symbol, prev_date, attr_date]).fetchall()

    if len(rows) < 2:
        return 0.0
    return rows[1]["close"] - rows[0]["close"]

def _get_close_price(symbol: str, d: date) -> float:
    """Get closing price for a date."""
    with db_conn() as conn:
        row = conn.execute(
            "SELECT close FROM daily_prices WHERE symbol = %s AND date = %s",
            [symbol, d]
        ).fetchone()
    return row["close"] if row else 0.0

def _get_iv_change(symbol: str, prev_date: date, attr_date: date) -> float:
    """
    Get ATM IV change between two dates.

    Uses portfolio_greeks_history or a dedicated IV history table.
    Falls back to 0 if data unavailable (conservative: no vega attribution).
    """
    # Implementation: query IV surface snapshots or options chain history
    # For MVP: return 0.0 (theta and delta attribution are more reliable)
    return 0.0

def _get_actual_options_pnl(symbol: str, attr_date: date) -> float:
    """
    Get actual options P&L for a symbol on a given date.

    Computed from position mark-to-market changes.
    """
    with db_conn() as conn:
        row = conn.execute("""
            SELECT COALESCE(SUM(
                (current_price - prev_close_price) * quantity * 100
            ), 0) as pnl
            FROM positions
            WHERE symbol = %s AND asset_type = 'option' AND status = 'open'
        """, [symbol]).fetchone()
    return row["pnl"] if row else 0.0
```

### 3. Store attribution in DB

```python
def _store_attribution(attribution: dict, attr_date: date) -> None:
    """Upsert attribution row into options_pnl_attribution table."""
    with db_conn() as conn:
        conn.execute("""
            INSERT INTO options_pnl_attribution
                (date, symbol, delta_pnl, gamma_pnl, theta_pnl, vega_pnl,
                 unexplained_pnl, total_pnl)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (date, symbol) DO UPDATE SET
                delta_pnl = EXCLUDED.delta_pnl,
                gamma_pnl = EXCLUDED.gamma_pnl,
                theta_pnl = EXCLUDED.theta_pnl,
                vega_pnl = EXCLUDED.vega_pnl,
                unexplained_pnl = EXCLUDED.unexplained_pnl,
                total_pnl = EXCLUDED.total_pnl
        """, [
            attr_date, attribution["symbol"],
            attribution["delta_pnl"], attribution["gamma_pnl"],
            attribution["theta_pnl"], attribution["vega_pnl"],
            attribution["unexplained_pnl"], attribution["total_pnl"],
        ])
```

### 4. Scheduler integration

In `scripts/scheduler.py`, add a daily EOD job:

```python
# Run at 16:15 ET (15 minutes after market close)
scheduler.add_job(
    run_pnl_attribution,
    trigger="cron",
    hour=16, minute=15,
    timezone="US/Eastern",
    id="options_pnl_attribution",
    name="Daily options P&L attribution",
)

async def run_pnl_attribution():
    from quantstack.core.options.engine import compute_greek_pnl_attribution
    results = compute_greek_pnl_attribution()
    logger.info(f"P&L attribution computed for {len(results)} symbols")
```

## Tests to Write

File: `tests/unit/options/test_pnl_attribution.py`

1. **test_known_delta_pnl** -- Position with delta_dollars=$5,000, spot moves from $100 to $102 (+2%). `delta_pnl = 5000 * 0.02 = $100`. Verify.
2. **test_known_gamma_pnl** -- Position with gamma_dollars=$250 (per 1% move), spot moves 3%. `gamma_pnl = 250 * (3/1)^2 * 0.01 = 250 * 9 * 0.01 = $22.50`. Verify.
3. **test_theta_pnl_one_day** -- Position with theta_dollars=-$50/day. After 1 day, `theta_pnl = -$50`. Verify.
4. **test_vega_pnl_iv_increase** -- Vega_dollars=$200, IV increases by 0.05 (5 vol points). `vega_pnl = 200 * 0.05 = $10`. Verify.
5. **test_unexplained_captures_residual** -- Actual PnL = $150, attributed = $120. `unexplained = $30`. Verify.
6. **test_no_greeks_snapshot_returns_none** -- No portfolio_greeks_history for the date. Verify symbol skipped.
7. **test_upsert_overwrites_existing** -- Insert attribution, then recompute same date/symbol with different values. Verify updated.
8. **test_weekend_prev_date_skips_to_friday** -- Attribution date is Monday. `prev_date` should resolve to Friday (not Saturday/Sunday).
9. **test_zero_price_change_delta_pnl_zero** -- Flat day. All directional P&L components should be zero. Theta still non-zero.
10. **test_multiple_symbols_independent** -- 3 symbols, each with different Greeks and price moves. Verify independent computation.

## Edge Cases

- **Missing price data**: If `daily_prices` has no row for `attr_date`, `price_change = 0`. This means delta and gamma attribution are zero, and the entire actual P&L goes to `unexplained`. This is correct behavior -- the system acknowledges it cannot attribute without data.
- **IV data unavailability**: `_get_iv_change` returns 0 for MVP. All vega P&L goes to `unexplained`. This is conservative and correct. When IV history is added, vega attribution will improve and unexplained will shrink.
- **Positions opened and closed same day**: These contribute to actual P&L but may not appear in start-of-day Greeks (they did not exist at snapshot time). Their P&L will be entirely unexplained. Acceptable for daily attribution.
- **Large unexplained residual**: If `|unexplained| > |total_pnl| * 0.5`, the attribution is unreliable. Consider logging a warning. Common causes: stale Greeks, missing IV data, positions opened/closed intra-day, jumps in underlying.
- **Greeks snapshot timing**: Using the earliest snapshot of the day assumes Greeks are stable from pre-market through close. For volatile days, intraday Greeks changes contribute to unexplained. Enhancement: use time-weighted average Greeks (requires multiple intraday snapshots from section-05).
- **Negative gamma P&L on large moves**: Short gamma positions (short options) have negative gamma_dollars. A large move produces `gamma_pnl = negative * move^2`, which is a large negative number. This is correct -- it reflects the convexity cost of being short gamma.
- **ON CONFLICT for upsert**: Requires the unique index `idx_opa_date_symbol` from section-01. If that index is missing, the upsert fails with a Postgres error. Section-01 must be completed first.
