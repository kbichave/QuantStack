# Section 05 -- Portfolio Greeks Aggregation

## Description

Implement portfolio-level Greeks aggregation in dollar terms. Computes per-position, per-symbol, per-strategy, and portfolio-total Greeks from all open options positions. Snapshots are stored in the `portfolio_greeks_history` table (created in section-01) for time-series analysis and hedging decisions.

## Files to Create

| File | Purpose |
|------|---------|
| (none -- all code goes in existing files) | |

## Files to Modify

| File | Change |
|------|--------|
| `src/quantstack/core/options/engine.py` | Add `compute_portfolio_greeks()` function |
| `src/quantstack/core/options/models.py` | Add `PortfolioGreeks` dataclass |
| `src/quantstack/execution/execution_monitor.py` | Call `compute_portfolio_greeks()` in periodic check cycle |

## What to Implement

### 1. `PortfolioGreeks` dataclass in `models.py`

```python
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class GreeksBreakdown:
    """Greeks for a single aggregation unit (position, symbol, or strategy)."""
    delta_dollars: float = 0.0
    gamma_dollars: float = 0.0
    theta_dollars: float = 0.0
    vega_dollars: float = 0.0
    rho_dollars: float = 0.0

@dataclass
class PortfolioGreeks:
    """Portfolio-level Greeks in dollar terms with drill-down breakdowns."""
    snapshot_time: datetime = field(default_factory=datetime.utcnow)

    # Portfolio totals
    total_delta: float = 0.0    # dollar delta
    total_gamma: float = 0.0    # dollar gamma
    total_theta: float = 0.0    # dollar theta per day
    total_vega: float = 0.0     # dollar vega per 1% IV move
    total_rho: float = 0.0      # dollar rho per 1% rate move

    # Breakdowns
    per_symbol: dict[str, GreeksBreakdown] = field(default_factory=dict)
    per_strategy: dict[str, GreeksBreakdown] = field(default_factory=dict)

    def to_db_row(self) -> dict:
        """Convert to dict matching portfolio_greeks_history table columns."""
        import json
        return {
            "snapshot_time": self.snapshot_time,
            "symbol_greeks": json.dumps({
                sym: {"delta": g.delta_dollars, "gamma": g.gamma_dollars,
                      "theta": g.theta_dollars, "vega": g.vega_dollars, "rho": g.rho_dollars}
                for sym, g in self.per_symbol.items()
            }),
            "strategy_greeks": json.dumps({
                strat: {"delta": g.delta_dollars, "gamma": g.gamma_dollars,
                        "theta": g.theta_dollars, "vega": g.vega_dollars, "rho": g.rho_dollars}
                for strat, g in self.per_strategy.items()
            }),
            "portfolio_delta": self.total_delta,
            "portfolio_gamma": self.total_gamma,
            "portfolio_theta": self.total_theta,
            "portfolio_vega": self.total_vega,
            "portfolio_rho": self.total_rho,
        }
```

### 2. `compute_portfolio_greeks()` in `engine.py`

```python
from quantstack.db import db_conn
from quantstack.core.options.models import PortfolioGreeks, GreeksBreakdown

def compute_portfolio_greeks() -> PortfolioGreeks:
    """
    Compute portfolio-level Greeks in dollar terms from all open options positions.

    Dollar conversion:
      delta_dollars = delta * spot * quantity * 100  (multiplier)
      gamma_dollars = 0.5 * gamma * spot^2 * quantity * 100 * 0.01  (for 1% move)
      theta_dollars = theta * quantity * 100  (already in $/day from BS)
      vega_dollars  = vega * quantity * 100  ($/1-vol-point from BS)
      rho_dollars   = rho * quantity * 100

    Steps:
    1. Query all open options positions from DB (asset_type='option', status='open')
    2. For each position: fetch current spot price, compute Greeks via compute_greeks_dispatch
    3. Aggregate: position -> symbol -> strategy -> total
    4. Return PortfolioGreeks with all breakdowns
    """
    portfolio = PortfolioGreeks()

    with db_conn() as conn:
        rows = conn.execute("""
            SELECT p.id, p.symbol, p.strategy_name, p.quantity, p.strike,
                   p.expiry, p.option_type, p.entry_price
            FROM positions p
            WHERE p.asset_type = 'option' AND p.status = 'open'
        """).fetchall()

    if not rows:
        return portfolio

    # Group by symbol for spot price fetching (one quote per symbol)
    symbols = set(row["symbol"] for row in rows)
    spot_prices = _fetch_spot_prices(symbols)  # helper: returns {symbol: float}

    for row in rows:
        symbol = row["symbol"]
        spot = spot_prices.get(symbol, 0)
        if spot <= 0:
            logger.warning(f"No spot price for {symbol}, skipping Greeks computation")
            continue

        strike = row["strike"]
        expiry = row["expiry"]
        qty = row["quantity"]
        opt_type = row["option_type"]

        # Compute time to expiry
        from datetime import date
        dte = (expiry - date.today()).days
        tte = max(dte / 365.0, 1 / 365.0)  # floor at 1 day

        # Get IV from chain or use last known
        iv = _get_current_iv(symbol, strike, expiry, opt_type)  # helper
        if iv <= 0:
            iv = 0.30  # fallback: assume 30% vol

        greeks = compute_greeks_dispatch(
            spot=spot, strike=strike, time_to_expiry=tte,
            volatility=iv, option_type=opt_type,
        )

        # Dollar conversion
        multiplier = 100  # standard equity option multiplier
        delta_d = greeks["delta"] * spot * qty * multiplier
        gamma_d = 0.5 * greeks["gamma"] * (spot ** 2) * qty * multiplier * 0.01
        theta_d = greeks["theta"] * qty * multiplier
        vega_d  = greeks["vega"] * qty * multiplier
        rho_d   = greeks["rho"] * qty * multiplier

        # Accumulate to symbol breakdown
        if symbol not in portfolio.per_symbol:
            portfolio.per_symbol[symbol] = GreeksBreakdown()
        sym_g = portfolio.per_symbol[symbol]
        sym_g.delta_dollars += delta_d
        sym_g.gamma_dollars += gamma_d
        sym_g.theta_dollars += theta_d
        sym_g.vega_dollars += vega_d
        sym_g.rho_dollars += rho_d

        # Accumulate to strategy breakdown
        strategy = row.get("strategy_name", "unknown")
        if strategy not in portfolio.per_strategy:
            portfolio.per_strategy[strategy] = GreeksBreakdown()
        strat_g = portfolio.per_strategy[strategy]
        strat_g.delta_dollars += delta_d
        strat_g.gamma_dollars += gamma_d
        strat_g.theta_dollars += theta_d
        strat_g.vega_dollars += vega_d
        strat_g.rho_dollars += rho_d

        # Accumulate to portfolio totals
        portfolio.total_delta += delta_d
        portfolio.total_gamma += gamma_d
        portfolio.total_theta += theta_d
        portfolio.total_vega += vega_d
        portfolio.total_rho += rho_d

    return portfolio
```

### 3. Store snapshot in DB

```python
def store_greeks_snapshot(greeks: PortfolioGreeks) -> None:
    """Persist a portfolio Greeks snapshot to portfolio_greeks_history."""
    row = greeks.to_db_row()
    with db_conn() as conn:
        conn.execute("""
            INSERT INTO portfolio_greeks_history
                (snapshot_time, symbol_greeks, strategy_greeks,
                 portfolio_delta, portfolio_gamma, portfolio_theta,
                 portfolio_vega, portfolio_rho)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """, [
            row["snapshot_time"], row["symbol_greeks"], row["strategy_greeks"],
            row["portfolio_delta"], row["portfolio_gamma"], row["portfolio_theta"],
            row["portfolio_vega"], row["portfolio_rho"],
        ])
```

### 4. Integration with execution monitor

In `execution_monitor.py`, add a call in the periodic check cycle (the main monitoring loop that runs every N seconds):

```python
from quantstack.core.options.engine import compute_portfolio_greeks, store_greeks_snapshot

# Inside the periodic check method:
greeks = compute_portfolio_greeks()
store_greeks_snapshot(greeks)
logger.info(
    f"Portfolio Greeks: delta=${greeks.total_delta:,.0f} "
    f"gamma=${greeks.total_gamma:,.0f} theta=${greeks.total_theta:,.0f}/day "
    f"vega=${greeks.total_vega:,.0f}"
)
```

## Tests to Write

File: `tests/unit/options/test_portfolio_greeks.py`

1. **test_empty_portfolio_returns_zeros** -- No open options positions. All totals should be 0.
2. **test_single_long_call_delta_dollars** -- 1 long call, delta=0.50, spot=100, qty=1. `delta_dollars = 0.50 * 100 * 1 * 100 = 5000`.
3. **test_single_short_put_delta_dollars** -- 1 short put, delta=-0.30, qty=-1. Verify sign: short put has positive delta dollars (delta is negative, qty is negative, product is positive).
4. **test_multi_position_aggregation** -- 3 positions across 2 symbols. Verify per-symbol breakdowns sum to portfolio totals.
5. **test_per_strategy_breakdown** -- 2 positions with different `strategy_name`. Verify each strategy's Greeks are isolated.
6. **test_gamma_dollar_conversion** -- gamma=0.05, spot=100, qty=1. `gamma_dollars = 0.5 * 0.05 * 100^2 * 1 * 100 * 0.01 = 250`. Verify.
7. **test_theta_dollar_is_daily** -- theta=-0.05 (per contract per day from BS). `theta_dollars = -0.05 * 1 * 100 = -5.0`. Verify negative (theta decays).
8. **test_to_db_row_json_roundtrip** -- Create PortfolioGreeks, call `to_db_row()`, parse JSONB fields, verify structure matches.
9. **test_store_and_retrieve_snapshot** -- Insert via `store_greeks_snapshot()`, query back, verify all fields match.
10. **test_missing_spot_price_skipped** -- One symbol has no spot price. Verify it is skipped (logged) and other symbols still computed.

## Edge Cases

- **No open options positions**: Return all-zero `PortfolioGreeks`. Do not skip the DB insert -- a zero snapshot is valuable for confirming the system ran and there was no exposure.
- **Expired positions still marked open**: If a position has `expiry < today` but `status = 'open'`, the Greeks computation will use `tte = 1/365` (floor). This is technically wrong -- the position should be closed. Log a warning and flag for the supervisor.
- **Missing IV data**: Fallback to 30% vol. This is a coarse assumption. Log the fallback so it is visible in traces. Consider using the most recent IV from `options_pnl_attribution` as a better fallback.
- **Very large delta exposure**: If `total_delta > $100,000`, this is a signal for the hedging engine (section-06). The snapshot stores this for the hedging engine to query.
- **Concurrent snapshot writes**: Multiple instances writing simultaneously is fine -- each is a separate row with its own `snapshot_time`. No conflict.
- **Sign conventions**: Long positions have positive `qty`, short positions have negative `qty`. Delta for puts is negative. The product `delta * qty` captures the net exposure correctly. Short puts: `delta(-0.3) * qty(-1) = +0.3` (positive delta exposure, correct).
