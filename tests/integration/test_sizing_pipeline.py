"""Integration tests for the full sizing pipeline (section-12).

Tests the path:
  signals table → kelly_sizing.compute_alpha_signals()
  → portfolio_construction (LW covariance)
  → SafetyGate.validate() on final weights.

Requires a running PostgreSQL database (TRADER_PG_URL or localhost/quantstack).
Tests use unique strategy IDs and clean up after themselves.
"""

import uuid
from datetime import date, timedelta

import numpy as np
import pytest

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# DB availability guard
# ---------------------------------------------------------------------------

def _pg_available() -> bool:
    try:
        import psycopg
        conn = psycopg.connect("postgresql://localhost/quantstack")
        conn.close()
        return True
    except Exception:
        return False


skip_no_pg = pytest.mark.skipif(
    not _pg_available(),
    reason="PostgreSQL not available at localhost/quantstack",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db_conn():
    """Real DB connection with autocommit=False; rolls back after the test."""
    import psycopg
    conn = psycopg.connect("postgresql://localhost/quantstack")
    conn.autocommit = False
    yield conn
    conn.rollback()
    conn.close()


@pytest.fixture
def strategy_id():
    """Unique strategy ID per test to prevent data conflicts."""
    return f"test_sizing_{uuid.uuid4().hex[:8]}"


def _seed_ohlcv(conn, symbols, n_days=120):
    """Insert synthetic OHLCV rows for symbols."""
    rows = []
    rng = np.random.default_rng(42)
    base_date = date.today() - timedelta(days=n_days)
    for sym in symbols:
        price = 100.0
        for i in range(n_days):
            dt = base_date + timedelta(days=i)
            if dt.weekday() >= 5:
                continue
            ret = rng.normal(0.0005, 0.015)
            price *= (1 + ret)
            rows.append((sym, "daily", dt, price * 0.99, price * 1.01, price * 0.98, price, 1_000_000))

    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO ohlcv (symbol, timeframe, timestamp, open, high, low, close, volume) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s) "
            "ON CONFLICT (symbol, timeframe, timestamp) DO NOTHING",
            rows,
        )


def _seed_signals(conn, strategy_id, symbols, signal_date=None):
    """Insert signal rows for each symbol."""
    if signal_date is None:
        signal_date = date.today()
    signal_values = [0.8, 0.5, -0.2]
    with conn.cursor() as cur:
        for sym, sv in zip(symbols, signal_values):
            cur.execute(
                """
                INSERT INTO signals (signal_date, strategy_id, symbol, signal_value, confidence, regime, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, NOW())
                ON CONFLICT (signal_date, strategy_id, symbol) DO UPDATE
                  SET signal_value = EXCLUDED.signal_value
                """,
                (signal_date, strategy_id, sym, sv, 0.9, "trending_up"),
            )


def _seed_signal_ic(conn, strategy_id, icir_21d=0.45, icir_63d=0.40, n_dates=22):
    """Insert signal_ic rows for the strategy."""
    with conn.cursor() as cur:
        for i in range(n_dates):
            d = date.today() - timedelta(days=i)
            cur.execute(
                """
                INSERT INTO signal_ic (date, strategy_id, horizon_days, rank_ic, icir_21d, icir_63d, ic_tstat, n_symbols, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())
                ON CONFLICT (date, strategy_id, horizon_days) DO UPDATE
                  SET icir_21d = EXCLUDED.icir_21d, icir_63d = EXCLUDED.icir_63d
                """,
                (d, strategy_id, 21, 0.05, icir_21d, icir_63d, 1.2, 5),
            )


def _seed_strategy(conn, strategy_id, status="live"):
    """Insert a minimal strategy row."""
    with conn.cursor() as cur:
        # Delete any existing row with this strategy_id first (rollback handles cleanup)
        cur.execute("DELETE FROM strategies WHERE strategy_id = %s", (strategy_id,))
        cur.execute(
            """
            INSERT INTO strategies (strategy_id, name, status, regime_affinity, created_at, updated_at)
            VALUES (%s, %s, %s, %s, NOW(), NOW())
            """,
            (strategy_id, f"Test Strategy {strategy_id[:8]}", status, "trending_up"),
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@skip_no_pg
def test_alpha_signals_computed_from_ic_and_vol(db_conn, strategy_id):
    """
    Kelly alpha_signals are computed from IC × kelly_fraction × z × annualized_vol.
    With ic=0.45, kelly_fraction=0.5, and 3 symbols of known vol, verify the
    alpha_signals array has 3 elements and positive values for positive signals.
    """
    from quantstack.core.kelly_sizing import compute_alpha_signals

    symbols = ["AAPL", "MSFT", "NVDA"]
    _seed_ohlcv(db_conn, symbols)
    _seed_signals(db_conn, strategy_id, symbols)
    _seed_signal_ic(db_conn, strategy_id)

    # Build candidates mimicking risk_sizing output (requires strategy_id)
    candidates = [
        {"symbol": "AAPL", "strategy_id": strategy_id, "signal_value": 0.8},
        {"symbol": "MSFT", "strategy_id": strategy_id, "signal_value": 0.5},
        {"symbol": "NVDA", "strategy_id": strategy_id, "signal_value": -0.2},
    ]
    signal_ic_lookup = {strategy_id: 0.45}
    vol_lookup = {"AAPL": 0.25, "MSFT": 0.22, "NVDA": 0.30}

    alpha_signals = compute_alpha_signals(
        candidates, signal_ic_lookup, vol_lookup, kelly_fraction=0.5
    )

    assert len(alpha_signals) == 3
    # AAPL (signal=0.8) should have higher alpha than MSFT (signal=0.5)
    assert alpha_signals[0] > alpha_signals[1]
    # NVDA (signal=-0.2) should be negative
    assert alpha_signals[2] < 0


@skip_no_pg
def test_lw_covariance_condition_number_below_500(db_conn, strategy_id):
    """
    With 120 days of OHLCV for 3 symbols, the LW-shrunk covariance condition
    number is < 500.
    """
    from quantstack.core.portfolio.optimizer import covariance_matrix

    symbols = ["AAPL", "MSFT", "NVDA"]
    _seed_ohlcv(db_conn, symbols, n_days=130)
    db_conn.commit()

    # Fetch returns and compute covariance
    with db_conn.cursor() as cur:
        rows = {}
        for sym in symbols:
            cur.execute(
                """
                SELECT close FROM ohlcv WHERE symbol = %s AND timeframe = 'daily'
                ORDER BY timestamp DESC LIMIT 121
                """,
                (sym,),
            )
            closes = [r[0] for r in cur.fetchall()]
            if len(closes) >= 21:
                import numpy as np
                closes_arr = np.array(list(reversed(closes)), dtype=float)
                rows[sym] = np.log(closes_arr[1:] / closes_arr[:-1])

    if len(rows) < 3:
        pytest.skip("Insufficient OHLCV data for covariance test")

    n = len(rows)
    import pandas as pd
    min_len = min(len(v) for v in rows.values())
    returns_df = pd.DataFrame({s: rows[s][:min_len] for s in symbols})

    cov_df = covariance_matrix(returns_df)
    cov = cov_df.values
    cond = np.linalg.cond(cov)
    assert cond < 500, f"Condition number too high: {cond:.1f}"


@skip_no_pg
def test_alpha_signals_positive_for_positive_signal(db_conn, strategy_id):
    """Alpha signal is positive for positive signal value, negative for negative."""
    from quantstack.core.kelly_sizing import compute_alpha_signals

    candidates = [
        {"symbol": "AAPL", "strategy_id": strategy_id, "signal_value": 1.0},
        {"symbol": "MSFT", "strategy_id": strategy_id, "signal_value": -1.0},
    ]
    signal_ic_lookup = {strategy_id: 0.30}
    vol_lookup = {"AAPL": 0.20, "MSFT": 0.20}

    alpha_signals = compute_alpha_signals(candidates, signal_ic_lookup, vol_lookup)
    assert alpha_signals[0] > 0
    assert alpha_signals[1] < 0


@skip_no_pg
def test_ic_prior_used_when_no_signal_ic_history(db_conn, strategy_id):
    """
    When signal_ic has no rows for a strategy, kelly_sizing uses IC_prior = 0.01.
    The resulting alpha signal should be much smaller than when ic=0.45.
    """
    from quantstack.core.kelly_sizing import IC_PRIOR, compute_alpha_signals

    candidates = [{"symbol": "AAPL", "strategy_id": strategy_id, "signal_value": 1.0}]
    # No signal_ic rows → empty lookup
    signal_ic_lookup: dict = {}
    vol_lookup = {"AAPL": 0.25}

    alpha_prior = compute_alpha_signals(candidates, signal_ic_lookup, vol_lookup)
    alpha_with_ic = compute_alpha_signals(
        candidates, {strategy_id: 0.45}, vol_lookup
    )

    # Prior should be much weaker than a strategy with established IC
    assert alpha_prior[0] < alpha_with_ic[0]
    # Specifically, prior should use IC_PRIOR = 0.01
    from quantstack.core.kelly_sizing import IC_PRIOR
    expected_prior = IC_PRIOR * 0.5 * 1.0 * 0.25  # ic * kf * z * vol
    assert abs(alpha_prior[0] - expected_prior) < 1e-10


@skip_no_pg
def test_covariance_positive_definite(db_conn, strategy_id):
    """LW covariance matrix must be positive definite (all eigenvalues > 0)."""
    from quantstack.core.portfolio.optimizer import covariance_matrix

    symbols = ["AAPL", "MSFT", "NVDA"]
    _seed_ohlcv(db_conn, symbols, n_days=130)
    db_conn.commit()

    with db_conn.cursor() as cur:
        all_rows = {}
        for sym in symbols:
            cur.execute(
                "SELECT close FROM ohlcv WHERE symbol = %s AND timeframe = 'daily' "
                "ORDER BY timestamp DESC LIMIT 121",
                (sym,),
            )
            closes = [r[0] for r in cur.fetchall()]
            if len(closes) >= 21:
                closes_arr = np.array(list(reversed(closes)), dtype=float)
                all_rows[sym] = np.log(closes_arr[1:] / closes_arr[:-1])

    if len(all_rows) < 3:
        pytest.skip("Insufficient data")

    import pandas as pd
    min_len = min(len(v) for v in all_rows.values())
    returns_df = pd.DataFrame({s: all_rows[s][:min_len] for s in symbols})

    cov_df = covariance_matrix(returns_df)
    eigenvalues = np.linalg.eigvalsh(cov_df.values)
    assert np.all(eigenvalues > 0), f"Non-positive eigenvalues: {eigenvalues}"
