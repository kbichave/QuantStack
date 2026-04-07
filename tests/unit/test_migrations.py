"""
Integration tests for DB schema migrations (section-01).

These tests require a live PostgreSQL connection. They are skipped automatically
when TRADER_PG_URL is not set or the DB is unreachable.

All tests rely on idempotency — running migrations twice must not raise.
"""

import os

import pytest

def _local_pg_reachable() -> bool:
    try:
        import psycopg
        conn = psycopg.connect("postgresql://localhost/quantstack", connect_timeout=2)
        conn.close()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _local_pg_reachable(),
    reason="postgresql://localhost/quantstack not reachable — skipping DB migration tests",
)


@pytest.fixture(scope="module")
def conn():
    """Live PgConnection using localhost (avoids Docker host.docker.internal in TRADER_PG_URL)."""
    import os
    from quantstack.db import PgConnection, _get_pg_pool, run_migrations_pg

    # Override the pool DSN to use localhost directly.
    orig = os.environ.get("TRADER_PG_URL")
    os.environ["TRADER_PG_URL"] = "postgresql://localhost/quantstack"
    # Reset the pool so it picks up the new URL.
    import quantstack.db as _db
    _db._pg_pool = None
    _db._migrations_done = False
    try:
        from quantstack.db import pg_conn
        with pg_conn() as c:
            run_migrations_pg(c)
            yield c
    finally:
        # Restore original env
        _db._pg_pool = None
        _db._migrations_done = False
        if orig is not None:
            os.environ["TRADER_PG_URL"] = orig
        else:
            os.environ.pop("TRADER_PG_URL", None)


def test_all_five_tables_created(conn):
    """After run_migrations_pg(), all five new schema elements exist."""
    tables = {"signals", "signal_ic", "pnl_attribution", "regime_state", "system_state"}
    rows = conn.execute(
        "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
    ).fetchall()
    existing = {r[0] for r in rows}
    for t in tables:
        assert t in existing, f"Table '{t}' was not created by migrations"


def test_signals_table_idempotent(conn):
    """run_migrations_pg() called a second time does not raise."""
    from quantstack.db import run_migrations_pg

    # The module-level _migrations_done flag means this is a no-op — still must not raise.
    run_migrations_pg(conn)


def test_signals_pk_is_date_strategy_symbol(conn):
    """signals table primary key is (signal_date, strategy_id, symbol)."""
    rows = conn.execute("""
        SELECT kcu.column_name
        FROM information_schema.table_constraints tc
        JOIN information_schema.key_column_usage kcu
            ON tc.constraint_name = kcu.constraint_name
        WHERE tc.table_name = 'signals'
          AND tc.constraint_type = 'PRIMARY KEY'
        ORDER BY kcu.ordinal_position
    """).fetchall()
    pk_cols = [r[0] for r in rows]
    assert pk_cols == ["signal_date", "strategy_id", "symbol"], (
        f"signals PK columns: {pk_cols}"
    )


def test_signal_ic_pk_has_no_symbol_column(conn):
    """signal_ic primary key is (date, strategy_id, horizon_days) — no symbol column."""
    rows = conn.execute("""
        SELECT kcu.column_name
        FROM information_schema.table_constraints tc
        JOIN information_schema.key_column_usage kcu
            ON tc.constraint_name = kcu.constraint_name
        WHERE tc.table_name = 'signal_ic'
          AND tc.constraint_type = 'PRIMARY KEY'
        ORDER BY kcu.ordinal_position
    """).fetchall()
    pk_cols = [r[0] for r in rows]
    assert "symbol" not in pk_cols, (
        f"signal_ic PK must not contain 'symbol', got: {pk_cols}"
    )
    assert pk_cols == ["date", "strategy_id", "horizon_days"], (
        f"signal_ic PK columns: {pk_cols}"
    )


def test_pnl_attribution_conflict_do_nothing(conn):
    """Duplicate insert into pnl_attribution with same (date, symbol, strategy_id) is a no-op."""
    from datetime import date

    test_date = date(2020, 1, 2)
    conn.execute(
        "DELETE FROM pnl_attribution WHERE date = %s AND symbol = 'TEST_MIGR' AND strategy_id = 'TEST_MIGR'",
        (test_date,),
    )
    # First insert
    conn.execute(
        """
        INSERT INTO pnl_attribution (date, symbol, strategy_id, total_pnl, market_pnl, sector_pnl, alpha_pnl, residual_pnl)
        VALUES (%s, 'TEST_MIGR', 'TEST_MIGR', 100.0, 50.0, 30.0, 15.0, 5.0)
        ON CONFLICT (date, symbol, strategy_id) DO NOTHING
        """,
        (test_date,),
    )
    # Second insert — must not raise or update
    conn.execute(
        """
        INSERT INTO pnl_attribution (date, symbol, strategy_id, total_pnl, market_pnl, sector_pnl, alpha_pnl, residual_pnl)
        VALUES (%s, 'TEST_MIGR', 'TEST_MIGR', 999.0, 0.0, 0.0, 0.0, 0.0)
        ON CONFLICT (date, symbol, strategy_id) DO NOTHING
        """,
        (test_date,),
    )
    row = conn.execute(
        "SELECT total_pnl FROM pnl_attribution WHERE date = %s AND symbol = 'TEST_MIGR' AND strategy_id = 'TEST_MIGR'",
        (test_date,),
    ).fetchone()
    assert row is not None
    assert float(row[0]) == 100.0, "Second insert should be a no-op; first value must be preserved"
    # Clean up
    conn.execute(
        "DELETE FROM pnl_attribution WHERE date = %s AND symbol = 'TEST_MIGR' AND strategy_id = 'TEST_MIGR'",
        (test_date,),
    )


def test_regime_state_current_row_query(conn):
    """regime_state can be queried with ORDER BY detected_at DESC LIMIT 1 without error."""
    # Must not raise; result may be None if table is empty.
    row = conn.execute(
        "SELECT regime FROM regime_state ORDER BY detected_at DESC LIMIT 1"
    ).fetchone()
    # row is None or a tuple — either is valid
    assert row is None or isinstance(row[0], str)


def test_system_state_risk_free_rate_roundtrip(conn):
    """risk_free_rate_daily row exists in system_state after migration and has a numeric value."""
    row = conn.execute(
        "SELECT value FROM system_state WHERE key = 'risk_free_rate_daily'"
    ).fetchone()
    assert row is not None, "risk_free_rate_daily row missing from system_state"
    assert float(row[0]) > 0, f"risk_free_rate_daily should be positive, got: {row[0]}"


# ---------------------------------------------------------------------------
# Institutional gap migration tests (section-01)
# ---------------------------------------------------------------------------


def _table_exists(conn, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM pg_tables WHERE schemaname = 'public' AND tablename = %s",
        (table_name,),
    ).fetchone()
    return row is not None


def _column_exists(conn, table_name: str, column_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM information_schema.columns WHERE table_name = %s AND column_name = %s",
        (table_name, column_name),
    ).fetchone()
    return row is not None


def _get_columns(conn, table_name: str) -> set:
    rows = conn.execute(
        "SELECT column_name FROM information_schema.columns WHERE table_name = %s",
        (table_name,),
    ).fetchall()
    return {r[0] for r in rows}


def test_strategies_has_ic_gate_grandfathered_until(conn):
    """M1: strategies table has ic_gate_grandfathered_until column."""
    assert _column_exists(conn, "strategies", "ic_gate_grandfathered_until")


def test_tca_results_has_forecast_columns(conn):
    """M2: tca_results has ac_expected_cost_bps and forecast_error_bps columns."""
    assert _column_exists(conn, "tca_results", "ac_expected_cost_bps")
    assert _column_exists(conn, "tca_results", "forecast_error_bps")


def test_tca_coefficients_table_exists(conn):
    """M3: tca_coefficients table exists with correct columns."""
    assert _table_exists(conn, "tca_coefficients")
    cols = _get_columns(conn, "tca_coefficients")
    expected = {"updated_at", "symbol_group", "eta", "gamma", "beta", "n_trades_in_fit", "r_squared"}
    assert expected <= cols, f"Missing columns: {expected - cols}"


def test_symbol_execution_quality_table_exists(conn):
    """M4: symbol_execution_quality table exists with correct columns."""
    assert _table_exists(conn, "symbol_execution_quality")
    cols = _get_columns(conn, "symbol_execution_quality")
    expected = {"symbol", "week_ending", "mean_abs_error_bps", "quality_scalar", "n_trades"}
    assert expected <= cols, f"Missing columns: {expected - cols}"


def test_strategy_mmc_table_exists(conn):
    """M5: strategy_mmc table exists with correct columns."""
    assert _table_exists(conn, "strategy_mmc")
    cols = _get_columns(conn, "strategy_mmc")
    expected = {
        "date", "strategy_id", "mmc_score", "signal_correlation_to_portfolio",
        "capital_weight_scalar", "n_days_in_window", "computed_at",
    }
    assert expected <= cols, f"Missing columns: {expected - cols}"


def test_alt_data_ic_table_exists(conn):
    """M6: alt_data_ic table exists with correct columns."""
    assert _table_exists(conn, "alt_data_ic")
    cols = _get_columns(conn, "alt_data_ic")
    expected = {"date", "signal_source", "symbol", "rank_ic", "icir_21d", "n_observations"}
    assert expected <= cols, f"Missing columns: {expected - cols}"


def test_institutional_gaps_migration_idempotent(conn):
    """Running _migrate_institutional_gaps_pg twice does not raise."""
    from quantstack.db import _migrate_institutional_gaps_pg

    _migrate_institutional_gaps_pg(conn)
    _migrate_institutional_gaps_pg(conn)


def test_grandfathered_live_strategies(conn):
    """M1: live strategies get ic_gate_grandfathered_until set to ~90 days in future."""
    from datetime import datetime, timedelta, timezone

    test_id = "_test_grandfather_live_001"
    # Clean up any leftover from previous test runs
    conn.execute("DELETE FROM strategies WHERE strategy_id = %s", (test_id,))
    # Insert a live strategy with NULL grandfathered_until
    conn.execute(
        """
        INSERT INTO strategies (strategy_id, name, status, parameters, entry_rules, exit_rules)
        VALUES (%s, %s, 'live', '{}', '{}', '{}')
        """,
        (test_id, test_id),
    )
    # Run the migration again — should set the grandfathered date
    from quantstack.db import _migrate_institutional_gaps_pg
    _migrate_institutional_gaps_pg(conn)

    row = conn.execute(
        "SELECT ic_gate_grandfathered_until FROM strategies WHERE strategy_id = %s",
        (test_id,),
    ).fetchone()
    assert row is not None and row[0] is not None, "grandfathered_until should be set for live strategies"
    # Should be roughly 90 days in the future (allow 1 day margin)
    now = datetime.now(timezone.utc)
    delta = row[0] - now
    assert timedelta(days=88) < delta < timedelta(days=92), f"Expected ~90 days in future, got {delta}"
    # Clean up
    conn.execute("DELETE FROM strategies WHERE strategy_id = %s", (test_id,))
