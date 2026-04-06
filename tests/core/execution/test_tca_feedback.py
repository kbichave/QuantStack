"""Tests for TCA forecast persistence (section-06).

Tests use a live PostgreSQL connection (skipped if unavailable) to verify
that save_result() correctly persists ac_expected_cost_bps and forecast_error_bps.
"""

from __future__ import annotations

import uuid

import pytest


def _local_pg_reachable() -> bool:
    try:
        import psycopg2
        conn = psycopg2.connect("postgresql://localhost/quantstack", connect_timeout=2)
        conn.close()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _local_pg_reachable(),
    reason="postgresql://localhost/quantstack not reachable",
)


@pytest.fixture(scope="module", autouse=True)
def _setup_pg():
    """Ensure migrations are run and pool uses localhost."""
    import os
    import quantstack.db as _db

    orig = os.environ.get("TRADER_PG_URL")
    os.environ["TRADER_PG_URL"] = "postgresql://localhost/quantstack"
    _db._pg_pool = None
    _db._migrations_done = False
    from quantstack.db import pg_conn, run_migrations_pg
    with pg_conn() as c:
        run_migrations_pg(c)
    yield
    _db._pg_pool = None
    _db._migrations_done = False
    if orig is not None:
        os.environ["TRADER_PG_URL"] = orig
    else:
        os.environ.pop("TRADER_PG_URL", None)


def _unique_trade_id() -> str:
    return f"test_tca_{uuid.uuid4().hex[:8]}"


class TestTCAForecastPersistence:
    """Tests for save_result() with forecast error columns."""

    def test_save_result_with_forecast(self):
        """After save_result() with a forecast, ac_expected_cost_bps and forecast_error_bps are populated."""
        from quantstack.core.execution.tca_engine import ExecAlgo, OrderSide, PreTradeForecast, TradeTCAResult
        from quantstack.core.execution.tca_storage import TCAStore
        from quantstack.db import pg_conn

        trade_id = _unique_trade_id()
        forecast = PreTradeForecast(
            symbol="AAPL", side=OrderSide.BUY, shares=100, arrival_price=150.0,
            spread_cost_bps=2.0, market_impact_bps=5.0, timing_cost_bps=1.0,
            commission_bps=0.5, total_expected_bps=8.5,
            participation_rate=0.05, adv_fraction=0.05, is_liquid=True,
            recommended_algo=ExecAlgo.VWAP, algo_rationale="test",
            min_alpha_bps=17.0,
        )
        result = TradeTCAResult(
            trade_id=trade_id, symbol="AAPL", side=OrderSide.BUY,
            shortfall_vs_arrival_bps=12.0, shortfall_vs_vwap_bps=3.0,
            shortfall_vs_twap_bps=None, shortfall_vs_prev_close_bps=None,
            shortfall_dollar=18.0, is_favorable=False,
        )

        store = TCAStore()
        store.save_result(result, forecast=forecast)

        with pg_conn() as conn:
            row = conn.execute(
                "SELECT ac_expected_cost_bps, forecast_error_bps FROM tca_results WHERE trade_id = %s",
                [trade_id],
            ).fetchone()

        assert row is not None
        assert row[0] == pytest.approx(8.5)  # ac_expected_cost_bps
        assert row[1] == pytest.approx(3.5)  # forecast_error_bps = 12.0 - 8.5

        # Cleanup
        with pg_conn() as conn:
            conn.execute("DELETE FROM tca_results WHERE trade_id = %s", [trade_id])

    def test_forecast_error_sign_convention(self):
        """forecast_error_bps = realized - expected. Positive = worse than forecast."""
        from quantstack.core.execution.tca_engine import ExecAlgo, OrderSide, PreTradeForecast, TradeTCAResult
        from quantstack.core.execution.tca_storage import TCAStore
        from quantstack.db import pg_conn

        trade_id = _unique_trade_id()
        forecast = PreTradeForecast(
            symbol="MSFT", side=OrderSide.BUY, shares=50, arrival_price=300.0,
            spread_cost_bps=1.0, market_impact_bps=3.0, timing_cost_bps=0.5,
            commission_bps=0.5, total_expected_bps=5.0,
            participation_rate=0.03, adv_fraction=0.03, is_liquid=True,
            recommended_algo=ExecAlgo.TWAP, algo_rationale="test",
            min_alpha_bps=10.0,
        )
        # Favorable fill: realized cost 2.0 < expected 5.0 → error = -3.0
        result = TradeTCAResult(
            trade_id=trade_id, symbol="MSFT", side=OrderSide.BUY,
            shortfall_vs_arrival_bps=2.0, shortfall_vs_vwap_bps=None,
            shortfall_vs_twap_bps=None, shortfall_vs_prev_close_bps=None,
            shortfall_dollar=3.0, is_favorable=True,
        )

        store = TCAStore()
        store.save_result(result, forecast=forecast)

        with pg_conn() as conn:
            row = conn.execute(
                "SELECT forecast_error_bps FROM tca_results WHERE trade_id = %s",
                [trade_id],
            ).fetchone()

        assert row is not None
        assert row[0] == pytest.approx(-3.0)  # Negative = better than forecast

        with pg_conn() as conn:
            conn.execute("DELETE FROM tca_results WHERE trade_id = %s", [trade_id])

    def test_save_result_without_forecast_nulls(self):
        """Without a forecast, ac_expected_cost_bps and forecast_error_bps are NULL."""
        from quantstack.core.execution.tca_engine import OrderSide, TradeTCAResult
        from quantstack.core.execution.tca_storage import TCAStore
        from quantstack.db import pg_conn

        trade_id = _unique_trade_id()
        result = TradeTCAResult(
            trade_id=trade_id, symbol="GOOG", side=OrderSide.SELL,
            shortfall_vs_arrival_bps=7.0, shortfall_vs_vwap_bps=None,
            shortfall_vs_twap_bps=None, shortfall_vs_prev_close_bps=None,
            shortfall_dollar=10.5, is_favorable=False,
        )

        store = TCAStore()
        store.save_result(result)  # No forecast

        with pg_conn() as conn:
            row = conn.execute(
                "SELECT ac_expected_cost_bps, forecast_error_bps FROM tca_results WHERE trade_id = %s",
                [trade_id],
            ).fetchone()

        assert row is not None
        assert row[0] is None  # ac_expected_cost_bps
        assert row[1] is None  # forecast_error_bps

        with pg_conn() as conn:
            conn.execute("DELETE FROM tca_results WHERE trade_id = %s", [trade_id])

    def test_save_result_upsert_idempotent(self):
        """Saving the same trade_id twice is an upsert, not a duplicate."""
        from quantstack.core.execution.tca_engine import ExecAlgo, OrderSide, PreTradeForecast, TradeTCAResult
        from quantstack.core.execution.tca_storage import TCAStore
        from quantstack.db import pg_conn

        trade_id = _unique_trade_id()
        forecast = PreTradeForecast(
            symbol="TSLA", side=OrderSide.BUY, shares=20, arrival_price=200.0,
            spread_cost_bps=3.0, market_impact_bps=4.0, timing_cost_bps=1.0,
            commission_bps=0.5, total_expected_bps=8.5,
            participation_rate=1.0, adv_fraction=1.0, is_liquid=False,
            recommended_algo=ExecAlgo.IMMEDIATE, algo_rationale="test",
            min_alpha_bps=17.0,
        )
        result = TradeTCAResult(
            trade_id=trade_id, symbol="TSLA", side=OrderSide.BUY,
            shortfall_vs_arrival_bps=10.0, shortfall_vs_vwap_bps=None,
            shortfall_vs_twap_bps=None, shortfall_vs_prev_close_bps=None,
            shortfall_dollar=4.0, is_favorable=False,
        )

        store = TCAStore()
        store.save_result(result, forecast=forecast)
        store.save_result(result, forecast=forecast)  # Second call — must not raise

        with pg_conn() as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM tca_results WHERE trade_id = %s", [trade_id]
            ).fetchone()[0]
        assert count == 1

        with pg_conn() as conn:
            conn.execute("DELETE FROM tca_results WHERE trade_id = %s", [trade_id])
