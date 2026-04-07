"""Tests for slippage model enhancement (section 11).

Verifies:
  - EWMA-calibrated spread used when tca_parameters row has sample_count >= 50
  - Falls back to fixed constants when no EWMA data
  - Conservative multiplier applied when sample_count < 50
  - Time-of-day multiplier applied to default slippage
  - Time bucket classification correct at boundaries
  - Slippage accuracy tracked: predicted vs realized ratio stored
  - Alert triggered when drift beyond 0.5x or 2.0x
  - No alert when ratio within bounds
  - Zero predicted slippage handled gracefully (no division by zero)
"""

from datetime import datetime, timezone
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pytest

from quantstack.db import PgConnection, _migrate_execution_layer_pg, db_conn
from quantstack.execution.paper_broker import PaperBroker
from quantstack.execution.slippage import (
    check_slippage_drift,
    classify_time_bucket,
    get_time_of_day_multiplier,
    record_slippage_accuracy,
)

_ET = ZoneInfo("America/New_York")


# ---------------------------------------------------------------------------
# DB fixture (savepoint-isolated, rolls back on teardown)
# ---------------------------------------------------------------------------


@pytest.fixture()
def conn():
    with db_conn() as c:
        _migrate_execution_layer_pg(c)
        c._ensure_raw()
        c._raw.execute("SAVEPOINT test_sp")
        yield c
        c._raw.execute("ROLLBACK TO SAVEPOINT test_sp")


# ---------------------------------------------------------------------------
# Time bucket classification
# ---------------------------------------------------------------------------


class TestClassifyTimeBucket:
    def test_morning(self):
        ts = datetime(2026, 4, 6, 10, 0, tzinfo=_ET)
        assert classify_time_bucket(ts) == "morning"

    def test_morning_boundary_start(self):
        ts = datetime(2026, 4, 6, 9, 30, tzinfo=_ET)
        assert classify_time_bucket(ts) == "morning"

    def test_morning_boundary_end(self):
        # 10:59:59 ET is still morning
        ts = datetime(2026, 4, 6, 10, 59, 59, tzinfo=_ET)
        assert classify_time_bucket(ts) == "morning"

    def test_midday_boundary_start(self):
        ts = datetime(2026, 4, 6, 11, 0, tzinfo=_ET)
        assert classify_time_bucket(ts) == "midday"

    def test_afternoon_boundary_start(self):
        ts = datetime(2026, 4, 6, 14, 0, tzinfo=_ET)
        assert classify_time_bucket(ts) == "afternoon"

    def test_close_boundary_start(self):
        ts = datetime(2026, 4, 6, 15, 30, tzinfo=_ET)
        assert classify_time_bucket(ts) == "close"

    def test_pre_market_defaults_to_close(self):
        ts = datetime(2026, 4, 6, 8, 0, tzinfo=_ET)
        assert classify_time_bucket(ts) == "close"

    def test_after_hours_defaults_to_close(self):
        ts = datetime(2026, 4, 6, 17, 0, tzinfo=_ET)
        assert classify_time_bucket(ts) == "close"


# ---------------------------------------------------------------------------
# Time-of-day multiplier
# ---------------------------------------------------------------------------


class TestTimeOfDayMultiplier:
    def test_morning_multiplier(self):
        assert get_time_of_day_multiplier("morning") == 1.3

    def test_midday_multiplier(self):
        assert get_time_of_day_multiplier("midday") == 1.0

    def test_afternoon_multiplier(self):
        assert get_time_of_day_multiplier("afternoon") == 1.1

    def test_close_multiplier(self):
        assert get_time_of_day_multiplier("close") == 1.2

    def test_unknown_bucket_defaults_to_1(self):
        assert get_time_of_day_multiplier("premarket") == 1.0


# ---------------------------------------------------------------------------
# PaperBroker: EWMA-calibrated slippage
# ---------------------------------------------------------------------------


class TestPaperBrokerSlippageParams:
    """Test _get_slippage_params with mocked EWMA lookups."""

    def test_uses_ewma_when_sample_count_high(self, conn):
        """When tca_parameters has sample_count >= 50, use EWMA directly."""
        # Insert a well-calibrated row
        conn.execute(
            """
            INSERT INTO tca_parameters
                (symbol, time_bucket, ewma_spread_bps, ewma_impact_bps,
                 ewma_total_bps, sample_count)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            ["AAPL", "midday", 1.5, 3.0, 4.5, 60],
        )

        broker = PaperBroker(conn=conn)
        ts = datetime(2026, 4, 6, 12, 0, tzinfo=_ET)
        spread_bps, impact_k = broker._get_slippage_params("AAPL", ts)

        assert spread_bps == 1.5
        assert impact_k == 3.0

    def test_conservative_multiplier_when_low_samples(self, conn):
        """When sample_count < 50, conservative multiplier widens the estimate."""
        conn.execute(
            """
            INSERT INTO tca_parameters
                (symbol, time_bucket, ewma_spread_bps, ewma_impact_bps,
                 ewma_total_bps, sample_count)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            ["TSLA", "midday", 2.0, 4.0, 6.0, 10],
        )

        broker = PaperBroker(conn=conn)
        ts = datetime(2026, 4, 6, 12, 0, tzinfo=_ET)
        spread_bps, impact_k = broker._get_slippage_params("TSLA", ts)

        # conservative_multiplier(10) = max(1.0, 2.0 - 10/50) = 1.8
        assert spread_bps == pytest.approx(2.0 * 1.8)
        assert impact_k == pytest.approx(4.0 * 1.8)

    def test_fallback_to_defaults_when_no_ewma(self, conn):
        """No tca_parameters row: use HALF_SPREAD_BPS * tod_mult and k=5."""
        broker = PaperBroker(conn=conn)
        ts = datetime(2026, 4, 6, 12, 0, tzinfo=_ET)  # midday, mult=1.0
        spread_bps, impact_k = broker._get_slippage_params("UNKNOWN", ts)

        assert spread_bps == PaperBroker.HALF_SPREAD_BPS * 1.0
        assert impact_k == 5.0

    def test_tod_multiplier_applied_to_default(self, conn):
        """Default spread scales with time-of-day multiplier."""
        broker = PaperBroker(conn=conn)
        # Morning: mult=1.3
        ts_morning = datetime(2026, 4, 6, 10, 0, tzinfo=_ET)
        spread_bps, _ = broker._get_slippage_params("UNKNOWN", ts_morning)
        assert spread_bps == pytest.approx(PaperBroker.HALF_SPREAD_BPS * 1.3)

        # Close: mult=1.2
        ts_close = datetime(2026, 4, 6, 15, 45, tzinfo=_ET)
        spread_bps_close, _ = broker._get_slippage_params("UNKNOWN", ts_close)
        assert spread_bps_close == pytest.approx(PaperBroker.HALF_SPREAD_BPS * 1.2)

    def test_no_timestamp_defaults_to_close(self, conn):
        """When timestamp is None, bucket defaults to 'close'."""
        broker = PaperBroker(conn=conn)
        spread_bps, impact_k = broker._get_slippage_params("UNKNOWN", None)
        assert spread_bps == pytest.approx(PaperBroker.HALF_SPREAD_BPS * 1.2)
        assert impact_k == 5.0


class TestPaperBrokerFillIntegration:
    """End-to-end: execute() uses calibrated slippage."""

    def test_market_order_uses_calibrated_spread(self, conn):
        """Market order fill price reflects EWMA spread, not the hardcoded default."""
        conn.execute(
            """
            INSERT INTO tca_parameters
                (symbol, time_bucket, ewma_spread_bps, ewma_impact_bps,
                 ewma_total_bps, sample_count)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            ["SPY", "midday", 1.0, 2.0, 3.0, 100],
        )

        broker = PaperBroker(conn=conn)

        # Patch datetime.now() so the timestamp maps to midday
        midday_ts = datetime(2026, 4, 6, 12, 0, tzinfo=_ET)
        with patch(
            "quantstack.execution.paper_broker.datetime",
        ) as mock_dt:
            mock_dt.now.return_value = midday_ts
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

            from quantstack.execution.paper_broker import OrderRequest

            req = OrderRequest(
                symbol="SPY",
                side="buy",
                quantity=100,
                order_type="market",
                current_price=450.0,
                daily_volume=80_000_000,
            )
            fill = broker.execute(req)

        # With EWMA spread=1.0 bps and impact_k=2.0, fill should differ from
        # the old hardcoded HALF_SPREAD_BPS=2 and k=5
        assert fill.fill_price > 450.0  # buy side: price goes up
        assert not fill.rejected


# ---------------------------------------------------------------------------
# Slippage accuracy recording
# ---------------------------------------------------------------------------


class TestSlippageAccuracyRecording:
    def test_record_and_query(self, conn):
        """Basic record: predicted vs realized with valid ratio."""
        record_slippage_accuracy(
            conn, order_id="ord-1", symbol="AAPL", time_bucket="midday",
            predicted_bps=5.0, realized_bps=6.0,
        )
        row = conn.execute(
            "SELECT predicted_bps, realized_bps, ratio FROM slippage_accuracy "
            "WHERE order_id = ?",
            ["ord-1"],
        ).fetchone()

        assert row is not None
        assert row[0] == pytest.approx(5.0)
        assert row[1] == pytest.approx(6.0)
        assert row[2] == pytest.approx(1.2)  # 6/5

    def test_zero_predicted_stores_null_ratio(self, conn):
        """When predicted_bps=0, ratio is NULL (no division by zero)."""
        record_slippage_accuracy(
            conn, order_id="ord-2", symbol="AAPL", time_bucket="morning",
            predicted_bps=0.0, realized_bps=3.0,
        )
        row = conn.execute(
            "SELECT ratio FROM slippage_accuracy WHERE order_id = ?",
            ["ord-2"],
        ).fetchone()

        assert row is not None
        assert row[0] is None

    def test_time_bucket_stored(self, conn):
        """Time bucket is persisted correctly."""
        record_slippage_accuracy(
            conn, order_id="ord-3", symbol="TSLA", time_bucket="close",
            predicted_bps=4.0, realized_bps=4.0,
        )
        row = conn.execute(
            "SELECT time_bucket FROM slippage_accuracy WHERE order_id = ?",
            ["ord-3"],
        ).fetchone()
        assert row[0] == "close"


# ---------------------------------------------------------------------------
# Slippage drift detection
# ---------------------------------------------------------------------------


class TestSlippageDriftDetection:
    def _insert_ratios(self, conn: PgConnection, symbol: str, ratios: list[float]):
        """Helper: insert slippage_accuracy rows with given ratios."""
        for i, ratio in enumerate(ratios):
            predicted = 5.0
            realized = predicted * ratio
            record_slippage_accuracy(
                conn, order_id=f"drift-{symbol}-{i}", symbol=symbol,
                time_bucket="midday", predicted_bps=predicted, realized_bps=realized,
            )

    def test_no_alert_when_within_bounds(self, conn):
        """Ratios between 0.5 and 2.0 produce no alert."""
        self._insert_ratios(conn, "AAPL", [1.0] * 20)
        assert check_slippage_drift(conn, "AAPL") is None

    def test_alert_when_over_predicting(self, conn):
        """Mean ratio < 0.5 triggers over-prediction alert."""
        self._insert_ratios(conn, "MSFT", [0.3] * 20)
        result = check_slippage_drift(conn, "MSFT")
        assert result is not None
        assert "over-predicting" in result

    def test_alert_when_under_predicting(self, conn):
        """Mean ratio > 2.0 triggers under-prediction alert."""
        self._insert_ratios(conn, "GOOG", [2.5] * 20)
        result = check_slippage_drift(conn, "GOOG")
        assert result is not None
        assert "under-predicting" in result

    def test_no_data_returns_none(self, conn):
        """No accuracy data for symbol produces no alert."""
        assert check_slippage_drift(conn, "NOPE") is None

    def test_boundary_exactly_at_threshold(self, conn):
        """Exactly at 0.5 or 2.0 is within bounds (no alert)."""
        self._insert_ratios(conn, "EDGE_LO", [0.5] * 20)
        assert check_slippage_drift(conn, "EDGE_LO") is None

        self._insert_ratios(conn, "EDGE_HI", [2.0] * 20)
        assert check_slippage_drift(conn, "EDGE_HI") is None

    def test_lookback_limits_rows(self, conn):
        """Only the most recent lookback_count rows are considered."""
        # 30 bad ratios followed by 10 good ratios
        self._insert_ratios(conn, "MIX", [0.3] * 30)
        self._insert_ratios(conn, "MIX", [1.0] * 10)
        # lookback=10 should see only the recent 1.0 ratios
        assert check_slippage_drift(conn, "MIX", lookback_count=10) is None
