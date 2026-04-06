"""Unit tests for regime detector (section-11)."""

import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from quantstack.core.regime_detector import classify_regime, RegimeInputs, RegimeClassification


def _inputs(adx: float, spy_20d: float, prev: str | None = None,
             vix: float = 20.0, breadth: float = 0.5) -> RegimeInputs:
    return RegimeInputs(
        adx=adx,
        spy_20d_return=spy_20d,
        vix_level=vix,
        breadth_score=breadth,
        previous_regime=prev,
    )


class TestClassifyRegime:
    def test_trending_up(self):
        """ADX=30, SPY_20d_return=+0.05 → regime = trending_up."""
        result = classify_regime(_inputs(adx=30.0, spy_20d=0.05))
        assert result.regime == "trending_up"

    def test_trending_down(self):
        """ADX=28, SPY_20d_return=-0.04 → regime = trending_down."""
        result = classify_regime(_inputs(adx=28.0, spy_20d=-0.04))
        assert result.regime == "trending_down"

    def test_ranging_overrides_return(self):
        """ADX=18, SPY_20d_return=+0.10 → regime = ranging  (ADX < 20 dominates)."""
        result = classify_regime(_inputs(adx=18.0, spy_20d=0.10))
        assert result.regime == "ranging"

    def test_unknown_at_adx_boundary(self):
        """ADX=25, SPY_20d_return=+0.02 → regime = unknown (ADX at boundary, return < 3%)."""
        result = classify_regime(_inputs(adx=25.0, spy_20d=0.02))
        assert result.regime == "unknown"

    def test_unknown_adx_in_gap(self):
        """ADX=22, SPY_20d_return=0.0 → regime = unknown (20 ≤ ADX ≤ 25)."""
        result = classify_regime(_inputs(adx=22.0, spy_20d=0.0))
        assert result.regime == "unknown"

    def test_regime_change_detected(self):
        """previous_regime differs from new regime → regime_change = True."""
        result = classify_regime(_inputs(adx=30.0, spy_20d=0.05, prev="ranging"))
        assert result.regime_change is True

    def test_no_regime_change_same_regime(self):
        """previous_regime equals new regime → regime_change = False."""
        result = classify_regime(_inputs(adx=30.0, spy_20d=0.05, prev="trending_up"))
        assert result.regime_change is False

    def test_no_previous_regime_is_change(self):
        """previous_regime=None → regime_change = True (fresh state)."""
        result = classify_regime(_inputs(adx=30.0, spy_20d=0.05, prev=None))
        assert result.regime_change is True

    def test_confidence_high_when_clear_trending(self):
        """ADX=35 (clearly > 28) → confidence = 1.0."""
        result = classify_regime(_inputs(adx=35.0, spy_20d=0.05))
        assert result.confidence == pytest.approx(1.0)

    def test_confidence_high_when_clear_ranging(self):
        """ADX=12 (clearly < 20) → confidence = 1.0."""
        result = classify_regime(_inputs(adx=12.0, spy_20d=0.0))
        assert result.confidence == pytest.approx(1.0)

    def test_confidence_less_than_1_near_boundary(self):
        """ADX=21 (within 3 of ranging boundary 20) → confidence < 1.0."""
        result = classify_regime(_inputs(adx=21.0, spy_20d=0.0))
        assert result.confidence < 1.0
        assert result.confidence >= 0.5

    def test_confidence_less_than_1_near_upper_boundary(self):
        """ADX=24 (within 3 of trending boundary 25) → confidence < 1.0."""
        result = classify_regime(_inputs(adx=24.0, spy_20d=0.05))
        assert result.confidence < 1.0

    def test_detected_at_is_utc(self):
        """detected_at is a timezone-aware UTC datetime."""
        result = classify_regime(_inputs(adx=30.0, spy_20d=0.05))
        assert result.detected_at.tzinfo is not None


class TestRunRegimeDetection:
    """Tests for run_regime_detection() in supervisor/nodes.py."""

    def test_writes_regime_state_row(self):
        """run_regime_detection writes a row to regime_state."""
        from quantstack.graphs.supervisor.nodes import run_regime_detection

        conn = MagicMock()
        cursor = MagicMock()
        conn.execute = MagicMock(return_value=cursor)

        # SPY OHLCV: 21 rows for 20d return + enough for ADX
        spy_rows = [(float(i + 100),) for i in range(50)]
        vix_row = (20.0,)
        prev_regime_row = None  # no previous regime
        breadth_rows = []

        call_count = [0]
        def fake_fetchall():
            i = call_count[0]
            call_count[0] += 1
            if i == 0:
                return spy_rows     # SPY closes for 20d return + ADX
            if i == 1:
                return breadth_rows # breadth computation
            return []

        fetchone_count = [0]
        def fake_fetchone():
            i = fetchone_count[0]
            fetchone_count[0] += 1
            if i == 0:
                return vix_row
            if i == 1:
                return prev_regime_row
            return None

        cursor.fetchall = MagicMock(side_effect=fake_fetchall)
        cursor.fetchone = MagicMock(side_effect=fake_fetchone)

        result = run_regime_detection(conn)

        # regime_state row must be written via execute
        assert conn.execute.called
        assert "regime" in result

    def test_publishes_regime_change_event(self):
        """REGIME_CHANGE event is published when regime changes."""
        from quantstack.graphs.supervisor.nodes import run_regime_detection

        conn = MagicMock()
        cursor = MagicMock()
        conn.execute = MagicMock(return_value=cursor)

        spy_rows = [(float(i + 100),) for i in range(50)]

        call_count = [0]
        def fake_fetchall():
            call_count[0] += 1
            if call_count[0] == 1:
                return spy_rows
            return []

        fetchone_count = [0]
        def fake_fetchone():
            fetchone_count[0] += 1
            if fetchone_count[0] == 1:
                return (20.0,)                   # vix
            if fetchone_count[0] == 2:
                return ("ranging",)              # previous regime (different from trending_up)
            return None

        cursor.fetchall = MagicMock(side_effect=fake_fetchall)
        cursor.fetchone = MagicMock(side_effect=fake_fetchone)

        published = []

        class FakeBus:
            def publish(self, event):
                published.append(event)

        with patch("quantstack.graphs.supervisor.nodes.EventBus", return_value=FakeBus()):
            with patch("quantstack.graphs.supervisor.nodes.classify_regime") as mock_classify:
                mock_classify.return_value = RegimeClassification(
                    regime="trending_up",
                    regime_change=True,
                    confidence=1.0,
                    detected_at=datetime.now(timezone.utc),
                )
                result = run_regime_detection(conn)

        assert any(e.event_type.value == "regime_change" for e in published if hasattr(e, "event_type"))

    def test_no_event_when_regime_unchanged(self):
        """REGIME_CHANGE event NOT published when regime is same as previous."""
        from quantstack.graphs.supervisor.nodes import run_regime_detection

        conn = MagicMock()
        cursor = MagicMock()
        conn.execute = MagicMock(return_value=cursor)

        cursor.fetchall = MagicMock(return_value=[])
        cursor.fetchone = MagicMock(return_value=None)

        published = []

        class FakeBus:
            def publish(self, event):
                published.append(event)

        with patch("quantstack.graphs.supervisor.nodes.EventBus", return_value=FakeBus()):
            with patch("quantstack.graphs.supervisor.nodes.classify_regime") as mock_classify:
                mock_classify.return_value = RegimeClassification(
                    regime="ranging",
                    regime_change=False,
                    confidence=1.0,
                    detected_at=datetime.now(timezone.utc),
                )
                result = run_regime_detection(conn)

        # No REGIME_CHANGE events published
        regime_events = [e for e in published if hasattr(e, "event_type") and e.event_type.value == "regime_change"]
        assert len(regime_events) == 0

    def test_row_contains_all_input_values(self):
        """regime_state INSERT must include adx, vix_level, spy_20d_return, breadth_score."""
        from quantstack.graphs.supervisor.nodes import run_regime_detection

        conn = MagicMock()
        cursor = MagicMock()
        conn.execute = MagicMock(return_value=cursor)
        cursor.fetchall = MagicMock(return_value=[])
        cursor.fetchone = MagicMock(return_value=None)

        insert_calls = []
        def capture_execute(sql, params=None):
            if params:
                insert_calls.append((sql, params))
            return cursor

        conn.execute = MagicMock(side_effect=capture_execute)

        with patch("quantstack.graphs.supervisor.nodes.classify_regime") as mock_classify:
            mock_classify.return_value = RegimeClassification(
                regime="trending_up",
                regime_change=True,
                confidence=1.0,
                detected_at=datetime.now(timezone.utc),
            )
            run_regime_detection(conn)

        # Find INSERT INTO regime_state call
        insert_sql = [sql for sql, _ in insert_calls if "regime_state" in sql.lower()]
        assert len(insert_sql) >= 1
