"""Tests for drift detection pre-cache behavior (section-04).

Verifies that drift severity influences confidence penalty and cache TTL.
"""

import json
from dataclasses import dataclass, field
from datetime import date
from unittest.mock import MagicMock, patch, call

import pytest

from quantstack.signal_engine.brief import SignalBrief


@dataclass
class FakeDriftReport:
    """Minimal stand-in for DriftReport."""

    overall_psi: float = 0.0
    severity: str = "NONE"
    drifted_features: list[str] = field(default_factory=list)


def _make_brief(confidence: float = 0.5) -> SignalBrief:
    return SignalBrief(
        date=date.today(),
        market_overview="test",
        market_bias="neutral",
        risk_environment="normal",
        overall_confidence=confidence,
    )


def _mock_engine_deps():
    """Return patches needed to run SignalEngine.run() without real infra."""
    return {
        "collectors": patch(
            "quantstack.signal_engine.engine.SignalEngine._run_collectors",
            return_value=({}, []),
        ),
        "build": patch(
            "quantstack.signal_engine.engine.SignalEngine._build_brief",
            side_effect=lambda sym, out, fail: _make_brief(0.75),
        ),
        "cache_get": patch(
            "quantstack.signal_engine.cache.get", return_value=None
        ),
        "cache_put": patch("quantstack.signal_engine.cache.put"),
        "db_conn": patch("quantstack.signal_engine.engine.db_conn"),
    }


class TestDriftConfidencePenalty:
    """Drift severity applies the correct confidence penalty."""

    @pytest.mark.asyncio
    async def test_drift_none_no_penalty(self):
        """NONE drift -> no confidence change, default TTL."""
        drift = FakeDriftReport(severity="NONE", overall_psi=0.02)

        with patch("quantstack.signal_engine.engine.DriftDetector") as mock_dd:
            mock_dd.return_value.check_drift_from_brief.return_value = drift
            deps = _mock_engine_deps()
            with deps["collectors"], deps["build"], deps["cache_get"], \
                 deps["cache_put"] as mock_put, deps["db_conn"]:
                from quantstack.signal_engine.engine import SignalEngine
                brief = await SignalEngine().run("AAPL")

        assert brief.overall_confidence == 0.75
        mock_put.assert_called_once()
        assert mock_put.call_args.kwargs.get("ttl") is None

    @pytest.mark.asyncio
    async def test_drift_warning_halves_ttl_and_penalizes(self):
        """WARNING drift -> confidence -0.10, TTL 1800s."""
        drift = FakeDriftReport(severity="WARNING", overall_psi=0.15)

        with patch("quantstack.signal_engine.engine.DriftDetector") as mock_dd:
            mock_dd.return_value.check_drift_from_brief.return_value = drift
            deps = _mock_engine_deps()
            with deps["collectors"], deps["build"], deps["cache_get"], \
                 deps["cache_put"] as mock_put, deps["db_conn"]:
                from quantstack.signal_engine.engine import SignalEngine
                brief = await SignalEngine().run("AAPL")

        assert abs(brief.overall_confidence - 0.65) < 0.01
        assert mock_put.call_args.kwargs.get("ttl") == 1800

    @pytest.mark.asyncio
    async def test_drift_critical_short_ttl_and_heavy_penalty(self):
        """CRITICAL drift -> confidence -0.30, TTL 300s."""
        drift = FakeDriftReport(
            severity="CRITICAL", overall_psi=0.35, drifted_features=["rsi_14"]
        )

        with patch("quantstack.signal_engine.engine.DriftDetector") as mock_dd:
            mock_dd.return_value.check_drift_from_brief.return_value = drift
            deps = _mock_engine_deps()
            with deps["collectors"], deps["build"], deps["cache_get"], \
                 deps["cache_put"] as mock_put, deps["db_conn"]:
                from quantstack.signal_engine.engine import SignalEngine
                brief = await SignalEngine().run("AAPL")

        assert abs(brief.overall_confidence - 0.45) < 0.01
        assert mock_put.call_args.kwargs.get("ttl") == 300

    @pytest.mark.asyncio
    async def test_confidence_penalty_floors_at_zero(self):
        """Confidence does not go below 0.0."""
        drift = FakeDriftReport(
            severity="CRITICAL", overall_psi=0.5, drifted_features=["x"]
        )

        with patch("quantstack.signal_engine.engine.DriftDetector") as mock_dd:
            mock_dd.return_value.check_drift_from_brief.return_value = drift
            deps = _mock_engine_deps()
            build_patch = patch(
                "quantstack.signal_engine.engine.SignalEngine._build_brief",
                side_effect=lambda sym, out, fail: _make_brief(0.15),
            )
            with deps["collectors"], build_patch, deps["cache_get"], \
                 deps["cache_put"], deps["db_conn"]:
                from quantstack.signal_engine.engine import SignalEngine
                brief = await SignalEngine().run("AAPL")

        assert brief.overall_confidence == 0.0


class TestDriftSystemEvent:
    """CRITICAL drift inserts a system_events row."""

    @pytest.mark.asyncio
    async def test_critical_inserts_system_event(self):
        drift = FakeDriftReport(
            severity="CRITICAL", overall_psi=0.4, drifted_features=["vol_20d"]
        )

        with patch("quantstack.signal_engine.engine.DriftDetector") as mock_dd:
            mock_dd.return_value.check_drift_from_brief.return_value = drift
            deps = _mock_engine_deps()
            with deps["collectors"], deps["build"], deps["cache_get"], \
                 deps["cache_put"], deps["db_conn"] as mock_db:
                mock_conn = MagicMock()
                mock_db.return_value.__enter__ = MagicMock(return_value=mock_conn)
                mock_db.return_value.__exit__ = MagicMock(return_value=False)

                from quantstack.signal_engine.engine import SignalEngine
                await SignalEngine().run("AAPL")

            event_calls = [
                c for c in mock_conn.execute.call_args_list
                if "system_events" in str(c)
            ]
            assert len(event_calls) >= 1, "Expected DRIFT_CRITICAL system_events insert"

    @pytest.mark.asyncio
    async def test_warning_no_system_event(self):
        drift = FakeDriftReport(severity="WARNING", overall_psi=0.15)

        with patch("quantstack.signal_engine.engine.DriftDetector") as mock_dd:
            mock_dd.return_value.check_drift_from_brief.return_value = drift
            deps = _mock_engine_deps()
            with deps["collectors"], deps["build"], deps["cache_get"], \
                 deps["cache_put"], deps["db_conn"] as mock_db:
                mock_conn = MagicMock()
                mock_db.return_value.__enter__ = MagicMock(return_value=mock_conn)
                mock_db.return_value.__exit__ = MagicMock(return_value=False)

                from quantstack.signal_engine.engine import SignalEngine
                await SignalEngine().run("AAPL")

            event_calls = [
                c for c in mock_conn.execute.call_args_list
                if "system_events" in str(c)
            ]
            assert len(event_calls) == 0, "WARNING should not insert system_events"
