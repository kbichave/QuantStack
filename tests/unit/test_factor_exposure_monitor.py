"""Unit tests for factor exposure monitoring module."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from quantstack.risk.factor_exposure import (
    FACTOR_CONFIG_DEFAULTS,
    FactorExposureSnapshot,
    check_factor_drift,
    compute_factor_exposure,
    load_factor_config,
    persist_factor_snapshot,
)


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def _make_positions(specs: list[tuple[str, float, str]]) -> list[dict]:
    """Create position dicts from (symbol, market_value, sector) tuples."""
    return [
        {"symbol": s, "quantity": 100, "market_value": mv, "sector": sec}
        for s, mv, sec in specs
    ]


# ---------------------------------------------------------------------------
# Factor config defaults
# ---------------------------------------------------------------------------


class TestFactorConfigDefaults:
    def test_correct_values(self):
        assert FACTOR_CONFIG_DEFAULTS["beta_drift_threshold"] == "0.3"
        assert FACTOR_CONFIG_DEFAULTS["sector_max_pct"] == "40"
        assert FACTOR_CONFIG_DEFAULTS["momentum_crowding_pct"] == "70"
        assert FACTOR_CONFIG_DEFAULTS["benchmark_symbol"] == "SPY"


# ---------------------------------------------------------------------------
# load_factor_config
# ---------------------------------------------------------------------------


class TestLoadFactorConfig:
    def test_loads_from_db(self):
        conn = MagicMock()
        cursor = MagicMock()
        cursor.fetchall.return_value = [
            {"config_key": "beta_drift_threshold", "value": "0.5"},
            {"config_key": "sector_max_pct", "value": "50"},
        ]
        conn.execute.return_value = cursor

        @contextmanager
        def mock_db():
            yield conn

        with patch("quantstack.risk.factor_exposure.db_conn", mock_db):
            config = load_factor_config()

        assert config["beta_drift_threshold"] == "0.5"
        assert config["sector_max_pct"] == "50"

    def test_falls_back_to_defaults_on_error(self):
        @contextmanager
        def failing_db():
            raise Exception("DB down")
            yield  # noqa: unreachable

        with patch("quantstack.risk.factor_exposure.db_conn", failing_db):
            config = load_factor_config()

        assert config == FACTOR_CONFIG_DEFAULTS


# ---------------------------------------------------------------------------
# compute_factor_exposure
# ---------------------------------------------------------------------------


class TestComputeFactorExposure:
    def test_sector_weights_sum_to_one(self):
        positions = _make_positions([
            ("AAPL", 5000.0, "Technology"),
            ("JNJ", 3000.0, "Healthcare"),
            ("XOM", 2000.0, "Energy"),
        ])

        with patch("quantstack.risk.factor_exposure._compute_portfolio_beta", return_value=1.0):
            snapshot = _run(compute_factor_exposure(positions, "SPY"))

        total = sum(snapshot.sector_weights.values())
        assert abs(total - 1.0) < 1e-9

    def test_single_position_portfolio(self):
        positions = _make_positions([("AAPL", 10000.0, "Technology")])

        with patch("quantstack.risk.factor_exposure._compute_portfolio_beta", return_value=1.2):
            snapshot = _run(compute_factor_exposure(positions, "SPY"))

        assert snapshot.sector_weights == {"Technology": 1.0}
        assert snapshot.top_sector == "Technology"
        assert snapshot.top_sector_pct == 100.0

    def test_no_sector_data_falls_back(self):
        positions = [
            {"symbol": "AAPL", "quantity": 100, "market_value": 5000.0},
            {"symbol": "MSFT", "quantity": 50, "market_value": 3000.0},
        ]

        with patch("quantstack.risk.factor_exposure._compute_portfolio_beta", return_value=1.0):
            snapshot = _run(compute_factor_exposure(positions, "SPY"))

        assert "unknown" in snapshot.sector_weights
        assert snapshot.sector_weights["unknown"] == 1.0

    def test_empty_positions(self):
        snapshot = _run(compute_factor_exposure([], "SPY"))
        assert snapshot.portfolio_beta == 0.0
        assert snapshot.sector_weights == {}

    def test_beta_from_known_series(self):
        """Beta should reflect weighted average of position betas."""
        with patch("quantstack.risk.factor_exposure._compute_portfolio_beta", return_value=1.35):
            positions = _make_positions([("AAPL", 10000.0, "Tech")])
            snapshot = _run(compute_factor_exposure(positions, "SPY"))

        assert snapshot.portfolio_beta == 1.35


# ---------------------------------------------------------------------------
# check_factor_drift
# ---------------------------------------------------------------------------


class TestCheckFactorDrift:
    def _make_exposure(self, **overrides):
        defaults = {
            "portfolio_beta": 1.0,
            "sector_weights": {"Technology": 0.3, "Healthcare": 0.7},
            "top_sector": "Healthcare",
            "top_sector_pct": 70.0,
            "style_scores": {},
            "momentum_crowding_pct": 50.0,
            "benchmark_symbol": "SPY",
        }
        defaults.update(overrides)
        return FactorExposureSnapshot(**defaults)

    def test_beta_drift_triggers_alert(self):
        exposure = self._make_exposure(portfolio_beta=1.5, top_sector_pct=30.0)
        config = {"beta_drift_threshold": "0.3", "sector_max_pct": "80", "momentum_crowding_pct": "90"}

        alerts = _run(check_factor_drift(exposure, config))
        assert len(alerts) == 1
        assert alerts[0]["category"] == "factor_drift"
        assert "beta" in alerts[0]["title"].lower()

    def test_beta_drift_critical_severity(self):
        exposure = self._make_exposure(portfolio_beta=1.7, top_sector_pct=30.0)
        config = {"beta_drift_threshold": "0.3", "sector_max_pct": "80", "momentum_crowding_pct": "90"}

        alerts = _run(check_factor_drift(exposure, config))
        assert alerts[0]["severity"] == "critical"  # drift 0.7 >= 2*0.3

    def test_beta_drift_warning_severity(self):
        exposure = self._make_exposure(portfolio_beta=1.4, top_sector_pct=30.0)
        config = {"beta_drift_threshold": "0.3", "sector_max_pct": "80", "momentum_crowding_pct": "90"}

        alerts = _run(check_factor_drift(exposure, config))
        assert alerts[0]["severity"] == "warning"  # drift 0.4 < 2*0.3

    def test_sector_concentration_triggers_alert(self):
        exposure = self._make_exposure(portfolio_beta=1.0, top_sector_pct=55.0)
        config = {"beta_drift_threshold": "0.3", "sector_max_pct": "40", "momentum_crowding_pct": "90"}

        alerts = _run(check_factor_drift(exposure, config))
        sector_alerts = [a for a in alerts if "sector" in a["title"].lower()]
        assert len(sector_alerts) == 1

    def test_momentum_crowding_triggers_alert(self):
        exposure = self._make_exposure(portfolio_beta=1.0, top_sector_pct=30.0, momentum_crowding_pct=80.0)
        config = {"beta_drift_threshold": "0.3", "sector_max_pct": "80", "momentum_crowding_pct": "70"}

        alerts = _run(check_factor_drift(exposure, config))
        mom_alerts = [a for a in alerts if "momentum" in a["title"].lower()]
        assert len(mom_alerts) == 1

    def test_all_within_thresholds_no_alerts(self):
        exposure = self._make_exposure(portfolio_beta=1.1, top_sector_pct=35.0, momentum_crowding_pct=50.0)
        config = {"beta_drift_threshold": "0.3", "sector_max_pct": "40", "momentum_crowding_pct": "70"}

        alerts = _run(check_factor_drift(exposure, config))
        assert alerts == []

    def test_custom_config_changes_thresholds(self):
        exposure = self._make_exposure(portfolio_beta=1.4, top_sector_pct=35.0)
        # Default config would trigger beta alert (drift 0.4 > 0.3)
        default_alerts = _run(check_factor_drift(exposure, FACTOR_CONFIG_DEFAULTS))
        assert len(default_alerts) > 0

        # Custom config with higher threshold
        custom_config = dict(FACTOR_CONFIG_DEFAULTS)
        custom_config["beta_drift_threshold"] = "0.5"
        custom_alerts = _run(check_factor_drift(exposure, custom_config))
        assert len(custom_alerts) == 0


# ---------------------------------------------------------------------------
# persist_factor_snapshot
# ---------------------------------------------------------------------------


class TestPersistFactorSnapshot:
    def test_inserts_row(self):
        conn = MagicMock()
        cursor = MagicMock()
        conn.execute.return_value = cursor

        @contextmanager
        def mock_db():
            yield conn

        snapshot = FactorExposureSnapshot(
            portfolio_beta=1.2,
            sector_weights={"Tech": 0.6, "Health": 0.4},
            style_scores={"momentum": 0.5},
            momentum_crowding_pct=45.0,
            benchmark_symbol="SPY",
            alerts_triggered=1,
        )

        with patch("quantstack.risk.factor_exposure.db_conn", mock_db):
            persist_factor_snapshot(snapshot)

        conn.execute.assert_called_once()
        sql = conn.execute.call_args[0][0]
        assert "factor_exposure_history" in sql
