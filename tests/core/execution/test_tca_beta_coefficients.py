"""Tests for TCA β=0.6 power law and coefficient loading (section-07)."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest


class TestBetaPowerLaw:
    """Verify the market impact formula uses β=0.6 (not 0.5)."""

    def test_impact_uses_beta_06(self):
        """pre_trade_forecast uses η × pr^0.6 × σ_bps, not 0.5 × σ × √pr."""
        from quantstack.core.execution.tca_engine import (
            DEFAULT_BETA,
            DEFAULT_ETA,
            OrderSide,
            pre_trade_forecast,
        )

        # Reset coefficients cache to force defaults
        import quantstack.core.execution.tca_engine as _mod
        _mod._coefficients_loaded = False
        _mod._loaded_coefficients = {}

        # Patch db_conn to avoid DB dependency
        with patch.object(_mod, "db_conn", side_effect=Exception("no DB")):
            _mod._coefficients_loaded = False
            _mod._loaded_coefficients = {}

            forecast = pre_trade_forecast(
                symbol="AAPL",
                side=OrderSide.BUY,
                shares=1000,
                arrival_price=150.0,
                adv=1_000_000,
                daily_volatility_pct=1.5,
            )

        participation_rate = 1000 / 1_000_000
        sigma_bps = 1.5 * 100
        # Expected: DEFAULT_ETA × pr^0.6 × sigma_bps
        expected_impact = DEFAULT_ETA * (participation_rate ** DEFAULT_BETA) * sigma_bps
        assert forecast.market_impact_bps == pytest.approx(expected_impact, rel=0.01)

    def test_impact_different_from_sqrt_law(self):
        """At participation > 1%, β=0.6 gives higher impact than β=0.5."""
        from quantstack.core.execution.tca_engine import (
            DEFAULT_BETA,
            DEFAULT_ETA,
            OrderSide,
            pre_trade_forecast,
        )

        import quantstack.core.execution.tca_engine as _mod
        _mod._coefficients_loaded = False
        _mod._loaded_coefficients = {}

        with patch.object(_mod, "db_conn", side_effect=Exception("no DB")):
            _mod._coefficients_loaded = False
            _mod._loaded_coefficients = {}

            forecast = pre_trade_forecast(
                symbol="TEST",
                side=OrderSide.BUY,
                shares=50_000,  # 5% participation
                arrival_price=100.0,
                adv=1_000_000,
                daily_volatility_pct=2.0,
            )

        pr = 50_000 / 1_000_000
        sigma_bps = 200.0
        # Old formula: 0.5 * sigma * sqrt(pr)
        old_impact = 0.5 * sigma_bps * np.sqrt(pr)
        # New formula: eta * pr^0.6 * sigma_bps
        new_impact = DEFAULT_ETA * (pr ** DEFAULT_BETA) * sigma_bps

        # The impacts should be different (new model)
        assert forecast.market_impact_bps == pytest.approx(new_impact, rel=0.01)
        assert abs(new_impact - old_impact) > 0.1  # Measurably different


class TestCoefficientLoading:
    """Verify coefficient loading from tca_coefficients table."""

    def test_defaults_when_no_db(self):
        """When DB is unavailable, defaults are used."""
        from quantstack.core.execution.tca_engine import (
            DEFAULT_BETA,
            DEFAULT_ETA,
            DEFAULT_GAMMA,
            _get_coefficients_for_adv,
        )

        import quantstack.core.execution.tca_engine as _mod
        _mod._coefficients_loaded = False
        _mod._loaded_coefficients = {}

        with patch.object(_mod, "db_conn", side_effect=Exception("no DB")):
            _mod._coefficients_loaded = False
            eta, gamma, beta = _get_coefficients_for_adv(1_000_000, 100.0)

        assert eta == DEFAULT_ETA
        assert gamma == DEFAULT_GAMMA
        assert beta == DEFAULT_BETA

    def test_loads_large_cap_coefficients(self):
        """Large cap symbol uses large_cap coefficients from DB."""
        from quantstack.core.execution.tca_engine import _get_coefficients_for_adv

        import quantstack.core.execution.tca_engine as _mod
        _mod._coefficients_loaded = True
        _mod._loaded_coefficients = {
            "large_cap": (0.18, 0.35, 0.58),
            "small_cap": (0.22, 0.40, 0.62),
        }

        # ADV $20M > threshold → large_cap
        eta, gamma, beta = _get_coefficients_for_adv(200_000, 100.0)
        assert (eta, gamma, beta) == (0.18, 0.35, 0.58)

    def test_loads_small_cap_coefficients(self):
        """Small cap symbol uses small_cap coefficients from DB."""
        from quantstack.core.execution.tca_engine import _get_coefficients_for_adv

        import quantstack.core.execution.tca_engine as _mod
        _mod._coefficients_loaded = True
        _mod._loaded_coefficients = {
            "large_cap": (0.18, 0.35, 0.58),
            "small_cap": (0.22, 0.40, 0.62),
        }

        # ADV $5M < threshold → small_cap
        eta, gamma, beta = _get_coefficients_for_adv(50_000, 100.0)
        assert (eta, gamma, beta) == (0.22, 0.40, 0.62)

    def test_falls_back_to_market_wide(self):
        """When specific group missing, falls back to market_wide."""
        from quantstack.core.execution.tca_engine import _get_coefficients_for_adv

        import quantstack.core.execution.tca_engine as _mod
        _mod._coefficients_loaded = True
        _mod._loaded_coefficients = {
            "market_wide": (0.15, 0.30, 0.60),
        }

        eta, gamma, beta = _get_coefficients_for_adv(200_000, 100.0)
        assert (eta, gamma, beta) == (0.15, 0.30, 0.60)

    def test_falls_back_to_defaults_when_empty(self):
        """Empty loaded coefficients → module defaults."""
        from quantstack.core.execution.tca_engine import (
            DEFAULT_BETA,
            DEFAULT_ETA,
            DEFAULT_GAMMA,
            _get_coefficients_for_adv,
        )

        import quantstack.core.execution.tca_engine as _mod
        _mod._coefficients_loaded = True
        _mod._loaded_coefficients = {}

        eta, gamma, beta = _get_coefficients_for_adv(200_000, 100.0)
        assert eta == DEFAULT_ETA
        assert gamma == DEFAULT_GAMMA
        assert beta == DEFAULT_BETA

    def test_malformed_db_rows_fallback(self):
        """If DB returns None values, _load falls back to empty dict."""
        from quantstack.core.execution.tca_engine import _load_coefficients_from_db

        class FakeCursor:
            def fetchall(self):
                return [("large_cap", None, 0.35, 0.6)]  # eta is None

        class FakeConn:
            def execute(self, sql):
                return FakeCursor()

        class FakeCtx:
            def __enter__(self):
                return FakeConn()
            def __exit__(self, *a):
                pass

        import quantstack.core.execution.tca_engine as _mod
        with patch.object(_mod, "db_conn", return_value=FakeCtx()):
            result = _load_coefficients_from_db()

        # Row with None eta should be skipped
        assert "large_cap" not in result
