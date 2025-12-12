# Copyright 2024 QuantCore Contributors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for pysabr adapter."""

import pytest
import numpy as np
import pandas as pd


class TestPySABRAdapter:
    """Test suite for SABR adapter functions."""

    def test_fit_sabr_surface_basic(self):
        """Test basic SABR surface fitting."""
        from quantcore.options.adapters.pysabr_adapter import fit_sabr_surface

        # Create sample IV data with typical smile
        strikes = [90, 95, 100, 105, 110]
        ivs = [0.28, 0.24, 0.22, 0.23, 0.26]  # Smile pattern

        quotes = pd.DataFrame({"strike": strikes, "iv": ivs})

        result = fit_sabr_surface(
            quotes=quotes,
            forward=100.0,
            time_to_expiry=30 / 365,
            beta=1.0,
        )

        assert "params" in result
        assert "fit_quality" in result
        assert result["fit_quality"]["r_squared"] > 0.8

    def test_fit_sabr_params_reasonable(self):
        """Test SABR parameters are in reasonable ranges."""
        from quantcore.options.adapters.pysabr_adapter import fit_sabr_surface

        strikes = [90, 95, 100, 105, 110]
        ivs = [0.28, 0.24, 0.22, 0.23, 0.26]
        quotes = pd.DataFrame({"strike": strikes, "iv": ivs})

        result = fit_sabr_surface(quotes, 100.0, 30 / 365, beta=1.0)

        params = result["params_dict"]

        # Alpha should be positive
        assert params["alpha"] > 0

        # Rho should be in [-1, 1]
        assert -1 <= params["rho"] <= 1

        # Volvol should be positive
        assert params["volvol"] > 0

    def test_interpolate_sabr_vol(self):
        """Test SABR volatility interpolation."""
        from quantcore.options.adapters.pysabr_adapter import (
            fit_sabr_surface,
            interpolate_sabr_vol,
        )

        strikes = [90, 95, 100, 105, 110]
        ivs = [0.28, 0.24, 0.22, 0.23, 0.26]
        quotes = pd.DataFrame({"strike": strikes, "iv": ivs})

        result = fit_sabr_surface(quotes, 100.0, 30 / 365)

        # Interpolate at known strike
        vol_100 = interpolate_sabr_vol(result["params"], 100.0, 100.0, 30 / 365)

        # Should be close to fitted ATM vol
        assert 0.20 < vol_100 < 0.25

    def test_interpolate_extrapolation(self):
        """Test interpolation handles extrapolation."""
        from quantcore.options.adapters.pysabr_adapter import (
            fit_sabr_surface,
            interpolate_sabr_vol,
        )

        strikes = [95, 100, 105]
        ivs = [0.24, 0.22, 0.23]
        quotes = pd.DataFrame({"strike": strikes, "iv": ivs})

        result = fit_sabr_surface(quotes, 100.0, 30 / 365)

        # Extrapolate to strike outside training range
        vol_85 = interpolate_sabr_vol(result["params"], 85.0, 100.0, 30 / 365)

        # Should return a reasonable vol (not crash)
        assert 0.10 < vol_85 < 0.50

    def test_get_sabr_smile(self):
        """Test generating vol smile from SABR params."""
        from quantcore.options.adapters.pysabr_adapter import (
            fit_sabr_surface,
            get_sabr_smile,
        )

        strikes = [90, 95, 100, 105, 110]
        ivs = [0.28, 0.24, 0.22, 0.23, 0.26]
        quotes = pd.DataFrame({"strike": strikes, "iv": ivs})

        result = fit_sabr_surface(quotes, 100.0, 30 / 365)

        # Generate smile
        smile = get_sabr_smile(
            result["params"],
            strikes=[85, 90, 95, 100, 105, 110, 115],
            forward=100.0,
            time_to_expiry=30 / 365,
        )

        assert len(smile) == 7
        assert "strike" in smile.columns
        assert "iv" in smile.columns

    def test_get_sabr_skew_metrics(self):
        """Test skew metrics extraction."""
        from quantcore.options.adapters.pysabr_adapter import (
            fit_sabr_surface,
            get_sabr_skew_metrics,
        )

        # Create data with downside skew (puts more expensive)
        strikes = [90, 95, 100, 105, 110]
        ivs = [0.30, 0.25, 0.22, 0.21, 0.20]  # Negative skew
        quotes = pd.DataFrame({"strike": strikes, "iv": ivs})

        result = fit_sabr_surface(quotes, 100.0, 30 / 365)

        metrics = get_sabr_skew_metrics(result["params"], 100.0, 30 / 365)

        assert "atm_vol" in metrics
        assert "skew_25d" in metrics
        assert "risk_reversal_25d" in metrics

        # With negative skew, puts > calls
        assert metrics["skew_25d"] > 0

    def test_fit_sabr_insufficient_data(self):
        """Test error handling for insufficient data."""
        from quantcore.options.adapters.pysabr_adapter import fit_sabr_surface

        # Only 2 points
        quotes = pd.DataFrame({"strike": [100, 105], "iv": [0.22, 0.23]})

        with pytest.raises(ValueError):
            fit_sabr_surface(quotes, 100.0, 30 / 365)

    def test_fit_sabr_invalid_data(self):
        """Test filtering of invalid data."""
        from quantcore.options.adapters.pysabr_adapter import fit_sabr_surface

        strikes = [90, 95, 100, 105, 110, 115]
        ivs = [0.28, 0.24, 0.22, -0.1, 0.26, 6.0]  # Invalid: negative and >5
        quotes = pd.DataFrame({"strike": strikes, "iv": ivs})

        # Should filter invalid and still fit with valid points
        result = fit_sabr_surface(quotes, 100.0, 30 / 365)

        assert "params" in result

    def test_fit_term_structure(self):
        """Test fitting across multiple expiries."""
        from quantcore.options.adapters.pysabr_adapter import fit_term_structure_sabr

        # Create chain data with multiple expiries
        data = []
        for dte in [14, 30, 60]:
            for strike in [95, 100, 105]:
                iv = 0.22 + 0.01 * abs(strike - 100) / 5  # Simple smile
                data.append({"strike": strike, "iv": iv, "dte": dte})

        chain = pd.DataFrame(data)

        result = fit_term_structure_sabr(
            chain_data=chain,
            spot=100.0,
            risk_free_rate=0.05,
        )

        assert "expiry_fits" in result
        assert len(result["expiries"]) >= 2


class TestPySABRFallback:
    """Test fallback when pysabr not available."""

    def test_scipy_fallback(self):
        """Test scipy-based fitting works as fallback."""
        from quantcore.options.adapters.pysabr_adapter import fit_sabr_surface

        strikes = [90, 95, 100, 105, 110]
        ivs = [0.28, 0.24, 0.22, 0.23, 0.26]
        quotes = pd.DataFrame({"strike": strikes, "iv": ivs})

        # Should work regardless of pysabr availability
        result = fit_sabr_surface(quotes, 100.0, 30 / 365)

        assert "params" in result or "params_dict" in result
