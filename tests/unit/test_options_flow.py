# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for options_flow signal computation (pure math — no Alpaca API calls)."""

import math
import pytest

from quant_pod.signal_engine.collectors.options_flow import (
    _bs_delta,
    _bs_gamma,
    _bs_price,
    _norm_cdf,
    compute_options_flow_signals,
)


# ---------------------------------------------------------------------------
# Black-Scholes helpers
# ---------------------------------------------------------------------------


class TestBlackScholesHelpers:
    def test_norm_cdf_symmetry(self):
        assert abs(_norm_cdf(0.0) - 0.5) < 1e-10
        assert abs(_norm_cdf(1.0) + _norm_cdf(-1.0) - 1.0) < 1e-10

    def test_norm_cdf_extremes(self):
        assert _norm_cdf(10.0) > 0.9999
        assert _norm_cdf(-10.0) < 0.0001

    def test_bs_price_atm_call(self):
        """ATM call with 30 days, 20% vol should be a few % of spot."""
        price = _bs_price(spot=100, strike=100, dte_years=30 / 365, iv=0.20, r=0.05, is_call=True)
        assert 1.0 < price < 10.0

    def test_bs_price_deep_itm_call(self):
        """Deep ITM call ≈ spot - PV(strike)."""
        price = _bs_price(spot=200, strike=100, dte_years=1 / 365, iv=0.20, r=0.05, is_call=True)
        assert price > 95  # approximately spot - strike

    def test_bs_price_zero_dte(self):
        assert _bs_price(spot=100, strike=100, dte_years=0, iv=0.20, r=0.05, is_call=True) == 0.0

    def test_bs_gamma_positive(self):
        g = _bs_gamma(spot=100, strike=100, dte_years=30 / 365, iv=0.20, r=0.05)
        assert g > 0

    def test_bs_gamma_atm_peaks(self):
        """Gamma is highest ATM."""
        g_atm = _bs_gamma(100, 100, 30 / 365, 0.20, 0.05)
        g_otm = _bs_gamma(100, 120, 30 / 365, 0.20, 0.05)
        assert g_atm > g_otm

    def test_bs_delta_call_in_zero_one(self):
        d = _bs_delta(spot=100, strike=100, dte_years=30 / 365, iv=0.20, r=0.05, is_call=True)
        assert 0.0 < d < 1.0

    def test_bs_delta_put_in_neg_one_zero(self):
        d = _bs_delta(spot=100, strike=100, dte_years=30 / 365, iv=0.20, r=0.05, is_call=False)
        assert -1.0 < d < 0.0

    def test_bs_delta_deep_itm_call_near_one(self):
        d = _bs_delta(spot=200, strike=100, dte_years=1 / 365, iv=0.20, r=0.05, is_call=True)
        assert d > 0.99


# ---------------------------------------------------------------------------
# compute_options_flow_signals
# ---------------------------------------------------------------------------


def _make_chain(spot: float, strikes: list[float], iv: float = 0.20, oi: int = 1000) -> list[dict]:
    """Build a synthetic chain with BS Greeks for testing."""
    contracts = []
    for st in strikes:
        for option_type, is_call in [("call", True), ("put", False)]:
            dte = 30 / 365
            gamma = _bs_gamma(spot, st, dte, iv, 0.05)
            delta = _bs_delta(spot, st, dte, iv, 0.05, is_call)
            contracts.append(
                {
                    "option_type": option_type,
                    "strike": st,
                    "open_interest": oi,
                    "implied_volatility": iv,
                    "gamma": gamma,
                    "delta": delta,
                }
            )
    return contracts


class TestComputeOptionsFlowSignals:
    def test_returns_dict(self):
        chain = _make_chain(spot=100, strikes=[95, 100, 105])
        result = compute_options_flow_signals(chain, spot=100)
        assert isinstance(result, dict)

    def test_expected_keys(self):
        chain = _make_chain(spot=100, strikes=[95, 100, 105])
        result = compute_options_flow_signals(chain, spot=100)
        for key in ("gex", "gamma_flip", "above_gamma_flip", "dex", "max_pain", "n_contracts"):
            assert key in result

    def test_n_contracts_correct(self):
        chain = _make_chain(spot=100, strikes=[95, 100, 105])
        result = compute_options_flow_signals(chain, spot=100)
        assert result["n_contracts"] == 6  # 3 strikes × 2 sides

    def test_empty_chain(self):
        result = compute_options_flow_signals([], spot=100)
        assert result["gex"] is None
        assert result["max_pain"] is None

    def test_zero_spot(self):
        chain = _make_chain(spot=100, strikes=[100])
        result = compute_options_flow_signals(chain, spot=0)
        assert result["gex"] is None

    def test_gex_is_numeric(self):
        chain = _make_chain(spot=100, strikes=[90, 95, 100, 105, 110])
        result = compute_options_flow_signals(chain, spot=100)
        assert result["gex"] is not None
        assert isinstance(result["gex"], float)

    def test_max_pain_is_strike(self):
        strikes = [90.0, 95.0, 100.0, 105.0, 110.0]
        chain = _make_chain(spot=100, strikes=strikes)
        result = compute_options_flow_signals(chain, spot=100)
        assert result["max_pain"] in strikes

    def test_above_gamma_flip_binary(self):
        chain = _make_chain(spot=100, strikes=list(range(80, 125, 5)))
        result = compute_options_flow_signals(chain, spot=100)
        if result["above_gamma_flip"] is not None:
            assert result["above_gamma_flip"] in (0, 1)

    def test_call_oi_equals_sum(self):
        chain = _make_chain(spot=100, strikes=[95, 100, 105], oi=500)
        result = compute_options_flow_signals(chain, spot=100)
        assert result["call_oi"] == 1500  # 3 strikes × 500

    def test_vrp_computed_with_realized_vol(self):
        chain = _make_chain(spot=100, strikes=[100], iv=0.25)
        result = compute_options_flow_signals(chain, spot=100, realized_vol_30d=0.18)
        assert result["vrp"] is not None
        # VRP = IV - RV = 0.25 - 0.18 = 0.07 (approx)
        assert abs(result["vrp"] - 0.07) < 0.02

    def test_vrp_none_without_realized_vol(self):
        chain = _make_chain(spot=100, strikes=[100], iv=0.25)
        result = compute_options_flow_signals(chain, spot=100, realized_vol_30d=None)
        assert result["vrp"] is None

    def test_max_pain_symmetric_chain(self):
        """Symmetric call/put OI → max pain should be near ATM."""
        chain = _make_chain(spot=100, strikes=[90, 95, 100, 105, 110], oi=1000)
        result = compute_options_flow_signals(chain, spot=100)
        # With symmetric equal OI, max pain should be close to ATM (100)
        assert abs(result["max_pain"] - 100) <= 10

    def test_no_oi_contracts_filtered(self):
        """Contracts with open_interest=None or 0 should not crash."""
        chain = [
            {"option_type": "call", "strike": 100, "open_interest": None, "gamma": 0.01, "delta": 0.5, "implied_volatility": 0.20},
            {"option_type": "put", "strike": 100, "open_interest": 0, "gamma": 0.01, "delta": -0.5, "implied_volatility": 0.20},
        ]
        result = compute_options_flow_signals(chain, spot=100)
        assert isinstance(result, dict)

    def test_iv_skew_computed(self):
        """Build OTM puts with higher IV than OTM calls → positive skew."""
        spot = 100.0
        dte = 30 / 365
        contracts = [
            # OTM put: delta ≈ -0.25 → strike ≈ 95
            {
                "option_type": "put",
                "strike": 95,
                "open_interest": 500,
                "implied_volatility": 0.30,
                "delta": -0.25,
                "gamma": _bs_gamma(spot, 95, dte, 0.30, 0.05),
            },
            # OTM call: delta ≈ 0.25 → strike ≈ 105
            {
                "option_type": "call",
                "strike": 105,
                "open_interest": 500,
                "implied_volatility": 0.18,
                "delta": 0.25,
                "gamma": _bs_gamma(spot, 105, dte, 0.18, 0.05),
            },
        ]
        result = compute_options_flow_signals(contracts, spot=spot)
        assert result["iv_skew"] is not None
        assert result["iv_skew"] > 0  # put IV > call IV → positive skew
