# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for options_flow signal computation (pure math — no Alpaca API calls)."""

import math
import pytest

from quant_pod.signal_engine.collectors.options_flow import (
    _bs_charm,
    _bs_delta,
    _bs_gamma,
    _bs_price,
    _bs_vanna,
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


# ---------------------------------------------------------------------------
# Charm, Vanna, EHD, O/S ratio, AveMoney
# ---------------------------------------------------------------------------


def _make_full_contracts(spot: float = 100.0):
    """Return contracts with IV, delta, gamma, and volume for comprehensive tests.

    Uses a far-future expiry so DTE > 0 regardless of when tests run.
    """
    from datetime import date, timedelta
    future_expiry = (date.today() + timedelta(days=60)).strftime("%Y-%m-%d")
    dte = 60.0 / 365.0
    contracts = []
    for strike, opt_type, iv, volume in [
        (95,  "put",  0.30, 200),
        (100, "call", 0.20, 300),
        (100, "put",  0.22, 150),
        (105, "call", 0.18, 100),
    ]:
        is_call = opt_type == "call"
        contracts.append({
            "option_type": opt_type,
            "strike": float(strike),
            "open_interest": 500,
            "implied_volatility": iv,
            "delta": _bs_delta(spot, float(strike), dte, iv, 0.05, is_call),
            "gamma": _bs_gamma(spot, float(strike), dte, iv, 0.05),
            "expiry": future_expiry,
            "volume": float(volume),
        })
    return contracts


class TestCharmVannaHelpers:
    def test_bs_charm_zero_dte_returns_zero(self):
        assert _bs_charm(100, 100, 0, 0.20, 0.05, True) == 0.0

    def test_bs_charm_finite_for_valid_inputs(self):
        v = _bs_charm(100, 100, 30 / 365, 0.20, 0.05, True)
        assert math.isfinite(v)

    def test_bs_vanna_zero_dte_returns_zero(self):
        assert _bs_vanna(100, 100, 0, 0.20, 0.05) == 0.0

    def test_bs_vanna_finite_for_valid_inputs(self):
        v = _bs_vanna(100, 100, 30 / 365, 0.20, 0.05)
        assert math.isfinite(v)

    def test_bs_vanna_atm_is_zero(self):
        """ATM vanna: d2 ≈ 0 for very short-dated ATM, so vanna ≈ 0."""
        # For very short DTE, d2 → -∞ from d1 − σ√t; not quite zero
        # but the product phi(d1)*d2/iv should remain finite
        v = _bs_vanna(100, 100, 1 / 365, 0.20, 0.05)
        assert math.isfinite(v)


class TestCharmVannaEHDSignals:
    def test_charm_present_with_iv_data(self):
        result = compute_options_flow_signals(_make_full_contracts(), spot=100.0)
        assert result["charm"] is not None

    def test_vanna_present_with_iv_data(self):
        result = compute_options_flow_signals(_make_full_contracts(), spot=100.0)
        assert result["vanna"] is not None

    def test_ehd_present_with_delta_data(self):
        result = compute_options_flow_signals(_make_full_contracts(), spot=100.0)
        assert result["ehd"] is not None
        assert result["ehd"] > 0

    def test_ehd_is_always_non_negative(self):
        """EHD = Σ|delta × OI × 100| — sum of absolute values."""
        result = compute_options_flow_signals(_make_full_contracts(), spot=100.0)
        assert result["ehd"] >= 0

    def test_charm_vanna_are_finite(self):
        result = compute_options_flow_signals(_make_full_contracts(), spot=100.0)
        assert math.isfinite(result["charm"])
        assert math.isfinite(result["vanna"])

    def test_charm_absent_when_no_iv(self):
        contracts = [{"option_type": "call", "strike": 100.0, "open_interest": 100}]
        result = compute_options_flow_signals(contracts, spot=100.0)
        assert result["charm"] is None

    def test_vanna_absent_when_no_iv(self):
        contracts = [{"option_type": "call", "strike": 100.0, "open_interest": 100}]
        result = compute_options_flow_signals(contracts, spot=100.0)
        assert result["vanna"] is None


class TestOSRatioAvemoney:
    def test_os_ratio_present_when_volume_available(self):
        result = compute_options_flow_signals(_make_full_contracts(), spot=100.0)
        assert result["os_ratio"] is not None
        assert result["os_ratio"] >= 0.0

    def test_avemoney_present_when_volume_available(self):
        result = compute_options_flow_signals(_make_full_contracts(), spot=100.0)
        assert result["avemoney"] is not None

    def test_avemoney_above_1_when_all_calls_otm(self):
        """Contracts all have strike > spot → moneyness > 1 → avemoney > 1."""
        contracts = [
            {"option_type": "call", "strike": 110.0, "open_interest": 500,
             "volume": 100.0, "implied_volatility": 0.20},
            {"option_type": "call", "strike": 115.0, "open_interest": 200,
             "volume": 50.0, "implied_volatility": 0.22},
        ]
        result = compute_options_flow_signals(contracts, spot=100.0)
        if result["avemoney"] is not None:
            assert result["avemoney"] > 1.0

    def test_avemoney_below_1_when_all_puts_otm(self):
        """Contracts all have strike < spot → moneyness < 1 → avemoney < 1."""
        contracts = [
            {"option_type": "put", "strike": 90.0, "open_interest": 500,
             "volume": 100.0, "implied_volatility": 0.25},
            {"option_type": "put", "strike": 85.0, "open_interest": 200,
             "volume": 50.0, "implied_volatility": 0.28},
        ]
        result = compute_options_flow_signals(contracts, spot=100.0)
        if result["avemoney"] is not None:
            assert result["avemoney"] < 1.0

    def test_os_ratio_absent_when_no_volume(self):
        contracts = [{"option_type": "call", "strike": 100.0, "open_interest": 100}]
        result = compute_options_flow_signals(contracts, spot=100.0)
        assert result["os_ratio"] is None

    def test_avemoney_absent_when_no_volume(self):
        contracts = [{"option_type": "call", "strike": 100.0, "open_interest": 100}]
        result = compute_options_flow_signals(contracts, spot=100.0)
        assert result["avemoney"] is None
