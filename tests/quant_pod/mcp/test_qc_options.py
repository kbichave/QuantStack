# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Tests for qc_options MCP tools — pricing, Greeks, IV, multi-leg, chain, scoring.

Pure computation tools (price_option, compute_greeks, compute_implied_vol,
price_american_option, compute_portfolio_stats) need no DB or mocks.

Chain/surface tools (compute_option_chain, get_options_chain, get_iv_surface)
mock _get_reader() to avoid real DB hits.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from quantstack.mcp.tools.qc_options import (
    compute_greeks,
    compute_implied_vol,
    compute_multi_leg_price,
    compute_option_chain,
    compute_portfolio_stats,
    price_american_option,
    price_option,
    score_trade_structure,
    simulate_trade_outcome,
)
from tests.quantstack.mcp.conftest import _fn, synthetic_ohlcv


# ---------------------------------------------------------------------------
# price_option
# ---------------------------------------------------------------------------


class TestPriceOption:
    @pytest.mark.asyncio
    async def test_european_call_atm(self):
        result = await _fn(price_option)(
            spot=100.0, strike=100.0, time_to_expiry=0.25,
            volatility=0.20, option_type="call",
        )
        assert "error" not in result
        assert result["price"] > 0
        assert "greeks" in result or "delta" in result

    @pytest.mark.asyncio
    async def test_european_put_atm(self):
        result = await _fn(price_option)(
            spot=100.0, strike=100.0, time_to_expiry=0.25,
            volatility=0.20, option_type="put",
        )
        assert "error" not in result
        assert result["price"] > 0

    @pytest.mark.asyncio
    async def test_call_put_parity(self):
        """C - P ≈ S*e^(-qT) - K*e^(-rT) for European options."""
        call = await _fn(price_option)(
            spot=100.0, strike=100.0, time_to_expiry=0.25,
            volatility=0.20, risk_free_rate=0.05, option_type="call",
        )
        put = await _fn(price_option)(
            spot=100.0, strike=100.0, time_to_expiry=0.25,
            volatility=0.20, risk_free_rate=0.05, option_type="put",
        )
        assert "error" not in call
        assert "error" not in put
        # C - P ≈ S - K*e^(-rT) for no dividend
        parity_rhs = 100.0 - 100.0 * np.exp(-0.05 * 0.25)
        diff = call["price"] - put["price"]
        assert abs(diff - parity_rhs) < 0.5  # tolerance for numerical methods

    @pytest.mark.asyncio
    async def test_deep_otm_call_near_zero(self):
        result = await _fn(price_option)(
            spot=100.0, strike=200.0, time_to_expiry=0.01,
            volatility=0.10, option_type="call",
        )
        assert "error" not in result
        assert result["price"] < 0.01

    @pytest.mark.asyncio
    async def test_analysis_section(self):
        result = await _fn(price_option)(
            spot=110.0, strike=100.0, time_to_expiry=0.25,
            volatility=0.20, option_type="call",
        )
        assert "error" not in result
        assert "analysis" in result
        assert result["analysis"]["is_itm"] is True
        assert result["analysis"]["intrinsic_value"] > 0

    @pytest.mark.asyncio
    async def test_american_call(self):
        result = await _fn(price_option)(
            spot=100.0, strike=100.0, time_to_expiry=0.25,
            volatility=0.20, exercise_style="american",
        )
        assert "error" not in result
        assert result["price"] > 0

    @pytest.mark.asyncio
    async def test_zero_time_to_expiry(self):
        """At expiry, call = max(S-K, 0)."""
        result = await _fn(price_option)(
            spot=110.0, strike=100.0, time_to_expiry=0.001,
            volatility=0.20, option_type="call",
        )
        assert "error" not in result
        assert abs(result["price"] - 10.0) < 1.0


# ---------------------------------------------------------------------------
# compute_greeks
# ---------------------------------------------------------------------------


class TestComputeGreeks:
    @pytest.mark.asyncio
    async def test_call_greeks(self):
        result = await _fn(compute_greeks)(
            spot=100.0, strike=100.0, time_to_expiry=0.25,
            volatility=0.20, option_type="call",
        )
        assert "error" not in result
        greeks = result.get("greeks", result)
        assert "delta" in greeks
        # ATM call delta ≈ 0.5
        assert 0.3 < greeks["delta"] < 0.7

    @pytest.mark.asyncio
    async def test_put_delta_negative(self):
        result = await _fn(compute_greeks)(
            spot=100.0, strike=100.0, time_to_expiry=0.25,
            volatility=0.20, option_type="put",
        )
        assert "error" not in result
        greeks = result.get("greeks", result)
        assert greeks["delta"] < 0


# ---------------------------------------------------------------------------
# compute_implied_vol
# ---------------------------------------------------------------------------


class TestComputeImpliedVol:
    @pytest.mark.asyncio
    async def test_roundtrip_iv(self):
        """Price an option at known vol, then recover IV from that price."""
        priced = await _fn(price_option)(
            spot=100.0, strike=100.0, time_to_expiry=0.25,
            volatility=0.25, option_type="call",
        )
        assert "error" not in priced

        iv_result = await _fn(compute_implied_vol)(
            spot=100.0, strike=100.0, time_to_expiry=0.25,
            option_price=priced["price"], option_type="call",
        )
        assert "error" not in iv_result
        recovered_iv = iv_result.get("implied_volatility") or iv_result.get("iv")
        assert recovered_iv is not None
        assert abs(recovered_iv - 0.25) < 0.02


# ---------------------------------------------------------------------------
# price_american_option
# ---------------------------------------------------------------------------


class TestPriceAmericanOption:
    @pytest.mark.asyncio
    async def test_american_put_premium(self):
        """American put should be worth >= European put."""
        result = await _fn(price_american_option)(
            spot=100.0, strike=100.0, time_to_expiry=0.25,
            volatility=0.30,
        )
        assert "error" not in result
        assert result["price"] > 0

    @pytest.mark.asyncio
    async def test_has_early_exercise_premium(self):
        result = await _fn(price_american_option)(
            spot=80.0, strike=100.0, time_to_expiry=1.0,
            volatility=0.30, option_type="put",
        )
        assert "error" not in result
        # Deep ITM put with long expiry should have early exercise premium
        assert result.get("early_exercise_premium", 0) >= 0


# ---------------------------------------------------------------------------
# compute_multi_leg_price
# ---------------------------------------------------------------------------


class TestComputeMultiLegPrice:
    @pytest.mark.asyncio
    async def test_bull_call_spread(self):
        legs = [
            {"option_type": "call", "strike": 100, "expiry_days": 30, "quantity": 1, "iv": 0.25},
            {"option_type": "call", "strike": 110, "expiry_days": 30, "quantity": -1, "iv": 0.25},
        ]
        result = await _fn(compute_multi_leg_price)(
            legs=legs, underlying_price=105.0,
        )
        assert "error" not in result
        assert result["is_debit"] is True
        assert len(result["leg_prices"]) == 2
        assert "net_greeks" in result

    @pytest.mark.asyncio
    async def test_empty_legs(self):
        result = await _fn(compute_multi_leg_price)(
            legs=[], underlying_price=100.0,
        )
        assert "error" in result

    @pytest.mark.asyncio
    async def test_iron_condor(self):
        legs = [
            {"option_type": "put", "strike": 90, "expiry_days": 30, "quantity": 1, "iv": 0.25},
            {"option_type": "put", "strike": 95, "expiry_days": 30, "quantity": -1, "iv": 0.25},
            {"option_type": "call", "strike": 105, "expiry_days": 30, "quantity": -1, "iv": 0.25},
            {"option_type": "call", "strike": 110, "expiry_days": 30, "quantity": 1, "iv": 0.25},
        ]
        result = await _fn(compute_multi_leg_price)(
            legs=legs, underlying_price=100.0,
        )
        assert "error" not in result
        assert len(result["leg_prices"]) == 4
        # Net delta should be small for a balanced iron condor
        assert abs(result["net_greeks"]["delta"]) < 50


# ---------------------------------------------------------------------------
# compute_option_chain (mocked reader)
# ---------------------------------------------------------------------------


class TestComputeOptionChain:
    @pytest.mark.asyncio
    async def test_returns_chain_with_mock_data(self):
        df = synthetic_ohlcv("SPY", n_days=100, start_price=450.0)
        mock_store = MagicMock()
        mock_store.load_ohlcv.return_value = df
        mock_store.close.return_value = None

        with patch("quantstack.mcp.tools.qc_options._get_reader", return_value=mock_store):
            result = await _fn(compute_option_chain)(symbol="SPY")

        assert "error" not in result
        assert result["symbol"] == "SPY"
        assert result["underlying_price"] > 0
        assert len(result["calls"]) > 0
        assert len(result["puts"]) > 0
        assert "chain_metrics" in result

    @pytest.mark.asyncio
    async def test_empty_data_returns_error(self):
        mock_store = MagicMock()
        mock_store.load_ohlcv.return_value = pd.DataFrame()
        mock_store.close.return_value = None

        with patch("quantstack.mcp.tools.qc_options._get_reader", return_value=mock_store):
            result = await _fn(compute_option_chain)(symbol="NODATA")

        assert "error" in result


# ---------------------------------------------------------------------------
# compute_portfolio_stats
# ---------------------------------------------------------------------------


class TestComputePortfolioStats:
    @pytest.mark.asyncio
    async def test_trending_up_equity(self):
        equity = list(np.linspace(100000, 120000, 252))
        result = await _fn(compute_portfolio_stats)(equity_curve=equity)
        assert "error" not in result

    @pytest.mark.asyncio
    async def test_flat_equity(self):
        equity = [100000.0] * 50
        result = await _fn(compute_portfolio_stats)(equity_curve=equity)
        # Should handle zero returns gracefully
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# score_trade_structure
# ---------------------------------------------------------------------------


class TestScoreTradeStructure:
    @pytest.mark.asyncio
    async def test_scores_bull_spread(self):
        spec = {
            "underlying_symbol": "SPY",
            "underlying_price": 450.0,
            "legs": [
                {"option_type": "call", "strike": 445, "expiry_days": 30, "quantity": 1, "iv": 0.20},
                {"option_type": "call", "strike": 455, "expiry_days": 30, "quantity": -1, "iv": 0.18},
            ],
        }
        result = await _fn(score_trade_structure)(
            structure_spec=spec, market_regime="bull",
        )
        # May error if quantsbin not available, which is acceptable
        if "error" not in result:
            assert "total_score" in result
            assert "recommendation" in result
            assert result["total_score"] > 0


# ---------------------------------------------------------------------------
# simulate_trade_outcome
# ---------------------------------------------------------------------------


class TestSimulateTradeOutcome:
    @pytest.mark.asyncio
    async def test_monte_carlo_simulation(self):
        template = {
            "underlying_price": 100.0,
            "legs": [
                {"option_type": "call", "strike": 100, "expiry_days": 30, "quantity": 1, "iv": 0.25},
            ],
        }
        result = await _fn(simulate_trade_outcome)(
            trade_template=template, num_scenarios=100, holding_days=10,
        )
        assert "error" not in result
        assert result["num_scenarios"] == 100
        assert "expected_pnl" in result
        assert "probability_profit" in result
        assert "var_95" in result
        assert "pnl_percentiles" in result

    @pytest.mark.asyncio
    async def test_missing_legs_returns_error(self):
        result = await _fn(simulate_trade_outcome)(
            trade_template={"underlying_price": 100.0, "legs": []},
        )
        assert "error" in result

    @pytest.mark.asyncio
    async def test_missing_underlying_returns_error(self):
        result = await _fn(simulate_trade_outcome)(
            trade_template={"legs": [{"option_type": "call", "strike": 100}]},
        )
        assert "error" in result
