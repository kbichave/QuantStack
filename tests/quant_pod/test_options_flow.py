# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for quant_pod.tools.options_flow_tools — Sprint 4.

Tests UOA signal computation from a canned options chain (no network calls).
All tests call _compute_signal() directly to avoid Alpha Vantage API dependency.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest
from quant_pod.tools.options_flow_tools import (
    OptionsContract,
    OptionsFlowClient,
    OptionsFlowSignal,
)


def _contract(
    option_type: str = "call",
    volume: int = 1000,
    open_interest: int = 200,
    implied_volatility: float = 0.25,
    last_price: float = 2.5,
    bid: float = 2.4,
    ask: float = 2.6,
    strike: float = 450.0,
) -> OptionsContract:
    expiry = (date.today() + timedelta(days=14)).isoformat()
    return OptionsContract(
        symbol="SPY",
        expiry=expiry,
        strike=strike,
        option_type=option_type,
        volume=volume,
        open_interest=open_interest,
        implied_volatility=implied_volatility,
        last_price=last_price,
        bid=bid,
        ask=ask,
    )


@pytest.fixture
def client() -> OptionsFlowClient:
    return OptionsFlowClient(api_key="dummy")


# ---------------------------------------------------------------------------
# OptionsContract derived properties
# ---------------------------------------------------------------------------


class TestOptionsContractProperties:
    def test_volume_oi_ratio(self):
        c = _contract(volume=600, open_interest=200)
        assert abs(c.volume_oi_ratio - 3.0) < 1e-6

    def test_mid_price(self):
        c = _contract(bid=2.4, ask=2.6)
        assert abs(c.mid_price - 2.5) < 1e-6

    def test_notional_premium(self):
        c = _contract(volume=100, bid=2.4, ask=2.6)  # mid=2.5
        assert abs(c.notional_premium - 100 * 2.5 * 100) < 1.0  # $25,000

    def test_zero_oi_no_divzero(self):
        c = _contract(volume=100, open_interest=0)
        assert c.volume_oi_ratio == 100.0  # volume / max(OI, 1)


# ---------------------------------------------------------------------------
# _compute_signal — pure logic, no network
# ---------------------------------------------------------------------------


class TestComputeSignal:
    def _bullish_chain(self) -> list:
        """Heavy call volume, net premium positive."""
        return [
            _contract("call", volume=5000, open_interest=500, last_price=3.0, bid=2.9, ask=3.1),
            _contract("call", volume=3000, open_interest=200, last_price=2.0, bid=1.9, ask=2.1),
            _contract("put", volume=500, open_interest=300, last_price=1.0, bid=0.9, ask=1.1),
        ]

    def _bearish_chain(self) -> list:
        """Heavy put volume, net premium negative."""
        return [
            _contract("call", volume=500, open_interest=300),
            _contract("put", volume=5000, open_interest=500, last_price=3.0, bid=2.9, ask=3.1),
            _contract("put", volume=3000, open_interest=200, last_price=2.0, bid=1.9, ask=2.1),
        ]

    def test_bullish_chain_is_bullish(self, client):
        signal = client._compute_signal("SPY", self._bullish_chain())
        assert signal.flow_bias == "BULLISH"

    def test_bearish_chain_is_bearish(self, client):
        signal = client._compute_signal("SPY", self._bearish_chain())
        assert signal.flow_bias == "BEARISH"

    def test_score_in_range(self, client):
        contracts = self._bullish_chain() + self._bearish_chain()
        signal = client._compute_signal("SPY", contracts)
        assert 0.0 <= signal.unusual_score <= 100.0

    def test_put_call_ratio_computed(self, client):
        signal = client._compute_signal("SPY", self._bullish_chain())
        # Heavy calls → P/C ratio < 1
        assert signal.put_call_ratio < 1.0

    def test_net_premium_positive_for_bullish(self, client):
        signal = client._compute_signal("SPY", self._bullish_chain())
        assert signal.net_premium_usd > 0

    def test_unusual_contracts_counted(self, client):
        """vol/OI > 3 AND notional > $50k → unusual."""
        unusual = _contract(
            "call",
            volume=2000,  # vol/OI = 2000/100 = 20 > 3
            open_interest=100,
            last_price=3.0,
            bid=2.9,
            ask=3.1,  # notional = 2000 * 3.0 * 100 = $600k > $50k
        )
        normal = _contract("call", volume=100, open_interest=1000)  # vol/OI = 0.1
        signal = client._compute_signal("SPY", [unusual, normal])
        assert signal.n_unusual_contracts >= 1

    def test_high_iv_contributes_to_score(self, client):
        high_iv = _contract("call", implied_volatility=0.70, volume=500, open_interest=100)
        low_iv = _contract("call", implied_volatility=0.20, volume=500, open_interest=100)

        sig_high = client._compute_signal("SPY", [high_iv])
        sig_low = client._compute_signal("SPY", [low_iv])
        # High IV should score higher
        assert sig_high.unusual_score >= sig_low.unusual_score

    def test_returns_options_flow_signal(self, client):
        signal = client._compute_signal("SPY", self._bullish_chain())
        assert isinstance(signal, OptionsFlowSignal)
        assert signal.symbol == "SPY"

    def test_empty_chain_returns_empty_signal(self, client):
        signal = client._empty_signal("QQQ")
        assert signal.flow_bias == "NEUTRAL"
        assert signal.unusual_score == 0.0
        assert signal.symbol == "QQQ"


# ---------------------------------------------------------------------------
# OptionsFlowSignal summary property
# ---------------------------------------------------------------------------


class TestOptionsFlowSignalSummary:
    def test_summary_contains_symbol(self, client):
        contracts = [
            _contract("call", volume=1000, open_interest=200),
            _contract("put", volume=500, open_interest=100),
        ]
        signal = client._compute_signal("AAPL", contracts)
        assert "AAPL" in signal.summary

    def test_summary_contains_score(self, client):
        contracts = [_contract("call")]
        signal = client._compute_signal("SPY", contracts)
        assert "score=" in signal.summary
