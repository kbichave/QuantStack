# Copyright 2024 QuantCore Contributors
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for QuantCore MCP tool tests."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def mock_datastore():
    """Mock DataStore for testing."""
    with patch("quantstack.data.storage.DataStore") as mock:
        store = MagicMock()

        # Create sample OHLCV data with enough bars
        dates = pd.date_range("2024-01-01", periods=150, freq="D")
        df = pd.DataFrame(
            {
                "open": np.linspace(100, 120, 150),
                "high": np.linspace(101, 122, 150),
                "low": np.linspace(99, 118, 150),
                "close": np.linspace(100, 120, 150),
                "volume": np.random.randint(1000000, 5000000, 150),
            },
            index=dates,
        )

        store.load_ohlcv.return_value = df
        store.close.return_value = None
        mock.return_value = store

        yield mock


@pytest.fixture
def sample_equity_curve():
    """Sample equity curve for testing."""
    return [100, 102, 101, 104, 103, 106, 108, 107, 110, 112]


@pytest.fixture
def sample_structure_spec():
    """Sample options structure specification."""
    return {
        "underlying_symbol": "SPY",
        "underlying_price": 450.0,
        "legs": [
            {
                "option_type": "call",
                "strike": 450.0,
                "expiry_days": 30,
                "quantity": 1,
                "iv": 0.20,
            }
        ],
    }


@pytest.fixture
def sample_trade_template():
    """Sample trade template."""
    return {
        "template_id": "SPY_vertical_bullish_30d",
        "symbol": "SPY",
        "direction": "bullish",
        "structure_type": "Bull Call Spread",
        "underlying_price": 450.0,
        "legs": [
            {
                "option_type": "call",
                "strike": 450.0,
                "expiry_days": 30,
                "quantity": 1,
                "iv": 0.20,
            },
            {
                "option_type": "call",
                "strike": 455.0,
                "expiry_days": 30,
                "quantity": -1,
                "iv": 0.18,
            },
        ],
        "risk_profile": {
            "max_profit": 300,
            "max_loss": -200,
            "break_evens": [452.0],
        },
        "greeks": {"delta": 25.0, "gamma": 0.5, "theta": -5.0, "vega": 10.0},
        "validation": {"is_defined_risk": True},
    }
