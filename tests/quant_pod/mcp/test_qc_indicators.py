# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Tests for qc_indicators.py MCP tools.

Covers compute_technical_indicators, compute_all_features, list_available_indicators,
compute_feature_matrix, and compute_quantagent_features.
All tools use _get_reader() for data — mocked with synthetic OHLCV.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from quantstack.mcp.tools.qc_indicators import (
    compute_all_features,
    compute_feature_matrix,
    compute_quantagent_features,
    compute_technical_indicators,
    list_available_indicators,
)
from tests.quantstack.mcp.conftest import _fn, synthetic_ohlcv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_store(df: pd.DataFrame) -> MagicMock:
    """Build a mock PgDataStore that returns *df* from load_ohlcv."""
    store = MagicMock()
    store.load_ohlcv.return_value = df
    store.close.return_value = None
    return store


def _empty_store() -> MagicMock:
    return _mock_store(pd.DataFrame())


# ---------------------------------------------------------------------------
# list_available_indicators — pure computation, no I/O
# ---------------------------------------------------------------------------


class TestListAvailableIndicators:
    @pytest.mark.asyncio
    async def test_returns_categories(self):
        result = await _fn(list_available_indicators)()
        assert "categories" in result
        assert "signal_hierarchy" in result
        assert "total_indicators" in result
        assert result["total_indicators"] == 200

    @pytest.mark.asyncio
    async def test_hierarchy_tiers(self):
        result = await _fn(list_available_indicators)()
        hierarchy = result["signal_hierarchy"]
        assert "tier_1_retail" in hierarchy
        assert "tier_2_smart_money" in hierarchy
        assert "tier_3_institutional" in hierarchy
        assert "tier_4_regime_macro" in hierarchy

    @pytest.mark.asyncio
    async def test_categories_have_indicators(self):
        result = await _fn(list_available_indicators)()
        for cat_name, cat in result["categories"].items():
            assert "indicators" in cat, f"Category {cat_name} missing 'indicators'"
            assert len(cat["indicators"]) > 0


# ---------------------------------------------------------------------------
# compute_technical_indicators
# ---------------------------------------------------------------------------


class TestComputeTechnicalIndicators:
    @pytest.mark.asyncio
    async def test_happy_path(self):
        df = synthetic_ohlcv("SPY", n_days=300)
        with patch(
            "quantstack.mcp.tools.qc_indicators._get_reader",
            return_value=_mock_store(df),
        ):
            result = await _fn(compute_technical_indicators)(symbol="SPY")
        assert "error" not in result
        assert result["symbol"] == "SPY"
        assert result["rows"] > 0
        assert len(result["indicators_computed"]) > 0
        assert "data" in result

    @pytest.mark.asyncio
    async def test_empty_data_returns_error(self):
        with patch(
            "quantstack.mcp.tools.qc_indicators._get_reader",
            return_value=_empty_store(),
        ):
            result = await _fn(compute_technical_indicators)(symbol="NOPE")
        assert "error" in result
        assert result["symbol"] == "NOPE"

    @pytest.mark.asyncio
    async def test_filter_specific_indicators(self):
        df = synthetic_ohlcv("SPY", n_days=300)
        with patch(
            "quantstack.mcp.tools.qc_indicators._get_reader",
            return_value=_mock_store(df),
        ):
            result = await _fn(compute_technical_indicators)(
                symbol="SPY", indicators=["RSI", "MACD"]
            )
        assert "error" not in result
        # Should have RSI/MACD related columns but NOT all indicators
        ind_names = [c.lower() for c in result["indicators_computed"]]
        has_rsi = any("rsi" in c for c in ind_names)
        assert has_rsi, f"Expected RSI in {result['indicators_computed']}"

    @pytest.mark.asyncio
    async def test_end_date_filter(self):
        df = synthetic_ohlcv("SPY", n_days=300)
        with patch(
            "quantstack.mcp.tools.qc_indicators._get_reader",
            return_value=_mock_store(df),
        ):
            result = await _fn(compute_technical_indicators)(
                symbol="SPY", end_date="2023-06-01"
            )
        assert "error" not in result
        # Should have fewer rows than the full dataset
        assert result["rows"] < 300

    @pytest.mark.asyncio
    async def test_end_date_too_early_returns_error(self):
        df = synthetic_ohlcv("SPY", n_days=300)
        with patch(
            "quantstack.mcp.tools.qc_indicators._get_reader",
            return_value=_mock_store(df),
        ):
            result = await _fn(compute_technical_indicators)(
                symbol="SPY", end_date="2020-01-01"
            )
        assert "error" in result

    @pytest.mark.asyncio
    async def test_weekly_timeframe(self):
        df = synthetic_ohlcv("SPY", n_days=300)
        with patch(
            "quantstack.mcp.tools.qc_indicators._get_reader",
            return_value=_mock_store(df),
        ):
            result = await _fn(compute_technical_indicators)(
                symbol="SPY", timeframe="weekly"
            )
        # Should still compute — timeframe changes the Timeframe enum passed to TechnicalIndicators
        assert result["symbol"] == "SPY"


# ---------------------------------------------------------------------------
# compute_all_features
# ---------------------------------------------------------------------------


class TestComputeAllFeatures:
    @pytest.mark.asyncio
    async def test_happy_path(self):
        df = synthetic_ohlcv("QQQ", n_days=300)
        with patch(
            "quantstack.mcp.tools.qc_indicators._get_reader",
            return_value=_mock_store(df),
        ):
            result = await _fn(compute_all_features)(symbol="QQQ")
        assert "error" not in result
        assert result["symbol"] == "QQQ"
        assert result["total_features"] > 0
        assert result["rows"] > 0
        assert "feature_names" in result

    @pytest.mark.asyncio
    async def test_empty_data_returns_error(self):
        with patch(
            "quantstack.mcp.tools.qc_indicators._get_reader",
            return_value=_empty_store(),
        ):
            result = await _fn(compute_all_features)(symbol="EMPTY")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_end_date_filter(self):
        df = synthetic_ohlcv("QQQ", n_days=300)
        with patch(
            "quantstack.mcp.tools.qc_indicators._get_reader",
            return_value=_mock_store(df),
        ):
            result = await _fn(compute_all_features)(
                symbol="QQQ", end_date="2023-06-01"
            )
        assert "error" not in result
        assert result["rows"] < 300


# ---------------------------------------------------------------------------
# compute_feature_matrix
# ---------------------------------------------------------------------------


class TestComputeFeatureMatrix:
    @pytest.mark.asyncio
    async def test_happy_path(self):
        df = synthetic_ohlcv("AAPL", n_days=300)
        with patch(
            "quantstack.mcp.tools.qc_indicators._get_reader",
            return_value=_mock_store(df),
        ):
            result = await _fn(compute_feature_matrix)(symbol="AAPL")
        assert "error" not in result
        assert result["symbol"] == "AAPL"
        assert result["total_features"] > 0
        assert "latest_values" in result
        assert "feature_names" in result

    @pytest.mark.asyncio
    async def test_include_all_flag(self):
        df = synthetic_ohlcv("AAPL", n_days=300)
        with patch(
            "quantstack.mcp.tools.qc_indicators._get_reader",
            return_value=_mock_store(df),
        ):
            result_core = await _fn(compute_feature_matrix)(
                symbol="AAPL", include_all=False
            )
            result_all = await _fn(compute_feature_matrix)(
                symbol="AAPL", include_all=True
            )
        assert "error" not in result_core, f"Core errored: {result_core.get('error')}"
        assert "error" not in result_all, f"All errored: {result_all.get('error')}"
        # include_all=True should produce at least as many features
        assert result_all["total_features"] >= result_core["total_features"]

    @pytest.mark.asyncio
    async def test_empty_data_returns_error(self):
        with patch(
            "quantstack.mcp.tools.qc_indicators._get_reader",
            return_value=_empty_store(),
        ):
            result = await _fn(compute_feature_matrix)(symbol="NONE")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_latest_values_are_clean(self):
        """latest_values should have no NaN — they should be replaced with None."""
        df = synthetic_ohlcv("AAPL", n_days=300)
        with patch(
            "quantstack.mcp.tools.qc_indicators._get_reader",
            return_value=_mock_store(df),
        ):
            result = await _fn(compute_feature_matrix)(symbol="AAPL")
        for k, v in result["latest_values"].items():
            if v is not None:
                assert isinstance(v, float), f"{k}={v!r} is not float or None"


# ---------------------------------------------------------------------------
# compute_quantagent_features
# ---------------------------------------------------------------------------


class TestComputeQuantAgentFeatures:
    @pytest.mark.asyncio
    async def test_happy_path(self):
        df = synthetic_ohlcv("TSLA", n_days=200)
        with patch(
            "quantstack.mcp.tools.qc_indicators._get_reader",
            return_value=_mock_store(df),
        ):
            result = await _fn(compute_quantagent_features)(symbol="TSLA")
        assert "error" not in result
        assert result["symbol"] == "TSLA"
        assert "pattern_features" in result
        assert "trend_features" in result
        assert "signals" in result
        assert "summary" in result

    @pytest.mark.asyncio
    async def test_empty_data_returns_error(self):
        with patch(
            "quantstack.mcp.tools.qc_indicators._get_reader",
            return_value=_empty_store(),
        ):
            result = await _fn(compute_quantagent_features)(symbol="NONE")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_insufficient_data_returns_error(self):
        """Need 50+ bars — 30 should fail."""
        df = synthetic_ohlcv("SHORT", n_days=30)
        with patch(
            "quantstack.mcp.tools.qc_indicators._get_reader",
            return_value=_mock_store(df),
        ):
            result = await _fn(compute_quantagent_features)(symbol="SHORT")
        assert "error" in result
        assert "Insufficient" in result["error"] or "50" in result["error"]

    @pytest.mark.asyncio
    async def test_summary_keys(self):
        df = synthetic_ohlcv("MSFT", n_days=200)
        with patch(
            "quantstack.mcp.tools.qc_indicators._get_reader",
            return_value=_mock_store(df),
        ):
            result = await _fn(compute_quantagent_features)(symbol="MSFT")
        summary = result["summary"]
        assert "trend_regime" in summary
        assert "trend_strength" in summary
        assert "trend_quality" in summary
        assert "is_consolidating" in summary
        assert "has_pullback_signal" in summary
        assert "has_breakout_signal" in summary

    @pytest.mark.asyncio
    async def test_end_date_filter(self):
        df = synthetic_ohlcv("MSFT", n_days=200)
        with patch(
            "quantstack.mcp.tools.qc_indicators._get_reader",
            return_value=_mock_store(df),
        ):
            result = await _fn(compute_quantagent_features)(
                symbol="MSFT", end_date="2023-06-01"
            )
        assert "error" not in result
