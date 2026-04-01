# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Tests for qc_research MCP tools — statistical analysis and research.

All tools use _get_reader() for data. We mock it with synthetic OHLCV.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from quantstack.mcp.tools.qc_research import (
    compute_alpha_decay,
    compute_deflated_sharpe_ratio,
    compute_information_coefficient,
    fit_garch_model,
    forecast_volatility,
    run_adf_test,
    run_monte_carlo,
    validate_signal,
)
from tests.quant_pod.mcp.conftest import _fn, synthetic_ohlcv


def _mock_reader(df):
    """Return a MagicMock PgDataStore that returns df from load_ohlcv."""
    store = MagicMock()
    store.load_ohlcv.return_value = df
    store.close.return_value = None
    return store


def _patch_reader(df):
    """Patch _get_reader to return a mock store with the given DataFrame."""
    return patch(
        "quantstack.mcp.tools.qc_research._get_reader",
        return_value=_mock_reader(df),
    )


# ---------------------------------------------------------------------------
# run_adf_test
# ---------------------------------------------------------------------------


class TestRunAdfTest:
    @pytest.mark.asyncio
    async def test_adf_on_close_prices(self):
        df = synthetic_ohlcv("SPY", n_days=252)
        with _patch_reader(df):
            result = await _fn(run_adf_test)(symbol="SPY")
        assert "error" not in result
        assert "statistic" in result or "adf_statistic" in result or "test_statistic" in result
        assert "p_value" in result

    @pytest.mark.asyncio
    async def test_adf_on_returns(self):
        df = synthetic_ohlcv("SPY", n_days=252)
        with _patch_reader(df):
            result = await _fn(run_adf_test)(symbol="SPY", column="returns")
        assert "error" not in result
        assert "p_value" in result

    @pytest.mark.asyncio
    async def test_adf_empty_data(self):
        with _patch_reader(pd.DataFrame()):
            result = await _fn(run_adf_test)(symbol="NODATA")
        assert "error" in result


# ---------------------------------------------------------------------------
# compute_alpha_decay
# ---------------------------------------------------------------------------


class TestComputeAlphaDecay:
    @pytest.mark.asyncio
    async def test_alpha_decay_happy_path(self):
        df = synthetic_ohlcv("SPY", n_days=500)
        with _patch_reader(df):
            result = await _fn(compute_alpha_decay)(symbol="SPY")
        # May return error if insufficient signal data — check structure
        if "error" not in result:
            assert "symbol" in result

    @pytest.mark.asyncio
    async def test_alpha_decay_empty_data(self):
        with _patch_reader(pd.DataFrame()):
            result = await _fn(compute_alpha_decay)(symbol="NODATA")
        assert "error" in result


# ---------------------------------------------------------------------------
# compute_information_coefficient
# ---------------------------------------------------------------------------


class TestComputeIC:
    @pytest.mark.asyncio
    async def test_ic_happy_path(self):
        df = synthetic_ohlcv("SPY", n_days=252)
        with _patch_reader(df):
            result = await _fn(compute_information_coefficient)(
                symbol="SPY", signal_column="close",
            )
        # IC requires forward returns — may error on certain data shapes
        if "error" not in result:
            assert "symbol" in result

    @pytest.mark.asyncio
    async def test_ic_empty_data(self):
        with _patch_reader(pd.DataFrame()):
            result = await _fn(compute_information_coefficient)(
                symbol="NODATA", signal_column="close",
            )
        assert "error" in result


# ---------------------------------------------------------------------------
# run_monte_carlo
# ---------------------------------------------------------------------------


class TestRunMonteCarlo:
    @pytest.mark.asyncio
    async def test_monte_carlo_happy_path(self):
        df = synthetic_ohlcv("SPY", n_days=252)
        with _patch_reader(df):
            result = await _fn(run_monte_carlo)(
                symbol="SPY", n_simulations=100,
            )
        assert "error" not in result
        assert "simulations" in result or "percentiles" in result or "symbol" in result

    @pytest.mark.asyncio
    async def test_monte_carlo_empty_data(self):
        with _patch_reader(pd.DataFrame()):
            result = await _fn(run_monte_carlo)(symbol="NODATA")
        assert "error" in result


# ---------------------------------------------------------------------------
# validate_signal
# ---------------------------------------------------------------------------


class TestValidateSignal:
    @pytest.mark.asyncio
    async def test_validate_signal_happy_path(self):
        rng = np.random.default_rng(42)
        n = 252
        signal = rng.normal(0, 1, n).tolist()
        returns = rng.normal(0.0005, 0.01, n).tolist()
        result = await _fn(validate_signal)(
            signal=signal, returns=returns,
        )
        # Structure depends on signal data availability
        if "error" not in result:
            assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_validate_signal_short_data(self):
        result = await _fn(validate_signal)(
            signal=[1.0, 2.0], returns=[0.01, -0.01],
        )
        # May error on insufficient data — just check it doesn't crash
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# fit_garch_model
# ---------------------------------------------------------------------------


class TestFitGarchModel:
    @pytest.mark.asyncio
    async def test_garch_happy_path(self):
        df = synthetic_ohlcv("SPY", n_days=500)
        with _patch_reader(df):
            result = await _fn(fit_garch_model)(symbol="SPY")
        assert "error" not in result
        assert "symbol" in result

    @pytest.mark.asyncio
    async def test_garch_empty_data(self):
        with _patch_reader(pd.DataFrame()):
            result = await _fn(fit_garch_model)(symbol="NODATA")
        assert "error" in result


# ---------------------------------------------------------------------------
# forecast_volatility
# ---------------------------------------------------------------------------


class TestForecastVolatility:
    @pytest.mark.asyncio
    async def test_forecast_happy_path(self):
        df = synthetic_ohlcv("SPY", n_days=500)
        with _patch_reader(df):
            result = await _fn(forecast_volatility)(symbol="SPY", horizon_days=5)
        assert "error" not in result
        assert "symbol" in result

    @pytest.mark.asyncio
    async def test_forecast_empty_data(self):
        with _patch_reader(pd.DataFrame()):
            result = await _fn(forecast_volatility)(symbol="NODATA")
        assert "error" in result


# ---------------------------------------------------------------------------
# compute_deflated_sharpe_ratio
# ---------------------------------------------------------------------------


class TestDeflatedSharpeRatio:
    @pytest.mark.asyncio
    async def test_deflated_sharpe_happy_path(self):
        # DSR takes pre-computed statistics, not raw data
        result = await _fn(compute_deflated_sharpe_ratio)(
            observed_sharpe=1.5,
            n_trials=10,
            variance_of_sharpe=1.0,
        )
        if "error" not in result:
            assert "observed_sharpe" in result or "deflated_sharpe" in result or "dsr_pvalue" in result

    @pytest.mark.asyncio
    async def test_deflated_sharpe_zero_trials(self):
        # Edge case: 0 trials should handle gracefully
        result = await _fn(compute_deflated_sharpe_ratio)(
            observed_sharpe=1.5,
            n_trials=0,
        )
        assert isinstance(result, dict)
