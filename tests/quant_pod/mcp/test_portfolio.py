# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Tests for portfolio optimization MCP tools (portfolio.py).

Covers optimize_portfolio and compute_hrp_weights with mocked data dependencies.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from quantstack.mcp.tools.portfolio import (
    _cluster_variance,
    _hrp_correlation_distance,
    _hrp_quasi_diag,
    _hrp_recursive_bisection,
    _load_returns,
    _max_sharpe,
    _min_variance,
    _portfolio_stats,
    _risk_parity,
    _run_hrp,
    compute_hrp_weights,
    optimize_portfolio,
)
from tests.quant_pod.mcp.conftest import _fn, synthetic_ohlcv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _multi_asset_returns(
    symbols: list[str] | None = None, n_days: int = 300, seed: int = 42
) -> pd.DataFrame:
    """Build a synthetic log-returns DataFrame for multiple assets."""
    symbols = symbols or ["AAPL", "MSFT", "GOOG"]
    rng = np.random.default_rng(seed)
    prices = {}
    for i, sym in enumerate(symbols):
        t = np.arange(n_days)
        trend = 100 + t * (0.02 + 0.01 * i)
        noise = rng.normal(0, 0.5, n_days)
        prices[sym] = trend + noise

    price_df = pd.DataFrame(
        prices,
        index=pd.date_range("2023-01-01", periods=n_days, freq="1D"),
    )
    returns = np.log(price_df / price_df.shift(1)).dropna()
    return returns


def _mock_store_for_symbols(symbols: list[str], n_days: int = 400) -> MagicMock:
    """Build a mock PgDataStore that returns synthetic OHLCV per symbol."""
    store = MagicMock()

    def load_ohlcv(symbol, tf):
        if symbol in symbols:
            return synthetic_ohlcv(symbol=symbol, n_days=n_days, seed=hash(symbol) % 2**31)
        return pd.DataFrame()

    store.load_ohlcv = MagicMock(side_effect=load_ohlcv)
    store.close = MagicMock()
    return store


# ---------------------------------------------------------------------------
# Pure math / internal function tests
# ---------------------------------------------------------------------------


class TestHRPInternals:
    """Test HRP algorithm internals without any I/O."""

    def test_correlation_distance_identity(self):
        """Perfect correlation => distance 0; zero correlation => distance ~0.707."""
        corr = pd.DataFrame(
            [[1.0, 0.0], [0.0, 1.0]], columns=["A", "B"], index=["A", "B"]
        )
        dist = _hrp_correlation_distance(corr)
        assert dist[0, 0] == 0.0  # diagonal
        assert dist[1, 1] == 0.0
        np.testing.assert_almost_equal(dist[0, 1], np.sqrt(0.5), decimal=6)

    def test_correlation_distance_perfect(self):
        """Perfect positive correlation => distance 0."""
        corr = pd.DataFrame(
            [[1.0, 1.0], [1.0, 1.0]], columns=["A", "B"], index=["A", "B"]
        )
        dist = _hrp_correlation_distance(corr)
        assert dist[0, 1] == 0.0

    def test_quasi_diag_produces_valid_permutation(self):
        """Leaf order should be a permutation of [0, n-1]."""
        returns = _multi_asset_returns(["A", "B", "C", "D"])
        corr = returns.corr()
        dist = _hrp_correlation_distance(corr)
        n = len(dist)
        condensed = dist[np.triu_indices(n, k=1)]
        from scipy.cluster.hierarchy import linkage

        link = linkage(condensed, method="ward")
        order = _hrp_quasi_diag(link)
        assert sorted(order) == list(range(4))

    def test_recursive_bisection_weights_sum_to_one(self):
        """Weights from recursive bisection should sum to 1.0."""
        returns = _multi_asset_returns(["A", "B", "C"])
        cov = returns.cov()
        sorted_idx = list(range(len(cov)))
        weights = _hrp_recursive_bisection(cov, sorted_idx)
        np.testing.assert_almost_equal(weights.sum(), 1.0, decimal=6)

    def test_cluster_variance_raises_on_degenerate(self):
        """Degenerate covariance (zero variance) should raise ValueError."""
        cov = pd.DataFrame(
            np.zeros((3, 3)), columns=["A", "B", "C"], index=["A", "B", "C"]
        )
        with pytest.raises(ValueError, match="Degenerate"):
            _cluster_variance(cov, [0, 1])

    def test_run_hrp_full_pipeline(self):
        """Full HRP pipeline returns valid weights dict."""
        returns = _multi_asset_returns(["A", "B", "C", "D"])
        weights, link, order = _run_hrp(returns)

        assert isinstance(weights, dict)
        assert len(weights) == 4
        np.testing.assert_almost_equal(sum(weights.values()), 1.0, decimal=6)
        assert all(w > 0 for w in weights.values())


class TestOptimizers:
    """Test each optimizer on synthetic returns."""

    def test_min_variance_weights_sum_to_one(self):
        returns = _multi_asset_returns()
        weights = _min_variance(returns)
        np.testing.assert_almost_equal(sum(weights.values()), 1.0, decimal=4)
        assert all(w >= -0.01 for w in weights.values())  # non-negative (within tolerance)

    def test_risk_parity_weights_sum_to_one(self):
        returns = _multi_asset_returns()
        weights = _risk_parity(returns)
        np.testing.assert_almost_equal(sum(weights.values()), 1.0, decimal=4)

    def test_max_sharpe_weights_sum_to_one(self):
        returns = _multi_asset_returns()
        weights = _max_sharpe(returns, risk_free_rate=0.05)
        np.testing.assert_almost_equal(sum(weights.values()), 1.0, decimal=4)

    def test_max_sharpe_with_zero_rf(self):
        returns = _multi_asset_returns()
        weights = _max_sharpe(returns, risk_free_rate=0.0)
        assert len(weights) == 3
        np.testing.assert_almost_equal(sum(weights.values()), 1.0, decimal=4)


class TestPortfolioStats:
    """Test the _portfolio_stats helper."""

    def test_stats_structure(self):
        returns = _multi_asset_returns()
        weights = {"AAPL": 0.4, "MSFT": 0.3, "GOOG": 0.3}
        stats = _portfolio_stats(weights, returns, risk_free_rate=0.05)
        assert "expected_return" in stats
        assert "expected_volatility" in stats
        assert "sharpe_ratio" in stats
        assert "diversification_ratio" in stats
        assert "risk_contributions" in stats
        assert isinstance(stats["risk_contributions"], dict)
        assert set(stats["risk_contributions"].keys()) == {"AAPL", "MSFT", "GOOG"}

    def test_equal_weight_risk_contributions(self):
        """With equal weights on identical-vol assets, risk contributions should be roughly equal."""
        # Use uniform random returns so assets are similar
        rng = np.random.default_rng(0)
        data = pd.DataFrame(
            rng.normal(0, 0.01, (200, 3)),
            columns=["A", "B", "C"],
            index=pd.date_range("2023-01-01", periods=200),
        )
        weights = {"A": 1 / 3, "B": 1 / 3, "C": 1 / 3}
        stats = _portfolio_stats(weights, data, 0.0)
        # Risk contributions should be close to 33.33%
        for rc in stats["risk_contributions"].values():
            assert 20 < rc < 45  # generous tolerance for random data


class TestLoadReturns:
    """Test _load_returns with mocked data sources."""

    def test_load_returns_happy_path(self):
        symbols = ["AAPL", "MSFT", "GOOG"]
        store = _mock_store_for_symbols(symbols)
        with patch("quantstack.mcp.tools.portfolio._get_reader", return_value=store):
            returns = _load_returns(symbols, lookback_days=252)
        assert isinstance(returns, pd.DataFrame)
        assert len(returns) >= 20
        # All requested symbols present
        for s in symbols:
            assert s in returns.columns

    def test_load_returns_insufficient_symbols_raises(self):
        """Only 1 symbol has data => should raise ValueError."""
        store = _mock_store_for_symbols(["AAPL"])  # only AAPL has data
        with patch("quantstack.mcp.tools.portfolio._get_reader", return_value=store):
            with pytest.raises(ValueError, match="at least 2 symbols"):
                _load_returns(["AAPL", "NOPE"], lookback_days=252)

    def test_load_returns_fallback_to_provider(self):
        """When PgDataStore returns empty, falls back to DataProviderRegistry."""
        store = MagicMock()
        store.load_ohlcv.return_value = pd.DataFrame()
        store.close.return_value = None

        registry = MagicMock()

        def fetch_ohlcv(symbol, asset_class, tf, start, end):
            return synthetic_ohlcv(symbol=symbol, n_days=400, seed=hash(symbol) % 2**31)

        registry.fetch_ohlcv = MagicMock(side_effect=fetch_ohlcv)

        with (
            patch("quantstack.mcp.tools.portfolio._get_reader", return_value=store),
            patch("quantstack.mcp.tools.portfolio.DataProviderRegistry") as MockReg,
            patch("quantstack.mcp.tools.portfolio.get_settings"),
        ):
            MockReg.from_settings.return_value = registry
            returns = _load_returns(["AAPL", "MSFT"], lookback_days=252)

        assert isinstance(returns, pd.DataFrame)
        assert len(returns) >= 20

    def test_load_returns_too_few_observations_raises(self):
        """Very short data gets filtered out by len >= lookback_days // 2 check."""
        symbols = ["A", "B"]
        store = MagicMock()

        def load_ohlcv(sym, tf):
            return synthetic_ohlcv(symbol=sym, n_days=10)

        store.load_ohlcv = MagicMock(side_effect=load_ohlcv)
        store.close = MagicMock()

        with patch("quantstack.mcp.tools.portfolio._get_reader", return_value=store):
            # 10-day data < lookback_days//2 (126) → both symbols excluded → "at least 2"
            with pytest.raises(ValueError, match="at least 2 symbols"):
                _load_returns(symbols, lookback_days=252)


# ---------------------------------------------------------------------------
# MCP tool tests (async)
# ---------------------------------------------------------------------------


class TestOptimizePortfolioTool:
    @pytest.mark.asyncio
    async def test_invalid_method_returns_error(self):
        result = await _fn(optimize_portfolio)(symbols=["A", "B"], method="bogus")
        assert result["success"] is False
        assert "Unknown method" in result["error"]

    @pytest.mark.asyncio
    async def test_single_symbol_returns_error(self):
        result = await _fn(optimize_portfolio)(symbols=["AAPL"])
        assert result["success"] is False
        assert "at least 2" in result["error"]

    @pytest.mark.asyncio
    async def test_hrp_happy_path(self):
        store = _mock_store_for_symbols(["AAPL", "MSFT", "GOOG"])
        with patch("quantstack.mcp.tools.portfolio._get_reader", return_value=store):
            result = await _fn(optimize_portfolio)(
                symbols=["AAPL", "MSFT", "GOOG"], method="hrp"
            )
        assert result["success"] is True
        assert result["method"] == "hrp"
        assert "weights" in result
        assert "sharpe_ratio" in result
        assert "risk_contributions" in result
        # Weights should sum to ~1
        np.testing.assert_almost_equal(
            sum(result["weights"].values()), 1.0, decimal=4
        )

    @pytest.mark.asyncio
    async def test_min_variance_happy_path(self):
        store = _mock_store_for_symbols(["AAPL", "MSFT"])
        with patch("quantstack.mcp.tools.portfolio._get_reader", return_value=store):
            result = await _fn(optimize_portfolio)(
                symbols=["AAPL", "MSFT"], method="min_variance"
            )
        assert result["success"] is True
        assert result["method"] == "min_variance"

    @pytest.mark.asyncio
    async def test_risk_parity_happy_path(self):
        store = _mock_store_for_symbols(["AAPL", "MSFT", "GOOG"])
        with patch("quantstack.mcp.tools.portfolio._get_reader", return_value=store):
            result = await _fn(optimize_portfolio)(
                symbols=["AAPL", "MSFT", "GOOG"], method="risk_parity"
            )
        assert result["success"] is True
        assert result["method"] == "risk_parity"

    @pytest.mark.asyncio
    async def test_max_sharpe_happy_path(self):
        store = _mock_store_for_symbols(["AAPL", "MSFT"])
        with patch("quantstack.mcp.tools.portfolio._get_reader", return_value=store):
            result = await _fn(optimize_portfolio)(
                symbols=["AAPL", "MSFT"], method="max_sharpe", risk_free_rate=0.04
            )
        assert result["success"] is True
        assert result["method"] == "max_sharpe"

    @pytest.mark.asyncio
    async def test_equal_weight_happy_path(self):
        store = _mock_store_for_symbols(["AAPL", "MSFT", "GOOG"])
        with patch("quantstack.mcp.tools.portfolio._get_reader", return_value=store):
            result = await _fn(optimize_portfolio)(
                symbols=["AAPL", "MSFT", "GOOG"], method="equal_weight"
            )
        assert result["success"] is True
        # Equal weight: each should be ~0.333
        for w in result["weights"].values():
            np.testing.assert_almost_equal(w, 1 / 3, decimal=4)

    @pytest.mark.asyncio
    async def test_data_error_returns_failure(self):
        """When _load_returns raises, the tool catches and returns success=False."""
        store = MagicMock()
        store.load_ohlcv.return_value = pd.DataFrame()
        store.close.return_value = None
        with (
            patch("quantstack.mcp.tools.portfolio._get_reader", return_value=store),
            patch(
                "quantstack.mcp.tools.portfolio.DataProviderRegistry.from_settings",
                side_effect=Exception("no settings"),
            ),
            patch("quantstack.mcp.tools.portfolio.get_settings"),
        ):
            result = await _fn(optimize_portfolio)(symbols=["A", "B"])
        assert result["success"] is False
        assert "error" in result


class TestComputeHRPWeightsTool:
    @pytest.mark.asyncio
    async def test_single_symbol_returns_error(self):
        result = await _fn(compute_hrp_weights)(symbols=["AAPL"])
        assert result["success"] is False
        assert "at least 2" in result["error"]

    @pytest.mark.asyncio
    async def test_happy_path(self):
        store = _mock_store_for_symbols(["AAPL", "MSFT", "GOOG"])
        with patch("quantstack.mcp.tools.portfolio._get_reader", return_value=store):
            result = await _fn(compute_hrp_weights)(
                symbols=["AAPL", "MSFT", "GOOG"]
            )
        assert result["success"] is True
        assert result["method"] == "hrp"
        assert "weights" in result
        assert "leaf_order" in result
        assert "cluster_merges" in result
        assert "annualized_vols" in result
        assert "lowest_correlations" in result
        assert "highest_correlations" in result
        assert "risk_contributions" in result
        np.testing.assert_almost_equal(
            sum(result["weights"].values()), 1.0, decimal=4
        )

    @pytest.mark.asyncio
    async def test_two_symbols(self):
        """Minimum viable case: 2 symbols."""
        store = _mock_store_for_symbols(["A", "B"])
        with patch("quantstack.mcp.tools.portfolio._get_reader", return_value=store):
            result = await _fn(compute_hrp_weights)(symbols=["A", "B"])
        assert result["success"] is True
        assert len(result["weights"]) == 2
        # With 2 symbols, there's only 1 cluster merge
        assert len(result["cluster_merges"]) == 1

    @pytest.mark.asyncio
    async def test_data_failure_returns_error(self):
        store = MagicMock()
        store.load_ohlcv.return_value = pd.DataFrame()
        store.close.return_value = None
        with (
            patch("quantstack.mcp.tools.portfolio._get_reader", return_value=store),
            patch(
                "quantstack.mcp.tools.portfolio.DataProviderRegistry.from_settings",
                side_effect=Exception("offline"),
            ),
            patch("quantstack.mcp.tools.portfolio.get_settings"),
        ):
            result = await _fn(compute_hrp_weights)(symbols=["X", "Y"])
        assert result["success"] is False
        assert "error" in result
