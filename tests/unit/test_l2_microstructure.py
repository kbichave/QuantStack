# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for L2 microstructure signals (pure math — no IBKR API calls)."""

import numpy as np
import pandas as pd
import pytest

from quantstack.signal_engine.collectors.l2_microstructure import (
    compute_book_signals,
    kyle_lambda_df,
    kyle_lambda_ohlcv,
)


@pytest.fixture
def ohlcv_with_volume():
    dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
    np.random.seed(42)
    close = pd.Series(100 + np.cumsum(np.random.randn(100) * 0.5), index=dates)
    volume = pd.Series(
        np.random.randint(500_000, 5_000_000, 100).astype(float), index=dates
    )
    return close, volume


# ---------------------------------------------------------------------------
# Kyle's Lambda (OHLCV proxy)
# ---------------------------------------------------------------------------


class TestKyleLambdaOHLCV:
    def test_returns_series(self, ohlcv_with_volume):
        close, volume = ohlcv_with_volume
        result = kyle_lambda_ohlcv(close, volume)
        assert isinstance(result, pd.Series)

    def test_same_length_as_input(self, ohlcv_with_volume):
        close, volume = ohlcv_with_volume
        result = kyle_lambda_ohlcv(close, volume)
        assert len(result) == len(close)

    def test_nan_in_warmup(self, ohlcv_with_volume):
        close, volume = ohlcv_with_volume
        period = 22
        result = kyle_lambda_ohlcv(close, volume, period=period)
        assert result.iloc[:period].isna().all()

    def test_non_nan_after_warmup(self, ohlcv_with_volume):
        close, volume = ohlcv_with_volume
        result = kyle_lambda_ohlcv(close, volume, period=22)
        assert result.iloc[22:].notna().any()

    def test_zero_volume_handled(self, ohlcv_with_volume):
        close, volume = ohlcv_with_volume
        volume = volume.copy()
        volume.iloc[5] = 0  # should not crash
        result = kyle_lambda_ohlcv(close, volume)
        assert isinstance(result, pd.Series)

    def test_high_volume_low_lambda(self):
        """With large volume for small price moves, lambda should be small."""
        dates = pd.date_range(start="2023-01-01", periods=50, freq="D")
        close = pd.Series(100 + np.random.randn(50) * 0.01, index=dates)  # tiny moves
        volume = pd.Series(np.full(50, 10_000_000.0), index=dates)  # huge volume
        result = kyle_lambda_ohlcv(close, volume, period=22)
        valid = result.dropna()
        if len(valid) > 0:
            assert valid.abs().mean() < 1e-4  # very low impact per share


class TestKyleLambdaDf:
    def test_returns_dataframe(self, ohlcv_with_volume):
        close, volume = ohlcv_with_volume
        result = kyle_lambda_df(close, volume)
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self, ohlcv_with_volume):
        close, volume = ohlcv_with_volume
        result = kyle_lambda_df(close, volume)
        assert {"kyle_lambda", "kyle_lambda_zscore", "high_impact"}.issubset(
            set(result.columns)
        )

    def test_high_impact_binary(self, ohlcv_with_volume):
        close, volume = ohlcv_with_volume
        result = kyle_lambda_df(close, volume)
        vals = result["high_impact"].unique()
        assert set(vals).issubset({0, 1})

    def test_zscore_mean_near_zero(self, ohlcv_with_volume):
        close, volume = ohlcv_with_volume
        result = kyle_lambda_df(close, volume)
        valid_z = result["kyle_lambda_zscore"].dropna()
        if len(valid_z) > 10:
            assert abs(valid_z.mean()) < 1.5  # should be approximately zero-centred


# ---------------------------------------------------------------------------
# compute_book_signals (pure math from L2 snapshot)
# ---------------------------------------------------------------------------


class TestComputeBookSignals:
    def _book(self, bid_price=100.0, ask_price=100.02, bid_size=1000, ask_size=500):
        return {
            "bids": [(bid_price, bid_size), (bid_price - 0.01, bid_size // 2)],
            "asks": [(ask_price, ask_size), (ask_price + 0.01, ask_size // 2)],
        }

    def test_returns_dict(self):
        result = compute_book_signals(self._book())
        assert isinstance(result, dict)

    def test_expected_keys(self):
        result = compute_book_signals(self._book())
        assert {"obi", "micro_price", "quoted_spread_bps"}.issubset(result.keys())

    def test_obi_in_minus_one_to_one(self):
        result = compute_book_signals(self._book())
        assert -1.0 <= result["obi"] <= 1.0

    def test_obi_positive_when_bid_dominant(self):
        book = {"bids": [(100.0, 2000)], "asks": [(100.02, 500)]}
        result = compute_book_signals(book)
        assert result["obi"] > 0  # more bids = bullish pressure

    def test_obi_negative_when_ask_dominant(self):
        book = {"bids": [(100.0, 300)], "asks": [(100.02, 3000)]}
        result = compute_book_signals(book)
        assert result["obi"] < 0  # more asks = selling pressure

    def test_obi_zero_when_balanced(self):
        book = {"bids": [(100.0, 1000)], "asks": [(100.02, 1000)]}
        result = compute_book_signals(book)
        assert result["obi"] == 0.0

    def test_micro_price_between_bid_ask(self):
        result = compute_book_signals(self._book(bid_price=100.0, ask_price=100.02))
        assert 100.0 <= result["micro_price"] <= 100.02

    def test_micro_price_closer_to_ask_when_bid_dominant(self):
        """More bid volume → micro-price pulls toward ask (queue-weighted)."""
        book = {"bids": [(100.0, 3000)], "asks": [(100.02, 100)]}
        result = compute_book_signals(book)
        mid = (100.0 + 100.02) / 2
        assert result["micro_price"] > mid

    def test_spread_bps_positive(self):
        result = compute_book_signals(self._book())
        assert result["quoted_spread_bps"] > 0

    def test_spread_bps_zero_locked_market(self):
        book = {"bids": [(100.0, 1000)], "asks": [(100.0, 1000)]}
        result = compute_book_signals(book)
        assert result["quoted_spread_bps"] == 0.0

    def test_empty_bids_returns_none(self):
        result = compute_book_signals({"bids": [], "asks": [(100.02, 500)]})
        assert result["obi"] is None

    def test_empty_asks_returns_none(self):
        result = compute_book_signals({"bids": [(100.0, 1000)], "asks": []})
        assert result["obi"] is None

    def test_spread_2_cents_100_dollar_stock(self):
        """$0.02 spread on $100 stock = 2bps."""
        book = {"bids": [(100.00, 500)], "asks": [(100.02, 500)]}
        result = compute_book_signals(book)
        assert abs(result["quoted_spread_bps"] - 2.0) < 0.1
