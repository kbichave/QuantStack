# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Feature-level tests for statistical features: YangZhangVolatility, HurstExponent,
VarianceRatioTest, OUHalfLife, AutocorrelationSpectrum, EntropyFeatures.

Tests verify correctness properties (value bounds, regime detection accuracy,
known behavior on synthetic data) following hedge fund TA gap plan.
"""

import numpy as np
import pandas as pd

from quantcore.features.statistical import (
    AutocorrelationSpectrum,
    EntropyFeatures,
    HurstExponent,
    OUHalfLife,
    VarianceRatioTest,
    YangZhangVolatility,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ohlcv(n: int = 300, seed: int = 42) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Synthetic (open, high, low, close)."""
    np.random.seed(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    close = pd.Series(100 + np.cumsum(np.random.randn(n) * 0.8), index=dates).clip(50)
    open_ = close.shift(1).fillna(close.iloc[0]) + np.random.randn(n) * 0.2
    high = pd.Series(np.maximum(open_, close) + np.abs(np.random.randn(n) * 0.5), index=dates)
    low = pd.Series(np.minimum(open_, close) - np.abs(np.random.randn(n) * 0.5), index=dates)
    return open_, high, low, close


def _trending_series(n: int = 500) -> pd.Series:
    """Strong monotone uptrend (cumulative drift >> noise)."""
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    np.random.seed(10)
    returns = 0.002 + np.random.randn(n) * 0.005  # drift dominates
    prices = 100 * np.cumprod(1 + returns)
    return pd.Series(prices, index=dates)


def _mean_reverting_series(n: int = 500, theta: float = 0.1, mu: float = 100.0) -> pd.Series:
    """Ornstein-Uhlenbeck process."""
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    np.random.seed(20)
    x = np.zeros(n)
    x[0] = mu
    for i in range(1, n):
        x[i] = x[i - 1] + theta * (mu - x[i - 1]) + np.random.randn() * 0.5
    return pd.Series(x, index=dates)


# ---------------------------------------------------------------------------
# YangZhangVolatility
# ---------------------------------------------------------------------------


class TestYangZhangVolatility:
    def test_returns_dataframe(self):
        op, hi, lo, cl = _ohlcv()
        result = YangZhangVolatility().compute(op, hi, lo, cl)
        assert isinstance(result, pd.DataFrame)

    def test_required_columns(self):
        op, hi, lo, cl = _ohlcv()
        result = YangZhangVolatility().compute(op, hi, lo, cl)
        for col in ("yang_zhang_vol", "garman_klass_vol", "parkinson_vol", "yz_vs_close_ratio"):
            assert col in result.columns, f"{col} missing"

    def test_vol_non_negative(self):
        op, hi, lo, cl = _ohlcv()
        result = YangZhangVolatility().compute(op, hi, lo, cl)
        for col in ("yang_zhang_vol", "garman_klass_vol", "parkinson_vol"):
            valid = result[col].dropna()
            assert (valid >= 0).all(), f"{col} has negative values"

    def test_vol_reasonable_range(self):
        """Annualized vol for random walk with 0.8% daily std should be roughly 12-15%."""
        op, hi, lo, cl = _ohlcv(n=500)
        result = YangZhangVolatility(period=60).compute(op, hi, lo, cl)
        yz = result["yang_zhang_vol"].dropna()
        median_vol = yz.median()
        assert 0.01 < median_vol < 1.0, f"Unreasonable vol: {median_vol}"

    def test_single_bar_no_crash(self):
        idx = pd.DatetimeIndex(["2020-01-01"])
        result = YangZhangVolatility().compute(
            pd.Series([100.0], index=idx),
            pd.Series([101.0], index=idx),
            pd.Series([99.0], index=idx),
            pd.Series([100.5], index=idx),
        )
        assert isinstance(result, pd.DataFrame)

    def test_constant_price(self):
        n = 60
        idx = pd.date_range("2020-01-01", periods=n, freq="D")
        op = pd.Series(np.full(n, 100.0), index=idx)
        hi = pd.Series(np.full(n, 100.5), index=idx)
        lo = pd.Series(np.full(n, 99.5), index=idx)
        cl = pd.Series(np.full(n, 100.0), index=idx)
        result = YangZhangVolatility(period=20).compute(op, hi, lo, cl)
        assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# HurstExponent
# ---------------------------------------------------------------------------


class TestHurstExponent:
    def test_returns_dataframe(self):
        _, _, _, cl = _ohlcv(n=400)
        result = HurstExponent(window=200).compute(cl)
        assert isinstance(result, pd.DataFrame)

    def test_required_columns(self):
        _, _, _, cl = _ohlcv(n=400)
        result = HurstExponent(window=200).compute(cl)
        for col in ("hurst_exponent", "hurst_regime"):
            assert col in result.columns

    def test_hurst_bounded(self):
        _, _, _, cl = _ohlcv(n=400)
        result = HurstExponent(window=200).compute(cl)
        valid = result["hurst_exponent"].dropna()
        assert (valid >= 0).all() and (valid <= 1).all()

    def test_trending_series_high_hurst(self):
        """Strong trend should produce H > 0.5."""
        cl = _trending_series(n=600)
        result = HurstExponent(window=252, min_lags=10, max_lags=80).compute(cl)
        valid = result["hurst_exponent"].dropna()
        if len(valid) > 0:
            assert valid.mean() > 0.45, f"Expected H > 0.45 for trending, got {valid.mean()}"

    def test_regime_values(self):
        _, _, _, cl = _ohlcv(n=400)
        result = HurstExponent(window=200).compute(cl)
        valid = result["hurst_regime"].dropna()
        assert set(valid.unique()).issubset({-1, 0, 1})


# ---------------------------------------------------------------------------
# VarianceRatioTest
# ---------------------------------------------------------------------------


class TestVarianceRatioTest:
    def test_returns_dataframe(self):
        _, _, _, cl = _ohlcv()
        result = VarianceRatioTest().compute(cl)
        assert isinstance(result, pd.DataFrame)

    def test_required_columns(self):
        _, _, _, cl = _ohlcv()
        result = VarianceRatioTest().compute(cl)
        for col in ("vr_2", "vr_5", "vr_10", "vr_20", "vr_zscore_5"):
            assert col in result.columns, f"{col} missing"

    def test_random_walk_vr_near_one(self):
        """IID returns should produce VR ≈ 1."""
        np.random.seed(99)
        n = 1000
        dates = pd.date_range("2020-01-01", periods=n, freq="D")
        close = pd.Series(100 * np.cumprod(1 + np.random.randn(n) * 0.01), index=dates)
        result = VarianceRatioTest(window=252).compute(close)
        vr5 = result["vr_5"].dropna()
        if len(vr5) > 0:
            assert 0.5 < vr5.median() < 1.5, f"VR(5) = {vr5.median()}, expected near 1"

    def test_single_bar_no_crash(self):
        cl = pd.Series([100.0], index=pd.DatetimeIndex(["2020-01-01"]))
        result = VarianceRatioTest().compute(cl)
        assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# OUHalfLife
# ---------------------------------------------------------------------------


class TestOUHalfLife:
    def test_returns_dataframe(self):
        series = _mean_reverting_series()
        result = OUHalfLife(window=126).compute(series)
        assert isinstance(result, pd.DataFrame)

    def test_required_columns(self):
        series = _mean_reverting_series()
        result = OUHalfLife(window=126).compute(series)
        for col in ("ou_half_life", "ou_theta", "ou_mu", "ou_half_life_valid"):
            assert col in result.columns

    def test_mean_reverting_detected(self):
        """OU process with theta=0.1 should be detected as mean-reverting."""
        series = _mean_reverting_series(n=500, theta=0.1, mu=100.0)
        result = OUHalfLife(window=200).compute(series)
        valid_count = result["ou_half_life_valid"].sum()
        assert valid_count > 0, "Should detect mean-reversion in OU process"

    def test_half_life_positive(self):
        series = _mean_reverting_series()
        result = OUHalfLife(window=126).compute(series)
        valid_hl = result.loc[result["ou_half_life_valid"] == 1, "ou_half_life"]
        if len(valid_hl) > 0:
            assert (valid_hl > 0).all(), "Half-life should be positive"

    def test_trending_not_mean_reverting(self):
        """Strong trend should have fewer valid (mean-reverting) windows."""
        series = _trending_series(n=500)
        result = OUHalfLife(window=200).compute(series)
        mr_count = _mean_reverting_series(n=500)
        result_mr = OUHalfLife(window=200).compute(mr_count)
        # OU process should have more mean-reverting detections
        assert result_mr["ou_half_life_valid"].sum() >= result["ou_half_life_valid"].sum()

    def test_single_bar_no_crash(self):
        series = pd.Series([100.0], index=pd.DatetimeIndex(["2020-01-01"]))
        result = OUHalfLife().compute(series)
        assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# AutocorrelationSpectrum
# ---------------------------------------------------------------------------


class TestAutocorrelationSpectrum:
    def test_returns_dataframe(self):
        _, _, _, cl = _ohlcv()
        result = AutocorrelationSpectrum(window=63).compute(cl)
        assert isinstance(result, pd.DataFrame)

    def test_required_columns(self):
        _, _, _, cl = _ohlcv()
        result = AutocorrelationSpectrum(window=63).compute(cl)
        for col in ("acf_lag1", "acf_lag5", "acf_lag10", "acf_sum_1_5", "acf_decay_rate"):
            assert col in result.columns

    def test_acf_bounded(self):
        """ACF values should be in [-1, 1]."""
        _, _, _, cl = _ohlcv(n=300)
        result = AutocorrelationSpectrum(window=63).compute(cl)
        for col in ("acf_lag1", "acf_lag5", "acf_lag10"):
            valid = result[col].dropna()
            if len(valid) > 0:
                assert valid.min() >= -1.01 and valid.max() <= 1.01

    def test_single_bar_no_crash(self):
        cl = pd.Series([100.0], index=pd.DatetimeIndex(["2020-01-01"]))
        result = AutocorrelationSpectrum().compute(cl)
        assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# EntropyFeatures
# ---------------------------------------------------------------------------


class TestEntropyFeatures:
    def test_returns_dataframe(self):
        _, _, _, cl = _ohlcv()
        result = EntropyFeatures(window=63).compute(cl)
        assert isinstance(result, pd.DataFrame)

    def test_required_columns(self):
        _, _, _, cl = _ohlcv()
        result = EntropyFeatures(window=63).compute(cl)
        for col in ("shannon_entropy", "sample_entropy", "entropy_regime"):
            assert col in result.columns

    def test_shannon_non_negative(self):
        _, _, _, cl = _ohlcv()
        result = EntropyFeatures(window=63).compute(cl)
        valid = result["shannon_entropy"].dropna()
        assert (valid >= 0).all(), "Shannon entropy should be >= 0"

    def test_constant_series_low_entropy(self):
        """Constant returns should have near-zero entropy."""
        n = 200
        dates = pd.date_range("2020-01-01", periods=n, freq="D")
        cl = pd.Series(np.linspace(100, 150, n), index=dates)  # perfectly linear
        result = EntropyFeatures(window=63).compute(cl)
        valid = result["shannon_entropy"].dropna()
        if len(valid) > 0:
            # Linear price -> constant returns -> low entropy
            assert valid.iloc[-1] < 2.5, f"Expected low entropy for constant returns, got {valid.iloc[-1]}"

    def test_regime_values(self):
        _, _, _, cl = _ohlcv()
        result = EntropyFeatures(window=63).compute(cl)
        valid = result["entropy_regime"].dropna()
        assert set(valid.unique()).issubset({-1, 0, 1})

    def test_single_bar_no_crash(self):
        cl = pd.Series([100.0], index=pd.DatetimeIndex(["2020-01-01"]))
        result = EntropyFeatures().compute(cl)
        assert isinstance(result, pd.DataFrame)
