# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for institutional momentum factors and cross-sectional dispersion."""

import numpy as np
import pandas as pd

from quantstack.core.features.momentum_factors import (
    CrossSectionalDispersion,
    InstitutionalMomentumFactors,
)


def _close(n: int = 400, seed: int = 42) -> pd.Series:
    np.random.seed(seed)
    dates = pd.date_range("2019-01-01", periods=n, freq="D")
    return pd.Series(100 * np.cumprod(1 + np.random.randn(n) * 0.01), index=dates)


class TestInstitutionalMomentumFactors:
    def test_returns_dataframe(self):
        cl = _close()
        result = InstitutionalMomentumFactors().compute_single(cl)
        assert isinstance(result, pd.DataFrame)

    def test_required_columns(self):
        cl = _close()
        result = InstitutionalMomentumFactors().compute_single(cl)
        for col in (
            "mom_12_1",
            "mom_7_1",
            "residual_momentum",
            "vol_adjusted_momentum",
        ):
            assert col in result.columns

    def test_residual_nan_without_market(self):
        cl = _close()
        result = InstitutionalMomentumFactors().compute_single(cl)
        assert result["residual_momentum"].isna().all()

    def test_residual_with_market(self):
        cl = _close(seed=42)
        mkt = _close(seed=99)
        result = InstitutionalMomentumFactors().compute_single(cl, market_close=mkt)
        valid = result["residual_momentum"].dropna()
        assert len(valid) > 0

    def test_vol_adjusted_finite(self):
        cl = _close()
        result = InstitutionalMomentumFactors().compute_single(cl)
        valid = result["vol_adjusted_momentum"].dropna()
        assert np.isfinite(valid).all()

    def test_uptrend_positive_momentum(self):
        n = 400
        dates = pd.date_range("2019-01-01", periods=n, freq="D")
        cl = pd.Series(np.linspace(100, 200, n), index=dates)
        result = InstitutionalMomentumFactors().compute_single(cl)
        mom = result["mom_12_1"].dropna()
        assert mom.iloc[-1] > 0

    def test_cross_section_ranks(self):
        closes = pd.DataFrame(
            {
                "A": _close(seed=1),
                "B": _close(seed=2),
                "C": _close(seed=3),
            }
        )
        result = InstitutionalMomentumFactors().compute_cross_section(closes)
        assert "A_cs_rank" in result.columns
        valid = result["A_cs_rank"].dropna()
        if len(valid) > 0:
            assert (valid >= 0).all() and (valid <= 1).all()

    def test_single_bar_no_crash(self):
        cl = pd.Series([100.0], index=pd.DatetimeIndex(["2020-01-01"]))
        result = InstitutionalMomentumFactors().compute_single(cl)
        assert isinstance(result, pd.DataFrame)


class TestCrossSectionalDispersion:
    def _universe(self, n_symbols: int = 10, n_bars: int = 200) -> dict[str, pd.Series]:
        dates = pd.date_range("2020-01-01", periods=n_bars, freq="D")
        return {
            f"SYM{i}": pd.Series(
                100 * np.cumprod(1 + np.random.RandomState(i).randn(n_bars) * 0.01),
                index=dates,
            )
            for i in range(n_symbols)
        }

    def test_returns_dataframe(self):
        result = CrossSectionalDispersion().compute(self._universe())
        assert isinstance(result, pd.DataFrame)

    def test_required_columns(self):
        result = CrossSectionalDispersion().compute(self._universe())
        for col in ("cs_dispersion", "cs_dispersion_zscore", "cs_correlation_mean"):
            assert col in result.columns

    def test_dispersion_non_negative(self):
        result = CrossSectionalDispersion().compute(self._universe())
        valid = result["cs_dispersion"].dropna()
        assert (valid >= 0).all()

    def test_correlation_bounded(self):
        result = CrossSectionalDispersion().compute(self._universe())
        valid = result["cs_correlation_mean"].dropna()
        if len(valid) > 0:
            assert valid.min() >= -1.01 and valid.max() <= 1.01

    def test_too_few_symbols_returns_nan(self):
        dates = pd.date_range("2020-01-01", periods=50, freq="D")
        closes = {"A": pd.Series(np.ones(50) * 100, index=dates)}
        result = CrossSectionalDispersion(min_symbols=5).compute(closes)
        assert result["cs_dispersion"].isna().all()
