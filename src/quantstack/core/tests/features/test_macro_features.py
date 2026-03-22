# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for FRED-sourced macro features."""

import numpy as np
import pandas as pd

from quantstack.core.features.macro_features import (
    CopperGoldRatio,
    CreditSpreadFeatures,
    DXYMomentum,
    EquityBondCorrelation,
    MOVEIndex,
    RealYieldFeatures,
    VolOfVol,
)


def _series(
    n: int = 300, base: float = 2.0, noise: float = 0.1, seed: int = 42
) -> pd.Series:
    np.random.seed(seed)
    dates = pd.date_range("2022-01-01", periods=n, freq="D")
    return pd.Series(base + np.cumsum(np.random.randn(n) * noise), index=dates)


def _close(n: int = 300, seed: int = 42) -> pd.Series:
    np.random.seed(seed)
    dates = pd.date_range("2022-01-01", periods=n, freq="D")
    return pd.Series(100 * np.cumprod(1 + np.random.randn(n) * 0.01), index=dates)


class TestRealYieldFeatures:
    def test_returns_dataframe(self):
        ry = _series(base=1.5, noise=0.05)
        be = _series(base=2.2, noise=0.03, seed=99)
        result = RealYieldFeatures().compute(ry, be)
        assert isinstance(result, pd.DataFrame)

    def test_required_columns(self):
        ry = _series(base=1.5)
        be = _series(base=2.2, seed=99)
        result = RealYieldFeatures().compute(ry, be)
        for col in (
            "real_yield_10y",
            "breakeven_10y",
            "real_yield_momentum",
            "breakeven_momentum",
            "growth_value_signal",
        ):
            assert col in result.columns

    def test_growth_value_signal_values(self):
        ry = _series(base=1.5)
        be = _series(base=2.2, seed=99)
        result = RealYieldFeatures().compute(ry, be)
        assert set(result["growth_value_signal"].unique()).issubset({-1, 0, 1})


class TestCreditSpreadFeatures:
    def test_returns_dataframe(self):
        oas = _series(base=4.0, noise=0.2)
        result = CreditSpreadFeatures().compute(oas)
        assert isinstance(result, pd.DataFrame)

    def test_required_columns(self):
        oas = _series(base=4.0, noise=0.2)
        result = CreditSpreadFeatures().compute(oas)
        for col in ("hy_oas", "hy_oas_zscore", "hy_oas_momentum", "credit_regime"):
            assert col in result.columns

    def test_regime_values(self):
        oas = _series(base=4.0, noise=0.2)
        result = CreditSpreadFeatures().compute(oas)
        assert set(result["credit_regime"].unique()).issubset(
            {"TIGHT", "NORMAL", "WIDE"}
        )


class TestCopperGoldRatio:
    def test_returns_dataframe(self):
        cu = _series(base=4.0, noise=0.1)
        au = _series(base=1800.0, noise=10.0, seed=99)
        result = CopperGoldRatio().compute(cu, au)
        assert isinstance(result, pd.DataFrame)

    def test_ratio_positive(self):
        cu = _series(base=4.0, noise=0.01)
        au = _series(base=1800.0, noise=1.0, seed=99)
        result = CopperGoldRatio().compute(cu, au)
        valid = result["copper_gold_ratio"].dropna()
        assert (valid > 0).all()

    def test_regime_values(self):
        cu = _series(base=4.0, noise=0.1)
        au = _series(base=1800.0, noise=10.0, seed=99)
        result = CopperGoldRatio().compute(cu, au)
        assert set(result["cg_regime"].unique()).issubset(
            {"EXPANSION", "NEUTRAL", "CONTRACTION"}
        )


class TestDXYMomentum:
    def test_returns_dataframe(self):
        dxy = _series(base=103.0, noise=0.3)
        result = DXYMomentum().compute(dxy)
        assert isinstance(result, pd.DataFrame)

    def test_regime_values(self):
        dxy = _series(base=103.0, noise=0.3)
        result = DXYMomentum().compute(dxy)
        assert set(result["dxy_regime"].unique()).issubset(
            {"STRONG", "NEUTRAL", "WEAK"}
        )


class TestMOVEIndex:
    def test_returns_dataframe(self):
        move = _series(base=90.0, noise=5.0)
        result = MOVEIndex().compute(move)
        assert isinstance(result, pd.DataFrame)

    def test_elevated_binary(self):
        move = _series(base=90.0, noise=5.0)
        result = MOVEIndex().compute(move)
        vals = result["move_elevated"].unique()
        assert set(vals).issubset({0, 1})

    def test_elevated_fires_above_threshold(self):
        n = 100
        dates = pd.date_range("2022-01-01", periods=n, freq="D")
        move = pd.Series(np.full(n, 120.0), index=dates)  # always above 100
        result = MOVEIndex().compute(move)
        assert result["move_elevated"].sum() == n


class TestEquityBondCorrelation:
    def test_returns_dataframe(self):
        eq = _close().pct_change()
        bd = _close(seed=77).pct_change()
        result = EquityBondCorrelation().compute(eq, bd)
        assert isinstance(result, pd.DataFrame)

    def test_corr_bounded(self):
        eq = _close().pct_change()
        bd = _close(seed=77).pct_change()
        result = EquityBondCorrelation().compute(eq, bd)
        valid = result["eq_bond_corr_60d"].dropna()
        assert valid.min() >= -1.01 and valid.max() <= 1.01

    def test_regime_values(self):
        eq = _close().pct_change()
        bd = _close(seed=77).pct_change()
        result = EquityBondCorrelation().compute(eq, bd)
        assert set(result["eq_bond_corr_regime"].unique()).issubset(
            {"NORMAL", "CRISIS"}
        )


class TestVolOfVol:
    def test_returns_dataframe(self):
        cl = _close()
        result = VolOfVol().compute(cl)
        assert isinstance(result, pd.DataFrame)

    def test_required_columns(self):
        cl = _close()
        result = VolOfVol().compute(cl)
        for col in ("vov", "vov_zscore", "vov_spike"):
            assert col in result.columns

    def test_vov_non_negative(self):
        cl = _close()
        result = VolOfVol().compute(cl)
        valid = result["vov"].dropna()
        assert (valid >= 0).all()

    def test_spike_binary(self):
        cl = _close()
        result = VolOfVol().compute(cl)
        vals = result["vov_spike"].unique()
        assert set(vals).issubset({0, 1})
