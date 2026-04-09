"""Tests for 4.7 — Feature Multicollinearity VIF Filter (QS-B6)."""

import numpy as np
import pandas as pd
import pytest

from quantstack.core.validation.orthogonalization import (
    CorrelationFilter,
    FeatureOrthogonalizer,
    VIFFilter,
)


def _make_collinear_df(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Create a DataFrame with one collinear column."""
    rng = np.random.default_rng(seed)
    a = rng.normal(0, 1, n)
    b = rng.normal(0, 1, n)
    c = a + rng.normal(0, 0.01, n)  # nearly identical to A
    d = rng.normal(0, 1, n)
    return pd.DataFrame({"A": a, "B": b, "C_collinear": c, "D": d})


def _make_orthogonal_df(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Create a DataFrame with orthogonal columns."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "X1": rng.normal(0, 1, n),
        "X2": rng.normal(0, 1, n),
        "X3": rng.normal(0, 1, n),
    })


class TestVIFFilter:
    def test_drops_collinear_feature(self):
        """Column C (≈A + noise) should be dropped due to high VIF."""
        df = _make_collinear_df()
        vif = VIFFilter(threshold=10.0)
        result = vif.fit_transform(df)

        # Either A or C should be dropped (they're interchangeable)
        assert result.shape[1] < df.shape[1]
        dropped = set(df.columns) - set(result.columns)
        assert dropped & {"A", "C_collinear"}

    def test_keeps_orthogonal_features(self):
        """Independent columns should all survive VIF filtering."""
        df = _make_orthogonal_df()
        vif = VIFFilter(threshold=10.0)
        result = vif.fit_transform(df)
        assert list(result.columns) == list(df.columns)

    def test_get_result(self):
        """get_result() returns correct counts."""
        df = _make_collinear_df()
        vif = VIFFilter(threshold=10.0)
        vif.fit_transform(df)
        res = vif.get_result()
        assert res.original_features == 4
        assert res.selected_features + len(res.removed_features) == 4

    def test_too_few_rows_keeps_all(self):
        """With too few rows, VIF warns and keeps all features."""
        df = _make_collinear_df(n=3)
        vif = VIFFilter(threshold=10.0)
        result = vif.fit_transform(df)
        assert result.shape[1] == df.shape[1]


class TestFeatureOrthogonalizerWithVIF:
    def test_pipeline_corr_then_vif(self):
        """Full pipeline: correlation filter → VIF filter."""
        df = _make_collinear_df(n=200)
        ortho = FeatureOrthogonalizer(
            correlation_threshold=0.85,
            use_vif_filter=True,
            vif_threshold=10.0,
        )
        result = ortho.fit_transform(df)
        assert result.shape[1] <= df.shape[1]

    def test_pipeline_without_vif(self):
        """Without VIF, only correlation filter runs."""
        df = _make_collinear_df(n=200)
        ortho = FeatureOrthogonalizer(
            correlation_threshold=0.85,
            use_vif_filter=False,
        )
        result = ortho.fit_transform(df)
        # Correlation filter should still drop C_collinear (corr > 0.85 with A)
        assert result.shape[1] < df.shape[1]
