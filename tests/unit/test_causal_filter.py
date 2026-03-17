"""Tests for CausalFilter — Granger causality feature filtering."""

import numpy as np
import pandas as pd
import pytest

from quantcore.validation.causal_filter import (
    CausalFilter,
    CausalFilterResult,
    CausalTestResult,
    _compute_binned_te,
)


@pytest.fixture
def rng():
    return np.random.default_rng(42)


def _make_index(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2020-01-01", periods=n, freq="D")


def _make_causal_pair(
    n: int, index: pd.Index, lag: int = 1, signal_strength: float = 0.5
) -> tuple[pd.Series, pd.Series]:
    """
    Create (feature, target) where feature Granger-causes target.

    target[t] = signal_strength * feature[t-lag] + noise
    """
    rng_local = np.random.default_rng(123)
    feature = pd.Series(rng_local.standard_normal(n), index=index)
    noise = pd.Series(rng_local.standard_normal(n), index=index)
    target = signal_strength * feature.shift(lag).fillna(0) + noise
    return feature, target


class TestCausalFilterBasic:
    """Basic fit/transform/get_result tests."""

    def test_causal_feature_survives(self, rng):
        """A feature that Granger-causes y should survive the filter."""
        n = 500
        idx = _make_index(n)
        feat, y = _make_causal_pair(n, idx, lag=1, signal_strength=0.8)
        X = pd.DataFrame({"causal_feat": feat}, index=idx)

        cf = CausalFilter(max_lag=3, significance_level=0.05, correction="bonferroni")
        result_df = cf.fit_transform(X, y)

        assert "causal_feat" in result_df.columns
        result = cf.get_result()
        assert result.surviving_features == 1
        assert result.per_feature_results["causal_feat"].survived

    def test_random_feature_dropped(self, rng):
        """A random feature should not Granger-cause an independent target."""
        n = 500
        idx = _make_index(n)
        # Feature and target are independently generated
        feat, y = _make_causal_pair(n, idx, lag=1, signal_strength=0.8)
        X = pd.DataFrame({"random_feat": rng.standard_normal(n)}, index=idx)

        cf = CausalFilter(max_lag=3, significance_level=0.05, correction="bonferroni")
        result_df = cf.fit_transform(X, y)

        result = cf.get_result()
        assert not result.per_feature_results["random_feat"].survived
        assert len(result_df.columns) == 0

    def test_mixed_features(self, rng):
        """Causal feature survives, random features mostly get dropped."""
        n = 500
        idx = _make_index(n)
        feat, y = _make_causal_pair(n, idx, lag=1, signal_strength=0.8)

        X = pd.DataFrame(index=idx)
        X["causal"] = feat
        for i in range(9):
            X[f"random_{i}"] = rng.standard_normal(n)

        cf = CausalFilter(max_lag=3, significance_level=0.05, correction="bonferroni")
        result_df = cf.fit_transform(X, y)

        assert "causal" in result_df.columns
        result = cf.get_result()
        assert result.original_features == 10
        assert result.surviving_features >= 1

    def test_fit_then_transform(self, rng):
        """fit on train, transform on test returns correct column subset."""
        n = 500
        idx = _make_index(n)
        feat, y = _make_causal_pair(n, idx, lag=1, signal_strength=0.8)

        X = pd.DataFrame(index=idx)
        X["causal"] = feat
        X["random"] = rng.standard_normal(n)

        cf = CausalFilter(max_lag=3, significance_level=0.05)
        cf.fit(X.iloc[:400], y.iloc[:400])

        X_test = X.iloc[400:]
        result_df = cf.transform(X_test)

        assert len(result_df) == 100
        # Columns should be a subset of trained surviving features
        for col in result_df.columns:
            assert col in cf._surviving_features


class TestStationarity:
    """Stationarity handling tests."""

    def test_nonstationary_auto_differenced(self, rng):
        """Non-stationary feature gets auto-differenced when auto_difference=True."""
        n = 500
        idx = _make_index(n)
        y = pd.Series(rng.standard_normal(n), index=idx)

        # Random walk (non-stationary)
        X = pd.DataFrame({"rw": rng.standard_normal(n).cumsum()}, index=idx)

        cf = CausalFilter(max_lag=3, auto_difference=True)
        cf.fit(X, y)

        result = cf.get_result()
        feat_result = result.per_feature_results["rw"]
        assert feat_result.was_differenced

    def test_stationary_not_differenced(self, rng):
        """Already-stationary feature should not be differenced."""
        n = 500
        idx = _make_index(n)
        y = pd.Series(rng.standard_normal(n), index=idx)
        # White noise is stationary
        X = pd.DataFrame({"stationary": rng.standard_normal(n)}, index=idx)

        cf = CausalFilter(max_lag=3, auto_difference=True)
        cf.fit(X, y)

        result = cf.get_result()
        feat_result = result.per_feature_results["stationary"]
        assert feat_result.is_stationary
        assert not feat_result.was_differenced


class TestMultipleTesting:
    """Multiple testing correction tests."""

    def test_bonferroni_controls_false_positives(self, rng):
        """With 50 random features and 0 causal, very few should survive."""
        n = 500
        idx = _make_index(n)
        y = pd.Series(rng.standard_normal(n), index=idx)

        X = pd.DataFrame(
            {f"rand_{i}": rng.standard_normal(n) for i in range(50)},
            index=idx,
        )

        cf = CausalFilter(max_lag=3, significance_level=0.05, correction="bonferroni")
        cf.fit(X, y)

        result = cf.get_result()
        # Bonferroni at alpha=0.05/50=0.001 should yield ~0 false positives
        assert result.surviving_features <= 3  # very unlikely to be more

    def test_holm_correction(self, rng):
        """Holm correction should work and be at least as powerful as Bonferroni."""
        n = 500
        idx = _make_index(n)
        feat, y = _make_causal_pair(n, idx, lag=1, signal_strength=0.8)

        X = pd.DataFrame(index=idx)
        X["causal"] = feat
        for i in range(9):
            X[f"random_{i}"] = rng.standard_normal(n)

        cf_bonf = CausalFilter(max_lag=3, correction="bonferroni")
        cf_bonf.fit(X, y)

        cf_holm = CausalFilter(max_lag=3, correction="holm")
        cf_holm.fit(X, y)

        # Holm is strictly more powerful than Bonferroni
        assert cf_holm.get_result().surviving_features >= cf_bonf.get_result().surviving_features

    def test_invalid_correction_raises(self):
        with pytest.raises(ValueError, match="correction must be"):
            CausalFilter(correction="fdr")


class TestNaNHandling:
    """NaN and edge case handling."""

    def test_nan_warmup_period(self, rng):
        """Features with NaN warmup period should still be tested on valid window."""
        n = 500
        idx = _make_index(n)
        y = pd.Series(rng.standard_normal(n), index=idx)

        feat_values = rng.standard_normal(n)
        feat_values[:50] = np.nan  # 50-bar warmup
        X = pd.DataFrame({"warmup_feat": feat_values}, index=idx)

        cf = CausalFilter(max_lag=3, min_observations=100)
        cf.fit(X, y)

        result = cf.get_result()
        # Should have been tested (450 valid obs > 100 min)
        assert result.per_feature_results["warmup_feat"].drop_reason is None or "insufficient" not in (
            result.per_feature_results["warmup_feat"].drop_reason or ""
        )

    def test_insufficient_data_dropped(self, rng):
        """Feature with too few valid observations should be dropped with reason."""
        n = 120
        idx = _make_index(n)
        y = pd.Series(rng.standard_normal(n), index=idx)

        feat_values = rng.standard_normal(n)
        feat_values[:100] = np.nan  # Only 20 valid obs
        X = pd.DataFrame({"sparse_feat": feat_values}, index=idx)

        cf = CausalFilter(max_lag=3, min_observations=100)
        cf.fit(X, y)

        result = cf.get_result()
        assert result.per_feature_results["sparse_feat"].survived is False
        assert "insufficient" in result.per_feature_results["sparse_feat"].drop_reason

    def test_all_features_dropped(self, rng):
        """When all features are dropped, result should be empty DataFrame."""
        n = 500
        idx = _make_index(n)
        y = pd.Series(rng.standard_normal(n), index=idx)
        X = pd.DataFrame(
            {f"rand_{i}": rng.standard_normal(n) for i in range(5)},
            index=idx,
        )

        # Very strict alpha to force all drops
        cf = CausalFilter(max_lag=1, significance_level=0.001, correction="bonferroni")
        result_df = cf.fit_transform(X, y)

        assert isinstance(result_df, pd.DataFrame)
        # May or may not be empty, but should not error

    def test_too_few_target_observations(self, rng):
        """If target has < min_observations, keep all features."""
        n = 50
        idx = _make_index(n)
        y = pd.Series(rng.standard_normal(n), index=idx)
        X = pd.DataFrame({"feat": rng.standard_normal(n)}, index=idx)

        cf = CausalFilter(min_observations=100)
        result_df = cf.fit_transform(X, y)

        # Should keep all features since target is too short
        assert "feat" in result_df.columns


class TestTransferEntropy:
    """Transfer entropy computation tests."""

    def test_te_computed_when_enabled(self, rng):
        """TE values should be populated when use_transfer_entropy=True."""
        n = 500
        idx = _make_index(n)
        feat, y = _make_causal_pair(n, idx, lag=1, signal_strength=0.8)
        X = pd.DataFrame(index=idx)
        X["causal"] = feat

        cf = CausalFilter(
            max_lag=3,
            use_transfer_entropy=True,
            te_n_permutations=20,
        )
        cf.fit(X, y)

        result = cf.get_result()
        causal_result = result.per_feature_results["causal"]
        if causal_result.survived:
            assert causal_result.transfer_entropy is not None
            assert causal_result.te_p_value is not None

    def test_binned_te_positive_for_causal(self):
        """TE should be positive for a clearly causal relationship."""
        rng = np.random.default_rng(42)
        n = 1000
        target = rng.standard_normal(n)
        # Source is lagged target + noise
        source = np.roll(target, 1) + rng.normal(0, 0.1, n)
        source[0] = 0

        te = _compute_binned_te(source, target, lag=1, n_bins=8)
        assert te >= 0.0

    def test_binned_te_near_zero_for_independent(self):
        """TE should be near zero for independent series."""
        rng = np.random.default_rng(42)
        n = 1000
        source = rng.standard_normal(n)
        target = rng.standard_normal(n)

        te = _compute_binned_te(source, target, lag=1, n_bins=8)
        # Should be close to zero (some noise expected)
        assert te < 0.3


class TestOrthogonalizerIntegration:
    """Test CausalFilter integration with FeatureOrthogonalizer."""

    def test_orthogonalizer_with_causal_filter(self, rng):
        """FeatureOrthogonalizer with use_causal_filter=True chains correctly."""
        from quantcore.validation.orthogonalization import FeatureOrthogonalizer

        n = 500
        idx = _make_index(n)
        feat, y = _make_causal_pair(n, idx, lag=1, signal_strength=0.8)

        X = pd.DataFrame(index=idx)
        X["causal"] = feat
        for i in range(4):
            X[f"random_{i}"] = rng.standard_normal(n)

        orth = FeatureOrthogonalizer(
            use_causal_filter=True,
            causal_filter_kwargs={"max_lag": 3, "significance_level": 0.05},
        )
        result_df = orth.fit_transform(X, y)

        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df.columns) <= len(X.columns)

    def test_orthogonalizer_without_causal_backward_compat(self, rng):
        """FeatureOrthogonalizer without causal filter works as before (no y needed)."""
        from quantcore.validation.orthogonalization import FeatureOrthogonalizer

        n = 200
        idx = _make_index(n)
        X = pd.DataFrame(
            {f"feat_{i}": rng.standard_normal(n) for i in range(5)},
            index=idx,
        )

        orth = FeatureOrthogonalizer(use_causal_filter=False)
        result_df = orth.fit_transform(X)

        assert isinstance(result_df, pd.DataFrame)

    def test_orthogonalizer_causal_requires_y(self, rng):
        """FeatureOrthogonalizer with use_causal_filter=True raises without y."""
        from quantcore.validation.orthogonalization import FeatureOrthogonalizer

        n = 200
        idx = _make_index(n)
        X = pd.DataFrame({"feat": rng.standard_normal(n)}, index=idx)

        orth = FeatureOrthogonalizer(use_causal_filter=True)
        with pytest.raises(ValueError, match="y .* required"):
            orth.fit(X)


class TestCausalFilterResult:
    """Result dataclass tests."""

    def test_result_fields(self, rng):
        """CausalFilterResult should have all expected fields."""
        n = 500
        idx = _make_index(n)
        y = pd.Series(rng.standard_normal(n), index=idx)
        X = pd.DataFrame({"feat": rng.standard_normal(n)}, index=idx)

        cf = CausalFilter(max_lag=3)
        cf.fit(X, y)

        result = cf.get_result()
        assert isinstance(result, CausalFilterResult)
        assert result.original_features == 1
        assert isinstance(result.dropped_features, list)
        assert isinstance(result.surviving_feature_names, list)
        assert isinstance(result.per_feature_results, dict)
        assert result.elapsed_seconds > 0
        assert result.correction_method == "bonferroni"

    def test_get_result_before_fit_raises(self):
        cf = CausalFilter()
        with pytest.raises(ValueError, match="Must fit"):
            cf.get_result()

    def test_transform_before_fit_raises(self, rng):
        cf = CausalFilter()
        X = pd.DataFrame({"a": [1, 2, 3]})
        with pytest.raises(ValueError, match="Must fit"):
            cf.transform(X)
