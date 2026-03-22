"""
Causal feature filtering via Granger causality and transfer entropy.

Drops features that fail to Granger-cause forward returns,
reducing spurious associations and improving OOS performance.
"""

import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from loguru import logger
from statsmodels.tsa.stattools import grangercausalitytests

from quantstack.core.research.stat_tests import adf_test


@dataclass
class CausalTestResult:
    """Result of a single feature's causal test."""

    feature_name: str
    granger_p_value: float
    granger_f_stat: float
    best_lag: int
    is_stationary: bool
    was_differenced: bool
    transfer_entropy: float | None = None
    te_p_value: float | None = None
    survived: bool = False
    drop_reason: str | None = None


@dataclass
class CausalFilterResult:
    """Aggregate result from CausalFilter."""

    original_features: int
    surviving_features: int
    dropped_features: list[str]
    surviving_feature_names: list[str]
    per_feature_results: dict[str, CausalTestResult] = field(default_factory=dict)
    alpha_used: float = 0.05
    correction_method: str = "bonferroni"
    elapsed_seconds: float = 0.0


def _compute_binned_te(
    source: np.ndarray,
    target: np.ndarray,
    lag: int,
    n_bins: int = 8,
) -> float:
    """
    Compute transfer entropy TE(source -> target) using binned estimator.

    TE = H(target_t | target_{t-lag}) - H(target_t | target_{t-lag}, source_{t-lag})

    Args:
        source: Source time series (feature)
        target: Target time series (returns)
        lag: Time lag
        n_bins: Number of bins for discretization

    Returns:
        Transfer entropy in nats
    """
    n = len(source) - lag
    if n < 20:
        return 0.0

    # Discretize into equal-frequency bins
    src_lagged = pd.qcut(source[:-lag], n_bins, labels=False, duplicates="drop")
    tgt_lagged = pd.qcut(target[:-lag], n_bins, labels=False, duplicates="drop")
    tgt_current = pd.qcut(target[lag:], n_bins, labels=False, duplicates="drop")

    # Align lengths after qcut (handles edge cases)
    min_len = min(len(src_lagged), len(tgt_lagged), len(tgt_current))
    src_lagged = src_lagged[:min_len]
    tgt_lagged = tgt_lagged[:min_len]
    tgt_current = tgt_current[:min_len]

    # Joint and marginal counts
    # TE = sum p(y_t, y_{t-k}, x_{t-k}) * log[ p(y_t | y_{t-k}, x_{t-k}) / p(y_t | y_{t-k}) ]
    joint_xyz = {}  # (y_t, y_{t-k}, x_{t-k}) -> count
    joint_yz = {}  # (y_t, y_{t-k}) -> count
    marginal_yz_x = {}  # (y_{t-k}, x_{t-k}) -> count
    marginal_y = {}  # (y_{t-k},) -> count

    for i in range(min_len):
        yt = tgt_current[i]
        ytk = tgt_lagged[i]
        xtk = src_lagged[i]

        # Skip NaN bins
        if np.isnan(yt) or np.isnan(ytk) or np.isnan(xtk):
            continue

        key_xyz = (yt, ytk, xtk)
        key_yz = (yt, ytk)
        key_yz_x = (ytk, xtk)
        key_y = (ytk,)

        joint_xyz[key_xyz] = joint_xyz.get(key_xyz, 0) + 1
        joint_yz[key_yz] = joint_yz.get(key_yz, 0) + 1
        marginal_yz_x[key_yz_x] = marginal_yz_x.get(key_yz_x, 0) + 1
        marginal_y[key_y] = marginal_y.get(key_y, 0) + 1

    total = sum(joint_xyz.values())
    if total == 0:
        return 0.0

    te = 0.0
    for (yt, ytk, xtk), count_xyz in joint_xyz.items():
        p_xyz = count_xyz / total
        p_yt_given_ytk_xtk = count_xyz / marginal_yz_x.get((ytk, xtk), 1)
        p_yt_given_ytk = joint_yz.get((yt, ytk), 1) / marginal_y.get((ytk,), 1)

        if p_yt_given_ytk > 0 and p_yt_given_ytk_xtk > 0:
            te += p_xyz * np.log(p_yt_given_ytk_xtk / p_yt_given_ytk)

    return max(te, 0.0)


def _te_permutation_test(
    source: np.ndarray,
    target: np.ndarray,
    lag: int,
    n_permutations: int,
    n_bins: int = 8,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """
    Permutation test for transfer entropy significance.

    Returns:
        (observed_te, p_value)
    """
    if rng is None:
        rng = np.random.default_rng(42)

    observed_te = _compute_binned_te(source, target, lag, n_bins)

    surpass_count = 0
    for _ in range(n_permutations):
        shuffled_source = rng.permutation(source)
        shuffled_te = _compute_binned_te(shuffled_source, target, lag, n_bins)
        if shuffled_te >= observed_te:
            surpass_count += 1

    p_value = (surpass_count + 1) / (n_permutations + 1)
    return observed_te, p_value


class CausalFilter:
    """
    Filter features by Granger causality on forward returns.

    Tests whether each feature Granger-causes the target (forward returns)
    after checking stationarity. Applies Bonferroni or Holm correction
    for multiple testing. Optionally computes transfer entropy for
    nonlinear causality detection.

    Follows sklearn-style fit/transform/fit_transform API.
    """

    def __init__(
        self,
        max_lag: int = 5,
        significance_level: float = 0.05,
        correction: str = "bonferroni",
        use_transfer_entropy: bool = False,
        te_n_permutations: int = 100,
        stationarity_alpha: float = 0.05,
        auto_difference: bool = True,
        n_jobs: int = 1,
        min_observations: int = 100,
    ):
        """
        Initialize causal filter.

        Args:
            max_lag: Maximum lag for Granger test (5 = 1 week on D1)
            significance_level: Pre-correction significance level
            correction: Multiple testing correction ("bonferroni" or "holm")
            use_transfer_entropy: Enable nonlinear TE test (compute-heavy)
            te_n_permutations: Permutation count for TE significance
            stationarity_alpha: ADF test significance threshold
            auto_difference: Auto-difference non-stationary features
            n_jobs: Parallelism for TE computation
            min_observations: Min valid rows per feature after NaN removal
        """
        if correction not in ("bonferroni", "holm"):
            raise ValueError(
                f"correction must be 'bonferroni' or 'holm', got '{correction}'"
            )

        self.max_lag = max_lag
        self.significance_level = significance_level
        self.correction = correction
        self.use_transfer_entropy = use_transfer_entropy
        self.te_n_permutations = te_n_permutations
        self.stationarity_alpha = stationarity_alpha
        self.auto_difference = auto_difference
        self.n_jobs = n_jobs
        self.min_observations = min_observations

        self._surviving_features: list[str] = []
        self._per_feature_results: dict[str, CausalTestResult] = {}
        self._result: CausalFilterResult | None = None
        self._fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "CausalFilter":
        """
        Test each feature for Granger causality on y.

        Args:
            X: Feature matrix (rows = time, columns = features)
            y: Target series (forward returns)

        Returns:
            self
        """
        start = time.monotonic()
        n_features = len(X.columns)
        logger.info(
            f"CausalFilter: testing {n_features} features (max_lag={self.max_lag})"
        )

        # Clean target
        y_clean = y.replace([np.inf, -np.inf], np.nan).dropna()
        if len(y_clean) < self.min_observations:
            logger.warning(
                f"Target has only {len(y_clean)} valid observations "
                f"(min={self.min_observations}). Keeping all features."
            )
            self._surviving_features = list(X.columns)
            self._fitted = True
            self._result = CausalFilterResult(
                original_features=n_features,
                surviving_features=n_features,
                dropped_features=[],
                surviving_feature_names=list(X.columns),
                elapsed_seconds=time.monotonic() - start,
            )
            return self

        # Check target stationarity
        y_for_test = self._ensure_stationary(y_clean, "target")

        # Test each feature
        raw_results: list[CausalTestResult] = []
        for col in X.columns:
            result = self._test_feature(X[col], y_for_test, y_clean.index, col)
            raw_results.append(result)
            self._per_feature_results[col] = result

        # Collect testable features (those that got a valid p-value)
        testable = [r for r in raw_results if r.drop_reason is None]
        n_tests = len(testable)

        if n_tests == 0:
            logger.warning("No features had enough data for Granger testing")
            self._surviving_features = []
            self._fitted = True
            self._build_result(n_features, start)
            return self

        # Multiple testing correction
        if self.correction == "bonferroni":
            adjusted_alpha = self.significance_level / n_tests
            for r in testable:
                r.survived = r.granger_p_value < adjusted_alpha
        else:
            # Holm step-down: sort p-values ascending, compare to alpha/(n-rank+1)
            sorted_results = sorted(testable, key=lambda r: r.granger_p_value)
            for rank, r in enumerate(sorted_results):
                threshold = self.significance_level / (n_tests - rank)
                if r.granger_p_value >= threshold:
                    # This and all subsequent features fail
                    for r2 in sorted_results[rank:]:
                        r2.survived = False
                    break
                r.survived = True

        # Transfer entropy (optional, on Granger survivors only)
        if self.use_transfer_entropy:
            survivors = [r for r in testable if r.survived]
            self._compute_transfer_entropy_batch(X, y_clean, survivors)

        self._surviving_features = [r.feature_name for r in raw_results if r.survived]
        self._fitted = True
        self._build_result(n_features, start)

        logger.info(
            f"CausalFilter: {n_features} -> {len(self._surviving_features)} features "
            f"({n_features - len(self._surviving_features)} dropped, "
            f"correction={self.correction}, "
            f"{time.monotonic() - start:.1f}s)"
        )

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return only columns that survived causal filtering."""
        if not self._fitted:
            raise ValueError("Must fit before transform")
        present = [c for c in self._surviving_features if c in X.columns]
        return X[present]

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Fit and transform."""
        self.fit(X, y)
        return self.transform(X)

    def get_result(self) -> CausalFilterResult:
        """Get detailed filtering result for diagnostics."""
        if self._result is None:
            raise ValueError("Must fit before get_result")
        return self._result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_stationary(self, series: pd.Series, name: str) -> pd.Series:
        """Check stationarity and difference if needed."""
        result = adf_test(series, significance_level=self.stationarity_alpha)
        if result.is_significant:
            return series

        if self.auto_difference:
            diffed = series.diff().dropna()
            result2 = adf_test(diffed, significance_level=self.stationarity_alpha)
            if result2.is_significant:
                logger.debug(f"  {name}: non-stationary, differenced -> stationary")
                return diffed
            logger.debug(f"  {name}: still non-stationary after differencing")
            return diffed

        return series

    def _test_feature(
        self,
        feature: pd.Series,
        y_stationary: pd.Series,
        y_index: pd.Index,
        feature_name: str,
    ) -> CausalTestResult:
        """Run Granger causality test for a single feature."""
        # Clean feature
        feat_clean = feature.replace([np.inf, -np.inf], np.nan).dropna()

        # Align on common index
        common_idx = feat_clean.index.intersection(y_index)
        if len(common_idx) < self.min_observations:
            return CausalTestResult(
                feature_name=feature_name,
                granger_p_value=1.0,
                granger_f_stat=0.0,
                best_lag=0,
                is_stationary=False,
                was_differenced=False,
                survived=False,
                drop_reason=f"insufficient_data ({len(common_idx)} < {self.min_observations})",
            )

        feat_aligned = feat_clean.loc[common_idx]

        # Stationarity check on feature
        adf_result = adf_test(feat_aligned, significance_level=self.stationarity_alpha)
        is_stationary = adf_result.is_significant
        was_differenced = False

        feat_for_test = feat_aligned
        if not is_stationary and self.auto_difference:
            feat_for_test = feat_aligned.diff().dropna()
            was_differenced = True
            # Re-check after differencing
            adf_result2 = adf_test(
                feat_for_test, significance_level=self.stationarity_alpha
            )
            is_stationary = adf_result2.is_significant

        # Align with stationary target on common index after differencing
        common_test_idx = feat_for_test.index.intersection(y_stationary.index)
        if len(common_test_idx) < self.min_observations:
            return CausalTestResult(
                feature_name=feature_name,
                granger_p_value=1.0,
                granger_f_stat=0.0,
                best_lag=0,
                is_stationary=is_stationary,
                was_differenced=was_differenced,
                survived=False,
                drop_reason=f"insufficient_data_after_diff ({len(common_test_idx)})",
            )

        # Build 2-column DataFrame for grangercausalitytests
        # Column order: [y, x] — tests whether x Granger-causes y
        test_data = pd.DataFrame(
            {
                "y": y_stationary.loc[common_test_idx].values,
                "x": feat_for_test.loc[common_test_idx].values,
            }
        )
        test_data = test_data.dropna()

        if len(test_data) < self.max_lag + self.min_observations:
            return CausalTestResult(
                feature_name=feature_name,
                granger_p_value=1.0,
                granger_f_stat=0.0,
                best_lag=0,
                is_stationary=is_stationary,
                was_differenced=was_differenced,
                survived=False,
                drop_reason=f"insufficient_data_for_lag ({len(test_data)})",
            )

        # Run Granger test
        try:
            gc_results = grangercausalitytests(
                test_data[["y", "x"]],
                maxlag=self.max_lag,
                verbose=False,
            )
        except Exception as e:
            logger.debug(f"  {feature_name}: Granger test failed: {e}")
            return CausalTestResult(
                feature_name=feature_name,
                granger_p_value=1.0,
                granger_f_stat=0.0,
                best_lag=0,
                is_stationary=is_stationary,
                was_differenced=was_differenced,
                survived=False,
                drop_reason=f"granger_error: {e}",
            )

        # Extract best (lowest) p-value across lags, using ssr_ftest
        best_p = 1.0
        best_f = 0.0
        best_lag = 1
        for lag_val in range(1, self.max_lag + 1):
            if lag_val not in gc_results:
                continue
            tests_dict = gc_results[lag_val][0]
            p_val = tests_dict["ssr_ftest"][1]
            f_val = tests_dict["ssr_ftest"][0]
            if p_val < best_p:
                best_p = p_val
                best_f = f_val
                best_lag = lag_val

        return CausalTestResult(
            feature_name=feature_name,
            granger_p_value=best_p,
            granger_f_stat=best_f,
            best_lag=best_lag,
            is_stationary=is_stationary,
            was_differenced=was_differenced,
        )

    def _compute_transfer_entropy_batch(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        results: list[CausalTestResult],
    ) -> None:
        """Compute transfer entropy for a batch of features."""
        if not results:
            return

        rng = np.random.default_rng(42)
        y_arr = y.values

        def _compute_single(feat_name: str) -> tuple[str, float, float]:
            feat_arr = X[feat_name].reindex(y.index).values
            # Drop NaN pairs
            mask = ~(np.isnan(feat_arr) | np.isnan(y_arr))
            te_val, te_p = _te_permutation_test(
                feat_arr[mask],
                y_arr[mask],
                lag=1,
                n_permutations=self.te_n_permutations,
                rng=rng,
            )
            return feat_name, te_val, te_p

        if self.n_jobs > 1:
            with ProcessPoolExecutor(max_workers=self.n_jobs) as pool:
                futures = {
                    pool.submit(_compute_single, r.feature_name): r for r in results
                }
                for future in futures:
                    feat_name, te_val, te_p = future.result()
                    res = self._per_feature_results[feat_name]
                    res.transfer_entropy = te_val
                    res.te_p_value = te_p
        else:
            for r in results:
                _, te_val, te_p = _compute_single(r.feature_name)
                r.transfer_entropy = te_val
                r.te_p_value = te_p

        logger.info(
            f"  Transfer entropy computed for {len(results)} features "
            f"({self.te_n_permutations} permutations each)"
        )

    def _build_result(self, original_count: int, start_time: float) -> None:
        """Build the aggregate CausalFilterResult."""
        dropped = [
            name for name, r in self._per_feature_results.items() if not r.survived
        ]
        self._result = CausalFilterResult(
            original_features=original_count,
            surviving_features=len(self._surviving_features),
            dropped_features=dropped,
            surviving_feature_names=list(self._surviving_features),
            per_feature_results=dict(self._per_feature_results),
            alpha_used=self.significance_level,
            correction_method=self.correction,
            elapsed_seconds=time.monotonic() - start_time,
        )
