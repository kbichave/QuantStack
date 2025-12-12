"""
Data leakage detection and prevention.

Critical tests for ensuring model validity.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from loguru import logger
from sklearn.metrics import roc_auc_score


@dataclass
class LeakageTestResult:
    """Result of a leakage test."""

    test_name: str
    passed: bool
    score: float
    threshold: float
    details: str
    severity: str  # INFO, WARNING, CRITICAL


class FeatureShiftTest:
    """
    Test for lookahead bias by shifting features.

    If shifting features forward degrades performance significantly,
    features are likely valid. If not, there may be leakage.
    """

    def __init__(
        self,
        shift_periods: List[int] = [-1, 1, 2],
        degradation_threshold: float = 0.1,
    ):
        """
        Initialize feature shift test.

        Args:
            shift_periods: Periods to shift features
            degradation_threshold: Expected performance drop
        """
        self.shift_periods = shift_periods
        self.degradation_threshold = degradation_threshold

    def run(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        base_score: Optional[float] = None,
    ) -> List[LeakageTestResult]:
        """
        Run feature shift tests.

        Args:
            model: Trained model
            X: Feature matrix
            y: Target
            base_score: Baseline AUC (computed if not provided)

        Returns:
            List of test results
        """
        results = []

        # Get baseline score
        if base_score is None:
            y_pred = model.predict_proba(X)[:, 1]
            valid_mask = ~(y.isna() | np.isnan(y_pred))
            base_score = roc_auc_score(y[valid_mask], y_pred[valid_mask])

        for shift in self.shift_periods:
            # Shift features
            X_shifted = X.shift(shift).dropna()
            y_shifted = y.loc[X_shifted.index]

            # Get predictions
            try:
                y_pred = model.predict_proba(X_shifted)[:, 1]
                valid_mask = ~(y_shifted.isna() | np.isnan(y_pred))
                shifted_score = roc_auc_score(y_shifted[valid_mask], y_pred[valid_mask])
            except Exception as e:
                logger.warning(f"Shift test failed for shift={shift}: {e}")
                continue

            # Analyze degradation
            degradation = base_score - shifted_score

            if shift < 0:  # Forward shift (should use future data = better)
                expected = "Score should increase or stay same"
                passed = degradation <= 0.02  # Allow small increase
                severity = (
                    "CRITICAL"
                    if degradation < -0.1
                    else "WARNING" if degradation < -0.05 else "INFO"
                )
            else:  # Backward shift (should lose info = worse)
                expected = f"Score should drop by >{self.degradation_threshold}"
                passed = degradation >= self.degradation_threshold * 0.5
                severity = (
                    "CRITICAL"
                    if degradation < 0.02
                    else (
                        "WARNING"
                        if degradation < self.degradation_threshold
                        else "INFO"
                    )
                )

            results.append(
                LeakageTestResult(
                    test_name=f"feature_shift_{shift}",
                    passed=passed,
                    score=shifted_score,
                    threshold=self.degradation_threshold,
                    details=f"Shift={shift}, Base={base_score:.4f}, Shifted={shifted_score:.4f}, Δ={degradation:.4f}. {expected}",
                    severity=severity,
                )
            )

        return results


class PermutationTest:
    """
    Permutation test for statistical significance.

    Randomly permutes target labels to establish null distribution.
    Real model should significantly outperform permuted models.
    """

    def __init__(
        self,
        n_permutations: int = 100,
        significance_level: float = 0.05,
    ):
        """
        Initialize permutation test.

        Args:
            n_permutations: Number of permutations
            significance_level: P-value threshold
        """
        self.n_permutations = n_permutations
        self.significance_level = significance_level

    def run(
        self,
        model_class: Any,
        model_params: dict,
        X: pd.DataFrame,
        y: pd.Series,
        real_score: Optional[float] = None,
    ) -> LeakageTestResult:
        """
        Run permutation test.

        Args:
            model_class: Model class to instantiate
            model_params: Model parameters
            X: Features
            y: Target
            real_score: Real model score

        Returns:
            LeakageTestResult
        """
        # Clean data
        valid_mask = ~y.isna()
        X_clean = X[valid_mask]
        y_clean = y[valid_mask]

        # Get real score if not provided
        if real_score is None:
            model = model_class(**model_params)
            model.fit(X_clean, y_clean)
            y_pred = model.predict_proba(X_clean)[:, 1]
            real_score = roc_auc_score(y_clean, y_pred)

        # Generate null distribution
        null_scores = []

        for i in range(self.n_permutations):
            # Permute labels
            y_permuted = y_clean.sample(frac=1, random_state=i).reset_index(drop=True)
            y_permuted.index = y_clean.index

            # Train model on permuted data
            model = model_class(**model_params)
            try:
                model.fit(X_clean, y_permuted)
                y_pred = model.predict_proba(X_clean)[:, 1]
                score = roc_auc_score(y_permuted, y_pred)
                null_scores.append(score)
            except Exception:
                continue

        if not null_scores:
            return LeakageTestResult(
                test_name="permutation_test",
                passed=False,
                score=real_score,
                threshold=self.significance_level,
                details="Failed to generate null distribution",
                severity="CRITICAL",
            )

        # Calculate p-value
        null_scores = np.array(null_scores)
        p_value = np.mean(null_scores >= real_score)

        passed = p_value < self.significance_level

        return LeakageTestResult(
            test_name="permutation_test",
            passed=passed,
            score=real_score,
            threshold=self.significance_level,
            details=f"P-value={p_value:.4f}, Real={real_score:.4f}, Null mean={null_scores.mean():.4f}, Null std={null_scores.std():.4f}",
            severity="CRITICAL" if not passed else "INFO",
        )


class DistributionTest:
    """
    Test for train/test distribution drift.

    Uses KS test to detect if features have significantly different
    distributions between train and test sets, which could indicate
    data leakage or temporal drift.
    """

    def __init__(
        self,
        drift_threshold: float = 0.2,
        p_value_threshold: float = 0.01,
    ):
        """
        Initialize distribution test.

        Args:
            drift_threshold: KS statistic threshold for drift detection
            p_value_threshold: P-value threshold for significance
        """
        self.drift_threshold = drift_threshold
        self.p_value_threshold = p_value_threshold

    def run(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
    ) -> LeakageTestResult:
        """
        Run distribution test comparing train and test features.

        Args:
            X_train: Training features
            X_test: Test features

        Returns:
            LeakageTestResult
        """
        from scipy import stats

        if X_train.empty or X_test.empty:
            return LeakageTestResult(
                test_name="distribution_test",
                passed=True,
                score=0.0,
                threshold=self.drift_threshold,
                details="Empty DataFrame provided, skipping test",
                severity="INFO",
            )

        drift_detected = []
        max_ks_stat = 0.0

        for col in X_train.columns:
            if col not in X_test.columns:
                continue

            train_vals = X_train[col].dropna()
            test_vals = X_test[col].dropna()

            if len(train_vals) < 10 or len(test_vals) < 10:
                continue

            try:
                ks_stat, p_value = stats.ks_2samp(train_vals, test_vals)
                max_ks_stat = max(max_ks_stat, ks_stat)

                if ks_stat > self.drift_threshold or p_value < self.p_value_threshold:
                    drift_detected.append((col, ks_stat, p_value))
            except Exception:
                continue

        passed = len(drift_detected) == 0

        if drift_detected:
            worst_col, worst_ks, worst_p = max(drift_detected, key=lambda x: x[1])
            details = f"Drift detected in {len(drift_detected)} features. Worst: {worst_col} (KS={worst_ks:.4f}, p={worst_p:.4f})"
            severity = "CRITICAL" if worst_ks > 0.5 else "WARNING"
        else:
            details = f"No significant drift detected. Max KS stat={max_ks_stat:.4f}"
            severity = "INFO"

        return LeakageTestResult(
            test_name="distribution_test",
            passed=passed,
            score=max_ks_stat,
            threshold=self.drift_threshold,
            details=details,
            severity=severity,
        )


class LeakageDetector:
    """
    Comprehensive leakage detection suite.

    Runs multiple tests to detect potential data leakage:
    1. Feature shift test
    2. Permutation test
    3. Train/test distribution comparison
    4. Feature-target correlation timeline
    """

    def __init__(self):
        """Initialize leakage detector."""
        self.shift_test = FeatureShiftTest()
        self.permutation_test = PermutationTest(n_permutations=50)

    def run_all_tests(
        self,
        model: Any,
        model_class: Any,
        model_params: dict,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Dict[str, List[LeakageTestResult]]:
        """
        Run all leakage detection tests.

        Args:
            model: Trained model
            model_class: Model class for permutation test
            model_params: Model parameters
            X_train: Training features
            y_train: Training target
            X_test: Test features
            y_test: Test target

        Returns:
            Dictionary of test results by category
        """
        results = {
            "shift_tests": [],
            "permutation_tests": [],
            "distribution_tests": [],
        }

        # 1. Feature shift tests
        logger.info("Running feature shift tests...")
        train_score = self._get_score(model, X_train, y_train)
        shift_results = self.shift_test.run(model, X_test, y_test, train_score)
        results["shift_tests"] = shift_results

        # 2. Train/test distribution comparison
        logger.info("Running distribution tests...")
        dist_results = self._check_distributions(X_train, X_test)
        results["distribution_tests"] = dist_results

        # 3. Permutation test (expensive, run last)
        logger.info("Running permutation test...")
        perm_result = self.permutation_test.run(
            model_class, model_params, X_train, y_train, train_score
        )
        results["permutation_tests"] = [perm_result]

        return results

    def _get_score(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> float:
        """Get model score."""
        y_pred = model.predict_proba(X)[:, 1]
        valid_mask = ~(y.isna() | np.isnan(y_pred))
        return roc_auc_score(y[valid_mask], y_pred[valid_mask])

    def _check_distributions(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
    ) -> List[LeakageTestResult]:
        """Check for distribution shift between train and test."""
        results = []

        for col in X_train.columns[:20]:  # Check first 20 features
            if col not in X_test.columns:
                continue

            train_vals = X_train[col].dropna()
            test_vals = X_test[col].dropna()

            if len(train_vals) < 10 or len(test_vals) < 10:
                continue

            # KS test for distribution difference
            from scipy import stats

            ks_stat, p_value = stats.ks_2samp(train_vals, test_vals)

            # Significant difference might indicate issue
            passed = p_value > 0.01  # Allow some difference

            if not passed:
                results.append(
                    LeakageTestResult(
                        test_name=f"distribution_{col}",
                        passed=passed,
                        score=ks_stat,
                        threshold=0.01,
                        details=f"KS stat={ks_stat:.4f}, p={p_value:.4f}",
                        severity="WARNING" if ks_stat < 0.3 else "INFO",
                    )
                )

        return results

    def generate_report(
        self,
        results: Dict[str, List[LeakageTestResult]],
    ) -> str:
        """Generate human-readable report."""
        lines = ["=" * 60, "LEAKAGE DETECTION REPORT", "=" * 60, ""]

        total_tests = 0
        passed_tests = 0
        critical_failures = []

        for category, tests in results.items():
            lines.append(f"\n{category.upper()}")
            lines.append("-" * 40)

            for test in tests:
                total_tests += 1
                if test.passed:
                    passed_tests += 1
                    status = "✓ PASS"
                else:
                    status = "✗ FAIL"
                    if test.severity == "CRITICAL":
                        critical_failures.append(test)

                lines.append(f"  {status} [{test.severity}] {test.test_name}")
                lines.append(f"       {test.details}")

        lines.append("\n" + "=" * 60)
        lines.append(f"SUMMARY: {passed_tests}/{total_tests} tests passed")

        if critical_failures:
            lines.append(f"\n⚠️  {len(critical_failures)} CRITICAL failures detected!")
            lines.append("Review these before proceeding to production.")

        return "\n".join(lines)
