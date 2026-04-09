"""
Treatment effect estimation for causal alpha signals.

Supports Double Machine Learning (econml) with an OLS fallback when econml
is not installed. The OLS path is useful for rapid hypothesis screening but
should not be trusted for final signal weighting — install econml for
production use.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from quantstack.core.causal.models import CausalHypothesis, TreatmentEffect


class TreatmentEstimator:
    """Estimate average treatment effects from observational data."""

    def __init__(self, method: str = "dml") -> None:
        if method not in ("dml", "psm"):
            raise ValueError(f"Unknown method {method!r} — use 'dml' or 'psm'")
        self.method = method

    def estimate(
        self,
        features: pd.DataFrame,
        treatment_col: str,
        outcome_col: str,
        confounders: list[str],
        hypothesis: CausalHypothesis | None = None,
    ) -> TreatmentEffect:
        """
        Estimate the average treatment effect (ATE).

        Parameters
        ----------
        features : pd.DataFrame
            Panel with treatment, outcome, and confounder columns.
        treatment_col : str
            Name of the treatment variable column.
        outcome_col : str
            Name of the outcome variable column.
        confounders : list[str]
            Confounder column names to control for.
        hypothesis : CausalHypothesis, optional
            The hypothesis being tested. Used for labeling results.

        Returns
        -------
        TreatmentEffect
        """
        df = features[[treatment_col, outcome_col, *confounders]].dropna()
        if df.shape[0] < 30:
            logger.warning(
                "Only {} rows for treatment effect — need >=30, "
                "returning null effect",
                df.shape[0],
            )
            return self._null_effect(hypothesis, treatment_col, outcome_col)

        try:
            return self._estimate_econml(
                df, treatment_col, outcome_col, confounders, hypothesis
            )
        except ImportError:
            logger.warning(
                "econml not installed — falling back to OLS regression. "
                "Install econml for Double ML estimation."
            )
            return self._estimate_ols(
                df, treatment_col, outcome_col, confounders, hypothesis
            )

    # ------------------------------------------------------------------
    # econml Double ML
    # ------------------------------------------------------------------

    def _estimate_econml(
        self,
        df: pd.DataFrame,
        treatment_col: str,
        outcome_col: str,
        confounders: list[str],
        hypothesis: CausalHypothesis | None,
    ) -> TreatmentEffect:
        from econml.dml import LinearDML
        from sklearn.linear_model import LassoCV

        Y = df[outcome_col].values
        T = df[treatment_col].values.reshape(-1, 1)
        X = None  # no heterogeneity dimensions
        W = df[confounders].values if confounders else None

        model = LinearDML(
            model_y=LassoCV(cv=5),
            model_t=LassoCV(cv=5),
            random_state=42,
        )
        model.fit(Y, T, X=X, W=W)

        ate = float(model.const_marginal_ate(X)[0])
        ci = model.const_marginal_ate_interval(X, alpha=0.05)
        ci_lower = float(ci[0][0])
        ci_upper = float(ci[1][0])

        # Approximate p-value from CI: if CI excludes zero, p < 0.05
        se = (ci_upper - ci_lower) / (2 * 1.96)
        p_value = float(
            2 * (1 - self._normal_cdf(abs(ate) / se)) if se > 0 else 1.0
        )

        return TreatmentEffect(
            hypothesis=hypothesis or self._default_hypothesis(
                treatment_col, outcome_col
            ),
            ate=ate,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            p_value=p_value,
            method="dml",
            refutation_passed=False,  # must call refute() separately
            regime_stability=1.0,
        )

    # ------------------------------------------------------------------
    # OLS fallback
    # ------------------------------------------------------------------

    def _estimate_ols(
        self,
        df: pd.DataFrame,
        treatment_col: str,
        outcome_col: str,
        confounders: list[str],
        hypothesis: CausalHypothesis | None,
    ) -> TreatmentEffect:
        """Simple OLS: outcome ~ treatment + confounders."""
        from scipy import stats

        regressors = [treatment_col, *confounders]
        X = df[regressors].values
        # Add intercept
        X = np.column_stack([np.ones(X.shape[0]), X])
        y = df[outcome_col].values

        # OLS closed form: beta = (X'X)^-1 X'y
        try:
            XtX_inv = np.linalg.inv(X.T @ X)
        except np.linalg.LinAlgError:
            logger.error("Singular matrix in OLS — returning null effect")
            return self._null_effect(hypothesis, treatment_col, outcome_col)

        beta = XtX_inv @ (X.T @ y)
        residuals = y - X @ beta
        n, k = X.shape
        mse = float(np.sum(residuals**2) / (n - k))
        se_beta = np.sqrt(np.diag(XtX_inv) * mse)

        # Treatment coefficient is at index 1 (after intercept)
        ate = float(beta[1])
        se = float(se_beta[1])
        t_stat = ate / se if se > 0 else 0.0
        p_value = float(2 * stats.t.sf(abs(t_stat), df=n - k))
        ci_lower = ate - 1.96 * se
        ci_upper = ate + 1.96 * se

        return TreatmentEffect(
            hypothesis=hypothesis or self._default_hypothesis(
                treatment_col, outcome_col
            ),
            ate=ate,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            p_value=p_value,
            method="ols_fallback",
            refutation_passed=False,
            regime_stability=1.0,
        )

    # ------------------------------------------------------------------
    # Refutation
    # ------------------------------------------------------------------

    def refute(
        self,
        features: pd.DataFrame,
        treatment_col: str,
        outcome_col: str,
        confounders: list[str],
        method: str = "placebo",
    ) -> dict:
        """
        Run a refutation test to check robustness of the treatment effect.

        Parameters
        ----------
        method : str
            One of "placebo" (random treatment), "random_common_cause"
            (add random confounder), or "subset" (use data subset).

        Returns
        -------
        dict
            {"passed": bool, "refutation_type": str,
             "original_effect": float, "refuted_effect": float}
        """
        df = features[[treatment_col, outcome_col, *confounders]].dropna()

        # Estimate original effect
        original = self.estimate(
            features, treatment_col, outcome_col, confounders
        )
        original_ate = original.ate

        if method == "placebo":
            # Replace treatment with random noise — effect should vanish
            df_placebo = df.copy()
            rng = np.random.default_rng(42)
            df_placebo[treatment_col] = rng.permutation(
                df_placebo[treatment_col].values
            )
            refuted = self.estimate(
                df_placebo, treatment_col, outcome_col, confounders
            )
            refuted_ate = refuted.ate
            # Pass if refuted effect is much smaller than original
            passed = abs(refuted_ate) < 0.5 * abs(original_ate) if original_ate != 0 else True

        elif method == "random_common_cause":
            # Add a random confounder — effect should stay similar
            df_rcc = df.copy()
            rng = np.random.default_rng(42)
            random_col = "_random_common_cause"
            df_rcc[random_col] = rng.standard_normal(len(df_rcc))
            refuted = self.estimate(
                df_rcc,
                treatment_col,
                outcome_col,
                [*confounders, random_col],
            )
            refuted_ate = refuted.ate
            # Pass if effect is stable (within 20% of original)
            passed = (
                abs(refuted_ate - original_ate) < 0.2 * abs(original_ate)
                if original_ate != 0
                else True
            )

        elif method == "subset":
            # Use random 80% subset — effect should be similar
            rng = np.random.default_rng(42)
            idx = rng.choice(len(df), size=int(0.8 * len(df)), replace=False)
            df_subset = df.iloc[idx]
            refuted = self.estimate(
                df_subset, treatment_col, outcome_col, confounders
            )
            refuted_ate = refuted.ate
            passed = (
                abs(refuted_ate - original_ate) < 0.3 * abs(original_ate)
                if original_ate != 0
                else True
            )

        else:
            raise ValueError(
                f"Unknown refutation method {method!r} — "
                "use 'placebo', 'random_common_cause', or 'subset'"
            )

        return {
            "passed": passed,
            "refutation_type": method,
            "original_effect": original_ate,
            "refuted_effect": refuted_ate,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normal_cdf(x: float) -> float:
        """Standard normal CDF without scipy import."""
        import math

        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    @staticmethod
    def _null_effect(
        hypothesis: CausalHypothesis | None,
        treatment_col: str,
        outcome_col: str,
    ) -> TreatmentEffect:
        hyp = hypothesis or TreatmentEstimator._default_hypothesis(
            treatment_col, outcome_col
        )
        return TreatmentEffect(
            hypothesis=hyp,
            ate=0.0,
            ci_lower=0.0,
            ci_upper=0.0,
            p_value=1.0,
            method="null",
            refutation_passed=False,
            regime_stability=0.0,
        )

    @staticmethod
    def _default_hypothesis(
        treatment_col: str, outcome_col: str
    ) -> CausalHypothesis:
        return CausalHypothesis(
            treatment=treatment_col,
            outcome=outcome_col,
            expected_direction="positive",
            description=f"Auto-generated: {treatment_col} -> {outcome_col}",
        )
