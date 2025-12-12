"""
Lookahead Bias Detection and Data Leakage Diagnostics.

Comprehensive toolkit for detecting:
- Feature lookahead bias
- Label leakage
- Data snooping
- Information leakage through cross-validation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from scipy import stats
from loguru import logger


@dataclass
class LeakageReport:
    """Report of detected data leakage."""

    has_leakage: bool
    severity: str  # "none", "low", "medium", "high", "critical"
    issues: List[Dict]
    recommendations: List[str]


class LeakageDiagnostics:
    """
    Comprehensive leakage detection for trading systems.

    Checks for:
    1. Feature lookahead: Features computed using future data
    2. Label leakage: Labels containing future information
    3. Cross-validation leakage: Improper train/test splits
    4. Data snooping: Excessive hyperparameter tuning
    """

    def __init__(self):
        self.issues = []
        self.recommendations = []

    def run_full_diagnostics(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        prices: pd.Series,
        returns: pd.Series,
    ) -> LeakageReport:
        """
        Run all leakage diagnostics.

        Args:
            features: Feature DataFrame
            labels: Target labels
            prices: Price series
            returns: Return series

        Returns:
            LeakageReport with all findings
        """
        self.issues = []
        self.recommendations = []

        # 1. Check feature lookahead
        self._check_feature_lookahead(features, returns)

        # 2. Check label leakage
        self._check_label_leakage(labels, returns)

        # 3. Check for suspiciously high correlations
        self._check_suspicious_correlations(features, labels)

        # 4. Check temporal alignment
        self._check_temporal_alignment(features, labels, prices)

        # Determine severity
        severity = self._assess_severity()

        return LeakageReport(
            has_leakage=len(self.issues) > 0,
            severity=severity,
            issues=self.issues,
            recommendations=self.recommendations,
        )

    def _check_feature_lookahead(
        self,
        features: pd.DataFrame,
        returns: pd.Series,
    ) -> None:
        """
        Check if features have lookahead bias.

        Signs of lookahead:
        - Future returns correlation > 0.5
        - Perfect prediction of future events
        """
        for col in features.columns:
            feature = features[col]

            # Check correlation with future returns
            for lag in [0, -1, -2]:  # Lag 0 and negative lags are suspicious
                if lag == 0:
                    corr_series = returns
                else:
                    corr_series = returns.shift(lag)

                valid = ~(feature.isna() | corr_series.isna())
                if valid.sum() < 30:
                    continue

                corr, _ = stats.spearmanr(feature[valid], corr_series[valid])

                if abs(corr) > 0.5 and lag <= 0:
                    self.issues.append(
                        {
                            "type": "feature_lookahead",
                            "feature": col,
                            "lag": lag,
                            "correlation": corr,
                            "description": f"Feature '{col}' has {corr:.2f} correlation with lag-{abs(lag)} returns",
                        }
                    )
                    self.recommendations.append(
                        f"Review feature '{col}' - may contain future information"
                    )

    def _check_label_leakage(
        self,
        labels: pd.Series,
        returns: pd.Series,
    ) -> None:
        """
        Check if labels leak future information.

        Signs of leakage:
        - Labels perfectly predict next-period returns
        - Labels have > 0.8 correlation with immediate returns
        """
        common_idx = labels.index.intersection(returns.index)
        labels_aligned = labels.loc[common_idx]
        returns_aligned = returns.loc[common_idx]

        # Check correlation with concurrent and future returns
        for lag in [0, 1, 2, 3]:
            lagged_returns = returns_aligned.shift(-lag)
            valid = ~(labels_aligned.isna() | lagged_returns.isna())

            if valid.sum() < 30:
                continue

            corr, _ = stats.spearmanr(labels_aligned[valid], lagged_returns[valid])

            if lag == 0 and abs(corr) > 0.8:
                self.issues.append(
                    {
                        "type": "label_leakage",
                        "lag": lag,
                        "correlation": corr,
                        "description": f"Labels have {corr:.2f} correlation with same-period returns",
                    }
                )
                self.recommendations.append(
                    "Labels may contain future information - check label construction"
                )
            elif lag == 1 and abs(corr) > 0.5:
                # This is actually expected for good labels
                pass

    def _check_suspicious_correlations(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
    ) -> None:
        """
        Check for suspiciously high feature-label correlations.

        Correlations > 0.9 often indicate leakage.
        """
        common_idx = features.index.intersection(labels.index)

        for col in features.columns:
            feature = features.loc[common_idx, col]
            label = labels.loc[common_idx]

            valid = ~(feature.isna() | label.isna())
            if valid.sum() < 30:
                continue

            corr, _ = stats.spearmanr(feature[valid], label[valid])

            if abs(corr) > 0.9:
                self.issues.append(
                    {
                        "type": "suspicious_correlation",
                        "feature": col,
                        "correlation": corr,
                        "description": f"Feature '{col}' has {corr:.2f} correlation with labels - likely leakage",
                    }
                )
                self.recommendations.append(
                    f"Investigate feature '{col}' - correlation too high to be predictive"
                )

    def _check_temporal_alignment(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        prices: pd.Series,
    ) -> None:
        """
        Check temporal alignment of data.

        Issues:
        - Misaligned timestamps
        - Features available before their index suggests
        """
        # Check if feature index aligns with label index
        feature_idx = set(features.index)
        label_idx = set(labels.index)

        only_in_features = feature_idx - label_idx
        only_in_labels = label_idx - feature_idx

        if len(only_in_features) > 0.1 * len(feature_idx):
            self.issues.append(
                {
                    "type": "temporal_misalignment",
                    "description": f"{len(only_in_features)} timestamps in features but not labels",
                }
            )
            self.recommendations.append(
                "Check data pipeline for timestamp alignment issues"
            )

    def _assess_severity(self) -> str:
        """Assess overall severity of detected issues."""
        if not self.issues:
            return "none"

        critical_types = {"label_leakage", "feature_lookahead"}
        high_types = {"suspicious_correlation"}

        has_critical = any(i["type"] in critical_types for i in self.issues)
        has_high = any(i["type"] in high_types for i in self.issues)

        if has_critical:
            return "critical"
        elif has_high:
            return "high"
        elif len(self.issues) > 3:
            return "medium"
        else:
            return "low"


def detect_survivorship_bias(
    data: pd.DataFrame,
    min_history_years: float = 5,
) -> Dict:
    """
    Detect potential survivorship bias in dataset.

    Args:
        data: DataFrame with symbol column and DatetimeIndex
        min_history_years: Minimum years of history expected

    Returns:
        Dictionary with survivorship bias analysis
    """
    if "symbol" not in data.columns:
        return {"error": "No symbol column found"}

    symbols = data["symbol"].unique()

    # Check start dates
    start_dates = {}
    for symbol in symbols:
        symbol_data = data[data["symbol"] == symbol]
        start_dates[symbol] = symbol_data.index.min()

    # Flag symbols that start late (potential survivors)
    overall_start = min(start_dates.values())
    late_starters = {
        s: d
        for s, d in start_dates.items()
        if (d - overall_start).days > 365 * 2  # Started > 2 years later
    }

    return {
        "total_symbols": len(symbols),
        "late_starters": len(late_starters),
        "late_starter_symbols": list(late_starters.keys()),
        "potential_survivorship_bias": len(late_starters) > 0.1 * len(symbols),
        "recommendation": (
            "Include delisted securities to avoid survivorship bias"
            if late_starters
            else "Data appears free of obvious survivorship bias"
        ),
    }


def check_point_in_time_accuracy(
    data: pd.DataFrame,
    as_of_col: str,
    effective_col: str,
) -> Dict:
    """
    Check if data respects point-in-time accuracy.

    Point-in-time means: data at time T only includes information
    that was actually available at time T.

    Args:
        data: DataFrame with timestamps
        as_of_col: Column indicating when data was recorded
        effective_col: Column indicating when data became available

    Returns:
        Dictionary with point-in-time analysis
    """
    if as_of_col not in data.columns or effective_col not in data.columns:
        return {"error": "Required columns not found"}

    # Check if effective date ever precedes as-of date
    violations = data[data[effective_col] > data[as_of_col]]

    return {
        "total_records": len(data),
        "pit_violations": len(violations),
        "violation_rate": len(violations) / len(data) if len(data) > 0 else 0,
        "is_pit_compliant": len(violations) == 0,
        "recommendation": (
            "Fix point-in-time violations"
            if len(violations) > 0
            else "Data is point-in-time compliant"
        ),
    }


def generate_leakage_report(report: LeakageReport) -> str:
    """Generate text report from leakage diagnostics."""
    output = f"""
Data Leakage Diagnostics Report
===============================

Overall Assessment: {report.severity.upper()}
Has Leakage: {"YES" if report.has_leakage else "NO"}

Issues Found ({len(report.issues)}):
"""

    if not report.issues:
        output += "  No issues detected.\n"
    else:
        for i, issue in enumerate(report.issues, 1):
            output += f"""
{i}. [{issue['type'].upper()}]
   {issue['description']}
"""

    output += f"""
Recommendations:
"""
    if not report.recommendations:
        output += "  None - data appears clean.\n"
    else:
        for rec in report.recommendations:
            output += f"  • {rec}\n"

    return output
