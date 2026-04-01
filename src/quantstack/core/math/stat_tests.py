# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Core statistical tests — pure functions with no upward dependencies.

Extracted from core.research.stat_tests so that lower layers (core.validation)
can use adf_test without depending on the research layer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from loguru import logger
from statsmodels.tsa.stattools import adfuller


@dataclass
class TestResult:
    """Result of a statistical test."""

    test_name: str
    statistic: float
    p_value: float
    is_significant: bool
    critical_values: dict[str, float] | None = None
    additional_info: dict | None = None


def adf_test(
    series: pd.Series,
    max_lags: int | None = None,
    regression: str = "c",
    significance_level: float = 0.05,
) -> TestResult:
    """
    Augmented Dickey-Fuller test for stationarity.

    Args:
        series: Time series to test
        max_lags: Maximum number of lags to include
        regression: Type of regression ('c' = constant, 'ct' = constant + trend, 'n' = none)
        significance_level: Significance level for hypothesis test

    Returns:
        TestResult with ADF statistic, p-value, and critical values
    """
    series = series.dropna()

    if len(series) < 20:
        logger.warning("ADF test requires at least 20 observations")
        return TestResult(
            test_name="ADF",
            statistic=np.nan,
            p_value=1.0,
            is_significant=False,
            additional_info={"error": "insufficient_data"},
        )

    result = adfuller(series, maxlag=max_lags, regression=regression, autolag="AIC")

    adf_stat, p_value, used_lag, nobs, critical_values, icbest = result

    return TestResult(
        test_name="ADF",
        statistic=adf_stat,
        p_value=p_value,
        is_significant=bool(p_value < significance_level),
        critical_values=critical_values,
        additional_info={
            "used_lag": used_lag,
            "nobs": nobs,
            "ic_best": icbest,
            "regression": regression,
            "interpretation": (
                "stationary" if p_value < significance_level else "non-stationary"
            ),
        },
    )
