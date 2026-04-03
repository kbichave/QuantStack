"""Fractional differentiation for minimum-memory-loss stationarity.

Implements fixed-width window fractional differentiation from
Advances in Financial Machine Learning (Lopez de Prado, Chapter 5).

Standard first-differencing (d=1) achieves stationarity but destroys all
memory in the series. Fractional differencing with d < 1 preserves partial
memory while still achieving stationarity, yielding better ML features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger
from statsmodels.tsa.stattools import adfuller


def _compute_weights(d: float, window: int) -> np.ndarray:
    """Compute fractional differentiation weights using binomial series.

    w[0] = 1, w[k] = w[k-1] * (d - k + 1) / k  for k = 1..window-1
    """
    weights = np.zeros(window)
    weights[0] = 1.0
    for k in range(1, window):
        weights[k] = -weights[k - 1] * (d - k + 1) / k
    return weights


def frac_diff(series: pd.Series, d: float, window: int = 100) -> pd.Series:
    """Apply fixed-width window fractional differentiation.

    Args:
        series: Input price or feature series.
        d: Differencing order (0 = original, 1 = first difference).
        window: Number of terms in the weight expansion.

    Returns:
        Fractionally differenced series (first window-1 values are NaN).
    """
    if len(series) == 0:
        return pd.Series(dtype=float)

    window = min(window, len(series))
    weights = _compute_weights(d, window)

    result = np.full(len(series), np.nan)
    values = series.values.astype(float)

    for i in range(window - 1, len(values)):
        # Dot product of weights with the window of values ending at i
        result[i] = np.dot(weights, values[i - window + 1 : i + 1][::-1])

    return pd.Series(result, index=series.index)


def find_min_d(
    series: pd.Series,
    d_range: tuple[float, float] = (0.0, 1.0),
    step: float = 0.05,
    window: int = 100,
    adf_threshold: float = 0.05,
) -> float:
    """Find the minimum d that achieves stationarity.

    Iterates d from d_range[0]+step upward. Returns the first d where
    the ADF test p-value < adf_threshold.

    If no d in range achieves stationarity, returns d_range[1].
    """
    d_values = np.arange(d_range[0] + step, d_range[1] + step / 2, step)

    for d in d_values:
        diffed = frac_diff(series, d=d, window=window).dropna()
        if len(diffed) < 20:
            continue
        try:
            adf_pvalue = adfuller(diffed, maxlag=min(10, len(diffed) // 5), autolag=None)[1]
        except Exception:
            continue
        if adf_pvalue < adf_threshold:
            logger.debug(f"find_min_d: d={d:.2f} achieves stationarity (p={adf_pvalue:.4f})")
            return round(d, 4)

    logger.warning("find_min_d: no d achieved stationarity, returning upper bound")
    return d_range[1]


def batch_find_min_d(
    features_df: pd.DataFrame,
    window: int = 100,
    adf_threshold: float = 0.05,
) -> dict[str, float]:
    """Find minimum d for each column in a features DataFrame.

    Returns dict mapping column_name -> optimal_d.
    """
    results = {}
    for col in features_df.columns:
        results[col] = find_min_d(
            features_df[col].dropna(),
            window=window,
            adf_threshold=adf_threshold,
        )
    return results
