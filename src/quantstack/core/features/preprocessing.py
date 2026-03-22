"""
ML preprocessing transforms.

Transforms applied to price/volume series before feature computation.
These are not features themselves — they prepare the input data.

Includes:
- FractionalDifferentiator: Marcos Lopez de Prado (2018) fixed-width window
  fractional differentiation. Preserves memory while achieving stationarity.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller


class FractionalDifferentiator:
    """
    Fixed-width window fractional differentiation (De Prado 2018, Ch. 5).

    Fractional differentiation of order d ∈ (0, 1) preserves more memory
    than full first-differencing (d=1) while still achieving stationarity.
    This gives ML models access to long-range price level information that
    simple returns discard.

    Weight formula:
        w_0 = 1
        w_k = -w_{k-1} × (d - k + 1) / k  for k ≥ 1

    Parameters
    ----------
    max_d : float
        Maximum differentiation order to try. Default 1.0.
    d_step : float
        Step size for d search. Default 0.05.
    adf_threshold : float
        P-value threshold for ADF stationarity test. Default 0.05.
    min_weight : float
        Weights below this magnitude are truncated. Default 1e-5.
    """

    def __init__(
        self,
        max_d: float = 1.0,
        d_step: float = 0.05,
        adf_threshold: float = 0.05,
        min_weight: float = 1e-5,
    ) -> None:
        self.max_d = max_d
        self.d_step = d_step
        self.adf_threshold = adf_threshold
        self.min_weight = min_weight

    def _get_weights(self, d: float, size: int) -> np.ndarray:
        """Compute fractional differentiation weights."""
        w = [1.0]
        for k in range(1, size):
            w_next = -w[-1] * (d - k + 1) / k
            if abs(w_next) < self.min_weight:
                break
            w.append(w_next)
        return np.array(w)

    def transform(
        self, series: pd.Series, d: float | None = None, window: int | None = None
    ) -> pd.Series:
        """
        Apply fractional differentiation.

        Parameters
        ----------
        series : pd.Series
            Price or volume series.
        d : float, optional
            Differentiation order. If None, uses find_min_d().
        window : int, optional
            Fixed window size. If None, determined by weight truncation.

        Returns
        -------
        pd.Series — fractionally differentiated series (NaN-padded at start).
        """
        if d is None:
            d = self.find_min_d(series)

        values = series.values
        n = len(values)

        weights = self._get_weights(d, window or n)
        w_len = len(weights)

        result = np.full(n, np.nan)
        for i in range(w_len - 1, n):
            window_vals = values[i - w_len + 1 : i + 1][::-1]
            if not np.any(np.isnan(window_vals)):
                result[i] = np.dot(weights, window_vals)

        return pd.Series(result, index=series.index, name=series.name)

    def find_min_d(self, series: pd.Series, threshold: float | None = None) -> float:
        """
        Find minimum d that makes the series stationary (ADF test).

        Uses binary search over [0, max_d] with step d_step.

        Parameters
        ----------
        series : pd.Series
            Input series.
        threshold : float, optional
            ADF p-value threshold. Defaults to self.adf_threshold.

        Returns
        -------
        float — minimum d ∈ [0, max_d] that achieves stationarity.
        """
        if threshold is None:
            threshold = self.adf_threshold

        # Check if already stationary
        clean = series.dropna()
        if len(clean) < 20:
            return 0.0

        try:
            pval = adfuller(clean.values, maxlag=10, autolag="AIC")[1]
            if pval < threshold:
                return 0.0
        except (ValueError, np.linalg.LinAlgError):
            return 1.0

        best_d = self.max_d
        d = self.d_step
        while d <= self.max_d:
            diffed = self.transform(series, d=d)
            clean_diffed = diffed.dropna()
            if len(clean_diffed) < 20:
                d += self.d_step
                continue
            try:
                pval = adfuller(clean_diffed.values, maxlag=10, autolag="AIC")[1]
                if pval < threshold:
                    best_d = d
                    break
            except (ValueError, np.linalg.LinAlgError):
                pass
            d += self.d_step

        return best_d

    def transform_df(
        self,
        df: pd.DataFrame,
        columns: list[str],
        d_map: dict[str, float] | None = None,
    ) -> pd.DataFrame:
        """
        Apply fractional differentiation to multiple columns.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.
        columns : list[str]
            Columns to transform.
        d_map : dict[str, float], optional
            Pre-computed d values per column. If None, auto-detects.

        Returns
        -------
        pd.DataFrame with transformed columns (original columns replaced).
        """
        result = df.copy()
        for col in columns:
            if col not in df.columns:
                continue
            d = d_map[col] if d_map and col in d_map else None
            result[col] = self.transform(df[col], d=d)
        return result
