"""
Fourier calendar features.

Sine/cosine embeddings of calendar effects (day-of-week, month-of-year, etc.)
for use as ML features. Avoids one-hot explosion while capturing cyclical
patterns that affect market behavior (Monday effect, January effect, etc.).
"""

import numpy as np
import pandas as pd


class FourierCalendarFeatures:
    """
    Fourier sine/cosine embeddings of calendar effects.

    Creates smooth cyclical features for: day-of-week, day-of-month,
    month-of-year, week-of-year. Also flags structural calendar events
    (month-end, quarter-end, options expiration).
    """

    def compute(self, index: pd.DatetimeIndex) -> pd.DataFrame:
        """
        Parameters
        ----------
        index : pd.DatetimeIndex
            Datetime index to compute calendar features for.

        Returns
        -------
        pd.DataFrame with columns:
            sin_dow, cos_dow       – Day-of-week (period=5 trading days)
            sin_dom, cos_dom       – Day-of-month (period=21 trading days)
            sin_moy, cos_moy       – Month-of-year (period=12)
            sin_woy, cos_woy       – Week-of-year (period=52)
            is_month_end           – 1 on last trading day of month
            is_quarter_end         – 1 on last trading day of quarter
            is_opex                – 1 on third Friday of month (options expiration)
        """
        dow = index.dayofweek  # 0=Monday, 4=Friday
        dom = index.day
        moy = index.month
        woy = index.isocalendar().week.values.astype(float)

        result = pd.DataFrame(index=index)

        # Fourier embeddings
        result["sin_dow"] = np.sin(2 * np.pi * dow / 5)
        result["cos_dow"] = np.cos(2 * np.pi * dow / 5)
        result["sin_dom"] = np.sin(2 * np.pi * dom / 21)
        result["cos_dom"] = np.cos(2 * np.pi * dom / 21)
        result["sin_moy"] = np.sin(2 * np.pi * moy / 12)
        result["cos_moy"] = np.cos(2 * np.pi * moy / 12)
        result["sin_woy"] = np.sin(2 * np.pi * woy / 52)
        result["cos_woy"] = np.cos(2 * np.pi * woy / 52)

        # Structural events
        result["is_month_end"] = index.is_month_end.astype(int)
        result["is_quarter_end"] = index.is_quarter_end.astype(int)

        # Options expiration: third Friday of each month
        is_friday = dow == 4
        # Third Friday = Friday where day is between 15 and 21
        is_opex = is_friday & (dom >= 15) & (dom <= 21)
        result["is_opex"] = is_opex.astype(int)

        return result
