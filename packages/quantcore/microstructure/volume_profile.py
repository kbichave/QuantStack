"""
Volume profile and intraday seasonality analysis.

Models typical intraday volume patterns for relative volume analysis.
"""

from dataclasses import dataclass
from typing import Dict, Optional, List
import pandas as pd
import numpy as np
from loguru import logger


@dataclass
class VolumeProfileStats:
    """Volume profile statistics."""

    avg_volume_by_hour: Dict[int, float]
    std_volume_by_hour: Dict[int, float]
    typical_pattern: pd.Series  # Normalized hourly pattern


class IntradaySeasonality:
    """
    Model intraday volume seasonality.

    Typical US equity pattern:
    - High volume at open (9:30-10:30)
    - Lower volume mid-day (11:00-14:00)
    - Increasing volume into close (14:30-16:00)
    """

    # Typical US equity intraday volume pattern (normalized)
    TYPICAL_US_EQUITY_PATTERN = {
        9: 1.5,  # First hour high
        10: 1.2,
        11: 0.8,
        12: 0.7,  # Lunch lull
        13: 0.7,
        14: 0.9,
        15: 1.2,  # Pickup into close
    }

    def __init__(self, lookback_days: int = 20):
        """
        Initialize seasonality analyzer.

        Args:
            lookback_days: Days to use for pattern estimation
        """
        self.lookback_days = lookback_days
        self._learned_pattern: Optional[Dict[int, float]] = None

    def fit(self, df: pd.DataFrame) -> VolumeProfileStats:
        """
        Learn intraday volume pattern from data.

        Args:
            df: DataFrame with volume and datetime index

        Returns:
            VolumeProfileStats
        """
        if df.empty:
            return self._default_stats()

        # Group by hour
        df_copy = df.copy()
        df_copy["hour"] = df_copy.index.hour

        hourly_stats = df_copy.groupby("hour")["volume"].agg(["mean", "std"])

        avg_by_hour = hourly_stats["mean"].to_dict()
        std_by_hour = hourly_stats["std"].to_dict()

        # Normalize pattern
        total_avg = np.mean(list(avg_by_hour.values()))
        normalized = {h: v / total_avg for h, v in avg_by_hour.items()}

        self._learned_pattern = normalized

        return VolumeProfileStats(
            avg_volume_by_hour=avg_by_hour,
            std_volume_by_hour=std_by_hour,
            typical_pattern=pd.Series(normalized),
        )

    def _default_stats(self) -> VolumeProfileStats:
        """Return default statistics when no data."""
        return VolumeProfileStats(
            avg_volume_by_hour=self.TYPICAL_US_EQUITY_PATTERN,
            std_volume_by_hour={
                h: v * 0.3 for h, v in self.TYPICAL_US_EQUITY_PATTERN.items()
            },
            typical_pattern=pd.Series(self.TYPICAL_US_EQUITY_PATTERN),
        )

    def get_expected_volume_ratio(self, hour: int) -> float:
        """
        Get expected volume ratio for an hour.

        Args:
            hour: Hour of day (0-23)

        Returns:
            Expected volume ratio vs daily average
        """
        pattern = self._learned_pattern or self.TYPICAL_US_EQUITY_PATTERN
        return pattern.get(hour, 1.0)

    def get_adjusted_volume_ratio(
        self,
        volume: float,
        hour: int,
        daily_avg_volume: float,
    ) -> float:
        """
        Get seasonality-adjusted volume ratio.

        Adjusts raw volume ratio for expected intraday pattern.

        Args:
            volume: Current volume
            hour: Hour of day
            daily_avg_volume: Daily average volume

        Returns:
            Adjusted volume ratio
        """
        expected_ratio = self.get_expected_volume_ratio(hour)
        expected_volume = daily_avg_volume * expected_ratio

        if expected_volume <= 0:
            return 1.0

        return volume / expected_volume


class VolumeProfileAnalyzer:
    """
    Comprehensive volume profile analysis.

    Features:
    - Intraday seasonality modeling
    - Relative volume vs expected
    - Volume anomaly detection
    - Volume distribution analysis
    """

    def __init__(self):
        """Initialize volume profile analyzer."""
        self.seasonality = IntradaySeasonality()

    def compute_features(
        self,
        df: pd.DataFrame,
        fit_seasonality: bool = True,
    ) -> pd.DataFrame:
        """
        Compute volume profile features.

        Args:
            df: OHLCV DataFrame
            fit_seasonality: Whether to fit seasonality from data

        Returns:
            DataFrame with volume profile features
        """
        result = df.copy()
        volume = result["volume"]

        # Fit seasonality
        if fit_seasonality:
            self.seasonality.fit(result)

        # Hour of day
        result["hour"] = result.index.hour

        # Expected volume ratio for each hour
        result["expected_vol_ratio"] = result["hour"].map(
            lambda h: self.seasonality.get_expected_volume_ratio(h)
        )

        # Daily average volume
        daily_avg = volume.rolling(
            window=20 * 7, min_periods=20
        ).mean()  # ~20 days for hourly
        result["daily_avg_volume"] = daily_avg

        # Seasonality-adjusted volume ratio
        result["adjusted_vol_ratio"] = result.apply(
            lambda row: self.seasonality.get_adjusted_volume_ratio(
                row["volume"],
                row["hour"],
                row["daily_avg_volume"],
            ),
            axis=1,
        )

        # Volume anomaly score
        vol_mean = result["adjusted_vol_ratio"].rolling(20).mean()
        vol_std = result["adjusted_vol_ratio"].rolling(20).std()
        result["volume_anomaly_zscore"] = (
            result["adjusted_vol_ratio"] - vol_mean
        ) / vol_std

        # Volume trend (is volume picking up or declining)
        result["volume_trend"] = volume.rolling(5).mean() / volume.rolling(20).mean()

        # High/low volume flags
        result["unusual_high_volume"] = (result["volume_anomaly_zscore"] > 2).astype(
            int
        )
        result["unusual_low_volume"] = (result["volume_anomaly_zscore"] < -2).astype(
            int
        )

        # Volume concentration (what % of daily volume in this bar)
        daily_volume = volume.resample("D").sum().reindex(result.index, method="ffill")
        result["volume_concentration"] = volume / daily_volume * 100

        return result

    def get_volume_distribution(
        self,
        df: pd.DataFrame,
        price_levels: int = 20,
    ) -> pd.DataFrame:
        """
        Calculate volume distribution across price levels.

        Creates a volume profile showing volume at each price level.

        Args:
            df: OHLCV DataFrame
            price_levels: Number of price buckets

        Returns:
            DataFrame with price levels and volume
        """
        typical_price = (df["high"] + df["low"] + df["close"]) / 3

        price_min = typical_price.min()
        price_max = typical_price.max()

        # Create price buckets
        bins = np.linspace(price_min, price_max, price_levels + 1)
        labels = [(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)]

        df_copy = df.copy()
        df_copy["price_level"] = pd.cut(typical_price, bins=bins, labels=labels)

        # Sum volume at each level
        volume_profile = df_copy.groupby("price_level")["volume"].sum()

        return pd.DataFrame(
            {
                "price_level": volume_profile.index.astype(float),
                "volume": volume_profile.values,
                "volume_pct": volume_profile.values / volume_profile.sum() * 100,
            }
        )

    def find_high_volume_nodes(
        self,
        df: pd.DataFrame,
        threshold_pct: float = 70,
    ) -> List[float]:
        """
        Find high-volume price nodes (support/resistance).

        Args:
            df: OHLCV DataFrame
            threshold_pct: Percentile threshold for high volume

        Returns:
            List of price levels with high volume
        """
        profile = self.get_volume_distribution(df)

        threshold = np.percentile(profile["volume"], threshold_pct)
        high_volume_nodes = profile[profile["volume"] >= threshold][
            "price_level"
        ].tolist()

        return high_volume_nodes
