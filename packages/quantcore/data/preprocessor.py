"""
Data preprocessing for market data.

Handles cleaning, validation, gap filling, and timezone normalization.
"""

import pandas as pd
import pytz
from loguru import logger

from quantcore.config.settings import get_settings
from quantcore.config.timeframes import Timeframe


class DataPreprocessor:
    """
    Preprocessor for cleaning and normalizing OHLCV data.

    Features:
    - Timezone normalization
    - Gap detection and handling
    - Outlier detection
    - Data validation
    - Market hours filtering
    """

    # US market hours (Eastern Time)
    MARKET_OPEN_HOUR = 9
    MARKET_OPEN_MINUTE = 30
    MARKET_CLOSE_HOUR = 16
    MARKET_CLOSE_MINUTE = 0

    def __init__(self):
        """Initialize the preprocessor."""
        settings = get_settings()
        self.market_tz = pytz.timezone(settings.market_timezone)

    def preprocess(
        self,
        df: pd.DataFrame,
        timeframe: Timeframe,
        remove_outliers: bool = True,
        fill_gaps: bool = False,
        market_hours_only: bool = True,
    ) -> pd.DataFrame:
        """
        Full preprocessing pipeline.

        Args:
            df: Raw OHLCV DataFrame
            timeframe: Data timeframe
            remove_outliers: Whether to remove outliers
            fill_gaps: Whether to fill gaps (forward fill)
            market_hours_only: Filter to market hours (for intraday only)

        Returns:
            Cleaned DataFrame
        """
        if df.empty:
            return df.copy()

        result = df.copy()

        # Ensure DatetimeIndex
        if not isinstance(result.index, pd.DatetimeIndex):
            logger.warning("Converting index to DatetimeIndex")
            result.index = pd.to_datetime(result.index)

        # Normalize timezone
        result = self._normalize_timezone(result)

        # Sort by timestamp
        result = result.sort_index()

        # Remove duplicates
        initial_len = len(result)
        result = result[~result.index.duplicated(keep="last")]
        if len(result) < initial_len:
            logger.info(f"Removed {initial_len - len(result)} duplicate rows")

        # Filter market hours for intraday data
        if market_hours_only and timeframe in [Timeframe.H1, Timeframe.H4]:
            result = self._filter_market_hours(result)

        # Remove outliers
        if remove_outliers:
            result = self._remove_outliers(result)

        # Validate OHLCV relationships
        result = self._validate_ohlcv(result)

        # Fill gaps if requested
        if fill_gaps:
            result = self._fill_gaps(result, timeframe)

        # Drop rows with NaN values
        result = result.dropna()

        logger.info(f"Preprocessed data: {len(result)} rows")
        return result

    def _normalize_timezone(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize index to market timezone."""
        if df.index.tz is None:
            # Assume UTC if no timezone
            df.index = df.index.tz_localize("UTC").tz_convert(self.market_tz)
        elif df.index.tz != self.market_tz:
            df.index = df.index.tz_convert(self.market_tz)
        return df

    def _filter_market_hours(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter to regular market hours only."""
        if df.empty:
            return df

        # Get hour and minute
        hours = df.index.hour

        # Market hours: 9:30 AM to 4:00 PM ET
        # For hourly data, we include bars starting from 9:00 (covers 9:30)
        # through 15:00 (covers until 16:00)
        market_open = self.MARKET_OPEN_HOUR
        market_close = self.MARKET_CLOSE_HOUR

        # Include bars where trading occurs
        mask = (hours >= market_open) & (hours < market_close)

        # Also filter weekends
        mask = mask & (df.index.dayofweek < 5)

        filtered = df[mask]

        if len(filtered) < len(df):
            logger.debug(f"Filtered {len(df) - len(filtered)} non-market-hours rows")

        return filtered

    def _remove_outliers(
        self,
        df: pd.DataFrame,
        price_zscore_threshold: float = 5.0,
        volume_zscore_threshold: float = 10.0,
    ) -> pd.DataFrame:
        """
        Remove outliers based on z-score.

        Uses rolling statistics to detect anomalous values.
        """
        if df.empty or len(df) < 20:
            return df

        result = df.copy()

        # Calculate returns for price outlier detection
        returns = result["close"].pct_change()

        # Rolling z-score of returns
        rolling_mean = returns.rolling(20, min_periods=5).mean()
        rolling_std = returns.rolling(20, min_periods=5).std()
        zscore = (returns - rolling_mean) / rolling_std

        # Mark outliers
        price_outliers = zscore.abs() > price_zscore_threshold

        # Volume outliers
        vol_rolling_mean = result["volume"].rolling(20, min_periods=5).mean()
        vol_rolling_std = result["volume"].rolling(20, min_periods=5).std()
        vol_zscore = (result["volume"] - vol_rolling_mean) / vol_rolling_std
        volume_outliers = vol_zscore.abs() > volume_zscore_threshold

        # Remove outliers
        outliers = price_outliers | volume_outliers
        outliers = outliers.fillna(False)

        if outliers.sum() > 0:
            logger.info(f"Removing {outliers.sum()} outlier rows")
            result = result[~outliers]

        return result

    def _validate_ohlcv(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and fix OHLCV relationships."""
        result = df.copy()

        # Fix high < low (swap them)
        invalid_hl = result["high"] < result["low"]
        if invalid_hl.sum() > 0:
            logger.warning(f"Fixing {invalid_hl.sum()} rows with high < low")
            result.loc[invalid_hl, ["high", "low"]] = result.loc[invalid_hl, ["low", "high"]].values

        # Ensure high >= max(open, close)
        max_oc = result[["open", "close"]].max(axis=1)
        invalid_h = result["high"] < max_oc
        if invalid_h.sum() > 0:
            logger.warning(f"Fixing {invalid_h.sum()} rows with high < max(open, close)")
            result.loc[invalid_h, "high"] = max_oc[invalid_h]

        # Ensure low <= min(open, close)
        min_oc = result[["open", "close"]].min(axis=1)
        invalid_l = result["low"] > min_oc
        if invalid_l.sum() > 0:
            logger.warning(f"Fixing {invalid_l.sum()} rows with low > min(open, close)")
            result.loc[invalid_l, "low"] = min_oc[invalid_l]

        # Ensure positive prices
        for col in ["open", "high", "low", "close"]:
            invalid = result[col] <= 0
            if invalid.sum() > 0:
                logger.warning(f"Removing {invalid.sum()} rows with non-positive {col}")
                result = result[~invalid]

        # Ensure non-negative volume
        invalid_vol = result["volume"] < 0
        if invalid_vol.sum() > 0:
            logger.warning(f"Fixing {invalid_vol.sum()} rows with negative volume")
            result.loc[invalid_vol, "volume"] = 0

        return result

    def _fill_gaps(
        self,
        df: pd.DataFrame,
        timeframe: Timeframe,
        max_gap_periods: int = 5,
    ) -> pd.DataFrame:
        """
        Fill gaps in data using forward fill.

        Only fills small gaps; larger gaps are left as NaN.
        """
        if df.empty:
            return df

        # Determine expected frequency
        freq_map = {
            Timeframe.H1: "1H",
            Timeframe.H4: "4H",
            Timeframe.D1: "1D",
            Timeframe.W1: "1W",
        }
        freq = freq_map.get(timeframe, "1H")

        # Reindex to expected frequency
        full_index = pd.date_range(
            start=df.index.min(),
            end=df.index.max(),
            freq=freq,
            tz=df.index.tz,
        )

        result = df.reindex(full_index)

        # Forward fill, but only for small gaps
        for col in result.columns:
            result[col] = result[col].fillna(method="ffill", limit=max_gap_periods)

        # Remove rows that couldn't be filled
        result = result.dropna()

        if len(result) > len(df):
            logger.info(f"Filled {len(result) - len(df)} gap rows")

        return result

    def detect_gaps(
        self,
        df: pd.DataFrame,
        timeframe: Timeframe,
    ) -> pd.DataFrame:
        """
        Detect gaps in the data.

        Args:
            df: OHLCV DataFrame
            timeframe: Data timeframe

        Returns:
            DataFrame with gap information
        """
        if df.empty or len(df) < 2:
            return pd.DataFrame(columns=["start", "end", "gap_periods"])

        # Calculate expected time delta
        delta_map = {
            Timeframe.H1: pd.Timedelta(hours=1),
            Timeframe.H4: pd.Timedelta(hours=4),
            Timeframe.D1: pd.Timedelta(days=1),
            Timeframe.W1: pd.Timedelta(weeks=1),
        }
        expected_delta = delta_map.get(timeframe, pd.Timedelta(hours=1))

        # Find gaps
        time_diffs = pd.Series(df.index).diff()
        gaps = time_diffs > expected_delta * 1.5  # Allow some tolerance

        if not gaps.any():
            return pd.DataFrame(columns=["start", "end", "gap_periods"])

        # Build gap report
        gap_starts = df.index[:-1][gaps.values[1:]]
        gap_ends = df.index[1:][gaps.values[1:]]
        gap_periods = (gap_ends - gap_starts) / expected_delta

        return pd.DataFrame(
            {
                "start": gap_starts,
                "end": gap_ends,
                "gap_periods": gap_periods.astype(int),
            }
        )

    def get_data_quality_report(
        self,
        df: pd.DataFrame,
        timeframe: Timeframe,
    ) -> dict:
        """
        Generate a data quality report.

        Args:
            df: OHLCV DataFrame
            timeframe: Data timeframe

        Returns:
            Dictionary with quality metrics
        """
        if df.empty:
            return {"status": "empty", "rows": 0}

        report = {
            "rows": len(df),
            "start_date": df.index.min().isoformat(),
            "end_date": df.index.max().isoformat(),
            "missing_values": df.isna().sum().to_dict(),
            "duplicates": df.index.duplicated().sum(),
        }

        # Gap analysis
        gaps = self.detect_gaps(df, timeframe)
        report["gaps"] = {
            "count": len(gaps),
            "total_missing_periods": gaps["gap_periods"].sum() if len(gaps) > 0 else 0,
        }

        # OHLCV validity
        report["invalid_ohlcv"] = {
            "high_lt_low": (df["high"] < df["low"]).sum(),
            "high_lt_close": (df["high"] < df["close"]).sum(),
            "low_gt_close": (df["low"] > df["close"]).sum(),
            "negative_prices": (df[["open", "high", "low", "close"]] < 0).any(axis=1).sum(),
            "negative_volume": (df["volume"] < 0).sum(),
        }

        # Basic statistics
        report["price_stats"] = {
            "mean": df["close"].mean(),
            "std": df["close"].std(),
            "min": df["close"].min(),
            "max": df["close"].max(),
        }

        return report
