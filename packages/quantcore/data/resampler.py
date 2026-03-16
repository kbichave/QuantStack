"""
Multi-timeframe bar resampling.

Provides:
- Build 4H bars from 1H data
- Align daily/weekly bars to hourly timestamps
- Cross-timeframe feature alignment
"""


import numpy as np
import pandas as pd
from loguru import logger

from quantcore.config.timeframes import Timeframe


class TimeframeResampler:
    """
    Resamples OHLCV data across timeframes.

    Key features:
    - Build higher timeframe bars from lower TF data
    - Align higher TF data to lower TF timestamps (forward-fill)
    - Proper handling of market hours and sessions
    """

    # Resample rules for each timeframe.
    # Values are pandas offset aliases — must match what pd.DataFrame.resample() accepts.
    # "1D" / "W-FRI" for daily/weekly; "Xh" for hours; "Xmin" for minutes; "Xs" for seconds.
    # Note: "1M" in pandas means 1 Month — use "1min" for 1-minute bars.
    RESAMPLE_RULES = {
        Timeframe.W1:  "W-FRI",   # Week ending Friday
        Timeframe.D1:  "1D",
        Timeframe.H4:  "4h",
        Timeframe.H1:  "1h",
        Timeframe.M30: "30min",
        Timeframe.M15: "15min",
        Timeframe.M5:  "5min",
        Timeframe.M1:  "1min",
        Timeframe.S5:  "5s",
    }

    def __init__(
        self,
        market_open: str = "09:30",
        market_close: str = "16:00",
        timezone: str = "America/New_York",
    ):
        """
        Initialize resampler.

        Args:
            market_open: Market open time (HH:MM)
            market_close: Market close time (HH:MM)
            timezone: Market timezone
        """
        self.market_open = market_open
        self.market_close = market_close
        self.timezone = timezone

    def resample_to_higher_tf(
        self,
        df: pd.DataFrame,
        target_tf: Timeframe,
    ) -> pd.DataFrame:
        """
        Resample OHLCV data to a higher timeframe.

        Args:
            df: DataFrame with OHLCV data and DatetimeIndex
            target_tf: Target timeframe to resample to

        Returns:
            Resampled DataFrame
        """
        if df.empty:
            return df

        rule = self.RESAMPLE_RULES[target_tf]

        # Resample with proper OHLCV aggregation
        resampled = df.resample(rule).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        })

        # Drop rows where all values are NaN (no data in that period)
        resampled = resampled.dropna(how="all")

        logger.debug(f"Resampled {len(df)} bars to {len(resampled)} {target_tf.value} bars")
        return resampled

    def build_4h_from_1h(self, df_1h: pd.DataFrame) -> pd.DataFrame:
        """
        Build 4-hour bars from 1-hour data.

        4H bars are anchored to market hours:
        - 09:30-13:30 (first 4H bar)
        - 13:30-16:00 (partial second bar, or extended hours)

        Args:
            df_1h: 1-hour OHLCV DataFrame

        Returns:
            4-hour OHLCV DataFrame
        """
        if df_1h.empty:
            return df_1h

        # Use standard 4H resampling
        df_4h = self.resample_to_higher_tf(df_1h, Timeframe.H4)

        return df_4h

    def build_daily_from_intraday(self, df_intraday: pd.DataFrame) -> pd.DataFrame:
        """
        Build daily bars from intraday data.

        Args:
            df_intraday: Intraday OHLCV DataFrame (1H or smaller)

        Returns:
            Daily OHLCV DataFrame
        """
        if df_intraday.empty:
            return df_intraday

        return self.resample_to_higher_tf(df_intraday, Timeframe.D1)

    def build_weekly_from_daily(self, df_daily: pd.DataFrame) -> pd.DataFrame:
        """
        Build weekly bars from daily data.

        Week ends on Friday.

        Args:
            df_daily: Daily OHLCV DataFrame

        Returns:
            Weekly OHLCV DataFrame
        """
        if df_daily.empty:
            return df_daily

        return self.resample_to_higher_tf(df_daily, Timeframe.W1)

    def align_higher_tf_to_lower(
        self,
        df_lower: pd.DataFrame,
        df_higher: pd.DataFrame,
        prefix: str = "",
    ) -> pd.DataFrame:
        """
        Align higher timeframe data to lower timeframe index.

        Uses shift(1) + forward-fill to propagate higher TF values to lower TF bars.
        This ensures no lookahead bias - at any lower TF bar, we only see
        the most recently COMPLETED higher TF bar, not the current incomplete one.

        Example:
            - At Monday 10:00 AM, daily features come from Friday's completed bar
            - At Monday 4:00 PM, daily features still come from Friday's bar
            - At Tuesday 9:30 AM, daily features come from Monday's completed bar

        Args:
            df_lower: Lower timeframe DataFrame (target index)
            df_higher: Higher timeframe DataFrame to align
            prefix: Column name prefix for aligned data

        Returns:
            DataFrame with higher TF data aligned to lower TF index
        """
        if df_lower.empty or df_higher.empty:
            return df_lower

        # CRITICAL: Shift higher TF data by 1 period BEFORE reindexing
        # This ensures we only see COMPLETED bars, avoiding lookahead bias
        # Without shift: at 9:30 AM Monday, we'd see Monday's daily bar (lookahead!)
        # With shift: at 9:30 AM Monday, we see Friday's completed daily bar (correct)
        df_higher_shifted = df_higher.shift(1)

        # Reindex shifted data to lower TF with forward fill
        aligned = df_higher_shifted.reindex(df_lower.index, method="ffill")

        # Add prefix to column names
        if prefix:
            aligned = aligned.add_prefix(f"{prefix}_")

        return aligned

    def build_multi_timeframe_dataset(
        self,
        df_1h: pd.DataFrame,
        include_4h: bool = True,
        include_daily: bool = True,
        include_weekly: bool = True,
    ) -> dict[Timeframe, pd.DataFrame]:
        """
        Build complete multi-timeframe dataset from hourly data.

        Args:
            df_1h: 1-hour OHLCV DataFrame
            include_4h: Include 4-hour bars
            include_daily: Include daily bars
            include_weekly: Include weekly bars

        Returns:
            Dictionary mapping Timeframe to DataFrame
        """
        result = {Timeframe.H1: df_1h}

        if include_4h:
            result[Timeframe.H4] = self.build_4h_from_1h(df_1h)

        if include_daily:
            result[Timeframe.D1] = self.build_daily_from_intraday(df_1h)

        if include_weekly:
            if Timeframe.D1 in result:
                result[Timeframe.W1] = self.build_weekly_from_daily(result[Timeframe.D1])
            else:
                # Build weekly directly from hourly
                daily = self.build_daily_from_intraday(df_1h)
                result[Timeframe.W1] = self.build_weekly_from_daily(daily)

        return result

    def get_last_completed_bar(
        self,
        df: pd.DataFrame,
        target_tf: Timeframe,
        as_of: pd.Timestamp,
    ) -> pd.Series | None:
        """
        Get the last completed bar for a timeframe as of a given timestamp.

        This is critical for avoiding lookahead bias - we only want to use
        bars that have fully completed before the as_of time.

        Args:
            df: DataFrame with OHLCV data for target timeframe
            target_tf: Timeframe of the data
            as_of: Reference timestamp

        Returns:
            Series with the last completed bar, or None if no valid bar
        """
        if df.empty:
            return None

        # Get all bars that ended before as_of
        completed_bars = df[df.index < as_of]

        if completed_bars.empty:
            return None

        return completed_bars.iloc[-1]

    def compute_partial_candle_features(
        self,
        df_1h: pd.DataFrame,
        target_tf: Timeframe,
    ) -> pd.DataFrame:
        """
        Compute features for the current in-progress candle at a higher timeframe.

        These features capture real-time context WITHOUT lookahead:
        - Partial candle's open, high, low (expanding from period start)
        - Current price position within partial candle's range
        - Progress through the period (e.g., 3rd hour of 7-hour day)

        Example at Monday 11:00 AM:
        - partial_D1_open: Monday's opening price (9:30)
        - partial_D1_high: Highest price so far today (9:30-11:00)
        - partial_D1_low: Lowest price so far today (9:30-11:00)
        - partial_D1_range_position: Where current close is in today's range
        - partial_D1_progress: ~0.28 (2 hours into ~7 hour day)

        Args:
            df_1h: 1-hour OHLCV DataFrame
            target_tf: Target timeframe (H4, D1, or W1)

        Returns:
            DataFrame with partial candle features aligned to 1H index
        """
        if df_1h.empty:
            return pd.DataFrame(index=df_1h.index)

        rule = self.RESAMPLE_RULES[target_tf]
        prefix = f"partial_{target_tf.value}"

        # Create period grouper
        grouper = pd.Grouper(freq=rule)

        # For each 1H bar, compute expanding stats from start of its period
        results = []

        for _period_start, period_group in df_1h.groupby(grouper):
            if period_group.empty:
                continue

            # Expanding open (first value of period, constant)
            period_open = period_group["open"].iloc[0]

            # Expanding high/low (cumulative max/min from period start)
            expanding_high = period_group["high"].expanding().max()
            expanding_low = period_group["low"].expanding().min()

            # Current close
            current_close = period_group["close"]

            # Range position: where is current close within partial bar's range?
            # 0 = at low, 1 = at high, 0.5 = middle
            partial_range = expanding_high - expanding_low
            range_position = np.where(
                partial_range > 0,
                (current_close - expanding_low) / partial_range,
                0.5  # If range is 0, position is middle
            )

            # Progress through period (0 to 1)
            n_bars = len(period_group)
            progress = np.arange(1, n_bars + 1) / max(n_bars, 1)

            # Distance from period open
            distance_from_open = (current_close - period_open) / period_open

            # Build features for this period
            period_features = pd.DataFrame({
                f"{prefix}_open": period_open,
                f"{prefix}_high": expanding_high.values,
                f"{prefix}_low": expanding_low.values,
                f"{prefix}_range": partial_range.values,
                f"{prefix}_range_position": range_position,
                f"{prefix}_progress": progress,
                f"{prefix}_distance_from_open": distance_from_open.values,
            }, index=period_group.index)

            results.append(period_features)

        if not results:
            return pd.DataFrame(index=df_1h.index)

        return pd.concat(results).reindex(df_1h.index)

    def align_all_timeframes(
        self,
        data: dict[Timeframe, pd.DataFrame],
        base_tf: Timeframe = Timeframe.H1,
        include_partial_candles: bool = True,
    ) -> pd.DataFrame:
        """
        Align all timeframes to base timeframe index.

        Creates a single DataFrame with columns prefixed by timeframe.
        Higher TF data is forward-filled to ensure no lookahead.

        Includes two types of higher TF features:
        1. Previous completed candle (shifted by 1 period) - e.g., "1D_close"
        2. Current partial candle (in-progress) - e.g., "partial_1D_high"

        Args:
            data: Dictionary of Timeframe -> DataFrame
            base_tf: Base timeframe to align to
            include_partial_candles: Include partial candle features (default True)

        Returns:
            Single DataFrame with all timeframes aligned
        """
        if base_tf not in data:
            raise ValueError(f"Base timeframe {base_tf} not in data")

        base_df = data[base_tf].copy()

        # Collect all DataFrames to concat at once (avoids fragmentation)
        dfs_to_concat = [base_df.add_prefix(f"{base_tf.value}_")]

        # Align each higher timeframe
        from quantcore.config.timeframes import TIMEFRAME_HIERARCHY

        for tf in TIMEFRAME_HIERARCHY:
            if tf == base_tf or tf not in data:
                continue

            # 1. Previous completed candle features (with shift to avoid lookahead)
            aligned = self.align_higher_tf_to_lower(
                base_df,
                data[tf],
                prefix=tf.value,
            )
            dfs_to_concat.append(aligned)

            # 2. Current partial candle features (no lookahead - uses data up to current bar)
            if include_partial_candles:
                partial_features = self.compute_partial_candle_features(
                    base_df,
                    tf,
                )
                dfs_to_concat.append(partial_features)

        # Concat all at once to avoid DataFrame fragmentation
        return pd.concat(dfs_to_concat, axis=1)


    def build_multi_timeframe_intraday(
        self,
        df_1m: pd.DataFrame,
        include_m5: bool = True,
        include_m15: bool = True,
        include_m30: bool = True,
        include_h1: bool = True,
    ) -> dict[Timeframe, pd.DataFrame]:
        """Build an intraday multi-timeframe dataset from 1-minute bars.

        Analogous to ``build_multi_timeframe_dataset`` but anchored at 1-minute
        resolution instead of 1-hour.  The M5/M15/M30/H1 bars are produced by
        resampling so you get a self-consistent set without fetching each
        timeframe separately from the broker.

        Args:
            df_1m: 1-minute OHLCV DataFrame with DatetimeIndex.
            include_m5:  Include 5-minute bars.
            include_m15: Include 15-minute bars.
            include_m30: Include 30-minute bars.
            include_h1:  Include hourly bars (bridging intraday ↔ swing pipeline).

        Returns:
            Dict mapping Timeframe → DataFrame.
        """
        result: dict[Timeframe, pd.DataFrame] = {Timeframe.M1: df_1m}

        if include_m5:
            result[Timeframe.M5] = self.resample_to_higher_tf(df_1m, Timeframe.M5)
        if include_m15:
            result[Timeframe.M15] = self.resample_to_higher_tf(df_1m, Timeframe.M15)
        if include_m30:
            result[Timeframe.M30] = self.resample_to_higher_tf(df_1m, Timeframe.M30)
        if include_h1:
            result[Timeframe.H1] = self.resample_to_higher_tf(df_1m, Timeframe.H1)

        return result


def build_multi_tf_from_hourly(
    df_1h: pd.DataFrame,
) -> dict[Timeframe, pd.DataFrame]:
    """
    Convenience function to build multi-TF dataset from hourly data.

    Args:
        df_1h: 1-hour OHLCV DataFrame

    Returns:
        Dictionary mapping Timeframe to DataFrame
    """
    resampler = TimeframeResampler()
    return resampler.build_multi_timeframe_dataset(df_1h)


def align_daily_to_hourly(
    df_hourly: pd.DataFrame,
    df_daily: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convenience function to align daily data to hourly index.

    Args:
        df_hourly: Hourly DataFrame (provides target index)
        df_daily: Daily DataFrame to align

    Returns:
        Daily data aligned to hourly index
    """
    resampler = TimeframeResampler()
    return resampler.align_higher_tf_to_lower(df_hourly, df_daily, prefix="D1")


# Backward compatibility alias
OHLCVResampler = TimeframeResampler
