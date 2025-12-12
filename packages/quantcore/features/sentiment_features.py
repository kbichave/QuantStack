"""
News sentiment features for trading analysis.

Reusable sentiment feature engineering for any asset class:
- Daily sentiment aggregation
- Rolling window features (7/14/30-day)
- Exponential decay-weighted features
- Sentiment momentum and volatility
- Missing data handling with neutral imputation
"""

from typing import List, Optional
import pandas as pd
import numpy as np
from loguru import logger

from quantcore.features.base import FeatureBase
from quantcore.config.timeframes import Timeframe


class SentimentFeatures(FeatureBase):
    """
    News sentiment features for trading.

    Converts discrete news articles into continuous time-series features
    aligned with price data. Reusable for commodities, equities, or any asset.

    Features:
    - Daily aggregates: mean, std, count, bullish/bearish ratios
    - Rolling windows: 7/14/30-day moving averages
    - Decay-weighted: exponential moving averages
    - Momentum: sentiment change over time
    - Data availability indicator (for missing data handling)
    """

    def __init__(
        self,
        timeframe: Timeframe,
        rolling_windows: List[int] = None,
        decay_halflifes: List[int] = None,
    ):
        """
        Initialize sentiment features calculator.

        Args:
            timeframe: Timeframe for alignment (typically D1 for news)
            rolling_windows: Rolling window sizes in days (default: [7, 14, 30])
            decay_halflifes: Half-life periods for EWM (default: [7, 14])
        """
        super().__init__(timeframe)

        self.rolling_windows = rolling_windows or [7, 14, 30]
        self.decay_halflifes = decay_halflifes or [7, 14]

    def compute(
        self,
        df: pd.DataFrame,
        news_data: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Compute sentiment features.

        Args:
            df: OHLCV DataFrame with DatetimeIndex
            news_data: News sentiment DataFrame from fetch_news_sentiment()
                Expected columns: overall_sentiment_score, overall_sentiment_label,
                ticker_sentiment_score, relevance_score
                Expected index: DatetimeIndex (time_published)

        Returns:
            DataFrame with original data plus sentiment features
        """
        result = df.copy()

        if news_data is None or news_data.empty:
            logger.debug("No news data provided, using neutral sentiment")
            return self._add_neutral_features(result)

        # Aggregate news to daily level
        daily_sentiment = self._aggregate_daily_sentiment(news_data)

        if daily_sentiment.empty:
            logger.debug("No daily sentiment after aggregation")
            return self._add_neutral_features(result)

        # Align with price data
        aligned = self._align_with_price_data(result, daily_sentiment)

        # Merge aligned sentiment into result
        for col in aligned.columns:
            result[col] = aligned[col]

        # Compute rolling features
        result = self._compute_rolling_features(result)

        # Compute decay-weighted features
        result = self._compute_decay_features(result)

        # Compute momentum features
        result = self._compute_momentum_features(result)

        # Compute extreme sentiment signals
        result = self._compute_sentiment_signals(result)

        return result

    def _aggregate_daily_sentiment(
        self,
        news_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Aggregate news articles to daily sentiment features.

        Args:
            news_df: News DataFrame with datetime index

        Returns:
            DataFrame with daily aggregated sentiment
        """
        if news_df.empty:
            return pd.DataFrame()

        # Ensure we have a datetime index
        if not isinstance(news_df.index, pd.DatetimeIndex):
            logger.warning("News data index is not DatetimeIndex")
            return pd.DataFrame()

        # Group by date
        news_df = news_df.copy()
        news_df["date"] = news_df.index.date

        agg_dict = {}

        # Overall sentiment score
        if "overall_sentiment_score" in news_df.columns:
            agg_dict["overall_sentiment_score"] = ["mean", "std", "count"]

        # Ticker-specific sentiment
        if "ticker_sentiment_score" in news_df.columns:
            agg_dict["ticker_sentiment_score"] = "mean"

        # Relevance score
        if "relevance_score" in news_df.columns:
            agg_dict["relevance_score"] = "mean"

        if not agg_dict:
            logger.warning("No sentiment columns found in news data")
            return pd.DataFrame()

        # Perform aggregation
        daily = news_df.groupby("date").agg(agg_dict)

        # Flatten column names
        daily.columns = [
            f"{col}_{stat}" if stat != "mean" else col for col, stat in daily.columns
        ]

        # Rename columns
        column_mapping = {
            "overall_sentiment_score": "news_sentiment_mean",
            "overall_sentiment_score_std": "news_sentiment_std",
            "overall_sentiment_score_count": "news_count",
            "ticker_sentiment_score": "news_ticker_sentiment",
            "relevance_score": "news_relevance_mean",
        }
        daily = daily.rename(columns=column_mapping)

        # Fill NaN std with 0 (single article days)
        if "news_sentiment_std" in daily.columns:
            daily["news_sentiment_std"] = daily["news_sentiment_std"].fillna(0)

        # Compute bullish/bearish ratios if label available
        if "overall_sentiment_label" in news_df.columns:
            label_counts = (
                news_df.groupby("date")["overall_sentiment_label"]
                .apply(
                    lambda x: pd.Series(
                        {
                            "bullish": (
                                (x == "Bullish").sum() / len(x) if len(x) > 0 else 0
                            ),
                            "bearish": (
                                (x == "Bearish").sum() / len(x) if len(x) > 0 else 0
                            ),
                        }
                    )
                )
                .unstack()
            )

            if "bullish" in label_counts.columns:
                daily["news_bullish_ratio"] = label_counts["bullish"]
            if "bearish" in label_counts.columns:
                daily["news_bearish_ratio"] = label_counts["bearish"]

        # Convert index to datetime
        daily.index = pd.to_datetime(daily.index)

        return daily

    def _align_with_price_data(
        self,
        df: pd.DataFrame,
        sentiment_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Align daily sentiment with price data.

        Uses forward-fill for missing days and imputes neutral (0.0)
        for dates with no historical data.

        Args:
            df: Price DataFrame
            sentiment_df: Daily sentiment DataFrame

        Returns:
            Aligned sentiment DataFrame with same index as df
        """
        # Create output DataFrame with price index
        aligned = pd.DataFrame(index=df.index)

        # Get date part of price index
        if isinstance(df.index, pd.DatetimeIndex):
            price_dates = df.index.normalize()
        else:
            price_dates = pd.to_datetime(df.index).normalize()

        # Normalize sentiment dates
        sentiment_dates = sentiment_df.index.normalize()

        # Reindex sentiment to price dates
        for col in sentiment_df.columns:
            # Create series with sentiment dates
            sentiment_series = pd.Series(
                sentiment_df[col].values,
                index=sentiment_dates,
            )

            # Reindex to price dates
            aligned_series = sentiment_series.reindex(price_dates)

            # Forward-fill (use last known sentiment for missing days)
            aligned_series = aligned_series.ffill()

            # Determine imputation value
            if col in ["news_count"]:
                impute_value = 0
            elif col in ["news_bullish_ratio", "news_bearish_ratio"]:
                impute_value = 0.5  # Neutral
            else:
                impute_value = 0.0  # Neutral sentiment

            # Fill remaining NaN (before first news)
            aligned_series = aligned_series.fillna(impute_value)

            # Assign to aligned DataFrame
            aligned[col] = aligned_series.values

        # Add data availability indicator
        has_data = ~sentiment_df["news_sentiment_mean"].reindex(price_dates).isna()
        has_data = has_data.ffill().fillna(False)
        aligned["news_data_available"] = has_data.astype(int).values

        return aligned

    def _compute_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute rolling window features."""
        result = df.copy()

        if "news_sentiment_mean" not in result.columns:
            return result

        sentiment = result["news_sentiment_mean"]

        for window in self.rolling_windows:
            # Moving average
            result[f"news_sentiment_ma_{window}d"] = sentiment.rolling(
                window, min_periods=1
            ).mean()

            # Rolling volatility
            result[f"news_sentiment_vol_{window}d"] = (
                sentiment.rolling(window, min_periods=2).std().fillna(0)
            )

        # Z-score vs 30-day rolling mean
        ma_30 = sentiment.rolling(30, min_periods=1).mean()
        std_30 = sentiment.rolling(30, min_periods=2).std()
        std_30_safe = std_30.replace(0, np.nan)
        result["news_sentiment_zscore"] = (sentiment - ma_30) / std_30_safe
        result["news_sentiment_zscore"] = result["news_sentiment_zscore"].fillna(0)

        return result

    def _compute_decay_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute exponential decay-weighted features."""
        result = df.copy()

        if "news_sentiment_mean" not in result.columns:
            return result

        sentiment = result["news_sentiment_mean"]

        for halflife in self.decay_halflifes:
            # Compute alpha from halflife
            # alpha = 1 - exp(-ln(2) / halflife)
            alpha = 1 - np.exp(-np.log(2) / halflife)

            # EWM
            result[f"news_sentiment_ewm_{halflife}d"] = sentiment.ewm(
                alpha=alpha, adjust=False
            ).mean()

        return result

    def _compute_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute sentiment momentum features."""
        result = df.copy()

        if "news_sentiment_mean" not in result.columns:
            return result

        sentiment = result["news_sentiment_mean"]

        # Sentiment change (momentum)
        result["news_sentiment_change_1d"] = sentiment.diff(1)
        result["news_sentiment_change_7d"] = sentiment.diff(7)

        # Sentiment momentum (rate of change)
        sentiment_safe = sentiment.replace(0, np.nan)
        result["news_sentiment_momentum"] = (
            sentiment.diff(7) / sentiment_safe.shift(7).abs()
        ).fillna(0)

        # Sentiment trend (direction of 7-day MA)
        if "news_sentiment_ma_7d" in result.columns:
            ma_7 = result["news_sentiment_ma_7d"]
            result["news_sentiment_trend"] = np.sign(ma_7.diff(1))

        # Sentiment reversal (change in direction)
        if "news_sentiment_trend" in result.columns:
            result["news_sentiment_reversal"] = (
                result["news_sentiment_trend"].diff().abs() > 0
            ).astype(int)

        return result

    def _compute_sentiment_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute sentiment-based trading signals."""
        result = df.copy()

        if "news_sentiment_mean" not in result.columns:
            return result

        sentiment = result["news_sentiment_mean"]

        # Extreme sentiment (>1.5 std from rolling mean)
        if "news_sentiment_zscore" in result.columns:
            zscore = result["news_sentiment_zscore"]
            result["news_extreme_bullish"] = (zscore > 1.5).astype(int)
            result["news_extreme_bearish"] = (zscore < -1.5).astype(int)

        # Sentiment spike (large daily change)
        if "news_sentiment_change_1d" in result.columns:
            change = result["news_sentiment_change_1d"]
            change_std = change.rolling(30, min_periods=5).std()
            change_std_safe = change_std.replace(0, np.nan)
            result["news_sentiment_spike"] = (
                (change.abs() / change_std_safe) > 2
            ).astype(int)

        # Sentiment divergence from count (high count but neutral sentiment)
        if "news_count" in result.columns:
            count = result["news_count"]
            count_ma = count.rolling(14, min_periods=1).mean()
            high_volume = count > count_ma * 1.5
            neutral_sentiment = sentiment.abs() < 0.1
            result["news_volume_divergence"] = (high_volume & neutral_sentiment).astype(
                int
            )

        return result

    def _add_neutral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add neutral sentiment features when no news data available."""
        result = df.copy()

        for name in self.get_feature_names():
            if name == "news_data_available":
                result[name] = 0
            elif name == "news_count":
                result[name] = 0
            elif name in ["news_bullish_ratio", "news_bearish_ratio"]:
                result[name] = 0.5
            elif (
                "spike" in name
                or "extreme" in name
                or "reversal" in name
                or "divergence" in name
            ):
                result[name] = 0
            else:
                result[name] = 0.0

        return result

    def get_feature_names(self) -> List[str]:
        """Return list of sentiment feature names."""
        features = [
            # Daily aggregates
            "news_sentiment_mean",
            "news_sentiment_std",
            "news_count",
            "news_ticker_sentiment",
            "news_relevance_mean",
            "news_bullish_ratio",
            "news_bearish_ratio",
            "news_data_available",
            # Z-score
            "news_sentiment_zscore",
            # Momentum
            "news_sentiment_change_1d",
            "news_sentiment_change_7d",
            "news_sentiment_momentum",
            "news_sentiment_trend",
            "news_sentiment_reversal",
            # Signals
            "news_extreme_bullish",
            "news_extreme_bearish",
            "news_sentiment_spike",
            "news_volume_divergence",
        ]

        # Rolling windows
        for window in self.rolling_windows:
            features.append(f"news_sentiment_ma_{window}d")
            features.append(f"news_sentiment_vol_{window}d")

        # Decay features
        for halflife in self.decay_halflifes:
            features.append(f"news_sentiment_ewm_{halflife}d")

        return features
