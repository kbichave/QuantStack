"""
Commodity feature factory.

Orchestrates all commodity-specific feature calculations.
"""

from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from loguru import logger

from quantcore.data.base import AssetClass
from quantcore.config.timeframes import Timeframe
from quantcore.features.trend import TrendFeatures
from quantcore.features.momentum import MomentumFeatures
from quantcore.features.volatility import VolatilityFeatures
from quantcore.features.volume import VolumeFeatures
from quantcore.features.technical_indicators import TechnicalIndicators
from quantcore.features.market_structure import MarketStructureFeatures
from quantcore.features.gann import GannFeatures
from quantcore.features.sentiment_features import SentimentFeatures
from quantcore.features.commodity.spread_features import SpreadFeatures
from quantcore.features.commodity.curve_features import CurveFeatures
from quantcore.features.commodity.seasonality_features import SeasonalityFeatures
from quantcore.features.commodity.event_features import EventFeatures
from quantcore.features.commodity.microstructure_features import MicrostructureFeatures
from quantcore.features.commodity.cross_asset_features import CrossAssetFeatures


class CommodityFeatureFactory:
    """
    Feature factory for commodities.

    Orchestrates both shared TA features and commodity-specific features:
    - Spread features (WTI-Brent, crack spreads)
    - Curve features (contango/backwardation)
    - Seasonality features (time-of-day, EIA cycle)
    - Event features (EIA, OPEC proximity)
    - Microstructure features (volume imbalance, VWAP)
    - Cross-asset features (USD, XLE correlations)
    - Gann features (swing/pivot points, retracements, price-time)
    - Market structure features (swings, support/resistance)
    - Mean reversion features (z-score, percentile rank)
    - News sentiment features
    """

    def __init__(
        self,
        include_technical_indicators: bool = True,
        include_spread_features: bool = True,
        include_curve_features: bool = True,
        include_seasonality_features: bool = True,
        include_event_features: bool = True,
        include_microstructure_features: bool = True,
        include_cross_asset_features: bool = True,
        include_gann_features: bool = True,
        include_market_structure_features: bool = True,
        include_mean_reversion_features: bool = True,
        include_sentiment_features: bool = True,
        enable_moving_averages: bool = True,
        enable_oscillators: bool = True,
        enable_volatility_indicators: bool = True,
        enable_volume_indicators: bool = True,
        mr_lookback: int = 20,
        mr_zscore_threshold: float = 2.0,
    ):
        """
        Initialize commodity feature factory.

        Args:
            include_technical_indicators: Include standard TA indicators
            include_spread_features: Include spread-based features
            include_curve_features: Include curve/term structure features
            include_seasonality_features: Include time-based features
            include_event_features: Include event proximity features
            include_microstructure_features: Include volume/VWAP features
            include_cross_asset_features: Include cross-asset correlation features
            include_gann_features: Include Gann swing/retracement features
            include_market_structure_features: Include swing/market structure features
            include_mean_reversion_features: Include mean reversion features
            include_sentiment_features: Include news sentiment features
            enable_moving_averages: Enable MA indicators
            enable_oscillators: Enable oscillator indicators
            enable_volatility_indicators: Enable volatility indicators
            enable_volume_indicators: Enable volume indicators
            mr_lookback: Mean reversion lookback period
            mr_zscore_threshold: Z-score threshold for mean reversion signals
        """
        self.include_technical_indicators = include_technical_indicators
        self.include_spread_features = include_spread_features
        self.include_curve_features = include_curve_features
        self.include_seasonality_features = include_seasonality_features
        self.include_event_features = include_event_features
        self.include_microstructure_features = include_microstructure_features
        self.include_cross_asset_features = include_cross_asset_features
        self.include_gann_features = include_gann_features
        self.include_market_structure_features = include_market_structure_features
        self.include_mean_reversion_features = include_mean_reversion_features
        self.include_sentiment_features = include_sentiment_features

        # TA indicator settings
        self.enable_moving_averages = enable_moving_averages
        self.enable_oscillators = enable_oscillators
        self.enable_volatility_indicators = enable_volatility_indicators
        self.enable_volume_indicators = enable_volume_indicators

        # Mean reversion settings
        self.mr_lookback = mr_lookback
        self.mr_zscore_threshold = mr_zscore_threshold

    def get_asset_class(self) -> AssetClass:
        """Get asset class this factory supports."""
        return AssetClass.COMMODITY_FUTURES

    def compute_features(
        self,
        data: Dict[Timeframe, pd.DataFrame],
        spread_data: Optional[pd.DataFrame] = None,
        crack_spread_data: Optional[pd.DataFrame] = None,
        curve_data: Optional[Dict[str, pd.DataFrame]] = None,
        event_calendar: Optional[pd.DataFrame] = None,
        cross_asset_data: Optional[Dict[str, pd.DataFrame]] = None,
        news_sentiment_data: Optional[pd.DataFrame] = None,
    ) -> Dict[Timeframe, pd.DataFrame]:
        """
        Compute all features for commodity data.

        Args:
            data: Dictionary mapping timeframe to OHLCV DataFrame
            spread_data: Pre-computed WTI-Brent spread data
            crack_spread_data: Pre-computed crack spread data
            curve_data: Futures curve data
            event_calendar: Event calendar DataFrame
            cross_asset_data: Cross-asset OHLCV data
            news_sentiment_data: News sentiment DataFrame from fetch_news_sentiment()

        Returns:
            Dictionary mapping timeframe to DataFrame with features
        """
        result = {}

        for timeframe, df in data.items():
            if df.empty:
                result[timeframe] = df
                continue

            logger.info(f"Computing commodity features for {timeframe.value}")

            # Compute features for this timeframe
            featured_df = self._compute_timeframe_features(
                df=df,
                timeframe=timeframe,
                spread_data=spread_data,
                crack_spread_data=crack_spread_data,
                curve_data=curve_data,
                event_calendar=event_calendar,
                cross_asset_data=cross_asset_data,
                news_sentiment_data=news_sentiment_data,
            )

            result[timeframe] = featured_df
            logger.info(f"  {timeframe.value}: {len(featured_df.columns)} features")

        return result

    def _compute_timeframe_features(
        self,
        df: pd.DataFrame,
        timeframe: Timeframe,
        spread_data: Optional[pd.DataFrame] = None,
        crack_spread_data: Optional[pd.DataFrame] = None,
        curve_data: Optional[Dict[str, pd.DataFrame]] = None,
        event_calendar: Optional[pd.DataFrame] = None,
        cross_asset_data: Optional[Dict[str, pd.DataFrame]] = None,
        news_sentiment_data: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Compute all features for a single timeframe."""
        result = df.copy()

        # 1. Standard TA features
        if self.include_technical_indicators:
            result = self._compute_ta_features(result, timeframe)

        # 2. Spread features
        if self.include_spread_features:
            spread_calc = SpreadFeatures(timeframe)
            result = spread_calc.compute(result, spread_data, crack_spread_data)

        # 3. Curve features
        if self.include_curve_features:
            curve_calc = CurveFeatures(timeframe)
            result = curve_calc.compute(result, curve_data)

        # 4. Seasonality features
        if self.include_seasonality_features:
            seasonality_calc = SeasonalityFeatures(timeframe)
            result = seasonality_calc.compute(result)

        # 5. Event features
        if self.include_event_features:
            event_calc = EventFeatures(timeframe)
            result = event_calc.compute(result, event_calendar)

        # 6. Microstructure features
        if self.include_microstructure_features:
            micro_calc = MicrostructureFeatures(timeframe)
            result = micro_calc.compute(result)

        # 7. Cross-asset features
        if self.include_cross_asset_features:
            cross_calc = CrossAssetFeatures(timeframe)
            result = cross_calc.compute(result, cross_asset_data)

        # 8. Market structure features (swings)
        if self.include_market_structure_features:
            market_structure = MarketStructureFeatures(timeframe)
            result = market_structure.compute(result)

        # 9. Gann features
        if self.include_gann_features:
            gann = GannFeatures(timeframe)
            result = gann.compute(result)

        # 10. Mean reversion features
        if self.include_mean_reversion_features:
            result = self._compute_mean_reversion_features(result)

        # 11. News sentiment features
        if self.include_sentiment_features:
            sentiment = SentimentFeatures(timeframe)
            result = sentiment.compute(result, news_sentiment_data)

        # Drop any fully empty columns
        result = result.dropna(axis=1, how="all")

        return result

    def _compute_mean_reversion_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute mean reversion specific features.

        Features:
        - Z-score of price
        - Z-score change (reversion confirmation)
        - Distance from moving average
        - Percentile rank
        - Mean reversion signals
        """
        result = df.copy()
        close = result["close"]

        # Z-score of price
        ma = close.rolling(self.mr_lookback).mean()
        std = close.rolling(self.mr_lookback).std()
        result["mr_zscore"] = (close - ma) / (std + 1e-8)

        # Z-score change (for reversion confirmation)
        result["mr_zscore_change"] = result["mr_zscore"].diff()

        # Reversion signal
        # Long: was oversold, now turning up
        oversold = result["mr_zscore"].shift(1) < -self.mr_zscore_threshold
        turning_up = result["mr_zscore_change"] > 0.2
        result["mr_long_signal"] = (oversold & turning_up).astype(int)

        # Short: was overbought, now turning down
        overbought = result["mr_zscore"].shift(1) > self.mr_zscore_threshold
        turning_down = result["mr_zscore_change"] < -0.2
        result["mr_short_signal"] = (overbought & turning_down).astype(int)

        # Distance from MA (percentage)
        result["mr_ma_distance"] = (close - ma) / ma * 100

        # Percentile rank
        result["mr_percentile"] = close.rolling(self.mr_lookback).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1],
            raw=False,
        )

        # Stretch magnitude (for signal strength)
        result["mr_stretch"] = result["mr_zscore"].abs()

        # Mean reversion target (rolling mean)
        result["mr_target"] = ma

        # Expected return to mean
        result["mr_expected_return"] = (ma - close) / close * 100

        return result

    def _compute_ta_features(
        self,
        df: pd.DataFrame,
        timeframe: Timeframe,
    ) -> pd.DataFrame:
        """Compute standard technical analysis features."""
        result = df.copy()

        # Trend features
        trend = TrendFeatures(timeframe)
        result = trend.compute(result)

        # Momentum features
        momentum = MomentumFeatures(timeframe)
        result = momentum.compute(result)

        # Volatility features
        volatility = VolatilityFeatures(timeframe)
        result = volatility.compute(result)

        # Volume features
        volume = VolumeFeatures(timeframe)
        result = volume.compute(result)

        # Technical indicators (if enabled)
        try:
            ti = TechnicalIndicators(
                timeframe,
                enable_moving_averages=self.enable_moving_averages,
                enable_oscillators=self.enable_oscillators,
                enable_volatility=self.enable_volatility_indicators,
                enable_volume=self.enable_volume_indicators,
                enable_hilbert=False,  # Expensive, disabled by default
            )
            result = ti.compute(result)
        except Exception as e:
            logger.warning(f"Failed to compute technical indicators: {e}")

        return result

    def get_feature_names(self, timeframe: Timeframe) -> List[str]:
        """
        Get list of all feature names for a timeframe.

        Args:
            timeframe: Timeframe to get features for

        Returns:
            List of feature names
        """
        features = []

        # TA features (approximate, depends on actual computation)
        features.extend(
            [
                "ema_fast",
                "ema_medium",
                "ema_slow",
                "rsi",
                "macd",
                "macd_signal",
                "macd_histogram",
                "atr",
                "bbands_upper",
                "bbands_lower",
                "obv",
                "volume_sma",
            ]
        )

        # Spread features
        if self.include_spread_features:
            spread_calc = SpreadFeatures(timeframe)
            features.extend(spread_calc.get_feature_names())

        # Curve features
        if self.include_curve_features:
            curve_calc = CurveFeatures(timeframe)
            features.extend(curve_calc.get_feature_names())

        # Seasonality features
        if self.include_seasonality_features:
            seasonality_calc = SeasonalityFeatures(timeframe)
            features.extend(seasonality_calc.get_feature_names())

        # Event features
        if self.include_event_features:
            event_calc = EventFeatures(timeframe)
            features.extend(event_calc.get_feature_names())

        # Microstructure features
        if self.include_microstructure_features:
            micro_calc = MicrostructureFeatures(timeframe)
            features.extend(micro_calc.get_feature_names())

        # Cross-asset features
        if self.include_cross_asset_features:
            cross_calc = CrossAssetFeatures(timeframe)
            features.extend(cross_calc.get_feature_names())

        # Market structure features
        if self.include_market_structure_features:
            market_structure = MarketStructureFeatures(timeframe)
            features.extend(market_structure.get_feature_names())

        # Gann features
        if self.include_gann_features:
            gann = GannFeatures(timeframe)
            features.extend(gann.get_feature_names())

        # Mean reversion features
        if self.include_mean_reversion_features:
            features.extend(
                [
                    "mr_zscore",
                    "mr_zscore_change",
                    "mr_long_signal",
                    "mr_short_signal",
                    "mr_ma_distance",
                    "mr_percentile",
                    "mr_stretch",
                    "mr_target",
                    "mr_expected_return",
                ]
            )

        # Sentiment features
        if self.include_sentiment_features:
            sentiment = SentimentFeatures(timeframe)
            features.extend(sentiment.get_feature_names())

        return list(set(features))  # Remove duplicates

    def get_feature_groups(self) -> Dict[str, List[str]]:
        """
        Get features organized by group.

        Returns:
            Dictionary mapping group name to list of feature names
        """
        return {
            "ta_trend": ["ema_fast", "ema_medium", "ema_slow", "regression_slope"],
            "ta_momentum": ["rsi", "macd", "macd_signal", "momentum_score"],
            "ta_volatility": ["atr", "bbands_upper", "bbands_lower", "realized_vol"],
            "ta_volume": ["obv", "volume_sma", "relative_volume"],
            "spread": [
                "wti_brent_spread",
                "wti_brent_zscore",
                "crack_spread",
                "crack_zscore",
            ],
            "curve": [
                "is_contango",
                "curve_slope",
                "roll_yield",
                "carry_signal",
            ],
            "seasonality": [
                "is_eia_day",
                "is_driving_season",
                "eia_proximity",
            ],
            "event": [
                "days_to_eia",
                "eia_proximity",
                "high_event_risk",
            ],
            "microstructure": [
                "volume_imbalance",
                "vwap_deviation",
                "gap_pct",
            ],
            "cross_asset": [
                "wti_usd_corr",
                "wti_xle_corr",
                "vix_level",
                "cross_asset_signal",
            ],
            "market_structure": [
                "probable_swing_low",
                "probable_swing_high",
                "bars_since_swing_low",
                "bars_since_swing_high",
                "trend_structure",
                "near_support",
                "near_resistance",
            ],
            "gann": [
                "gann_swing_high",
                "gann_swing_low",
                "gann_retracement_382",
                "gann_retracement_500",
                "gann_retracement_618",
                "gann_range_position",
                "gann_vs_1x1_low",
                "gann_near_any_level",
                "gann_oversold",
                "gann_overbought",
            ],
            "mean_reversion": [
                "mr_zscore",
                "mr_zscore_change",
                "mr_long_signal",
                "mr_short_signal",
                "mr_ma_distance",
                "mr_percentile",
                "mr_stretch",
                "mr_expected_return",
            ],
            "sentiment": [
                "news_sentiment_mean",
                "news_sentiment_zscore",
                "news_sentiment_momentum",
                "news_bullish_ratio",
                "news_extreme_bullish",
                "news_extreme_bearish",
                "news_data_available",
            ],
        }
