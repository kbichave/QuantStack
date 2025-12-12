"""
Multi-timeframe feature factory.

Orchestrates feature computation across all timeframes with cross-TF context injection.
"""

from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from loguru import logger

from quantcore.validation.input_validation import DataFrameValidator
from quantcore.config.timeframes import (
    Timeframe,
    TIMEFRAME_HIERARCHY,
    get_higher_timeframes,
)
from quantcore.features.trend import TrendFeatures
from quantcore.features.momentum import MomentumFeatures
from quantcore.features.volatility import VolatilityFeatures
from quantcore.features.volume import VolumeFeatures
from quantcore.features.market_structure import MarketStructureFeatures
from quantcore.features.rrg import RRGFeatures
from quantcore.features.waves import WaveFeatures
from quantcore.features.technical_indicators import TechnicalIndicators
from quantcore.features.trendlines import TrendlineFeatures
from quantcore.features.candlestick_patterns import CandlestickPatternFeatures
from quantcore.features.quantagents_trend import QuantAgentsTrendFeatures
from quantcore.features.quantagents_pattern import QuantAgentsPatternFeatures
from quantcore.features.gann import GannFeatures
from quantcore.features.sentiment_features import SentimentFeatures


class MultiTimeframeFeatureFactory:
    """
    Factory for computing features across all timeframes.

    Key responsibilities:
    1. Compute all feature groups for each timeframe
    2. Inject higher-TF context into lower-TF data
    3. Apply proper lagging to prevent lookahead
    4. Handle benchmark data for RRG features
    """

    def __init__(
        self,
        include_waves: bool = True,
        include_rrg: bool = True,
        include_technical_indicators: bool = True,
        enable_moving_averages: bool = True,
        enable_oscillators: bool = True,
        enable_volatility_indicators: bool = True,
        enable_volume_indicators: bool = True,
        enable_hilbert: bool = False,
        include_trendlines: bool = True,
        include_candlestick_patterns: bool = True,
        include_quant_trend: bool = True,
        include_quant_pattern: bool = True,
        include_gann_features: bool = True,
        include_mean_reversion: bool = True,
        include_sentiment_features: bool = True,
        trendline_lookback: int = 50,
        mr_lookback: int = 20,
        mr_zscore_threshold: float = 2.0,
    ):
        """
        Initialize the feature factory.

        Args:
            include_waves: Whether to compute wave features (default True)
            include_rrg: Whether to compute RRG features (default True).
                        Set to False for synthetic tests without benchmark data.
            include_technical_indicators: Whether to compute AlphaVantage indicators
            enable_moving_averages: Enable MA indicators (SMA, EMA, etc.)
            enable_oscillators: Enable oscillator indicators (RSI, MACD, etc.)
            enable_volatility_indicators: Enable volatility indicators (BB, ATR, etc.)
            enable_volume_indicators: Enable volume indicators (OBV, AD, etc.)
            enable_hilbert: Enable Hilbert Transform indicators (computationally expensive)
            include_trendlines: Whether to compute trendline features (default True)
            include_candlestick_patterns: Whether to compute candlestick patterns (default True)
            include_quant_trend: Whether to compute QuantAgents trend features (default True)
            include_quant_pattern: Whether to compute QuantAgents pattern features (default True)
            include_gann_features: Whether to compute Gann features (default True)
            include_mean_reversion: Whether to compute mean reversion features (default True)
            include_sentiment_features: Whether to compute news sentiment features (default True)
            trendline_lookback: Lookback period for trendline fitting (default 50)
            mr_lookback: Mean reversion lookback period (default 20)
            mr_zscore_threshold: Z-score threshold for mean reversion signals (default 2.0)
        """
        self._feature_computers: Dict[Timeframe, dict] = {}
        self.include_waves = include_waves
        self.include_rrg = include_rrg
        self.include_technical_indicators = include_technical_indicators
        self.include_trendlines = include_trendlines
        self.include_candlestick_patterns = include_candlestick_patterns
        self.include_quant_trend = include_quant_trend
        self.include_quant_pattern = include_quant_pattern
        self.include_gann_features = include_gann_features
        self.include_mean_reversion = include_mean_reversion
        self.include_sentiment_features = include_sentiment_features
        self.mr_lookback = mr_lookback
        self.mr_zscore_threshold = mr_zscore_threshold

        # Initialize feature computers for each timeframe
        for tf in Timeframe:
            self._feature_computers[tf] = {
                "trend": TrendFeatures(tf),
                "momentum": MomentumFeatures(tf),
                "volatility": VolatilityFeatures(tf),
                "volume": VolumeFeatures(tf),
                "market_structure": MarketStructureFeatures(tf),
            }

            # Technical indicators from AlphaVantage
            if include_technical_indicators:
                self._feature_computers[tf]["technical_indicators"] = (
                    TechnicalIndicators(
                        tf,
                        enable_moving_averages=enable_moving_averages,
                        enable_oscillators=enable_oscillators,
                        enable_volatility=enable_volatility_indicators,
                        enable_volume=enable_volume_indicators,
                        enable_hilbert=enable_hilbert,
                    )
                )

            # RRG features require benchmark data
            if include_rrg:
                self._feature_computers[tf]["rrg"] = RRGFeatures(tf)

            # Wave features primarily for 4H and Daily
            if include_waves and tf in [Timeframe.H4, Timeframe.D1]:
                self._feature_computers[tf]["waves"] = WaveFeatures(tf)

            # Trendline features (QuantAgent-inspired)
            if include_trendlines:
                self._feature_computers[tf]["trendlines"] = TrendlineFeatures(
                    tf, lookback_period=trendline_lookback
                )

            # Candlestick pattern recognition
            if include_candlestick_patterns:
                self._feature_computers[tf]["candlestick_patterns"] = (
                    CandlestickPatternFeatures(tf)
                )

            # QuantAgents trend features (multi-horizon trend analysis)
            if include_quant_trend:
                self._feature_computers[tf]["quant_trend"] = QuantAgentsTrendFeatures(
                    tf
                )

            # QuantAgents pattern features (price action patterns)
            if include_quant_pattern:
                self._feature_computers[tf]["quant_pattern"] = (
                    QuantAgentsPatternFeatures(tf, lookback_period=trendline_lookback)
                )

            # Gann features (swing/pivot points, retracements, price-time)
            if include_gann_features:
                self._feature_computers[tf]["gann"] = GannFeatures(tf)

            # Sentiment features (news sentiment)
            if include_sentiment_features:
                self._feature_computers[tf]["sentiment"] = SentimentFeatures(tf)

    def compute_all_timeframes(
        self,
        data: Dict[Timeframe, pd.DataFrame],
        benchmark_data: Optional[Dict[Timeframe, pd.DataFrame]] = None,
        news_sentiment_data: Optional[pd.DataFrame] = None,
        lag_features: bool = True,
    ) -> Dict[Timeframe, pd.DataFrame]:
        """
        Compute features for all timeframes with cross-TF context.

        Args:
            data: Dictionary mapping timeframe to OHLCV DataFrame
            benchmark_data: Optional benchmark data for RRG calculations
            news_sentiment_data: Optional news sentiment DataFrame
            lag_features: Whether to lag features by 1 bar (prevent lookahead)

        Returns:
            Dictionary mapping timeframe to DataFrame with features

        Raises:
            ValueError: If any input DataFrame fails validation
        """
        # Validate input data
        if not data:
            raise ValueError("FeatureFactory: data dictionary is empty")

        for tf, df in data.items():
            # Only strictly validate base timeframe (1H)
            # Resampled timeframes (4H, D1, W1) naturally have gaps from resampling
            is_base_tf = tf == Timeframe.H1
            validation_result = DataFrameValidator.validate_ohlcv(
                df, name=f"data[{tf.value}]", raise_on_error=is_base_tf
            )
            if is_base_tf:
                validation_result.log_warnings()
            elif not validation_result.is_valid:
                # For resampled TFs, just log warnings and forward-fill NaN values
                logger.warning(
                    f"Resampled {tf.value} has gaps, forward-filling NaN values"
                )
                df.ffill(inplace=True)
                df.bfill(inplace=True)  # Fill any remaining at start

        # Validate benchmark data if provided
        if benchmark_data:
            for tf, df in benchmark_data.items():
                is_base_tf = tf == Timeframe.H1
                validation_result = DataFrameValidator.validate_ohlcv(
                    df, name=f"benchmark_data[{tf.value}]", raise_on_error=is_base_tf
                )
                if not is_base_tf and not validation_result.is_valid:
                    df.ffill(inplace=True)
                    df.bfill(inplace=True)

        result: Dict[Timeframe, pd.DataFrame] = {}

        # First pass: compute features for each timeframe independently
        for tf in TIMEFRAME_HIERARCHY:
            if tf not in data or data[tf].empty:
                logger.warning(f"No data for {tf.value}, skipping")
                continue

            logger.info(f"Computing features for {tf.value}")

            benchmark_df = None
            if benchmark_data and tf in benchmark_data:
                benchmark_df = benchmark_data[tf]

            result[tf] = self._compute_features_for_timeframe(
                data[tf],
                tf,
                benchmark_df,
                news_sentiment_data,
            )

        # Second pass: inject higher-TF context into lower-TF data
        for tf in TIMEFRAME_HIERARCHY[1:]:  # Skip weekly (has no higher TF)
            if tf not in result:
                continue

            higher_tfs = get_higher_timeframes(tf)
            for higher_tf in higher_tfs:
                if higher_tf in result:
                    result[tf] = self.inject_higher_tf_context(
                        result[tf],
                        result[higher_tf],
                        higher_tf,
                    )

        # Third pass: lag features if requested
        if lag_features:
            for tf in result:
                feature_cols = self._get_all_feature_names(tf)
                result[tf] = self._lag_feature_columns(result[tf], feature_cols)

        return result

    def _compute_features_for_timeframe(
        self,
        df: pd.DataFrame,
        timeframe: Timeframe,
        benchmark_df: Optional[pd.DataFrame] = None,
        news_sentiment_data: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Compute all features for a single timeframe.

        Order matters:
        - Technical indicators first (comprehensive suite of base indicators)
        - Volatility before market structure (ATR used in swings)
        - Momentum after technical indicators (to avoid duplicates)
        - Wave features computed last (depend on ATR and market structure)
        """
        result = df.copy()

        computers = self._feature_computers[timeframe]

        # Technical indicators first (if enabled) - provides base indicators
        if "technical_indicators" in computers:
            result = computers["technical_indicators"].compute(result)
            logger.debug(f"Computed AlphaVantage indicators for {timeframe.value}")

        # Compute in dependency order
        result = computers["volatility"].compute(result)
        result = computers["trend"].compute(result)
        result = computers["momentum"].compute(
            result
        )  # Uses RSI/MACD from technical_indicators if available
        result = computers["volume"].compute(result)
        result = computers["market_structure"].compute(result)

        # RRG requires benchmark (if enabled)
        if "rrg" in computers:
            result = computers["rrg"].compute(result, benchmark_df)

        # Wave features (if enabled for this timeframe)
        if "waves" in computers:
            result = computers["waves"].compute(result)
            logger.debug(f"Computed wave features for {timeframe.value}")

        # Trendline features (QuantAgent-inspired)
        if "trendlines" in computers:
            result = computers["trendlines"].compute(result)
            logger.debug(f"Computed trendline features for {timeframe.value}")

        # Candlestick pattern recognition
        if "candlestick_patterns" in computers:
            result = computers["candlestick_patterns"].compute(result)
            logger.debug(f"Computed candlestick pattern features for {timeframe.value}")

        # QuantAgents trend features (requires price and volatility data)
        if "quant_trend" in computers:
            result = computers["quant_trend"].compute(result)
            logger.debug(f"Computed QuantAgents trend features for {timeframe.value}")

        # QuantAgents pattern features (can use swing data if available)
        if "quant_pattern" in computers:
            result = computers["quant_pattern"].compute(result)
            logger.debug(f"Computed QuantAgents pattern features for {timeframe.value}")

        # Gann features (swing/pivot, retracements, price-time)
        if "gann" in computers:
            result = computers["gann"].compute(result)
            logger.debug(f"Computed Gann features for {timeframe.value}")

        # Mean reversion features
        if self.include_mean_reversion:
            result = self._compute_mean_reversion_features(result)
            logger.debug(f"Computed mean reversion features for {timeframe.value}")

        # Sentiment features (news sentiment)
        if "sentiment" in computers:
            result = computers["sentiment"].compute(result, news_sentiment_data)
            logger.debug(f"Computed sentiment features for {timeframe.value}")

        logger.debug(f"Computed {len(result.columns)} features for {timeframe.value}")

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

    def inject_higher_tf_context(
        self,
        df_lower: pd.DataFrame,
        df_higher: pd.DataFrame,
        tf_higher: Timeframe,
    ) -> pd.DataFrame:
        """
        Inject higher timeframe context into lower timeframe data.

        This allows lower TF to see higher TF trends, regimes, etc.
        Uses forward-fill to propagate higher TF values.

        Args:
            df_lower: Lower timeframe DataFrame
            df_higher: Higher timeframe DataFrame with features
            tf_higher: Higher timeframe identifier

        Returns:
            Lower TF DataFrame with higher TF context columns
        """
        if df_higher.empty:
            return df_lower

        result = df_lower.copy()
        prefix = f"{tf_higher.value}_"

        # Key features to inject from higher timeframe
        context_features = [
            # Trend context
            "ema_alignment",
            "zscore_price",
            "regression_slope",
            "price_above_fast_ema",
            # Momentum context
            "rsi",
            "momentum_score",
            "macd_cross",
            # Volatility context
            "atr",
            "vol_regime",
            "bb_position",
            # Market structure
            "trend_structure",
            "probable_swing_low",
            "probable_swing_high",
            # RRG context (only for higher TFs)
            "rrg_quadrant",
            "rrg_long_favorable",
            "rs_ratio",
            "rs_momentum",
            # Wave context (from 4H/D1 to lower TFs)
            "wave_role",
            "wave_stage",
            "wave_conf",
            "prob_impulse_up",
            "prob_impulse_down",
            "prob_corr_down",
            "prob_corr_up",
            # Trendline context (QuantAgent-inspired)
            "tl_support_slope_close",
            "tl_resist_slope_close",
            "tl_dist_to_support",
            "tl_dist_to_resist",
            "tl_channel_width",
            "tl_price_position",
            "tl_support_angle",
            "tl_resist_angle",
            # Candlestick pattern context
            "cdl_net_signal",
            "cdl_bullish_count",
            "cdl_bearish_count",
            "cdl_double_bottom",
            # QuantAgents trend context
            "qa_trend_slope_short",
            "qa_trend_slope_med",
            "qa_trend_slope_long",
            "qa_trend_regime",
            "qa_trend_quality_med",
            "qa_trend_strength_med",
            "qa_trend_alignment_score",
            # QuantAgents pattern context
            "qa_pattern_is_pullback",
            "qa_pattern_is_breakout",
            "qa_pattern_consolidation",
            "qa_pattern_range_position",
            "qa_pattern_mr_opportunity",
            # Gann features context
            "gann_swing_high",
            "gann_swing_low",
            "gann_range_position",
            "gann_retracement_382",
            "gann_retracement_500",
            "gann_retracement_618",
            "gann_near_any_level",
            "gann_oversold",
            "gann_overbought",
            "gann_vs_1x1_low",
            "gann_vs_1x1_high",
            # Mean reversion context
            "mr_zscore",
            "mr_zscore_change",
            "mr_long_signal",
            "mr_short_signal",
            "mr_ma_distance",
            "mr_percentile",
            "mr_stretch",
            # Sentiment context
            "news_sentiment_mean",
            "news_sentiment_zscore",
            "news_sentiment_momentum",
            "news_bullish_ratio",
            "news_extreme_bullish",
            "news_extreme_bearish",
            "news_data_available",
        ]

        # Filter to features that exist in higher TF data
        available_features = [f for f in context_features if f in df_higher.columns]

        # Reindex higher TF to lower TF index with forward fill
        higher_reindexed = df_higher[available_features].reindex(
            result.index,
            method="ffill",
        )

        # Add with prefix
        for col in available_features:
            result[f"{prefix}{col}"] = higher_reindexed[col]

        logger.debug(
            f"Injected {len(available_features)} features from {tf_higher.value}"
        )

        return result

    def _lag_feature_columns(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        lag: int = 1,
    ) -> pd.DataFrame:
        """Lag feature columns to prevent lookahead bias."""
        result = df.copy()

        # Don't lag OHLCV columns or higher-TF context (already lagged implicitly)
        ohlcv_cols = {"open", "high", "low", "close", "volume"}

        for col in feature_cols:
            if col in result.columns and col not in ohlcv_cols:
                # Don't re-lag higher TF features
                if not any(col.startswith(f"{tf.value}_") for tf in Timeframe):
                    result[col] = result[col].shift(lag)

        return result

    def _get_all_feature_names(self, timeframe: Timeframe) -> List[str]:
        """Get all feature names for a timeframe (deduplicated)."""
        names = []
        seen = set()
        for computer in self._feature_computers[timeframe].values():
            for name in computer.get_feature_names():
                if name not in seen:
                    names.append(name)
                    seen.add(name)
        return names

    def get_feature_group_mapping(
        self,
        feature_names: List[str],
    ) -> Dict[str, str]:
        """
        Create mapping from feature names to their group tags.

        Groups are used for aggregating feature importance by category.

        Args:
            feature_names: List of feature names

        Returns:
            Dictionary mapping feature_name -> group_tag
        """
        mapping = {}

        for feature_name in feature_names:
            # Determine group based on prefix or known patterns
            if (
                feature_name.startswith("ema_")
                or feature_name.startswith("price_dist_")
                or feature_name.startswith("zscore_")
                or feature_name.startswith("regression_")
            ):
                mapping[feature_name] = "ta_trend"
            elif (
                feature_name.startswith("rsi")
                or feature_name.startswith("macd_")
                or feature_name.startswith("stoch_")
                or feature_name.startswith("momentum_")
                or feature_name.startswith("roc_")
            ):
                mapping[feature_name] = "ta_momentum"
            elif (
                feature_name.startswith("atr")
                or feature_name.startswith("bb_")
                or feature_name.startswith("vol_")
                or feature_name.startswith("realized_vol")
            ):
                mapping[feature_name] = "ta_volatility"
            elif (
                feature_name.startswith("volume_")
                or feature_name.startswith("obv")
                or feature_name.startswith("vwap")
                or feature_name.startswith("ad_")
            ):
                mapping[feature_name] = "ta_volume"
            elif feature_name.startswith("tl_"):
                mapping[feature_name] = "trendlines"
            elif feature_name.startswith("cdl_"):
                mapping[feature_name] = "candlestick"
            elif feature_name.startswith("qa_trend_"):
                mapping[feature_name] = "qa_trend"
            elif feature_name.startswith("qa_pattern_"):
                mapping[feature_name] = "qa_pattern"
            elif feature_name.startswith("rrg_") or feature_name.startswith("rs_"):
                mapping[feature_name] = "rrg"
            elif feature_name.startswith("wave_") or feature_name.startswith("prob_"):
                mapping[feature_name] = "waves"
            elif "swing" in feature_name or "trend_structure" in feature_name:
                mapping[feature_name] = "market_structure"
            elif feature_name.startswith("gann_"):
                mapping[feature_name] = "gann"
            elif feature_name.startswith("mr_"):
                mapping[feature_name] = "mean_reversion"
            elif feature_name.startswith("news_"):
                mapping[feature_name] = "sentiment"
            else:
                # Default to "other" for unclassified features
                mapping[feature_name] = "other"

        return mapping

    def get_feature_names_for_ml(
        self,
        timeframe: Timeframe,
        include_higher_tf: bool = True,
        include_wave_features: bool = True,
    ) -> List[str]:
        """
        Get feature names suitable for ML model input.

        Excludes raw price columns and non-numeric features.

        Args:
            timeframe: Timeframe to get features for
            include_higher_tf: Include injected higher-TF features
            include_wave_features: Include wave context features

        Returns:
            List of feature column names
        """
        # Categorical features to exclude
        categorical = ["rrg_quadrant", "wave_role"]

        # Base features
        features = self._get_all_feature_names(timeframe)

        # Remove categorical/string features
        features = [f for f in features if f not in categorical]

        # Add higher TF features if requested
        if include_higher_tf:
            for higher_tf in get_higher_timeframes(timeframe):
                prefix = f"{higher_tf.value}_"
                higher_features = self._get_all_feature_names(higher_tf)
                higher_features = [f for f in higher_features if f not in categorical]
                features.extend([f"{prefix}{f}" for f in higher_features])

        # Add wave one-hot encoded features for ML
        if include_wave_features and self.include_waves:
            wave_roles = [
                "impulse_up",
                "impulse_down",
                "impulse_up_terminal",
                "impulse_down_terminal",
                "corr_up",
                "corr_down",
                "abc_complete",
            ]
            # Add as binary features for the primary wave timeframe (H4)
            if timeframe == Timeframe.H1:
                for role in wave_roles:
                    features.append(f"H4_wave_role_{role}")

        return features

    def compute_single_bar(
        self,
        current_bar: pd.Series,
        historical_data: Dict[Timeframe, pd.DataFrame],
        benchmark_data: Optional[Dict[Timeframe, pd.DataFrame]] = None,
    ) -> Dict[Timeframe, pd.Series]:
        """
        Compute features for a single new bar (for live trading).

        This is optimized for real-time use where we only need
        features for the most recent bar.

        Args:
            current_bar: New OHLCV bar (with proper index)
            historical_data: Recent historical data per TF
            benchmark_data: Benchmark data per TF

        Returns:
            Dictionary of feature Series for each TF
        """
        # For each TF, append new bar and compute features
        result = {}

        for tf in TIMEFRAME_HIERARCHY:
            if tf not in historical_data:
                continue

            # Append new bar to history
            df = historical_data[tf].copy()
            if isinstance(current_bar.name, pd.Timestamp):
                if current_bar.name not in df.index:
                    df.loc[current_bar.name] = current_bar[
                        ["open", "high", "low", "close", "volume"]
                    ]

            # Compute features
            benchmark_df = benchmark_data.get(tf) if benchmark_data else None
            features_df = self._compute_features_for_timeframe(df, tf, benchmark_df)

            # Get last row
            result[tf] = features_df.iloc[-1]

        return result
