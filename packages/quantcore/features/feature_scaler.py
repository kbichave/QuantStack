"""
Feature Scaling for Options Trading.

Handles price and feature scaling across different equity price ranges
to ensure ML/RL models work properly across all symbols.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Literal
import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class ScalerConfig:
    """Configuration for feature scaling."""

    method: Literal["zscore", "minmax", "robust", "rank"] = "zscore"
    clip_outliers: bool = True
    clip_std: float = 3.0
    rolling_window: Optional[int] = None  # If set, use rolling normalization


class FeatureScaler:
    """
    Scale features to handle different price ranges across equities.

    Key principles:
    - Price-derived features (like ATR, spreads) need normalization
    - Return-based features are already normalized
    - Use robust scaling for features with outliers
    """

    # Features that are inherently normalized (returns, ratios, z-scores)
    ALREADY_NORMALIZED = [
        "returns",
        "log_returns",
        "pct_change",
        "rsi",
        "willr",
        "stoch",
        "stochf",
        "bb_position",
        "zscore",
        "percentile",
        "rank",
        "iv_rank",
        "iv_percentile",
        "put_call_ratio",
        "spy_correlation",
        "qqq_correlation",
        "sector_correlation",
        "spy_beta",
        "relative_strength",
        "rs_zscore",
        "market_regime_bullish",
        "market_regime_bearish",
    ]

    # Features that need price normalization (divide by close)
    PRICE_DEPENDENT = [
        "atr",
        "true_range",
        "bb_width",
        "high",
        "low",
        "open",
        "close",
        "ma_",
        "ema_",
        "sma_",
        "upper_band",
        "lower_band",
        "middle_band",
    ]

    def __init__(
        self,
        config: Optional[ScalerConfig] = None,
    ):
        """
        Initialize feature scaler.

        Args:
            config: Scaler configuration
        """
        self.config = config or ScalerConfig()
        self._fit_params: Dict[str, Tuple[float, float]] = {}

    def fit(
        self,
        df: pd.DataFrame,
        feature_columns: Optional[List[str]] = None,
    ) -> "FeatureScaler":
        """
        Fit scaler to training data.

        Args:
            df: Training DataFrame
            feature_columns: Columns to fit (None = all numeric)

        Returns:
            Self for chaining
        """
        if feature_columns is None:
            feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()

        for col in feature_columns:
            if col not in df.columns:
                continue

            values = df[col].dropna()
            if len(values) == 0:
                continue

            if self.config.method == "zscore":
                self._fit_params[col] = (values.mean(), values.std())
            elif self.config.method == "minmax":
                self._fit_params[col] = (values.min(), values.max())
            elif self.config.method == "robust":
                self._fit_params[col] = (
                    values.median(),
                    values.quantile(0.75) - values.quantile(0.25),
                )
            elif self.config.method == "rank":
                self._fit_params[col] = (0, len(values))

        return self

    def transform(
        self,
        df: pd.DataFrame,
        feature_columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Transform features using fitted parameters.

        Args:
            df: DataFrame to transform
            feature_columns: Columns to transform

        Returns:
            Transformed DataFrame
        """
        result = df.copy()

        if feature_columns is None:
            feature_columns = [c for c in df.columns if c in self._fit_params]

        for col in feature_columns:
            if col not in self._fit_params or col not in result.columns:
                continue

            center, scale = self._fit_params[col]

            if scale == 0:
                result[col] = 0.0
                continue

            if self.config.method in ["zscore", "robust"]:
                result[col] = (result[col] - center) / scale
            elif self.config.method == "minmax":
                result[col] = (result[col] - center) / (scale - center)

            # Clip outliers
            if self.config.clip_outliers:
                result[col] = result[col].clip(
                    -self.config.clip_std, self.config.clip_std
                )

        return result

    def fit_transform(
        self,
        df: pd.DataFrame,
        feature_columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(df, feature_columns).transform(df, feature_columns)

    def normalize_price_features(
        self,
        df: pd.DataFrame,
        price_column: str = "close",
    ) -> pd.DataFrame:
        """
        Normalize price-dependent features by dividing by price.

        This makes ATR, BB width, etc. comparable across different priced stocks.

        Args:
            df: DataFrame with features
            price_column: Column containing price for normalization

        Returns:
            DataFrame with normalized features
        """
        result = df.copy()

        if price_column not in result.columns:
            logger.warning(
                f"Price column {price_column} not found, skipping normalization"
            )
            return result

        price = result[price_column].replace(0, np.nan)

        for col in result.columns:
            # Check if column is price-dependent
            is_price_dependent = any(
                pattern in col.lower() for pattern in self.PRICE_DEPENDENT
            )

            # Skip if already normalized
            is_normalized = any(
                pattern in col.lower() for pattern in self.ALREADY_NORMALIZED
            )

            if is_price_dependent and not is_normalized:
                # Convert to percentage of price
                result[f"{col}_pct"] = result[col] / price * 100

        return result


class MultiSymbolScaler:
    """
    Fit and transform features across multiple symbols.

    Maintains separate scaling parameters per feature, fitting on
    pooled data from all symbols for cross-sectional comparability.
    """

    def __init__(
        self,
        config: Optional[ScalerConfig] = None,
    ):
        """
        Initialize multi-symbol scaler.

        Args:
            config: Scaler configuration
        """
        self.config = config or ScalerConfig()
        self._fit_params: Dict[str, Tuple[float, float]] = {}
        self._per_symbol_params: Dict[str, Dict[str, Tuple[float, float]]] = {}

    def fit(
        self,
        symbol_data: Dict[str, pd.DataFrame],
        feature_columns: List[str],
        pooled: bool = True,
    ) -> "MultiSymbolScaler":
        """
        Fit scaler on multi-symbol data.

        Args:
            symbol_data: Dictionary of symbol -> DataFrame
            feature_columns: Features to scale
            pooled: If True, pool all symbols for fitting. If False, fit per symbol.

        Returns:
            Self for chaining
        """
        if pooled:
            # Pool all data for universal scaling
            all_data = []
            for symbol, df in symbol_data.items():
                if len(df) > 0:
                    all_data.append(df[feature_columns].dropna())

            if all_data:
                pooled_df = pd.concat(all_data, axis=0)

                for col in feature_columns:
                    if col not in pooled_df.columns:
                        continue

                    values = pooled_df[col].dropna()
                    if len(values) == 0:
                        continue

                    if self.config.method == "zscore":
                        self._fit_params[col] = (values.mean(), values.std())
                    elif self.config.method == "robust":
                        self._fit_params[col] = (
                            values.median(),
                            values.quantile(0.75) - values.quantile(0.25),
                        )
                    elif self.config.method == "minmax":
                        self._fit_params[col] = (values.min(), values.max())
        else:
            # Fit per symbol
            for symbol, df in symbol_data.items():
                self._per_symbol_params[symbol] = {}

                for col in feature_columns:
                    if col not in df.columns:
                        continue

                    values = df[col].dropna()
                    if len(values) == 0:
                        continue

                    if self.config.method == "zscore":
                        self._per_symbol_params[symbol][col] = (
                            values.mean(),
                            values.std(),
                        )
                    elif self.config.method == "robust":
                        self._per_symbol_params[symbol][col] = (
                            values.median(),
                            values.quantile(0.75) - values.quantile(0.25),
                        )

        return self

    def transform(
        self,
        symbol: str,
        df: pd.DataFrame,
        feature_columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Transform features for a symbol.

        Args:
            symbol: Symbol name
            df: DataFrame to transform
            feature_columns: Columns to transform

        Returns:
            Transformed DataFrame
        """
        result = df.copy()

        # Get parameters (pooled or per-symbol)
        params = self._per_symbol_params.get(symbol, self._fit_params)

        if feature_columns is None:
            feature_columns = list(params.keys())

        for col in feature_columns:
            if col not in params or col not in result.columns:
                continue

            center, scale = params[col]

            if scale == 0 or np.isnan(scale):
                result[col] = 0.0
                continue

            if self.config.method in ["zscore", "robust"]:
                result[col] = (result[col] - center) / scale
            elif self.config.method == "minmax":
                if scale != center:
                    result[col] = (result[col] - center) / (scale - center)

            # Clip outliers
            if self.config.clip_outliers:
                result[col] = result[col].clip(
                    -self.config.clip_std, self.config.clip_std
                )

        return result

    def transform_all(
        self,
        symbol_data: Dict[str, pd.DataFrame],
        feature_columns: Optional[List[str]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Transform all symbols.

        Args:
            symbol_data: Dictionary of symbol -> DataFrame
            feature_columns: Columns to transform

        Returns:
            Dictionary of symbol -> transformed DataFrame
        """
        return {
            symbol: self.transform(symbol, df, feature_columns)
            for symbol, df in symbol_data.items()
        }


def get_feature_groups() -> Dict[str, List[str]]:
    """
    Get feature groups for selective scaling.

    Returns:
        Dictionary mapping group name to feature patterns
    """
    return {
        "price_level": [
            "close",
            "open",
            "high",
            "low",
            "mid",
            "bid",
            "ask",
            "strike",
            "underlying_price",
        ],
        "volatility": [
            "atr",
            "true_range",
            "realized_vol",
            "iv",
            "bb_width",
            "range",
            "spread",
        ],
        "momentum": [
            "macd",
            "rsi",
            "adx",
            "cci",
            "mfi",
            "williams_r",
        ],
        "volume": [
            "volume",
            "turnover",
            "obv",
            "ad",
            "volume_zscore",
        ],
        "options": [
            "iv_rank",
            "iv_percentile",
            "skew",
            "put_call_ratio",
            "delta",
            "gamma",
            "theta",
            "vega",
        ],
        "cross_sectional": [
            "spy_correlation",
            "spy_beta",
            "relative_strength",
            "sector_relative_perf",
        ],
    }


def select_features_for_ml(
    df: pd.DataFrame,
    exclude_raw_prices: bool = True,
    exclude_identifiers: bool = True,
) -> List[str]:
    """
    Select features suitable for ML training.

    Args:
        df: DataFrame with features
        exclude_raw_prices: Exclude raw price columns
        exclude_identifiers: Exclude identifier columns

    Returns:
        List of feature column names
    """
    feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    exclude_patterns = []

    if exclude_raw_prices:
        exclude_patterns.extend(["close", "open", "high", "low", "volume"])

    if exclude_identifiers:
        exclude_patterns.extend(["index", "id", "timestamp", "date"])

    features = []
    for col in feature_cols:
        col_lower = col.lower()

        # Skip excluded patterns
        if any(pattern in col_lower for pattern in exclude_patterns):
            continue

        features.append(col)

    return features
