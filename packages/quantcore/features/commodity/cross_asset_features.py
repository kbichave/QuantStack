"""
Cross-asset features for commodity trading.

Computes correlations and relationships with USD, equities, rates, and other commodities.
"""

from typing import List, Optional, Dict
import pandas as pd
import numpy as np
from loguru import logger

from quantcore.features.base import FeatureBase
from quantcore.config.timeframes import Timeframe


class CrossAssetFeatures(FeatureBase):
    """
    Cross-asset features for commodity trading.

    Features:
    - USD correlation and divergence
    - Energy equity correlation (XLE, XOM)
    - Treasury rates correlation
    - VIX relationship
    - Other commodity correlations (Gold, Copper)
    """

    def __init__(
        self,
        timeframe: Timeframe,
        correlation_lookback: int = 20,
        divergence_lookback: int = 10,
    ):
        """
        Initialize cross-asset feature calculator.

        Args:
            timeframe: Timeframe for parameter adjustment
            correlation_lookback: Lookback for correlation calculations
            divergence_lookback: Lookback for divergence detection
        """
        super().__init__(timeframe)

        # Adjust lookback based on timeframe
        if timeframe == Timeframe.H1:
            self.corr_lookback = correlation_lookback
            self.div_lookback = divergence_lookback
        elif timeframe == Timeframe.H4:
            self.corr_lookback = max(10, correlation_lookback // 2)
            self.div_lookback = max(5, divergence_lookback // 2)
        elif timeframe == Timeframe.D1:
            self.corr_lookback = max(10, correlation_lookback)
            self.div_lookback = max(5, divergence_lookback)
        else:
            self.corr_lookback = max(8, correlation_lookback)
            self.div_lookback = max(4, divergence_lookback)

    def compute(
        self,
        df: pd.DataFrame,
        cross_asset_data: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> pd.DataFrame:
        """
        Compute cross-asset features.

        Args:
            df: Main OHLCV DataFrame (e.g., WTI)
            cross_asset_data: Dictionary of DataFrames for cross-asset symbols

        Returns:
            DataFrame with cross-asset features added
        """
        result = df.copy()

        if cross_asset_data is None or len(cross_asset_data) == 0:
            # Add empty cross-asset features
            return self._add_empty_cross_asset_features(result)

        # USD features
        if "USD" in cross_asset_data or "DXY" in cross_asset_data:
            usd_data = cross_asset_data.get("USD") or cross_asset_data.get("DXY")
            result = self._compute_usd_features(result, usd_data)

        # Energy equity features
        if "XLE" in cross_asset_data:
            result = self._compute_xle_features(result, cross_asset_data["XLE"])

        # Individual energy stocks
        for stock in ["XOM", "CVX", "OIH"]:
            if stock in cross_asset_data:
                result = self._compute_stock_features(
                    result, cross_asset_data[stock], stock.lower()
                )

        # Treasury rates features
        if "UST10Y" in cross_asset_data:
            result = self._compute_rates_features(result, cross_asset_data["UST10Y"])

        # VIX features
        if "VIX" in cross_asset_data:
            result = self._compute_vix_features(result, cross_asset_data["VIX"])

        # Other commodities
        if "GLD" in cross_asset_data:
            result = self._compute_commodity_features(
                result, cross_asset_data["GLD"], "gold"
            )
        if "COPX" in cross_asset_data:
            result = self._compute_commodity_features(
                result, cross_asset_data["COPX"], "copper"
            )

        # Aggregate cross-asset signal
        result = self._compute_aggregate_signal(result)

        return result

    def _compute_usd_features(
        self,
        df: pd.DataFrame,
        usd_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute USD-related features."""
        result = df.copy()

        # Align data
        common_idx = df.index.intersection(usd_data.index)
        if len(common_idx) == 0:
            return self._add_empty_usd_features(result)

        wti_returns = df.loc[common_idx, "close"].pct_change()
        usd_returns = usd_data.loc[common_idx, "close"].pct_change()

        # Rolling correlation
        result["wti_usd_corr"] = wti_returns.rolling(self.corr_lookback).corr(
            usd_returns
        )

        # USD momentum
        usd_mom = usd_data["close"].pct_change(self.div_lookback)
        result["usd_momentum"] = result.index.map(
            lambda x: usd_mom.loc[x] if x in usd_mom.index else np.nan
        )

        # WTI-USD divergence (WTI up while USD up = divergence from typical negative corr)
        wti_mom = df["close"].pct_change(self.div_lookback)
        result["wti_usd_divergence"] = np.where(
            (wti_mom > 0) & (usd_mom.reindex(df.index) > 0),
            1,  # Both up = unusual
            np.where(
                (wti_mom < 0) & (usd_mom.reindex(df.index) < 0),
                1,  # Both down = unusual
                0,
            ),
        )

        # USD z-score
        result["usd_zscore"] = result.index.map(
            lambda x: (
                self._single_zscore(usd_data["close"], x, self.corr_lookback)
                if x in usd_data.index
                else np.nan
            )
        )

        # USD regime (strong/weak/neutral)
        usd_sma = usd_data["close"].rolling(self.corr_lookback).mean()
        usd_strength = usd_data["close"] / usd_sma - 1
        result["usd_regime"] = result.index.map(
            lambda x: (
                "STRONG"
                if x in usd_strength.index and usd_strength.loc[x] > 0.02
                else (
                    "WEAK"
                    if x in usd_strength.index and usd_strength.loc[x] < -0.02
                    else "NEUTRAL"
                )
            )
        )

        return result

    def _compute_xle_features(
        self,
        df: pd.DataFrame,
        xle_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute energy sector ETF features."""
        result = df.copy()

        common_idx = df.index.intersection(xle_data.index)
        if len(common_idx) == 0:
            return self._add_empty_xle_features(result)

        wti_returns = df.loc[common_idx, "close"].pct_change()
        xle_returns = xle_data.loc[common_idx, "close"].pct_change()

        # Rolling correlation
        result["wti_xle_corr"] = wti_returns.rolling(self.corr_lookback).corr(
            xle_returns
        )

        # XLE relative strength
        xle_roc = xle_data["close"].pct_change(self.div_lookback)
        wti_roc = df["close"].pct_change(self.div_lookback)
        result["xle_relative_strength"] = xle_roc.reindex(df.index) - wti_roc

        # WTI-XLE divergence
        result["wti_xle_divergence"] = np.where(
            (wti_roc > 0.02) & (xle_roc.reindex(df.index) < -0.02),
            -1,  # WTI up, XLE down = bearish for WTI
            np.where(
                (wti_roc < -0.02) & (xle_roc.reindex(df.index) > 0.02),
                1,  # WTI down, XLE up = bullish for WTI
                0,
            ),
        )

        # XLE beta (sensitivity of WTI to XLE)
        result["wti_xle_beta"] = wti_returns.rolling(self.corr_lookback).cov(
            xle_returns
        ) / xle_returns.rolling(self.corr_lookback).var().replace(0, np.nan)

        return result

    def _compute_stock_features(
        self,
        df: pd.DataFrame,
        stock_data: pd.DataFrame,
        stock_name: str,
    ) -> pd.DataFrame:
        """Compute individual stock correlation features."""
        result = df.copy()

        common_idx = df.index.intersection(stock_data.index)
        if len(common_idx) == 0:
            result[f"wti_{stock_name}_corr"] = np.nan
            return result

        wti_returns = df.loc[common_idx, "close"].pct_change()
        stock_returns = stock_data.loc[common_idx, "close"].pct_change()

        # Rolling correlation
        result[f"wti_{stock_name}_corr"] = wti_returns.rolling(self.corr_lookback).corr(
            stock_returns
        )

        return result

    def _compute_rates_features(
        self,
        df: pd.DataFrame,
        rates_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute treasury rates features."""
        result = df.copy()

        common_idx = df.index.intersection(rates_data.index)
        if len(common_idx) == 0:
            return self._add_empty_rates_features(result)

        wti_returns = df.loc[common_idx, "close"].pct_change()
        rates_returns = rates_data.loc[common_idx, "close"].pct_change()

        # Rolling correlation
        result["wti_rates_corr"] = wti_returns.rolling(self.corr_lookback).corr(
            rates_returns
        )

        # Rates momentum
        rates_mom = rates_data["close"].pct_change(self.div_lookback)
        result["rates_momentum"] = result.index.map(
            lambda x: rates_mom.loc[x] if x in rates_mom.index else np.nan
        )

        # Rates regime (rising/falling)
        rates_trend = (
            rates_data["close"]
            .rolling(self.corr_lookback)
            .apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0)
        )
        result["rates_trend"] = result.index.map(
            lambda x: rates_trend.loc[x] if x in rates_trend.index else np.nan
        )
        result["rates_rising"] = (result["rates_trend"] > 0).astype(int)

        return result

    def _compute_vix_features(
        self,
        df: pd.DataFrame,
        vix_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute VIX-related features."""
        result = df.copy()

        common_idx = df.index.intersection(vix_data.index)
        if len(common_idx) == 0:
            return self._add_empty_vix_features(result)

        wti_returns = df.loc[common_idx, "close"].pct_change()
        vix_returns = vix_data.loc[common_idx, "close"].pct_change()

        # Rolling correlation
        result["wti_vix_corr"] = wti_returns.rolling(self.corr_lookback).corr(
            vix_returns
        )

        # VIX level
        result["vix_level"] = result.index.map(
            lambda x: vix_data.loc[x, "close"] if x in vix_data.index else np.nan
        )

        # VIX z-score
        result["vix_zscore"] = result.index.map(
            lambda x: (
                self._single_zscore(vix_data["close"], x, self.corr_lookback * 2)
                if x in vix_data.index
                else np.nan
            )
        )

        # VIX spike (sudden increase)
        vix_change = vix_data["close"].pct_change()
        result["vix_spike"] = result.index.map(
            lambda x: int(vix_change.loc[x] > 0.1) if x in vix_change.index else 0
        )

        # VIX regime
        result["vix_regime"] = np.where(
            result["vix_zscore"] > 1.5,
            "HIGH",
            np.where(result["vix_zscore"] < -1, "LOW", "NORMAL"),
        )

        # Risk-off signal (high VIX)
        result["risk_off"] = (result["vix_zscore"] > 1).astype(int)

        return result

    def _compute_commodity_features(
        self,
        df: pd.DataFrame,
        commodity_data: pd.DataFrame,
        commodity_name: str,
    ) -> pd.DataFrame:
        """Compute other commodity correlation features."""
        result = df.copy()

        common_idx = df.index.intersection(commodity_data.index)
        if len(common_idx) == 0:
            result[f"wti_{commodity_name}_corr"] = np.nan
            return result

        wti_returns = df.loc[common_idx, "close"].pct_change()
        commodity_returns = commodity_data.loc[common_idx, "close"].pct_change()

        # Rolling correlation
        result[f"wti_{commodity_name}_corr"] = wti_returns.rolling(
            self.corr_lookback
        ).corr(commodity_returns)

        # Relative performance
        commodity_roc = commodity_data["close"].pct_change(self.div_lookback)
        wti_roc = df["close"].pct_change(self.div_lookback)
        result[f"{commodity_name}_relative"] = commodity_roc.reindex(df.index) - wti_roc

        return result

    def _compute_aggregate_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute aggregate cross-asset signal."""
        result = df.copy()

        # Collect correlation columns
        corr_cols = [col for col in result.columns if col.endswith("_corr")]

        if len(corr_cols) == 0:
            result["cross_asset_signal"] = 0
            result["cross_asset_strength"] = 0
            return result

        # Average correlation magnitude
        corr_data = result[corr_cols].abs()
        result["avg_correlation"] = corr_data.mean(axis=1)

        # Cross-asset signal (based on divergences and regimes)
        signal = 0

        # USD impact
        if "wti_usd_divergence" in result.columns:
            signal += (
                result["wti_usd_divergence"] * -1
            )  # Divergence = mean reversion opportunity

        # XLE divergence
        if "wti_xle_divergence" in result.columns:
            signal += result["wti_xle_divergence"]

        # VIX impact
        if "risk_off" in result.columns:
            signal += result["risk_off"] * -1  # Risk-off = bearish for oil

        result["cross_asset_signal"] = signal
        result["cross_asset_strength"] = result["avg_correlation"]

        return result

    def _single_zscore(
        self,
        series: pd.Series,
        timestamp: pd.Timestamp,
        lookback: int,
    ) -> float:
        """Calculate z-score for a single timestamp."""
        if timestamp not in series.index:
            return np.nan

        idx_loc = series.index.get_loc(timestamp)
        if idx_loc < lookback:
            return np.nan

        window = series.iloc[max(0, idx_loc - lookback) : idx_loc + 1]
        mean = window.mean()
        std = window.std()

        if std == 0 or pd.isna(std):
            return 0

        return (series.loc[timestamp] - mean) / std

    def _add_empty_cross_asset_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add empty cross-asset feature columns."""
        result = df.copy()
        for col in self.get_feature_names():
            result[col] = np.nan
        return result

    def _add_empty_usd_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add empty USD feature columns."""
        result = df.copy()
        for col in [
            "wti_usd_corr",
            "usd_momentum",
            "wti_usd_divergence",
            "usd_zscore",
            "usd_regime",
        ]:
            result[col] = np.nan
        return result

    def _add_empty_xle_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add empty XLE feature columns."""
        result = df.copy()
        for col in [
            "wti_xle_corr",
            "xle_relative_strength",
            "wti_xle_divergence",
            "wti_xle_beta",
        ]:
            result[col] = np.nan
        return result

    def _add_empty_rates_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add empty rates feature columns."""
        result = df.copy()
        for col in ["wti_rates_corr", "rates_momentum", "rates_trend", "rates_rising"]:
            result[col] = np.nan
        return result

    def _add_empty_vix_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add empty VIX feature columns."""
        result = df.copy()
        for col in [
            "wti_vix_corr",
            "vix_level",
            "vix_zscore",
            "vix_spike",
            "vix_regime",
            "risk_off",
        ]:
            result[col] = np.nan
        return result

    def get_feature_names(self) -> List[str]:
        """Get list of feature names produced by this class."""
        return [
            # USD features
            "wti_usd_corr",
            "usd_momentum",
            "wti_usd_divergence",
            "usd_zscore",
            "usd_regime",
            # XLE features
            "wti_xle_corr",
            "xle_relative_strength",
            "wti_xle_divergence",
            "wti_xle_beta",
            # Individual stocks
            "wti_xom_corr",
            "wti_cvx_corr",
            "wti_oih_corr",
            # Rates features
            "wti_rates_corr",
            "rates_momentum",
            "rates_trend",
            "rates_rising",
            # VIX features
            "wti_vix_corr",
            "vix_level",
            "vix_zscore",
            "vix_spike",
            "vix_regime",
            "risk_off",
            # Other commodities
            "wti_gold_corr",
            "gold_relative",
            "wti_copper_corr",
            "copper_relative",
            # Aggregate
            "avg_correlation",
            "cross_asset_signal",
            "cross_asset_strength",
        ]
