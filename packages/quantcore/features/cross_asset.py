"""
Cross-Asset Features for Options Trading.

Adds market-wide and sector-relative features to individual symbols:
- SPY/QQQ correlation and relative strength
- Sector ETF relative performance
- Market regime indicators
- Beta estimation
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from loguru import logger


class CrossAssetFeatures:
    """
    Compute cross-asset features for individual symbols.

    Features:
    - spy_correlation_20d: Rolling correlation with SPY
    - spy_relative_strength: Performance relative to SPY
    - spy_beta: Rolling beta to SPY
    - qqq_correlation_20d: Rolling correlation with QQQ
    - sector_relative_perf: Performance relative to sector ETF
    - market_regime_bullish: Whether SPY is above key MAs
    """

    DEFAULT_LOOKBACK = 20
    BETA_LOOKBACK = 60

    def __init__(
        self,
        lookback: int = 20,
        beta_lookback: int = 60,
    ):
        """
        Initialize cross-asset features.

        Args:
            lookback: Lookback period for correlations
            beta_lookback: Lookback period for beta calculation
        """
        self.lookback = lookback
        self.beta_lookback = beta_lookback

    def get_feature_names(self) -> List[str]:
        """Get list of feature names computed by this class."""
        return [
            "returns",
            f"spy_correlation_{self.lookback}d",
            "spy_relative_strength",
            "spy_rs_zscore",
            "spy_beta",
            "spy_relative_volume",
            f"qqq_correlation_{self.lookback}d",
            "qqq_relative_strength",
            "qqq_rs_zscore",
            "qqq_beta",
            "sector_relative_perf",
            f"sector_correlation_{self.lookback}d",
            "outperforming_sector",
            "market_regime_bullish",
            "market_regime_bearish",
            "market_above_200ma",
        ]

    def compute(
        self,
        symbol_data: pd.DataFrame,
        spy_data: Optional[pd.DataFrame] = None,
        qqq_data: Optional[pd.DataFrame] = None,
        sector_etf_data: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Compute cross-asset features.

        Args:
            symbol_data: OHLCV data for the target symbol
            spy_data: OHLCV data for SPY (market proxy)
            qqq_data: OHLCV data for QQQ (tech proxy)
            sector_etf_data: OHLCV data for sector ETF

        Returns:
            DataFrame with cross-asset features added
        """
        result = symbol_data.copy()

        # Compute symbol returns
        result["returns"] = result["close"].pct_change()

        # SPY features
        if spy_data is not None and not spy_data.empty:
            result = self._add_benchmark_features(result, spy_data, "spy")

        # QQQ features
        if qqq_data is not None and not qqq_data.empty:
            result = self._add_benchmark_features(result, qqq_data, "qqq")

        # Sector ETF features
        if sector_etf_data is not None and not sector_etf_data.empty:
            result = self._add_sector_features(result, sector_etf_data)

        # Market regime features (from SPY)
        if spy_data is not None and not spy_data.empty:
            result = self._add_market_regime(result, spy_data)

        return result

    def _add_benchmark_features(
        self,
        symbol_df: pd.DataFrame,
        benchmark_df: pd.DataFrame,
        prefix: str,
    ) -> pd.DataFrame:
        """Add features relative to a benchmark (SPY or QQQ)."""
        result = symbol_df.copy()

        # Align data
        benchmark = benchmark_df.copy()
        benchmark["bench_close"] = benchmark["close"]
        benchmark["bench_returns"] = benchmark["close"].pct_change()

        # Join on index
        aligned = result.join(
            benchmark[["bench_close", "bench_returns"]],
            how="left",
            rsuffix="_bench",
        )

        # Forward fill missing benchmark data
        aligned["bench_close"] = aligned["bench_close"].ffill()
        aligned["bench_returns"] = aligned["bench_returns"].ffill()

        # Rolling correlation
        result[f"{prefix}_correlation_{self.lookback}d"] = (
            aligned["returns"].rolling(self.lookback).corr(aligned["bench_returns"])
        )

        # Relative strength (symbol return - benchmark return over lookback)
        symbol_perf = result["close"].pct_change(self.lookback)
        bench_perf = aligned["bench_close"].pct_change(self.lookback)
        result[f"{prefix}_relative_strength"] = symbol_perf - bench_perf

        # Relative strength z-score
        rs = result[f"{prefix}_relative_strength"]
        result[f"{prefix}_rs_zscore"] = (rs - rs.rolling(60).mean()) / rs.rolling(
            60
        ).std()

        # Rolling beta
        result[f"{prefix}_beta"] = self._compute_rolling_beta(
            aligned["returns"],
            aligned["bench_returns"],
            window=self.beta_lookback,
        )

        # Relative volume (symbol vol / benchmark vol)
        if "volume" in result.columns and "volume" in benchmark_df.columns:
            aligned_vol = result.join(
                benchmark_df[["volume"]].rename(columns={"volume": "bench_volume"}),
                how="left",
            )
            aligned_vol["bench_volume"] = aligned_vol["bench_volume"].ffill()

            symbol_vol_avg = result["volume"].rolling(20).mean()
            bench_vol_avg = aligned_vol["bench_volume"].rolling(20).mean()

            result[f"{prefix}_relative_volume"] = (
                symbol_vol_avg / bench_vol_avg.replace(0, np.nan)
            )

        return result

    def _add_sector_features(
        self,
        symbol_df: pd.DataFrame,
        sector_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Add sector-relative features."""
        result = symbol_df.copy()

        # Align sector data
        sector = sector_df.copy()
        sector["sector_close"] = sector["close"]
        sector["sector_returns"] = sector["close"].pct_change()

        aligned = result.join(
            sector[["sector_close", "sector_returns"]],
            how="left",
        )
        aligned["sector_close"] = aligned["sector_close"].ffill()
        aligned["sector_returns"] = aligned["sector_returns"].ffill()

        # Sector relative performance
        symbol_perf = result["close"].pct_change(self.lookback)
        sector_perf = aligned["sector_close"].pct_change(self.lookback)
        result["sector_relative_perf"] = symbol_perf - sector_perf

        # Sector correlation
        result[f"sector_correlation_{self.lookback}d"] = (
            aligned["returns"].rolling(self.lookback).corr(aligned["sector_returns"])
        )

        # Outperforming sector flag
        result["outperforming_sector"] = (result["sector_relative_perf"] > 0).astype(
            int
        )

        return result

    def _add_market_regime(
        self,
        symbol_df: pd.DataFrame,
        spy_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Add market regime indicators based on SPY."""
        result = symbol_df.copy()

        # SPY moving averages
        spy = spy_df.copy()
        spy["ma_20"] = spy["close"].rolling(20).mean()
        spy["ma_50"] = spy["close"].rolling(50).mean()
        spy["ma_200"] = spy["close"].rolling(200).mean()

        # Join regime indicators
        spy["regime_bullish"] = (
            (spy["close"] > spy["ma_20"])
            & (spy["close"] > spy["ma_50"])
            & (spy["ma_20"] > spy["ma_50"])
        ).astype(int)

        spy["regime_bearish"] = (
            (spy["close"] < spy["ma_20"])
            & (spy["close"] < spy["ma_50"])
            & (spy["ma_20"] < spy["ma_50"])
        ).astype(int)

        spy["above_200ma"] = (spy["close"] > spy["ma_200"]).astype(int)

        result = result.join(
            spy[["regime_bullish", "regime_bearish", "above_200ma"]].add_prefix(
                "market_"
            ),
            how="left",
        )

        # Forward fill regime indicators
        for col in [
            "market_regime_bullish",
            "market_regime_bearish",
            "market_above_200ma",
        ]:
            if col in result.columns:
                result[col] = result[col].ffill().fillna(0)

        return result

    def _compute_rolling_beta(
        self,
        asset_returns: pd.Series,
        market_returns: pd.Series,
        window: int,
    ) -> pd.Series:
        """
        Compute rolling beta using covariance / variance.

        Beta = Cov(asset, market) / Var(market)
        """
        cov = asset_returns.rolling(window).cov(market_returns)
        var = market_returns.rolling(window).var()

        beta = cov / var.replace(0, np.nan)
        return beta.clip(-5, 5)  # Clip extreme values


class CrossAssetFeatureBuilder:
    """
    Builder for cross-asset features across multiple symbols.

    Usage:
        builder = CrossAssetFeatureBuilder()
        builder.set_market_data(spy_data, qqq_data)
        builder.set_sector_etfs({'TECHNOLOGY': xlk_data, ...})

        # Build features for each symbol
        for symbol in symbols:
            features = builder.build_features(symbol_data, symbol, sector)
    """

    def __init__(self):
        """Initialize feature builder."""
        self.spy_data: Optional[pd.DataFrame] = None
        self.qqq_data: Optional[pd.DataFrame] = None
        self.sector_etfs: Dict[str, pd.DataFrame] = {}
        self._feature_computer = CrossAssetFeatures()

    def set_market_data(
        self,
        spy_data: pd.DataFrame,
        qqq_data: Optional[pd.DataFrame] = None,
    ) -> "CrossAssetFeatureBuilder":
        """
        Set market benchmark data.

        Args:
            spy_data: SPY OHLCV data
            qqq_data: QQQ OHLCV data (optional)
        """
        self.spy_data = spy_data
        self.qqq_data = qqq_data
        return self

    def set_sector_etfs(
        self,
        sector_etfs: Dict[str, pd.DataFrame],
    ) -> "CrossAssetFeatureBuilder":
        """
        Set sector ETF data.

        Args:
            sector_etfs: Dict of sector name -> OHLCV data
        """
        self.sector_etfs = sector_etfs
        return self

    def build_features(
        self,
        symbol_data: pd.DataFrame,
        symbol: str,
        sector: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Build cross-asset features for a symbol.

        Args:
            symbol_data: Symbol OHLCV data
            symbol: Ticker symbol
            sector: Sector name for sector ETF lookup

        Returns:
            DataFrame with cross-asset features
        """
        sector_data = None
        if sector and sector in self.sector_etfs:
            sector_data = self.sector_etfs[sector]

        return self._feature_computer.compute(
            symbol_data=symbol_data,
            spy_data=self.spy_data,
            qqq_data=self.qqq_data,
            sector_etf_data=sector_data,
        )


# Sector ETF mapping
SECTOR_ETF_MAP = {
    "TECHNOLOGY": "XLK",
    "HEALTHCARE": "XLV",
    "FINANCIALS": "XLF",
    "CONSUMER_DISCRETIONARY": "XLY",
    "CONSUMER_STAPLES": "XLP",
    "INDUSTRIALS": "XLI",
    "ENERGY": "XLE",
    "MATERIALS": "XLB",
    "UTILITIES": "XLU",
    "REAL_ESTATE": "XLRE",
    "COMMUNICATION": "XLC",
}


def get_sector_etf(sector: str) -> Optional[str]:
    """Get sector ETF symbol for a sector."""
    return SECTOR_ETF_MAP.get(sector.upper())
