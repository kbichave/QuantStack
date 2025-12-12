"""
Liquidity analysis and spread estimation.

Critical for hourly MR strategies where spread widening destroys edge.
"""

from dataclasses import dataclass
from typing import Optional, Dict
import pandas as pd
import numpy as np
from loguru import logger


@dataclass
class LiquidityFeatures:
    """Liquidity metrics for a bar."""

    estimated_spread_bps: float
    volume_vs_avg: float
    volume_imbalance: float
    volatility_of_volatility: float
    liquidity_score: float  # 0-1, higher = better liquidity
    is_liquid: bool


class SpreadEstimator:
    """
    Estimate bid-ask spread from OHLCV data.

    Uses multiple estimation methods:
    1. High-Low spread estimator (Corwin-Schultz)
    2. Close-to-close volatility ratio
    3. Volume-weighted estimation
    """

    @staticmethod
    def corwin_schultz_spread(
        high: pd.Series,
        low: pd.Series,
        window: int = 20,
    ) -> pd.Series:
        """
        Corwin-Schultz high-low spread estimator.

        Based on the observation that daily high-low ratios
        capture both volatility and spread.
        """
        # Beta calculation
        log_hl = np.log(high / low) ** 2

        # Two-day high-low
        h2 = high.rolling(2).max()
        l2 = low.rolling(2).min()
        log_hl2 = np.log(h2 / l2) ** 2

        # Gamma
        gamma = log_hl2 - 2 * log_hl

        # Beta
        beta = log_hl.rolling(window).mean()

        # Alpha
        sqrt_2 = np.sqrt(2)
        sqrt_8_pi = np.sqrt(8 / np.pi)

        k = sqrt_2 - 1

        alpha_term = (np.sqrt(beta) * (sqrt_2 - 1)) / (3 - 2 * sqrt_2)
        alpha_term = alpha_term - np.sqrt(gamma / (3 - 2 * sqrt_2))

        # Spread estimation
        spread = 2 * (np.exp(alpha_term) - 1) / (1 + np.exp(alpha_term))
        spread = spread.clip(lower=0)  # Spread can't be negative

        return spread * 10000  # Convert to basis points

    @staticmethod
    def roll_spread(
        close: pd.Series,
        window: int = 20,
    ) -> pd.Series:
        """
        Roll (1984) implied spread estimator.

        Uses serial covariance of price changes.
        """
        returns = close.pct_change()

        # Covariance of consecutive returns
        cov = returns.rolling(window).apply(
            lambda x: np.cov(x[:-1], x[1:])[0, 1] if len(x) > 1 else 0, raw=True
        )

        # Spread = 2 * sqrt(-cov) if cov < 0
        spread = np.where(cov < 0, 2 * np.sqrt(-cov), 0)

        return pd.Series(spread, index=close.index) * 10000  # bps

    @staticmethod
    def volume_adjusted_spread(
        high: pd.Series,
        low: pd.Series,
        volume: pd.Series,
        window: int = 20,
    ) -> pd.Series:
        """
        Volume-adjusted spread estimation.

        Lower volume typically means wider spreads.
        """
        # Base spread from high-low
        base_spread = (high - low) / ((high + low) / 2) * 10000

        # Volume adjustment factor
        avg_volume = volume.rolling(window).mean()
        volume_ratio = volume / avg_volume

        # Widen spread when volume is low
        volume_adjustment = np.where(
            volume_ratio < 1, 1 + (1 - volume_ratio) * 0.5, 1  # Up to 50% wider
        )

        return base_spread * volume_adjustment


class LiquidityAnalyzer:
    """
    Comprehensive liquidity analysis for microstructure-aware trading.

    Features:
    - Spread estimation (multiple methods)
    - Volume profile analysis
    - Liquidity scoring
    - Volatility-of-volatility detection
    """

    def __init__(
        self,
        min_volume_ratio: float = 0.3,
        max_spread_bps: float = 20.0,
        vol_of_vol_threshold: float = 2.0,
    ):
        """
        Initialize liquidity analyzer.

        Args:
            min_volume_ratio: Minimum volume vs average to be liquid
            max_spread_bps: Maximum spread in bps to be liquid
            vol_of_vol_threshold: Vol-of-vol z-score for burst detection
        """
        self.min_volume_ratio = min_volume_ratio
        self.max_spread_bps = max_spread_bps
        self.vol_of_vol_threshold = vol_of_vol_threshold
        self.spread_estimator = SpreadEstimator()

    def compute_features(
        self,
        df: pd.DataFrame,
        window: int = 20,
    ) -> pd.DataFrame:
        """
        Compute all liquidity features.

        Args:
            df: OHLCV DataFrame
            window: Lookback window

        Returns:
            DataFrame with liquidity features added
        """
        result = df.copy()

        high = result["high"]
        low = result["low"]
        close = result["close"]
        volume = result["volume"]

        # Spread estimates (ensemble)
        cs_spread = self.spread_estimator.corwin_schultz_spread(high, low, window)
        roll_spread = self.spread_estimator.roll_spread(close, window)
        vol_spread = self.spread_estimator.volume_adjusted_spread(
            high, low, volume, window
        )

        # Ensemble spread (median of methods)
        result["estimated_spread_bps"] = pd.concat(
            [cs_spread, roll_spread, vol_spread], axis=1
        ).median(axis=1)

        # Volume vs average
        avg_volume = volume.rolling(window).mean()
        result["volume_vs_avg"] = volume / avg_volume

        # Volume imbalance (OBV-based intraday)
        price_change = close.diff()
        signed_volume = np.where(
            price_change > 0, volume, np.where(price_change < 0, -volume, 0)
        )
        result["volume_imbalance"] = (
            pd.Series(signed_volume, index=df.index).rolling(5).sum()
        )
        result["volume_imbalance_norm"] = result["volume_imbalance"] / avg_volume

        # Volatility of volatility
        returns = close.pct_change()
        rolling_vol = returns.rolling(window).std()
        vol_of_vol = rolling_vol.rolling(window).std()
        vol_of_vol_mean = vol_of_vol.rolling(window * 2).mean()
        vol_of_vol_std = vol_of_vol.rolling(window * 2).std()
        result["vol_of_vol_zscore"] = (vol_of_vol - vol_of_vol_mean) / vol_of_vol_std

        # Liquidity score (0-1)
        result["liquidity_score"] = self._compute_liquidity_score(result)

        # Is liquid flag
        result["is_liquid"] = (
            (result["volume_vs_avg"] >= self.min_volume_ratio)
            & (result["estimated_spread_bps"] <= self.max_spread_bps)
            & (result["vol_of_vol_zscore"].abs() <= self.vol_of_vol_threshold)
        ).astype(int)

        return result

    def _compute_liquidity_score(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute composite liquidity score (0-1).

        Higher = better liquidity.
        """
        scores = pd.DataFrame(index=df.index)

        # Volume score (higher is better)
        vol_ratio = df["volume_vs_avg"].clip(0, 3)  # Cap at 3x
        scores["volume"] = vol_ratio / 3

        # Spread score (lower is better)
        spread = df["estimated_spread_bps"].clip(0, 50)
        scores["spread"] = 1 - (spread / 50)

        # Vol-of-vol score (lower is better)
        vov_abs = df["vol_of_vol_zscore"].abs().clip(0, 4)
        scores["vov"] = 1 - (vov_abs / 4)

        # Weighted average
        weights = {"volume": 0.4, "spread": 0.4, "vov": 0.2}

        return sum(scores[k] * v for k, v in weights.items())

    def get_liquidity_at_bar(
        self,
        df: pd.DataFrame,
        idx: int = -1,
    ) -> LiquidityFeatures:
        """
        Get liquidity features for a specific bar.

        Args:
            df: DataFrame with liquidity features
            idx: Bar index

        Returns:
            LiquidityFeatures dataclass
        """
        row = df.iloc[idx]

        return LiquidityFeatures(
            estimated_spread_bps=float(row.get("estimated_spread_bps", 10)),
            volume_vs_avg=float(row.get("volume_vs_avg", 1)),
            volume_imbalance=float(row.get("volume_imbalance_norm", 0)),
            volatility_of_volatility=float(row.get("vol_of_vol_zscore", 0)),
            liquidity_score=float(row.get("liquidity_score", 0.5)),
            is_liquid=bool(row.get("is_liquid", True)),
        )


class OrderFlowProxy:
    """
    Proxy for order flow analysis from OHLCV data.

    Since we don't have L2 data, we estimate order flow
    from price/volume patterns.
    """

    @staticmethod
    def compute_buying_pressure(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
    ) -> pd.Series:
        """
        Estimate buying pressure from OHLCV.

        Uses the position of close within the high-low range.
        """
        hl_range = high - low
        hl_range = hl_range.replace(0, np.nan)

        # Close location value (CLV)
        clv = ((close - low) - (high - close)) / hl_range

        # Accumulation/Distribution
        ad = clv * volume

        return ad

    @staticmethod
    def compute_smart_money_flow(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
        window: int = 10,
    ) -> pd.Series:
        """
        Estimate smart money flow.

        Smart money tends to accumulate during low-volume periods
        and distribute during high-volume periods.
        """
        avg_volume = volume.rolling(window).mean()
        is_high_volume = volume > avg_volume * 1.5

        returns = close.pct_change()

        # High volume moves = less informed
        # Low volume moves = more informed
        smart_move = np.where(~is_high_volume, returns * volume, returns * volume * 0.5)

        return pd.Series(smart_move, index=close.index).rolling(window).sum()
