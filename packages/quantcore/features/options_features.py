"""
Options-specific feature engineering.

Features:
- IV rank and percentile
- Term structure slope
- Put/call volume ratio
- Skew metrics
"""

from typing import List, Optional
import pandas as pd
import numpy as np
from loguru import logger

from quantcore.features.base import FeatureBase
from quantcore.config.timeframes import Timeframe


class OptionsFeatures(FeatureBase):
    """
    Options-specific features for trading signals.

    Simple approach for v1 MVP (no full IV surface):
    - IV rank: Current IV vs historical range
    - IV percentile: Percentile of current IV in history
    - Term structure slope: Front vs back month IV
    - Put/call volume ratio
    - ATM IV level
    """

    def __init__(
        self,
        timeframe: Timeframe = Timeframe.D1,
        iv_lookback: int = 252,  # 1 year
        vol_lookback: int = 20,
    ):
        """
        Initialize options features.

        Args:
            timeframe: Timeframe for features
            iv_lookback: Lookback period for IV rank calculation
            vol_lookback: Lookback for volume features
        """
        super().__init__(timeframe)
        self.iv_lookback = iv_lookback
        self.vol_lookback = vol_lookback

    def compute(
        self,
        df: pd.DataFrame,
        options_data: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Compute options features.

        Args:
            df: DataFrame with underlying OHLCV data
            options_data: Optional DataFrame with options chain data

        Returns:
            DataFrame with options features added
        """
        result = df.copy()

        # If we have options data, compute options-specific features
        if options_data is not None and not options_data.empty:
            result = self._compute_from_options_chain(result, options_data)

        # Compute realized volatility features (always available)
        result = self._compute_realized_vol_features(result)

        return result

    def _compute_from_options_chain(
        self,
        df: pd.DataFrame,
        options_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute features from options chain data."""
        result = df.copy()

        # Get ATM IV
        if "iv" in options_data.columns and "strike" in options_data.columns:
            atm_iv = self._get_atm_iv(options_data, result["close"].iloc[-1])
            result["opt_atm_iv"] = atm_iv

        # Put/Call volume ratio
        if "volume" in options_data.columns and "option_type" in options_data.columns:
            pc_ratio = self._compute_put_call_ratio(options_data)
            result["opt_put_call_vol_ratio"] = pc_ratio

        # Put/Call OI ratio
        if "open_interest" in options_data.columns:
            pc_oi_ratio = self._compute_put_call_oi_ratio(options_data)
            result["opt_put_call_oi_ratio"] = pc_oi_ratio

        # Term structure slope
        if "expiry" in options_data.columns and "iv" in options_data.columns:
            term_slope = self._compute_term_structure_slope(options_data)
            result["opt_term_structure_slope"] = term_slope

        return result

    def _compute_realized_vol_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute realized volatility-based features."""
        result = df.copy()
        close = result["close"]

        # Log returns
        log_returns = np.log(close / close.shift(1))

        # Realized volatility (annualized)
        realized_vol = log_returns.rolling(20).std() * np.sqrt(252)
        result["opt_realized_vol_20d"] = realized_vol

        # Historical realized volatility for comparison
        realized_vol_60d = log_returns.rolling(60).std() * np.sqrt(252)
        result["opt_realized_vol_60d"] = realized_vol_60d

        # Vol ratio (short-term vs long-term)
        result["opt_vol_ratio"] = realized_vol / (realized_vol_60d + 1e-8)

        return result

    def _get_atm_iv(
        self,
        options_data: pd.DataFrame,
        underlying_price: float,
    ) -> float:
        """Get ATM implied volatility."""
        if options_data.empty:
            return np.nan

        df = options_data.copy()
        df["strike_diff"] = abs(df["strike"] - underlying_price)

        # Get ATM options (closest to underlying price)
        atm_options = df.nsmallest(2, "strike_diff")

        if "iv" in atm_options.columns:
            return atm_options["iv"].mean()

        return np.nan

    def _compute_put_call_ratio(self, options_data: pd.DataFrame) -> float:
        """Compute put/call volume ratio."""
        calls = options_data[options_data["option_type"].str.upper() == "CALL"]
        puts = options_data[options_data["option_type"].str.upper() == "PUT"]

        call_volume = calls["volume"].sum()
        put_volume = puts["volume"].sum()

        if call_volume > 0:
            return put_volume / call_volume
        return np.nan

    def _compute_put_call_oi_ratio(self, options_data: pd.DataFrame) -> float:
        """Compute put/call open interest ratio."""
        calls = options_data[options_data["option_type"].str.upper() == "CALL"]
        puts = options_data[options_data["option_type"].str.upper() == "PUT"]

        call_oi = calls["open_interest"].sum()
        put_oi = puts["open_interest"].sum()

        if call_oi > 0:
            return put_oi / call_oi
        return np.nan

    def _compute_term_structure_slope(self, options_data: pd.DataFrame) -> float:
        """
        Compute term structure slope.

        Positive slope = contango (normal)
        Negative slope = backwardation (fear)
        """
        if options_data.empty or "expiry" not in options_data.columns:
            return np.nan

        # Get unique expiries sorted
        expiries = pd.to_datetime(options_data["expiry"]).unique()
        expiries = sorted(expiries)

        if len(expiries) < 2:
            return np.nan

        # Get front and back month ATM IVs
        front_expiry = expiries[0]
        back_expiry = expiries[-1] if len(expiries) > 2 else expiries[1]

        front_options = options_data[
            pd.to_datetime(options_data["expiry"]) == front_expiry
        ]
        back_options = options_data[
            pd.to_datetime(options_data["expiry"]) == back_expiry
        ]

        front_iv = front_options["iv"].mean() if not front_options.empty else np.nan
        back_iv = back_options["iv"].mean() if not back_options.empty else np.nan

        if pd.notna(front_iv) and pd.notna(back_iv) and front_iv > 0:
            return (back_iv - front_iv) / front_iv

        return np.nan

    def get_feature_names(self) -> List[str]:
        """Return list of feature names."""
        return [
            "opt_atm_iv",
            "opt_put_call_vol_ratio",
            "opt_put_call_oi_ratio",
            "opt_term_structure_slope",
            "opt_realized_vol_20d",
            "opt_realized_vol_60d",
            "opt_vol_ratio",
        ]


def compute_iv_rank(
    current_iv: float,
    iv_history: pd.Series,
    lookback: int = 252,
) -> float:
    """
    Compute IV rank.

    IV Rank = (Current IV - 52w Low) / (52w High - 52w Low)

    Args:
        current_iv: Current implied volatility
        iv_history: Historical IV series
        lookback: Lookback period (default 252 = 1 year)

    Returns:
        IV rank (0-100)
    """
    if iv_history.empty or pd.isna(current_iv):
        return np.nan

    recent = iv_history.tail(lookback)
    iv_high = recent.max()
    iv_low = recent.min()

    if iv_high == iv_low:
        return 50.0  # No range, return middle

    rank = (current_iv - iv_low) / (iv_high - iv_low) * 100
    return rank


def compute_iv_percentile(
    current_iv: float,
    iv_history: pd.Series,
    lookback: int = 252,
) -> float:
    """
    Compute IV percentile.

    Percentage of days in lookback where IV was lower than current.

    Args:
        current_iv: Current implied volatility
        iv_history: Historical IV series
        lookback: Lookback period

    Returns:
        IV percentile (0-100)
    """
    if iv_history.empty or pd.isna(current_iv):
        return np.nan

    recent = iv_history.tail(lookback)
    percentile = (recent < current_iv).mean() * 100
    return percentile


def compute_skew(
    options_chain: pd.DataFrame,
    underlying_price: float,
    option_type: str = "PUT",
) -> float:
    """
    Compute volatility skew.

    Skew = IV(25-delta OTM) - IV(ATM)

    Args:
        options_chain: Options chain DataFrame
        underlying_price: Current underlying price
        option_type: Option type to measure skew

    Returns:
        Skew value
    """
    if options_chain.empty:
        return np.nan

    options = options_chain[
        options_chain["option_type"].str.upper() == option_type.upper()
    ]

    if options.empty or "delta" not in options.columns or "iv" not in options.columns:
        return np.nan

    # Find ATM option (delta ~= 0.5 for calls, -0.5 for puts)
    target_atm_delta = 0.5 if option_type.upper() == "CALL" else -0.5
    options = options.copy()
    options["atm_diff"] = abs(options["delta"] - target_atm_delta)
    atm_option = options.loc[options["atm_diff"].idxmin()]
    atm_iv = atm_option["iv"]

    # Find 25-delta OTM option
    target_otm_delta = 0.25 if option_type.upper() == "CALL" else -0.25
    options["otm_diff"] = abs(options["delta"] - target_otm_delta)
    otm_option = options.loc[options["otm_diff"].idxmin()]
    otm_iv = otm_option["iv"]

    return otm_iv - atm_iv


def classify_vol_regime(iv_rank: float) -> str:
    """
    Classify volatility regime from IV rank.

    Args:
        iv_rank: IV rank (0-100)

    Returns:
        "LOW", "MEDIUM", or "HIGH"
    """
    if pd.isna(iv_rank):
        return "MEDIUM"

    if iv_rank < 30:
        return "LOW"
    elif iv_rank > 70:
        return "HIGH"
    else:
        return "MEDIUM"
