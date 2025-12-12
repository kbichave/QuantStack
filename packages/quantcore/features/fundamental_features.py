"""
Fundamental and Earnings Features for Options Trading.

Provides features derived from:
- Earnings calendar: days_to_earnings, days_since_earnings, earnings_surprise
- Company fundamentals: P/E ratio, beta, dividend yield, market cap
- Derived features: earnings_volatility_factor, pre_earnings_iv_expansion

Usage:
    earnings_features = EarningsFeatures(store)
    fundamentals_features = FundamentalFeatures(store)

    # Add to feature DataFrame
    df = earnings_features.compute(df, symbol="AAPL")
    df = fundamentals_features.compute(df, symbol="AAPL")
"""

from datetime import datetime, date, timedelta
from typing import Optional, Dict, Any, List
import pandas as pd
import numpy as np
from loguru import logger

from quantcore.data.options_storage import OptionsDataStore, get_options_store
from quantcore.features.base import FeatureBase
from quantcore.config.timeframes import Timeframe


class EarningsFeatures(FeatureBase):
    """
    Features derived from earnings calendar.

    Features:
    - days_to_earnings: Days until next earnings (0-365, NaN if unknown)
    - days_since_earnings: Days since last earnings (0-365, NaN if unknown)
    - earnings_window: Boolean flag if within N days of earnings
    - last_surprise: Last earnings surprise (EPS beat/miss)
    - last_surprise_pct: Last earnings surprise as percentage
    - earnings_volatility_factor: Expected IV expansion near earnings
    """

    def __init__(
        self,
        store: Optional[OptionsDataStore] = None,
        timeframe: Timeframe = Timeframe.D1,
        earnings_window_days: int = 5,
    ):
        """
        Initialize earnings features.

        Args:
            store: Options data store
            timeframe: Timeframe for features
            earnings_window_days: Days before/after earnings to flag
        """
        super().__init__(timeframe)
        self.store = store or get_options_store()
        self.earnings_window_days = earnings_window_days

    def compute(
        self,
        df: pd.DataFrame,
        symbol: Optional[str] = None,
        earnings_data: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Compute earnings features.

        Args:
            df: DataFrame with DatetimeIndex
            symbol: Stock symbol (required if earnings_data not provided)
            earnings_data: Pre-loaded earnings DataFrame (optional)

        Returns:
            DataFrame with earnings features added
        """
        result = df.copy()

        if result.empty:
            return result

        # Load earnings data if not provided
        if earnings_data is None and symbol:
            earnings_data = self.store.load_earnings(symbol=symbol)

        if earnings_data is None or earnings_data.empty:
            # No earnings data - fill with NaN
            result["earn_days_to"] = np.nan
            result["earn_days_since"] = np.nan
            result["earn_window"] = False
            result["earn_last_surprise"] = np.nan
            result["earn_last_surprise_pct"] = np.nan
            result["earn_vol_factor"] = 1.0
            return result

        # Ensure report_date is datetime
        if "report_date" in earnings_data.columns:
            earnings_data = earnings_data.copy()
            earnings_data["report_date"] = pd.to_datetime(earnings_data["report_date"])
        else:
            logger.warning("No report_date column in earnings data")
            return result

        # Sort earnings by date
        earnings_data = earnings_data.sort_values("report_date")
        earnings_dates = earnings_data["report_date"].values

        # Initialize feature columns
        result["earn_days_to"] = np.nan
        result["earn_days_since"] = np.nan
        result["earn_window"] = False
        result["earn_last_surprise"] = np.nan
        result["earn_last_surprise_pct"] = np.nan
        result["earn_vol_factor"] = 1.0

        # Compute features for each row
        for idx in result.index:
            current_date = pd.Timestamp(idx)

            # Days to next earnings
            future_earnings = earnings_dates[earnings_dates > current_date]
            if len(future_earnings) > 0:
                next_earnings = pd.Timestamp(future_earnings[0])
                days_to = (next_earnings - current_date).days
                result.loc[idx, "earn_days_to"] = days_to

                # Earnings window flag
                if days_to <= self.earnings_window_days:
                    result.loc[idx, "earn_window"] = True

            # Days since last earnings
            past_earnings = earnings_dates[earnings_dates <= current_date]
            if len(past_earnings) > 0:
                last_earnings = pd.Timestamp(past_earnings[-1])
                days_since = (current_date - last_earnings).days
                result.loc[idx, "earn_days_since"] = days_since

                # Last earnings surprise
                last_row = earnings_data[earnings_data["report_date"] == last_earnings]
                if not last_row.empty:
                    if "surprise" in last_row.columns:
                        result.loc[idx, "earn_last_surprise"] = last_row[
                            "surprise"
                        ].iloc[0]
                    if "surprise_pct" in last_row.columns:
                        result.loc[idx, "earn_last_surprise_pct"] = last_row[
                            "surprise_pct"
                        ].iloc[0]

            # Earnings volatility factor
            # IV typically expands ~2x in week before earnings
            days_to_val = result.loc[idx, "earn_days_to"]
            if pd.notna(days_to_val) and days_to_val <= 7:
                # Linear ramp from 1.0 at 7 days to 2.0 at 0 days
                vol_factor = 1.0 + (1.0 - days_to_val / 7.0)
                result.loc[idx, "earn_vol_factor"] = vol_factor

        return result

    def get_feature_names(self) -> List[str]:
        """Return list of feature names."""
        return [
            "earn_days_to",
            "earn_days_since",
            "earn_window",
            "earn_last_surprise",
            "earn_last_surprise_pct",
            "earn_vol_factor",
        ]


class FundamentalFeatures(FeatureBase):
    """
    Features derived from company fundamentals.

    Features:
    - fund_pe_ratio: Price-to-earnings ratio
    - fund_pe_zscore: P/E ratio z-score vs sector/market
    - fund_forward_pe: Forward P/E ratio
    - fund_peg_ratio: PEG ratio (P/E to growth)
    - fund_beta: Stock beta (market sensitivity)
    - fund_dividend_yield: Dividend yield
    - fund_market_cap_log: Log of market cap
    - fund_price_to_52w_high: Current price vs 52-week high
    - fund_price_to_52w_low: Current price vs 52-week low
    """

    def __init__(
        self,
        store: Optional[OptionsDataStore] = None,
        timeframe: Timeframe = Timeframe.D1,
    ):
        """
        Initialize fundamental features.

        Args:
            store: Options data store
            timeframe: Timeframe for features
        """
        super().__init__(timeframe)
        self.store = store or get_options_store()

        # Cache for fundamentals (don't need to reload every bar)
        self._fundamentals_cache: Dict[str, Dict[str, Any]] = {}

    def compute(
        self,
        df: pd.DataFrame,
        symbol: Optional[str] = None,
        fundamentals: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """
        Compute fundamental features.

        Args:
            df: DataFrame with price data (needs 'close' column)
            symbol: Stock symbol
            fundamentals: Pre-loaded fundamentals dict (optional)

        Returns:
            DataFrame with fundamental features added
        """
        result = df.copy()

        if result.empty:
            return result

        # Load fundamentals if not provided
        if fundamentals is None and symbol:
            # Check cache first
            if symbol in self._fundamentals_cache:
                fundamentals = self._fundamentals_cache[symbol]
            else:
                fundamentals = self.store.load_fundamentals(symbol)
                if fundamentals:
                    self._fundamentals_cache[symbol] = fundamentals

        # Initialize with NaN
        for feat in self.get_feature_names():
            result[feat] = np.nan

        if fundamentals is None:
            return result

        # Static fundamental features (same for all rows)
        pe_ratio = fundamentals.get("pe_ratio")
        forward_pe = fundamentals.get("forward_pe")
        peg_ratio = fundamentals.get("peg_ratio")
        beta = fundamentals.get("beta")
        dividend_yield = fundamentals.get("dividend_yield")
        market_cap = fundamentals.get("market_cap")
        high_52w = fundamentals.get("fifty_two_week_high")
        low_52w = fundamentals.get("fifty_two_week_low")

        # Fill static features
        if pe_ratio is not None:
            result["fund_pe_ratio"] = float(pe_ratio)

        if forward_pe is not None:
            result["fund_forward_pe"] = float(forward_pe)

        if peg_ratio is not None:
            result["fund_peg_ratio"] = float(peg_ratio)

        if beta is not None:
            result["fund_beta"] = float(beta)

        if dividend_yield is not None:
            result["fund_dividend_yield"] = float(dividend_yield)

        if market_cap is not None and market_cap > 0:
            result["fund_market_cap_log"] = np.log10(float(market_cap))

        # Dynamic features (depend on current price)
        if "close" in result.columns:
            if high_52w is not None and high_52w > 0:
                result["fund_price_to_52w_high"] = result["close"] / float(high_52w)

            if low_52w is not None and low_52w > 0:
                result["fund_price_to_52w_low"] = result["close"] / float(low_52w)

        # P/E z-score (placeholder - would need sector data for proper calculation)
        # For now, use a simple normalization
        if pe_ratio is not None and pe_ratio > 0:
            # Assume market average P/E ~20, std ~10
            result["fund_pe_zscore"] = (float(pe_ratio) - 20) / 10

        return result

    def get_feature_names(self) -> List[str]:
        """Return list of feature names."""
        return [
            "fund_pe_ratio",
            "fund_pe_zscore",
            "fund_forward_pe",
            "fund_peg_ratio",
            "fund_beta",
            "fund_dividend_yield",
            "fund_market_cap_log",
            "fund_price_to_52w_high",
            "fund_price_to_52w_low",
        ]

    def clear_cache(self) -> None:
        """Clear fundamentals cache."""
        self._fundamentals_cache.clear()


class CombinedFundamentalFeatures:
    """
    Combines earnings and fundamental features for convenience.

    Also adds derived features:
    - earnings_beta_interaction: Earnings window * beta (high beta = more volatile around earnings)
    - pre_earnings_iv_expansion: Expected IV expansion factor
    """

    def __init__(
        self,
        store: Optional[OptionsDataStore] = None,
        timeframe: Timeframe = Timeframe.D1,
        earnings_window_days: int = 5,
    ):
        """
        Initialize combined features.

        Args:
            store: Options data store
            timeframe: Timeframe for features
            earnings_window_days: Days before/after earnings to flag
        """
        self.store = store or get_options_store()
        self.earnings_features = EarningsFeatures(
            store, timeframe, earnings_window_days
        )
        self.fundamental_features = FundamentalFeatures(store, timeframe)

    def compute(
        self,
        df: pd.DataFrame,
        symbol: str,
        earnings_data: Optional[pd.DataFrame] = None,
        fundamentals: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """
        Compute all fundamental and earnings features.

        Args:
            df: DataFrame with price data
            symbol: Stock symbol
            earnings_data: Pre-loaded earnings data (optional)
            fundamentals: Pre-loaded fundamentals (optional)

        Returns:
            DataFrame with all features added
        """
        # Add earnings features
        result = self.earnings_features.compute(df, symbol, earnings_data)

        # Add fundamental features
        result = self.fundamental_features.compute(result, symbol, fundamentals)

        # Add derived interaction features
        result = self._add_derived_features(result)

        return result

    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived/interaction features."""
        result = df.copy()

        # Earnings-beta interaction
        # High beta stocks tend to move more around earnings
        if "earn_window" in result.columns and "fund_beta" in result.columns:
            beta = result["fund_beta"].fillna(1.0)
            window = result["earn_window"].astype(float)
            result["earn_beta_interaction"] = window * beta

        # Pre-earnings IV expansion factor
        # Combines days_to_earnings with expected vol expansion
        if "earn_days_to" in result.columns:
            days_to = result["earn_days_to"]

            # Sigmoid-like expansion curve
            # Max expansion at 0 days, tapering to 1.0 at 14+ days
            expansion = np.where(
                days_to.isna(), 1.0, 1.0 + np.maximum(0, 1.0 - days_to / 14) * 1.5
            )
            result["earn_pre_iv_expansion"] = expansion

        return result

    def get_feature_names(self) -> List[str]:
        """Return list of all feature names."""
        return (
            self.earnings_features.get_feature_names()
            + self.fundamental_features.get_feature_names()
            + ["earn_beta_interaction", "earn_pre_iv_expansion"]
        )


def fetch_and_store_fundamentals(
    symbol: str,
    fetcher,  # AlphaVantageClient
    store: Optional[OptionsDataStore] = None,
) -> Dict[str, Any]:
    """
    Fetch company fundamentals and store in options database.

    Args:
        symbol: Stock symbol
        fetcher: AlphaVantageClient instance
        store: Options data store

    Returns:
        Fundamentals dict
    """
    store = store or get_options_store()

    try:
        data = fetcher.fetch_company_overview(symbol)
        if data:
            store.save_fundamentals(data)
            logger.info(f"Fetched and stored fundamentals for {symbol}")
        return data
    except Exception as e:
        logger.error(f"Failed to fetch fundamentals for {symbol}: {e}")
        return {}


def fetch_and_store_earnings(
    symbol: str,
    fetcher,  # AlphaVantageClient
    store: Optional[OptionsDataStore] = None,
    horizon: str = "12month",
) -> int:
    """
    Fetch earnings calendar and store in options database.

    Args:
        symbol: Stock symbol
        fetcher: AlphaVantageClient instance
        store: Options data store
        horizon: Time horizon ("3month", "6month", "12month")

    Returns:
        Number of records saved
    """
    store = store or get_options_store()

    try:
        df = fetcher.fetch_earnings_calendar(symbol=symbol, horizon=horizon)
        if not df.empty:
            records = store.save_earnings(df)
            logger.info(f"Fetched and stored {records} earnings records for {symbol}")
            return records
        return 0
    except Exception as e:
        logger.error(f"Failed to fetch earnings for {symbol}: {e}")
        return 0


def update_fundamentals_batch(
    symbols: List[str],
    fetcher,  # AlphaVantageClient
    store: Optional[OptionsDataStore] = None,
) -> Dict[str, bool]:
    """
    Update fundamentals for multiple symbols.

    Args:
        symbols: List of stock symbols
        fetcher: AlphaVantageClient instance
        store: Options data store

    Returns:
        Dict of symbol -> success status
    """
    results = {}

    for symbol in symbols:
        try:
            data = fetch_and_store_fundamentals(symbol, fetcher, store)
            results[symbol] = bool(data)
        except Exception as e:
            logger.error(f"Failed to update fundamentals for {symbol}: {e}")
            results[symbol] = False

    return results
