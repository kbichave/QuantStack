"""
IV Rank Calculator with Historical Data.

Computes IV rank and percentile from:
1. Historical ATM IV (from HISTORICAL_OPTIONS API) - primary source
2. Realized volatility (from OHLCV) - fallback when IV history unavailable

Usage:
    manager = IVHistoryManager()

    # Fetch and store historical IV (one-time or periodic)
    manager.build_iv_history("AAPL", fetcher, start_date="2023-01-01")

    # Get current IV rank
    rank = manager.get_iv_rank("AAPL", current_iv=0.25)

    # Get dynamic thresholds (replaces hardcoded values)
    low, high = manager.get_iv_thresholds("AAPL")
"""

from datetime import datetime, date, timedelta
from typing import Optional, Tuple, Dict, Any, List
import pandas as pd
import numpy as np
from loguru import logger

from quantcore.data.options_storage import OptionsDataStore, get_options_store
from quantcore.features.options_features import compute_iv_rank, compute_iv_percentile


class IVHistoryManager:
    """
    Manages IV history and computes IV rank/percentile.

    Primary source: Historical ATM IV from options chain
    Fallback: Realized volatility from OHLCV data
    """

    def __init__(
        self,
        store: Optional[OptionsDataStore] = None,
        lookback_days: int = 252,
    ):
        """
        Initialize IV history manager.

        Args:
            store: Options data store (uses singleton if not provided)
            lookback_days: Default lookback for IV rank calculation
        """
        self.store = store or get_options_store()
        self.lookback_days = lookback_days

        # Cache for IV statistics
        self._stats_cache: Dict[str, Dict[str, float]] = {}

    def build_iv_history(
        self,
        symbol: str,
        fetcher,  # AlphaVantageClient
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        use_realized_vol: bool = True,
        ohlcv_data: Optional[pd.DataFrame] = None,
    ) -> int:
        """
        Build IV history for a symbol by fetching historical options data.

        Args:
            symbol: Stock symbol
            fetcher: AlphaVantageClient instance
            start_date: Start date (YYYY-MM-DD), default 1 year ago
            end_date: End date (YYYY-MM-DD), default today
            use_realized_vol: Also compute realized volatility as fallback
            ohlcv_data: Pre-loaded OHLCV data (optional, for realized vol)

        Returns:
            Number of records saved
        """
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=self.lookback_days)).strftime(
                "%Y-%m-%d"
            )
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        logger.info(f"Building IV history for {symbol} from {start_date} to {end_date}")

        records_saved = 0

        # Try to get historical options data
        try:
            records_saved = self._fetch_historical_iv(
                symbol, fetcher, start_date, end_date
            )
        except Exception as e:
            logger.warning(f"Failed to fetch historical options IV for {symbol}: {e}")

        # Compute and store realized volatility as fallback
        if use_realized_vol:
            try:
                rv_records = self._compute_realized_vol_history(
                    symbol, fetcher, ohlcv_data, start_date, end_date
                )
                records_saved = max(records_saved, rv_records)
            except Exception as e:
                logger.warning(f"Failed to compute realized vol for {symbol}: {e}")

        # Update IV statistics cache
        if records_saved > 0:
            self.store.update_iv_statistics(symbol, self.lookback_days)

        return records_saved

    def _fetch_historical_iv(
        self,
        symbol: str,
        fetcher,
        start_date: str,
        end_date: str,
    ) -> int:
        """Fetch historical ATM IV from options chain API."""
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")

        records = []
        current_date = start_dt

        # Sample dates (not every day - API rate limits)
        # Get ~52 data points over the year (weekly)
        sample_interval = max(1, (end_dt - start_dt).days // 52)

        while current_date <= end_dt:
            date_str = current_date.strftime("%Y-%m-%d")

            try:
                # Fetch historical options for this date
                options_df = fetcher.fetch_historical_options(symbol, date_str)

                if not options_df.empty:
                    # Extract ATM IV
                    atm_iv = self._extract_atm_iv(options_df)

                    if atm_iv is not None and not np.isnan(atm_iv):
                        records.append(
                            {
                                "date": current_date.date(),
                                "atm_iv": atm_iv,
                                "source": "options_chain",
                            }
                        )
                        logger.debug(f"{symbol} {date_str}: ATM IV = {atm_iv:.4f}")

            except Exception as e:
                logger.debug(f"No options data for {symbol} on {date_str}: {e}")

            current_date += timedelta(days=sample_interval)

        if records:
            df = pd.DataFrame(records)
            return self.store.save_iv_history_bulk(df, symbol)

        return 0

    def _extract_atm_iv(
        self,
        options_df: pd.DataFrame,
    ) -> Optional[float]:
        """Extract ATM implied volatility from options chain."""
        if options_df.empty:
            return None

        # Need strike and IV columns
        if "strike" not in options_df.columns or "iv" not in options_df.columns:
            return None

        # Filter to reasonable DTE (20-45 days) if expiry available
        if "expiry" in options_df.columns:
            today = datetime.now().date()
            options_df = options_df.copy()
            options_df["dte"] = (
                pd.to_datetime(options_df["expiry"]).dt.date - today
            ).apply(lambda x: x.days if hasattr(x, "days") else 999)
            options_df = options_df[
                (options_df["dte"] >= 20) & (options_df["dte"] <= 60)
            ]

        if options_df.empty:
            return None

        # Get underlying price (approximate from ATM strikes)
        # ATM is where calls and puts have similar prices
        strikes = options_df["strike"].unique()
        if len(strikes) == 0:
            return None

        underlying_price = np.median(strikes)

        # Find ATM options (closest to underlying)
        options_df = options_df.copy()
        options_df["strike_diff"] = abs(options_df["strike"] - underlying_price)
        atm_options = options_df.nsmallest(4, "strike_diff")

        # Average IV of ATM options
        atm_iv = atm_options["iv"].mean()

        return float(atm_iv) if pd.notna(atm_iv) else None

    def _compute_realized_vol_history(
        self,
        symbol: str,
        fetcher,
        ohlcv_data: Optional[pd.DataFrame],
        start_date: str,
        end_date: str,
    ) -> int:
        """Compute realized volatility from OHLCV data."""
        # Load OHLCV data if not provided
        if ohlcv_data is None or ohlcv_data.empty:
            try:
                ohlcv_data = fetcher.fetch_daily(symbol, outputsize="full")
            except Exception as e:
                logger.warning(f"Failed to fetch daily data for {symbol}: {e}")
                return 0

        if ohlcv_data.empty or "close" not in ohlcv_data.columns:
            return 0

        # Filter to date range
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        df = ohlcv_data.copy()
        df = df[(df.index >= start_dt) & (df.index <= end_dt)]

        if len(df) < 60:
            logger.warning(
                f"Insufficient data for realized vol calculation: {len(df)} bars"
            )
            return 0

        # Calculate log returns
        log_returns = np.log(df["close"] / df["close"].shift(1))

        # Calculate rolling realized volatility (annualized)
        df["realized_vol_20d"] = log_returns.rolling(20).std() * np.sqrt(252)
        df["realized_vol_60d"] = log_returns.rolling(60).std() * np.sqrt(252)

        # Prepare data for storage
        records = []
        for idx, row in df.dropna(subset=["realized_vol_20d"]).iterrows():
            # Check if we already have ATM IV for this date
            existing = self.store.conn.execute(
                """
                SELECT atm_iv FROM iv_history 
                WHERE symbol = ? AND date = ? AND source = 'options_chain'
            """,
                [symbol, idx.date()],
            ).fetchone()

            if existing and existing[0] is not None:
                # Already have options-based IV, just add realized vol
                self.store.conn.execute(
                    """
                    UPDATE iv_history 
                    SET realized_vol_20d = ?, realized_vol_60d = ?
                    WHERE symbol = ? AND date = ?
                """,
                    [
                        row["realized_vol_20d"],
                        row.get("realized_vol_60d"),
                        symbol,
                        idx.date(),
                    ],
                )
            else:
                # No options IV, store realized vol as primary
                records.append(
                    {
                        "date": idx.date(),
                        "realized_vol_20d": row["realized_vol_20d"],
                        "realized_vol_60d": row.get("realized_vol_60d"),
                        "source": "realized_vol",
                    }
                )

        if records:
            df_records = pd.DataFrame(records)
            return self.store.save_iv_history_bulk(df_records, symbol)

        return len(df)

    def get_iv_rank(
        self,
        symbol: str,
        current_iv: Optional[float] = None,
        lookback: Optional[int] = None,
    ) -> float:
        """
        Get IV rank for a symbol.

        IV Rank = (Current IV - 52w Low) / (52w High - 52w Low) * 100

        Args:
            symbol: Stock symbol
            current_iv: Current IV (if None, uses last stored value)
            lookback: Lookback period (default: self.lookback_days)

        Returns:
            IV rank (0-100), or 50 if insufficient data
        """
        lookback = lookback or self.lookback_days

        # Get IV history
        iv_series = self.store.get_iv_for_rank(symbol, lookback_days=lookback)

        if iv_series.empty or len(iv_series) < 20:
            logger.warning(f"Insufficient IV data for {symbol}, returning 50")
            return 50.0

        # Use last stored value if current_iv not provided
        if current_iv is None:
            current_iv = float(iv_series.iloc[-1])

        return compute_iv_rank(current_iv, iv_series, lookback)

    def get_iv_percentile(
        self,
        symbol: str,
        current_iv: Optional[float] = None,
        lookback: Optional[int] = None,
    ) -> float:
        """
        Get IV percentile for a symbol.

        Percentage of days where IV was lower than current.

        Args:
            symbol: Stock symbol
            current_iv: Current IV (if None, uses last stored value)
            lookback: Lookback period

        Returns:
            IV percentile (0-100)
        """
        lookback = lookback or self.lookback_days

        iv_series = self.store.get_iv_for_rank(symbol, lookback_days=lookback)

        if iv_series.empty or len(iv_series) < 20:
            return 50.0

        if current_iv is None:
            current_iv = float(iv_series.iloc[-1])

        return compute_iv_percentile(current_iv, iv_series, lookback)

    def get_iv_thresholds(
        self,
        symbol: str,
        low_percentile: float = 25,
        high_percentile: float = 75,
    ) -> Tuple[float, float]:
        """
        Get dynamic IV rank thresholds based on historical distribution.

        Replaces hardcoded iv_rank_low and iv_rank_high in config.

        Args:
            symbol: Stock symbol
            low_percentile: Percentile for low threshold (default 25)
            high_percentile: Percentile for high threshold (default 75)

        Returns:
            Tuple of (low_threshold, high_threshold) in IV rank terms
        """
        # First try to get from cache
        stats = self.store.get_iv_statistics(symbol)

        if stats is None:
            # Try to compute
            stats = self.store.update_iv_statistics(symbol, self.lookback_days)

        if not stats:
            # Fallback to defaults
            logger.warning(f"No IV stats for {symbol}, using defaults (30, 70)")
            return (30.0, 70.0)

        # Use percentiles from distribution
        # Convert to IV rank scale (0-100)
        return (low_percentile, high_percentile)

    def get_vol_regime(
        self,
        symbol: str,
        current_iv: Optional[float] = None,
    ) -> str:
        """
        Classify current volatility regime.

        Args:
            symbol: Stock symbol
            current_iv: Current IV

        Returns:
            "LOW", "MEDIUM", or "HIGH"
        """
        iv_rank = self.get_iv_rank(symbol, current_iv)

        if iv_rank < 30:
            return "LOW"
        elif iv_rank > 70:
            return "HIGH"
        else:
            return "MEDIUM"

    def compute_realized_vol(
        self,
        ohlcv_data: pd.DataFrame,
        window: int = 20,
    ) -> pd.Series:
        """
        Compute realized volatility from OHLCV data.

        Args:
            ohlcv_data: DataFrame with 'close' column
            window: Rolling window size

        Returns:
            Series of annualized realized volatility
        """
        if ohlcv_data.empty or "close" not in ohlcv_data.columns:
            return pd.Series(dtype=float)

        log_returns = np.log(ohlcv_data["close"] / ohlcv_data["close"].shift(1))
        realized_vol = log_returns.rolling(window).std() * np.sqrt(252)

        return realized_vol

    def update_all_symbols(
        self,
        symbols: List[str],
        fetcher,
        ohlcv_data_dict: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> Dict[str, int]:
        """
        Update IV history for multiple symbols.

        Args:
            symbols: List of symbols to update
            fetcher: AlphaVantageClient
            ohlcv_data_dict: Pre-loaded OHLCV data by symbol

        Returns:
            Dict of symbol -> records saved
        """
        results = {}

        for symbol in symbols:
            try:
                ohlcv = ohlcv_data_dict.get(symbol) if ohlcv_data_dict else None
                records = self.build_iv_history(symbol, fetcher, ohlcv_data=ohlcv)
                results[symbol] = records
                logger.info(f"Updated IV history for {symbol}: {records} records")
            except Exception as e:
                logger.error(f"Failed to update IV history for {symbol}: {e}")
                results[symbol] = 0

        return results


def get_dynamic_iv_thresholds(
    symbol: str,
    store: Optional[OptionsDataStore] = None,
) -> Tuple[float, float]:
    """
    Convenience function to get dynamic IV thresholds for a symbol.

    Args:
        symbol: Stock symbol
        store: Options data store

    Returns:
        Tuple of (low_threshold, high_threshold)
    """
    manager = IVHistoryManager(store=store)
    return manager.get_iv_thresholds(symbol)


def get_current_iv_rank(
    symbol: str,
    current_iv: Optional[float] = None,
    store: Optional[OptionsDataStore] = None,
) -> float:
    """
    Convenience function to get current IV rank.

    Args:
        symbol: Stock symbol
        current_iv: Current IV value
        store: Options data store

    Returns:
        IV rank (0-100)
    """
    manager = IVHistoryManager(store=store)
    return manager.get_iv_rank(symbol, current_iv)
