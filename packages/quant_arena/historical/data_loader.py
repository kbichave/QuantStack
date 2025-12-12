# Copyright 2024 QuantArena Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Data loader for historical simulation.

Loads daily OHLCV data via QuantCore MCP and aligns calendars across symbols.
Provides an iterator interface for stepping through trading days.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Dict, Iterator, List, Optional, Tuple

import pandas as pd
from loguru import logger

from quant_arena.historical.universe import SymbolUniverse


@dataclass
class MarketSnapshot:
    """
    Market data snapshot for a single trading day.

    Contains OHLCV data for all symbols in the universe that have
    data available on that date.
    """

    date: date
    data: Dict[str, Dict[str, float]]  # {symbol: {open, high, low, close, volume}}
    available_symbols: List[str]

    def get_close(self, symbol: str) -> Optional[float]:
        """Get closing price for a symbol."""
        if symbol in self.data:
            return self.data[symbol].get("close")
        return None

    def get_prices(self) -> Dict[str, float]:
        """Get closing prices for all symbols."""
        return {sym: data["close"] for sym, data in self.data.items()}

    def __repr__(self) -> str:
        return f"MarketSnapshot(date={self.date}, symbols={self.available_symbols})"


@dataclass
class MTFSnapshot:
    """
    Multi-timeframe market data snapshot.

    Contains OHLCV data at multiple timeframes for all symbols:
    - 1H: Hourly bars for execution timing
    - 4H: 4-hour bars for swing context
    - D1: Daily bars for intermediate trend
    - W1: Weekly bars for macro regime
    """

    date: date
    hour: int  # Hour of day (e.g., 10, 14)
    data: Dict[str, Dict[str, pd.DataFrame]]  # {symbol: {timeframe: DataFrame}}
    available_symbols: List[str]

    def get_timeframe_data(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Get data for a specific symbol and timeframe."""
        if symbol in self.data and timeframe in self.data[symbol]:
            return self.data[symbol][timeframe]
        return None

    def get_latest_close(self, symbol: str, timeframe: str = "1H") -> Optional[float]:
        """Get latest closing price for a symbol at specified timeframe."""
        df = self.get_timeframe_data(symbol, timeframe)
        if df is not None and not df.empty:
            close_col = "close" if "close" in df.columns else "Close"
            return float(df[close_col].iloc[-1])
        return None

    def to_dict(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """Get all timeframe data for a symbol as dict."""
        return self.data.get(symbol, {})

    def __repr__(self) -> str:
        return f"MTFSnapshot(date={self.date}, hour={self.hour}, symbols={self.available_symbols})"


class DataLoader:
    """
    Loads and manages historical market data for simulation.

    Uses QuantCore MCP tools (load_market_data) to fetch daily OHLCV.
    Aligns trading calendars across multiple symbols and provides
    an iterator interface for stepping through the simulation.

    Usage:
        universe = SymbolUniverse(["SPY", "QQQ", "WTI"])
        loader = DataLoader(universe)
        await loader.load_data()

        for date, snapshot in loader.iterate_days():
            print(f"{date}: SPY close = {snapshot.get_close('SPY')}")
    """

    def __init__(
        self,
        universe: SymbolUniverse,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        enable_mtf: bool = False,
    ):
        """
        Initialize data loader.

        Args:
            universe: Symbol universe to load data for
            start_date: Start date (None = earliest common date)
            end_date: End date (None = today)
            enable_mtf: Enable multi-timeframe data loading
        """
        self.universe = universe
        self.start_date = start_date
        self.end_date = end_date or date.today()
        self.enable_mtf = enable_mtf

        # Data storage: {ticker: DataFrame with DatetimeIndex}
        self._data: Dict[str, pd.DataFrame] = {}

        # MTF data storage: {ticker: {timeframe: DataFrame}}
        self._mtf_data: Dict[str, Dict[str, pd.DataFrame]] = {}

        # Aligned trading calendar
        self._trading_days: List[date] = []

        # Pre-computed snapshots for fast iteration
        self._snapshots: Dict[date, MarketSnapshot] = {}

        self._loaded = False
        self._mtf_loaded = False

    async def load_data(self) -> None:
        """
        Load data for all symbols in universe.

        Fetches data via QuantCore MCP, aligns calendars, and
        pre-computes snapshots for efficient iteration.
        """
        if self._loaded:
            logger.info("Data already loaded, skipping")
            return

        logger.info(f"Loading data for {len(self.universe)} symbols...")

        # Load data for each ticker
        for logical_name in self.universe.symbols:
            ticker = self.universe.get_ticker(logical_name)
            try:
                df = await self._fetch_ticker_data(ticker)
                if df is not None and not df.empty:
                    self._data[logical_name] = df
                    logger.info(f"Loaded {logical_name} ({ticker}): {len(df)} bars")
                else:
                    logger.warning(f"No data for {logical_name} ({ticker})")
            except Exception as e:
                logger.error(f"Failed to load {logical_name}: {e}")

        if not self._data:
            raise RuntimeError("No data loaded for any symbol")

        # Align calendars
        self._align_calendars()

        # Pre-compute snapshots
        self._build_snapshots()

        self._loaded = True
        logger.info(f"Data loaded: {len(self._trading_days)} trading days")

    async def _fetch_ticker_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV data for a single ticker from DuckDB.

        Uses the local DuckDB database which contains daily and hourly data.
        """
        try:
            import duckdb
            from pathlib import Path

            # Use simulation copy of database to avoid lock conflicts with MCP server
            base_path = Path(__file__).parent.parent.parent.parent / "data"
            db_path = str(base_path / "trader_sim.duckdb")

            # Fall back to original if sim copy doesn't exist
            if not Path(db_path).exists():
                db_path = str(base_path / "trader.duckdb")

            conn = duckdb.connect(db_path, read_only=True)

            query = """
                SELECT timestamp, open, high, low, close, volume
                FROM ohlcv
                WHERE symbol = ? AND timeframe = ?
                ORDER BY timestamp
            """

            df = conn.execute(query, [ticker, "1D"]).fetchdf()
            conn.close()

            if df is not None and not df.empty:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df = df.set_index("timestamp")
                logger.info(f"Loaded {len(df)} rows for {ticker} from DuckDB (daily)")
                return df

            logger.warning(f"No daily data found for {ticker} in DuckDB")
            return None

        except ImportError:
            logger.error("DuckDB not available - cannot load data")
            return None
        except Exception as e:
            logger.error(f"DuckDB load failed for {ticker}: {e}")
            return None

    def _align_calendars(self) -> None:
        """
        Align trading calendars across all symbols.

        Uses SPY as the reference calendar (most liquid US market),
        then filters to dates where at least one symbol has data.
        """
        if not self._data:
            return

        # Use SPY as reference if available, otherwise first symbol
        reference = "SPY" if "SPY" in self._data else list(self._data.keys())[0]
        ref_dates = set(self._data[reference].index.date)

        # Get union of all dates (for symbols that may have different calendars)
        all_dates: set = set()
        for df in self._data.values():
            all_dates.update(df.index.date)

        # Filter by date range
        effective_start = self.start_date
        if effective_start is None:
            effective_start = self.universe.get_earliest_common_date()

        filtered_dates = [
            d for d in sorted(all_dates) if effective_start <= d <= self.end_date
        ]

        self._trading_days = filtered_dates
        logger.info(
            f"Aligned calendar: {len(self._trading_days)} trading days "
            f"from {self._trading_days[0]} to {self._trading_days[-1]}"
        )

    def _build_snapshots(self) -> None:
        """Pre-compute market snapshots for each trading day."""
        ticker_to_logical = self.universe.ticker_to_logical_map()

        for day in self._trading_days:
            day_data = {}
            available = []

            for logical_name, df in self._data.items():
                # Get data for this day
                try:
                    # Handle both datetime and date index
                    if isinstance(df.index, pd.DatetimeIndex):
                        day_df = df[df.index.date == day]
                    else:
                        day_df = df.loc[[day]] if day in df.index else pd.DataFrame()

                    if not day_df.empty:
                        row = day_df.iloc[-1]  # Use last row if multiple
                        day_data[logical_name] = {
                            "open": float(row.get("open", row.get("Open", 0))),
                            "high": float(row.get("high", row.get("High", 0))),
                            "low": float(row.get("low", row.get("Low", 0))),
                            "close": float(row.get("close", row.get("Close", 0))),
                            "volume": float(row.get("volume", row.get("Volume", 0))),
                        }
                        available.append(logical_name)
                except Exception as e:
                    logger.debug(f"No data for {logical_name} on {day}: {e}")

            self._snapshots[day] = MarketSnapshot(
                date=day,
                data=day_data,
                available_symbols=available,
            )

    def iterate_days(self) -> Iterator[Tuple[date, MarketSnapshot]]:
        """
        Iterate through trading days chronologically.

        Yields:
            Tuple of (date, MarketSnapshot) for each trading day
        """
        if not self._loaded:
            raise RuntimeError("Data not loaded. Call load_data() first.")

        for day in self._trading_days:
            yield day, self._snapshots[day]

    def get_day(self, day: date) -> Optional[MarketSnapshot]:
        """Get market snapshot for a specific day."""
        return self._snapshots.get(day)

    def get_lookback(
        self,
        current_date: date,
        days: int,
    ) -> List[MarketSnapshot]:
        """
        Get lookback window of snapshots.

        Args:
            current_date: Current simulation date
            days: Number of trading days to look back

        Returns:
            List of MarketSnapshots, oldest first
        """
        try:
            current_idx = self._trading_days.index(current_date)
        except ValueError:
            return []

        start_idx = max(0, current_idx - days)
        lookback_dates = self._trading_days[start_idx:current_idx]

        return [self._snapshots[d] for d in lookback_dates]

    def get_price_history(
        self,
        symbol: str,
        end_date: date,
        days: int = 252,
    ) -> pd.Series:
        """
        Get price history for a symbol.

        Args:
            symbol: Logical symbol name
            end_date: End date for history
            days: Number of trading days

        Returns:
            Series of closing prices indexed by date
        """
        if symbol not in self._data:
            return pd.Series(dtype=float)

        df = self._data[symbol]

        # Filter by date
        if isinstance(df.index, pd.DatetimeIndex):
            mask = df.index.date <= end_date
        else:
            mask = df.index <= end_date

        filtered = df[mask].tail(days)

        close_col = "close" if "close" in filtered.columns else "Close"
        return filtered[close_col]

    @property
    def trading_days(self) -> List[date]:
        """Get list of trading days."""
        return self._trading_days.copy()

    @property
    def start(self) -> Optional[date]:
        """Get actual start date of loaded data."""
        return self._trading_days[0] if self._trading_days else None

    @property
    def end(self) -> Optional[date]:
        """Get actual end date of loaded data."""
        return self._trading_days[-1] if self._trading_days else None

    def __len__(self) -> int:
        """Number of trading days."""
        return len(self._trading_days)

    def __repr__(self) -> str:
        return (
            f"DataLoader(symbols={len(self.universe)}, "
            f"days={len(self._trading_days)}, loaded={self._loaded})"
        )

    # =========================================================================
    # MULTI-TIMEFRAME DATA LOADING
    # =========================================================================

    async def load_mtf_data(self) -> None:
        """
        Load multi-timeframe data for all symbols.

        Fetches data at 1H, 4H, 1D, and 1W timeframes.
        - 1H and 1D are loaded directly from DuckDB
        - 4H is resampled from 1H
        - 1W is resampled from 1D
        """
        if self._mtf_loaded:
            logger.info("MTF data already loaded, skipping")
            return

        logger.info(f"Loading MTF data for {len(self.universe)} symbols...")

        # Load 1H and 1D first, then derive 4H and 1W
        base_timeframes = ["1H", "1D"]
        derived_timeframes = ["4H", "1W"]

        for logical_name in self.universe.symbols:
            ticker = self.universe.get_ticker(logical_name)
            self._mtf_data[logical_name] = {}

            # Load base timeframes from DuckDB
            for tf in base_timeframes:
                try:
                    df = await self._fetch_ticker_data_tf(ticker, tf)
                    if df is not None and not df.empty:
                        self._mtf_data[logical_name][tf] = df
                        logger.info(f"Loaded {logical_name} {tf}: {len(df):,} bars")
                except Exception as e:
                    logger.warning(f"Failed to load {logical_name} {tf}: {e}")

            # Create derived timeframes via resampling
            for tf in derived_timeframes:
                try:
                    df = await self._fetch_ticker_data_tf(ticker, tf)
                    if df is not None and not df.empty:
                        self._mtf_data[logical_name][tf] = df
                        logger.info(
                            f"Created {logical_name} {tf}: {len(df):,} bars (resampled)"
                        )
                except Exception as e:
                    logger.warning(f"Failed to create {logical_name} {tf}: {e}")

        self._mtf_loaded = True
        logger.info("MTF data loaded successfully")

    async def _fetch_ticker_data_tf(
        self,
        ticker: str,
        timeframe: str,
    ) -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV data for a ticker at a specific timeframe from DuckDB.

        DuckDB has: 1H (hourly) and 1D (daily) timeframes.
        4H and 1W are resampled from available data.

        Args:
            ticker: The ticker symbol
            timeframe: Timeframe string (1H, 4H, 1D, 1W)

        Returns:
            DataFrame with OHLCV data or None
        """
        try:
            import duckdb
            from pathlib import Path

            base_path = Path(__file__).parent.parent.parent.parent / "data"
            db_path = str(base_path / "trader_sim.duckdb")

            if not Path(db_path).exists():
                db_path = str(base_path / "trader.duckdb")

            conn = duckdb.connect(db_path, read_only=True)

            # Map requested timeframe to available DB timeframes
            # DuckDB has: 1H, 1D
            db_tf = None
            needs_resample = False

            if timeframe in ["1H"]:
                db_tf = "1H"
            elif timeframe in ["1D"]:
                db_tf = "1D"
            elif timeframe == "4H":
                # Load 1H data and resample to 4H
                db_tf = "1H"
                needs_resample = True
            elif timeframe == "1W":
                # Load 1D data and resample to weekly
                db_tf = "1D"
                needs_resample = True
            else:
                db_tf = timeframe

            query = """
                SELECT timestamp, open, high, low, close, volume
                FROM ohlcv
                WHERE symbol = ? AND timeframe = ?
                ORDER BY timestamp
            """

            df = conn.execute(query, [ticker, db_tf]).fetchdf()
            conn.close()

            if df is not None and not df.empty:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df = df.set_index("timestamp")

                # Resample if needed
                if needs_resample:
                    df = self._resample_data(df, timeframe)
                    logger.debug(
                        f"Resampled {ticker} from {db_tf} to {timeframe}: {len(df)} bars"
                    )
                else:
                    logger.debug(f"Loaded {ticker} {timeframe}: {len(df)} bars")

                return df

            logger.warning(f"No {timeframe} data found for {ticker}")
            return None

        except Exception as e:
            logger.warning(f"DuckDB load failed for {ticker} {timeframe}: {e}")
            return None

    def _resample_data(
        self,
        df: pd.DataFrame,
        target_tf: str,
    ) -> pd.DataFrame:
        """
        Resample OHLCV data to a different timeframe.

        Args:
            df: Source DataFrame with OHLCV data
            target_tf: Target timeframe (4H, 1W, etc.)

        Returns:
            Resampled DataFrame
        """
        resample_rules = {
            "1H": "1h",
            "4H": "4h",
            "1D": "1D",
            "1W": "W-FRI",
        }

        rule = resample_rules.get(target_tf)
        if rule is None:
            return df

        try:
            # Detect column names (handle both uppercase and lowercase)
            cols = df.columns.str.lower()
            close_col = "close" if "close" in cols.tolist() else "Close"
            open_col = "open" if "open" in cols.tolist() else "Open"
            high_col = "high" if "high" in cols.tolist() else "High"
            low_col = "low" if "low" in cols.tolist() else "Low"
            vol_col = "volume" if "volume" in cols.tolist() else "Volume"

            # Handle case where columns might already be lowercase
            if close_col not in df.columns:
                close_col = (
                    df.columns[df.columns.str.lower() == "close"][0]
                    if any(df.columns.str.lower() == "close")
                    else "close"
                )
            if open_col not in df.columns:
                open_col = (
                    df.columns[df.columns.str.lower() == "open"][0]
                    if any(df.columns.str.lower() == "open")
                    else "open"
                )
            if high_col not in df.columns:
                high_col = (
                    df.columns[df.columns.str.lower() == "high"][0]
                    if any(df.columns.str.lower() == "high")
                    else "high"
                )
            if low_col not in df.columns:
                low_col = (
                    df.columns[df.columns.str.lower() == "low"][0]
                    if any(df.columns.str.lower() == "low")
                    else "low"
                )
            if vol_col not in df.columns:
                vol_col = (
                    df.columns[df.columns.str.lower() == "volume"][0]
                    if any(df.columns.str.lower() == "volume")
                    else "volume"
                )

            resampled = df.resample(rule).agg(
                {
                    open_col: "first",
                    high_col: "max",
                    low_col: "min",
                    close_col: "last",
                    vol_col: "sum",
                }
            )

            # Rename columns to lowercase
            resampled.columns = ["open", "high", "low", "close", "volume"]
            return resampled.dropna()

        except Exception as e:
            logger.warning(f"Resample failed: {e}")
            return df

    def get_mtf_snapshot(
        self,
        current_date: date,
        hour: int = 10,
        lookback_bars: Dict[str, int] = None,
    ) -> MTFSnapshot:
        """
        Get multi-timeframe data snapshot for a specific date/time.

        Args:
            current_date: The simulation date
            hour: Hour of day (e.g., 10 for 10 AM)
            lookback_bars: Dict of {timeframe: num_bars} for lookback
                          Default: {"1H": 50, "4H": 30, "D1": 50, "W1": 20}

        Returns:
            MTFSnapshot with data at all timeframes
        """
        if lookback_bars is None:
            lookback_bars = {"1H": 50, "4H": 30, "1D": 50, "1W": 20}

        # Target datetime
        target_dt = datetime.combine(
            current_date, datetime.min.time().replace(hour=hour)
        )

        snapshot_data = {}
        available = []

        for logical_name in self.universe.symbols:
            if logical_name not in self._mtf_data:
                continue

            symbol_data = {}
            has_data = False

            for tf, df in self._mtf_data[logical_name].items():
                if df is None or df.empty:
                    continue

                # Filter to data before target datetime
                if isinstance(df.index, pd.DatetimeIndex):
                    mask = df.index <= target_dt
                else:
                    mask = df.index <= target_dt.date()

                filtered = df[mask]

                # Get lookback bars
                n_bars = lookback_bars.get(tf, 50)
                symbol_data[tf] = filtered.tail(n_bars).copy()

                if not symbol_data[tf].empty:
                    has_data = True

            if has_data:
                snapshot_data[logical_name] = symbol_data
                available.append(logical_name)

        return MTFSnapshot(
            date=current_date,
            hour=hour,
            data=snapshot_data,
            available_symbols=available,
        )

    def get_symbol_mtf_data(
        self,
        symbol: str,
        end_date: date,
        lookback_bars: Dict[str, int] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Get MTF data for a single symbol up to a specific date.

        Args:
            symbol: Logical symbol name
            end_date: End date for data
            lookback_bars: Dict of {timeframe: num_bars}

        Returns:
            Dict mapping timeframe -> DataFrame
        """
        if lookback_bars is None:
            lookback_bars = {"1H": 50, "4H": 30, "1D": 50, "1W": 20}

        if symbol not in self._mtf_data:
            return {}

        result = {}
        target_dt = datetime.combine(end_date, datetime.max.time())

        for tf, df in self._mtf_data[symbol].items():
            if df is None or df.empty:
                continue

            # Filter by date
            if isinstance(df.index, pd.DatetimeIndex):
                mask = df.index <= target_dt
            else:
                mask = df.index <= end_date

            filtered = df[mask]
            n_bars = lookback_bars.get(tf, 50)
            result[tf] = filtered.tail(n_bars).copy()

        return result

    def get_intraday_bars(
        self,
        current_date: date,
        symbol: str,
        timeframe: str = "4H",
    ) -> List[int]:
        """
        Get list of intraday bar hours for a date.

        For simulation purposes, we process at specific hours during
        the trading day (e.g., 10 AM and 2 PM for 4H bars).

        Args:
            current_date: The simulation date
            symbol: Symbol to check
            timeframe: Timeframe (1H, 4H)

        Returns:
            List of hours to process (e.g., [10, 14] for 4H)
        """
        # Standard US market hours: 9:30 AM - 4:00 PM ET
        # 4H bars roughly at: 10 AM (open + 30min), 2 PM
        if timeframe == "4H":
            return [10, 14]
        elif timeframe == "1H":
            return [10, 11, 12, 13, 14, 15]
        else:
            return [16]  # Daily - end of day only
