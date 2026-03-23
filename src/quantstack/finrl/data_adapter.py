"""
Data adapter — bridges DataStore / Alpha Vantage to FinRL DataFrame format.

FinRL expects: date, tic, open, high, low, close, volume + indicator columns.
Our DataStore has OHLCV in DuckDB. This adapter translates between the two.

Also provides a custom Alpha Vantage data processor compatible with FinRL's
processor interface.

Usage:
    from quantstack.finrl.data_adapter import FinRLDataAdapter

    adapter = FinRLDataAdapter()
    df = adapter.fetch_and_format(["SPY", "QQQ"], "2023-01-01", "2024-01-01")
"""

from __future__ import annotations

import os
from datetime import datetime

import numpy as np
import pandas as pd
import requests
from loguru import logger


class FinRLDataAdapter:
    """
    Converts market data from our DataStore or Alpha Vantage into the
    DataFrame format that FinRL environments expect.

    FinRL format:
        date (str YYYY-MM-DD), tic (str), open, high, low, close, volume
        + optional indicator columns
    """

    def __init__(self, api_key: str | None = None):
        self._api_key = api_key or os.getenv("ALPHA_VANTAGE_API_KEY", "")

    def fetch_and_format(
        self,
        symbols: list[str],
        start_date: str,
        end_date: str,
        add_indicators: bool = True,
        indicator_list: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV for symbols and return in FinRL-compatible format.

        Tries DataStore first, falls back to Alpha Vantage.
        """
        frames = []
        for symbol in symbols:
            df = self._fetch_symbol(symbol, start_date, end_date)
            if not df.empty:
                frames.append(df)

        if not frames:
            logger.warning("[FinRLDataAdapter] No data fetched for any symbol.")
            return pd.DataFrame()

        combined = pd.concat(frames, ignore_index=True)
        combined = combined.sort_values(["date", "tic"]).reset_index(drop=True)

        if add_indicators:
            combined = self._add_technical_indicators(combined, indicator_list)

        return combined

    def from_datastore(
        self,
        symbols: list[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Load from DataStore (DuckDB) and format for FinRL."""
        try:
            store = DataStore(read_only=True)
            frames = []
            for symbol in symbols:
                try:
                    rows = store.conn.execute(
                        """
                        SELECT date, open, high, low, close, volume
                        FROM ohlcv_daily
                        WHERE symbol = ? AND date >= ? AND date <= ?
                        ORDER BY date
                        """,
                        [symbol, start_date, end_date],
                    ).fetchall()
                    if rows:
                        df = pd.DataFrame(
                            rows,
                            columns=["date", "open", "high", "low", "close", "volume"],
                        )
                        df["tic"] = symbol
                        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
                        frames.append(df)
                except Exception as e:
                    logger.debug(f"[FinRLDataAdapter] DataStore query for {symbol}: {e}")
            store.close()

            if frames:
                return pd.concat(frames, ignore_index=True)
        except Exception:
            logger.debug("[FinRLDataAdapter] DataStore query failed.")

        return pd.DataFrame()

    def from_alphavantage(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Fetch from Alpha Vantage and format for FinRL."""
        if not self._api_key:
            logger.warning("[FinRLDataAdapter] No ALPHA_VANTAGE_API_KEY set.")
            return pd.DataFrame()

        url = "https://www.alphavantage.co/query"
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": symbol,
            "outputsize": "full",
            "apikey": self._api_key,
        }

        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            if "Note" in data or "Information" in data:
                logger.warning(f"[FinRLDataAdapter] AV rate limit for {symbol}")
                return pd.DataFrame()

            ts = data.get("Time Series (Daily)", {})
            if not ts:
                return pd.DataFrame()

            records = []
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")

            for date_str, ohlcv in ts.items():
                dt = datetime.strptime(date_str, "%Y-%m-%d")
                if start_dt <= dt <= end_dt:
                    records.append(
                        {
                            "date": date_str,
                            "tic": symbol,
                            "open": float(ohlcv["1. open"]),
                            "high": float(ohlcv["2. high"]),
                            "low": float(ohlcv["3. low"]),
                            "close": float(ohlcv["4. close"]),
                            "volume": float(ohlcv["5. volume"]),
                        }
                    )

            if not records:
                return pd.DataFrame()

            return pd.DataFrame(records).sort_values("date").reset_index(drop=True)

        except requests.RequestException as e:
            logger.error(f"[FinRLDataAdapter] AV fetch failed for {symbol}: {e}")
            return pd.DataFrame()

    def _fetch_symbol(
        self, symbol: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Fetch single symbol — DataStore first, AV fallback."""
        df = self.from_datastore([symbol], start_date, end_date)
        if not df.empty:
            return df
        return self.from_alphavantage(symbol, start_date, end_date)

    def _add_technical_indicators(
        self,
        df: pd.DataFrame,
        indicator_list: list[str] | None = None,
    ) -> pd.DataFrame:
        """Add technical indicators to the DataFrame.

        Uses FinRL's FeatureEngineer when available, falls back to
        manual computation for core indicators.
        """
        try:
            fe = FeatureEngineer(
                use_technical_indicator=True,
                tech_indicator_list=indicator_list
                or [
                    "macd",
                    "boll_ub",
                    "boll_lb",
                    "rsi_30",
                    "cci_30",
                    "dx_30",
                    "close_30_sma",
                    "close_60_sma",
                ],
                use_vix=False,
                use_turbulence=False,
            )
            return fe.preprocess_data(df)
        except Exception as e:
            logger.debug(f"[FinRLDataAdapter] FeatureEngineer failed: {e}")
            return self._add_basic_indicators(df)

    @staticmethod
    def _add_basic_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Fallback: add basic indicators without FinRL dependency."""
        result = df.copy()

        for tic in result["tic"].unique():
            mask = result["tic"] == tic
            close = result.loc[mask, "close"]

            # SMA
            result.loc[mask, "close_30_sma"] = close.rolling(30).mean()
            result.loc[mask, "close_60_sma"] = close.rolling(60).mean()

            # RSI (14-period)
            delta = close.diff()
            gain = delta.clip(lower=0).rolling(14).mean()
            loss = (-delta.clip(upper=0)).rolling(14).mean()
            rs = gain / (loss + 1e-10)
            result.loc[mask, "rsi_30"] = 100 - (100 / (1 + rs))

            # Bollinger Bands
            sma20 = close.rolling(20).mean()
            std20 = close.rolling(20).std()
            result.loc[mask, "boll_ub"] = sma20 + 2 * std20
            result.loc[mask, "boll_lb"] = sma20 - 2 * std20

            # MACD
            ema12 = close.ewm(span=12).mean()
            ema26 = close.ewm(span=26).mean()
            result.loc[mask, "macd"] = ema12 - ema26

            # Turbulence (simplified)
            returns = close.pct_change()
            result.loc[mask, "turbulence"] = (
                returns.rolling(20).apply(
                    lambda x: float(np.sum((x - x.mean()) ** 2)), raw=True
                )
            )

        return result.fillna(0)

    @staticmethod
    def ohlcv_to_finrl(
        ohlcv_df: pd.DataFrame, symbol: str
    ) -> pd.DataFrame:
        """Convert a raw OHLCV DataFrame (with DatetimeIndex) to FinRL format."""
        df = ohlcv_df.copy()
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
            date_col = df.columns[0]
            df = df.rename(columns={date_col: "date"})
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        df["tic"] = symbol
        required = ["date", "tic", "open", "high", "low", "close", "volume"]
        for col in required:
            if col not in df.columns:
                df[col] = 0
        return df[required + [c for c in df.columns if c not in required]]
