"""
AlphaVantageAdapter — wraps the existing AlphaVantageClient.

AlphaVantage is the legacy/fallback provider:
  - Free tier: 5 calls/min (research and backtesting only)
  - Supports equities, some FX pairs, and commodity spot prices
  - No native 4H bars — H4 is built by resampling from H1

Supported timeframes
--------------------
M1, M5, M15, M30  → TIME_SERIES_INTRADAY (recent 30 days; use fetch_intraday)
H1                → TIME_SERIES_INTRADAY_EXTENDED (full history via monthly slices)
H4                → H1 fetched then resampled via OHLCVResampler
D1                → TIME_SERIES_DAILY_ADJUSTED
W1                → TIME_SERIES_WEEKLY_ADJUSTED
S5                → not supported (raises ValueError)
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
from loguru import logger

from quantcore.config.timeframes import Timeframe
from quantcore.data.base import AssetClass, AssetClassAdapter
from quantcore.data.fetcher import AlphaVantageClient
from quantcore.data.provider_enum import DataProvider
from quantcore.data.resampler import TimeframeResampler

# AlphaVantage interval strings for intraday endpoints
_AV_INTRADAY_INTERVALS = {
    Timeframe.M1: "1min",
    Timeframe.M5: "5min",
    Timeframe.M15: "15min",
    Timeframe.M30: "30min",
    Timeframe.H1: "60min",
}

_SUPPORTED_TIMEFRAMES = {*_AV_INTRADAY_INTERVALS, Timeframe.H4, Timeframe.D1, Timeframe.W1}


def _filter_by_date(
    df: pd.DataFrame,
    start_date: datetime | None,
    end_date: datetime | None,
) -> pd.DataFrame:
    if df.empty:
        return df
    if start_date:
        df = df[df.index >= pd.Timestamp(start_date)]
    if end_date:
        df = df[df.index <= pd.Timestamp(end_date)]
    return df


class AlphaVantageAdapter(AssetClassAdapter):
    """AssetClassAdapter wrapping AlphaVantageClient.

    The underlying client is not modified — this class is purely a
    translation layer from its timeframe-specific methods to the
    unified ``fetch_ohlcv`` contract.

    Args:
        api_key: Alpha Vantage API key.  Falls back to ``ALPHA_VANTAGE_API_KEY``
                 env var if not provided.
    """

    def __init__(self, api_key: str | None = None) -> None:
        self._client = AlphaVantageClient(api_key=api_key)
        self._resampler = TimeframeResampler()

    # ── AssetClassAdapter interface ───────────────────────────────────────────

    @property
    def asset_class(self) -> AssetClass:
        return AssetClass.EQUITY

    @property
    def provider(self) -> DataProvider:
        return DataProvider.ALPHA_VANTAGE

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: Timeframe,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> pd.DataFrame:
        """Fetch OHLCV bars from Alpha Vantage.

        Returns a DataFrame with DatetimeIndex named "timestamp" and
        lowercase float64 columns [open, high, low, close, volume],
        sorted ascending.  Empty DataFrame on fetch failure.

        Raises:
            ValueError: If ``timeframe`` is not supported (e.g. S5).
        """
        if timeframe not in _SUPPORTED_TIMEFRAMES:
            raise ValueError(
                f"AlphaVantageAdapter does not support {timeframe}. "
                f"Supported: {sorted(tf.name for tf in _SUPPORTED_TIMEFRAMES)}"
            )

        try:
            df = self._fetch(symbol, timeframe)
        except Exception as exc:
            logger.warning(f"AlphaVantage fetch failed for {symbol} {timeframe}: {exc}")
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        df = _filter_by_date(df, start_date, end_date)
        return df.sort_index()

    def get_available_symbols(self) -> list[str]:
        # Alpha Vantage does not expose a symbol list endpoint.
        return []

    # ── Internal routing ──────────────────────────────────────────────────────

    def _fetch(self, symbol: str, timeframe: Timeframe) -> pd.DataFrame:
        if timeframe in _AV_INTRADAY_INTERVALS:
            interval = _AV_INTRADAY_INTERVALS[timeframe]

            if timeframe == Timeframe.H1:
                # H1 uses the extended-history endpoint (monthly slices) for
                # full multi-year history.  fetch_intraday only returns ~30 days.
                logger.debug(f"[AV] Fetching full H1 history for {symbol}")
                return self._client.fetch_all_intraday_history(symbol, interval="60min")

            # Sub-hourly timeframes: recent data only (~30 days via compact/full)
            logger.debug(f"[AV] Fetching {interval} intraday for {symbol}")
            return self._client.fetch_intraday(symbol, interval=interval)

        if timeframe == Timeframe.H4:
            # H4 is not a native AV interval — derive from H1.
            logger.debug(f"[AV] Building H4 from H1 for {symbol}")
            df_1h = self._client.fetch_all_intraday_history(symbol, interval="60min")
            return self._resampler.resample_to_higher_tf(df_1h, Timeframe.H4)

        if timeframe == Timeframe.D1:
            logger.debug(f"[AV] Fetching daily for {symbol}")
            return self._client.fetch_daily(symbol)

        if timeframe == Timeframe.W1:
            logger.debug(f"[AV] Fetching weekly for {symbol}")
            return self._client.fetch_weekly(symbol)

        # Should never reach here given _SUPPORTED_TIMEFRAMES check above.
        raise ValueError(f"Unhandled timeframe: {timeframe}")
