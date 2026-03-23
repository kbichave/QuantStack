"""
FinancialDatasetsAdapter — OHLCV adapter for financialdatasets.ai.

A low-priority fallback for price data (Alpaca/Polygon are preferred for
latency and rate limits), but the same API key is used by FundamentalsProvider
for the datasets that make this provider unique: financial statements, earnings,
insider trades, institutional ownership, etc.

Supported timeframes
--------------------
D1, W1   → native ``day`` / ``week`` interval
H1       → native ``hour`` interval
H4       → resampled from H1 via TimeframeResampler
M1–M30   → native ``minute`` interval with multiplier
S5       → not supported (raises ValueError)
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
from loguru import logger

from quantstack.config.timeframes import Timeframe
from quantstack.data.base import AssetClass, AssetClassAdapter
from quantstack.data.provider_enum import DataProvider
from quantstack.data.resampler import TimeframeResampler

from .financial_datasets_client import FinancialDatasetsClient

# Map Timeframe → (interval, interval_multiplier) for the API.
_TF_TO_API: dict[Timeframe, tuple[str, int]] = {
    Timeframe.M1: ("minute", 1),
    Timeframe.M5: ("minute", 5),
    Timeframe.M15: ("minute", 15),
    Timeframe.M30: ("minute", 30),
    Timeframe.H1: ("hour", 1),
    Timeframe.D1: ("day", 1),
    Timeframe.W1: ("week", 1),
}

# H4 is derived from H1 via resampling (same approach as AlphaVantageAdapter).
_SUPPORTED_TIMEFRAMES = {*_TF_TO_API, Timeframe.H4}


class FinancialDatasetsAdapter(AssetClassAdapter):
    """AssetClassAdapter wrapping FinancialDatasetsClient for OHLCV data.

    Args:
        api_key: FinancialDatasets.ai API key.
        base_url: API base URL (default: production).
        rate_limit_rpm: Requests per minute (default: 1000 for Developer tier).
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.financialdatasets.ai",
        rate_limit_rpm: int = 1000,
    ) -> None:
        self._client = FinancialDatasetsClient(
            api_key=api_key,
            base_url=base_url,
            rate_limit_rpm=rate_limit_rpm,
        )
        self._resampler = TimeframeResampler()

    # ── AssetClassAdapter interface ────────────────────────────────────────

    @property
    def asset_class(self) -> AssetClass:
        return AssetClass.EQUITY

    @property
    def provider(self) -> DataProvider:
        return DataProvider.FINANCIAL_DATASETS

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: Timeframe,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> pd.DataFrame:
        """Fetch OHLCV bars from FinancialDatasets.ai.

        Returns a DataFrame with DatetimeIndex named "timestamp" and
        lowercase float64 columns [open, high, low, close, volume],
        sorted ascending.  Empty DataFrame on fetch failure.

        Raises:
            ValueError: If ``timeframe`` is not supported.
        """
        if timeframe not in _SUPPORTED_TIMEFRAMES:
            raise ValueError(
                f"FinancialDatasetsAdapter does not support {timeframe}. "
                f"Supported: {sorted(tf.name for tf in _SUPPORTED_TIMEFRAMES)}"
            )

        try:
            df = self._fetch(symbol, timeframe, start_date, end_date)
        except Exception as exc:
            logger.warning(
                f"FinancialDatasets fetch failed for {symbol} {timeframe}: {exc}"
            )
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        return df.sort_index()

    def get_available_symbols(self) -> list[str]:
        # 30,000+ tickers — too many to enumerate.
        return []

    # ── Internal routing ───────────────────────────────────────────────────

    def _fetch(
        self,
        symbol: str,
        timeframe: Timeframe,
        start_date: datetime | None,
        end_date: datetime | None,
    ) -> pd.DataFrame:
        if timeframe == Timeframe.H4:
            # H4 is not a native API interval — derive from H1.
            logger.debug(f"[FinancialDatasets] Building H4 from H1 for {symbol}")
            df_1h = self._fetch_native(symbol, Timeframe.H1, start_date, end_date)
            if df_1h.empty:
                return df_1h
            return self._resampler.resample_to_higher_tf(df_1h, Timeframe.H4)

        return self._fetch_native(symbol, timeframe, start_date, end_date)

    def _fetch_native(
        self,
        symbol: str,
        timeframe: Timeframe,
        start_date: datetime | None,
        end_date: datetime | None,
    ) -> pd.DataFrame:
        interval, multiplier = _TF_TO_API[timeframe]
        start_str = start_date.strftime("%Y-%m-%d") if start_date else None
        end_str = end_date.strftime("%Y-%m-%d") if end_date else None

        logger.debug(
            f"[FinancialDatasets] Fetching {interval}×{multiplier} for {symbol}"
        )

        # Use paginated fetch to get all available bars.
        prices = self._client.get_all_historical_prices(
            ticker=symbol,
            interval=interval,
            interval_multiplier=multiplier,
            start_date=start_str,
            end_date=end_str,
        )

        if not prices:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        return self._prices_to_df(prices)

    @staticmethod
    def _prices_to_df(prices: list[dict]) -> pd.DataFrame:
        """Convert API price dicts to the standard OHLCV DataFrame contract."""
        rows = []
        for p in prices:
            # API returns: open, close, high, low, volume, time, time_milliseconds
            ts = p.get("time") or p.get("time_milliseconds")
            if ts is None:
                continue
            rows.append(
                {
                    "timestamp": ts,
                    "open": float(p["open"]),
                    "high": float(p["high"]),
                    "low": float(p["low"]),
                    "close": float(p["close"]),
                    "volume": float(p.get("volume", 0)),
                }
            )

        if not rows:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp").sort_index()
        df.index.name = "timestamp"
        return df
