"""
AlpacaAdapter — wraps alpaca-py StockHistoricalDataClient.

Why Alpaca for historical data:
  - Free tier: unlimited historical bars (no per-call limit) for paper/backtest
  - Supports M1 through D1 natively via TimeFrame enum
  - Same API key used for live/paper trading (no separate data key needed)
  - Clean DataFrame output with VWAP and trade_count included

Requires: ``alpaca-py>=0.20.0``  (``uv pip install -e ".[alpaca]"``)

Supported timeframes
--------------------
M1, M5, M15, M30  → alpaca TimeFrame(1/5/15/30, TimeFrameUnit.Minute)
H1                → alpaca TimeFrame(1, TimeFrameUnit.Hour)
H4                → H1 fetched then resampled
D1                → alpaca TimeFrame(1, TimeFrameUnit.Day)
W1                → D1 fetched then resampled
S5                → not supported (Alpaca minimum granularity is 1 minute)
"""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
from loguru import logger

from quantcore.config.timeframes import Timeframe
from quantcore.data.base import AssetClass, AssetClassAdapter
from quantcore.data.provider_enum import DataProvider
from quantcore.data.resampler import TimeframeResampler

try:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    _ALPACA_AVAILABLE = True
except ImportError:
    _ALPACA_AVAILABLE = False

try:
    from alpaca.data.historical import OptionHistoricalDataClient
    from alpaca.data.requests import OptionSnapshotRequest

    _ALPACA_OPTIONS_AVAILABLE = True
except ImportError:
    _ALPACA_OPTIONS_AVAILABLE = False


# Timeframes that Alpaca serves natively (no resample needed)
_ALPACA_NATIVE = {
    Timeframe.M1: (1, "Minute"),
    Timeframe.M5: (5, "Minute"),
    Timeframe.M15: (15, "Minute"),
    Timeframe.M30: (30, "Minute"),
    Timeframe.H1: (1, "Hour"),
    Timeframe.D1: (1, "Day"),
}

_SUPPORTED_TIMEFRAMES = {*_ALPACA_NATIVE, Timeframe.H4, Timeframe.W1}


def _require_alpaca() -> None:
    if not _ALPACA_AVAILABLE:
        raise ImportError("alpaca-py is not installed. Run: uv pip install -e '.[alpaca]'")


def _to_alpaca_tf(timeframe: Timeframe) -> TimeFrame:
    amount, unit_name = _ALPACA_NATIVE[timeframe]
    unit = getattr(TimeFrameUnit, unit_name)
    return TimeFrame(amount, unit)


def _bars_to_df(bar_set) -> pd.DataFrame:
    """Convert alpaca-py BarSet to standard OHLCV DataFrame."""
    if not bar_set or not bar_set.data:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    # BarSet.data is Dict[str, List[Bar]]; we expect single-symbol calls
    rows = []
    for bars in bar_set.data.values():
        for bar in bars:
            rows.append(
                {
                    "timestamp": bar.timestamp,
                    "open": float(bar.open),
                    "high": float(bar.high),
                    "low": float(bar.low),
                    "close": float(bar.close),
                    "volume": float(bar.volume),
                    "vwap": float(bar.vwap) if bar.vwap is not None else None,
                    "trade_count": int(bar.trade_count) if bar.trade_count is not None else None,
                }
            )

    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()
    df.index.name = "timestamp"
    return df


def _parse_alpaca_expiry(contract_symbol: str) -> str | None:
    """Extract expiry date string (YYYY-MM-DD) from Alpaca contract symbol.

    Alpaca contract symbols follow the OCC format: <underlying><YYMMDD><C|P><strike*1000>.
    Examples: SPY240119C00480000, AAPL240119P00170000
    """
    import re

    match = re.search(r"([CP])(\d{6})\d+$", contract_symbol)
    if not match:
        return None
    # Find the date portion (comes before C/P)
    date_match = re.search(r"(\d{6})[CP]", contract_symbol)
    if not date_match:
        return None
    date_str = date_match.group(1)
    try:
        year = 2000 + int(date_str[:2])
        month = int(date_str[2:4])
        day = int(date_str[4:6])
        return f"{year:04d}-{month:02d}-{day:02d}"
    except (ValueError, IndexError):
        return None


def _parse_alpaca_strike(contract_symbol: str) -> float | None:
    """Extract strike price from Alpaca contract symbol.

    The last 8 digits encode strike * 1000 (e.g., 00480000 = $480.00).
    """
    import re

    match = re.search(r"[CP](\d{8})$", contract_symbol)
    if not match:
        return None
    try:
        return int(match.group(1)) / 1000.0
    except (ValueError, IndexError):
        return None


class AlpacaAdapter(AssetClassAdapter):
    """AssetClassAdapter wrapping Alpaca's StockHistoricalDataClient.

    Args:
        api_key:    Alpaca API key (``ALPACA_API_KEY`` env var as fallback).
        secret_key: Alpaca secret key (``ALPACA_SECRET_KEY`` env var as fallback).
        paper:      If True, connect to paper-trading endpoint.  Alpaca historical
                    data uses the same endpoint regardless, but this flag is passed
                    through for completeness.
    """

    def __init__(
        self,
        api_key: str | None = None,
        secret_key: str | None = None,
        paper: bool = True,
    ) -> None:
        _require_alpaca()
        import os

        self._api_key = api_key or os.getenv("ALPACA_API_KEY", "")
        self._secret_key = secret_key or os.getenv("ALPACA_SECRET_KEY", "")
        self._paper = paper
        self._client = StockHistoricalDataClient(
            api_key=self._api_key,
            secret_key=self._secret_key,
        )
        self._resampler = TimeframeResampler()

    # ── AssetClassAdapter interface ───────────────────────────────────────────

    @property
    def asset_class(self) -> AssetClass:
        return AssetClass.EQUITY

    @property
    def provider(self) -> DataProvider:
        return DataProvider.ALPACA

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: Timeframe,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> pd.DataFrame:
        """Fetch OHLCV bars from Alpaca.

        Returns a DataFrame with DatetimeIndex named "timestamp", lowercase
        float64 columns [open, high, low, close, volume], sorted ascending.
        Columns vwap and trade_count are included when available.

        Raises:
            ValueError: If ``timeframe`` is S5 (below Alpaca's 1-minute floor).
            ImportError: If alpaca-py is not installed.
        """
        if timeframe == Timeframe.S5:
            raise ValueError(
                "AlpacaAdapter minimum granularity is 1 minute (Timeframe.M1). "
                "Use PolygonTickAdapter or IBKRTickAdapter for sub-minute data."
            )
        if timeframe not in _SUPPORTED_TIMEFRAMES:
            raise ValueError(
                f"AlpacaAdapter does not support {timeframe}. "
                f"Supported: {sorted(tf.name for tf in _SUPPORTED_TIMEFRAMES)}"
            )

        try:
            df = self._fetch(symbol, timeframe, start_date, end_date)
        except Exception as exc:
            logger.warning(f"Alpaca fetch failed for {symbol} {timeframe}: {exc}")
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        return df.sort_index()

    def get_available_symbols(self) -> list[str]:
        # Alpaca supports thousands of US equities; returning the full list
        # is expensive.  Callers should query the assets endpoint directly
        # if they need a complete universe.
        return []

    # ── Internal fetch ────────────────────────────────────────────────────────

    def _fetch(
        self,
        symbol: str,
        timeframe: Timeframe,
        start_date: datetime | None,
        end_date: datetime | None,
    ) -> pd.DataFrame:
        if timeframe == Timeframe.H4:
            df_1h = self._fetch_native(symbol, Timeframe.H1, start_date, end_date)
            return self._resampler.resample_to_higher_tf(df_1h, Timeframe.H4)

        if timeframe == Timeframe.W1:
            df_1d = self._fetch_native(symbol, Timeframe.D1, start_date, end_date)
            return self._resampler.resample_to_higher_tf(df_1d, Timeframe.W1)

        return self._fetch_native(symbol, timeframe, start_date, end_date)

    def fetch_options_chain(
        self,
        symbol: str,
        expiry_min_days: int = 0,
        expiry_max_days: int = 60,
    ) -> list[dict] | None:
        """Fetch live options chain snapshot from Alpaca.

        Requires an Alpaca Options Data subscription. Returns None if
        options data is unavailable (no subscription, market closed, import error).
        """
        if not _ALPACA_OPTIONS_AVAILABLE:
            logger.debug("[AlpacaAdapter] alpaca-py options client not available")
            return None

        try:
            client = OptionHistoricalDataClient(
                api_key=self._api_key,
                secret_key=self._secret_key,
            )
            request = OptionSnapshotRequest(underlying_symbols=[symbol])
            snapshots = client.get_option_snapshot(request)
        except Exception as exc:
            logger.warning(f"[AlpacaAdapter] options snapshot failed for {symbol}: {exc}")
            return None

        if not snapshots:
            return None

        from datetime import date as _date, timedelta

        today = _date.today()
        min_expiry = today + timedelta(days=expiry_min_days)
        max_expiry = today + timedelta(days=expiry_max_days)

        contracts: list[dict] = []
        for contract_symbol, snap in snapshots.items():
            try:
                greeks = snap.greeks
                quote = snap.latest_quote
                trade = snap.latest_trade

                # Parse expiry from contract symbol (e.g., "SPY240119C00480000")
                expiry_str = _parse_alpaca_expiry(contract_symbol)
                if expiry_str is None:
                    continue
                expiry_date = _date.fromisoformat(expiry_str)

                if not (min_expiry <= expiry_date <= max_expiry):
                    continue

                dte = (expiry_date - today).days
                option_type = "call" if "C" in contract_symbol[len(symbol):] else "put"
                strike = _parse_alpaca_strike(contract_symbol)

                bid = float(quote.bid_price) if quote and quote.bid_price else None
                ask = float(quote.ask_price) if quote and quote.ask_price else None
                mid = round((bid + ask) / 2, 2) if bid and ask else None
                last = float(trade.price) if trade and trade.price else None

                contracts.append({
                    "contract_id": contract_symbol,
                    "underlying": symbol,
                    "expiry": expiry_str,
                    "strike": strike,
                    "option_type": option_type,
                    "dte": dte,
                    "bid": bid,
                    "ask": ask,
                    "mid": mid,
                    "last": last,
                    "iv": float(snap.implied_volatility) if snap.implied_volatility else None,
                    "delta": float(greeks.delta) if greeks and greeks.delta else None,
                    "gamma": float(greeks.gamma) if greeks and greeks.gamma else None,
                    "theta": float(greeks.theta) if greeks and greeks.theta else None,
                    "vega": float(greeks.vega) if greeks and greeks.vega else None,
                    "open_interest": int(snap.open_interest) if snap.open_interest else None,
                    "volume": int(snap.day.volume) if snap.day and snap.day.volume else None,
                    "source": "alpaca",
                })
            except Exception as exc:
                logger.debug(f"[AlpacaAdapter] skipping contract {contract_symbol}: {exc}")
                continue

        logger.info(f"[AlpacaAdapter] fetched {len(contracts)} option contracts for {symbol}")
        return contracts if contracts else None

    def _fetch_native(
        self,
        symbol: str,
        timeframe: Timeframe,
        start_date: datetime | None,
        end_date: datetime | None,
    ) -> pd.DataFrame:
        from alpaca.data.requests import StockBarsRequest

        # Alpaca requires timezone-aware datetimes
        def _as_utc(dt: datetime | None) -> datetime | None:
            if dt is None:
                return None
            return dt if dt.tzinfo else dt.replace(tzinfo=UTC)

        from alpaca.data.enums import DataFeed

        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=_to_alpaca_tf(timeframe),
            start=_as_utc(start_date),
            end=_as_utc(end_date),
            feed=DataFeed.IEX,  # Free tier; SIP requires paid subscription
        )
        logger.debug(f"[Alpaca] Fetching {timeframe.value} bars for {symbol}")
        bar_set = self._client.get_stock_bars(request)
        return _bars_to_df(bar_set)
