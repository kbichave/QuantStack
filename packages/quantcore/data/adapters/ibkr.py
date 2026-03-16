"""
IBKRDataAdapter — historical OHLCV via IB Gateway reqHistoricalData.

Uses the ``IBKRConnectionManager`` singleton so the same gateway socket is
shared with ``IBKRStreamingAdapter`` and ``IBKRBrokerClient`` without opening
multiple TCP connections.

Supported timeframes
--------------------
S5  → "5 secs"
M1  → "1 min"
M5  → "5 mins"
M15 → "15 mins"
M30 → "30 mins"
H1  → "1 hour"
H4  → "4 hours"
D1  → "1 day"
W1  → "1 week"

Requires: ib_insync>=0.9.86  (uv pip install -e ".[ibkr]")
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional

import pandas as pd
from loguru import logger

from quantcore.config.timeframes import Timeframe
from quantcore.data.base import AssetClass, AssetClassAdapter
from quantcore.data.provider_enum import DataProvider

try:
    import ib_insync as ib
    _IB_AVAILABLE = True
except ImportError:
    _IB_AVAILABLE = False


def _require_ibkr() -> None:
    if not _IB_AVAILABLE:
        raise ImportError(
            "ib_insync is required for IBKRDataAdapter. "
            "Run: uv pip install -e '.[ibkr]'"
        )


# IB barSizeSetting strings per timeframe
_BAR_SIZE: dict = {
    Timeframe.S5:  "5 secs",
    Timeframe.M1:  "1 min",
    Timeframe.M5:  "5 mins",
    Timeframe.M15: "15 mins",
    Timeframe.M30: "30 mins",
    Timeframe.H1:  "1 hour",
    Timeframe.H4:  "4 hours",
    Timeframe.D1:  "1 day",
    Timeframe.W1:  "1 week",
}


class IBKRDataAdapter(AssetClassAdapter):
    """Historical OHLCV adapter for Interactive Brokers.

    Args:
        host:      IB Gateway host (default 127.0.0.1).
        port:      Gateway port (4001 = IB Gateway, 7497 = TWS).
        client_id: Unique client ID; default 1.  Must differ from
                   IBKRStreamingAdapter (2) and IBKRBrokerClient (3).
        timeout:   reqHistoricalData timeout in seconds.
    """

    def __init__(
        self,
        host:      str = "127.0.0.1",
        port:      int = 4001,
        client_id: int = 1,
        timeout:   int = 30,
    ) -> None:
        _require_ibkr()
        self._host      = host
        self._port      = port
        self._client_id = client_id
        self._timeout   = timeout
        self._mgr       = None   # lazy: created on first fetch

    # ── AssetClassAdapter identity ────────────────────────────────────────────

    @property
    def provider(self) -> DataProvider:
        return DataProvider.IBKR

    @property
    def asset_class(self) -> List[AssetClass]:
        return [
            AssetClass.EQUITY,
            AssetClass.COMMODITY_FUTURES,
            AssetClass.FX,
            AssetClass.FIXED_INCOME,
        ]

    def get_available_symbols(self) -> List[str]:
        # IB has a massive universe; return a representative sample only.
        return ["SPY", "QQQ", "AAPL", "MSFT", "GC=F", "CL=F", "EUR/USD"]

    # ── Data fetching ─────────────────────────────────────────────────────────

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: Timeframe,
        start_date: Optional[datetime] = None,
        end_date:   Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Fetch historical bars from IB Gateway.

        Returns a DatetimeIndex (UTC, named "timestamp") DataFrame with columns
        [open, high, low, close, volume] as float64, sorted ascending.
        """
        if timeframe not in _BAR_SIZE:
            raise ValueError(
                f"IBKRDataAdapter does not support {timeframe}. "
                f"Supported: {list(_BAR_SIZE.keys())}"
            )

        bar_size = _BAR_SIZE[timeframe]
        duration = self._duration_str(timeframe, start_date, end_date)
        end_str  = _format_end_dt(end_date)

        ib_conn = self._get_connection()
        contract = self._resolve_contract(ib_conn, symbol)

        bars = ib_conn.reqHistoricalData(
            contract,
            endDateTime    = end_str,
            durationStr    = duration,
            barSizeSetting = bar_size,
            whatToShow     = "TRADES",
            useRTH         = True,
            formatDate     = 1,
            timeout        = self._timeout,
        )

        if not bars:
            logger.warning(f"[IBKR] No historical data for {symbol} {timeframe}")
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        df = pd.DataFrame([
            {
                "timestamp": _parse_bar_date(b.date),
                "open":      float(b.open),
                "high":      float(b.high),
                "low":       float(b.low),
                "close":     float(b.close),
                "volume":    float(b.volume),
            }
            for b in bars
        ])

        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp").sort_index()

        # Trim to requested range
        if start_date:
            sd = pd.Timestamp(start_date, tz="UTC") if start_date.tzinfo is None else pd.Timestamp(start_date)
            df = df[df.index >= sd]
        if end_date:
            ed = pd.Timestamp(end_date, tz="UTC") if end_date.tzinfo is None else pd.Timestamp(end_date)
            df = df[df.index <= ed]

        logger.info(f"[IBKR] Fetched {len(df)} {timeframe.value} bars for {symbol}")
        return df

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_connection(self) -> "ib.IB":
        """Return live IB connection via IBKRConnectionManager."""
        if self._mgr is None:
            try:
                from ibkr_mcp.connection import IBKRConnectionManager
                self._mgr = IBKRConnectionManager.get_instance(
                    host      = self._host,
                    port      = self._port,
                    client_id = self._client_id,
                    timeout   = self._timeout,
                )
            except ImportError:
                # ibkr_mcp not installed — create a standalone connection
                logger.debug("[IBKR] ibkr_mcp not found; using standalone IB connection")
                from quantcore.execution.broker import BrokerConnectionError
                _ib = ib.IB()
                try:
                    _ib.connect(
                        self._host, self._port,
                        clientId=self._client_id,
                        timeout=self._timeout,
                    )
                except Exception as exc:
                    raise BrokerConnectionError(
                        f"IB Gateway not reachable at {self._host}:{self._port}"
                    ) from exc
                return _ib
        return self._mgr.ib

    def _resolve_contract(self, ib_conn: "ib.IB", symbol: str) -> "ib.Contract":
        """Qualify a US equity contract by symbol."""
        contract   = ib.Stock(symbol, "SMART", "USD")
        qualified  = ib_conn.qualifyContracts(contract)
        return qualified[0] if qualified else contract

    @staticmethod
    def _duration_str(
        timeframe: Timeframe,
        start_date: Optional[datetime],
        end_date:   Optional[datetime],
    ) -> str:
        """Convert date range to IB durationStr (e.g. "30 D", "3 M")."""
        if start_date and end_date:
            days = (end_date - start_date).days + 1
        elif start_date:
            from datetime import datetime as _dt
            days = (_dt.now(timezone.utc) - start_date.replace(tzinfo=timezone.utc)
                    if start_date.tzinfo is None else
                    _dt.now(timezone.utc) - start_date).days + 1
        else:
            # Default look-back per timeframe
            defaults = {
                Timeframe.S5: 1, Timeframe.M1: 7, Timeframe.M5: 30,
                Timeframe.M15: 60, Timeframe.M30: 90, Timeframe.H1: 180,
                Timeframe.H4: 365, Timeframe.D1: 365 * 5, Timeframe.W1: 365 * 10,
            }
            days = defaults.get(timeframe, 365)

        if days <= 365:
            return f"{days} D"
        years = max(1, days // 365)
        return f"{years} Y"


def _format_end_dt(end_date: Optional[datetime]) -> str:
    """Format end datetime for IB endDateTime parameter ("" = now)."""
    if end_date is None:
        return ""
    if end_date.tzinfo is None:
        end_date = end_date.replace(tzinfo=timezone.utc)
    return end_date.strftime("%Y%m%d %H:%M:%S UTC")


def _parse_bar_date(date_val) -> datetime:
    """Parse ib_insync bar.date which is either a datetime or a date string."""
    if isinstance(date_val, datetime):
        return date_val if date_val.tzinfo else date_val.replace(tzinfo=timezone.utc)
    # String "YYYYMMDD" for daily/weekly bars
    if isinstance(date_val, str):
        if len(date_val) == 8:
            return datetime.strptime(date_val, "%Y%m%d").replace(tzinfo=timezone.utc)
        return datetime.fromisoformat(date_val).replace(tzinfo=timezone.utc)
    return datetime.fromtimestamp(float(date_val), tz=timezone.utc)
