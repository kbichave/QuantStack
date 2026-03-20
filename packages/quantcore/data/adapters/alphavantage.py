"""
AlphaVantageAdapter — wraps the existing AlphaVantageClient.

Two roles:
1. OHLCV fallback (behind Alpaca) for price data
2. PRIMARY provider for:
   - Options chains with full Greeks + IV ($49.99/mo premium)
   - Economic indicators (CPI, Fed Funds, NFP, GDP, Treasury Yields)
   - Earnings calendar + estimates
   - News sentiment
   - Company fundamentals

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

    # ── Options data ───────────────────────────────────────────────────────

    def fetch_options_chain(
        self,
        symbol: str,
        expiry_min_days: int = 0,
        expiry_max_days: int = 60,
        as_of_date: str | None = None,
    ) -> list[dict] | None:
        """
        Fetch end-of-day options chain with full Greeks from Alpha Vantage.

        Uses HISTORICAL_OPTIONS endpoint which returns full Greeks (delta, gamma,
        theta, vega, rho) + IV + OI. Data available back to 2020+.

        Args:
            symbol: Underlying ticker.
            expiry_min_days: Min DTE to include (default 0).
            expiry_max_days: Max DTE to include (default 60).
            as_of_date: Fetch chain snapshot for a specific date (YYYY-MM-DD).
                        None = latest available (most recent trading day).

        Returns list of contract dicts compatible with the options_flow collector,
        or None if data is unavailable.
        """
        from datetime import date as _date, timedelta

        try:
            df = self._client.fetch_historical_options(
                symbol, date=as_of_date,
            )

            if df is None or df.empty:
                # Fallback to realtime endpoint
                try:
                    df = self._client.fetch_realtime_options(symbol, require_greeks=True)
                except Exception:
                    pass

            if df is None or df.empty:
                return None

            today = _date.today()
            min_expiry = today + timedelta(days=expiry_min_days)
            max_expiry = today + timedelta(days=expiry_max_days)

            contracts: list[dict] = []
            for _, row in df.iterrows():
                # Client normalizes: "expiration" → "expiry", "type" → "option_type"
                expiry_str = str(row.get("expiry", row.get("expiration", "")))
                if not expiry_str or expiry_str == "nan":
                    continue

                try:
                    expiry_date = _date.fromisoformat(expiry_str[:10])
                except ValueError:
                    continue

                if not (min_expiry <= expiry_date <= max_expiry):
                    continue

                dte = (expiry_date - today).days
                bid = _safe_float(row.get("bid"))
                ask = _safe_float(row.get("ask"))
                mid = round((bid + ask) / 2, 4) if bid is not None and ask is not None else None

                contracts.append({
                    "contract_id": row.get("contract_id", row.get("contractID", "")),
                    "underlying": symbol,
                    "expiry": expiry_str[:10],
                    "strike": _safe_float(row.get("strike")),
                    "option_type": str(row.get("option_type", row.get("type", ""))).lower(),
                    "dte": dte,
                    "bid": bid,
                    "ask": ask,
                    "mid": mid,
                    "last": _safe_float(row.get("last")),
                    "iv": _safe_float(row.get("iv", row.get("implied_volatility"))),
                    "delta": _safe_float(row.get("delta")),
                    "gamma": _safe_float(row.get("gamma")),
                    "theta": _safe_float(row.get("theta")),
                    "vega": _safe_float(row.get("vega")),
                    "rho": _safe_float(row.get("rho")),
                    "open_interest": _safe_int(row.get("open_interest")),
                    "volume": _safe_int(row.get("volume")),
                    "source": "alpha_vantage",
                })

            logger.info(f"[AV] Fetched {len(contracts)} option contracts for {symbol}")
            return contracts if contracts else None

        except Exception as exc:
            logger.warning(f"[AV] Options chain failed for {symbol}: {exc}")
            return None

    def fetch_historical_options_chain(
        self,
        symbol: str,
        date_str: str | None = None,
    ) -> list[dict] | None:
        """
        Fetch historical options chain snapshot for a specific date.

        Uses Alpha Vantage HISTORICAL_OPTIONS endpoint (15yr history).
        Returns same format as fetch_options_chain.
        """
        try:
            df = self._client.fetch_historical_options(
                symbol, date=date_str, require_greeks=True,
            )
            if df is None or df.empty:
                return None

            contracts = []
            for _, row in df.iterrows():
                contracts.append({
                    "underlying": symbol,
                    "expiry": str(row.get("expiration", ""))[:10],
                    "strike": _safe_float(row.get("strike")),
                    "option_type": str(row.get("type", "")).lower(),
                    "bid": _safe_float(row.get("bid")),
                    "ask": _safe_float(row.get("ask")),
                    "last": _safe_float(row.get("last")),
                    "iv": _safe_float(row.get("implied_volatility")),
                    "delta": _safe_float(row.get("delta")),
                    "gamma": _safe_float(row.get("gamma")),
                    "theta": _safe_float(row.get("theta")),
                    "vega": _safe_float(row.get("vega")),
                    "open_interest": _safe_int(row.get("open_interest")),
                    "volume": _safe_int(row.get("volume")),
                    "source": "alpha_vantage_historical",
                })

            return contracts if contracts else None

        except Exception as exc:
            logger.warning(f"[AV] Historical options failed for {symbol}: {exc}")
            return None

    # ── Economic indicators ────────────────────────────────────────────────

    def fetch_economic(
        self,
        indicator: str,
        interval: str = "monthly",
    ) -> pd.DataFrame:
        """
        Fetch economic indicator time series.

        Supported indicators:
            CPI, FEDERAL_FUNDS_RATE, REAL_GDP, TREASURY_YIELD,
            UNEMPLOYMENT, NONFARM_PAYROLL, INFLATION, RETAIL_SALES,
            DURABLES

        Returns DataFrame with DatetimeIndex and 'value' column.
        """
        return self._client.fetch_economic_indicator(
            function=indicator, interval=interval,
        )

    def fetch_fed_funds_rate(self) -> pd.DataFrame:
        """Fetch Federal Funds Rate history (monthly)."""
        return self.fetch_economic("FEDERAL_FUNDS_RATE", interval="monthly")

    def fetch_cpi(self) -> pd.DataFrame:
        """Fetch CPI history (monthly)."""
        return self.fetch_economic("CPI", interval="monthly")

    def fetch_unemployment(self) -> pd.DataFrame:
        """Fetch unemployment rate history (monthly)."""
        return self.fetch_economic("UNEMPLOYMENT", interval="monthly")

    def fetch_treasury_yield(self, maturity: str = "10year") -> pd.DataFrame:
        """Fetch Treasury yield (daily). Maturity: 3month, 2year, 5year, 10year, 30year."""
        return self._client.fetch_economic_indicator(
            function="TREASURY_YIELD", interval="daily", maturity=maturity,
        )

    # ── Earnings ───────────────────────────────────────────────────────────

    def fetch_earnings(
        self,
        symbol: str | None = None,
        horizon: str = "3month",
    ) -> pd.DataFrame:
        """
        Fetch earnings calendar with dates and EPS estimates.

        Returns DataFrame with: symbol, report_date, estimate, fiscal_date_ending.
        """
        return self._client.fetch_earnings_calendar(symbol=symbol, horizon=horizon)

    # ── News sentiment ─────────────────────────────────────────────────────

    def fetch_news(
        self,
        tickers: str | None = None,
        topics: str | None = None,
        limit: int = 50,
    ) -> list[dict]:
        """
        Fetch market news with sentiment scores.

        Args:
            tickers: Comma-separated symbols (e.g., "AAPL,MSFT")
            topics: Topics filter (e.g., "earnings", "ipo", "technology")
            limit: Max articles to return

        Returns list of article dicts with sentiment data.
        """
        return self._client.fetch_news_sentiment(
            tickers=tickers, topics=topics, limit=limit,
        )

    # ── Company fundamentals ───────────────────────────────────────────────

    def fetch_fundamentals(self, symbol: str) -> dict:
        """
        Fetch company overview (P/E, EPS, market cap, sector, etc.).

        Returns dict with ~60 fundamental fields.
        """
        return self._client.fetch_company_overview(symbol)


# ── Helpers ────────────────────────────────────────────────────────────────

def _safe_float(val) -> float | None:
    if val is None:
        return None
    try:
        f = float(val)
        return None if f != f else f  # NaN guard
    except (TypeError, ValueError):
        return None


def _safe_int(val) -> int | None:
    if val is None:
        return None
    try:
        return int(float(val))
    except (TypeError, ValueError):
        return None
