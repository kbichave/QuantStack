"""
Earnings Calendar Manager.

Provides utilities to:
- Track upcoming earnings dates
- Calculate days to earnings for trading decisions
- Block trades during earnings blackout periods
"""

from dataclasses import dataclass
from datetime import date, datetime, timedelta

import pandas as pd
from loguru import logger


@dataclass
class EarningsEvent:
    """Single earnings event."""

    symbol: str
    report_date: date
    fiscal_date_ending: str | None = None
    estimate: float | None = None
    name: str | None = None

    @property
    def days_until(self) -> int:
        """Days until earnings."""
        return (self.report_date - date.today()).days

    @property
    def is_upcoming(self) -> bool:
        """Check if earnings is in the future."""
        return self.days_until >= 0

    @property
    def is_this_week(self) -> bool:
        """Check if earnings is within this week."""
        return 0 <= self.days_until <= 7


class EarningsManager:
    """
    Manages earnings calendar data for trading decisions.

    Features:
    - Track upcoming earnings for all symbols
    - Calculate days to earnings
    - Check if trading is blocked due to earnings blackout
    """

    def __init__(
        self,
        default_blackout_days: int = 5,
    ):
        """
        Initialize earnings manager.

        Args:
            default_blackout_days: Default days before earnings to block trades
        """
        self.default_blackout_days = default_blackout_days
        self._earnings_cache: dict[str, list[EarningsEvent]] = {}
        self._last_refresh: datetime | None = None

    def load_from_dataframe(self, df: pd.DataFrame) -> None:
        """
        Load earnings data from DataFrame.

        Args:
            df: DataFrame with columns: symbol, report_date, name, estimate
        """
        self._earnings_cache.clear()

        if df.empty:
            logger.warning("Empty earnings DataFrame provided")
            return

        # Parse dates
        if "report_date" in df.columns:
            df = df.copy()
            df["report_date"] = pd.to_datetime(df["report_date"]).dt.date
        elif "reportDate" in df.columns:
            df = df.copy()
            df["report_date"] = pd.to_datetime(df["reportDate"]).dt.date
        else:
            logger.error("No report_date column found in earnings data")
            return

        # Group by symbol
        for symbol in df["symbol"].unique():
            symbol_data = df[df["symbol"] == symbol].sort_values("report_date")
            events = []

            for _, row in symbol_data.iterrows():
                events.append(
                    EarningsEvent(
                        symbol=symbol,
                        report_date=row["report_date"],
                        fiscal_date_ending=row.get("fiscal_date_ending")
                        or row.get("fiscalDateEnding"),
                        estimate=row.get("estimate"),
                        name=row.get("name"),
                    )
                )

            self._earnings_cache[symbol] = events

        self._last_refresh = datetime.now()
        logger.info(f"Loaded earnings for {len(self._earnings_cache)} symbols")

    def get_days_to_earnings(
        self,
        symbol: str,
        as_of_date: date | None = None,
    ) -> int | None:
        """
        Get days until next earnings for a symbol.

        Args:
            symbol: Ticker symbol
            as_of_date: Date to calculate from (default: today)

        Returns:
            Days until next earnings, or None if unknown
        """
        if as_of_date is None:
            as_of_date = date.today()

        events = self._earnings_cache.get(symbol, [])

        for event in events:
            if event.report_date >= as_of_date:
                return (event.report_date - as_of_date).days

        return None  # No upcoming earnings found

    def get_next_earnings(
        self,
        symbol: str,
        as_of_date: date | None = None,
    ) -> EarningsEvent | None:
        """
        Get next earnings event for a symbol.

        Args:
            symbol: Ticker symbol
            as_of_date: Date to calculate from (default: today)

        Returns:
            Next EarningsEvent or None
        """
        if as_of_date is None:
            as_of_date = date.today()

        events = self._earnings_cache.get(symbol, [])

        for event in events:
            if event.report_date >= as_of_date:
                return event

        return None

    def is_in_blackout(
        self,
        symbol: str,
        blackout_days: int | None = None,
        as_of_date: date | None = None,
    ) -> bool:
        """
        Check if a symbol is in earnings blackout period.

        Args:
            symbol: Ticker symbol
            blackout_days: Days before earnings to block (uses default if None)
            as_of_date: Date to check (default: today)

        Returns:
            True if in blackout period
        """
        if blackout_days is None:
            blackout_days = self.default_blackout_days

        days_to_earnings = self.get_days_to_earnings(symbol, as_of_date)

        if days_to_earnings is None:
            return False  # No earnings data, allow trading

        return days_to_earnings <= blackout_days

    def get_blackout_symbols(
        self,
        symbols: list[str],
        blackout_days: int | None = None,
        as_of_date: date | None = None,
    ) -> list[str]:
        """
        Get list of symbols currently in blackout period.

        Args:
            symbols: List of symbols to check
            blackout_days: Days before earnings to block
            as_of_date: Date to check

        Returns:
            List of symbols in blackout
        """
        return [
            symbol
            for symbol in symbols
            if self.is_in_blackout(symbol, blackout_days, as_of_date)
        ]

    def get_earnings_summary(
        self,
        symbols: list[str],
        as_of_date: date | None = None,
    ) -> pd.DataFrame:
        """
        Get summary of earnings dates for multiple symbols.

        Args:
            symbols: List of symbols
            as_of_date: Date to calculate from

        Returns:
            DataFrame with symbol, days_to_earnings, report_date, in_blackout
        """
        if as_of_date is None:
            as_of_date = date.today()

        data = []
        for symbol in symbols:
            event = self.get_next_earnings(symbol, as_of_date)
            days = self.get_days_to_earnings(symbol, as_of_date)

            data.append(
                {
                    "symbol": symbol,
                    "days_to_earnings": days,
                    "report_date": event.report_date if event else None,
                    "in_blackout": self.is_in_blackout(symbol, as_of_date=as_of_date),
                }
            )

        return pd.DataFrame(data)

    def get_all_upcoming(
        self,
        days_ahead: int = 30,
    ) -> list[EarningsEvent]:
        """
        Get all upcoming earnings within N days.

        Args:
            days_ahead: Days to look ahead

        Returns:
            List of upcoming earnings events, sorted by date
        """
        cutoff = date.today() + timedelta(days=days_ahead)
        today = date.today()

        events = []
        for _symbol, symbol_events in self._earnings_cache.items():
            for event in symbol_events:
                if today <= event.report_date <= cutoff:
                    events.append(event)

        return sorted(events, key=lambda e: e.report_date)

    @property
    def symbols_with_earnings(self) -> list[str]:
        """Get list of symbols with earnings data."""
        return list(self._earnings_cache.keys())

    def __contains__(self, symbol: str) -> bool:
        """Check if symbol has earnings data."""
        return symbol in self._earnings_cache


# Global instance for convenience
_earnings_manager: EarningsManager | None = None


def get_earnings_manager() -> EarningsManager:
    """Get global earnings manager instance."""
    global _earnings_manager
    if _earnings_manager is None:
        _earnings_manager = EarningsManager()
    return _earnings_manager


def load_earnings_from_store(data_store) -> EarningsManager:
    """
    Load earnings data from data store into earnings manager.

    Args:
        data_store: DataStore instance

    Returns:
        EarningsManager with loaded data
    """
    manager = get_earnings_manager()

    # Load from PostgreSQL
    df = data_store.load_earnings_calendar()
    if not df.empty:
        manager.load_from_dataframe(df)

    return manager
