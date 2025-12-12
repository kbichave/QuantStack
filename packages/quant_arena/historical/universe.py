# Copyright 2024 QuantArena Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Symbol universe mapping for historical simulation.

Maps logical symbol names (used in config and agent logic) to actual
ticker symbols for data fetching. This allows strategies to use intuitive
names like "WTI" and "BRENT" while fetching data from appropriate ETF proxies.

Symbol Mapping:
    SPY   -> SPY   (S&P 500 ETF)
    QQQ   -> QQQ   (Nasdaq 100 ETF)
    IWM   -> IWM   (Russell 2000 ETF)
    WTI   -> USO   (United States Oil Fund - WTI proxy)
    BRENT -> BNO   (United States Brent Oil Fund)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Dict, List, Optional, Set


class AssetClass(Enum):
    """Asset class classification."""

    EQUITY_INDEX = "equity_index"
    COMMODITY = "commodity"
    SECTOR = "sector"
    BOND = "bond"


@dataclass
class SymbolInfo:
    """
    Information about a tradable symbol.

    Attributes:
        logical_name: Human-readable name used in configs (e.g., "WTI")
        ticker: Actual ticker for data fetching (e.g., "USO")
        description: Full name of the instrument
        asset_class: Classification for exposure tracking
        inception_date: Earliest date data is available
        is_etf: Whether this is an ETF (for cost modeling)
    """

    logical_name: str
    ticker: str
    description: str
    asset_class: AssetClass
    inception_date: Optional[str] = None  # YYYY-MM-DD
    is_etf: bool = True

    def get_inception_date(self) -> Optional[date]:
        """Parse inception date to date object."""
        if self.inception_date is None:
            return None
        return date.fromisoformat(self.inception_date)


# Default symbol mappings for historical QuantArena
DEFAULT_SYMBOL_MAP: Dict[str, SymbolInfo] = {
    # US Equity Index ETFs
    "SPY": SymbolInfo(
        logical_name="SPY",
        ticker="SPY",
        description="SPDR S&P 500 ETF Trust",
        asset_class=AssetClass.EQUITY_INDEX,
        inception_date="1993-01-29",
        is_etf=True,
    ),
    "QQQ": SymbolInfo(
        logical_name="QQQ",
        ticker="QQQ",
        description="Invesco QQQ Trust (Nasdaq 100)",
        asset_class=AssetClass.EQUITY_INDEX,
        inception_date="1999-03-10",
        is_etf=True,
    ),
    "IWM": SymbolInfo(
        logical_name="IWM",
        ticker="IWM",
        description="iShares Russell 2000 ETF",
        asset_class=AssetClass.EQUITY_INDEX,
        inception_date="2000-05-22",
        is_etf=True,
    ),
    # Commodity ETF Proxies
    "WTI": SymbolInfo(
        logical_name="WTI",
        ticker="USO",
        description="United States Oil Fund (WTI Crude Proxy)",
        asset_class=AssetClass.COMMODITY,
        inception_date="2006-04-10",
        is_etf=True,
    ),
    "BRENT": SymbolInfo(
        logical_name="BRENT",
        ticker="BNO",
        description="United States Brent Oil Fund",
        asset_class=AssetClass.COMMODITY,
        inception_date="2010-06-02",
        is_etf=True,
    ),
    # Alternative mappings (can be used if USO/BNO not available)
    "USO": SymbolInfo(
        logical_name="USO",
        ticker="USO",
        description="United States Oil Fund",
        asset_class=AssetClass.COMMODITY,
        inception_date="2006-04-10",
        is_etf=True,
    ),
    "BNO": SymbolInfo(
        logical_name="BNO",
        ticker="BNO",
        description="United States Brent Oil Fund",
        asset_class=AssetClass.COMMODITY,
        inception_date="2010-06-02",
        is_etf=True,
    ),
}


class SymbolUniverse:
    """
    Manages the symbol universe for historical simulation.

    Responsibilities:
    - Map logical names to actual tickers
    - Track asset class exposures
    - Filter symbols by available date range
    - Provide universe metadata

    Usage:
        universe = SymbolUniverse(["SPY", "QQQ", "WTI", "BRENT"])

        # Get actual ticker for data fetching
        ticker = universe.get_ticker("WTI")  # Returns "USO"

        # Get all tickers for a date
        tickers = universe.get_available_tickers(date(2010, 1, 1))
    """

    def __init__(
        self,
        symbols: List[str],
        custom_mappings: Optional[Dict[str, SymbolInfo]] = None,
    ):
        """
        Initialize symbol universe.

        Args:
            symbols: List of logical symbol names to include
            custom_mappings: Optional custom symbol mappings (overrides defaults)
        """
        self._mappings = {**DEFAULT_SYMBOL_MAP}
        if custom_mappings:
            self._mappings.update(custom_mappings)

        # Validate requested symbols
        self._symbols: List[str] = []
        for sym in symbols:
            if sym in self._mappings:
                self._symbols.append(sym)
            else:
                raise ValueError(
                    f"Unknown symbol: {sym}. Available: {list(self._mappings.keys())}"
                )

        self._symbols = sorted(set(self._symbols))

    @property
    def symbols(self) -> List[str]:
        """Get list of logical symbol names."""
        return self._symbols.copy()

    @property
    def tickers(self) -> List[str]:
        """Get list of actual tickers for data fetching."""
        return sorted(set(self.get_ticker(s) for s in self._symbols))

    def get_ticker(self, logical_name: str) -> str:
        """
        Map logical name to actual ticker.

        Args:
            logical_name: Logical symbol name (e.g., "WTI")

        Returns:
            Actual ticker symbol (e.g., "USO")
        """
        if logical_name not in self._mappings:
            raise KeyError(f"Unknown symbol: {logical_name}")
        return self._mappings[logical_name].ticker

    def get_logical_name(self, ticker: str) -> Optional[str]:
        """
        Reverse lookup: get logical name from ticker.

        Args:
            ticker: Actual ticker symbol

        Returns:
            Logical name or None if not found
        """
        for name, info in self._mappings.items():
            if info.ticker == ticker and name in self._symbols:
                return name
        return None

    def get_info(self, logical_name: str) -> SymbolInfo:
        """Get full symbol information."""
        if logical_name not in self._mappings:
            raise KeyError(f"Unknown symbol: {logical_name}")
        return self._mappings[logical_name]

    def get_available_symbols(self, as_of_date: date) -> List[str]:
        """
        Get symbols that have data available as of a date.

        Args:
            as_of_date: Date to check availability

        Returns:
            List of logical symbol names with data available
        """
        available = []
        for sym in self._symbols:
            info = self._mappings[sym]
            inception = info.get_inception_date()
            if inception is None or as_of_date >= inception:
                available.append(sym)
        return available

    def get_available_tickers(self, as_of_date: date) -> List[str]:
        """Get actual tickers available as of a date."""
        available_symbols = self.get_available_symbols(as_of_date)
        return sorted(set(self.get_ticker(s) for s in available_symbols))

    def get_earliest_common_date(self) -> date:
        """
        Get the earliest date when all symbols have data.

        Returns:
            Date when all symbols in universe have data available
        """
        latest_inception = date(1900, 1, 1)
        for sym in self._symbols:
            info = self._mappings[sym]
            inception = info.get_inception_date()
            if inception and inception > latest_inception:
                latest_inception = inception
        return latest_inception

    def get_asset_class(self, logical_name: str) -> AssetClass:
        """Get asset class for a symbol."""
        return self._mappings[logical_name].asset_class

    def get_symbols_by_asset_class(self, asset_class: AssetClass) -> List[str]:
        """Get all symbols of a given asset class."""
        return [
            sym
            for sym in self._symbols
            if self._mappings[sym].asset_class == asset_class
        ]

    def ticker_to_logical_map(self) -> Dict[str, str]:
        """Get mapping from tickers to logical names."""
        return {self.get_ticker(s): s for s in self._symbols}

    def logical_to_ticker_map(self) -> Dict[str, str]:
        """Get mapping from logical names to tickers."""
        return {s: self.get_ticker(s) for s in self._symbols}

    def __len__(self) -> int:
        """Number of symbols in universe."""
        return len(self._symbols)

    def __contains__(self, symbol: str) -> bool:
        """Check if symbol is in universe."""
        return symbol in self._symbols

    def __iter__(self):
        """Iterate over logical symbol names."""
        return iter(self._symbols)

    def __repr__(self) -> str:
        return f"SymbolUniverse({self._symbols})"
