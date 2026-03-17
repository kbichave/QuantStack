# Copyright 2024 QuantCore Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Base classes for asset class abstraction and data adapters.

This module provides the foundation for multi-asset class support in QuantCore.
Each asset class (equity, commodity futures, FX, etc.) implements the
`AssetClassAdapter` interface to provide standardized data access.

Example
-------
>>> from quantcore.data.base import AssetClass, AssetClassAdapter
>>>
>>> class CryptoAdapter(AssetClassAdapter):
...     @property
...     def asset_class(self):
...         return AssetClass.CRYPTO
...
...     def fetch_ohlcv(self, symbol, timeframe, start_date=None, end_date=None):
...         # Implement crypto-specific data fetching
...         pass
...
...     def get_available_symbols(self):
...         return ["BTC", "ETH", "SOL"]
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum

import pandas as pd

from quantcore.config.timeframes import Timeframe
from quantcore.data.provider_enum import DataProvider


class AssetClass(Enum):
    """
    Supported asset classes.

    Each asset class has different characteristics for data handling,
    feature engineering, and execution modeling.

    Attributes
    ----------
    EQUITY : str
        Stocks and ETFs. Includes corporate actions handling.
    COMMODITY_FUTURES : str
        Commodity futures contracts. Includes roll handling.
    FX : str
        Foreign exchange pairs. 24-hour market.
    FIXED_INCOME : str
        Bonds and interest rate products.
    CRYPTO : str
        Cryptocurrency markets. 24/7 trading.

    Examples
    --------
    >>> from quantcore.data.base import AssetClass
    >>> asset = AssetClass.EQUITY
    >>> asset.value
    'EQUITY'
    """

    EQUITY = "EQUITY"
    COMMODITY_FUTURES = "COMMODITY_FUTURES"
    FX = "FX"
    FIXED_INCOME = "FIXED_INCOME"
    CRYPTO = "CRYPTO"


class AssetClassAdapter(ABC):
    """
    Base class for asset-class-specific data adapters.

    Each asset class (equity, commodity, etc.) implements this interface
    to provide asset-class-specific data fetching, feature engineering,
    and execution models.

    This abstraction allows the rest of QuantCore to work with different
    asset classes through a unified interface.

    Methods
    -------
    fetch_ohlcv(symbol, timeframe, start_date, end_date)
        Fetch OHLCV data for a symbol.
    get_available_symbols()
        List available symbols for this asset class.
    get_execution_model()
        Get the execution cost model.
    get_regime_detector()
        Get the regime detection model.
    get_feature_factory()
        Get the feature factory.
    validate_symbol(symbol)
        Check if a symbol is valid.

    Examples
    --------
    >>> from quantcore.data.base import AssetClassAdapter, AssetClass
    >>> from quantcore.config.timeframes import Timeframe
    >>>
    >>> class MyEquityAdapter(AssetClassAdapter):
    ...     @property
    ...     def asset_class(self):
    ...         return AssetClass.EQUITY
    ...
    ...     def fetch_ohlcv(self, symbol, timeframe, start_date=None, end_date=None):
    ...         # Fetch from your data source
    ...         import pandas as pd
    ...         return pd.DataFrame()
    ...
    ...     def get_available_symbols(self):
    ...         return ["AAPL", "MSFT", "GOOGL"]
    >>>
    >>> adapter = MyEquityAdapter()
    >>> adapter.validate_symbol("AAPL")
    True

    Notes
    -----
    Subclasses must implement:

    - `asset_class` property
    - `fetch_ohlcv()` method
    - `get_available_symbols()` method

    Optional methods (`get_execution_model`, `get_regime_detector`,
    `get_feature_factory`) raise NotImplementedError by default.

    See Also
    --------
    quantcore.data.fetcher.AlphaVantageFetcher : Concrete implementation for equities.
    quantcore.data.manager.DataManager : High-level data management.
    """

    @property
    @abstractmethod
    def asset_class(self) -> AssetClass:
        """
        Asset class this adapter handles.

        Returns
        -------
        AssetClass
            The asset class enum value.
        """
        pass

    @property
    @abstractmethod
    def provider(self) -> DataProvider:
        """Which external data source this adapter wraps.

        Used by :class:`DataProviderRegistry` as a stable key for
        routing and fallback ordering.  Must be unique per adapter class.
        """
        pass

    @abstractmethod
    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: Timeframe,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for symbol.

        Parameters
        ----------
        symbol : str
            Symbol to fetch (e.g., "AAPL", "CL=F", "EURUSD").
        timeframe : Timeframe
            Data timeframe (e.g., Timeframe.DAILY, Timeframe.HOURLY).
        start_date : datetime, optional
            Start date filter. If None, fetches all available history.
        end_date : datetime, optional
            End date filter. If None, fetches up to current date.

        Returns
        -------
        pd.DataFrame
            DataFrame with OHLCV data and DatetimeIndex.

            Required columns:
            - open : float
            - high : float
            - low : float
            - close : float
            - volume : float

            Optional columns (asset-class specific):
            - adjusted_close : float (equities)
            - open_interest : float (futures)
            - vwap : float

        Raises
        ------
        ValueError
            If symbol is not valid for this asset class.
        ConnectionError
            If data source is unreachable.

        Examples
        --------
        >>> adapter = MyEquityAdapter()
        >>> df = adapter.fetch_ohlcv(
        ...     "AAPL",
        ...     Timeframe.DAILY,
        ...     start_date=datetime(2023, 1, 1),
        ...     end_date=datetime(2023, 12, 31)
        ... )
        >>> df.columns.tolist()
        ['open', 'high', 'low', 'close', 'volume']
        """
        pass

    @abstractmethod
    def get_available_symbols(self) -> list[str]:
        """
        Get list of available symbols for this asset class.

        Returns
        -------
        List[str]
            List of symbol strings that can be fetched.

        Examples
        --------
        >>> adapter = MyEquityAdapter()
        >>> symbols = adapter.get_available_symbols()
        >>> "AAPL" in symbols
        True
        """
        pass

    def get_execution_model(self):
        """
        Get execution cost model for this asset class.

        Returns
        -------
        ExecutionModel
            Execution cost model instance with asset-class-specific
            parameters (spread, slippage, market impact).

        Raises
        ------
        NotImplementedError
            If not implemented by subclass.

        Notes
        -----
        Different asset classes have different execution characteristics:

        - Equities: Typically lower spreads, discrete tick sizes
        - Futures: Roll costs, margin requirements
        - FX: Very tight spreads, 24-hour liquidity
        """
        raise NotImplementedError("Subclasses must implement get_execution_model")

    def get_regime_detector(self):
        """
        Get regime detector for this asset class.

        Returns
        -------
        RegimeDetector
            Regime detection model tuned for this asset class.

        Raises
        ------
        NotImplementedError
            If not implemented by subclass.

        Notes
        -----
        Regime detection is asset-class specific. For example:

        - Equities: Bull/bear/sideways markets
        - Commodities: Contango/backwardation, seasonal patterns
        - FX: Risk-on/risk-off environments
        """
        raise NotImplementedError("Subclasses must implement get_regime_detector")

    def get_feature_factory(self):
        """
        Get feature factory for this asset class.

        Returns
        -------
        FeatureFactory
            Feature factory with asset-class-specific features.

        Raises
        ------
        NotImplementedError
            If not implemented by subclass.

        Notes
        -----
        Each asset class may have unique features:

        - Equities: Fundamental ratios, sector rotation
        - Commodities: Term structure, seasonality
        - FX: Interest rate differentials, carry
        """
        raise NotImplementedError("Subclasses must implement get_feature_factory")

    def fetch_options_chain(
        self,
        symbol: str,
        expiry_min_days: int = 0,
        expiry_max_days: int = 60,
    ) -> list[dict] | None:
        """Fetch live options chain for a symbol.

        Parameters
        ----------
        symbol : str
            Underlying equity symbol (e.g., "SPY").
        expiry_min_days : int
            Minimum days to expiration to include (inclusive).
        expiry_max_days : int
            Maximum days to expiration to include (inclusive).

        Returns
        -------
        list[dict] or None
            List of contract dicts with keys:
            contract_id, underlying, expiry, strike, option_type,
            bid, ask, mid, iv, delta, gamma, theta, vega,
            open_interest, volume, dte.
            Returns None if this provider does not support options data.
        """
        return None  # default: provider does not support options

    def validate_symbol(self, symbol: str) -> bool:
        """
        Validate that a symbol is valid for this asset class.

        Parameters
        ----------
        symbol : str
            Symbol to validate.

        Returns
        -------
        bool
            True if symbol is valid and can be fetched.

        Examples
        --------
        >>> adapter = MyEquityAdapter()
        >>> adapter.validate_symbol("AAPL")
        True
        >>> adapter.validate_symbol("INVALID")
        False
        """
        return symbol in self.get_available_symbols()
