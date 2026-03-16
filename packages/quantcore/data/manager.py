"""
Unified data manager for multi-asset class support.

Manages data fetching across multiple asset classes through the
DataProviderRegistry, which supports priority-ordered provider fallback.
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime
import pandas as pd
from loguru import logger

from quantcore.data.base import AssetClass, AssetClassAdapter
from quantcore.data.registry import DataProviderRegistry
from quantcore.data.storage import DataStore
from quantcore.config.timeframes import Timeframe


class UnifiedDataManager:
    """
    Manages data across multiple asset classes.

    Uses a DataProviderRegistry for priority-ordered, fallback-aware
    data fetching.  Multiple providers can be registered per asset class;
    the registry tries them in order and returns the first non-empty result.

    Args:
        registry: Pre-built DataProviderRegistry.  If None, an empty registry
                  is created (adapters can be added via register_adapter).
        db_path:  Path to DuckDB storage.  Falls back to settings if None.
    """

    def __init__(
        self,
        registry: Optional[DataProviderRegistry] = None,
        db_path: Optional[str] = None,
    ):
        self.registry = registry or DataProviderRegistry()
        self.storage  = DataStore(db_path)

    # ── Backward-compat shim ──────────────────────────────────────────────────

    @property
    def adapters(self) -> Dict[AssetClass, AssetClassAdapter]:
        """Primary adapter for each registered asset class."""
        return {
            ac: self.registry.get_primary(ac)
            for ac, bucket in self.registry._adapters.items()
            if bucket
        }

    def register_adapter(self, adapter: AssetClassAdapter) -> None:
        """Register an adapter (appends to registry; does not replace existing ones).

        This shim preserves the old ``manager.register_adapter(adapter)`` call
        pattern.  New code should call ``self.registry.register(adapter)`` directly.
        """
        # Append at back so explicitly-registered adapters don't displace
        # the already-configured primary provider from from_settings().
        self.registry.register(adapter, priority=99)

    def get_adapter(self, asset_class: AssetClass) -> AssetClassAdapter:
        """Return the primary (highest-priority) adapter for the asset class."""
        return self.registry.get_primary(asset_class)

    def fetch_ohlcv(
        self,
        symbol: str,
        asset_class: AssetClass,
        timeframe: Timeframe,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for a symbol.

        Args:
            symbol: Symbol to fetch
            asset_class: Asset class of symbol
            timeframe: Data timeframe
            start_date: Optional start date filter
            end_date: Optional end date filter
            
        Returns:
            DataFrame with OHLCV data
        """
        return self.registry.fetch_ohlcv(symbol, asset_class, timeframe, start_date, end_date)
    
    def fetch_universe(
        self,
        symbols: List[Tuple[str, AssetClass]],
        timeframe: Timeframe,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Dict[Tuple[str, AssetClass], pd.DataFrame]:
        """
        Fetch data for multiple symbols across asset classes.
        
        Args:
            symbols: List of (symbol, asset_class) tuples
            timeframe: Data timeframe
            start_date: Optional start date filter
            end_date: Optional end date filter
            
        Returns:
            Dictionary mapping (symbol, asset_class) to DataFrame
        """
        results = {}
        
        for symbol, asset_class in symbols:
            try:
                data = self.fetch_ohlcv(
                    symbol=symbol,
                    asset_class=asset_class,
                    timeframe=timeframe,
                    start_date=start_date,
                    end_date=end_date,
                )
                results[(symbol, asset_class)] = data
                logger.debug(f"Fetched {symbol} ({asset_class.value}): {len(data)} bars")
            except Exception as e:
                logger.error(f"Failed to fetch {symbol} ({asset_class.value}): {e}")
                results[(symbol, asset_class)] = pd.DataFrame()
        
        return results
    
    def get_all_symbols(self) -> Dict[AssetClass, List[str]]:
        """
        Get all available symbols across all asset classes.
        
        Returns:
            Dictionary mapping asset class to list of symbols
        """
        result = {}
        for asset_class, adapter in self.adapters.items():
            result[asset_class] = adapter.get_available_symbols()
        return result
    
    def validate_universe(
        self,
        symbols: List[Tuple[str, AssetClass]],
    ) -> List[Tuple[str, AssetClass, bool]]:
        """
        Validate symbols in universe.
        
        Args:
            symbols: List of (symbol, asset_class) tuples
            
        Returns:
            List of (symbol, asset_class, is_valid) tuples
        """
        results = []
        for symbol, asset_class in symbols:
            try:
                adapter = self.get_adapter(asset_class)
                is_valid = adapter.validate_symbol(symbol)
                results.append((symbol, asset_class, is_valid))
            except Exception as e:
                logger.warning(f"Validation failed for {symbol} ({asset_class.value}): {e}")
                results.append((symbol, asset_class, False))
        
        return results

