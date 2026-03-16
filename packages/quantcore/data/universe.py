"""
Universe management for options trading.

Provides:
- S&P 500 constituent list with sector mapping
- Hard gating liquidity filters
- Dynamic universe updates
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set
import pandas as pd
from loguru import logger


class Sector(Enum):
    """GICS sector classifications."""
    TECHNOLOGY = "Technology"
    HEALTHCARE = "Healthcare"
    FINANCIALS = "Financials"
    CONSUMER_DISCRETIONARY = "Consumer Discretionary"
    CONSUMER_STAPLES = "Consumer Staples"
    INDUSTRIALS = "Industrials"
    ENERGY = "Energy"
    MATERIALS = "Materials"
    UTILITIES = "Utilities"
    REAL_ESTATE = "Real Estate"
    COMMUNICATION = "Communication Services"
    ETF = "ETF"


@dataclass
class UniverseSymbol:
    """Symbol with metadata."""
    symbol: str
    name: str
    sector: Sector
    is_etf: bool = False
    avg_option_volume: Optional[float] = None
    median_spread_pct: Optional[float] = None
    avg_oi: Optional[float] = None
    
    @property
    def is_liquid(self) -> bool:
        """Check if symbol passes liquidity filters."""
        # Hard gating thresholds from plan
        if self.avg_option_volume is not None and self.avg_option_volume < 10000:
            return False
        if self.median_spread_pct is not None and self.median_spread_pct > 0.05:
            return False
        if self.avg_oi is not None and self.avg_oi < 1000:
            return False
        return True


# Target universe: 27 tickers (16 stocks + 11 ETFs)
# Selected for high liquidity and options volume
INITIAL_LIQUID_UNIVERSE: Dict[str, UniverseSymbol] = {
    # ===== ETFs (11) =====
    "SPY": UniverseSymbol("SPY", "SPDR S&P 500 ETF", Sector.ETF, is_etf=True),
    "QQQ": UniverseSymbol("QQQ", "Invesco QQQ Trust", Sector.ETF, is_etf=True),
    "IWM": UniverseSymbol("IWM", "iShares Russell 2000", Sector.ETF, is_etf=True),
    "GDX": UniverseSymbol("GDX", "VanEck Gold Miners ETF", Sector.ETF, is_etf=True),
    "XLE": UniverseSymbol("XLE", "Energy Select Sector SPDR", Sector.ETF, is_etf=True),
    "XLF": UniverseSymbol("XLF", "Financial Select Sector SPDR", Sector.ETF, is_etf=True),
    "XLI": UniverseSymbol("XLI", "Industrial Select Sector SPDR", Sector.ETF, is_etf=True),
    "XLP": UniverseSymbol("XLP", "Consumer Staples Select Sector SPDR", Sector.ETF, is_etf=True),
    "XLV": UniverseSymbol("XLV", "Health Care Select Sector SPDR", Sector.ETF, is_etf=True),
    "XLY": UniverseSymbol("XLY", "Consumer Discretionary Select Sector SPDR", Sector.ETF, is_etf=True),
    "XME": UniverseSymbol("XME", "SPDR S&P Metals & Mining ETF", Sector.ETF, is_etf=True),
    
    # ===== Individual Stocks (16) =====
    # Technology
    "AAPL": UniverseSymbol("AAPL", "Apple Inc", Sector.TECHNOLOGY),
    "AMD": UniverseSymbol("AMD", "Advanced Micro Devices", Sector.TECHNOLOGY),
    "MSFT": UniverseSymbol("MSFT", "Microsoft Corp", Sector.TECHNOLOGY),
    "NVDA": UniverseSymbol("NVDA", "NVIDIA Corp", Sector.TECHNOLOGY),
    
    # Communication Services
    "GOOGL": UniverseSymbol("GOOGL", "Alphabet Inc Class A", Sector.COMMUNICATION),
    "META": UniverseSymbol("META", "Meta Platforms Inc", Sector.COMMUNICATION),
    "NFLX": UniverseSymbol("NFLX", "Netflix Inc", Sector.COMMUNICATION),
    
    # Consumer Discretionary
    "AMZN": UniverseSymbol("AMZN", "Amazon.com Inc", Sector.CONSUMER_DISCRETIONARY),
    "NKE": UniverseSymbol("NKE", "Nike Inc", Sector.CONSUMER_DISCRETIONARY),
    "TSLA": UniverseSymbol("TSLA", "Tesla Inc", Sector.CONSUMER_DISCRETIONARY),
    "BABA": UniverseSymbol("BABA", "Alibaba Group Holding", Sector.CONSUMER_DISCRETIONARY),
    
    # Financials
    "BAC": UniverseSymbol("BAC", "Bank of America", Sector.FINANCIALS),
    
    # Industrials
    "AAL": UniverseSymbol("AAL", "American Airlines Group", Sector.INDUSTRIALS),
    "BA": UniverseSymbol("BA", "Boeing Co", Sector.INDUSTRIALS),
    
    # Energy
    "XOM": UniverseSymbol("XOM", "Exxon Mobil", Sector.ENERGY),
}


class UniverseManager:
    """
    Manages the trading universe with liquidity filtering.
    
    Features:
    - Initial liquid universe of 50 names
    - Hard gating filters for options liquidity
    - Sector-based filtering
    - Dynamic updates based on market data
    """
    
    # Liquidity thresholds (from plan)
    MIN_AVG_OPTION_VOLUME = 10000  # contracts/day
    MAX_MEDIAN_SPREAD_PCT = 0.05   # 5% of mid
    MIN_AVG_OI = 1000              # open interest for 20-45 DTE
    
    def __init__(
        self,
        custom_symbols: Optional[Dict[str, UniverseSymbol]] = None,
    ):
        """
        Initialize universe manager.
        
        Args:
            custom_symbols: Optional custom universe (uses default if None)
        """
        self._universe = custom_symbols or INITIAL_LIQUID_UNIVERSE.copy()
        self._active_symbols: Set[str] = set(self._universe.keys())
        
    @property
    def symbols(self) -> List[str]:
        """Get list of active symbols."""
        return sorted(self._active_symbols)
    
    @property
    def etf_symbols(self) -> List[str]:
        """Get ETF symbols only."""
        return [s for s in self._active_symbols 
                if self._universe.get(s, UniverseSymbol(s, s, Sector.ETF)).is_etf]
    
    @property
    def equity_symbols(self) -> List[str]:
        """Get equity (non-ETF) symbols only."""
        return [s for s in self._active_symbols 
                if not self._universe.get(s, UniverseSymbol(s, s, Sector.ETF)).is_etf]
    
    def get_symbols_by_sector(self, sector: Sector) -> List[str]:
        """Get symbols for a specific sector."""
        return [s for s in self._active_symbols 
                if self._universe.get(s) and self._universe[s].sector == sector]
    
    def get_symbol_info(self, symbol: str) -> Optional[UniverseSymbol]:
        """Get symbol metadata."""
        return self._universe.get(symbol)
    
    def add_symbol(
        self,
        symbol: str,
        name: str,
        sector: Sector,
        is_etf: bool = False,
    ) -> None:
        """Add a symbol to the universe."""
        self._universe[symbol] = UniverseSymbol(
            symbol=symbol,
            name=name,
            sector=sector,
            is_etf=is_etf,
        )
        self._active_symbols.add(symbol)
        logger.info(f"Added {symbol} to universe")
    
    def remove_symbol(self, symbol: str) -> None:
        """Remove a symbol from active universe."""
        self._active_symbols.discard(symbol)
        logger.info(f"Removed {symbol} from active universe")
    
    def update_liquidity_metrics(
        self,
        symbol: str,
        avg_option_volume: Optional[float] = None,
        median_spread_pct: Optional[float] = None,
        avg_oi: Optional[float] = None,
    ) -> bool:
        """
        Update liquidity metrics for a symbol.
        
        Args:
            symbol: Symbol to update
            avg_option_volume: Average daily option volume
            median_spread_pct: Median bid-ask spread as % of mid
            avg_oi: Average open interest for 20-45 DTE options
            
        Returns:
            True if symbol passes liquidity gate, False otherwise
        """
        if symbol not in self._universe:
            logger.warning(f"Symbol {symbol} not in universe")
            return False
        
        info = self._universe[symbol]
        
        if avg_option_volume is not None:
            info.avg_option_volume = avg_option_volume
        if median_spread_pct is not None:
            info.median_spread_pct = median_spread_pct
        if avg_oi is not None:
            info.avg_oi = avg_oi
        
        # Apply hard gating
        passes_gate = self._check_liquidity_gate(info)
        
        if passes_gate:
            self._active_symbols.add(symbol)
        else:
            self._active_symbols.discard(symbol)
            logger.warning(f"Symbol {symbol} failed liquidity gate")
        
        return passes_gate
    
    def _check_liquidity_gate(self, info: UniverseSymbol) -> bool:
        """Apply hard liquidity gating."""
        if info.avg_option_volume is not None:
            if info.avg_option_volume < self.MIN_AVG_OPTION_VOLUME:
                return False
        
        if info.median_spread_pct is not None:
            if info.median_spread_pct > self.MAX_MEDIAN_SPREAD_PCT:
                return False
        
        if info.avg_oi is not None:
            if info.avg_oi < self.MIN_AVG_OI:
                return False
        
        return True
    
    def refresh_from_options_data(
        self,
        options_df: pd.DataFrame,
        min_dte: int = 20,
        max_dte: int = 45,
    ) -> Dict[str, bool]:
        """
        Refresh liquidity metrics from options chain data.
        
        Args:
            options_df: DataFrame with options data (must have underlying, volume, bid, ask, expiry)
            min_dte: Minimum days to expiry for OI calculation
            max_dte: Maximum days to expiry for OI calculation
            
        Returns:
            Dictionary of symbol -> passes_gate
        """
        if options_df.empty:
            return {}
        
        results = {}
        
        # Calculate DTE
        if "expiry" in options_df.columns:
            options_df = options_df.copy()
            options_df["dte"] = (pd.to_datetime(options_df["expiry"]) - pd.Timestamp.now()).dt.days
            
            # Filter to target DTE range
            dte_filtered = options_df[
                (options_df["dte"] >= min_dte) & 
                (options_df["dte"] <= max_dte)
            ]
        else:
            dte_filtered = options_df
        
        for symbol in options_df["underlying"].unique():
            symbol_data = dte_filtered[dte_filtered["underlying"] == symbol]
            
            if symbol_data.empty:
                continue
            
            # Calculate metrics
            avg_volume = symbol_data["volume"].mean() if "volume" in symbol_data.columns else None
            
            # Calculate spread percentage
            if "bid" in symbol_data.columns and "ask" in symbol_data.columns:
                valid_quotes = symbol_data[(symbol_data["bid"] > 0) & (symbol_data["ask"] > 0)]
                if not valid_quotes.empty:
                    mid = (valid_quotes["bid"] + valid_quotes["ask"]) / 2
                    spread = valid_quotes["ask"] - valid_quotes["bid"]
                    spread_pct = (spread / mid).median()
                else:
                    spread_pct = None
            else:
                spread_pct = None
            
            avg_oi = symbol_data["open_interest"].mean() if "open_interest" in symbol_data.columns else None
            
            # Update and check gate
            if symbol in self._universe:
                passes = self.update_liquidity_metrics(
                    symbol=symbol,
                    avg_option_volume=avg_volume,
                    median_spread_pct=spread_pct,
                    avg_oi=avg_oi,
                )
                results[symbol] = passes
        
        return results
    
    def get_universe_summary(self) -> pd.DataFrame:
        """Get summary DataFrame of universe with metrics."""
        data = []
        for symbol in sorted(self._universe.keys()):
            info = self._universe[symbol]
            data.append({
                "symbol": symbol,
                "name": info.name,
                "sector": info.sector.value,
                "is_etf": info.is_etf,
                "active": symbol in self._active_symbols,
                "avg_option_volume": info.avg_option_volume,
                "median_spread_pct": info.median_spread_pct,
                "avg_oi": info.avg_oi,
            })
        
        return pd.DataFrame(data)
    
    def __len__(self) -> int:
        """Number of active symbols."""
        return len(self._active_symbols)
    
    def __contains__(self, symbol: str) -> bool:
        """Check if symbol is in active universe."""
        return symbol in self._active_symbols
    
    def __iter__(self):
        """Iterate over active symbols."""
        return iter(sorted(self._active_symbols))

