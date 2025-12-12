"""
Options Trading Configuration Loader.

Provides centralized configuration management for:
- Universe of tradable symbols
- Per-ticker parameters
- Default settings
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any
import yaml
from loguru import logger


class VolatilityRegime(Enum):
    """Volatility regime classification."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class Sector(Enum):
    """Sector classifications."""

    TECHNOLOGY = "TECHNOLOGY"
    HEALTHCARE = "HEALTHCARE"
    FINANCIALS = "FINANCIALS"
    CONSUMER_DISCRETIONARY = "CONSUMER_DISCRETIONARY"
    CONSUMER_STAPLES = "CONSUMER_STAPLES"
    INDUSTRIALS = "INDUSTRIALS"
    ENERGY = "ENERGY"
    MATERIALS = "MATERIALS"
    UTILITIES = "UTILITIES"
    REAL_ESTATE = "REAL_ESTATE"
    COMMUNICATION = "COMMUNICATION"
    SMALL_CAP = "SMALL_CAP"
    BENCHMARK = "BENCHMARK"
    ETF = "ETF"


@dataclass
class SymbolConfig:
    """Configuration for a single symbol."""

    symbol: str
    name: str
    sector: Sector
    is_etf: bool = False
    sector_etf: Optional[str] = None
    use_for_regime: bool = False

    # Per-ticker parameters (from ticker_params.yaml)
    iv_rank_low: float = 30.0
    iv_rank_high: float = 70.0
    max_contracts: int = 10
    min_dte: int = 20
    max_dte: int = 45
    preferred_dte: int = 30
    earnings_buffer_days: int = 5
    volatility_regime: VolatilityRegime = VolatilityRegime.MEDIUM
    min_option_volume: int = 100
    max_spread_pct: float = 0.10
    notes: str = ""


@dataclass
class UniverseConfig:
    """Complete universe configuration."""

    benchmarks: List[SymbolConfig] = field(default_factory=list)
    sector_etfs: List[SymbolConfig] = field(default_factory=list)
    other_etfs: List[SymbolConfig] = field(default_factory=list)
    stocks: List[SymbolConfig] = field(default_factory=list)

    # Defaults
    default_min_dte: int = 20
    default_max_dte: int = 45
    default_preferred_dte: int = 30
    default_earnings_buffer_days: int = 5
    default_max_contracts: int = 10
    default_iv_rank_low: float = 30.0
    default_iv_rank_high: float = 70.0

    @property
    def all_symbols(self) -> List[SymbolConfig]:
        """Get all symbols in universe."""
        return self.benchmarks + self.sector_etfs + self.other_etfs + self.stocks

    @property
    def all_etfs(self) -> List[SymbolConfig]:
        """Get all ETF symbols."""
        return self.benchmarks + self.sector_etfs + self.other_etfs

    @property
    def symbol_list(self) -> List[str]:
        """Get list of all symbol strings."""
        return [s.symbol for s in self.all_symbols]

    @property
    def tradable_symbols(self) -> List[str]:
        """Get symbols that can be traded (excludes benchmark-only)."""
        return [s.symbol for s in self.all_symbols]

    def get_symbol(self, symbol: str) -> Optional[SymbolConfig]:
        """Get configuration for a specific symbol."""
        for s in self.all_symbols:
            if s.symbol == symbol:
                return s
        return None

    def get_sector_etf(self, sector: Sector) -> Optional[str]:
        """Get sector ETF for a given sector."""
        for etf in self.sector_etfs:
            if etf.sector == sector:
                return etf.symbol
        return None

    def get_symbols_by_sector(self, sector: Sector) -> List[str]:
        """Get all symbols in a sector."""
        return [s.symbol for s in self.all_symbols if s.sector == sector]


class OptionsConfigLoader:
    """
    Loads and manages options trading configuration.

    Usage:
        loader = OptionsConfigLoader()
        config = loader.load()

        # Get specific symbol config
        aapl_config = config.get_symbol("AAPL")

        # Get all symbols
        symbols = config.symbol_list
    """

    DEFAULT_CONFIG_DIR = Path(__file__).parent.parent.parent / "configs" / "options"

    def __init__(
        self,
        config_dir: Optional[Path] = None,
    ):
        """
        Initialize config loader.

        Args:
            config_dir: Directory containing config files (default: configs/options/)
        """
        self.config_dir = config_dir or self.DEFAULT_CONFIG_DIR
        self._universe_config: Optional[UniverseConfig] = None
        self._ticker_params: Dict[str, Dict[str, Any]] = {}

    def load(self) -> UniverseConfig:
        """
        Load universe and ticker configs.

        Returns:
            UniverseConfig with all symbols and parameters
        """
        if self._universe_config is not None:
            return self._universe_config

        # Load universe.yaml
        universe_path = self.config_dir / "universe.yaml"
        ticker_params_path = self.config_dir / "ticker_params.yaml"

        if not universe_path.exists():
            logger.warning(
                f"Universe config not found at {universe_path}, using defaults"
            )
            return self._create_default_config()

        with open(universe_path, "r") as f:
            universe_data = yaml.safe_load(f)

        # Load ticker params if exists
        if ticker_params_path.exists():
            with open(ticker_params_path, "r") as f:
                self._ticker_params = yaml.safe_load(f) or {}

        # Parse defaults
        defaults = universe_data.get("defaults", {})

        # Parse symbols
        config = UniverseConfig(
            default_min_dte=defaults.get("min_dte", 20),
            default_max_dte=defaults.get("max_dte", 45),
            default_preferred_dte=defaults.get("preferred_dte", 30),
            default_earnings_buffer_days=defaults.get("earnings_buffer_days", 5),
            default_max_contracts=defaults.get("max_contracts", 10),
            default_iv_rank_low=defaults.get("iv_rank_low", 30.0),
            default_iv_rank_high=defaults.get("iv_rank_high", 70.0),
        )

        # Parse benchmarks
        for item in universe_data.get("benchmarks", []):
            config.benchmarks.append(self._parse_symbol(item, defaults))

        # Parse sector ETFs
        for item in universe_data.get("sector_etfs", []):
            config.sector_etfs.append(self._parse_symbol(item, defaults))

        # Parse other ETFs
        for item in universe_data.get("other_etfs", []):
            config.other_etfs.append(self._parse_symbol(item, defaults))

        # Parse stocks
        for item in universe_data.get("stocks", []):
            config.stocks.append(self._parse_symbol(item, defaults))

        self._universe_config = config
        logger.info(f"Loaded {len(config.all_symbols)} symbols from config")

        return config

    def _parse_symbol(
        self,
        item: Dict[str, Any],
        defaults: Dict[str, Any],
    ) -> SymbolConfig:
        """Parse a symbol entry from YAML."""
        symbol = item["symbol"]

        # Get ticker-specific overrides
        ticker_overrides = self._ticker_params.get(symbol, {})

        # Parse sector
        sector_str = item.get("sector", "ETF")
        try:
            sector = Sector[sector_str]
        except KeyError:
            sector = Sector.ETF

        # Parse volatility regime
        vol_regime_str = ticker_overrides.get("volatility_regime", "MEDIUM")
        try:
            vol_regime = VolatilityRegime[vol_regime_str]
        except KeyError:
            vol_regime = VolatilityRegime.MEDIUM

        return SymbolConfig(
            symbol=symbol,
            name=item.get("name", symbol),
            sector=sector,
            is_etf=item.get("is_etf", False),
            sector_etf=item.get("sector_etf"),
            use_for_regime=item.get("use_for_regime", False),
            # Parameters with override chain: ticker_params > item > defaults
            iv_rank_low=ticker_overrides.get(
                "iv_rank_low",
                item.get("iv_rank_low", defaults.get("iv_rank_low", 30.0)),
            ),
            iv_rank_high=ticker_overrides.get(
                "iv_rank_high",
                item.get("iv_rank_high", defaults.get("iv_rank_high", 70.0)),
            ),
            max_contracts=ticker_overrides.get(
                "max_contracts",
                item.get("max_contracts", defaults.get("max_contracts", 10)),
            ),
            min_dte=ticker_overrides.get(
                "min_dte", item.get("min_dte", defaults.get("min_dte", 20))
            ),
            max_dte=ticker_overrides.get(
                "max_dte", item.get("max_dte", defaults.get("max_dte", 45))
            ),
            preferred_dte=ticker_overrides.get(
                "preferred_dte",
                item.get("preferred_dte", defaults.get("preferred_dte", 30)),
            ),
            earnings_buffer_days=ticker_overrides.get(
                "earnings_buffer_days",
                item.get(
                    "earnings_buffer_days", defaults.get("earnings_buffer_days", 5)
                ),
            ),
            volatility_regime=vol_regime,
            min_option_volume=ticker_overrides.get(
                "min_option_volume",
                item.get("min_option_volume", defaults.get("min_option_volume", 100)),
            ),
            max_spread_pct=ticker_overrides.get(
                "max_spread_pct",
                item.get("max_spread_pct", defaults.get("max_spread_pct", 0.10)),
            ),
            notes=ticker_overrides.get("notes", ""),
        )

    def _create_default_config(self) -> UniverseConfig:
        """Create default config when YAML not found."""
        return UniverseConfig(
            benchmarks=[
                SymbolConfig("SPY", "SPDR S&P 500 ETF", Sector.BENCHMARK, is_etf=True),
                SymbolConfig("QQQ", "Invesco QQQ Trust", Sector.BENCHMARK, is_etf=True),
            ],
            stocks=[
                SymbolConfig("AAPL", "Apple Inc", Sector.TECHNOLOGY),
                SymbolConfig("MSFT", "Microsoft Corp", Sector.TECHNOLOGY),
            ],
        )

    def get_symbol_params(self, symbol: str) -> Optional[SymbolConfig]:
        """
        Get parameters for a specific symbol.

        Args:
            symbol: Ticker symbol

        Returns:
            SymbolConfig or None if not found
        """
        config = self.load()
        return config.get_symbol(symbol)

    def get_earnings_buffer(self, symbol: str) -> int:
        """Get earnings buffer days for a symbol."""
        sym_config = self.get_symbol_params(symbol)
        if sym_config:
            return sym_config.earnings_buffer_days
        return 5  # Default

    def get_iv_thresholds(self, symbol: str) -> tuple[float, float]:
        """Get IV rank thresholds for a symbol."""
        sym_config = self.get_symbol_params(symbol)
        if sym_config:
            return sym_config.iv_rank_low, sym_config.iv_rank_high
        return 30.0, 70.0  # Defaults

    def get_max_contracts(self, symbol: str) -> int:
        """Get max contracts for a symbol."""
        sym_config = self.get_symbol_params(symbol)
        if sym_config:
            return sym_config.max_contracts
        return 10  # Default


# Singleton instance for convenience
_config_loader: Optional[OptionsConfigLoader] = None


def get_options_config() -> UniverseConfig:
    """Get options universe config (singleton)."""
    global _config_loader
    if _config_loader is None:
        _config_loader = OptionsConfigLoader()
    return _config_loader.load()


def get_symbol_config(symbol: str) -> Optional[SymbolConfig]:
    """Get config for a specific symbol."""
    global _config_loader
    if _config_loader is None:
        _config_loader = OptionsConfigLoader()
    return _config_loader.get_symbol_params(symbol)
