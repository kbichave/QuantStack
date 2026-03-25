"""
Universe management for options trading.

Provides:
- S&P 500 constituent list with sector mapping
- Hard gating liquidity filters
- Dynamic universe updates
"""

from dataclasses import dataclass
from enum import Enum

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
    description: str = ""          # one-liner: what the company/ETF does
    avg_option_volume: float | None = None
    median_spread_pct: float | None = None
    avg_oi: float | None = None

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


# Target universe: 78 tickers (27 ETFs/indices + 35 large-cap stocks + 16 speculative/emerging)
#
# Selection criteria:
#   - ETFs/indices: broadest market coverage + highest-volume vol/sector instruments
#   - Stocks: top ADV and options-volume name per sector; all pass 500k+ shares/day
#   - Speculative: high-beta / thematic names (quantum, space, crypto, fintech, AI infra)
#     — tracked for momentum and event-driven setups; liquidity gates apply per symbol
#
# All equity/ETF symbols are supported by Alpha Vantage TIME_SERIES_INTRADAY (adjusted)
# and TIME_SERIES_DAILY_ADJUSTED, which is the primary OHLCV source for this system.
# Exception: VIX is sourced from FRED (VIXCLS series) via economic_fetcher, not Alpha Vantage.
INITIAL_LIQUID_UNIVERSE: dict[str, UniverseSymbol] = {
    # ===== ETFs (15) — broad market, rates, gold, sector, vol, leveraged =====
    "SPY": UniverseSymbol("SPY", "SPDR S&P 500 ETF", Sector.ETF, is_etf=True),
    "QQQ": UniverseSymbol("QQQ", "Invesco QQQ Trust", Sector.ETF, is_etf=True),
    "IWM": UniverseSymbol("IWM", "iShares Russell 2000 ETF", Sector.ETF, is_etf=True),
    "TLT": UniverseSymbol(
        "TLT", "iShares 20+ Year Treasury Bond ETF", Sector.ETF, is_etf=True
    ),
    "GLD": UniverseSymbol("GLD", "SPDR Gold Shares", Sector.ETF, is_etf=True),
    "GDX": UniverseSymbol("GDX", "VanEck Gold Miners ETF", Sector.ETF, is_etf=True),
    "VIX": UniverseSymbol(
        "VIX", "CBOE Volatility Index", Sector.ETF, is_etf=True,
        description="CBOE Volatility Index — 30-day implied vol of S&P 500 options",
    ),
    "VXX": UniverseSymbol(
        "VXX", "iPath Series B S&P 500 VIX Short-Term ETN", Sector.ETF, is_etf=True
    ),
    "TQQQ": UniverseSymbol("TQQQ", "ProShares UltraPro QQQ", Sector.ETF, is_etf=True),
    "SQQQ": UniverseSymbol(
        "SQQQ", "ProShares UltraPro Short QQQ", Sector.ETF, is_etf=True
    ),
    "XLE": UniverseSymbol("XLE", "Energy Select Sector SPDR", Sector.ETF, is_etf=True),
    "XLF": UniverseSymbol(
        "XLF", "Financial Select Sector SPDR", Sector.ETF, is_etf=True
    ),
    "XLK": UniverseSymbol(
        "XLK", "Technology Select Sector SPDR", Sector.ETF, is_etf=True
    ),
    "XLV": UniverseSymbol(
        "XLV", "Health Care Select Sector SPDR", Sector.ETF, is_etf=True
    ),
    "XLI": UniverseSymbol(
        "XLI", "Industrial Select Sector SPDR", Sector.ETF, is_etf=True
    ),
    "XLP": UniverseSymbol(
        "XLP", "Consumer Staples Select Sector SPDR", Sector.ETF, is_etf=True
    ),
    # ===== Technology (10) =====
    "AAPL": UniverseSymbol("AAPL", "Apple Inc", Sector.TECHNOLOGY),
    "MSFT": UniverseSymbol("MSFT", "Microsoft Corp", Sector.TECHNOLOGY),
    "NVDA": UniverseSymbol("NVDA", "NVIDIA Corp", Sector.TECHNOLOGY),
    "AMD": UniverseSymbol("AMD", "Advanced Micro Devices", Sector.TECHNOLOGY),
    "AVGO": UniverseSymbol("AVGO", "Broadcom Inc", Sector.TECHNOLOGY),
    "INTC": UniverseSymbol("INTC", "Intel Corp", Sector.TECHNOLOGY),
    "ORCL": UniverseSymbol("ORCL", "Oracle Corp", Sector.TECHNOLOGY),
    "CRM": UniverseSymbol("CRM", "Salesforce Inc", Sector.TECHNOLOGY),
    "PLTR": UniverseSymbol("PLTR", "Palantir Technologies", Sector.TECHNOLOGY),
    "UBER": UniverseSymbol("UBER", "Uber Technologies Inc", Sector.TECHNOLOGY),
    # ===== Communication Services / Consumer Internet (4) =====
    "GOOGL": UniverseSymbol("GOOGL", "Alphabet Inc Class A", Sector.COMMUNICATION),
    "META": UniverseSymbol("META", "Meta Platforms Inc", Sector.COMMUNICATION),
    "AMZN": UniverseSymbol("AMZN", "Amazon.com Inc", Sector.COMMUNICATION),
    "NFLX": UniverseSymbol("NFLX", "Netflix Inc", Sector.COMMUNICATION),
    # ===== Consumer Discretionary (4) =====
    "TSLA": UniverseSymbol("TSLA", "Tesla Inc", Sector.CONSUMER_DISCRETIONARY),
    "HD": UniverseSymbol("HD", "Home Depot Inc", Sector.CONSUMER_DISCRETIONARY),
    "MCD": UniverseSymbol("MCD", "McDonald's Corp", Sector.CONSUMER_DISCRETIONARY),
    "COST": UniverseSymbol(
        "COST", "Costco Wholesale Corp", Sector.CONSUMER_DISCRETIONARY
    ),
    # ===== Consumer Staples (2) =====
    "WMT": UniverseSymbol("WMT", "Walmart Inc", Sector.CONSUMER_STAPLES),
    "KO": UniverseSymbol("KO", "Coca-Cola Co", Sector.CONSUMER_STAPLES),
    # ===== Financials (6) =====
    "JPM": UniverseSymbol("JPM", "JPMorgan Chase & Co", Sector.FINANCIALS),
    "BAC": UniverseSymbol("BAC", "Bank of America Corp", Sector.FINANCIALS),
    "GS": UniverseSymbol("GS", "Goldman Sachs Group", Sector.FINANCIALS),
    "V": UniverseSymbol("V", "Visa Inc", Sector.FINANCIALS),
    "MA": UniverseSymbol("MA", "Mastercard Inc", Sector.FINANCIALS),
    "C": UniverseSymbol("C", "Citigroup Inc", Sector.FINANCIALS),
    # ===== Healthcare (5) =====
    "UNH": UniverseSymbol("UNH", "UnitedHealth Group", Sector.HEALTHCARE),
    "JNJ": UniverseSymbol("JNJ", "Johnson & Johnson", Sector.HEALTHCARE),
    "LLY": UniverseSymbol("LLY", "Eli Lilly and Co", Sector.HEALTHCARE),
    "PFE": UniverseSymbol("PFE", "Pfizer Inc", Sector.HEALTHCARE),
    "ABBV": UniverseSymbol("ABBV", "AbbVie Inc", Sector.HEALTHCARE),
    # ===== Energy (2) =====
    "XOM": UniverseSymbol("XOM", "Exxon Mobil Corp", Sector.ENERGY),
    "CVX": UniverseSymbol("CVX", "Chevron Corp", Sector.ENERGY),
    # ===== Industrials (2) =====
    "BA": UniverseSymbol("BA", "Boeing Co", Sector.INDUSTRIALS),
    "CAT": UniverseSymbol("CAT", "Caterpillar Inc", Sector.INDUSTRIALS),
    # ===== Macro / Breadth / Rates Reference ETFs (11) =====
    # Used for credit signals, sector breadth, and regime detection.
    # Not primary trading targets — data must still be acquired for these.
    "XLY": UniverseSymbol("XLY", "Consumer Discretionary Select Sector SPDR", Sector.ETF, is_etf=True),
    "XLB": UniverseSymbol("XLB", "Materials Select Sector SPDR", Sector.ETF, is_etf=True),
    "XLRE": UniverseSymbol("XLRE", "Real Estate Select Sector SPDR", Sector.ETF, is_etf=True),
    "XLU": UniverseSymbol("XLU", "Utilities Select Sector SPDR", Sector.ETF, is_etf=True),
    "XLC": UniverseSymbol("XLC", "Communication Services Select Sector SPDR", Sector.ETF, is_etf=True),
    "MDY": UniverseSymbol("MDY", "SPDR S&P 400 Mid-Cap ETF", Sector.ETF, is_etf=True),
    "HYG": UniverseSymbol("HYG", "iShares HY Corporate Bond ETF", Sector.ETF, is_etf=True),
    "LQD": UniverseSymbol("LQD", "iShares IG Corporate Bond ETF", Sector.ETF, is_etf=True),
    "IEF": UniverseSymbol("IEF", "iShares 7-10 Year Treasury Bond ETF", Sector.ETF, is_etf=True),
    "SHY": UniverseSymbol("SHY", "iShares 1-3 Year Treasury Bond ETF", Sector.ETF, is_etf=True),
    "UUP": UniverseSymbol("UUP", "Invesco DB US Dollar Index Bullish Fund", Sector.ETF, is_etf=True),
    # ===== Speculative / Emerging (16) — high-beta, thematic, event-driven =====
    # AI infrastructure
    "ALAB": UniverseSymbol("ALAB", "Astera Labs Inc", Sector.TECHNOLOGY),
    "SMCI": UniverseSymbol("SMCI", "Super Micro Computer Inc", Sector.TECHNOLOGY),
    "NBIS": UniverseSymbol("NBIS", "Nebius Group NV", Sector.TECHNOLOGY),
    # Quantum computing
    "IONQ": UniverseSymbol("IONQ", "IonQ Inc", Sector.TECHNOLOGY),
    "RGTI": UniverseSymbol("RGTI", "Rigetti Computing Inc", Sector.TECHNOLOGY),
    "QBTS": UniverseSymbol("QBTS", "D-Wave Quantum Inc", Sector.TECHNOLOGY),
    # Space / aerospace
    "RKLB": UniverseSymbol("RKLB", "Rocket Lab USA Inc", Sector.INDUSTRIALS),
    "LUNR": UniverseSymbol("LUNR", "Intuitive Machines Inc", Sector.INDUSTRIALS),
    # eVTOL / air mobility
    "JOBY": UniverseSymbol("JOBY", "Joby Aviation Inc", Sector.INDUSTRIALS),
    "ACHR": UniverseSymbol("ACHR", "Archer Aviation Inc", Sector.INDUSTRIALS),
    # Crypto miners / treasury
    "MSTR": UniverseSymbol("MSTR", "MicroStrategy Inc", Sector.TECHNOLOGY),
    "MARA": UniverseSymbol("MARA", "Marathon Digital Holdings Inc", Sector.TECHNOLOGY),
    "RIOT": UniverseSymbol("RIOT", "Riot Platforms Inc", Sector.TECHNOLOGY),
    # Fintech / retail brokerage
    "HOOD": UniverseSymbol("HOOD", "Robinhood Markets Inc", Sector.FINANCIALS),
    "SOFI": UniverseSymbol("SOFI", "SoFi Technologies Inc", Sector.FINANCIALS),
    # Social media
    "RDDT": UniverseSymbol("RDDT", "Reddit Inc", Sector.COMMUNICATION),
}


# ---------------------------------------------------------------------------
# Named subsets — import these instead of hardcoding symbol lists anywhere.
# Add a ticker once (to INITIAL_LIQUID_UNIVERSE above), then reference it here
# if it belongs to a logical group. Everything else derives from these.
# ---------------------------------------------------------------------------

# Regime detection: cross-asset ETFs that anchor the macro picture.
CROSS_ASSET_ETFS: tuple[str, ...] = ("SPY", "QQQ", "IWM", "TLT", "GLD", "VIX")

# Breadth tracking: all 11 SPDR sector ETFs + broad indices.
SECTOR_ETFS: tuple[str, ...] = (
    "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLRE", "XLU", "XLC",
    "SPY", "QQQ", "IWM", "MDY",
)

# Credit / macro context: bond + dollar + gold ETFs.
CREDIT_ETFS: tuple[str, ...] = ("HYG", "LQD", "TLT", "IEF", "SHY", "GLD", "UUP")

# Watchlist fallback: broad liquid names used when screener has no results.
WATCHLIST_DEFAULT: tuple[str, ...] = ("SPY", "QQQ", "IWM", "XOM", "MSFT", "AAPL", "JPM", "GLD")

# Bootstrap default: minimal set for first-run setup and connectivity tests.
BOOTSTRAP_DEFAULT: tuple[str, ...] = ("SPY", "QQQ", "IWM", "TSLA", "NVDA")

# Strategy backtest default: diversified sample across sectors.
STRATEGY_BACKTEST_DEFAULT: tuple[str, ...] = ("SPY", "QQQ", "AAPL", "MSFT", "XOM")

# Speculative / emerging names: high-beta thematic tickers for event-driven setups.
SPECULATIVE_SYMBOLS: tuple[str, ...] = (
    "ALAB", "SMCI", "NBIS",
    "IONQ", "RGTI", "QBTS",
    "RKLB", "LUNR",
    "JOBY", "ACHR",
    "MSTR", "MARA", "RIOT",
    "HOOD", "SOFI", "RDDT",
)


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
    MAX_MEDIAN_SPREAD_PCT = 0.05  # 5% of mid
    MIN_AVG_OI = 1000  # open interest for 20-45 DTE

    def __init__(
        self,
        custom_symbols: dict[str, UniverseSymbol] | None = None,
    ):
        """
        Initialize universe manager.

        Args:
            custom_symbols: Optional custom universe (uses default if None)
        """
        self._universe = custom_symbols or INITIAL_LIQUID_UNIVERSE.copy()
        self._active_symbols: set[str] = set(self._universe.keys())

    @property
    def symbols(self) -> list[str]:
        """Get list of active symbols."""
        return sorted(self._active_symbols)

    @property
    def etf_symbols(self) -> list[str]:
        """Get ETF symbols only."""
        return [
            s
            for s in self._active_symbols
            if self._universe.get(s, UniverseSymbol(s, s, Sector.ETF)).is_etf
        ]

    @property
    def equity_symbols(self) -> list[str]:
        """Get equity (non-ETF) symbols only."""
        return [
            s
            for s in self._active_symbols
            if not self._universe.get(s, UniverseSymbol(s, s, Sector.ETF)).is_etf
        ]

    def get_symbols_by_sector(self, sector: Sector) -> list[str]:
        """Get symbols for a specific sector."""
        return [
            s
            for s in self._active_symbols
            if self._universe.get(s) and self._universe[s].sector == sector
        ]

    def get_symbol_info(self, symbol: str) -> UniverseSymbol | None:
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
        avg_option_volume: float | None = None,
        median_spread_pct: float | None = None,
        avg_oi: float | None = None,
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
    ) -> dict[str, bool]:
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
            options_df["dte"] = (
                pd.to_datetime(options_df["expiry"]) - pd.Timestamp.now()
            ).dt.days

            # Filter to target DTE range
            dte_filtered = options_df[
                (options_df["dte"] >= min_dte) & (options_df["dte"] <= max_dte)
            ]
        else:
            dte_filtered = options_df

        for symbol in options_df["underlying"].unique():
            symbol_data = dte_filtered[dte_filtered["underlying"] == symbol]

            if symbol_data.empty:
                continue

            # Calculate metrics
            avg_volume = (
                symbol_data["volume"].mean()
                if "volume" in symbol_data.columns
                else None
            )

            # Calculate spread percentage
            if "bid" in symbol_data.columns and "ask" in symbol_data.columns:
                valid_quotes = symbol_data[
                    (symbol_data["bid"] > 0) & (symbol_data["ask"] > 0)
                ]
                if not valid_quotes.empty:
                    mid = (valid_quotes["bid"] + valid_quotes["ask"]) / 2
                    spread = valid_quotes["ask"] - valid_quotes["bid"]
                    spread_pct = (spread / mid).median()
                else:
                    spread_pct = None
            else:
                spread_pct = None

            avg_oi = (
                symbol_data["open_interest"].mean()
                if "open_interest" in symbol_data.columns
                else None
            )

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
            data.append(
                {
                    "symbol": symbol,
                    "name": info.name,
                    "sector": info.sector.value,
                    "is_etf": info.is_etf,
                    "active": symbol in self._active_symbols,
                    "avg_option_volume": info.avg_option_volume,
                    "median_spread_pct": info.median_spread_pct,
                    "avg_oi": info.avg_oi,
                }
            )

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
