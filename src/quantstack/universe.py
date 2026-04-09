"""
Trading universe constants — zero quantstack dependencies.

This module is the canonical home for universe definitions: types, the master
symbol dict, and all named subsets.  It sits below both ``config`` and ``data``
in the dependency hierarchy so that neither package creates a circular import
when referencing it.

Dependency rule: this file may only import from stdlib and third-party packages.
Do NOT add any ``from quantstack.*`` imports here.
"""

from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Any


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
CROSS_ASSET_ETFS: tuple[str, ...] = ("SPY", "QQQ", "IWM", "TLT", "GLD", "VXX")

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


# ---------------------------------------------------------------------------
# Point-in-time universe filter (survivorship-bias guard)
# ---------------------------------------------------------------------------

def universe_as_of(dt: date, conn: Any) -> list[str]:
    """Return symbols that were active (not delisted, not pre-IPO) at *dt*.

    Uses the ``universe`` + ``company_overview`` tables.  Symbols with NULL
    ``ipo_date`` are conservatively included (assumed to have always existed).

    Args:
        dt: Reference date for the point-in-time snapshot.
        conn: A DB connection supporting ``conn.execute(sql, params).fetchall()``.
              Accepts the ``PgConnection`` wrapper from ``quantstack.db`` —
              no quantstack imports are added here to preserve the dependency rule.

    Returns:
        Sorted list of ticker symbols active at *dt*.
    """
    rows = conn.execute(
        """
        SELECT u.symbol
        FROM universe u
        JOIN company_overview co ON u.symbol = co.symbol
        WHERE u.is_active = TRUE
          AND (co.ipo_date IS NULL OR co.ipo_date <= %s)
          AND (co.delisted_at IS NULL OR co.delisted_at > %s)
        ORDER BY u.symbol
        """,
        [dt, dt],
    ).fetchall()
    return [r[0] for r in rows]
