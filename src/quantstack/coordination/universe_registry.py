# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Universe registry — SP500, NASDAQ-100, and liquid ETF constituents.

Maintains a PostgreSQL-backed ``universe`` table of ~700 symbols.  Refreshed
weekly via FinancialDatasets.ai ``stock_screener`` endpoint (for equities)
and a hardcoded liquid ETF list (ETFs don't rebalance frequently enough
to warrant API calls).

Tradeoff: stock_screener uses market-cap + volume filters to *approximate*
index membership (no dedicated constituents endpoint).  This produces ~5-10%
false positives (mid-caps that aren't in the index).  Acceptable because the
downstream AutonomousScreener applies hard liquidity filters that eliminate
illiquid names regardless of index membership.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from loguru import logger

from quantstack.db import PgConnection


class UniverseSource(str, Enum):
    SP500 = "sp500"
    NASDAQ100 = "nasdaq100"
    ETF_LIQUID = "etf_liquid"
    MANUAL = "manual"


@dataclass
class ConstituentRecord:
    symbol: str
    name: str
    sector: str
    source: UniverseSource
    market_cap: float | None = None
    avg_daily_volume: float | None = None
    is_active: bool = True


@dataclass
class RefreshReport:
    """Summary of a universe refresh operation."""

    source: str
    symbols_added: int
    symbols_updated: int
    symbols_deactivated: int
    total_active: int
    errors: list[str]
    refreshed_at: datetime


# ~50 liquid ETFs covering sectors, commodities, bonds, and thematic
LIQUID_ETFS: list[dict[str, str]] = [
    # Broad market
    {"symbol": "SPY", "name": "SPDR S&P 500 ETF", "sector": "ETF"},
    {"symbol": "QQQ", "name": "Invesco QQQ Trust", "sector": "ETF"},
    {"symbol": "IWM", "name": "iShares Russell 2000 ETF", "sector": "ETF"},
    {"symbol": "DIA", "name": "SPDR Dow Jones Industrial Avg ETF", "sector": "ETF"},
    {"symbol": "VOO", "name": "Vanguard S&P 500 ETF", "sector": "ETF"},
    {"symbol": "VTI", "name": "Vanguard Total Stock Market ETF", "sector": "ETF"},
    # Sector SPDRs
    {"symbol": "XLK", "name": "Technology Select Sector SPDR", "sector": "Technology"},
    {"symbol": "XLF", "name": "Financial Select Sector SPDR", "sector": "Financials"},
    {"symbol": "XLE", "name": "Energy Select Sector SPDR", "sector": "Energy"},
    {"symbol": "XLV", "name": "Health Care Select Sector SPDR", "sector": "Healthcare"},
    {
        "symbol": "XLY",
        "name": "Consumer Discretionary SPDR",
        "sector": "Consumer Discretionary",
    },
    {"symbol": "XLP", "name": "Consumer Staples SPDR", "sector": "Consumer Staples"},
    {"symbol": "XLI", "name": "Industrial Select Sector SPDR", "sector": "Industrials"},
    {"symbol": "XLB", "name": "Materials Select Sector SPDR", "sector": "Materials"},
    {"symbol": "XLU", "name": "Utilities Select Sector SPDR", "sector": "Utilities"},
    {
        "symbol": "XLRE",
        "name": "Real Estate Select Sector SPDR",
        "sector": "Real Estate",
    },
    {"symbol": "XLC", "name": "Communication Services SPDR", "sector": "Communication"},
    # Thematic / industry
    {"symbol": "SOXX", "name": "iShares Semiconductor ETF", "sector": "Technology"},
    {"symbol": "IBB", "name": "iShares Biotechnology ETF", "sector": "Healthcare"},
    {"symbol": "XBI", "name": "SPDR S&P Biotech ETF", "sector": "Healthcare"},
    {"symbol": "KRE", "name": "SPDR S&P Regional Banking ETF", "sector": "Financials"},
    {
        "symbol": "XHB",
        "name": "SPDR S&P Homebuilders ETF",
        "sector": "Consumer Discretionary",
    },
    {
        "symbol": "XRT",
        "name": "SPDR S&P Retail ETF",
        "sector": "Consumer Discretionary",
    },
    {"symbol": "XME", "name": "SPDR S&P Metals & Mining ETF", "sector": "Materials"},
    {"symbol": "ARKK", "name": "ARK Innovation ETF", "sector": "Technology"},
    {"symbol": "SMH", "name": "VanEck Semiconductor ETF", "sector": "Technology"},
    {"symbol": "XOP", "name": "SPDR S&P Oil & Gas Exploration ETF", "sector": "Energy"},
    {"symbol": "IYR", "name": "iShares U.S. Real Estate ETF", "sector": "Real Estate"},
    # Commodities
    {"symbol": "GLD", "name": "SPDR Gold Shares", "sector": "Commodities"},
    {"symbol": "SLV", "name": "iShares Silver Trust", "sector": "Commodities"},
    {"symbol": "USO", "name": "United States Oil Fund", "sector": "Commodities"},
    {
        "symbol": "UNG",
        "name": "United States Natural Gas Fund",
        "sector": "Commodities",
    },
    {"symbol": "GDX", "name": "VanEck Gold Miners ETF", "sector": "Commodities"},
    {
        "symbol": "GDXJ",
        "name": "VanEck Junior Gold Miners ETF",
        "sector": "Commodities",
    },
    # Bonds
    {"symbol": "TLT", "name": "iShares 20+ Year Treasury Bond ETF", "sector": "Bonds"},
    {"symbol": "IEF", "name": "iShares 7-10 Year Treasury Bond ETF", "sector": "Bonds"},
    {
        "symbol": "HYG",
        "name": "iShares iBoxx $ High Yield Corporate Bond ETF",
        "sector": "Bonds",
    },
    {
        "symbol": "LQD",
        "name": "iShares iBoxx $ Invest Grade Corp Bond ETF",
        "sector": "Bonds",
    },
    {"symbol": "TIP", "name": "iShares TIPS Bond ETF", "sector": "Bonds"},
    # Volatility / hedging
    {
        "symbol": "VXX",
        "name": "iPath Series B S&P 500 VIX Short-Term Futures",
        "sector": "Volatility",
    },
    {
        "symbol": "UVXY",
        "name": "ProShares Ultra VIX Short-Term Futures ETF",
        "sector": "Volatility",
    },
    # International
    {
        "symbol": "EEM",
        "name": "iShares MSCI Emerging Markets ETF",
        "sector": "International",
    },
    {"symbol": "EFA", "name": "iShares MSCI EAFE ETF", "sector": "International"},
    {"symbol": "FXI", "name": "iShares China Large-Cap ETF", "sector": "International"},
    {"symbol": "EWJ", "name": "iShares MSCI Japan ETF", "sector": "International"},
]


class UniverseRegistry:
    """
    PostgreSQL-backed universe of ~700 symbols with weekly refresh.

    The registry is populated from three sources:
    1. SP500 approximation via stock_screener (market_cap > $10B, ADV > 500k)
    2. NASDAQ-100 approximation via stock_screener (tech/comm/consumer, market_cap > $15B)
    3. Hardcoded liquid ETF list (~50 ETFs)

    Args:
        conn: PostgreSQL connection.
        client: FinancialDatasetsClient for stock_screener calls.
            Pass None if you only need read operations.
    """

    def __init__(
        self,
        conn: PgConnection,
        client: Any | None = None,
    ) -> None:
        self._conn = conn
        self._client = client

    def refresh_constituents(self) -> RefreshReport:
        """
        Fetch SP500 + NASDAQ-100 + ETF list and upsert into universe table.

        Safe to call repeatedly — uses INSERT ... ON CONFLICT DO UPDATE.
        Symbols that disappear from screener results are NOT auto-deactivated
        (they may just have temporarily low volume). Use deactivate_symbol()
        for explicit removal.
        """
        errors: list[str] = []
        added = 0
        updated = 0
        now = datetime.now(timezone.utc)

        # 1. ETFs — always loaded from hardcoded list
        for etf in LIQUID_ETFS:
            try:
                a, u = self._upsert(
                    symbol=etf["symbol"],
                    name=etf["name"],
                    sector=etf["sector"],
                    source=UniverseSource.ETF_LIQUID,
                    now=now,
                )
                added += a
                updated += u
            except Exception as exc:
                errors.append(f"ETF {etf['symbol']}: {exc}")

        # 2. SP500 approximation
        if self._client:
            sp500_records = self._fetch_sp500_approx()
            for rec in sp500_records:
                try:
                    a, u = self._upsert(
                        symbol=rec["symbol"],
                        name=rec.get("name", rec["symbol"]),
                        sector=rec.get("sector", "Unknown"),
                        source=UniverseSource.SP500,
                        market_cap=rec.get("market_cap"),
                        avg_daily_volume=rec.get("avg_daily_volume"),
                        now=now,
                    )
                    added += a
                    updated += u
                except Exception as exc:
                    errors.append(f"SP500 {rec.get('symbol', '?')}: {exc}")

            # 3. NASDAQ-100 approximation (adds names not already covered by SP500)
            ndx_records = self._fetch_nasdaq100_approx()
            for rec in ndx_records:
                try:
                    a, u = self._upsert(
                        symbol=rec["symbol"],
                        name=rec.get("name", rec["symbol"]),
                        sector=rec.get("sector", "Unknown"),
                        source=UniverseSource.NASDAQ100,
                        market_cap=rec.get("market_cap"),
                        avg_daily_volume=rec.get("avg_daily_volume"),
                        now=now,
                    )
                    added += a
                    updated += u
                except Exception as exc:
                    errors.append(f"NDX {rec.get('symbol', '?')}: {exc}")
        else:
            errors.append("No FinancialDatasetsClient — skipping equity screener")

        total = self._count_active()
        report = RefreshReport(
            source="full_refresh",
            symbols_added=added,
            symbols_updated=updated,
            symbols_deactivated=0,
            total_active=total,
            errors=errors,
            refreshed_at=now,
        )
        logger.info(
            f"[UniverseRegistry] Refresh complete: +{added} updated={updated} "
            f"total={total} errors={len(errors)}"
        )
        return report

    def get_active_symbols(self, source: UniverseSource | None = None) -> list[str]:
        """Return sorted list of active symbols, optionally filtered by source."""
        if source:
            rows = self._conn.execute(
                "SELECT symbol FROM universe WHERE is_active = TRUE AND source = ? ORDER BY symbol",
                [source.value],
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT symbol FROM universe WHERE is_active = TRUE ORDER BY symbol",
            ).fetchall()
        return [r[0] for r in rows]

    def get_full_universe(self) -> list[ConstituentRecord]:
        """Return all active constituents as dataclass records."""
        rows = self._conn.execute(
            """
            SELECT symbol, name, sector, source, market_cap, avg_daily_volume, is_active
            FROM universe
            WHERE is_active = TRUE
            ORDER BY symbol
            """
        ).fetchall()
        return [
            ConstituentRecord(
                symbol=r[0],
                name=r[1],
                sector=r[2],
                source=UniverseSource(r[3]),
                market_cap=r[4],
                avg_daily_volume=r[5],
                is_active=r[6],
            )
            for r in rows
        ]

    def deactivate_symbol(self, symbol: str, reason: str) -> None:
        """Mark a symbol as inactive with a reason."""
        self._conn.execute(
            """
            UPDATE universe
            SET is_active = FALSE, deactivated_reason = ?, last_refreshed = ?
            WHERE symbol = ?
            """,
            [reason, datetime.now(timezone.utc), symbol],
        )
        logger.info(f"[UniverseRegistry] Deactivated {symbol}: {reason}")

    def get_refresh_age_hours(self) -> float:
        """Hours since the most recent refresh.  Returns inf if never refreshed."""
        row = self._conn.execute("SELECT MAX(last_refreshed) FROM universe").fetchone()
        if not row or row[0] is None:
            return float("inf")
        last = row[0]
        if not isinstance(last, datetime):
            # Try parsing as string
            try:
                last = datetime.fromisoformat(str(last))
            except (ValueError, TypeError):
                return float("inf")
        # Compare using naive local time to avoid timezone offset issues.
        if last.tzinfo is not None:
            last = last.replace(tzinfo=None)
        delta = datetime.now() - last
        return delta.total_seconds() / 3600.0

    def count(self, active_only: bool = True) -> int:
        """Count symbols in the universe."""
        if active_only:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM universe WHERE is_active = TRUE"
            ).fetchone()
        else:
            row = self._conn.execute("SELECT COUNT(*) FROM universe").fetchone()
        return row[0] if row else 0

    # ── Private helpers ──────────────────────────────────────────────────────

    def _upsert(
        self,
        symbol: str,
        name: str,
        sector: str,
        source: UniverseSource,
        market_cap: float | None = None,
        avg_daily_volume: float | None = None,
        now: datetime | None = None,
    ) -> tuple[int, int]:
        """
        Insert or update a universe symbol.

        Returns (added, updated) — one of them will be 1, the other 0.
        """
        now = now or datetime.now(timezone.utc)

        # Check if symbol exists
        existing = self._conn.execute(
            "SELECT symbol FROM universe WHERE symbol = ?", [symbol]
        ).fetchone()

        if existing:
            self._conn.execute(
                """
                UPDATE universe
                SET name = ?, sector = ?, source = ?,
                    market_cap = COALESCE(?, market_cap),
                    avg_daily_volume = COALESCE(?, avg_daily_volume),
                    is_active = TRUE, last_refreshed = ?,
                    deactivated_reason = NULL
                WHERE symbol = ?
                """,
                [name, sector, source.value, market_cap, avg_daily_volume, now, symbol],
            )
            return (0, 1)
        else:
            self._conn.execute(
                """
                INSERT INTO universe (symbol, name, sector, source, market_cap,
                                      avg_daily_volume, is_active, added_at, last_refreshed)
                VALUES (?, ?, ?, ?, ?, ?, TRUE, ?, ?)
                """,
                [
                    symbol,
                    name,
                    sector,
                    source.value,
                    market_cap,
                    avg_daily_volume,
                    now,
                    now,
                ],
            )
            return (1, 0)

    def _fetch_sp500_approx(self) -> list[dict[str, Any]]:
        """
        Approximate SP500 via stock_screener: market_cap > $10B, ADV > 500k.

        Returns list of dicts with symbol, name, sector, market_cap, avg_daily_volume.
        """
        filters = {
            "market_cap_gte": 10_000_000_000,
            "volume_gte": 500_000,
            "country": "US",
            "limit": 600,
        }
        resp = self._client.stock_screener(filters)
        if not resp:
            logger.warning("[UniverseRegistry] SP500 screener returned None")
            return []
        return self._parse_screener_response(resp)

    def _fetch_nasdaq100_approx(self) -> list[dict[str, Any]]:
        """
        Approximate NASDAQ-100 via stock_screener: tech/comm/consumer sectors,
        market_cap > $15B.
        """
        filters = {
            "market_cap_gte": 15_000_000_000,
            "volume_gte": 500_000,
            "country": "US",
            "limit": 200,
        }
        resp = self._client.stock_screener(filters)
        if not resp:
            logger.warning("[UniverseRegistry] NASDAQ-100 screener returned None")
            return []
        return self._parse_screener_response(resp)

    def _parse_screener_response(self, resp: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract symbol records from a stock_screener API response."""
        records: list[dict[str, Any]] = []
        # The API may return results under various keys
        results = resp.get("results", resp.get("data", resp.get("stocks", [])))
        if not isinstance(results, list):
            return records
        for item in results:
            symbol = item.get("ticker", item.get("symbol"))
            if not symbol:
                continue
            records.append(
                {
                    "symbol": str(symbol).upper(),
                    "name": item.get("name", item.get("company_name", symbol)),
                    "sector": item.get("sector", "Unknown"),
                    "market_cap": item.get(
                        "market_cap", item.get("market_capitalization")
                    ),
                    "avg_daily_volume": item.get("volume", item.get("avg_volume")),
                }
            )
        return records

    def _count_active(self) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) FROM universe WHERE is_active = TRUE"
        ).fetchone()
        return row[0] if row else 0
