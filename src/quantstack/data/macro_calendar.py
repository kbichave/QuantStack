# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
MacroCalendarGenerator — populates EventCalendar with FOMC, CPI, NFP dates.

Two data sources:
1. **Hardcoded rules** for dates (FOMC publishes schedule years ahead,
   CPI is ~2nd Tuesday/Wednesday, NFP is first Friday). Zero API cost.
2. **Alpha Vantage economic indicators** for actual vs prior values
   (CPI reading, Fed Funds Rate, unemployment). Requires API key.

The calendar dates are generated programmatically. The actual values
are fetched from Alpha Vantage and stored in DuckDB for the events
collector and research pods to use.

Usage:
    from quantstack.data.macro_calendar import MacroCalendarGenerator

    gen = MacroCalendarGenerator()
    calendar = gen.build_calendar(2022, 2026)
    # calendar is an EventCalendar with ~200 events

    # Enrich with actual values from Alpha Vantage
    gen.fetch_economic_history(conn)
"""

from __future__ import annotations

import os
from datetime import date, datetime, time, timedelta
from typing import Any

from loguru import logger

from quantstack.data.adapters.alphavantage import AlphaVantageAdapter
from quantstack.shared.event_calendar import EventCalendar


# =============================================================================
# FOMC MEETING DATES (published by the Fed)
# Source: https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
# =============================================================================

# Announcement dates (day 2 of 2-day meetings, or single-day meetings)
_FOMC_DATES: dict[int, list[str]] = {
    2020: [
        "2020-01-29",
        "2020-03-03",
        "2020-03-15",
        "2020-04-29",
        "2020-06-10",
        "2020-07-29",
        "2020-09-16",
        "2020-11-05",
        "2020-12-16",
    ],
    2021: [
        "2021-01-27",
        "2021-03-17",
        "2021-04-28",
        "2021-06-16",
        "2021-07-28",
        "2021-09-22",
        "2021-11-03",
        "2021-12-15",
    ],
    2022: [
        "2022-01-26",
        "2022-03-16",
        "2022-05-04",
        "2022-06-15",
        "2022-07-27",
        "2022-09-21",
        "2022-11-02",
        "2022-12-14",
    ],
    2023: [
        "2023-02-01",
        "2023-03-22",
        "2023-05-03",
        "2023-06-14",
        "2023-07-26",
        "2023-09-20",
        "2023-11-01",
        "2023-12-13",
    ],
    2024: [
        "2024-01-31",
        "2024-03-20",
        "2024-05-01",
        "2024-06-12",
        "2024-07-31",
        "2024-09-18",
        "2024-11-07",
        "2024-12-18",
    ],
    2025: [
        "2025-01-29",
        "2025-03-19",
        "2025-05-07",
        "2025-06-18",
        "2025-07-30",
        "2025-09-17",
        "2025-10-29",
        "2025-12-17",
    ],
    2026: [
        "2026-01-28",
        "2026-03-18",
        "2026-04-29",
        "2026-06-17",
        "2026-07-29",
        "2026-09-16",
        "2026-11-04",
        "2026-12-16",
    ],
}


def _generate_cpi_dates(year: int) -> list[date]:
    """
    CPI is released ~2nd or 3rd Tuesday/Wednesday of each month at 8:30 AM ET.

    Approximation: 12th-15th of each month (actual dates vary slightly).
    For production, these should be verified against BLS schedule.
    """
    dates = []
    for month in range(1, 13):
        # Start from the 10th, find the first Tuesday or Wednesday
        for day in range(10, 16):
            try:
                d = date(year, month, day)
                if d.weekday() in (1, 2):  # Tuesday or Wednesday
                    dates.append(d)
                    break
            except ValueError:
                continue
    return dates


def _generate_nfp_dates(year: int) -> list[date]:
    """
    NFP is released first Friday of each month at 8:30 AM ET.
    """
    dates = []
    for month in range(1, 13):
        d = date(year, month, 1)
        # Find first Friday (weekday 4)
        days_until_friday = (4 - d.weekday()) % 7
        first_friday = d + timedelta(days=days_until_friday)
        dates.append(first_friday)
    return dates


def _generate_gdp_dates(year: int) -> list[date]:
    """
    GDP advance estimate: last week of January, April, July, October.
    Approximation: 28th of those months.
    """
    dates = []
    for month in [1, 4, 7, 10]:
        try:
            d = date(year, month, 28)
            # Adjust to weekday
            while d.weekday() >= 5:
                d -= timedelta(days=1)
            dates.append(d)
        except ValueError:
            pass
    return dates


class MacroCalendarGenerator:
    """
    Generate macro event calendar from hardcoded rules + Alpha Vantage data.
    """

    def build_calendar(
        self,
        start_year: int = 2020,
        end_year: int = 2026,
    ) -> "EventCalendar":
        """
        Build an EventCalendar populated with FOMC, CPI, NFP, GDP dates.

        Returns an EventCalendar instance ready to use for blackout checks.
        """
        calendar = EventCalendar()

        for year in range(start_year, end_year + 1):
            # FOMC
            fomc_dates = _FOMC_DATES.get(year, [])
            fomc_datetimes = []
            for ds in fomc_dates:
                try:
                    fomc_datetimes.append(datetime.fromisoformat(ds))
                except ValueError:
                    pass
            calendar.add_fomc_dates(fomc_datetimes)

            # CPI
            cpi_dates = _generate_cpi_dates(year)
            calendar.add_cpi_dates(
                [datetime.combine(d, time(8, 30)) for d in cpi_dates]
            )

            # NFP
            nfp_dates = _generate_nfp_dates(year)
            calendar.add_nfp_dates(
                [datetime.combine(d, time(8, 30)) for d in nfp_dates]
            )

            # OPEX (options expiration — third Friday)
            calendar.add_opex_dates(year)

        event_count = len(calendar._events)
        logger.info(
            f"[MacroCalendar] Built calendar {start_year}-{end_year}: "
            f"{event_count} events"
        )
        return calendar

    def fetch_economic_history(
        self,
        conn: Any,
        indicators: list[str] | None = None,
    ) -> dict[str, int]:
        """
        Fetch historical economic indicator values from Alpha Vantage
        and store in DuckDB for the research pods.

        Args:
            conn: DuckDB connection.
            indicators: Which indicators to fetch. Default: all major ones.

        Returns:
            Dict of indicator → rows fetched.
        """
        indicators = indicators or [
            "CPI",
            "FEDERAL_FUNDS_RATE",
            "UNEMPLOYMENT",
            "NONFARM_PAYROLL",
            "REAL_GDP",
            "INFLATION",
            "RETAIL_SALES",
        ]

        api_key = os.environ.get("ALPHA_VANTAGE_API_KEY", "")
        if not api_key:
            logger.warning(
                "[MacroCalendar] No ALPHA_VANTAGE_API_KEY — skipping economic fetch"
            )
            return {}

        adapter = AlphaVantageAdapter(api_key=api_key)
        results: dict[str, int] = {}

        # Ensure table exists
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS economic_indicators (
                indicator VARCHAR NOT NULL,
                date DATE NOT NULL,
                value DOUBLE,
                PRIMARY KEY (indicator, date)
            )
        """
        )

        for indicator in indicators:
            try:
                df = adapter.fetch_economic(indicator, interval="monthly")
                if df.empty:
                    results[indicator] = 0
                    continue

                rows_inserted = 0
                for idx, row in df.iterrows():
                    try:
                        conn.execute(
                            """
                            INSERT INTO economic_indicators (indicator, date, value)
                            VALUES (?, ?, ?)
                            ON CONFLICT (indicator, date) DO NOTHING
                            """,
                            [
                                indicator,
                                idx.date() if hasattr(idx, "date") else idx,
                                row["value"],
                            ],
                        )
                        rows_inserted += 1
                    except Exception:
                        pass

                results[indicator] = rows_inserted
                logger.debug(
                    f"[MacroCalendar] {indicator}: {rows_inserted} rows stored"
                )

            except Exception as exc:
                logger.warning(f"[MacroCalendar] Failed to fetch {indicator}: {exc}")
                results[indicator] = 0

        logger.info(
            f"[MacroCalendar] Economic history: "
            + ", ".join(f"{k}={v}" for k, v in results.items())
        )
        return results

    def get_fomc_dates(self, year: int) -> list[date]:
        """Get FOMC announcement dates for a year."""
        return [date.fromisoformat(d) for d in _FOMC_DATES.get(year, [])]

    def get_cpi_dates(self, year: int) -> list[date]:
        """Get approximate CPI release dates for a year."""
        return _generate_cpi_dates(year)

    def get_nfp_dates(self, year: int) -> list[date]:
        """Get NFP release dates for a year."""
        return _generate_nfp_dates(year)
