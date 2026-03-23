# Copyright 2024 QuantCore Contributors
# SPDX-License-Identifier: Apache-2.0

"""
TCA Storage — persist pre-trade forecasts and post-trade results to DuckDB.

Enables /reflect to analyze execution quality trends:
- Average slippage over time
- Algo recommendation accuracy
- Worst fills by symbol/time-of-day

Usage:
    from quantstack.core.execution.tca_storage import TCAStore

    store = TCAStore()
    store.save_forecast(forecast)
    store.save_result(result)
    stats = store.get_aggregate_stats(lookback_days=30)
    store.close()
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
from loguru import logger

from quantstack.core.execution.tca_engine import (
    OrderSide,
    PreTradeForecast,
    TradeTCAResult,
)

# ---------------------------------------------------------------------------
# Default DB path
# ---------------------------------------------------------------------------

_DEFAULT_DB_DIR = Path.home() / ".quant_pod"
_DEFAULT_DB_PATH = _DEFAULT_DB_DIR / "tca.duckdb"


# ---------------------------------------------------------------------------
# TCAStore
# ---------------------------------------------------------------------------


class TCAStore:
    """DuckDB persistence for TCA forecasts and post-trade results.

    Args:
        db_path: Path to DuckDB file. Defaults to ``~/.quant_pod/tca.duckdb``.
                 Pass ``:memory:`` for in-memory (tests).
    """

    def __init__(self, db_path: str | None = None) -> None:
        resolved_path = db_path or str(_DEFAULT_DB_PATH)

        # Ensure parent directory exists for file-backed databases
        if resolved_path != ":memory:":
            parent = Path(resolved_path).parent
            parent.mkdir(parents=True, exist_ok=True)

        self._conn = duckdb.connect(resolved_path)
        self._ensure_tables()
        logger.debug(f"[TCAStore] Opened at {resolved_path}")

    def _ensure_tables(self) -> None:
        """Create tables if they do not exist."""
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tca_forecasts (
                trade_id        VARCHAR PRIMARY KEY,
                symbol          VARCHAR NOT NULL,
                side            VARCHAR NOT NULL,
                shares          DOUBLE NOT NULL,
                arrival_price   DOUBLE NOT NULL,
                spread_bps      DOUBLE,
                impact_bps      DOUBLE,
                total_bps       DOUBLE,
                recommended_algo VARCHAR,
                timestamp       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tca_results (
                trade_id                    VARCHAR PRIMARY KEY,
                symbol                      VARCHAR NOT NULL,
                side                        VARCHAR NOT NULL,
                shares                      DOUBLE,
                fill_price                  DOUBLE NOT NULL,
                arrival_price               DOUBLE NOT NULL,
                shortfall_vs_arrival_bps    DOUBLE,
                shortfall_vs_vwap_bps       DOUBLE,
                shortfall_dollar            DOUBLE,
                is_favorable                BOOLEAN,
                timestamp                   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

    # ── Write methods ────────────────────────────────────────────────────────

    def save_forecast(self, forecast: PreTradeForecast) -> str:
        """Persist a pre-trade forecast.

        Args:
            forecast: PreTradeForecast from tca_engine.

        Returns:
            The trade_id used as the primary key (derived from symbol + timestamp
            if the forecast doesn't carry one).
        """
        trade_id = f"{forecast.symbol}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

        self._conn.execute(
            """
            INSERT INTO tca_forecasts
                (trade_id, symbol, side, shares, arrival_price,
                 spread_bps, impact_bps, total_bps, recommended_algo, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                trade_id,
                forecast.symbol,
                forecast.side.value,
                forecast.shares,
                forecast.arrival_price,
                forecast.spread_cost_bps,
                forecast.market_impact_bps,
                forecast.total_expected_bps,
                forecast.recommended_algo.value,
                datetime.now(timezone.utc),
            ],
        )
        logger.debug(f"[TCAStore] Saved forecast: {trade_id}")
        return trade_id

    def save_result(self, result: TradeTCAResult) -> None:
        """Persist a post-trade TCA result.

        Args:
            result: TradeTCAResult from tca_engine.post_trade_tca().
        """
        self._conn.execute(
            """
            INSERT OR REPLACE INTO tca_results
                (trade_id, symbol, side, shares, fill_price, arrival_price,
                 shortfall_vs_arrival_bps, shortfall_vs_vwap_bps,
                 shortfall_dollar, is_favorable, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                result.trade_id,
                result.symbol,
                result.side.value,
                None,  # shares not on TradeTCAResult — populated from forecast join
                0.0,  # fill_price not stored on result — computed from shortfall
                0.0,  # arrival_price same
                result.shortfall_vs_arrival_bps,
                result.shortfall_vs_vwap_bps,
                result.shortfall_dollar,
                result.is_favorable,
                datetime.now(timezone.utc),
            ],
        )
        logger.debug(f"[TCAStore] Saved result: {result.trade_id}")

    def save_result_raw(
        self,
        trade_id: str,
        symbol: str,
        side: str,
        shares: float,
        fill_price: float,
        arrival_price: float,
        shortfall_vs_arrival_bps: float,
        shortfall_vs_vwap_bps: float | None = None,
        shortfall_dollar: float = 0.0,
        is_favorable: bool = False,
    ) -> None:
        """Persist a post-trade result from raw values (no TradeTCAResult needed).

        Useful when the caller computes shortfall externally (e.g. from broker fills).
        """
        self._conn.execute(
            """
            INSERT OR REPLACE INTO tca_results
                (trade_id, symbol, side, shares, fill_price, arrival_price,
                 shortfall_vs_arrival_bps, shortfall_vs_vwap_bps,
                 shortfall_dollar, is_favorable, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                trade_id,
                symbol,
                side,
                shares,
                fill_price,
                arrival_price,
                shortfall_vs_arrival_bps,
                shortfall_vs_vwap_bps,
                shortfall_dollar,
                is_favorable,
                datetime.now(timezone.utc),
            ],
        )

    # ── Read methods ─────────────────────────────────────────────────────────

    def get_recent_results(
        self, limit: int = 50, symbol: str | None = None
    ) -> list[dict[str, Any]]:
        """Query recent post-trade TCA results.

        Args:
            limit: Maximum rows to return.
            symbol: Optional symbol filter.

        Returns:
            List of dicts with result columns.
        """
        query = "SELECT * FROM tca_results"
        params: list[Any] = []

        if symbol:
            query += " WHERE symbol = ?"
            params.append(symbol.upper())

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        rows = self._conn.execute(query, params).fetchall()
        columns = [desc[0] for desc in self._conn.description]
        return [dict(zip(columns, row)) for row in rows]

    def get_aggregate_stats(
        self, lookback_days: int = 30, symbol: str | None = None
    ) -> dict[str, Any]:
        """Compute aggregate TCA statistics for /reflect analysis.

        Args:
            lookback_days: Number of days to look back.
            symbol: Optional symbol filter.

        Returns:
            Dict with avg_slippage_bps, worst_fill, algo_accuracy, trade_count, etc.
        """
        where_clauses = ["timestamp >= CURRENT_TIMESTAMP - INTERVAL ? DAY"]
        params: list[Any] = [lookback_days]

        if symbol:
            where_clauses.append("symbol = ?")
            params.append(symbol.upper())

        where_sql = " AND ".join(where_clauses)

        # Aggregate result stats
        result_stats = self._conn.execute(
            f"""
            SELECT
                COUNT(*) AS trade_count,
                AVG(shortfall_vs_arrival_bps) AS avg_slippage_bps,
                MEDIAN(shortfall_vs_arrival_bps) AS median_slippage_bps,
                MAX(shortfall_vs_arrival_bps) AS worst_slippage_bps,
                MIN(shortfall_vs_arrival_bps) AS best_slippage_bps,
                SUM(shortfall_dollar) AS total_shortfall_dollar,
                AVG(CASE WHEN is_favorable THEN 1.0 ELSE 0.0 END) * 100
                    AS favorable_pct,
                AVG(shortfall_vs_vwap_bps) AS avg_vs_vwap_bps
            FROM tca_results
            WHERE {where_sql}
            """,
            params,
        ).fetchone()

        if result_stats is None or result_stats[0] == 0:
            return {
                "trade_count": 0,
                "lookback_days": lookback_days,
                "symbol_filter": symbol,
                "message": "No TCA results in the lookback window.",
            }

        (
            trade_count,
            avg_slip,
            median_slip,
            worst_slip,
            best_slip,
            total_dollar,
            fav_pct,
            avg_vwap,
        ) = result_stats

        # Worst fills (top 5)
        worst_fills = self._conn.execute(
            f"""
            SELECT trade_id, symbol, side, shortfall_vs_arrival_bps, timestamp
            FROM tca_results
            WHERE {where_sql}
            ORDER BY shortfall_vs_arrival_bps DESC
            LIMIT 5
            """,
            params,
        ).fetchall()

        worst_fill_list = [
            {
                "trade_id": r[0],
                "symbol": r[1],
                "side": r[2],
                "shortfall_bps": round(r[3], 2) if r[3] is not None else None,
                "timestamp": str(r[4]) if r[4] else None,
            }
            for r in worst_fills
        ]

        # Algo accuracy: compare forecast algo vs actual performance
        algo_stats = self._conn.execute(
            f"""
            SELECT
                f.recommended_algo,
                COUNT(*) AS count,
                AVG(r.shortfall_vs_arrival_bps) AS avg_shortfall_bps
            FROM tca_forecasts f
            JOIN tca_results r ON f.trade_id = r.trade_id
            WHERE r.{where_sql.replace('timestamp', 'r.timestamp').replace('symbol', 'r.symbol')}
            GROUP BY f.recommended_algo
            ORDER BY count DESC
            """,
            params,
        ).fetchall()

        algo_breakdown = [
            {
                "algo": r[0],
                "count": r[1],
                "avg_shortfall_bps": round(r[2], 2) if r[2] is not None else None,
            }
            for r in algo_stats
        ]

        # Execution quality verdict
        avg_slip_val = avg_slip or 0.0
        if avg_slip_val < 0:
            quality = "EXCELLENT"
        elif avg_slip_val < 5:
            quality = "GOOD"
        elif avg_slip_val < 15:
            quality = "ACCEPTABLE"
        else:
            quality = "POOR"

        return {
            "trade_count": trade_count,
            "lookback_days": lookback_days,
            "symbol_filter": symbol,
            "avg_slippage_bps": round(avg_slip_val, 2),
            "median_slippage_bps": (
                round(median_slip, 2) if median_slip is not None else None
            ),
            "worst_slippage_bps": (
                round(worst_slip, 2) if worst_slip is not None else None
            ),
            "best_slippage_bps": round(best_slip, 2) if best_slip is not None else None,
            "total_shortfall_dollar": (
                round(total_dollar, 2) if total_dollar is not None else 0.0
            ),
            "favorable_pct": round(fav_pct, 1) if fav_pct is not None else 0.0,
            "avg_vs_vwap_bps": round(avg_vwap, 2) if avg_vwap is not None else None,
            "execution_quality": quality,
            "worst_fills": worst_fill_list,
            "algo_breakdown": algo_breakdown,
        }

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def close(self) -> None:
        """Close the DuckDB connection."""
        try:
            self._conn.close()
            logger.debug("[TCAStore] Connection closed")
        except Exception:
            pass

    def __enter__(self) -> TCAStore:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
