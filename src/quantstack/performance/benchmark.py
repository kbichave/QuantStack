# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
BenchmarkTracker — daily benchmark ingestion + rolling comparison metrics.

Computes rolling Sharpe, Sortino, alpha, beta of the portfolio vs SPY (or any
benchmark) from the daily_equity and benchmark_daily tables.

Runs autonomously after EquityTracker.snapshot_daily() — same 16:05 ET job.

Usage:
    tracker = BenchmarkTracker(conn)
    tracker.update_benchmark("SPY")              # fetch + store today's SPY close
    tracker.compute_comparison("SPY", [30, 60, 90])  # rolling metrics
"""

from __future__ import annotations

import math
from datetime import date, timedelta
from typing import Any

import duckdb
from loguru import logger


class BenchmarkTracker:
    """
    Benchmark return ingestion and rolling comparison vs portfolio.

    Args:
        conn: DuckDB connection (write-enabled).
    """

    def __init__(self, conn: duckdb.DuckDBPyConnection) -> None:
        self._conn = conn

    def update_benchmark(
        self,
        benchmark: str = "SPY",
        as_of: date | None = None,
    ) -> dict[str, Any]:
        """
        Fetch and store the benchmark's daily close + return.

        Uses DataStore/FinancialDatasets to get the latest close price.
        Idempotent: skips if already stored for the date.

        Returns:
            Dict with benchmark data or {"skipped": True}.
        """
        snapshot_date = as_of or date.today()

        existing = self._conn.execute(
            "SELECT 1 FROM benchmark_daily WHERE date = ? AND benchmark = ?",
            [snapshot_date, benchmark],
        ).fetchone()
        if existing:
            return {"skipped": True, "date": str(snapshot_date), "benchmark": benchmark}

        # Get close price from DataStore
        close_price = self._fetch_close(benchmark, snapshot_date)
        if close_price is None:
            logger.warning(f"[Benchmark] No price for {benchmark} on {snapshot_date}")
            return {
                "error": "no_price",
                "benchmark": benchmark,
                "date": str(snapshot_date),
            }

        # Get previous close for daily return
        prev = self._conn.execute(
            "SELECT close_price FROM benchmark_daily WHERE benchmark = ? AND date < ? ORDER BY date DESC LIMIT 1",
            [benchmark, snapshot_date],
        ).fetchone()
        prev_close = float(prev[0]) if prev else None

        daily_return = (
            ((close_price - prev_close) / prev_close * 100) if prev_close else 0.0
        )

        # Cumulative return from first entry
        first = self._conn.execute(
            "SELECT close_price FROM benchmark_daily WHERE benchmark = ? ORDER BY date ASC LIMIT 1",
            [benchmark],
        ).fetchone()
        first_close = float(first[0]) if first else close_price
        cumulative = (
            ((close_price - first_close) / first_close * 100)
            if first_close > 0
            else 0.0
        )

        self._conn.execute(
            """
            INSERT INTO benchmark_daily (date, benchmark, close_price, daily_return_pct, cumulative_return)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                snapshot_date,
                benchmark,
                round(close_price, 4),
                round(daily_return, 4),
                round(cumulative, 4),
            ],
        )

        logger.debug(
            f"[Benchmark] {benchmark} {snapshot_date}: ${close_price:.2f} ({daily_return:+.2f}%)"
        )
        return {
            "date": str(snapshot_date),
            "benchmark": benchmark,
            "close_price": round(close_price, 4),
            "daily_return_pct": round(daily_return, 4),
        }

    def compute_comparison(
        self,
        benchmark: str = "SPY",
        windows: list[int] | None = None,
        as_of: date | None = None,
    ) -> list[dict[str, Any]]:
        """
        Compute rolling Sharpe, Sortino, alpha, beta for each window.

        Writes results to benchmark_comparison table.

        Args:
            benchmark: Benchmark ticker.
            windows: Rolling windows in trading days. Default: [30, 60, 90].
            as_of: Computation date. Default: today.

        Returns:
            List of comparison dicts per window.
        """
        snapshot_date = as_of or date.today()
        windows = windows or [30, 60, 90]
        results = []

        for window in windows:
            comp = self._compute_window(benchmark, window, snapshot_date)
            if comp is None:
                continue

            # Persist (upsert)
            self._conn.execute(
                """
                INSERT INTO benchmark_comparison
                    (date, benchmark, window_days, portfolio_sharpe, benchmark_sharpe,
                     portfolio_sortino, alpha, beta)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (date, benchmark, window_days) DO UPDATE SET
                    portfolio_sharpe = excluded.portfolio_sharpe,
                    benchmark_sharpe = excluded.benchmark_sharpe,
                    portfolio_sortino = excluded.portfolio_sortino,
                    alpha = excluded.alpha,
                    beta = excluded.beta
                """,
                [
                    snapshot_date,
                    benchmark,
                    window,
                    comp["portfolio_sharpe"],
                    comp["benchmark_sharpe"],
                    comp["portfolio_sortino"],
                    comp["alpha"],
                    comp["beta"],
                ],
            )
            results.append(comp)

        if results:
            logger.info(
                f"[Benchmark] {benchmark} comparison: "
                + " | ".join(
                    f"{r['window']}d Sharpe={r['portfolio_sharpe']:.2f} α={r['alpha']:.2f}"
                    for r in results
                )
            )
        return results

    def get_comparison(
        self,
        benchmark: str = "SPY",
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> list[dict[str, Any]]:
        """Query stored benchmark comparisons."""
        query = "SELECT * FROM benchmark_comparison WHERE benchmark = ?"
        params: list[Any] = [benchmark]

        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)

        query += " ORDER BY date, window_days"
        rows = self._conn.execute(query, params).fetchall()
        columns = [
            "date",
            "benchmark",
            "window_days",
            "portfolio_sharpe",
            "benchmark_sharpe",
            "portfolio_sortino",
            "alpha",
            "beta",
        ]
        return [dict(zip(columns, row)) for row in rows]

    # ── Internal ──────────────────────────────────────────────────────────

    def _compute_window(
        self,
        benchmark: str,
        window: int,
        as_of: date,
    ) -> dict[str, Any] | None:
        """Compute comparison metrics for a single rolling window."""
        start = as_of - timedelta(days=int(window * 1.6))  # calendar days buffer

        # Portfolio daily returns
        port_rows = self._conn.execute(
            "SELECT date, daily_return_pct FROM daily_equity WHERE date >= ? AND date <= ? ORDER BY date",
            [start, as_of],
        ).fetchall()

        # Benchmark daily returns
        bench_rows = self._conn.execute(
            "SELECT date, daily_return_pct FROM benchmark_daily WHERE benchmark = ? AND date >= ? AND date <= ? ORDER BY date",
            [benchmark, start, as_of],
        ).fetchall()

        if len(port_rows) < min(window, 10) or len(bench_rows) < min(window, 10):
            return None

        # Align dates
        port_dict = {r[0]: r[1] for r in port_rows}
        bench_dict = {r[0]: r[1] for r in bench_rows}
        common_dates = sorted(set(port_dict) & set(bench_dict))[-window:]

        if len(common_dates) < 10:
            return None

        port_rets = [port_dict[d] for d in common_dates]
        bench_rets = [bench_dict[d] for d in common_dates]
        n = len(port_rets)

        # Portfolio Sharpe
        port_mean = sum(port_rets) / n
        port_std = _std(port_rets)
        port_sharpe = (port_mean / port_std * math.sqrt(252)) if port_std > 0 else 0

        # Benchmark Sharpe
        bench_mean = sum(bench_rets) / n
        bench_std = _std(bench_rets)
        bench_sharpe = (bench_mean / bench_std * math.sqrt(252)) if bench_std > 0 else 0

        # Portfolio Sortino
        port_down = [r for r in port_rets if r < 0]
        port_down_std = (sum(r**2 for r in port_down) / max(len(port_down), 1)) ** 0.5
        port_sortino = (
            (port_mean / port_down_std * math.sqrt(252)) if port_down_std > 0 else 0
        )

        # Beta = Cov(port, bench) / Var(bench)
        cov = sum(
            (p - port_mean) * (b - bench_mean) for p, b in zip(port_rets, bench_rets)
        ) / max(n - 1, 1)
        bench_var = bench_std**2
        beta = (cov / bench_var) if bench_var > 0 else 0

        # Alpha = annualized(port_mean - beta * bench_mean)
        alpha = (port_mean - beta * bench_mean) * 252

        return {
            "window": window,
            "portfolio_sharpe": round(port_sharpe, 3),
            "benchmark_sharpe": round(bench_sharpe, 3),
            "portfolio_sortino": round(port_sortino, 3),
            "alpha": round(alpha, 3),
            "beta": round(beta, 3),
            "n_days": n,
        }

    def _fetch_close(self, symbol: str, as_of: date) -> float | None:
        """Fetch the close price for a symbol on a given date from DataStore."""
        try:
            from quantstack.config.timeframes import Timeframe
            from quantstack.data.storage import DataStore

            store = DataStore(read_only=True)
            df = store.load_ohlcv(symbol, Timeframe.D1)
            if df is None or df.empty:
                return None

            # Find closest date <= as_of
            mask = df.index <= str(as_of)
            if not mask.any():
                return None
            return float(df.loc[mask, "close"].iloc[-1])
        except Exception as exc:
            logger.debug(f"[Benchmark] Failed to fetch {symbol} close: {exc}")
            return None


def _std(values: list[float]) -> float:
    """Sample standard deviation."""
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    return (sum((v - mean) ** 2 for v in values) / (n - 1)) ** 0.5
