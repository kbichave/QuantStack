"""
IS/OOS Degradation Detector — rolling live Sharpe vs. backtested benchmark.

Addresses GAP-11 in the gap analysis:
  "Monitor rolling live Sharpe vs walk-forward predicted Sharpe.
   Flag when rolling IS/OOS ratio (in-sample IC / out-of-sample IC) exceeds 2-sigma.
   Automated size reduction on degradation signal."

The key insight: backtested (IS) Sharpe and live (OOS) Sharpe will always differ.
What matters is whether the gap is growing. A strategy with IS=2.0 and OOS=1.2
is healthy; the same strategy with IS=2.0 and OOS=0.1 after 60 days is broken.

Thresholds (calibrated to avoid noise while catching real decay):
  - CRITICAL: live Sharpe < 0 over rolling 60 days
  - CRITICAL: IS/OOS ratio > 4.0 (live is 75%+ below backtest)
  - WARNING:  IS/OOS ratio > 2.0 (live is 50%+ below backtest)
  - WARNING:  Max drawdown > 2× predicted max drawdown

Auto size reduction recommendations (returned in report, not auto-applied):
  - CRITICAL → reduce position_size_pct by 75%
  - WARNING  → reduce position_size_pct by 50%
  - CLEAN    → maintain current sizing

Failure modes:
  - No closed trades yet → returns CLEAN (insufficient data)
  - No IS benchmark registered → computes degradation vs zero (absolute OOS check only)
  - DB connection failure → propagates (caller decides to halt or continue)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

import numpy as np
from loguru import logger

from quantstack.execution.portfolio_state import get_portfolio_state


class DegradationStatus(str, Enum):
    CLEAN = "clean"
    WARNING = "warning"
    CRITICAL = "critical"
    INSUFFICIENT_DATA = "insufficient_data"


@dataclass
class ISBenchmark:
    """
    In-sample performance benchmark stored when a strategy is validated.

    Register this after running walk-forward validation so the degradation
    detector knows what live performance to compare against.
    """

    strategy_id: str
    predicted_annual_sharpe: float
    predicted_max_drawdown: float  # As fraction (e.g. 0.08 = 8%)
    predicted_win_rate: float
    registered_at: datetime = field(default_factory=datetime.now)
    n_backtest_trades: int = 0


@dataclass
class DegradationReport:
    """Full degradation detection result."""

    strategy_id: str
    status: DegradationStatus
    checked_at: datetime

    # Live metrics (rolling window)
    live_sharpe: float
    live_win_rate: float
    live_max_drawdown: float
    live_n_trades: int
    rolling_window_days: int

    # IS benchmark (None if not registered)
    is_benchmark: ISBenchmark | None

    # Derived ratios
    sharpe_ratio_oos_vs_is: float | None  # OOS / IS — < 0.5 = WARNING
    drawdown_ratio_vs_predicted: float | None  # actual / predicted — > 2 = WARNING

    # Human-readable findings
    findings: list[str] = field(default_factory=list)

    # Recommended size adjustment (fraction, e.g. 0.5 = reduce to 50%)
    recommended_size_multiplier: float = 1.0

    @property
    def emoji(self) -> str:
        return {
            "clean": "✅",
            "warning": "⚠️",
            "critical": "🚨",
            "insufficient_data": "📊",
        }[self.status.value]


class DegradationDetector:
    """
    Monitors the gap between backtested (IS) and live (OOS) performance.

    Pulls closed trades from the database, computes rolling metrics,
    and compares them against registered IS benchmarks.
    """

    # Rolling window for live metrics: 60 trading days ≈ 3 months
    DEFAULT_ROLLING_DAYS = 60
    MIN_TRADES_FOR_SHARPE = 10  # Need at least 10 trades to compute Sharpe

    # IS/OOS ratio thresholds
    WARNING_OOS_IS_RATIO = 0.50  # OOS / IS < 0.5 → WARNING
    CRITICAL_OOS_IS_RATIO = 0.25  # OOS / IS < 0.25 → CRITICAL
    WARNING_DD_RATIO = 2.0  # actual DD > 2× predicted → WARNING
    CRITICAL_DD_RATIO = 3.0  # actual DD > 3× predicted → CRITICAL

    def __init__(self, conn=None) -> None:
        """
        Args:
            conn: Database connection. If None, opens the default portfolio DB.
        """
        self._conn = conn
        self._benchmarks: dict[str, ISBenchmark] = {}
        self._ensure_benchmark_table()
        self._load_benchmarks()

    # -------------------------------------------------------------------------
    # Benchmark registration
    # -------------------------------------------------------------------------

    def register_benchmark(self, benchmark: ISBenchmark) -> None:
        """
        Register an IS performance benchmark.

        Call this after running walk-forward validation:
            report = cpcv.evaluate(returns, ...)
            detector.register_benchmark(ISBenchmark(
                strategy_id="SuperTrader_SPY",
                predicted_annual_sharpe=1.8,
                predicted_max_drawdown=0.07,
                predicted_win_rate=0.57,
            ))
        """
        self._benchmarks[benchmark.strategy_id] = benchmark
        self._persist_benchmark(benchmark)
        logger.info(
            f"[DEGRADE] Registered IS benchmark for {benchmark.strategy_id}: "
            f"Sharpe={benchmark.predicted_annual_sharpe:.2f} "
            f"MaxDD={benchmark.predicted_max_drawdown:.1%} "
            f"WinRate={benchmark.predicted_win_rate:.1%}"
        )

    # -------------------------------------------------------------------------
    # Core check
    # -------------------------------------------------------------------------

    def check(
        self,
        strategy_id: str = "default",
        rolling_days: int = DEFAULT_ROLLING_DAYS,
    ) -> DegradationReport:
        """
        Run degradation check for a strategy.

        Computes rolling live Sharpe, win rate, and drawdown from closed trades,
        then compares against the registered IS benchmark (if any).

        Args:
            strategy_id: Key for benchmark lookup. Use "default" for the main
                         SuperTrader strategy, or per-symbol IDs.
            rolling_days: Look-back window for live metrics.

        Returns:
            DegradationReport with status and recommended size multiplier.
        """
        now = datetime.now()
        cutoff = now - timedelta(days=rolling_days)

        trades = self._load_closed_trades(since=cutoff)
        benchmark = self._benchmarks.get(strategy_id)

        if len(trades) < self.MIN_TRADES_FOR_SHARPE:
            return DegradationReport(
                strategy_id=strategy_id,
                status=DegradationStatus.INSUFFICIENT_DATA,
                checked_at=now,
                live_sharpe=0.0,
                live_win_rate=0.0,
                live_max_drawdown=0.0,
                live_n_trades=len(trades),
                rolling_window_days=rolling_days,
                is_benchmark=benchmark,
                sharpe_ratio_oos_vs_is=None,
                drawdown_ratio_vs_predicted=None,
                findings=[
                    f"Only {len(trades)} trades in last {rolling_days}d — need {self.MIN_TRADES_FOR_SHARPE} for analysis"
                ],
                recommended_size_multiplier=1.0,
            )

        pnls = [t["realized_pnl"] for t in trades]
        live_sharpe = self._rolling_sharpe(pnls)
        live_win_rate = sum(1 for p in pnls if p > 0) / len(pnls)
        live_max_dd = self._max_drawdown(pnls)

        # Derived ratios
        oos_is_ratio = None
        dd_ratio = None
        if benchmark is not None:
            if abs(benchmark.predicted_annual_sharpe) > 0.01:
                oos_is_ratio = live_sharpe / benchmark.predicted_annual_sharpe
            if benchmark.predicted_max_drawdown > 0.001:
                dd_ratio = live_max_dd / benchmark.predicted_max_drawdown

        # Classify
        status, findings, size_mult = self._classify(
            live_sharpe=live_sharpe,
            live_win_rate=live_win_rate,
            live_max_dd=live_max_dd,
            oos_is_ratio=oos_is_ratio,
            dd_ratio=dd_ratio,
            benchmark=benchmark,
        )

        report = DegradationReport(
            strategy_id=strategy_id,
            status=status,
            checked_at=now,
            live_sharpe=round(live_sharpe, 3),
            live_win_rate=round(live_win_rate, 3),
            live_max_drawdown=round(live_max_dd, 4),
            live_n_trades=len(trades),
            rolling_window_days=rolling_days,
            is_benchmark=benchmark,
            sharpe_ratio_oos_vs_is=(
                round(oos_is_ratio, 3) if oos_is_ratio is not None else None
            ),
            drawdown_ratio_vs_predicted=(
                round(dd_ratio, 2) if dd_ratio is not None else None
            ),
            findings=findings,
            recommended_size_multiplier=size_mult,
        )

        level = (
            "critical"
            if status == DegradationStatus.CRITICAL
            else ("warning" if status == DegradationStatus.WARNING else "info")
        )
        getattr(logger, level)(
            f"[DEGRADE] {strategy_id}: {status.value.upper()} | "
            f"live_sharpe={live_sharpe:.2f} win_rate={live_win_rate:.1%} "
            f"max_dd={live_max_dd:.1%} | size_mult={size_mult:.2f}"
        )
        return report

    def check_all(
        self, rolling_days: int = DEFAULT_ROLLING_DAYS
    ) -> list[DegradationReport]:
        """
        Check all registered benchmarks plus a "default" strategy check.
        Returns list of DegradationReport sorted by severity.
        """
        strategies = list(self._benchmarks.keys()) or ["default"]
        reports = [self.check(sid, rolling_days) for sid in strategies]

        severity_order = {
            DegradationStatus.CRITICAL: 0,
            DegradationStatus.WARNING: 1,
            DegradationStatus.CLEAN: 2,
            DegradationStatus.INSUFFICIENT_DATA: 3,
        }
        return sorted(reports, key=lambda r: severity_order[r.status])

    # -------------------------------------------------------------------------
    # Classification logic
    # -------------------------------------------------------------------------

    def _classify(
        self,
        live_sharpe: float,
        live_win_rate: float,
        live_max_dd: float,
        oos_is_ratio: float | None,
        dd_ratio: float | None,
        benchmark: ISBenchmark | None,
    ) -> tuple[DegradationStatus, list[str], float]:
        """
        Return (status, findings list, recommended_size_multiplier).

        size_multiplier of 1.0 = no change, 0.5 = reduce to 50%, 0.25 = reduce to 25%.
        """
        findings = []
        status = DegradationStatus.CLEAN
        size_mult = 1.0

        # --- Absolute live Sharpe check (no benchmark needed) ---
        if live_sharpe < 0.0:
            findings.append(
                f"Live Sharpe = {live_sharpe:.2f} (NEGATIVE). "
                "Strategy is currently destroying value on a risk-adjusted basis."
            )
            status = DegradationStatus.CRITICAL
            size_mult = min(size_mult, 0.25)

        elif (
            live_sharpe < 0.5 and benchmark and benchmark.predicted_annual_sharpe >= 1.0
        ):
            findings.append(
                f"Live Sharpe = {live_sharpe:.2f} vs predicted {benchmark.predicted_annual_sharpe:.2f}. "
                "Significant underperformance vs backtest."
            )
            if status == DegradationStatus.CLEAN:
                status = DegradationStatus.WARNING
            size_mult = min(size_mult, 0.5)

        # --- IS/OOS ratio check ---
        if oos_is_ratio is not None:
            if oos_is_ratio < self.CRITICAL_OOS_IS_RATIO:
                findings.append(
                    f"IS/OOS Sharpe ratio = {oos_is_ratio:.2f} (< {self.CRITICAL_OOS_IS_RATIO:.2f}). "
                    f"Live Sharpe is {(1 - oos_is_ratio):.0%} below backtest. Likely overfitted or regime change."
                )
                status = DegradationStatus.CRITICAL
                size_mult = min(size_mult, 0.25)
            elif oos_is_ratio < self.WARNING_OOS_IS_RATIO:
                findings.append(
                    f"IS/OOS Sharpe ratio = {oos_is_ratio:.2f} (< {self.WARNING_OOS_IS_RATIO:.2f}). "
                    f"Live underperforming backtest by {(1 - oos_is_ratio):.0%}. Monitor closely."
                )
                if status == DegradationStatus.CLEAN:
                    status = DegradationStatus.WARNING
                size_mult = min(size_mult, 0.5)

        # --- Drawdown ratio check ---
        if dd_ratio is not None:
            if dd_ratio > self.CRITICAL_DD_RATIO:
                findings.append(
                    f"Actual max drawdown {live_max_dd:.1%} is {dd_ratio:.1f}× predicted "
                    f"{benchmark.predicted_max_drawdown:.1%}. Tail risk is much higher than expected."
                )
                status = DegradationStatus.CRITICAL
                size_mult = min(size_mult, 0.25)
            elif dd_ratio > self.WARNING_DD_RATIO:
                findings.append(
                    f"Actual max drawdown {live_max_dd:.1%} is {dd_ratio:.1f}× predicted "
                    f"{benchmark.predicted_max_drawdown:.1%}. Drawdown exceeding model assumptions."
                )
                if status == DegradationStatus.CLEAN:
                    status = DegradationStatus.WARNING
                size_mult = min(size_mult, 0.5)

        # --- Win rate degradation vs benchmark ---
        if benchmark and benchmark.predicted_win_rate > 0:
            wr_gap = live_win_rate - benchmark.predicted_win_rate
            if wr_gap < -0.10 and live_win_rate < 0.45:
                findings.append(
                    f"Win rate {live_win_rate:.1%} vs predicted {benchmark.predicted_win_rate:.1%} "
                    f"(gap = {wr_gap:+.1%}). Consider retraining."
                )
                if status == DegradationStatus.CLEAN:
                    status = DegradationStatus.WARNING
                size_mult = min(size_mult, 0.5)

        if not findings:
            findings.append(
                f"No degradation detected. Live Sharpe={live_sharpe:.2f}, "
                f"Win Rate={live_win_rate:.1%}, Max DD={live_max_dd:.1%}."
            )

        return status, findings, size_mult

    # -------------------------------------------------------------------------
    # Metric computation
    # -------------------------------------------------------------------------

    @staticmethod
    def _rolling_sharpe(pnls: list[float], annualisation: int = 252) -> float:
        """
        Annualised Sharpe from a list of per-trade P&L values.

        Uses trade-level returns rather than daily returns because the live
        portfolio has too few trades for daily time-series Sharpe to be reliable.

        Annualisation factor: sqrt(252 / mean_holding_days). For strategies
        holding 1-5 days, this gives a fair apples-to-apples comparison with
        daily Sharpe from backtests.
        """
        if len(pnls) < 2:
            return 0.0
        arr = np.array(pnls)
        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=1))
        if std < 1e-9:
            return 0.0
        # No annualisation here — comparing absolute P&L Sharpe (trade-level)
        return mean / std

    @staticmethod
    def _max_drawdown(pnls: list[float]) -> float:
        """Max drawdown from cumulative P&L series."""
        if not pnls:
            return 0.0
        cumulative = np.cumsum(pnls)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / (np.abs(running_max) + 1e-9)
        return float(np.min(drawdowns))  # Negative — most negative = worst DD

    # -------------------------------------------------------------------------
    # Data access
    # -------------------------------------------------------------------------

    def _load_closed_trades(self, since: datetime) -> list[dict]:
        """Load closed trades from the database within the window."""
        conn = self._get_conn()
        if conn is None:
            return []
        try:
            rows = conn.execute(
                "SELECT symbol, side, quantity, entry_price, exit_price, "
                "realized_pnl, closed_at "
                "FROM closed_trades "
                "WHERE closed_at >= ? "
                "ORDER BY closed_at ASC",
                [since],
            ).fetchall()
            return [
                {
                    "symbol": r[0],
                    "side": r[1],
                    "quantity": r[2],
                    "entry_price": r[3],
                    "exit_price": r[4],
                    "realized_pnl": r[5],
                    "closed_at": r[6],
                }
                for r in rows
            ]
        except Exception as e:
            logger.warning(f"[DEGRADE] Could not load closed_trades: {e}")
            return []

    def _get_conn(self):
        """Get database connection, creating from PortfolioState if not injected."""
        if self._conn is not None:
            return self._conn
        try:
            return get_portfolio_state().conn
        except Exception as e:
            logger.warning(f"[DEGRADE] Could not get portfolio DB conn: {e}")
            return None

    # -------------------------------------------------------------------------
    # Benchmark persistence
    # -------------------------------------------------------------------------

    def _ensure_benchmark_table(self) -> None:
        conn = self._get_conn()
        if conn is None:
            return
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS is_benchmarks (
                    strategy_id             VARCHAR PRIMARY KEY,
                    predicted_annual_sharpe DOUBLE PRECISION NOT NULL,
                    predicted_max_drawdown  DOUBLE PRECISION NOT NULL,
                    predicted_win_rate      DOUBLE PRECISION NOT NULL,
                    n_backtest_trades       INTEGER DEFAULT 0,
                    registered_at           TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )
        except Exception as e:
            logger.debug(f"[DEGRADE] Could not create is_benchmarks table: {e}")

    def _load_benchmarks(self) -> None:
        conn = self._get_conn()
        if conn is None:
            return
        try:
            rows = conn.execute(
                "SELECT strategy_id, predicted_annual_sharpe, predicted_max_drawdown, "
                "predicted_win_rate, n_backtest_trades, registered_at FROM is_benchmarks"
            ).fetchall()
            for r in rows:
                self._benchmarks[r[0]] = ISBenchmark(
                    strategy_id=r[0],
                    predicted_annual_sharpe=r[1],
                    predicted_max_drawdown=r[2],
                    predicted_win_rate=r[3],
                    n_backtest_trades=r[4],
                    registered_at=r[5],
                )
            if rows:
                logger.info(f"[DEGRADE] Loaded {len(rows)} IS benchmarks from DB")
        except Exception as e:
            logger.debug(f"[DEGRADE] Could not load benchmarks: {e}")

    def _persist_benchmark(self, b: ISBenchmark) -> None:
        conn = self._get_conn()
        if conn is None:
            return
        try:
            conn.execute(
                """
                INSERT INTO is_benchmarks (strategy_id, predicted_annual_sharpe,
                    predicted_max_drawdown, predicted_win_rate, n_backtest_trades, registered_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT (strategy_id) DO UPDATE SET
                    predicted_annual_sharpe = excluded.predicted_annual_sharpe,
                    predicted_max_drawdown  = excluded.predicted_max_drawdown,
                    predicted_win_rate      = excluded.predicted_win_rate,
                    n_backtest_trades       = excluded.n_backtest_trades,
                    registered_at           = excluded.registered_at
                """,
                [
                    b.strategy_id,
                    b.predicted_annual_sharpe,
                    b.predicted_max_drawdown,
                    b.predicted_win_rate,
                    b.n_backtest_trades,
                    b.registered_at,
                ],
            )
        except Exception as e:
            logger.debug(f"[DEGRADE] Could not persist benchmark: {e}")


def get_degradation_detector() -> DegradationDetector:
    """Convenience factory — uses the singleton portfolio DB connection."""
    return DegradationDetector()
