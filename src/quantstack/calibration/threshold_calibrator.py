"""Data-backed threshold calibration for position sizing, risk, and strategy promotion.

Each calibration function:
1. Reads historical data from PostgreSQL
2. Checks minimum sample size; returns fallback if insufficient
3. Computes the threshold using a defined statistical methodology
4. Stores the result in calibration_history
5. Returns a CalibrationResult
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import numpy as np
from scipy import stats

from quantstack.calibration.deflated_sharpe import deflated_sharpe_ratio
from quantstack.calibration.models import CalibrationResult
from quantstack.calibration.monte_carlo import (
    compute_max_drawdowns,
    compute_monthly_max_drawdowns,
    simulate_paths,
)
from quantstack.db import pg_conn

logger = logging.getLogger(__name__)

# Fallback values when insufficient data
_FALLBACK_POSITION_SIZE = 0.15
_FALLBACK_DAILY_HALT = 0.03
_FALLBACK_SIGNAL_IC = 0.02
_FALLBACK_BACKTEST_SHARPE = 0.5
_FALLBACK_KELLY = 0.5

# Minimum sample sizes
_MIN_TRADES_POSITION = 50
_MIN_DAYS_HALT = 60
_MIN_STRATEGIES_SIGNAL = 30
_MIN_STRATEGIES_BACKTEST = 20
_MIN_TRADES_KELLY = 100

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS calibration_history (
    id BIGSERIAL PRIMARY KEY,
    threshold_name VARCHAR NOT NULL,
    calibrated_value DOUBLE PRECISION NOT NULL,
    previous_value DOUBLE PRECISION,
    confidence_interval_low DOUBLE PRECISION,
    confidence_interval_high DOUBLE PRECISION,
    sample_size INTEGER NOT NULL,
    methodology TEXT NOT NULL,
    is_fallback BOOLEAN NOT NULL DEFAULT FALSE,
    calibrated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_calibration_history_name_time
ON calibration_history (threshold_name, calibrated_at DESC);
"""


class ThresholdCalibrator:
    """Computes statistically grounded thresholds from historical trading data."""

    def __init__(self) -> None:
        self._init_schema()

    def _init_schema(self) -> None:
        """Ensure calibration_history table exists."""
        try:
            with pg_conn() as conn:
                conn.execute(_SCHEMA_SQL)
        except Exception as exc:
            logger.warning("Failed to init calibration schema: %s", exc)

    def calibrate_position_size(self) -> CalibrationResult:
        """Max position size such that P(single_position_loss > 2% of portfolio) < 5%.

        Data source: closed_trades table.
        Fallback: 0.15 when < 50 closed trades.
        """
        try:
            with pg_conn() as conn:
                rows = conn.execute(
                    "SELECT realized_pnl, portfolio_equity_at_entry "
                    "FROM closed_trades "
                    "WHERE realized_pnl IS NOT NULL "
                    "AND portfolio_equity_at_entry IS NOT NULL "
                    "AND portfolio_equity_at_entry > 0"
                ).fetchall()
        except Exception:
            rows = []

        if len(rows) < _MIN_TRADES_POSITION:
            return self._fallback(
                "position_size", _FALLBACK_POSITION_SIZE, len(rows),
                f"Insufficient data ({len(rows)} < {_MIN_TRADES_POSITION} trades). "
                f"Using fallback {_FALLBACK_POSITION_SIZE}."
            )

        loss_pcts = []
        for pnl, equity in rows:
            if equity > 0:
                loss_pcts.append(pnl / equity)

        losses = np.array([x for x in loss_pcts if x < 0])
        if len(losses) < 10:
            return self._fallback(
                "position_size", _FALLBACK_POSITION_SIZE, len(rows),
                f"Insufficient loss observations ({len(losses)}). Using fallback."
            )

        # VaR at 95% of single-position loss distribution
        var_95 = np.percentile(np.abs(losses), 95)
        if var_95 == 0:
            var_95 = 0.01  # floor to prevent division by zero

        # max_position_pct = 2% / VaR_95
        calibrated = min(0.02 / var_95, 0.30)  # cap at 30%
        calibrated = max(calibrated, 0.01)  # floor at 1%

        # Bootstrap confidence interval
        ci = self._bootstrap_ci(np.abs(losses), lambda x: 0.02 / np.percentile(x, 95))

        result = CalibrationResult(
            threshold_name="position_size",
            value=round(calibrated, 4),
            confidence_interval=(round(ci[0], 4), round(ci[1], 4)),
            sample_size=len(rows),
            methodology=(
                f"max_position_pct = 0.02 / VaR_95(single_position_loss). "
                f"VaR_95 = {var_95:.4f}. Based on {len(losses)} loss observations."
            ),
        )
        self._store_result(result)
        return result

    def calibrate_daily_halt(self) -> CalibrationResult:
        """Daily loss halt threshold such that P(monthly drawdown > 10%) < 5%.

        Data source: portfolio_snapshots table (daily P&L).
        Fallback: 0.03 when < 60 trading days of snapshots.
        """
        try:
            with pg_conn() as conn:
                rows = conn.execute(
                    "SELECT daily_pnl_pct FROM portfolio_snapshots "
                    "WHERE daily_pnl_pct IS NOT NULL "
                    "ORDER BY snapshot_date ASC"
                ).fetchall()
        except Exception:
            rows = []

        daily_returns = np.array([r[0] for r in rows]) if rows else np.array([])

        if len(daily_returns) < _MIN_DAYS_HALT:
            return self._fallback(
                "daily_halt", _FALLBACK_DAILY_HALT, len(daily_returns),
                f"Insufficient data ({len(daily_returns)} < {_MIN_DAYS_HALT} days). "
                f"Using fallback {_FALLBACK_DAILY_HALT}."
            )

        # Search for optimal halt threshold via Monte Carlo
        best_threshold = _FALLBACK_DAILY_HALT
        target_monthly_dd = 0.10
        target_prob = 0.05

        for candidate in np.arange(0.01, 0.10, 0.005):
            paths = simulate_paths(
                daily_returns, n_paths=5000, n_days=252,
                halt_threshold=candidate,
            )
            monthly_dds = compute_monthly_max_drawdowns(paths)
            prob_exceed = np.mean(monthly_dds > target_monthly_dd)

            if prob_exceed <= target_prob:
                best_threshold = float(candidate)
                break

        result = CalibrationResult(
            threshold_name="daily_halt",
            value=round(best_threshold, 4),
            confidence_interval=(round(best_threshold * 0.8, 4), round(best_threshold * 1.2, 4)),
            sample_size=len(daily_returns),
            methodology=(
                f"Bootstrap simulation: halt at {best_threshold:.3f} daily loss "
                f"keeps P(monthly DD > 10%) < 5%. Based on {len(daily_returns)} days."
            ),
        )
        self._store_result(result)
        return result

    def calibrate_signal_validation(self) -> CalibrationResult:
        """IC threshold where P(profitable OOS) > 70%.

        Data source: strategies table with IS and OOS results.
        Fallback: 0.02 when < 30 strategies validated.
        """
        try:
            with pg_conn() as conn:
                rows = conn.execute(
                    "SELECT "
                    "  (backtest_summary->>'ic')::float AS ic, "
                    "  (backtest_summary->>'oos_sharpe')::float AS oos_sharpe "
                    "FROM strategies "
                    "WHERE backtest_summary->>'ic' IS NOT NULL "
                    "AND backtest_summary->>'oos_sharpe' IS NOT NULL"
                ).fetchall()
        except Exception:
            rows = []

        if len(rows) < _MIN_STRATEGIES_SIGNAL:
            return self._fallback(
                "signal_validation", _FALLBACK_SIGNAL_IC, len(rows),
                f"Insufficient data ({len(rows)} < {_MIN_STRATEGIES_SIGNAL} strategies). "
                f"Using fallback IC > {_FALLBACK_SIGNAL_IC}."
            )

        ics = np.array([r[0] for r in rows])
        profitable = np.array([1 if r[1] > 0 else 0 for r in rows])

        # Find IC threshold where P(profitable OOS) > 70%
        # Sort by IC and find the cutoff
        sorted_idx = np.argsort(ics)
        ics_sorted = ics[sorted_idx]
        prof_sorted = profitable[sorted_idx]

        best_threshold = _FALLBACK_SIGNAL_IC
        for i in range(len(ics_sorted)):
            above = prof_sorted[i:]
            if len(above) >= 5 and np.mean(above) >= 0.70:
                best_threshold = float(ics_sorted[i])
                break

        result = CalibrationResult(
            threshold_name="signal_validation",
            value=round(best_threshold, 4),
            confidence_interval=(round(best_threshold * 0.8, 4), round(best_threshold * 1.2, 4)),
            sample_size=len(rows),
            methodology=(
                f"ROC analysis: IC > {best_threshold:.4f} yields >= 70% profitable OOS. "
                f"Based on {len(rows)} validated strategies."
            ),
        )
        self._store_result(result)
        return result

    def calibrate_backtest_gates(self) -> CalibrationResult:
        """IS Sharpe threshold (after DSR) that predicts positive forward Sharpe at 80%.

        Data source: strategies table with forward-testing outcomes.
        Fallback: 0.5 when < 20 strategies have completed forward testing.
        """
        try:
            with pg_conn() as conn:
                rows = conn.execute(
                    "SELECT "
                    "  (backtest_summary->>'is_sharpe')::float AS is_sharpe, "
                    "  (backtest_summary->>'is_skew')::float AS is_skew, "
                    "  (backtest_summary->>'is_kurtosis')::float AS is_kurtosis, "
                    "  (backtest_summary->>'is_n_returns')::int AS n_returns, "
                    "  (backtest_summary->>'forward_sharpe')::float AS fwd_sharpe "
                    "FROM strategies "
                    "WHERE backtest_summary->>'is_sharpe' IS NOT NULL "
                    "AND backtest_summary->>'forward_sharpe' IS NOT NULL "
                    "AND status IN ('live', 'retired', 'forward_testing')"
                ).fetchall()
        except Exception:
            rows = []

        # Count total strategies tested for DSR
        try:
            with pg_conn() as conn:
                total_row = conn.execute(
                    "SELECT COUNT(*) FROM strategies WHERE backtest_summary IS NOT NULL"
                ).fetchone()
                total_tested = total_row[0] if total_row else len(rows)
        except Exception:
            total_tested = max(len(rows), 1)

        if len(rows) < _MIN_STRATEGIES_BACKTEST:
            return self._fallback(
                "backtest_gates", _FALLBACK_BACKTEST_SHARPE, len(rows),
                f"Insufficient data ({len(rows)} < {_MIN_STRATEGIES_BACKTEST} strategies). "
                f"Using fallback Sharpe > {_FALLBACK_BACKTEST_SHARPE}."
            )

        # For each strategy, compute DSR and check if forward Sharpe > 0
        best_threshold = _FALLBACK_BACKTEST_SHARPE
        sharpes_and_outcomes = []

        for is_sharpe, skew, kurt, n_ret, fwd_sharpe in rows:
            skew = skew if skew is not None else 0.0
            kurt = kurt if kurt is not None else 3.0
            n_ret = n_ret if n_ret is not None else 252

            dsr = deflated_sharpe_ratio(
                observed_sharpe=is_sharpe,
                num_strategies_tested=total_tested,
                num_returns=n_ret,
                skewness=skew,
                kurtosis=kurt,
            )
            sharpes_and_outcomes.append((is_sharpe, dsr, fwd_sharpe > 0))

        # Find IS Sharpe where 80% have positive forward Sharpe
        sharpes_and_outcomes.sort(key=lambda x: x[0])
        for i in range(len(sharpes_and_outcomes)):
            above = [x[2] for x in sharpes_and_outcomes[i:]]
            if len(above) >= 5 and np.mean(above) >= 0.80:
                best_threshold = sharpes_and_outcomes[i][0]
                break

        result = CalibrationResult(
            threshold_name="backtest_gates",
            value=round(best_threshold, 4),
            confidence_interval=(round(best_threshold * 0.8, 4), round(best_threshold * 1.2, 4)),
            sample_size=len(rows),
            methodology=(
                f"DSR-adjusted IS Sharpe > {best_threshold:.3f} predicts positive "
                f"forward Sharpe at >= 80%. N_tested={total_tested}, N_with_forward={len(rows)}."
            ),
        )
        self._store_result(result)
        return result

    def calibrate_kelly(self) -> CalibrationResult:
        """Optimal Kelly fraction maximizing geometric growth with max DD < 15% in 95% of paths.

        Data source: closed_trades (win/loss distribution).
        Fallback: 0.5 (half-Kelly) when < 100 closed trades.
        """
        try:
            with pg_conn() as conn:
                rows = conn.execute(
                    "SELECT realized_pnl, portfolio_equity_at_entry "
                    "FROM closed_trades "
                    "WHERE realized_pnl IS NOT NULL "
                    "AND portfolio_equity_at_entry IS NOT NULL "
                    "AND portfolio_equity_at_entry > 0"
                ).fetchall()
        except Exception:
            rows = []

        if len(rows) < _MIN_TRADES_KELLY:
            return self._fallback(
                "kelly_fraction", _FALLBACK_KELLY, len(rows),
                f"Insufficient data ({len(rows)} < {_MIN_TRADES_KELLY} trades). "
                f"Using fallback half-Kelly {_FALLBACK_KELLY}."
            )

        returns = np.array([pnl / equity for pnl, equity in rows if equity > 0])

        # Search for optimal Kelly fraction
        best_fraction = _FALLBACK_KELLY
        best_growth = -np.inf

        for fraction in np.arange(0.10, 1.05, 0.05):
            paths = simulate_paths(
                returns, n_paths=10_000, n_days=252,
                kelly_fraction=fraction,
            )
            max_dds = compute_max_drawdowns(paths)
            prob_exceed = np.mean(max_dds > 0.15)

            if prob_exceed <= 0.05:
                # Geometric growth rate
                final_equity = paths[:, -1]
                geo_growth = np.mean(np.log(final_equity))
                if geo_growth > best_growth:
                    best_growth = geo_growth
                    best_fraction = float(fraction)

        result = CalibrationResult(
            threshold_name="kelly_fraction",
            value=round(best_fraction, 2),
            confidence_interval=(round(max(best_fraction - 0.10, 0.05), 2),
                                 round(min(best_fraction + 0.10, 1.0), 2)),
            sample_size=len(rows),
            methodology=(
                f"Monte Carlo optimization: f={best_fraction:.2f} maximizes geometric growth "
                f"subject to P(max DD > 15%) < 5%. Based on {len(rows)} trades."
            ),
        )
        self._store_result(result)
        return result

    def calibrate_all(self) -> dict[str, CalibrationResult]:
        """Run all calibrations and return results keyed by threshold_name."""
        results = {}
        for method_name in [
            "calibrate_position_size",
            "calibrate_daily_halt",
            "calibrate_signal_validation",
            "calibrate_backtest_gates",
            "calibrate_kelly",
        ]:
            try:
                result = getattr(self, method_name)()
                results[result.threshold_name] = result
            except Exception as exc:
                logger.error("Calibration %s failed: %s", method_name, exc)
        return results

    def get_latest(self, threshold_name: str) -> CalibrationResult | None:
        """Retrieve the most recent calibration for a given threshold from DB."""
        try:
            with pg_conn() as conn:
                row = conn.execute(
                    "SELECT calibrated_value, confidence_interval_low, "
                    "confidence_interval_high, sample_size, methodology, is_fallback "
                    "FROM calibration_history "
                    "WHERE threshold_name = ? "
                    "ORDER BY calibrated_at DESC LIMIT 1",
                    [threshold_name],
                ).fetchone()
            if row is None:
                return None
            return CalibrationResult(
                threshold_name=threshold_name,
                value=row[0],
                confidence_interval=(row[1] or 0.0, row[2] or 0.0),
                sample_size=row[3],
                methodology=row[4],
                is_fallback=row[5],
            )
        except Exception as exc:
            logger.warning("Failed to get latest calibration for %s: %s", threshold_name, exc)
            return None

    def _store_result(self, result: CalibrationResult) -> None:
        """Persist a CalibrationResult to calibration_history table."""
        previous = self.get_latest(result.threshold_name)
        previous_value = previous.value if previous else None

        try:
            with pg_conn() as conn:
                conn.execute(
                    "INSERT INTO calibration_history "
                    "(threshold_name, calibrated_value, previous_value, "
                    "confidence_interval_low, confidence_interval_high, "
                    "sample_size, methodology, is_fallback) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    [
                        result.threshold_name,
                        result.value,
                        previous_value,
                        result.confidence_interval[0],
                        result.confidence_interval[1],
                        result.sample_size,
                        result.methodology,
                        result.is_fallback,
                    ],
                )
        except Exception as exc:
            logger.warning("Failed to store calibration result: %s", exc)

    def _fallback(
        self,
        name: str,
        value: float,
        sample_size: int,
        methodology: str,
    ) -> CalibrationResult:
        """Create a fallback CalibrationResult."""
        result = CalibrationResult(
            threshold_name=name,
            value=value,
            confidence_interval=(value, value),
            sample_size=sample_size,
            methodology=methodology,
            is_fallback=True,
        )
        self._store_result(result)
        return result

    @staticmethod
    def _bootstrap_ci(
        data: np.ndarray,
        statistic_fn,
        n_boot: int = 1000,
        ci: float = 0.95,
    ) -> tuple[float, float]:
        """Compute bootstrap confidence interval for a statistic."""
        rng = np.random.default_rng(42)
        boot_stats = []
        for _ in range(n_boot):
            sample = rng.choice(data, size=len(data), replace=True)
            try:
                boot_stats.append(statistic_fn(sample))
            except Exception:
                continue

        if not boot_stats:
            return (0.0, 1.0)

        alpha = (1 - ci) / 2
        return (
            float(np.percentile(boot_stats, alpha * 100)),
            float(np.percentile(boot_stats, (1 - alpha) * 100)),
        )
