# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Strategy Regression Test Suite (Phase 4.4).

For every strategy in `live` or `forward_testing` status that has a
`backtest_summary`, re-run the backtest on the SAME date range and assert
core metrics haven't degraded beyond a tolerance threshold.

This catches silent breakage when feature engineering, indicator logic,
or synthesis weights change — a code change in quantcore/features/ could
alter signals that a live strategy depends on.

Run:
    pytest tests/regression/test_strategy_regression.py -v
    pytest tests/regression/test_strategy_regression.py -v -k "strategy_name"

CI integration:
    Add to PR merge gate. Fails the build if any promoted strategy
    regresses beyond the 10% tolerance.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pytest
from quantstack.db import open_db_readonly
from quantstack.mcp.tools.backtesting import run_backtest
import asyncio

# ---- Tolerance thresholds ----
# Metrics are allowed to change by this relative amount from promotion values.
# Beyond this, the test fails.
SHARPE_TOLERANCE = 0.10  # 10% relative drop
WIN_RATE_TOLERANCE = 0.10  # 10% relative drop
MAX_DRAWDOWN_TOLERANCE = (
    0.15  # 15% relative increase (drawdown is negative — larger = worse)
)
PROFIT_FACTOR_TOLERANCE = 0.10


@dataclass
class PromotedStrategy:
    """A strategy with its promotion-time backtest metrics."""

    strategy_id: str
    name: str
    status: str
    backtest_summary: dict[str, Any]

    @property
    def symbol(self) -> str:
        return self.backtest_summary.get("symbol", "SPY")

    @property
    def start_date(self) -> str | None:
        return self.backtest_summary.get("start_date")

    @property
    def end_date(self) -> str | None:
        return self.backtest_summary.get("end_date")


def _load_promoted_strategies() -> list[PromotedStrategy]:
    """
    Load all strategies in live/forward_testing status with a backtest_summary.

    Returns an empty list if the DB is not available or no strategies qualify.
    This allows the test to be collected without failing during import.
    """
    try:
        conn = open_db_readonly()
        rows = conn.execute(
            """
            SELECT strategy_id, name, status, backtest_summary
            FROM strategies
            WHERE status IN ('live', 'forward_testing')
              AND backtest_summary IS NOT NULL
            ORDER BY name
            """
        ).fetchall()
        conn.close()

        strategies = []
        for row in rows:
            strategy_id, name, status, bs_raw = row
            bs = bs_raw if isinstance(bs_raw, dict) else json.loads(bs_raw or "{}")
            if not bs:
                continue
            strategies.append(
                PromotedStrategy(
                    strategy_id=strategy_id,
                    name=name,
                    status=status,
                    backtest_summary=bs,
                )
            )
        return strategies
    except Exception:
        return []


def _get_strategy_ids() -> list[str]:
    """Return strategy IDs for parametrize — must be a top-level callable for pytest."""
    return [s.strategy_id for s in _load_promoted_strategies()]


def _get_strategy_names() -> list[str]:
    """Return strategy names for test IDs."""
    return [s.name for s in _load_promoted_strategies()]


# Cache the loaded strategies so we don't hit the DB multiple times
_STRATEGIES: list[PromotedStrategy] | None = None


def _strategies() -> list[PromotedStrategy]:
    global _STRATEGIES
    if _STRATEGIES is None:
        _STRATEGIES = _load_promoted_strategies()
    return _STRATEGIES


def _strategy_params():
    """Generate pytest parametrize args from loaded strategies."""
    strats = _strategies()
    if not strats:
        return []
    return [pytest.param(s, id=f"{s.name}[{s.status}]") for s in strats]


@pytest.mark.regression
class TestStrategyRegression:
    """
    Regression tests for promoted strategies.

    Each test re-runs the backtest with the same parameters used at
    promotion time and asserts metrics haven't degraded beyond tolerance.
    """

    @pytest.fixture(autouse=True)
    def _skip_if_no_strategies(self):
        if not _strategies():
            pytest.skip("No promoted strategies with backtest_summary found in DB")

    @pytest.mark.parametrize("strategy", _strategy_params())
    def test_sharpe_ratio_not_degraded(self, strategy: PromotedStrategy):
        """Sharpe ratio should not drop more than 10% from promotion value."""
        promotion_sharpe = strategy.backtest_summary.get("sharpe_ratio")
        if promotion_sharpe is None:
            pytest.skip(f"{strategy.name}: no sharpe_ratio in backtest_summary")

        current = _run_backtest_for_strategy(strategy)
        current_sharpe = current.get("sharpe_ratio", 0.0)

        # Allow for some degradation but not beyond tolerance
        if promotion_sharpe > 0:
            min_acceptable = promotion_sharpe * (1 - SHARPE_TOLERANCE)
            assert current_sharpe >= min_acceptable, (
                f"{strategy.name}: Sharpe degraded from {promotion_sharpe:.3f} to "
                f"{current_sharpe:.3f} (min acceptable: {min_acceptable:.3f})"
            )
        else:
            # Promotion Sharpe was negative or zero — just ensure it hasn't gotten worse
            assert (
                current_sharpe >= promotion_sharpe - 0.1
            ), f"{strategy.name}: Sharpe worsened from {promotion_sharpe:.3f} to {current_sharpe:.3f}"

    @pytest.mark.parametrize("strategy", _strategy_params())
    def test_win_rate_not_degraded(self, strategy: PromotedStrategy):
        """Win rate should not drop more than 10% from promotion value."""
        promotion_wr = strategy.backtest_summary.get("win_rate")
        if promotion_wr is None:
            pytest.skip(f"{strategy.name}: no win_rate in backtest_summary")

        current = _run_backtest_for_strategy(strategy)
        current_wr = current.get("win_rate", 0.0)

        min_acceptable = promotion_wr * (1 - WIN_RATE_TOLERANCE)
        assert current_wr >= min_acceptable, (
            f"{strategy.name}: Win rate degraded from {promotion_wr:.1%} to "
            f"{current_wr:.1%} (min acceptable: {min_acceptable:.1%})"
        )

    @pytest.mark.parametrize("strategy", _strategy_params())
    def test_max_drawdown_not_worsened(self, strategy: PromotedStrategy):
        """Max drawdown should not increase more than 15% from promotion value."""
        promotion_dd = strategy.backtest_summary.get("max_drawdown")
        if promotion_dd is None:
            pytest.skip(f"{strategy.name}: no max_drawdown in backtest_summary")

        current = _run_backtest_for_strategy(strategy)
        current_dd = current.get("max_drawdown", 0.0)

        # Drawdown is typically negative — "worse" means more negative.
        # We compare absolute values: current abs(dd) should not exceed promotion abs(dd) by >15%
        max_acceptable_dd = abs(promotion_dd) * (1 + MAX_DRAWDOWN_TOLERANCE)
        assert abs(current_dd) <= max_acceptable_dd, (
            f"{strategy.name}: Drawdown worsened from {promotion_dd:.2%} to "
            f"{current_dd:.2%} (max acceptable: {-max_acceptable_dd:.2%})"
        )

    @pytest.mark.parametrize("strategy", _strategy_params())
    def test_profit_factor_not_degraded(self, strategy: PromotedStrategy):
        """Profit factor should not drop more than 10% from promotion value."""
        promotion_pf = strategy.backtest_summary.get("profit_factor")
        if promotion_pf is None:
            pytest.skip(f"{strategy.name}: no profit_factor in backtest_summary")

        current = _run_backtest_for_strategy(strategy)
        current_pf = current.get("profit_factor", 0.0)

        min_acceptable = promotion_pf * (1 - PROFIT_FACTOR_TOLERANCE)
        assert current_pf >= min_acceptable, (
            f"{strategy.name}: Profit factor degraded from {promotion_pf:.2f} to "
            f"{current_pf:.2f} (min acceptable: {min_acceptable:.2f})"
        )


# =============================================================================
# Backtest runner (cached per strategy to avoid redundant runs)
# =============================================================================

_BACKTEST_CACHE: dict[str, dict[str, Any]] = {}


def _run_backtest_for_strategy(strategy: PromotedStrategy) -> dict[str, Any]:
    """
    Run backtest for a strategy using the same parameters from promotion.

    Results are cached per strategy_id to avoid re-running across
    multiple metric assertions for the same strategy.
    """
    if strategy.strategy_id in _BACKTEST_CACHE:
        return _BACKTEST_CACHE[strategy.strategy_id]

    bs = strategy.backtest_summary
    result = asyncio.run(
        run_backtest(
            strategy_id=strategy.strategy_id,
            symbol=strategy.symbol,
            start_date=bs.get("start_date"),
            end_date=bs.get("end_date"),
            initial_capital=bs.get("initial_capital", 100_000.0),
            position_size_pct=bs.get("position_size_pct", 0.10),
        )
    )

    _BACKTEST_CACHE[strategy.strategy_id] = result
    return result
