"""Tests for borrowing / funding cost model (section 13).

Covers:
- FundingCostCalculator pure math (no DB)
- Position model fields for margin_used / cumulative_funding_cost
- PortfolioState.accrue_daily_funding integration (requires DB)
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from quantstack.execution.funding import (
    FundingCostCalculator,
    MARGIN_ANNUAL_RATE_DEFAULT,
    TRADING_DAYS_PER_YEAR,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture()
def conn():
    """Live DB connection with execution layer tables migrated.

    Each test runs inside a SAVEPOINT that is rolled back on teardown.
    """
    from quantstack.db import db_conn, _migrate_execution_layer_pg

    with db_conn() as c:
        _migrate_execution_layer_pg(c)
        c._ensure_raw()
        c._raw.execute("SAVEPOINT test_sp")
        yield c
        c._raw.execute("ROLLBACK TO SAVEPOINT test_sp")


# ============================================================================
# Pure-math tests — FundingCostCalculator
# ============================================================================


class TestDailyInterest:
    def test_standard_margin(self):
        """daily_interest(10_000) at 8% APR ≈ 3.17."""
        calc = FundingCostCalculator(annual_rate=0.08)
        result = calc.daily_interest(10_000)
        expected = 10_000 * 0.08 / 252
        assert abs(result - expected) < 0.01
        assert abs(result - 3.17) < 0.01

    def test_cumulative_five_days(self):
        """5 consecutive days of accrual ≈ 15.87."""
        calc = FundingCostCalculator(annual_rate=0.08)
        daily = calc.daily_interest(10_000)
        cumulative = daily * 5
        expected = 10_000 * 0.08 / 252 * 5
        assert abs(cumulative - expected) < 0.01
        assert abs(cumulative - 15.87) < 0.05

    def test_zero_margin(self):
        """Zero margin produces zero cost."""
        calc = FundingCostCalculator(annual_rate=0.08)
        assert calc.daily_interest(0.0) == 0.0

    def test_negative_margin_clamped(self):
        """Negative margin is clamped to zero cost."""
        calc = FundingCostCalculator(annual_rate=0.08)
        assert calc.daily_interest(-5_000) == 0.0

    def test_custom_rate(self):
        """Constructor accepts a custom annual rate."""
        calc = FundingCostCalculator(annual_rate=0.10)
        assert calc.annual_rate == 0.10
        result = calc.daily_interest(10_000)
        expected = 10_000 * 0.10 / 252
        assert abs(result - expected) < 0.01

    def test_rate_from_env(self, monkeypatch):
        """Falls back to MARGIN_ANNUAL_RATE env var when no rate passed."""
        monkeypatch.setenv("MARGIN_ANNUAL_RATE", "0.05")
        calc = FundingCostCalculator()
        assert calc.annual_rate == 0.05

    def test_default_rate(self):
        """Without env var or constructor arg, uses the module default."""
        # Ensure env var is unset
        os.environ.pop("MARGIN_ANNUAL_RATE", None)
        calc = FundingCostCalculator()
        assert calc.annual_rate == MARGIN_ANNUAL_RATE_DEFAULT


class TestAccrueFundingCosts:
    def test_filters_positions_with_margin(self):
        """Only positions with margin_used > 0 get charged."""
        calc = FundingCostCalculator(annual_rate=0.08)
        positions = [
            SimpleNamespace(symbol="AAPL", margin_used=10_000),
            SimpleNamespace(symbol="MSFT", margin_used=0.0),
            SimpleNamespace(symbol="GOOG", margin_used=5_000),
        ]
        result = calc.accrue_funding_costs(positions)
        symbols = [s for s, _ in result]
        assert "AAPL" in symbols
        assert "GOOG" in symbols
        assert "MSFT" not in symbols
        assert len(result) == 2

    def test_strategy_level_sum(self):
        """Sum of funding costs for two positions is correct."""
        calc = FundingCostCalculator(annual_rate=0.08)
        positions = [
            SimpleNamespace(symbol="AAPL", margin_used=10_000),
            SimpleNamespace(symbol="GOOG", margin_used=20_000),
        ]
        result = calc.accrue_funding_costs(positions)
        total = sum(cost for _, cost in result)
        expected = (10_000 + 20_000) * 0.08 / 252
        assert abs(total - expected) < 0.01


class TestFundingDeductedFromPnL:
    def test_adjusted_unrealized_pnl(self):
        """Funding cost should reduce unrealized P&L."""
        calc = FundingCostCalculator(annual_rate=0.08)
        unrealized_pnl = 500.0
        daily = calc.daily_interest(10_000)
        cumulative_5d = daily * 5
        adjusted = unrealized_pnl - cumulative_5d
        assert abs(adjusted - 484.13) < 0.05


# ============================================================================
# DB integration tests — Position model + PortfolioState.accrue_daily_funding
# ============================================================================


class TestPositionFundingFields:
    def test_margin_used_default(self):
        """Position.margin_used defaults to 0.0."""
        from quantstack.execution.portfolio_state import Position

        pos = Position(symbol="TEST", quantity=100, avg_cost=50.0)
        assert pos.margin_used == 0.0
        assert pos.cumulative_funding_cost == 0.0

    def test_margin_used_roundtrip(self, conn):
        """margin_used and cumulative_funding_cost survive DB upsert → read."""
        from quantstack.execution.portfolio_state import Position, PortfolioState

        state = PortfolioState(conn=conn)
        pos = Position(
            symbol="FUND_TEST",
            quantity=100,
            avg_cost=50.0,
            margin_used=8_000.0,
            cumulative_funding_cost=12.34,
        )
        state.upsert_position(pos)
        loaded = state.get_position("FUND_TEST")
        assert loaded is not None
        assert abs(loaded.margin_used - 8_000.0) < 0.01
        assert abs(loaded.cumulative_funding_cost - 12.34) < 0.01


class TestAccrueDailyFunding:
    def test_accrues_and_persists(self, conn):
        """accrue_daily_funding updates cumulative_funding_cost in DB."""
        from quantstack.execution.portfolio_state import Position, PortfolioState

        state = PortfolioState(conn=conn)
        pos = Position(
            symbol="ACCRUE_TEST",
            quantity=100,
            avg_cost=50.0,
            margin_used=10_000.0,
        )
        state.upsert_position(pos)

        calc = FundingCostCalculator(annual_rate=0.08)
        accruals = state.accrue_daily_funding(calculator=calc)

        assert len(accruals) == 1
        assert accruals[0][0] == "ACCRUE_TEST"
        expected_daily = 10_000 * 0.08 / 252
        assert abs(accruals[0][1] - expected_daily) < 0.01

        # Verify DB was updated
        loaded = state.get_position("ACCRUE_TEST")
        assert loaded is not None
        assert abs(loaded.cumulative_funding_cost - expected_daily) < 0.01

    def test_multiple_days_accumulate(self, conn):
        """Running accrual N times accumulates correctly."""
        from quantstack.execution.portfolio_state import Position, PortfolioState

        state = PortfolioState(conn=conn)
        pos = Position(
            symbol="MULTI_DAY",
            quantity=100,
            avg_cost=50.0,
            margin_used=10_000.0,
        )
        state.upsert_position(pos)

        calc = FundingCostCalculator(annual_rate=0.08)
        for _ in range(5):
            state.accrue_daily_funding(calculator=calc)

        loaded = state.get_position("MULTI_DAY")
        assert loaded is not None
        expected = 10_000 * 0.08 / 252 * 5
        assert abs(loaded.cumulative_funding_cost - expected) < 0.01

    def test_skips_zero_margin(self, conn):
        """Positions with zero margin_used are not accrued."""
        from quantstack.execution.portfolio_state import Position, PortfolioState

        state = PortfolioState(conn=conn)
        pos = Position(
            symbol="NO_MARGIN",
            quantity=100,
            avg_cost=50.0,
            margin_used=0.0,
        )
        state.upsert_position(pos)

        calc = FundingCostCalculator(annual_rate=0.08)
        accruals = state.accrue_daily_funding(calculator=calc)
        assert len(accruals) == 0

        loaded = state.get_position("NO_MARGIN")
        assert loaded is not None
        assert loaded.cumulative_funding_cost == 0.0
