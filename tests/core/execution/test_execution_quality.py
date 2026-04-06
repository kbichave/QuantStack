"""Tests for Section 09: Execution Quality Scoring.

Covers:
  - compute_quality_scalar threshold table
  - get_execution_quality_scalar with no rows (default 1.0)
  - Integration: quality scalar applied after ADV cap in RiskGate.check()
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from quantstack.core.execution.execution_quality import compute_quality_scalar


# ---------------------------------------------------------------------------
# Task 1: compute_quality_scalar threshold tests
# ---------------------------------------------------------------------------


class TestComputeQualityScalar:
    """Verify the threshold table mapping error_bps -> scalar."""

    def test_excellent_execution(self):
        assert compute_quality_scalar(3.0) == 1.1

    def test_acceptable_execution(self):
        assert compute_quality_scalar(10.0) == 1.0

    def test_poor_execution(self):
        assert compute_quality_scalar(20.0) == 0.7

    def test_very_poor_execution(self):
        assert compute_quality_scalar(35.0) == 0.5

    def test_boundary_at_5(self):
        # 5 bps is the lower boundary of the "acceptable" band
        assert compute_quality_scalar(5.0) == 1.0

    def test_boundary_at_15(self):
        # 15 bps is the upper boundary of the "acceptable" band
        assert compute_quality_scalar(15.0) == 1.0

    def test_boundary_just_above_15(self):
        assert compute_quality_scalar(15.01) == 0.7

    def test_boundary_at_30(self):
        # 30 bps is the upper boundary of the "poor" band
        assert compute_quality_scalar(30.0) == 0.7

    def test_boundary_just_above_30(self):
        assert compute_quality_scalar(30.01) == 0.5

    def test_zero_error(self):
        assert compute_quality_scalar(0.0) == 1.1


# ---------------------------------------------------------------------------
# Task 2: get_execution_quality_scalar
# ---------------------------------------------------------------------------


class TestGetExecutionQualityScalar:
    """Verify DB lookup returns scalar or default."""

    def test_no_rows_returns_default(self):
        from quantstack.execution.risk_gate import get_execution_quality_scalar

        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchone.return_value = None

        result = get_execution_quality_scalar("AAPL", mock_conn)
        assert result == 1.0

    def test_returns_scalar_from_db(self):
        from quantstack.execution.risk_gate import get_execution_quality_scalar

        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchone.return_value = (0.7,)

        result = get_execution_quality_scalar("AAPL", mock_conn)
        assert result == 0.7

    def test_symbol_uppercased(self):
        from quantstack.execution.risk_gate import get_execution_quality_scalar

        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchone.return_value = (1.1,)

        get_execution_quality_scalar("aapl", mock_conn)
        call_args = mock_conn.execute.call_args
        assert call_args[0][1] == ["AAPL"]


# ---------------------------------------------------------------------------
# Task 3: Integration — quality scalar applied after ADV cap
# ---------------------------------------------------------------------------


def _make_mock_portfolio(equity: float = 100_000.0, daily_pnl: float = 0.0):
    """Create a mock portfolio for RiskGate tests."""
    snapshot = MagicMock()
    snapshot.total_equity = equity
    snapshot.daily_pnl = daily_pnl

    portfolio = MagicMock()
    portfolio.get_snapshot.return_value = snapshot
    portfolio.get_position.return_value = None
    portfolio.get_positions.return_value = []
    return portfolio


class TestRiskGateQualityScalarIntegration:
    """Verify that quality_scalar is applied AFTER ADV cap in check()."""

    @patch("quantstack.execution.risk_gate.pg_conn")
    def test_quality_scalar_after_adv_cap(self, mock_pg_conn):
        """proposed=100, quality_scalar=0.7, ADV cap=90 -> final=63.

        Flow: ADV cap -> 90, then quality -> 90 * 0.7 = 63.
        """
        from quantstack.execution.risk_gate import RiskGate, RiskLimits

        # Mock the DB lookup to return quality_scalar=0.7
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchone.return_value = (0.7,)
        mock_pg_conn.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_pg_conn.return_value.__exit__ = MagicMock(return_value=False)

        limits = RiskLimits(
            max_position_pct=1.0,  # Very permissive — won't trigger
            max_position_notional=1_000_000.0,
            max_gross_exposure_pct=10.0,
            min_daily_volume=1,  # Allow low volume
            max_participation_pct=0.01,  # 1% of ADV
        )
        portfolio = _make_mock_portfolio()
        gate = RiskGate(limits=limits, portfolio=portfolio)

        # proposed_qty=100, daily_volume=9000 -> ADV cap = 9000 * 0.01 = 90
        verdict = gate.check(
            symbol="TEST",
            side="buy",
            quantity=100,
            current_price=10.0,
            daily_volume=9000,
        )

        assert verdict.approved is True
        # ADV cap: min(100, 9000*0.01=90) = 90, then quality: 90 * 0.7 = 63
        assert verdict.approved_quantity == 63

    @patch("quantstack.execution.risk_gate.pg_conn")
    def test_quality_scalar_no_db_row_default(self, mock_pg_conn):
        """No quality data -> scalar=1.0, quantity unchanged (after ADV cap)."""
        from quantstack.execution.risk_gate import RiskGate, RiskLimits

        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchone.return_value = None
        mock_pg_conn.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_pg_conn.return_value.__exit__ = MagicMock(return_value=False)

        limits = RiskLimits(
            max_position_pct=1.0,
            max_position_notional=1_000_000.0,
            max_gross_exposure_pct=10.0,
            min_daily_volume=1,
            max_participation_pct=1.0,  # Very permissive — won't cap
        )
        portfolio = _make_mock_portfolio()
        gate = RiskGate(limits=limits, portfolio=portfolio)

        verdict = gate.check(
            symbol="TEST",
            side="buy",
            quantity=100,
            current_price=10.0,
            daily_volume=1_000_000,
        )

        assert verdict.approved is True
        assert verdict.approved_quantity == 100

    @patch("quantstack.execution.risk_gate.pg_conn")
    def test_quality_scalar_db_failure_defaults_to_one(self, mock_pg_conn):
        """DB error -> scalar=1.0, trade not blocked."""
        from quantstack.execution.risk_gate import RiskGate, RiskLimits

        mock_pg_conn.return_value.__enter__ = MagicMock(
            side_effect=Exception("DB down")
        )
        mock_pg_conn.return_value.__exit__ = MagicMock(return_value=False)

        limits = RiskLimits(
            max_position_pct=1.0,
            max_position_notional=1_000_000.0,
            max_gross_exposure_pct=10.0,
            min_daily_volume=1,
            max_participation_pct=1.0,
        )
        portfolio = _make_mock_portfolio()
        gate = RiskGate(limits=limits, portfolio=portfolio)

        verdict = gate.check(
            symbol="TEST",
            side="buy",
            quantity=100,
            current_price=10.0,
            daily_volume=1_000_000,
        )

        assert verdict.approved is True
        assert verdict.approved_quantity == 100

    @patch("quantstack.execution.risk_gate.pg_conn")
    def test_quality_bonus_scalar(self, mock_pg_conn):
        """Excellent execution (scalar=1.1) gives a size bonus."""
        from quantstack.execution.risk_gate import RiskGate, RiskLimits

        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchone.return_value = (1.1,)
        mock_pg_conn.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_pg_conn.return_value.__exit__ = MagicMock(return_value=False)

        limits = RiskLimits(
            max_position_pct=1.0,
            max_position_notional=1_000_000.0,
            max_gross_exposure_pct=10.0,
            min_daily_volume=1,
            max_participation_pct=1.0,
        )
        portfolio = _make_mock_portfolio()
        gate = RiskGate(limits=limits, portfolio=portfolio)

        verdict = gate.check(
            symbol="TEST",
            side="buy",
            quantity=100,
            current_price=10.0,
            daily_volume=1_000_000,
        )

        assert verdict.approved is True
        # 100 * 1.1 = 110
        assert verdict.approved_quantity == 110
