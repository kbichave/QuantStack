"""Tests for execution monitor bookkeeping fields on Position + PortfolioState."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from quantstack.execution.portfolio_state import Position


class TestPositionMonitorFields:

    def test_monitor_fields_default_to_none(self):
        """New monitor fields default to None."""
        pos = Position(symbol="SPY", quantity=100, avg_cost=450.0)
        assert pos.monitor_last_check is None
        assert pos.monitor_hwm is None

    def test_monitor_fields_accept_values(self):
        """Monitor fields can be set explicitly."""
        now = datetime.now()
        pos = Position(
            symbol="SPY",
            quantity=100,
            avg_cost=450.0,
            monitor_last_check=now,
            monitor_hwm=455.0,
        )
        assert pos.monitor_last_check == now
        assert pos.monitor_hwm == 455.0


class TestUpdateMonitorState:
    """Test update_monitor_state() with mocked DB connections."""

    @pytest.fixture
    def state(self):
        """PortfolioState with a mocked PgConnection."""
        from quantstack.execution.portfolio_state import PortfolioState

        conn = MagicMock()
        # Seed cash check — simulate empty table
        conn.execute.return_value.fetchone.return_value = (0,)
        with patch("quantstack.execution.portfolio_state.run_migrations"):
            ps = PortfolioState(conn=conn)
        return ps, conn

    def test_update_monitor_state_existing_position(self, state):
        """update_monitor_state writes hwm and last_check for existing position."""
        ps, conn = state
        pos = Position(symbol="SPY", quantity=100, avg_cost=450.0, side="long")
        # Mock get_position to return the position
        ps.get_position = MagicMock(return_value=pos)

        now = datetime.now()
        result = ps.update_monitor_state("SPY", hwm=455.0, last_check=now)
        assert result is True
        # Verify the UPDATE was called
        conn.execute.assert_called()

    def test_update_monitor_state_nonexistent_position(self, state):
        """update_monitor_state returns False for missing position."""
        ps, conn = state
        ps.get_position = MagicMock(return_value=None)

        now = datetime.now()
        result = ps.update_monitor_state("AAPL", hwm=175.0, last_check=now)
        assert result is False

    def test_monitor_state_does_not_affect_stops(self, state):
        """update_monitor_state SQL only touches monitor fields + last_updated."""
        ps, conn = state
        pos = Position(
            symbol="SPY",
            quantity=100,
            avg_cost=450.0,
            stop_price=440.0,
            target_price=470.0,
            trailing_stop=5.0,
        )
        ps.get_position = MagicMock(return_value=pos)

        now = datetime.now()
        ps.update_monitor_state("SPY", hwm=455.0, last_check=now)

        # The SQL should only update monitor_hwm, monitor_last_check, last_updated
        update_call = [
            c for c in conn.execute.call_args_list
            if isinstance(c[0][0], str) and "monitor_hwm" in c[0][0]
        ]
        assert len(update_call) == 1
        sql = update_call[0][0][0]
        assert "stop_price" not in sql
        assert "target_price" not in sql
        assert "trailing_stop" not in sql
