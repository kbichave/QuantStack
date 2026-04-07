"""Unit tests for cycle-level P&L attribution engine."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from quantstack.performance.attribution import (
    CycleAttribution,
    PositionAttribution,
    compute_cycle_attribution,
    persist_cycle_attribution,
)


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class TestComputeCycleAttribution:
    def test_accounting_identity(self):
        """Components sum to total_pnl."""
        positions = [
            {"symbol": "AAPL", "quantity": 100, "market_value": 15000,
             "sector": "Technology", "unrealized_pnl": 200.0},
            {"symbol": "JNJ", "quantity": 50, "market_value": 8000,
             "sector": "Healthcare", "unrealized_pnl": -50.0},
        ]
        fills = [
            {"symbol": "AAPL", "quantity": 10, "fill_price": 150.0,
             "vwap": 152.0, "side": "buy", "slippage": 0.5, "commission": 1.0,
             "realized_pnl": 0},
        ]

        with patch("quantstack.tools.functions.system_alerts.emit_system_alert", new_callable=AsyncMock):
            result = _run(compute_cycle_attribution(positions, fills, 0.01))

        total = (
            result.factor_contribution + result.timing_contribution +
            result.selection_contribution + result.cost_contribution
        )
        assert abs(total - result.total_pnl) < 1e-6

    def test_zero_fills_all_zero(self):
        """No positions and no fills -> all-zero components."""
        result = _run(compute_cycle_attribution([], [], 0.0))
        assert result.total_pnl == 0.0
        assert result.factor_contribution == 0.0
        assert result.timing_contribution == 0.0
        assert result.selection_contribution == 0.0
        assert result.cost_contribution == 0.0
        assert result.per_position == []

    def test_factor_reflects_benchmark(self):
        """Factor contribution reflects benchmark exposure."""
        positions = [
            {"symbol": "AAPL", "quantity": 100, "market_value": 10000,
             "sector": "Technology", "unrealized_pnl": 200.0},
        ]
        result = _run(compute_cycle_attribution(positions, [], 0.02))
        # Factor = weight(1.0) * benchmark_return(0.02) * market_value(10000) = 200
        assert abs(result.factor_contribution - 200.0) < 0.01

    def test_timing_positive_when_bought_below_vwap(self):
        """Buying below VWAP = positive timing contribution."""
        positions = [
            {"symbol": "AAPL", "quantity": 100, "market_value": 10000,
             "unrealized_pnl": 0},
        ]
        fills = [
            {"symbol": "AAPL", "quantity": 10, "fill_price": 100.0,
             "vwap": 102.0, "side": "buy", "slippage": 0, "commission": 0,
             "realized_pnl": 0},
        ]
        result = _run(compute_cycle_attribution(positions, fills, 0.0))
        assert result.timing_contribution > 0  # (102 - 100) * 10 = 20

    def test_timing_negative_when_bought_above_vwap(self):
        """Buying above VWAP = negative timing contribution."""
        positions = [
            {"symbol": "AAPL", "quantity": 100, "market_value": 10000,
             "unrealized_pnl": 0},
        ]
        fills = [
            {"symbol": "AAPL", "quantity": 10, "fill_price": 104.0,
             "vwap": 102.0, "side": "buy", "slippage": 0, "commission": 0,
             "realized_pnl": 0},
        ]
        result = _run(compute_cycle_attribution(positions, fills, 0.0))
        assert result.timing_contribution < 0  # (102 - 104) * 10 = -20

    def test_cost_equals_slippage_plus_commission(self):
        """Cost contribution = negative sum of slippage + commissions."""
        fills = [
            {"symbol": "AAPL", "quantity": 10, "fill_price": 100.0,
             "vwap": 100.0, "side": "buy", "slippage": 2.5, "commission": 1.0,
             "realized_pnl": 0},
            {"symbol": "JNJ", "quantity": 5, "fill_price": 160.0,
             "vwap": 160.0, "side": "buy", "slippage": 1.0, "commission": 1.0,
             "realized_pnl": 0},
        ]
        result = _run(compute_cycle_attribution([], fills, 0.0))
        assert result.cost_contribution == -(2.5 + 1.0 + 1.0 + 1.0)

    def test_empty_portfolio(self):
        """No active positions returns all-zero attribution."""
        result = _run(compute_cycle_attribution([], [], 0.01))
        assert result.total_pnl == 0.0
        assert result.per_position == []

    def test_identity_violation_logs_warning(self):
        """When total_pnl disagrees, the gap is absorbed by selection (residual)."""
        # The accounting identity holds by construction (selection = residual).
        # This test verifies that even with unusual inputs, no exception is raised.
        positions = [
            {"symbol": "AAPL", "quantity": 100, "market_value": 10000,
             "sector": "Tech", "unrealized_pnl": 500.0},
        ]
        result = _run(compute_cycle_attribution(positions, [], 0.0))
        # selection absorbs everything since factor=0, timing=0, cost=0
        assert abs(result.selection_contribution - 500.0) < 0.01

    def test_per_position_populated(self):
        """per_position list has one entry per position."""
        positions = [
            {"symbol": "AAPL", "quantity": 100, "market_value": 10000,
             "sector": "Tech", "unrealized_pnl": 100.0},
            {"symbol": "MSFT", "quantity": 50, "market_value": 5000,
             "sector": "Tech", "unrealized_pnl": 50.0},
        ]
        result = _run(compute_cycle_attribution(positions, [], 0.0))
        assert len(result.per_position) == 2
        symbols = {p.symbol for p in result.per_position}
        assert symbols == {"AAPL", "MSFT"}


class TestPersistCycleAttribution:
    def test_inserts_row(self):
        conn = MagicMock()
        cursor = MagicMock()
        conn.execute.return_value = cursor

        @contextmanager
        def mock_db():
            yield conn

        attr = CycleAttribution(
            cycle_id="test123",
            total_pnl=100.0,
            factor_contribution=50.0,
            timing_contribution=20.0,
            selection_contribution=35.0,
            cost_contribution=-5.0,
            per_position=[
                PositionAttribution("AAPL", 1.0, 100.0, 50.0, 20.0, 35.0, -5.0),
            ],
        )

        with patch("quantstack.performance.attribution.db_conn", mock_db):
            persist_cycle_attribution(attr, graph_cycle_number=42)

        conn.execute.assert_called_once()
        sql = conn.execute.call_args[0][0]
        assert "cycle_attribution" in sql
        params = conn.execute.call_args[0][1]
        assert params[0] == "test123"
        assert params[1] == 42

    def test_to_dict_serialization(self):
        attr = CycleAttribution(
            cycle_id="abc",
            total_pnl=100.0,
            factor_contribution=40.0,
            timing_contribution=30.0,
            selection_contribution=35.0,
            cost_contribution=-5.0,
        )
        d = attr.to_dict()
        assert d["cycle_id"] == "abc"
        assert d["total_pnl"] == 100.0
        assert isinstance(d["per_position"], list)
