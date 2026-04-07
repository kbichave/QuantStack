"""Unit tests for RiskGate._check_market_hours() trading window enforcement."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock, patch


@dataclass
class MockPosition:
    quantity: int = 0
    current_price: float = 100.0
    symbol: str = "SPY"


def _make_gate_with_position(position_qty: int | None = None):
    """Create a RiskGate with a mocked portfolio.

    If position_qty is None, get_position returns None (no existing position).
    Otherwise returns a MockPosition with that quantity.
    """
    mock_portfolio = MagicMock()
    mock_snapshot = MagicMock()
    mock_snapshot.total_equity = 100_000.0
    mock_snapshot.daily_pnl = 0.0
    mock_portfolio.get_snapshot.return_value = mock_snapshot
    mock_portfolio.get_positions.return_value = []

    if position_qty is not None:
        mock_portfolio.get_position.return_value = MockPosition(quantity=position_qty)
    else:
        mock_portfolio.get_position.return_value = None

    from quantstack.execution.risk_gate import RiskGate, RiskLimits

    gate = RiskGate.__new__(RiskGate)
    gate.limits = RiskLimits()
    gate._portfolio = mock_portfolio
    gate._daily_halted = None
    return gate


class TestCheckMarketHours:
    def test_market_mode_allows_all(self):
        """During market hours, _check_market_hours returns None (passthrough)."""
        gate = _make_gate_with_position(None)
        with patch("quantstack.runners.get_operating_mode") as mock_mode:
            from quantstack.runners import OperatingMode

            mock_mode.return_value = OperatingMode.MARKET
            result = gate._check_market_hours("SPY", "buy")
        assert result is None

    def test_extended_rejects_new_long(self):
        """Buy with no position = new long entry -> rejected in extended hours."""
        gate = _make_gate_with_position(None)  # no existing position
        with patch("quantstack.runners.get_operating_mode") as mock_mode:
            from quantstack.runners import OperatingMode

            mock_mode.return_value = OperatingMode.EXTENDED
            result = gate._check_market_hours("SPY", "buy")
        assert result is not None
        assert not result.approved
        assert "Extended hours" in result.reason

    def test_extended_rejects_new_short(self):
        """Sell with no position = new short -> rejected in extended hours."""
        gate = _make_gate_with_position(None)
        with patch("quantstack.runners.get_operating_mode") as mock_mode:
            from quantstack.runners import OperatingMode

            mock_mode.return_value = OperatingMode.EXTENDED
            result = gate._check_market_hours("SPY", "sell")
        assert result is not None
        assert not result.approved

    def test_extended_allows_exit_sell(self):
        """Sell against long position = closing trade -> allowed in extended hours."""
        gate = _make_gate_with_position(100)  # long 100 shares
        with patch("quantstack.runners.get_operating_mode") as mock_mode:
            from quantstack.runners import OperatingMode

            mock_mode.return_value = OperatingMode.EXTENDED
            result = gate._check_market_hours("SPY", "sell")
        assert result is None  # allowed

    def test_extended_allows_cover_buy(self):
        """Buy against short position = covering -> allowed in extended hours."""
        gate = _make_gate_with_position(-50)  # short 50 shares
        with patch("quantstack.runners.get_operating_mode") as mock_mode:
            from quantstack.runners import OperatingMode

            mock_mode.return_value = OperatingMode.EXTENDED
            result = gate._check_market_hours("SPY", "buy")
        assert result is None  # allowed

    def test_overnight_rejects_new_entry(self):
        """New entry during overnight -> rejected."""
        gate = _make_gate_with_position(None)
        with patch("quantstack.runners.get_operating_mode") as mock_mode:
            from quantstack.runners import OperatingMode

            mock_mode.return_value = OperatingMode.OVERNIGHT
            result = gate._check_market_hours("AAPL", "buy")
        assert result is not None
        assert "overnight" in result.reason.lower()

    def test_weekend_rejects_new_entry(self):
        """New entry during weekend -> rejected."""
        gate = _make_gate_with_position(None)
        with patch("quantstack.runners.get_operating_mode") as mock_mode:
            from quantstack.runners import OperatingMode

            mock_mode.return_value = OperatingMode.WEEKEND
            result = gate._check_market_hours("AAPL", "buy")
        assert result is not None
        assert "weekend" in result.reason.lower()

    def test_sell_on_existing_short_is_rejected(self):
        """Selling more when already short = increasing exposure -> rejected."""
        gate = _make_gate_with_position(-50)  # short 50
        with patch("quantstack.runners.get_operating_mode") as mock_mode:
            from quantstack.runners import OperatingMode

            mock_mode.return_value = OperatingMode.EXTENDED
            result = gate._check_market_hours("SPY", "sell")
        assert result is not None
        assert not result.approved

    def test_buy_on_existing_long_is_rejected(self):
        """Buying more when already long = increasing exposure -> rejected."""
        gate = _make_gate_with_position(100)  # long 100
        with patch("quantstack.runners.get_operating_mode") as mock_mode:
            from quantstack.runners import OperatingMode

            mock_mode.return_value = OperatingMode.EXTENDED
            result = gate._check_market_hours("SPY", "buy")
        assert result is not None
        assert not result.approved
