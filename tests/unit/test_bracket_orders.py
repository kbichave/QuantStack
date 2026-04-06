"""Tests for bracket order support across broker implementations."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from quantstack.execution.paper_broker import BrokerProtocol, Fill, OrderRequest, PaperBroker


# ---------------------------------------------------------------------------
# Fill model extension
# ---------------------------------------------------------------------------


class TestFillBracketFields:

    def test_fill_bracket_fields_default_none(self):
        """Fill without bracket fields works (backward compat)."""
        fill = Fill(
            order_id="test-1",
            symbol="SPY",
            side="buy",
            requested_quantity=10,
            filled_quantity=10,
            fill_price=450.0,
            slippage_bps=5.0,
        )
        assert fill.bracket_stop_order_id is None
        assert fill.bracket_tp_order_id is None

    def test_fill_bracket_fields_populated(self):
        """Fill with bracket order IDs stores them correctly."""
        fill = Fill(
            order_id="test-1",
            symbol="SPY",
            side="buy",
            requested_quantity=10,
            filled_quantity=10,
            fill_price=450.0,
            slippage_bps=5.0,
            bracket_stop_order_id="stop-123",
            bracket_tp_order_id="tp-456",
        )
        assert fill.bracket_stop_order_id == "stop-123"
        assert fill.bracket_tp_order_id == "tp-456"


# ---------------------------------------------------------------------------
# Protocol compliance — supports_bracket_orders()
# ---------------------------------------------------------------------------


class TestBrokerBracketSupport:

    def test_paper_broker_no_bracket_support(self):
        """PaperBroker.supports_bracket_orders() returns False."""
        broker = PaperBroker.__new__(PaperBroker)
        assert broker.supports_bracket_orders() is False

    def test_etrade_broker_no_bracket_support(self):
        """EtradeBroker.supports_bracket_orders() returns False."""
        with patch("quantstack.execution.etrade_broker.ETradeAuthManager"), \
             patch("quantstack.execution.etrade_broker.ETradeClient"), \
             patch("quantstack.execution.etrade_broker.get_portfolio_state"):
            from quantstack.execution.etrade_broker import EtradeBroker
            broker = EtradeBroker.__new__(EtradeBroker)
            assert broker.supports_bracket_orders() is False

    def test_alpaca_broker_has_bracket_support(self):
        """AlpacaBroker.supports_bracket_orders() returns True."""
        with patch("quantstack.execution.alpaca_broker.TradingClient"):
            from quantstack.execution.alpaca_broker import AlpacaBroker
            broker = AlpacaBroker.__new__(AlpacaBroker)
            assert broker.supports_bracket_orders() is True

    def test_paper_broker_satisfies_protocol(self):
        """PaperBroker still satisfies BrokerProtocol after adding method."""
        broker = PaperBroker.__new__(PaperBroker)
        assert isinstance(broker, BrokerProtocol)


# ---------------------------------------------------------------------------
# AlpacaBroker.execute_bracket()
# ---------------------------------------------------------------------------


class TestAlpacaBracketExecution:

    @pytest.fixture
    def alpaca_env(self, monkeypatch):
        """AlpacaBroker with mocked TradingClient — patches stay active during test."""
        mock_settings = MagicMock()
        mock_settings.alpaca.api_key = "test-key"
        mock_settings.alpaca.secret_key = "test-secret"
        mock_settings.alpaca.paper = True

        patches = [
            patch("quantstack.execution.alpaca_broker.get_settings", return_value=mock_settings),
            patch("quantstack.execution.alpaca_broker.get_portfolio_state"),
            patch("quantstack.execution.alpaca_broker.get_kill_switch"),
            patch("quantstack.execution.alpaca_broker.TradingClient"),
        ]
        mocks = [p.start() for p in patches]
        mock_client = mocks[3].return_value  # TradingClient mock instance

        from quantstack.execution.alpaca_broker import AlpacaBroker
        broker = AlpacaBroker()

        yield broker, mock_client

        for p in patches:
            p.stop()

    def test_execute_bracket_submits_bracket_order(self, alpaca_env):
        """execute_bracket() submits order with bracket params."""
        from alpaca.trading.enums import OrderSide, OrderStatus
        broker, mock_client = alpaca_env

        mock_order = MagicMock()
        mock_order.id = "parent-123"
        mock_order.legs = [MagicMock(id="stop-leg-1"), MagicMock(id="tp-leg-2")]
        mock_order.status = OrderStatus.FILLED
        mock_order.filled_qty = "10"
        mock_order.filled_avg_price = "450.00"
        mock_order.symbol = "SPY"
        mock_order.side = OrderSide.BUY
        mock_order.qty = "10"
        mock_client.submit_order.return_value = mock_order
        mock_client.get_order_by_id.return_value = mock_order

        req = OrderRequest(symbol="SPY", side="buy", quantity=10, current_price=450.0)
        fill = broker.execute_bracket(req, stop_price=440.0, take_profit_price=470.0)

        mock_client.submit_order.assert_called_once()
        call_kwargs = mock_client.submit_order.call_args
        order_data = call_kwargs[1].get("order_data") or call_kwargs[0][0]
        assert order_data.order_class.value == "bracket"

    def test_execute_bracket_fallback_on_failure(self, alpaca_env):
        """execute_bracket() falls back to plain execute() on API failure."""
        from alpaca.trading.enums import OrderSide, OrderStatus
        broker, mock_client = alpaca_env

        fallback_order = MagicMock()
        fallback_order.id = "fallback-123"
        fallback_order.status = OrderStatus.FILLED
        fallback_order.filled_qty = "10"
        fallback_order.filled_avg_price = "450.00"
        fallback_order.symbol = "SPY"
        fallback_order.side = OrderSide.BUY
        fallback_order.qty = "10"
        fallback_order.legs = None

        mock_client.submit_order.side_effect = [
            Exception("Bracket API error"),
            fallback_order,
        ]
        mock_client.get_order_by_id.return_value = fallback_order

        req = OrderRequest(symbol="SPY", side="buy", quantity=10, current_price=450.0)
        fill = broker.execute_bracket(req, stop_price=440.0, take_profit_price=470.0)

        assert mock_client.submit_order.call_count == 2
        assert not fill.rejected

    def test_execute_bracket_requires_stop_price(self, alpaca_env):
        """execute_bracket() raises ValueError without stop_price."""
        broker, _ = alpaca_env
        req = OrderRequest(symbol="SPY", side="buy", quantity=10, current_price=450.0)
        with pytest.raises(ValueError, match="stop_price"):
            broker.execute_bracket(req, stop_price=None, take_profit_price=470.0)


# ---------------------------------------------------------------------------
# TradeService wiring
# ---------------------------------------------------------------------------


class TestTradeServiceBracket:

    @pytest.fixture
    def trade_deps(self):
        """Common trade_service dependencies as mocks."""
        portfolio = MagicMock()
        # Return a position with a valid current_price so the flow doesn't exit early
        mock_pos = MagicMock()
        mock_pos.current_price = 450.0
        portfolio.get_position.return_value = mock_pos
        portfolio.get_positions.return_value = []
        portfolio.get_snapshot.return_value = MagicMock()

        risk_gate = MagicMock()
        verdict = MagicMock()
        verdict.approved = True
        verdict.violations = []
        verdict.approved_quantity = 10
        risk_gate.check.return_value = verdict

        audit = MagicMock()
        kill_switch = MagicMock()
        kill_switch.is_active = False

        return portfolio, risk_gate, audit, kill_switch

    @pytest.mark.asyncio
    async def test_bracket_used_when_supported(self, trade_deps):
        """execute_trade() uses execute_bracket() when broker supports it."""
        portfolio, risk_gate, audit, kill_switch = trade_deps

        broker = MagicMock()
        broker.supports_bracket_orders.return_value = True
        fill = Fill(
            order_id="t1", symbol="SPY", side="buy",
            requested_quantity=10, filled_quantity=10,
            fill_price=450.0, slippage_bps=5.0,
            bracket_stop_order_id="stop-1", bracket_tp_order_id="tp-1",
        )
        broker.execute_bracket.return_value = fill

        from quantstack.execution.trade_service import execute_trade
        with patch("quantstack.execution.trade_service.db_conn"), \
             patch("quantstack.execution.trade_service._fire_hook"):
            result = await execute_trade(
                portfolio=portfolio, risk_gate=risk_gate, broker=broker,
                audit=audit, kill_switch=kill_switch, session_id="test",
                symbol="SPY", action="buy", reasoning="test",
                confidence=0.9, quantity=10,
                stop_price=440.0, target_price=470.0,
            )

        broker.execute_bracket.assert_called_once()
        broker.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_plain_execute_when_no_bracket_support(self, trade_deps):
        """execute_trade() uses plain execute() when broker doesn't support brackets."""
        portfolio, risk_gate, audit, kill_switch = trade_deps

        broker = MagicMock()
        broker.supports_bracket_orders.return_value = False
        fill = Fill(
            order_id="t1", symbol="SPY", side="buy",
            requested_quantity=10, filled_quantity=10,
            fill_price=450.0, slippage_bps=5.0,
        )
        broker.execute.return_value = fill

        from quantstack.execution.trade_service import execute_trade
        with patch("quantstack.execution.trade_service.db_conn"), \
             patch("quantstack.execution.trade_service._fire_hook"):
            result = await execute_trade(
                portfolio=portfolio, risk_gate=risk_gate, broker=broker,
                audit=audit, kill_switch=kill_switch, session_id="test",
                symbol="SPY", action="buy", reasoning="test",
                confidence=0.9, quantity=10,
                stop_price=440.0, target_price=470.0,
            )

        broker.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_plain_execute_when_no_stop_price(self, trade_deps):
        """execute_trade() uses plain execute() when stop_price is missing."""
        portfolio, risk_gate, audit, kill_switch = trade_deps

        broker = MagicMock()
        broker.supports_bracket_orders.return_value = True
        fill = Fill(
            order_id="t1", symbol="SPY", side="buy",
            requested_quantity=10, filled_quantity=10,
            fill_price=450.0, slippage_bps=5.0,
        )
        broker.execute.return_value = fill

        from quantstack.execution.trade_service import execute_trade
        with patch("quantstack.execution.trade_service.db_conn"), \
             patch("quantstack.execution.trade_service._fire_hook"):
            result = await execute_trade(
                portfolio=portfolio, risk_gate=risk_gate, broker=broker,
                audit=audit, kill_switch=kill_switch, session_id="test",
                symbol="SPY", action="buy", reasoning="test",
                confidence=0.9, quantity=10,
                stop_price=None, target_price=470.0,
            )

        broker.execute.assert_called_once()
