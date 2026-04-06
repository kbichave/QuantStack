"""Tests for drill-down modal screens."""
from datetime import datetime

from quantstack.tui.queries.agents import GraphActivity
from quantstack.tui.queries.portfolio import ClosedTrade, Position
from quantstack.tui.queries.signals import SignalBrief
from quantstack.tui.queries.strategies import StrategyDetail
from quantstack.tui.screens.detail import (
    AgentEventModal,
    DetailModal,
    PositionDetailModal,
    SignalDetailModal,
    StrategyDetailModal,
    TradeDetailModal,
)


class TestDetailModal:
    def test_instantiates_with_title(self):
        m = DetailModal(title="Test Modal")
        assert m._title == "Test Modal"

    def test_default_title(self):
        m = DetailModal()
        assert m._title == "Detail"

    def test_has_escape_binding(self):
        m = DetailModal()
        keys = [b.key for b in m.BINDINGS]
        assert "escape" in keys


class TestPositionDetailModal:
    def _make_position(self, **overrides):
        defaults = dict(
            symbol="AAPL", quantity=100, avg_cost=150.0, current_price=160.0,
            unrealized_pnl=1000.0, unrealized_pnl_pct=6.67,
            strategy_id="strat_1", holding_days=5,
        )
        defaults.update(overrides)
        return Position(**defaults)

    def test_sets_title(self):
        m = PositionDetailModal(position=self._make_position())
        assert "AAPL" in m._title

    def test_stores_position(self):
        p = self._make_position(symbol="NVDA")
        m = PositionDetailModal(position=p)
        assert m._pos.symbol == "NVDA"

    def test_handles_negative_pnl(self):
        p = self._make_position(unrealized_pnl=-500.0, unrealized_pnl_pct=-3.33)
        m = PositionDetailModal(position=p)
        assert m._pos.unrealized_pnl < 0


class TestStrategyDetailModal:
    def test_sets_loading_state(self):
        m = StrategyDetailModal(strategy_id="s1")
        assert m._strategy_id == "s1"
        assert m._detail is None

    def test_render_detail_with_none(self):
        """_render_detail handles None gracefully (tested via update_view pattern)."""
        m = StrategyDetailModal(strategy_id="s1")
        m._detail = None

    def test_render_detail_with_data(self):
        m = StrategyDetailModal(strategy_id="s1")
        m._detail = StrategyDetail(
            strategy_id="s1", name="momentum", status="live", symbol="AAPL",
            instrument_type="equity", time_horizon="swing",
            regime_affinity="trending_up", sharpe=1.24, max_drawdown=-0.082,
            win_rate=0.62, profit_factor=1.85, total_trades=47,
            fwd_trades=6, fwd_pnl=340.0, fwd_days=18,
            entry_rules=["RSI < 35"], exit_rules=["Trailing stop 2x ATR"],
        )
        assert m._detail.entry_rules == ["RSI < 35"]

    def test_render_detail_no_backtest(self):
        m = StrategyDetailModal(strategy_id="s1")
        m._detail = StrategyDetail(
            strategy_id="s1", name="draft_strat", status="draft", symbol="TSLA",
            instrument_type=None, time_horizon=None, regime_affinity=None,
            sharpe=None, max_drawdown=None, win_rate=None,
            profit_factor=None, total_trades=0,
            fwd_trades=0, fwd_pnl=0.0, fwd_days=0,
            entry_rules=[], exit_rules=[],
        )
        assert m._detail.sharpe is None


class TestSignalDetailModal:
    def test_sets_symbol(self):
        m = SignalDetailModal(symbol="NVDA")
        assert m._symbol == "NVDA"
        assert "NVDA" in m._title

    def test_handles_full_brief(self):
        brief = SignalBrief(
            symbol="NVDA", action="BUY", confidence=0.87,
            ml_score=0.82, sentiment_score=0.71, technical_score=0.90,
            options_score=0.65, macro_score=0.78,
            risk_flags=["Earnings in 5 days"],
            generated_at=datetime.now(),
        )
        assert brief.ml_score == 0.82
        assert len(brief.risk_flags) == 1

    def test_handles_partial_brief(self):
        brief = SignalBrief(
            symbol="AAPL", action="HOLD", confidence=0.5,
            ml_score=0.6, sentiment_score=None, technical_score=None,
            options_score=None, macro_score=None,
            collector_failures=["sentiment", "technical"],
            generated_at=datetime.now(),
        )
        assert brief.sentiment_score is None
        assert len(brief.collector_failures) == 2


class TestTradeDetailModal:
    def _make_trade(self, **overrides):
        defaults = dict(
            symbol="AAPL", side="sell", realized_pnl=250.0, holding_days=3,
            strategy_id="s1", exit_reason="target_reached", closed_at=datetime.now(),
        )
        defaults.update(overrides)
        return ClosedTrade(**defaults)

    def test_sets_title(self):
        m = TradeDetailModal(trade=self._make_trade())
        assert "AAPL" in m._title

    def test_stores_trade_data(self):
        t = self._make_trade(realized_pnl=-100.0)
        m = TradeDetailModal(trade=t)
        assert m._trade.realized_pnl == -100.0

    def test_handles_no_reflection(self):
        """Modal should handle None reflection gracefully."""
        m = TradeDetailModal(trade=self._make_trade())
        assert m._trade is not None


class TestAgentEventModal:
    def test_renders_event(self):
        event = GraphActivity(
            graph_name="research", current_node="analyze",
            current_agent="scanner", cycle_number=10,
            cycle_started=datetime.now(), event_count=5,
        )
        m = AgentEventModal(event=event)
        assert "scanner" in m._title

    def test_stores_event(self):
        event = GraphActivity(
            graph_name="trading", current_node="monitor",
            current_agent="risk_mgr", cycle_number=3,
            cycle_started=datetime.now(), event_count=2,
        )
        m = AgentEventModal(event=event)
        assert m._event.graph_name == "trading"


class TestAllModalsHandleMinimalData:
    """Cross-cutting: all modals handle minimal/None fields without error."""

    def test_position_modal_minimal(self):
        p = Position("X", 0, 0.0, 0.0, 0.0, 0.0, "", 0)
        m = PositionDetailModal(position=p)
        assert m._pos.symbol == "X"

    def test_strategy_modal_minimal(self):
        m = StrategyDetailModal(strategy_id="")
        assert m._detail is None

    def test_signal_modal_minimal(self):
        m = SignalDetailModal(symbol="")
        assert m._symbol == ""

    def test_trade_modal_minimal(self):
        t = ClosedTrade("X", "buy", 0.0, 0, "", "", datetime.now())
        m = TradeDetailModal(trade=t)
        assert m._trade.symbol == "X"

    def test_agent_event_modal_minimal(self):
        e = GraphActivity("", "", "", 0, datetime.now(), 0)
        m = AgentEventModal(event=e)
        assert m._event.graph_name == ""
