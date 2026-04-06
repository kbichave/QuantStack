"""Drill-down modal screens for detail views."""
from __future__ import annotations

from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Static

from loguru import logger

from quantstack.db import pg_conn
from quantstack.tui.queries.agents import GraphActivity
from quantstack.tui.queries.portfolio import (
    ClosedTrade,
    Position,
    fetch_trade_decision,
    fetch_trade_reflection,
)
from quantstack.tui.queries.signals import SignalBrief, fetch_signal_brief
from quantstack.tui.queries.strategies import StrategyDetail, fetch_strategy_detail


class DetailModal(ModalScreen):
    """Base modal screen with semi-transparent overlay and ESC to dismiss.

    Subclasses override compose_content() to yield their specific widgets.
    """

    BINDINGS = [Binding("escape", "dismiss", "Close")]

    def __init__(self, title: str = "Detail") -> None:
        super().__init__()
        self._title = title

    def compose(self) -> ComposeResult:
        with Vertical(id="modal-container"):
            yield Static(self._title, id="modal-title")
            with VerticalScroll(id="modal-body"):
                yield from self.compose_content()
            yield Static("ESC to close", id="modal-footer")

    def compose_content(self) -> ComposeResult:
        yield Static("No content")


class PositionDetailModal(DetailModal):
    """Detail view for an open position."""

    def __init__(self, position: Position) -> None:
        super().__init__(title=f"{position.symbol} — Position Detail")
        self._pos = position

    def compose_content(self) -> ComposeResult:
        p = self._pos
        pnl_sign = "+" if p.unrealized_pnl >= 0 else ""
        lines = [
            f"Symbol:          {p.symbol}",
            f"Quantity:        {p.quantity:g}",
            f"Entry Price:     ${p.avg_cost:,.2f}",
            f"Current Price:   ${p.current_price:,.2f}",
            "",
            f"P&L:             {pnl_sign}${p.unrealized_pnl:,.2f} ({pnl_sign}{p.unrealized_pnl_pct:.2f}%)",
            f"Strategy:        {p.strategy_id}",
            f"Days Held:       {p.holding_days}",
        ]
        yield Static("\n".join(lines), id="detail-content")


class StrategyDetailModal(DetailModal):
    """Detail view for a strategy with backtest and forward test metrics."""

    def __init__(self, strategy_id: str) -> None:
        super().__init__(title="Strategy Detail")
        self._strategy_id = strategy_id
        self._detail: StrategyDetail | None = None

    def compose_content(self) -> ComposeResult:
        yield Static("Loading...", id="detail-content")

    def on_mount(self) -> None:
        self._fetch_detail()

    @work(thread=True)
    def _fetch_detail(self) -> None:
        try:
            with pg_conn() as conn:
                self._detail = fetch_strategy_detail(conn, self._strategy_id)
        except Exception:
            logger.warning("StrategyDetailModal fetch failed", exc_info=True)
        self.app.call_from_thread(self._render_detail)

    def _render_detail(self) -> None:
        content = self.query_one("#detail-content", Static)
        d = self._detail
        if d is None:
            content.update("No strategy data available")
            return
        self._title = f"{d.symbol} {d.name} — Strategy Detail"
        self.query_one("#modal-title", Static).update(self._title)

        lines = [
            f"Status:          {d.status}",
            f"Symbol:          {d.symbol}",
            f"Type:            {d.instrument_type or '—'}",
            f"Horizon:         {d.time_horizon or '—'}",
            f"Regime Affinity: {d.regime_affinity or '—'}",
            "",
            "── Backtest Results ──",
            f"Sharpe:          {d.sharpe:.2f}" if d.sharpe is not None else "Sharpe:          —",
            f"Max Drawdown:    {d.max_drawdown:.1%}" if d.max_drawdown is not None else "Max Drawdown:    —",
            f"Win Rate:        {d.win_rate:.0%}" if d.win_rate is not None else "Win Rate:        —",
            f"Profit Factor:   {d.profit_factor:.2f}" if d.profit_factor is not None else "Profit Factor:   —",
            f"Total Trades:    {d.total_trades}",
            "",
            "── Forward Test ──",
            f"Days Elapsed:    {d.fwd_days}",
            f"Trades Taken:    {d.fwd_trades}",
            f"Cumulative P&L:  ${d.fwd_pnl:+,.2f}",
        ]

        if d.entry_rules:
            lines.append("")
            lines.append("── Entry Rules ──")
            for rule in d.entry_rules:
                lines.append(f"  * {rule}")

        if d.exit_rules:
            lines.append("")
            lines.append("── Exit Rules ──")
            for rule in d.exit_rules:
                lines.append(f"  * {rule}")

        if not d.entry_rules and not d.exit_rules:
            lines.append("")
            lines.append("No entry/exit rules defined")

        content.update("\n".join(lines))


class SignalDetailModal(DetailModal):
    """Detail view for a signal brief with contributing factors and risk flags."""

    def __init__(self, symbol: str) -> None:
        super().__init__(title=f"{symbol} — Signal Detail")
        self._symbol = symbol

    def compose_content(self) -> ComposeResult:
        yield Static("Loading...", id="detail-content")

    def on_mount(self) -> None:
        self._fetch_brief()

    @work(thread=True)
    def _fetch_brief(self) -> None:
        brief: SignalBrief | None = None
        try:
            with pg_conn() as conn:
                brief = fetch_signal_brief(conn, self._symbol)
        except Exception:
            logger.warning("SignalDetailModal fetch failed", exc_info=True)
        self.app.call_from_thread(self._render_brief, brief)

    def _render_brief(self, brief: SignalBrief | None) -> None:
        content = self.query_one("#detail-content", Static)
        if brief is None:
            content.update("No signal brief available")
            return

        def _score(val: float | None, label: str) -> str:
            if val is None:
                return f"{label + ':':22s} N/A"
            return f"{label + ':':22s} {val:.2f}"

        lines = [
            f"Action:          {brief.action}",
            f"Confidence:      {brief.confidence:.0%}",
            f"Generated:       {brief.generated_at}",
            "",
            "── Contributing Factors ──",
            _score(brief.ml_score, "ML Prediction"),
            _score(brief.sentiment_score, "Sentiment Score"),
            _score(brief.technical_score, "Technical Score"),
            _score(brief.options_score, "Options Flow"),
            _score(brief.macro_score, "Macro Alignment"),
        ]

        if brief.risk_flags:
            lines.append("")
            lines.append("── Risk Flags ──")
            for flag in brief.risk_flags:
                lines.append(f"  ! {flag}")

        if brief.collector_failures:
            lines.append("")
            lines.append("── Collector Failures ──")
            for fail in brief.collector_failures:
                lines.append(f"  x {fail}")

        content.update("\n".join(lines))


class TradeDetailModal(DetailModal):
    """Detail view for a closed trade with decision reasoning and reflection."""

    def __init__(self, trade: ClosedTrade) -> None:
        super().__init__(title=f"{trade.symbol} — Trade Detail")
        self._trade = trade

    def compose_content(self) -> ComposeResult:
        t = self._trade
        pnl_sign = "+" if t.realized_pnl >= 0 else ""
        lines = [
            f"Symbol:          {t.symbol}",
            f"Side:            {t.side}",
            f"P&L:             {pnl_sign}${t.realized_pnl:,.2f}",
            f"Holding Days:    {t.holding_days}",
            f"Strategy:        {t.strategy_id}",
            f"Exit Reason:     {t.exit_reason}",
            f"Closed At:       {t.closed_at}",
        ]
        yield Static("\n".join(lines), id="detail-content")
        yield Static("Loading reasoning...", id="reasoning-content")

    def on_mount(self) -> None:
        self._fetch_extras()

    @work(thread=True)
    def _fetch_extras(self) -> None:
        decision: str | None = None
        reflection: str | None = None
        try:
            with pg_conn() as conn:
                decision = fetch_trade_decision(conn, self._trade.symbol, self._trade.closed_at.date())
                reflection = fetch_trade_reflection(conn, self._trade.symbol, self._trade.closed_at)
        except Exception:
            logger.warning("TradeDetailModal fetch failed", exc_info=True)
        self.app.call_from_thread(self._render_extras, decision, reflection)

    def _render_extras(self, decision: str | None, reflection: str | None) -> None:
        reasoning = self.query_one("#reasoning-content", Static)
        lines = [
            "",
            "── Decision Reasoning ──",
            decision or "No decision reasoning recorded",
            "",
            "── Trade Reflection ──",
            reflection or "No reflection recorded",
        ]
        reasoning.update("\n".join(lines))


class AgentEventModal(DetailModal):
    """Detail view for an agent event."""

    def __init__(self, event: GraphActivity) -> None:
        super().__init__(title=f"{event.current_agent} — Event Detail")
        self._event = event

    def compose_content(self) -> ComposeResult:
        e = self._event
        lines = [
            f"Graph:       {e.graph_name}",
            f"Node:        {e.current_node}",
            f"Agent:       {e.current_agent}",
            f"Cycle:       {e.cycle_number}",
            f"Started:     {e.cycle_started}",
            f"Events:      {e.event_count}",
        ]
        yield Static("\n".join(lines), id="detail-content")
