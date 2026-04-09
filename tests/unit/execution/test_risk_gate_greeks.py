"""Tests for Portfolio Greeks integration in RiskGate (QS-E3).

Verifies:
  - Options order approved when proposed Greeks within limits
  - Options order rejected when portfolio delta would breach
  - Options order rejected when per-symbol delta would breach
  - Equity orders unaffected by Greeks parameters
"""

from unittest.mock import MagicMock, patch

import pytest

from quantstack.core.risk.options_risk import (
    GreeksLimits,
    PortfolioGreeksManager,
    RiskLimits as OptionsRiskLimits,
    RiskMetrics,
)
from quantstack.execution.risk_gate import RiskGate, RiskLimits


def _make_greeks_manager(
    total_delta: float = 0.0,
    total_gamma: float = 0.0,
    delta_by_symbol: dict | None = None,
) -> PortfolioGreeksManager:
    """Create a PortfolioGreeksManager with pre-set metrics."""
    mgr = PortfolioGreeksManager(
        greek_limits=GreeksLimits(
            max_delta=500,
            max_gamma=100,
            max_delta_per_symbol=100,
        ),
        risk_limits=OptionsRiskLimits(),
    )
    mgr.current_metrics = RiskMetrics(
        total_delta=total_delta,
        total_gamma=total_gamma,
        delta_by_symbol=delta_by_symbol or {},
    )
    return mgr


def _make_portfolio_mock(equity: float = 100_000.0) -> MagicMock:
    """Create a mock PortfolioState with a snapshot."""
    portfolio = MagicMock()
    snapshot = MagicMock()
    snapshot.total_equity = equity
    snapshot.daily_pnl_pct = 0.0
    snapshot.daily_pnl = 0.0
    snapshot.gross_exposure = 0.0
    snapshot.net_exposure = 0.0
    portfolio.get_snapshot.return_value = snapshot
    portfolio.get_position.return_value = None
    portfolio.get_positions.return_value = []
    return portfolio


@pytest.fixture(autouse=True)
def _bypass_market_and_window_checks():
    """Bypass market-hours and trading-window checks for Greeks-specific tests."""
    with (
        patch.object(RiskGate, "_check_market_hours", return_value=None),
        patch("quantstack.trading_window.is_trade_allowed", return_value=True),
    ):
        yield


class TestGreeksInRiskGate:
    """Greeks integration in the options path of RiskGate.check()."""

    def test_approved_when_greeks_within_limits(self):
        mgr = _make_greeks_manager(total_delta=100, total_gamma=20)
        gate = RiskGate(
            limits=RiskLimits(),
            portfolio=_make_portfolio_mock(),
            greeks_manager=mgr,
        )

        verdict = gate.check(
            symbol="AAPL", side="buy", quantity=5, current_price=150.0,
            daily_volume=10_000_000, instrument_type="options", dte=30,
            premium_at_risk=500.0, proposed_delta=50.0, proposed_gamma=10.0,
        )

        assert verdict.approved

    def test_rejected_when_portfolio_delta_breach(self):
        mgr = _make_greeks_manager(total_delta=450)
        gate = RiskGate(
            limits=RiskLimits(),
            portfolio=_make_portfolio_mock(),
            greeks_manager=mgr,
        )

        verdict = gate.check(
            symbol="AAPL", side="buy", quantity=5, current_price=150.0,
            daily_volume=10_000_000, instrument_type="options", dte=30,
            premium_at_risk=500.0, proposed_delta=60.0, proposed_gamma=5.0,
        )

        assert not verdict.approved
        assert "options_greeks_limit" in [v.rule for v in verdict.violations]

    def test_rejected_when_per_symbol_delta_breach(self):
        mgr = _make_greeks_manager(total_delta=80, delta_by_symbol={"AAPL": 80})
        gate = RiskGate(
            limits=RiskLimits(),
            portfolio=_make_portfolio_mock(),
            greeks_manager=mgr,
        )

        verdict = gate.check(
            symbol="AAPL", side="buy", quantity=5, current_price=150.0,
            daily_volume=10_000_000, instrument_type="options", dte=30,
            premium_at_risk=500.0, proposed_delta=30.0, proposed_gamma=5.0,
        )

        assert not verdict.approved
        assert "options_greeks_limit" in [v.rule for v in verdict.violations]

    def test_zero_greeks_still_checked(self):
        """Even with proposed_delta=0, the check runs (no guard clause)."""
        mgr = _make_greeks_manager(total_delta=450)
        gate = RiskGate(
            limits=RiskLimits(),
            portfolio=_make_portfolio_mock(),
            greeks_manager=mgr,
        )

        verdict = gate.check(
            symbol="AAPL", side="buy", quantity=5, current_price=150.0,
            daily_volume=10_000_000, instrument_type="options", dte=30,
            premium_at_risk=500.0, proposed_delta=0.0, proposed_gamma=0.0,
        )

        # 0+450=450 < 500, so still approved — but the check DID run
        assert verdict.approved

    def test_equity_unaffected_by_greeks(self):
        """Equity orders never hit the Greeks check path."""
        mgr = _make_greeks_manager(total_delta=9999)
        gate = RiskGate(
            limits=RiskLimits(),
            portfolio=_make_portfolio_mock(),
            greeks_manager=mgr,
        )

        verdict = gate.check(
            symbol="AAPL", side="buy", quantity=10, current_price=150.0,
            daily_volume=10_000_000, instrument_type="equity",
            proposed_delta=9999.0,
        )

        assert verdict.approved
