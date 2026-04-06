"""Tests for holding period and trading window enforcement in RiskGate.check()."""

from unittest.mock import MagicMock

import pytest

from quantstack.execution.risk_gate import RiskGate, RiskLimits
from quantstack.holding_period import HoldingType


def _clear_caches():
    from quantstack.config.settings import (
        get_holding_period_settings,
        get_trading_window_settings,
    )

    get_holding_period_settings.cache_clear()
    get_trading_window_settings.cache_clear()


@pytest.fixture()
def risk_gate():
    """Create a RiskGate with a mock portfolio that has a valid snapshot."""
    portfolio = MagicMock()
    snapshot = MagicMock()
    snapshot.total_equity = 100_000.0
    snapshot.daily_pnl = 0.0
    snapshot.positions_value = 0.0
    portfolio.get_snapshot.return_value = snapshot
    portfolio.get_position.return_value = None
    portfolio.get_positions.return_value = []
    return RiskGate(limits=RiskLimits(), portfolio=portfolio)


class TestRiskGateHoldingPeriod:
    """Holding period check in risk gate."""

    def test_holding_type_none_skips_check(self, risk_gate, monkeypatch):
        """RiskGate.check() with holding_type=None → skips holding period check."""
        monkeypatch.delenv("TRADING_WINDOW", raising=False)
        monkeypatch.delenv("TRADING_HOLDING_PERIODS", raising=False)
        _clear_caches()

        verdict = risk_gate.check(
            symbol="SPY",
            side="buy",
            quantity=10,
            current_price=450.0,
            daily_volume=80_000_000,
        )
        assert verdict.approved

    def test_allowed_holding_type_approved(self, risk_gate, monkeypatch):
        """RiskGate.check() with allowed holding_type → approved."""
        monkeypatch.delenv("TRADING_HOLDING_PERIODS", raising=False)
        monkeypatch.setenv("TRADING_WINDOW", "equity_swing")
        _clear_caches()

        verdict = risk_gate.check(
            symbol="SPY",
            side="buy",
            quantity=10,
            current_price=450.0,
            daily_volume=80_000_000,
            instrument_type="equity",
            holding_type=HoldingType.SHORT_SWING,
        )
        assert verdict.approved

    def test_disallowed_holding_type_rejected(self, risk_gate, monkeypatch):
        """RiskGate.check() with disallowed holding_type → rejected with reason."""
        monkeypatch.delenv("TRADING_HOLDING_PERIODS", raising=False)
        monkeypatch.setenv("TRADING_WINDOW", "options_weekly")
        _clear_caches()

        verdict = risk_gate.check(
            symbol="SPY",
            side="buy",
            quantity=10,
            current_price=450.0,
            daily_volume=80_000_000,
            holding_type=HoldingType.POSITION,
        )
        assert not verdict.approved
        assert "holding_period_not_allowed" in verdict.reason

    def test_default_config_all_types_approved(self, risk_gate, monkeypatch):
        """RiskGate.check() with default config (all allowed) → all types approved."""
        monkeypatch.delenv("TRADING_HOLDING_PERIODS", raising=False)
        monkeypatch.delenv("RESEARCH_HOLDING_PERIODS", raising=False)
        monkeypatch.delenv("TRADING_WINDOW", raising=False)
        monkeypatch.delenv("RESEARCH_WINDOW", raising=False)
        _clear_caches()

        for ht in HoldingType:
            verdict = risk_gate.check(
                symbol="SPY",
                side="buy",
                quantity=10,
                current_price=450.0,
                daily_volume=80_000_000,
                holding_type=ht,
            )
            assert verdict.approved, f"{ht} should be approved with default config"

    def test_existing_checks_still_work(self, risk_gate, monkeypatch):
        """Existing risk checks still work when holding_type parameter added."""
        monkeypatch.delenv("TRADING_WINDOW", raising=False)
        monkeypatch.delenv("TRADING_HOLDING_PERIODS", raising=False)
        _clear_caches()
        risk_gate.add_restricted("BANNED")
        verdict = risk_gate.check(
            symbol="BANNED",
            side="buy",
            quantity=10,
            current_price=50.0,
            daily_volume=1_000_000,
        )
        assert not verdict.approved
        assert "restricted" in verdict.reason.lower()


class TestRiskGateTradingWindow:
    """Trading window (instrument + DTE) enforcement in risk gate."""

    def test_equity_rejected_when_only_options_allowed(self, risk_gate, monkeypatch):
        """Equity trade rejected when TRADING_WINDOW only allows options."""
        monkeypatch.delenv("TRADING_HOLDING_PERIODS", raising=False)
        monkeypatch.setenv("TRADING_WINDOW", "options_weekly")
        _clear_caches()

        verdict = risk_gate.check(
            symbol="SPY",
            side="buy",
            quantity=10,
            current_price=450.0,
            daily_volume=80_000_000,
            instrument_type="equity",
        )
        assert not verdict.approved
        assert "trading_window_rejected" in verdict.reason

    def test_options_within_dte_approved(self, risk_gate, monkeypatch):
        """Options trade with DTE inside weekly window → approved."""
        monkeypatch.delenv("TRADING_HOLDING_PERIODS", raising=False)
        monkeypatch.setenv("TRADING_WINDOW", "options_weekly")
        _clear_caches()

        verdict = risk_gate.check(
            symbol="SPY",
            side="buy",
            quantity=1,
            current_price=450.0,
            daily_volume=80_000_000,
            instrument_type="options",
            dte=5,
        )
        assert verdict.approved

    def test_options_outside_dte_rejected(self, risk_gate, monkeypatch):
        """Options trade with DTE outside weekly window → rejected."""
        monkeypatch.delenv("TRADING_HOLDING_PERIODS", raising=False)
        monkeypatch.setenv("TRADING_WINDOW", "options_weekly")
        _clear_caches()

        verdict = risk_gate.check(
            symbol="SPY",
            side="buy",
            quantity=1,
            current_price=450.0,
            daily_volume=80_000_000,
            instrument_type="options",
            dte=30,
        )
        assert not verdict.approved
        assert "trading_window_rejected" in verdict.reason

    def test_all_window_allows_everything(self, risk_gate, monkeypatch):
        """TRADING_WINDOW=all allows both equity and options at any DTE."""
        monkeypatch.delenv("TRADING_HOLDING_PERIODS", raising=False)
        monkeypatch.setenv("TRADING_WINDOW", "all")
        _clear_caches()

        # Equity
        v = risk_gate.check(
            symbol="SPY", side="buy", quantity=10,
            current_price=450.0, daily_volume=80_000_000,
            instrument_type="equity",
        )
        assert v.approved

        # Options with large DTE
        v = risk_gate.check(
            symbol="SPY", side="buy", quantity=1,
            current_price=450.0, daily_volume=80_000_000,
            instrument_type="options", dte=500,
        )
        assert v.approved

    def test_mixed_windows(self, risk_gate, monkeypatch):
        """Mixed window: options_weekly,equity_swing allows both within bounds."""
        monkeypatch.delenv("TRADING_HOLDING_PERIODS", raising=False)
        monkeypatch.setenv("TRADING_WINDOW", "options_weekly,equity_swing")
        _clear_caches()

        # Equity allowed
        v = risk_gate.check(
            symbol="SPY", side="buy", quantity=10,
            current_price=450.0, daily_volume=80_000_000,
            instrument_type="equity",
        )
        assert v.approved

        # Options DTE=5 allowed (weekly)
        v = risk_gate.check(
            symbol="SPY", side="buy", quantity=1,
            current_price=450.0, daily_volume=80_000_000,
            instrument_type="options", dte=5,
        )
        assert v.approved

        # Options DTE=30 rejected (not in weekly)
        v = risk_gate.check(
            symbol="SPY", side="buy", quantity=1,
            current_price=450.0, daily_volume=80_000_000,
            instrument_type="options", dte=30,
        )
        assert not v.approved
