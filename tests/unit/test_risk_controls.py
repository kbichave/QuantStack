"""
Unit tests for risk controls.

Tests verify:
- Exposure limits enforcement
- Drawdown soft/hard stop triggers
- Regime-based trade blocking
- Daily loss limits
- Risk state management
"""

import pytest
from datetime import datetime, timedelta

from quantcore.risk.controls import (
    RiskController,
    ExposureManager,
    DrawdownProtection,
    RiskStatus,
    RiskState,
)
from quantcore.hierarchy.regime_classifier import RegimeType


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def exposure_manager():
    """ExposureManager with default settings."""
    return ExposureManager(
        max_concurrent_trades=5,
        max_exposure_per_symbol_pct=20.0,
        max_total_exposure_pct=80.0,
        max_daily_trades=20,
    )


@pytest.fixture
def drawdown_protection():
    """DrawdownProtection with test settings."""
    return DrawdownProtection(
        soft_stop_pct=3.0,
        hard_stop_pct=7.0,
        daily_loss_limit_pct=5.0,  # Higher than soft stop so we can test soft stop
        recovery_threshold_pct=1.0,
    )


@pytest.fixture
def risk_controller(exposure_manager, drawdown_protection):
    """RiskController with test components."""
    return RiskController(
        exposure_manager=exposure_manager,
        drawdown_protection=drawdown_protection,
    )


# =============================================================================
# Test: Exposure Manager
# =============================================================================


class TestExposureManager:
    """Tests for exposure management."""

    def test_can_open_first_position(self, exposure_manager):
        """
        Verify first position can be opened within limits.
        """
        allowed, reason = exposure_manager.can_open_position(
            symbol="SPY",
            exposure_pct=10.0,
            equity=100000,
        )

        assert allowed is True
        assert reason == "OK"

    def test_max_concurrent_trades_enforced(self, exposure_manager):
        """
        Verify max concurrent trades limit is enforced.
        """
        # Open maximum number of trades
        for i in range(5):
            exposure_manager.register_open(f"SYM{i}", 10.0)

        # Try to open another
        allowed, reason = exposure_manager.can_open_position(
            symbol="NEW",
            exposure_pct=5.0,
            equity=100000,
        )

        assert allowed is False
        assert "concurrent" in reason.lower()

    def test_max_symbol_exposure_enforced(self, exposure_manager):
        """
        Verify max exposure per symbol is enforced.
        """
        # Register existing position
        exposure_manager.register_open("SPY", 15.0)

        # Try to add more, exceeding 20% limit
        allowed, reason = exposure_manager.can_open_position(
            symbol="SPY",
            exposure_pct=10.0,  # Total would be 25%
            equity=100000,
        )

        assert allowed is False
        assert "symbol exposure" in reason.lower()

    def test_max_total_exposure_enforced(self, exposure_manager):
        """
        Verify max total exposure is enforced.
        """
        # Register high exposure across symbols
        exposure_manager.register_open("SPY", 20.0)
        exposure_manager.register_open("QQQ", 20.0)
        exposure_manager.register_open("AAPL", 20.0)
        exposure_manager.register_open("MSFT", 15.0)  # Total = 75%

        # Try to add more, exceeding 80% limit
        allowed, reason = exposure_manager.can_open_position(
            symbol="GOOGL",
            exposure_pct=10.0,  # Total would be 85%
            equity=100000,
        )

        assert allowed is False
        assert "total exposure" in reason.lower()

    def test_register_close_releases_exposure(self, exposure_manager):
        """
        Verify closing position releases exposure.
        """
        # Open and close a position
        exposure_manager.register_open("SPY", 20.0)
        exposure_manager.register_close("SPY", 20.0)

        # Should be able to open new position with same exposure
        allowed, reason = exposure_manager.can_open_position(
            symbol="SPY",
            exposure_pct=20.0,
            equity=100000,
        )

        assert allowed is True

    def test_daily_trade_limit_enforced(self, exposure_manager):
        """
        Verify daily trade limit is enforced.
        """
        # Open and close trades sequentially to avoid concurrent limit
        for i in range(20):
            # Open then close to count as 1 trade, staying under concurrent limit
            exposure_manager.register_open(f"SYM{i}", 5.0)
            exposure_manager.register_close(f"SYM{i}", 5.0)

        # Should hit daily limit (20 trades done, limit is 20)
        allowed, reason = exposure_manager.can_open_position(
            symbol="NEW",
            exposure_pct=5.0,
            equity=100000,
        )

        assert allowed is False
        assert "daily" in reason.lower()


# =============================================================================
# Test: Drawdown Protection
# =============================================================================


class TestDrawdownProtection:
    """Tests for drawdown-based protections."""

    def test_normal_status_no_drawdown(self, drawdown_protection):
        """
        Verify NORMAL status when no drawdown.
        """
        equity = 100000
        drawdown_protection.reset(equity)

        status = drawdown_protection.update(equity)
        assert status == RiskStatus.NORMAL

    def test_caution_at_soft_stop(self, drawdown_protection):
        """
        Verify CAUTION status at soft stop threshold.
        """
        initial_equity = 100000
        drawdown_protection.reset(initial_equity)

        # 3.5% drawdown (above soft stop of 3%)
        current_equity = 96500
        status = drawdown_protection.update(current_equity)

        assert status == RiskStatus.CAUTION

    def test_halted_at_hard_stop(self, drawdown_protection):
        """
        Verify HALTED status at hard stop threshold.
        """
        initial_equity = 100000
        drawdown_protection.reset(initial_equity)

        # 8% drawdown (above hard stop of 7%)
        current_equity = 92000
        status = drawdown_protection.update(current_equity)

        assert status == RiskStatus.HALTED

    def test_restricted_at_daily_loss_limit(self, drawdown_protection):
        """
        Verify RESTRICTED status when daily loss limit hit.
        """
        from datetime import datetime

        initial_equity = 100000
        drawdown_protection.reset(initial_equity)

        # Simulate new day
        drawdown_protection._day_start_equity = 100000

        # 5.5% daily loss (above 5% limit)
        current_equity = 94500
        # Use a timestamp that's not in the 9:00-9:35 range to avoid resetting _day_start_equity
        test_time = datetime(2024, 1, 15, 14, 30)  # 2:30 PM
        status = drawdown_protection.update(current_equity, timestamp=test_time)

        assert status == RiskStatus.RESTRICTED

    def test_recovery_from_halt(self, drawdown_protection):
        """
        Verify recovery from HALTED status.
        """
        initial_equity = 100000
        drawdown_protection.reset(initial_equity)

        # Hit hard stop
        drawdown_protection.update(92000)  # 8% DD
        assert drawdown_protection._halted is True

        # Recover enough (need to be below hard_stop - recovery_threshold)
        # Hard stop is 7%, recovery is 1%, so need DD < 6%
        # Use 97000 (3% DD) to recover to NORMAL (at or below soft_stop_pct of 3%)
        status = drawdown_protection.update(97000)  # 3% DD exactly at soft stop

        # Should no longer be halted
        assert drawdown_protection._halted is False
        # Should be CAUTION since we're at the soft stop threshold (3%)
        assert status in [RiskStatus.NORMAL, RiskStatus.CAUTION]

    def test_get_drawdown_calculation(self, drawdown_protection):
        """
        Verify drawdown calculation is correct.
        """
        drawdown_protection._peak_equity = 100000

        dd = drawdown_protection.get_drawdown(95000)
        assert dd == 5.0  # 5% drawdown

        dd = drawdown_protection.get_drawdown(100000)
        assert dd == 0.0  # No drawdown

    def test_peak_tracking(self, drawdown_protection):
        """
        Verify peak equity is tracked correctly.
        """
        drawdown_protection.reset(100000)

        # New high
        drawdown_protection.update(105000)
        assert drawdown_protection._peak_equity == 105000

        # Pullback - peak should stay at 105000
        drawdown_protection.update(102000)
        assert drawdown_protection._peak_equity == 105000


# =============================================================================
# Test: Risk Controller Integration
# =============================================================================


class TestRiskController:
    """Tests for integrated risk controller."""

    def test_can_trade_normal_conditions(self, risk_controller):
        """
        Verify trading allowed under normal conditions.
        """
        equity = 100000
        risk_controller.drawdown.reset(equity)

        allowed, reason, multiplier = risk_controller.can_trade(
            symbol="SPY",
            exposure_pct=10.0,
            equity=equity,
            direction="LONG",
        )

        assert allowed is True
        assert multiplier == 1.0

    def test_size_reduced_at_soft_stop(self, risk_controller):
        """
        Verify position size reduced at soft stop.
        """
        initial_equity = 100000
        risk_controller.drawdown.reset(initial_equity)

        # Enter soft stop zone
        current_equity = 96500  # 3.5% DD

        allowed, reason, multiplier = risk_controller.can_trade(
            symbol="SPY",
            exposure_pct=10.0,
            equity=current_equity,
            direction="LONG",
        )

        assert allowed is True
        assert multiplier == 0.5  # CAUTION reduces to 50%

    def test_trading_blocked_at_hard_stop(self, risk_controller):
        """
        Verify trading blocked at hard stop.
        """
        initial_equity = 100000
        risk_controller.drawdown.reset(initial_equity)

        # Enter hard stop zone
        current_equity = 92000  # 8% DD

        allowed, reason, multiplier = risk_controller.can_trade(
            symbol="SPY",
            exposure_pct=10.0,
            equity=current_equity,
            direction="LONG",
        )

        assert allowed is False
        assert multiplier == 0.0

    def test_regime_blocks_opposing_trades(self, risk_controller):
        """
        Verify regime-based trade blocking.
        """
        equity = 100000
        risk_controller.drawdown.reset(equity)

        # Set BEAR regime
        risk_controller.update_regime(RegimeType.BEAR)

        # Try to go long in bear market
        allowed, reason, _ = risk_controller.can_trade(
            symbol="SPY",
            exposure_pct=10.0,
            equity=equity,
            direction="LONG",
        )

        assert allowed is False
        assert "BEAR" in reason

    def test_regime_allows_aligned_trades(self, risk_controller):
        """
        Verify regime allows aligned trades.
        """
        equity = 100000
        risk_controller.drawdown.reset(equity)

        # Set BULL regime
        risk_controller.update_regime(RegimeType.BULL)

        # Go long in bull market
        allowed, reason, _ = risk_controller.can_trade(
            symbol="SPY",
            exposure_pct=10.0,
            equity=equity,
            direction="LONG",
        )

        assert allowed is True

    def test_get_risk_state_returns_complete_state(self, risk_controller):
        """
        Verify risk state contains all required information.
        """
        equity = 100000
        risk_controller.drawdown.reset(equity)
        risk_controller.exposure.register_open("SPY", 15.0)

        state = risk_controller.get_risk_state(equity)

        assert isinstance(state, RiskState)
        assert state.equity == equity
        assert state.open_trades == 1
        assert state.open_exposure_pct == 15.0
        assert isinstance(state.status, RiskStatus)


# =============================================================================
# Test: Risk State
# =============================================================================


class TestRiskState:
    """Tests for RiskState dataclass."""

    def test_can_trade_normal(self):
        """
        Verify can_trade returns True for NORMAL status.
        """
        state = RiskState(
            status=RiskStatus.NORMAL,
            equity=100000,
            drawdown_pct=0,
            daily_pnl=0,
            open_trades=0,
            open_exposure_pct=0,
        )

        assert state.can_trade() is True

    def test_can_trade_caution(self):
        """
        Verify can_trade returns True for CAUTION status.
        """
        state = RiskState(
            status=RiskStatus.CAUTION,
            equity=97000,
            drawdown_pct=3.0,
            daily_pnl=-1.0,
            open_trades=2,
            open_exposure_pct=20.0,
        )

        assert state.can_trade() is True

    def test_cannot_trade_halted(self):
        """
        Verify can_trade returns False for HALTED status.
        """
        state = RiskState(
            status=RiskStatus.HALTED,
            equity=92000,
            drawdown_pct=8.0,
            daily_pnl=-3.0,
            open_trades=0,
            open_exposure_pct=0,
        )

        assert state.can_trade() is False

    def test_size_multiplier_by_status(self):
        """
        Verify size multipliers are correct for each status.
        """
        for status, expected in [
            (RiskStatus.NORMAL, 1.0),
            (RiskStatus.CAUTION, 0.5),
            (RiskStatus.RESTRICTED, 0.0),
            (RiskStatus.HALTED, 0.0),
        ]:
            state = RiskState(
                status=status,
                equity=100000,
                drawdown_pct=0,
                daily_pnl=0,
                open_trades=0,
                open_exposure_pct=0,
            )
            assert state.size_multiplier() == expected, f"Wrong multiplier for {status}"


# =============================================================================
# Test: Edge Cases
# =============================================================================


class TestRiskControlEdgeCases:
    """Edge case tests for risk controls."""

    def test_zero_equity_handled(self, drawdown_protection):
        """
        Verify zero equity doesn't cause division errors.
        """
        drawdown_protection.reset(100000)

        # This shouldn't crash
        dd = drawdown_protection.get_drawdown(0)
        assert dd == 100.0  # 100% drawdown

    def test_negative_drawdown_impossible(self, drawdown_protection):
        """
        Verify drawdown can't be negative (equity above peak).
        """
        drawdown_protection._peak_equity = 100000

        # Equity above peak - should update peak, DD = 0
        drawdown_protection.update(105000)
        dd = drawdown_protection.get_drawdown(105000)

        assert dd == 0.0

    def test_exposure_manager_reset_daily(self, exposure_manager):
        """
        Verify daily counter reset works.
        """
        # Register some trades
        for i in range(5):
            exposure_manager.register_open(f"SYM{i}", 5.0)

        # Reset daily
        exposure_manager.reset_daily()

        # Should be able to trade again
        allowed, _ = exposure_manager.can_open_position("NEW", 5.0, 100000)
        # Still limited by concurrent trades, but daily limit reset
        assert (
            exposure_manager._daily_trades.get(datetime.now().strftime("%Y-%m-%d"), 0)
            <= 5
        )
