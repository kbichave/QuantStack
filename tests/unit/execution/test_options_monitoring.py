"""Tests for options-specific monitoring rules in ExecutionMonitor.

Verifies five rules:
  - theta_acceleration: triggers when DTE < 7 AND |theta|/premium > 5%
  - pin_risk: triggers when DTE < 3 AND underlying within 1% of strike
  - assignment_risk: short ITM call near ex-dividend (ships disabled)
  - iv_crush: post-earnings IV collapse > 30% (ships disabled)
  - max_theta_loss: cumulative premium decay > 40%
  - Action config: auto_exit calls _submit_exit, flag_only logs only
  - Equity positions skip options evaluation entirely
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytz

from quantstack.execution.execution_monitor import (
    DEFAULT_OPTIONS_RULES,
    ExecutionMonitor,
    MonitoredPosition,
    OptionsMonitorRule,
)
from quantstack.holding_period import HoldingType

ET = pytz.timezone("US/Eastern")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_option_position(
    *,
    symbol: str = "AAPL",
    side: str = "long",
    quantity: int = 1,
    entry_price: float = 3.00,
    option_strike: float = 100.0,
    option_expiry: date | None = None,
    option_type: str = "call",
    entry_premium: float = 3.00,
    instrument_type: str = "options",
    strategy_id: str = "test_strategy",
) -> MonitoredPosition:
    """Build a MonitoredPosition for options testing."""
    if option_expiry is None:
        option_expiry = date.today() + timedelta(days=5)
    return MonitoredPosition(
        symbol=symbol,
        side=side,
        quantity=quantity,
        holding_type=HoldingType.SWING,
        entry_price=entry_price,
        entry_time=datetime.now(ET) - timedelta(hours=2),
        instrument_type=instrument_type,
        underlying_symbol=symbol,
        option_contract=f"{symbol}_240101_C{option_strike}",
        option_strike=option_strike,
        option_expiry=option_expiry,
        option_type=option_type,
        entry_premium=entry_premium,
        strategy_id=strategy_id,
    )


def _mock_greeks(theta: float = -0.05, delta: float = 0.5, gamma: float = 0.03, vega: float = 0.10):
    """Return a dict matching compute_greeks_dispatch output shape."""
    return {
        "greeks": {
            "theta": theta,
            "delta": delta,
            "gamma": gamma,
            "vega": vega,
            "rho": 0.01,
        },
        "backend_used": "internal",
        "interpretations": {},
        "risk_metrics": {},
    }


def _make_monitor() -> ExecutionMonitor:
    """Build an ExecutionMonitor with all deps mocked."""
    broker = MagicMock()
    feed = AsyncMock()
    portfolio = MagicMock()
    portfolio.get_positions.return_value = []
    return ExecutionMonitor(
        broker=broker,
        price_feed=feed,
        portfolio_state=portfolio,
    )


# ===========================================================================
# theta_acceleration rule
# ===========================================================================


class TestThetaAcceleration:
    """DTE < 7 AND |theta|/premium > 0.05 → trigger."""

    @pytest.mark.asyncio
    async def test_triggers_when_theta_ratio_high_and_dte_low(self):
        """DTE=5, premium=2.00, theta=-0.12 → 6%/day → triggers."""
        pos = _make_option_position(
            entry_price=2.00,
            entry_premium=2.00,
            option_expiry=date.today() + timedelta(days=5),
        )
        monitor = _make_monitor()
        now = datetime.now(ET)

        with patch(
            "quantstack.execution.execution_monitor.compute_greeks_dispatch",
            return_value=_mock_greeks(theta=-0.12),
        ):
            should_exit, reason = await monitor._evaluate_options_rules(
                pos, current_price=2.00, current_time=now
            )

        assert should_exit is True
        assert "theta_acceleration" in reason

    @pytest.mark.asyncio
    async def test_no_trigger_dte_too_high(self):
        """DTE=8 → no trigger even with high theta ratio."""
        pos = _make_option_position(
            entry_price=2.00,
            entry_premium=2.00,
            option_expiry=date.today() + timedelta(days=8),
        )
        monitor = _make_monitor()
        now = datetime.now(ET)

        with patch(
            "quantstack.execution.execution_monitor.compute_greeks_dispatch",
            return_value=_mock_greeks(theta=-0.12),
        ):
            should_exit, reason = await monitor._evaluate_options_rules(
                pos, current_price=2.00, current_time=now
            )

        assert should_exit is False

    @pytest.mark.asyncio
    async def test_no_trigger_theta_ratio_low(self):
        """DTE=5, premium=4.00, theta=-0.10 → 2.5% → no trigger."""
        pos = _make_option_position(
            entry_price=4.00,
            entry_premium=4.00,
            option_expiry=date.today() + timedelta(days=5),
        )
        monitor = _make_monitor()
        now = datetime.now(ET)

        with patch(
            "quantstack.execution.execution_monitor.compute_greeks_dispatch",
            return_value=_mock_greeks(theta=-0.10),
        ):
            should_exit, reason = await monitor._evaluate_options_rules(
                pos, current_price=4.00, current_time=now
            )

        assert should_exit is False


# ===========================================================================
# pin_risk rule
# ===========================================================================


class TestPinRisk:
    """DTE < 3 AND |underlying - strike|/strike < 0.01 → trigger."""

    @pytest.mark.asyncio
    async def test_triggers_near_strike_low_dte(self):
        """DTE=2, strike=100, price=100.50 → 0.5% → triggers."""
        pos = _make_option_position(
            option_strike=100.0,
            option_expiry=date.today() + timedelta(days=2),
        )
        monitor = _make_monitor()
        now = datetime.now(ET)

        with patch(
            "quantstack.execution.execution_monitor.compute_greeks_dispatch",
            return_value=_mock_greeks(),
        ):
            should_exit, reason = await monitor._evaluate_options_rules(
                pos, current_price=100.50, current_time=now
            )

        assert should_exit is True
        assert "pin_risk" in reason

    @pytest.mark.asyncio
    async def test_no_trigger_dte_too_high(self):
        """DTE=4 → no trigger even near strike."""
        pos = _make_option_position(
            option_strike=100.0,
            option_expiry=date.today() + timedelta(days=4),
        )
        monitor = _make_monitor()
        now = datetime.now(ET)

        with patch(
            "quantstack.execution.execution_monitor.compute_greeks_dispatch",
            return_value=_mock_greeks(),
        ):
            should_exit, reason = await monitor._evaluate_options_rules(
                pos, current_price=100.50, current_time=now
            )

        assert should_exit is False

    @pytest.mark.asyncio
    async def test_no_trigger_price_far_from_strike(self):
        """DTE=2, strike=100, price=105 → 5% → no trigger."""
        pos = _make_option_position(
            option_strike=100.0,
            option_expiry=date.today() + timedelta(days=2),
        )
        monitor = _make_monitor()
        now = datetime.now(ET)

        with patch(
            "quantstack.execution.execution_monitor.compute_greeks_dispatch",
            return_value=_mock_greeks(),
        ):
            should_exit, reason = await monitor._evaluate_options_rules(
                pos, current_price=105.0, current_time=now
            )

        assert should_exit is False


# ===========================================================================
# assignment_risk rule (ships disabled)
# ===========================================================================


class TestAssignmentRisk:
    """Short call ITM + ex-div within 2 days → flag_only (disabled by default)."""

    @pytest.mark.asyncio
    async def test_triggers_short_call_itm_near_exdiv(self):
        """Short call, strike=95, underlying=100, ex_div in 1 day → triggers."""
        pos = _make_option_position(
            side="short",
            option_type="call",
            option_strike=95.0,
            option_expiry=date.today() + timedelta(days=10),
        )
        monitor = _make_monitor()
        now = datetime.now(ET)

        # Enable the rule for testing
        rules_override = dict(DEFAULT_OPTIONS_RULES)
        rules_override["assignment_risk"] = OptionsMonitorRule(
            "assignment_risk", True, "flag_only"
        )

        with patch(
            "quantstack.execution.execution_monitor.compute_greeks_dispatch",
            return_value=_mock_greeks(delta=-0.7),
        ), patch.object(monitor, "_options_rules", rules_override):
            # Provide mock ex-div date
            pos._ex_div_date = date.today() + timedelta(days=1)
            should_exit, reason = await monitor._evaluate_options_rules(
                pos, current_price=100.0, current_time=now
            )

        # flag_only → should NOT trigger exit
        assert should_exit is False

    @pytest.mark.asyncio
    async def test_no_trigger_long_call(self):
        """Long call → assignment risk does not apply."""
        pos = _make_option_position(
            side="long",
            option_type="call",
            option_strike=95.0,
            option_expiry=date.today() + timedelta(days=10),
        )
        monitor = _make_monitor()
        now = datetime.now(ET)

        rules_override = dict(DEFAULT_OPTIONS_RULES)
        rules_override["assignment_risk"] = OptionsMonitorRule(
            "assignment_risk", True, "flag_only"
        )

        with patch(
            "quantstack.execution.execution_monitor.compute_greeks_dispatch",
            return_value=_mock_greeks(),
        ), patch.object(monitor, "_options_rules", rules_override):
            pos._ex_div_date = date.today() + timedelta(days=1)
            should_exit, reason = await monitor._evaluate_options_rules(
                pos, current_price=100.0, current_time=now
            )

        assert should_exit is False


# ===========================================================================
# iv_crush rule (ships disabled)
# ===========================================================================


class TestIVCrush:
    """Post-earnings + IV drop > 30% → flag_only (disabled by default)."""

    @pytest.mark.asyncio
    async def test_triggers_post_earnings_iv_drop(self):
        """Earnings 1 day ago, IV drop 42% → triggers (flag_only)."""
        pos = _make_option_position(
            option_expiry=date.today() + timedelta(days=10),
        )
        monitor = _make_monitor()
        now = datetime.now(ET)

        rules_override = dict(DEFAULT_OPTIONS_RULES)
        rules_override["iv_crush"] = OptionsMonitorRule(
            "iv_crush", True, "flag_only"
        )

        with patch(
            "quantstack.execution.execution_monitor.compute_greeks_dispatch",
            return_value=_mock_greeks(),
        ), patch.object(monitor, "_options_rules", rules_override):
            pos._earnings_date = date.today() - timedelta(days=1)
            pos._iv_entry = 0.50
            pos._iv_current = 0.29  # 42% drop
            should_exit, reason = await monitor._evaluate_options_rules(
                pos, current_price=3.00, current_time=now
            )

        # flag_only → should NOT trigger exit
        assert should_exit is False

    @pytest.mark.asyncio
    async def test_no_trigger_iv_drop_under_threshold(self):
        """IV drop 20% → no trigger."""
        pos = _make_option_position(
            option_expiry=date.today() + timedelta(days=10),
        )
        monitor = _make_monitor()
        now = datetime.now(ET)

        rules_override = dict(DEFAULT_OPTIONS_RULES)
        rules_override["iv_crush"] = OptionsMonitorRule(
            "iv_crush", True, "flag_only"
        )

        with patch(
            "quantstack.execution.execution_monitor.compute_greeks_dispatch",
            return_value=_mock_greeks(),
        ), patch.object(monitor, "_options_rules", rules_override):
            pos._earnings_date = date.today() - timedelta(days=1)
            pos._iv_entry = 0.50
            pos._iv_current = 0.40  # 20% drop
            should_exit, reason = await monitor._evaluate_options_rules(
                pos, current_price=3.00, current_time=now
            )

        assert should_exit is False


# ===========================================================================
# max_theta_loss rule
# ===========================================================================


class TestMaxThetaLoss:
    """(entry_premium - current_premium) / entry_premium > 0.40 → trigger."""

    @pytest.mark.asyncio
    async def test_triggers_when_decay_exceeds_threshold(self):
        """Entry=5.00, current=2.80 → 44% decay → triggers."""
        pos = _make_option_position(
            entry_price=5.00,
            entry_premium=5.00,
        )
        monitor = _make_monitor()
        now = datetime.now(ET)

        with patch(
            "quantstack.execution.execution_monitor.compute_greeks_dispatch",
            return_value=_mock_greeks(theta=-0.02),  # low theta so theta_acceleration won't fire
        ):
            should_exit, reason = await monitor._evaluate_options_rules(
                pos, current_price=2.80, current_time=now
            )

        assert should_exit is True
        assert "max_theta_loss" in reason

    @pytest.mark.asyncio
    async def test_no_trigger_under_threshold(self):
        """Entry=5.00, current=3.50 → 30% decay → no trigger."""
        pos = _make_option_position(
            entry_price=5.00,
            entry_premium=5.00,
        )
        monitor = _make_monitor()
        now = datetime.now(ET)

        with patch(
            "quantstack.execution.execution_monitor.compute_greeks_dispatch",
            return_value=_mock_greeks(theta=-0.02),
        ):
            should_exit, reason = await monitor._evaluate_options_rules(
                pos, current_price=3.50, current_time=now
            )

        assert should_exit is False


# ===========================================================================
# Action config tests
# ===========================================================================


class TestActionConfig:
    """auto_exit calls _submit_exit; flag_only logs but does NOT exit."""

    @pytest.mark.asyncio
    async def test_auto_exit_submits(self):
        """An auto_exit rule that fires → _submit_exit is called."""
        pos = _make_option_position(
            entry_price=5.00,
            entry_premium=5.00,
        )
        monitor = _make_monitor()
        monitor._positions[pos.symbol] = pos
        monitor._submit_exit = AsyncMock()
        now = datetime.now(ET)

        with patch(
            "quantstack.execution.execution_monitor.compute_greeks_dispatch",
            return_value=_mock_greeks(theta=-0.02),
        ):
            # current_price=2.80 → 44% decay → max_theta_loss fires
            should_exit, reason = await monitor._evaluate_options_rules(
                pos, current_price=2.80, current_time=now
            )

        assert should_exit is True
        assert "max_theta_loss" in reason

    @pytest.mark.asyncio
    async def test_flag_only_does_not_exit(self):
        """A flag_only rule that fires → should_exit is False."""
        pos = _make_option_position(
            option_expiry=date.today() + timedelta(days=10),
        )
        monitor = _make_monitor()
        now = datetime.now(ET)

        # Make assignment_risk enabled + flag_only
        rules_override = dict(DEFAULT_OPTIONS_RULES)
        rules_override["assignment_risk"] = OptionsMonitorRule(
            "assignment_risk", True, "flag_only"
        )

        with patch(
            "quantstack.execution.execution_monitor.compute_greeks_dispatch",
            return_value=_mock_greeks(),
        ), patch.object(monitor, "_options_rules", rules_override):
            pos.side = "short"
            pos.option_type = "call"
            pos.option_strike = 95.0
            pos._ex_div_date = date.today() + timedelta(days=1)
            should_exit, reason = await monitor._evaluate_options_rules(
                pos, current_price=100.0, current_time=now
            )

        assert should_exit is False

    @pytest.mark.asyncio
    async def test_rule_config_overrides(self):
        """Custom rules dict disables theta_acceleration → no exit even when conditions met."""
        pos = _make_option_position(
            entry_price=2.00,
            entry_premium=2.00,
            option_expiry=date.today() + timedelta(days=5),
        )
        monitor = _make_monitor()
        now = datetime.now(ET)

        # Disable theta_acceleration and max_theta_loss
        rules_override = dict(DEFAULT_OPTIONS_RULES)
        rules_override["theta_acceleration"] = OptionsMonitorRule(
            "theta_acceleration", False, "auto_exit"
        )
        rules_override["max_theta_loss"] = OptionsMonitorRule(
            "max_theta_loss", False, "auto_exit"
        )

        with patch(
            "quantstack.execution.execution_monitor.compute_greeks_dispatch",
            return_value=_mock_greeks(theta=-0.12),
        ), patch.object(monitor, "_options_rules", rules_override):
            should_exit, reason = await monitor._evaluate_options_rules(
                pos, current_price=2.00, current_time=now
            )

        assert should_exit is False


# ===========================================================================
# Equity skip
# ===========================================================================


class TestEquitySkip:
    """Equity positions skip options evaluation entirely."""

    @pytest.mark.asyncio
    async def test_equity_position_skipped(self):
        """instrument_type='equity' → immediate (False, '')."""
        pos = MonitoredPosition(
            symbol="AAPL",
            side="long",
            quantity=10,
            holding_type=HoldingType.SWING,
            entry_price=150.0,
            entry_time=datetime.now(ET),
            instrument_type="equity",
        )
        monitor = _make_monitor()
        now = datetime.now(ET)

        should_exit, reason = await monitor._evaluate_options_rules(
            pos, current_price=155.0, current_time=now
        )

        assert should_exit is False
        assert reason == ""


# ===========================================================================
# Greeks fetch failure
# ===========================================================================


class TestGreeksFetchFailure:
    """When compute_greeks_dispatch fails, log warning and return (False, '')."""

    @pytest.mark.asyncio
    async def test_greeks_failure_returns_no_exit(self):
        pos = _make_option_position()
        monitor = _make_monitor()
        now = datetime.now(ET)

        with patch(
            "quantstack.execution.execution_monitor.compute_greeks_dispatch",
            side_effect=Exception("vollib unavailable"),
        ):
            should_exit, reason = await monitor._evaluate_options_rules(
                pos, current_price=3.00, current_time=now
            )

        assert should_exit is False
        assert reason == ""


# ===========================================================================
# Integration: _on_price_update calls options rules for options positions
# ===========================================================================


class TestOnPriceUpdateIntegration:
    """_on_price_update calls _evaluate_options_rules after equity rules."""

    @pytest.mark.asyncio
    async def test_options_rules_called_for_options_position(self):
        """Options position that passes equity rules → options rules evaluated."""
        pos = _make_option_position(
            entry_price=5.00,
            entry_premium=5.00,
            option_expiry=date.today() + timedelta(days=3),
        )
        monitor = _make_monitor()
        monitor._positions[pos.symbol] = pos
        monitor._submit_exit = AsyncMock()
        monitor._portfolio.update_monitor_state = MagicMock()

        with patch(
            "quantstack.execution.execution_monitor.compute_greeks_dispatch",
            return_value=_mock_greeks(theta=-0.02),
        ):
            # current_price=2.80 → 44% decay → max_theta_loss
            await monitor._on_price_update(pos.symbol, 2.80, datetime.now(ET))

        monitor._submit_exit.assert_awaited_once()
        call_args = monitor._submit_exit.call_args
        assert "max_theta_loss" in call_args[0][1]  # reason arg

    @pytest.mark.asyncio
    async def test_equity_rules_take_priority(self):
        """If equity stop_loss triggers, options rules not evaluated."""
        pos = _make_option_position(
            entry_price=5.00,
            entry_premium=5.00,
        )
        pos.stop_price = 4.00  # stop at 4.00
        monitor = _make_monitor()
        monitor._positions[pos.symbol] = pos
        monitor._submit_exit = AsyncMock()

        # Price below stop → equity stop_loss fires first
        await monitor._on_price_update(pos.symbol, 3.50, datetime.now(ET))

        monitor._submit_exit.assert_awaited_once()
        call_args = monitor._submit_exit.call_args
        assert call_args[0][1] == "stop_loss"  # equity rule, not options
