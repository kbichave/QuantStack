# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for the governance package (mandate + mandate_check + cio_agent).

All tests run without a database or LLM — DB access and time are mocked.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from quantstack.governance.mandate import (
    DailyMandate,
    _default_mandate,
    get_active_mandate,
)
from quantstack.governance.mandate_check import (
    MandateVerdict,
    mandate_check,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mandate(**overrides) -> DailyMandate:
    """Build a DailyMandate with sensible defaults, overridable per-field."""
    defaults = dict(
        mandate_id="test-mandate-001",
        date="2026-04-07",
        regime_assessment="normal",
        allowed_sectors=["Technology", "Healthcare", "Finance"],
        blocked_sectors=[],
        max_new_positions=5,
        max_daily_notional=50_000.0,
        strategy_directives={},
        risk_overrides={},
        focus_areas=["momentum"],
        reasoning="Test mandate",
        created_at=datetime(2026, 4, 7, 13, 0, tzinfo=timezone.utc),
    )
    defaults.update(overrides)
    return DailyMandate(**defaults)


# ---------------------------------------------------------------------------
# CIO agent tests
# ---------------------------------------------------------------------------

class TestCIOAgent:
    """Verify the CIO agent produces a valid DailyMandate."""

    def test_cio_produces_valid_daily_mandate(self):
        """generate_daily_mandate returns a DailyMandate with all fields present."""
        from quantstack.governance.cio_agent import generate_daily_mandate

        # Mock persist_mandate to avoid DB access
        with patch("quantstack.governance.cio_agent.persist_mandate"):
            mandate = asyncio.get_event_loop().run_until_complete(
                generate_daily_mandate()
            )

        assert isinstance(mandate, DailyMandate)
        assert mandate.mandate_id
        assert mandate.date
        assert mandate.regime_assessment
        assert isinstance(mandate.allowed_sectors, list)
        assert len(mandate.allowed_sectors) > 0
        assert isinstance(mandate.blocked_sectors, list)
        assert mandate.max_new_positions > 0
        assert mandate.max_daily_notional > 0
        assert isinstance(mandate.strategy_directives, dict)
        assert isinstance(mandate.risk_overrides, dict)
        assert isinstance(mandate.focus_areas, list)
        assert mandate.reasoning
        assert mandate.created_at is not None


# ---------------------------------------------------------------------------
# Mandate check gate tests
# ---------------------------------------------------------------------------

class TestMandateCheckGate:
    """Verify the mandate_check gate enforces mandate constraints."""

    def test_mandate_check_rejects_trade_in_blocked_sector(self):
        """Trades in a blocked sector are rejected."""
        mandate = _make_mandate(blocked_sectors=["Energy", "Materials"])

        with patch(
            "quantstack.governance.mandate_check.get_active_mandate",
            return_value=mandate,
        ):
            verdict = mandate_check(
                symbol="XOM",
                sector="Energy",
                side="buy",
                notional=5_000.0,
                strategy_id="swing_momentum",
            )

        assert not verdict.approved
        assert "blocked" in verdict.rejection_reason.lower()

    def test_mandate_check_rejects_when_max_new_positions_reached(self):
        """Reject when today's entry count has reached max_new_positions."""
        mandate = _make_mandate(max_new_positions=3)

        with (
            patch(
                "quantstack.governance.mandate_check.get_active_mandate",
                return_value=mandate,
            ),
            patch(
                "quantstack.governance.mandate_check._count_todays_entries",
                return_value=3,
            ),
        ):
            verdict = mandate_check(
                symbol="AAPL",
                sector="Technology",
                side="buy",
                notional=5_000.0,
                strategy_id="swing_momentum",
            )

        assert not verdict.approved
        assert "position" in verdict.rejection_reason.lower()

    def test_mandate_check_rejects_when_max_daily_notional_exceeded(self):
        """Reject when cumulative notional + proposed exceeds daily limit."""
        mandate = _make_mandate(max_daily_notional=50_000.0)

        with (
            patch(
                "quantstack.governance.mandate_check.get_active_mandate",
                return_value=mandate,
            ),
            patch(
                "quantstack.governance.mandate_check._count_todays_entries",
                return_value=0,
            ),
            patch(
                "quantstack.governance.mandate_check._sum_todays_notional",
                return_value=48_000.0,
            ),
        ):
            verdict = mandate_check(
                symbol="AAPL",
                sector="Technology",
                side="buy",
                notional=5_000.0,
                strategy_id="swing_momentum",
            )

        assert not verdict.approved
        assert "notional" in verdict.rejection_reason.lower()

    def test_mandate_check_approves_trade_within_all_constraints(self):
        """A trade that meets all constraints is approved."""
        mandate = _make_mandate(
            allowed_sectors=["Technology"],
            blocked_sectors=[],
            max_new_positions=5,
            max_daily_notional=50_000.0,
            strategy_directives={},
        )

        with (
            patch(
                "quantstack.governance.mandate_check.get_active_mandate",
                return_value=mandate,
            ),
            patch(
                "quantstack.governance.mandate_check._count_todays_entries",
                return_value=2,
            ),
            patch(
                "quantstack.governance.mandate_check._sum_todays_notional",
                return_value=20_000.0,
            ),
        ):
            verdict = mandate_check(
                symbol="AAPL",
                sector="Technology",
                side="buy",
                notional=5_000.0,
                strategy_id="swing_momentum",
            )

        assert verdict.approved
        assert verdict.rejection_reason is None

    def test_mandate_check_is_hard_gate(self):
        """mandate_check returns a MandateVerdict with bool approved field."""
        mandate = _make_mandate(blocked_sectors=["all"])

        with patch(
            "quantstack.governance.mandate_check.get_active_mandate",
            return_value=mandate,
        ):
            verdict = mandate_check(
                symbol="AAPL",
                sector="Technology",
                side="buy",
                notional=5_000.0,
                strategy_id="test",
            )

        assert isinstance(verdict, MandateVerdict)
        assert isinstance(verdict.approved, bool)

    def test_mandate_check_rejects_paused_strategy(self):
        """Trades for a paused strategy are rejected."""
        mandate = _make_mandate(
            strategy_directives={"swing_momentum": "pause"},
            max_new_positions=10,
            max_daily_notional=100_000.0,
        )

        with patch(
            "quantstack.governance.mandate_check.get_active_mandate",
            return_value=mandate,
        ):
            verdict = mandate_check(
                symbol="AAPL",
                sector="Technology",
                side="buy",
                notional=5_000.0,
                strategy_id="swing_momentum",
            )

        assert not verdict.approved
        assert "pause" in verdict.rejection_reason.lower()

    def test_mandate_check_rejects_exit_strategy(self):
        """Trades for a strategy with 'exit' directive are rejected."""
        mandate = _make_mandate(
            strategy_directives={"mean_reversion": "exit"},
            max_new_positions=10,
            max_daily_notional=100_000.0,
        )

        with patch(
            "quantstack.governance.mandate_check.get_active_mandate",
            return_value=mandate,
        ):
            verdict = mandate_check(
                symbol="AAPL",
                sector="Technology",
                side="buy",
                notional=5_000.0,
                strategy_id="mean_reversion",
            )

        assert not verdict.approved
        assert "exit" in verdict.rejection_reason.lower()


# ---------------------------------------------------------------------------
# Conservative default tests
# ---------------------------------------------------------------------------

class TestConservativeDefault:
    """Verify the conservative default mandate is safe."""

    def test_default_mandate_has_zero_new_positions(self):
        """The default mandate must not allow any new positions."""
        default = _default_mandate("2026-04-07")
        assert default.max_new_positions == 0

    def test_default_mandate_does_not_liquidate(self):
        """All strategy_directives in the default are 'pause', not 'exit'.

        This ensures existing positions are preserved — the conservative
        default freezes activity without forced liquidation.
        """
        default = _default_mandate("2026-04-07")
        for strategy_id, directive in default.strategy_directives.items():
            assert directive == "pause", (
                f"Strategy '{strategy_id}' has directive '{directive}' — "
                f"expected 'pause' (conservative default must not liquidate)"
            )

    def test_default_mandate_blocks_all_sectors(self):
        """The default mandate blocks all sectors."""
        default = _default_mandate("2026-04-07")
        assert "all" in [s.lower() for s in default.blocked_sectors]

    def test_get_active_mandate_returns_default_when_no_row_after_0930(self):
        """When DB has no mandate and it's past 09:30 ET, return conservative default."""
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchone.return_value = None
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_conn)
        mock_ctx.__exit__ = MagicMock(return_value=False)

        # 14:00 UTC = 10:00 ET (past 09:30 cutoff)
        fake_now = datetime(2026, 4, 7, 14, 0, 0, tzinfo=timezone.utc)

        with (
            patch("quantstack.governance.mandate.db_conn", return_value=mock_ctx),
            patch(
                "quantstack.governance.mandate.datetime",
                wraps=datetime,
            ) as mock_dt,
        ):
            mock_dt.now.return_value = fake_now
            mandate = get_active_mandate("2026-04-07")

        assert mandate is not None
        assert mandate.max_new_positions == 0
        assert "Conservative default" in mandate.reasoning


# ---------------------------------------------------------------------------
# Kill switch precedence / pre-mandate window
# ---------------------------------------------------------------------------

class TestPreMandateWindow:
    """Verify behaviour when no mandate exists (pre-09:30 ET)."""

    def test_mandate_check_approves_when_no_mandate(self):
        """Before 09:30 ET with no mandate in DB, trades are approved.

        This prevents the mandate gate from blocking early operations
        before the CIO agent has run.
        """
        with patch(
            "quantstack.governance.mandate_check.get_active_mandate",
            return_value=None,
        ):
            verdict = mandate_check(
                symbol="AAPL",
                sector="Technology",
                side="buy",
                notional=5_000.0,
                strategy_id="swing_momentum",
            )

        assert verdict.approved

    def test_get_active_mandate_returns_none_before_0930(self):
        """Before 09:30 ET (13:30 UTC), get_active_mandate returns None."""
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchone.return_value = None
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_conn)
        mock_ctx.__exit__ = MagicMock(return_value=False)

        # 12:00 UTC = 08:00 ET (before 09:30 cutoff)
        fake_now = datetime(2026, 4, 7, 12, 0, 0, tzinfo=timezone.utc)

        with (
            patch("quantstack.governance.mandate.db_conn", return_value=mock_ctx),
            patch(
                "quantstack.governance.mandate.datetime",
                wraps=datetime,
            ) as mock_dt,
        ):
            mock_dt.now.return_value = fake_now
            mandate = get_active_mandate("2026-04-07")

        assert mandate is None
