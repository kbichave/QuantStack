# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for RiskGate.

Uses in-memory TradingContext so every test starts with a clean portfolio.
Sentinel file is redirected to tmp_path to avoid touching real files.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
from quantstack.context import create_trading_context
from quantstack.execution.risk_gate import RiskGate, RiskLimits


@pytest.fixture
def halt_sentinel(tmp_path: Path) -> Path:
    return tmp_path / "DAILY_HALT_ACTIVE"


@pytest.fixture
def ctx():
    context = create_trading_context(db_path=":memory:", initial_cash=100_000.0)
    yield context
    context.db.close()


@pytest.fixture
def gate(ctx, halt_sentinel, monkeypatch) -> RiskGate:
    """Fresh RiskGate wired to the in-memory portfolio, with temp sentinel."""
    monkeypatch.setattr(RiskGate, "DAILY_HALT_SENTINEL", halt_sentinel)
    return RiskGate(
        limits=RiskLimits(
            max_position_pct=0.10,
            max_position_notional=20_000.0,
            max_gross_exposure_pct=1.50,
            daily_loss_limit_pct=0.02,
            min_daily_volume=500_000,
            max_participation_pct=0.01,
        ),
        portfolio=ctx.portfolio,
    )


class TestApprovedOrders:
    def test_small_buy_order_approved(self, gate):
        verdict = gate.check("SPY", "buy", 10, 450.0, daily_volume=10_000_000)
        assert verdict.approved

    def test_approved_quantity_set(self, gate):
        verdict = gate.check("SPY", "buy", 10, 450.0, daily_volume=10_000_000)
        assert verdict.approved_quantity is not None
        assert verdict.approved_quantity > 0

    def test_sell_order_approved(self, gate):
        verdict = gate.check("SPY", "sell", 10, 450.0, daily_volume=10_000_000)
        assert verdict.approved


class TestUnknownVolumeRejection:
    def test_volume_zero_is_rejected(self, gate):
        verdict = gate.check("SPY", "buy", 100, 450.0, daily_volume=0)
        assert not verdict.approved
        assert any(v.rule == "unknown_volume" for v in verdict.violations)

    def test_volume_zero_reason_is_informative(self, gate):
        verdict = gate.check("SPY", "buy", 100, 450.0, daily_volume=0)
        reason = verdict.reason
        assert "volume" in reason.lower()


class TestRestrictedSymbols:
    def test_restricted_symbol_rejected(self, ctx, halt_sentinel, monkeypatch):
        monkeypatch.setattr(RiskGate, "DAILY_HALT_SENTINEL", halt_sentinel)
        gate = RiskGate(
            limits=RiskLimits(restricted_symbols={"GME", "AMC"}),
            portfolio=ctx.portfolio,
        )
        verdict = gate.check("GME", "buy", 100, 20.0, daily_volume=10_000_000)
        assert not verdict.approved
        assert any(v.rule == "restricted_symbol" for v in verdict.violations)

    def test_non_restricted_symbol_passes(self, gate):
        verdict = gate.check("SPY", "buy", 10, 450.0, daily_volume=10_000_000)
        assert verdict.approved


class TestLiquidityCheck:
    def test_low_volume_rejected(self, gate):
        # min_daily_volume is 500_000; pass 100_000 (illiquid penny stock)
        verdict = gate.check("PENNY", "buy", 100, 5.0, daily_volume=100_000)
        assert not verdict.approved
        assert any(v.rule == "min_daily_volume" for v in verdict.violations)

    def test_high_volume_passes(self, gate):
        verdict = gate.check("SPY", "buy", 10, 450.0, daily_volume=80_000_000)
        assert verdict.approved


class TestParticipationRate:
    def test_order_capped_at_participation_limit(self, gate):
        # max_participation_pct=0.01 (1%); ADV=1_000_000 → max order = 10_000
        verdict = gate.check("SPY", "buy", 50_000, 450.0, daily_volume=1_000_000)
        # Should be approved but quantity scaled down
        assert verdict.approved
        assert verdict.approved_quantity is not None
        assert verdict.approved_quantity <= int(1_000_000 * 0.01)


class TestDailyHaltSentinel:
    def test_daily_halt_rejects_all_orders(self, gate):
        gate._daily_halted = date.today()
        verdict = gate.check("SPY", "buy", 10, 450.0, daily_volume=10_000_000)
        assert not verdict.approved
        assert any(v.rule == "daily_loss_halt" for v in verdict.violations)

    def test_halt_sentinel_written_on_trigger(self, gate, halt_sentinel):
        gate._daily_halted = date.today()
        gate._write_halt_sentinel()
        assert halt_sentinel.exists()

    def test_halt_sentinel_loaded_on_new_instance(
        self, ctx, halt_sentinel, monkeypatch
    ):
        """A restarted process should reload the halt from the sentinel file."""
        monkeypatch.setattr(RiskGate, "DAILY_HALT_SENTINEL", halt_sentinel)
        g1 = RiskGate(portfolio=ctx.portfolio)
        g1._daily_halted = date.today()
        g1._write_halt_sentinel()

        g2 = RiskGate(portfolio=ctx.portfolio)
        assert g2.is_halted()

    def test_reset_daily_halt_clears_halt_and_sentinel(self, gate, halt_sentinel):
        gate._daily_halted = date.today()
        gate._write_halt_sentinel()
        assert halt_sentinel.exists()

        gate.reset_daily_halt()
        assert not gate.is_halted()
        assert not halt_sentinel.exists()

    def test_reset_halt_when_sentinel_absent_is_safe(self, gate):
        gate._daily_halted = date.today()
        gate.reset_daily_halt()  # sentinel never written, should not raise


class TestRiskGateMethods:
    def test_is_halted_false_initially(self, gate):
        assert gate.is_halted() is False

    def test_approved_quantity_respected(self, gate):
        """Callers must use approved_quantity, not the original requested quantity."""
        verdict = gate.check("SPY", "buy", 50_000, 450.0, daily_volume=1_000_000)
        assert verdict.approved_quantity is not None
        # approved_quantity must be <= requested (never inflated)
        assert verdict.approved_quantity <= 50_000
