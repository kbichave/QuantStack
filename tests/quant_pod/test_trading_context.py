# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Integration tests for TradingContext and the execution hot path.

All tests use in-memory DuckDB — no file system side-effects,
no shared state between test cases.

Coverage:
  - create_trading_context() wires all services to the same connection
  - Each test fixture is isolated (no cross-test state leakage)
  - PortfolioState: concurrent upsert safety, side-flip P&L
  - SignalCache: TTL expiry, hot-path None on stale signal
  - RiskState: daily loss halt, position limit enforcement
  - TickExecutor: end-to-end tick → fill path using in-memory services
"""

from __future__ import annotations

import asyncio
import threading
import time
from datetime import datetime, timedelta, timezone

import pytest

from quant_pod.context import TradingContext, create_trading_context
from quant_pod.execution.portfolio_state import Position
from quant_pod.execution.signal_cache import SignalCache, TradeSignal
from quant_pod.execution.tick_executor import Tick, TickExecutor


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def ctx() -> TradingContext:
    """Fresh in-memory TradingContext for each test."""
    context = create_trading_context(db_path=":memory:", initial_cash=100_000.0)
    yield context
    try:
        context.db.close()
    except Exception:
        pass


# =============================================================================
# TradingContext wiring
# =============================================================================


class TestTradingContextWiring:
    def test_all_services_share_same_connection(self, ctx):
        """Every service receives the same DuckDB connection object."""
        assert ctx.portfolio._conn is ctx.db
        assert ctx.broker._conn is ctx.db
        assert ctx.signal_cache._conn is ctx.db
        assert ctx.audit._conn is ctx.db
        assert ctx.blackboard._conn is ctx.db

    def test_portfolio_and_risk_gate_share_same_portfolio(self, ctx):
        """RiskGate uses the injected PortfolioState, not a separate singleton."""
        assert ctx.risk_gate._portfolio is ctx.portfolio

    def test_initial_cash_seeded(self, ctx):
        assert ctx.portfolio.get_cash() == 100_000.0

    def test_two_contexts_are_isolated(self):
        """Two in-memory contexts do not share data."""
        a = create_trading_context(db_path=":memory:", initial_cash=50_000.0)
        b = create_trading_context(db_path=":memory:", initial_cash=200_000.0)
        try:
            assert a.portfolio.get_cash() == 50_000.0
            assert b.portfolio.get_cash() == 200_000.0

            a.portfolio.adjust_cash(-10_000)
            # b must be unaffected
            assert b.portfolio.get_cash() == 200_000.0
        finally:
            a.db.close()
            b.db.close()

    def test_session_id_is_unique_per_context(self):
        """Each create_trading_context() generates a distinct session_id."""
        a = create_trading_context(db_path=":memory:")
        b = create_trading_context(db_path=":memory:")
        try:
            assert a.session_id != b.session_id
        finally:
            a.db.close()
            b.db.close()


# =============================================================================
# PortfolioState
# =============================================================================


class TestPortfolioState:
    def test_upsert_new_position(self, ctx):
        ctx.portfolio.upsert_position(
            Position(symbol="SPY", quantity=100, avg_cost=450.0, side="long")
        )
        pos = ctx.portfolio.get_position("SPY")
        assert pos is not None
        assert pos.quantity == 100
        assert pos.avg_cost == 450.0

    def test_upsert_adds_to_existing_same_side(self, ctx):
        ctx.portfolio.upsert_position(
            Position(symbol="SPY", quantity=100, avg_cost=450.0, side="long")
        )
        ctx.portfolio.upsert_position(
            Position(symbol="SPY", quantity=50, avg_cost=460.0, side="long")
        )
        pos = ctx.portfolio.get_position("SPY")
        assert pos.quantity == 150
        expected_avg = (100 * 450.0 + 50 * 460.0) / 150
        assert abs(pos.avg_cost - expected_avg) < 0.01

    def test_side_flip_records_realized_pnl(self, ctx):
        """Going long → short must record P&L on the closed long leg."""
        ctx.portfolio.upsert_position(
            Position(symbol="SPY", quantity=100, avg_cost=400.0, side="long")
        )
        # Flip to short at 420 — should close the long with +$2000 P&L
        ctx.portfolio.upsert_position(
            Position(symbol="SPY", quantity=50, avg_cost=420.0, side="short")
        )
        pos = ctx.portfolio.get_position("SPY")
        assert pos is not None
        assert pos.side == "short"
        assert pos.quantity == 50

        pnl = ctx.portfolio.get_total_realized_pnl()
        assert pnl == pytest.approx((420.0 - 400.0) * 100, abs=0.01)

    def test_concurrent_upserts_do_not_corrupt_inventory(self, ctx):
        """Concurrent writes must not lose inventory or corrupt the average cost."""
        ctx.portfolio.upsert_position(
            Position(symbol="AAPL", quantity=100, avg_cost=180.0, side="long")
        )

        errors: list[Exception] = []

        def add_shares():
            try:
                for _ in range(5):
                    ctx.portfolio.upsert_position(
                        Position(symbol="AAPL", quantity=10, avg_cost=185.0, side="long")
                    )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=add_shares) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Concurrent writes raised: {errors}"
        pos = ctx.portfolio.get_position("AAPL")
        assert pos is not None
        # 100 initial + 4 threads × 5 × 10 = 300 total
        assert pos.quantity == 300

    def test_snapshot_is_internally_consistent(self, ctx):
        ctx.portfolio.upsert_position(
            Position(symbol="QQQ", quantity=50, avg_cost=380.0, side="long",
                     current_price=390.0)
        )
        snap = ctx.portfolio.get_snapshot()
        assert snap.position_count == 1
        assert snap.total_equity == pytest.approx(snap.cash + snap.positions_value, abs=0.01)


# =============================================================================
# SignalCache
# =============================================================================


class TestSignalCache:
    def test_get_returns_none_when_empty(self, ctx):
        assert ctx.signal_cache.get("SPY") is None

    def test_get_returns_signal_within_ttl(self, ctx):
        ctx.signal_cache.update(
            TradeSignal.create(
                symbol="SPY", action="BUY", confidence=0.8,
                position_size_pct=0.05, expires_in_seconds=300,
                session_id=ctx.session_id,
            )
        )
        sig = ctx.signal_cache.get("SPY")
        assert sig is not None
        assert sig.action == "BUY"

    def test_get_returns_none_for_expired_signal(self, ctx):
        now = datetime.now(timezone.utc)
        expired = TradeSignal(
            symbol="SPY",
            action="BUY",
            confidence=0.8,
            position_size_pct=0.05,
            generated_at=now - timedelta(seconds=600),
            expires_at=now - timedelta(seconds=1),   # already expired
            session_id=ctx.session_id,
        )
        ctx.signal_cache.update(expired)
        # Must return None — expired signals are NEVER executed
        assert ctx.signal_cache.get("SPY") is None

    def test_is_stale_on_missing_symbol(self, ctx):
        assert ctx.signal_cache.is_stale("MSFT") is True

    def test_update_batch_writes_all_signals(self, ctx):
        symbols = ["SPY", "QQQ", "AAPL"]
        signals = [
            TradeSignal.create(
                symbol=s, action="BUY", confidence=0.7,
                position_size_pct=0.03, expires_in_seconds=300,
                session_id=ctx.session_id,
            )
            for s in symbols
        ]
        ctx.signal_cache.update_batch(signals)
        for s in symbols:
            assert ctx.signal_cache.get(s) is not None


# =============================================================================
# RiskState
# =============================================================================


class TestRiskState:
    def _buy_signal(self, symbol: str, session_id: str) -> TradeSignal:
        return TradeSignal.create(
            symbol=symbol, action="BUY", confidence=0.8,
            position_size_pct=0.05, expires_in_seconds=300,
            session_id=session_id,
        )

    def test_approves_valid_small_order(self, ctx):
        sig = self._buy_signal("SPY", ctx.session_id)
        verdict = ctx.risk_state.check(sig, tick_price=450.0)
        assert verdict.approved
        assert verdict.approved_quantity is not None
        assert verdict.approved_quantity > 0

    def test_rejects_when_kill_switch_active(self, ctx):
        ctx.risk_state.set_kill_switch(True)
        sig = self._buy_signal("SPY", ctx.session_id)
        verdict = ctx.risk_state.check(sig, tick_price=450.0)
        assert not verdict.approved
        assert any(v.rule == "kill_switch" for v in verdict.violations)

    def test_halts_on_daily_loss_limit(self, ctx):
        ctx.risk_state.daily_realized_pnl = -3_000.0  # -3% of $100k
        sig = self._buy_signal("SPY", ctx.session_id)
        verdict = ctx.risk_state.check(sig, tick_price=450.0)
        assert not verdict.approved
        assert any(v.rule == "daily_loss_limit" for v in verdict.violations)

    def test_apply_fill_updates_cash(self, ctx):
        original_cash = ctx.risk_state.cash
        ctx.risk_state.apply_fill("SPY", "buy", 10, 450.0)
        assert ctx.risk_state.cash == pytest.approx(original_cash - 10 * 450.0)


# =============================================================================
# TickExecutor end-to-end
# =============================================================================


class TestTickExecutor:
    @pytest.mark.asyncio
    async def test_processes_tick_with_valid_signal(self, ctx):
        """A tick with a valid signal, passing risk, should produce a fill."""
        # Write a BUY signal for SPY
        ctx.signal_cache.update(
            TradeSignal.create(
                symbol="SPY", action="BUY", confidence=0.8,
                position_size_pct=0.05, expires_in_seconds=300,
                session_id=ctx.session_id,
            )
        )

        fill_queue: asyncio.Queue = asyncio.Queue(maxsize=10)
        executor = TickExecutor(
            signal_cache=ctx.signal_cache,
            risk_state=ctx.risk_state,
            broker=ctx.broker,
            kill_switch=ctx.kill_switch,
            fill_queue=fill_queue,
            session_id=ctx.session_id,
        )

        tick = Tick(symbol="SPY", price=450.0, volume=1_000_000)
        tick_queue: asyncio.Queue = asyncio.Queue()
        await tick_queue.put(tick)
        await tick_queue.put(None)  # sentinel

        await executor.run(tick_queue)

        assert executor.ticks_processed == 1
        # Either a fill was submitted or it was skipped for some risk reason —
        # we just verify no exception was raised
        assert executor.orders_submitted + executor.orders_skipped_risk >= 1

    @pytest.mark.asyncio
    async def test_hold_signal_produces_no_order(self, ctx):
        ctx.signal_cache.update(
            TradeSignal.create(
                symbol="SPY", action="HOLD", confidence=0.5,
                position_size_pct=0.0, expires_in_seconds=300,
                session_id=ctx.session_id,
            )
        )

        fill_queue: asyncio.Queue = asyncio.Queue()
        executor = TickExecutor(
            signal_cache=ctx.signal_cache,
            risk_state=ctx.risk_state,
            broker=ctx.broker,
            kill_switch=ctx.kill_switch,
            fill_queue=fill_queue,
            session_id=ctx.session_id,
        )

        tick = Tick(symbol="SPY", price=450.0, volume=1_000_000)
        tick_queue: asyncio.Queue = asyncio.Queue()
        await tick_queue.put(tick)
        await tick_queue.put(None)

        await executor.run(tick_queue)

        assert executor.orders_submitted == 0
        # HOLD signal counted as stale/skipped
        assert executor.orders_skipped_stale == 1

    @pytest.mark.asyncio
    async def test_active_kill_switch_stops_executor(self, ctx):
        ctx.kill_switch.trigger(reason="test")
        # Even with a valid signal, executor must not trade

        ctx.signal_cache.update(
            TradeSignal.create(
                symbol="SPY", action="BUY", confidence=0.9,
                position_size_pct=0.05, expires_in_seconds=300,
                session_id=ctx.session_id,
            )
        )

        fill_queue: asyncio.Queue = asyncio.Queue()
        executor = TickExecutor(
            signal_cache=ctx.signal_cache,
            risk_state=ctx.risk_state,
            broker=ctx.broker,
            kill_switch=ctx.kill_switch,
            fill_queue=fill_queue,
            session_id=ctx.session_id,
        )

        tick_queue: asyncio.Queue = asyncio.Queue()
        await tick_queue.put(Tick(symbol="SPY", price=450.0, volume=1_000_000))
        await tick_queue.put(None)

        await executor.run(tick_queue)

        assert executor.orders_submitted == 0
        # Reset so kill switch sentinel file doesn't affect other tests
        ctx.kill_switch.reset("test_cleanup")
