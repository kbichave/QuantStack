# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
LiveIntradayLoop — continuous intraday trading: market open → flatten → close.

Wires existing streaming, feature computation, and execution infrastructure
into a single long-running async process.

Pipeline:
    StreamingAdapter → BarPublisher → [LiveBarStore, IncrementalFeatureEngine]
        → [IntradayPositionManager.on_features, AsyncExecutionLoop.on_features]

Session lifecycle:
    1. Start adapter, subscribe symbols, warm up features
    2. Enable trading at market open (09:30 ET)
    3. Process bars continuously: features → signals → execution
    4. Disable new entries at cutoff (15:30 ET)
    5. Flatten all positions at flatten time (15:55 ET)
    6. Shutdown at market close (16:00 ET)

Usage:
    loop = LiveIntradayLoop(symbols=["SPY", "QQQ"], timeframe="M1")
    report = asyncio.run(loop.run())
"""

from __future__ import annotations

import asyncio
import os
import signal
import time
from dataclasses import dataclass, field
from datetime import date, datetime, time as dtime

import pytz
from loguru import logger

from quantstack.config.timeframes import Timeframe
from quantstack.context import create_trading_context
from quantstack.core.execution.async_execution_loop import AsyncExecutionLoop
from quantstack.core.execution.fill_tracker import FillEvent, FillTracker
from quantstack.core.execution.risk_gate import PreTradeRiskGate
from quantstack.core.execution.smart_order_router import SmartOrderRouter
from quantstack.data.storage import DataStore
from quantstack.data.streaming.base import BarEvent
from quantstack.data.streaming.incremental_features import IncrementalFeatureEngine
from quantstack.data.streaming.live_store import LiveBarStore
from quantstack.data.streaming.publisher import BarPublisher
from quantstack.execution.kill_switch import get_kill_switch
from quantstack.execution.paper_broker import OrderRequest, get_paper_broker
from quantstack.intraday.position_manager import IntradayPositionManager
from quantstack.intraday.signal_evaluator import IntradaySignalEvaluator

from quantstack.data.streaming.alpaca_stream import AlpacaStreamingAdapter
from quantstack.data.streaming.polygon_stream import PolygonStreamingAdapter
from quantstack.data.streaming.ibkr_stream import IBKRStreamingAdapter

ET = pytz.timezone("US/Eastern")

_TIMEFRAME_MAP = {
    "S5": Timeframe.S5,
    "M1": Timeframe.M1,
    "M5": Timeframe.M5,
    "M15": Timeframe.M15,
    "M30": Timeframe.M30,
}


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


@dataclass
class IntradayReport:
    """Summary of one intraday trading session."""

    date: date
    symbols: list[str]
    bars_processed: int = 0
    trades_submitted: int = 0
    trades_filled: int = 0
    positions_flattened: int = 0
    realized_pnl: float = 0.0
    max_drawdown_pct: float = 0.0
    errors: list[str] = field(default_factory=list)
    session_duration_seconds: float = 0.0


# ---------------------------------------------------------------------------
# LiveIntradayLoop
# ---------------------------------------------------------------------------


class LiveIntradayLoop:
    """Continuous intraday trading loop: market open → flatten → close.

    Args:
        symbols: Tickers to trade. If None, loaded from active intraday strategies.
        timeframe: Bar granularity ("M1", "M5", "S5").
        provider: Streaming provider ("alpaca", "polygon", "ibkr", "paper").
        paper_mode: Force paper trading (default True; live requires USE_REAL_TRADING=true).
        flatten_time_et: Time to flatten all positions (ET).
        entry_cutoff_et: No new entries after this time (ET).
        max_trades_per_day: Hard cap on trades per session.
        trailing_stop_atr_mult: ATR multiplier for trailing stops.
        max_hold_bars: Force exit after N bars (0 = disabled).
        strategies: Intraday strategies (list of dicts). If None, loaded from DB.
        db_path: Ignored — DataStore uses PostgreSQL. None = default.
        dry_run: If True, initialize everything but don't start streaming.
    """

    def __init__(
        self,
        symbols: list[str] | None = None,
        timeframe: str = "M1",
        provider: str = "alpaca",
        paper_mode: bool = True,
        flatten_time_et: str = "15:55",
        entry_cutoff_et: str = "15:30",
        max_trades_per_day: int = 50,
        trailing_stop_atr_mult: float = 2.0,
        max_hold_bars: int = 0,
        strategies: list[dict] | None = None,
        db_path: str | None = None,
        dry_run: bool = False,
    ) -> None:
        self._symbols = [s.upper() for s in (symbols or [])]
        self._timeframe_str = timeframe.upper()
        self._timeframe = _TIMEFRAME_MAP.get(self._timeframe_str, Timeframe.M1)
        self._provider = provider.lower()
        self._paper_mode = (
            paper_mode or not os.getenv("USE_REAL_TRADING", "").lower() == "true"
        )
        self._flatten_time_et = flatten_time_et
        self._entry_cutoff_et = entry_cutoff_et
        self._max_trades = max_trades_per_day
        self._trailing_atr = trailing_stop_atr_mult
        self._max_hold_bars = max_hold_bars
        self._strategies = strategies
        self._db_path = db_path
        self._dry_run = dry_run

        self._shutdown_event = asyncio.Event()
        self._bar_count = 0

    async def run(self) -> IntradayReport:
        """Run the full intraday session. Blocks until market close or shutdown."""
        t0 = time.monotonic()
        report = IntradayReport(date=date.today(), symbols=list(self._symbols))

        logger.info(
            f"[IntradayLoop] Starting | symbols={self._symbols} "
            f"timeframe={self._timeframe_str} provider={self._provider} "
            f"paper={self._paper_mode}"
        )

        # ── 1. Load strategies if not provided ─────────────────────────────
        strategies = self._strategies or self._load_strategies()
        if not strategies:
            logger.warning(
                "[IntradayLoop] No intraday strategies found — nothing to trade"
            )
            report.errors.append("No intraday strategies")
            return report

        # ── 2. Load symbols from strategies if not specified ───────────────
        if not self._symbols:
            self._symbols = self._symbols_from_strategies(strategies)
            report.symbols = list(self._symbols)

        if not self._symbols:
            logger.warning("[IntradayLoop] No symbols to trade")
            report.errors.append("No symbols")
            return report

        # ── 3. Build infrastructure ────────────────────────────────────────
        data_store = DataStore(db_path=self._db_path, read_only=True)
        fill_tracker = FillTracker(starting_cash=100_000)
        publisher = BarPublisher()
        live_store = LiveBarStore(data_store)
        feature_engine = IncrementalFeatureEngine(live_store, timeframe=self._timeframe)

        # Broker execute function (paper mode wrapper)
        async def broker_execute(
            symbol: str, side: str, quantity: float, reason: str
        ) -> dict:
            """Submit an exit order through the paper broker."""
            broker = get_paper_broker()
            req = OrderRequest(
                symbol=symbol,
                side=side,
                quantity=int(quantity),
                order_type="market",
            )
            fill = broker.execute(req)
            if fill:
                fill_tracker.update_fill(
                    FillEvent(
                        order_id=fill.order_id,
                        symbol=symbol,
                        side=side,
                        filled_qty=fill.filled_quantity,
                        avg_fill_price=fill.fill_price,
                    )
                )
            return {
                "fill": fill is not None,
                "symbol": symbol,
                "side": side,
                "reason": reason,
            }

        # Kill switch
        def kill_switch_active() -> bool:
            try:
                return get_kill_switch().is_active()
            except Exception:
                return False

        # Position manager
        position_manager = IntradayPositionManager(
            fill_tracker=fill_tracker,
            broker_execute_fn=broker_execute,
            kill_switch_fn=kill_switch_active,
            flatten_time_et=self._flatten_time_et,
            trailing_stop_atr_mult=self._trailing_atr,
            max_hold_bars=self._max_hold_bars,
        )

        # Signal evaluator
        evaluator = IntradaySignalEvaluator(
            strategies=strategies,
            position_manager=position_manager,
            entry_cutoff_et=self._entry_cutoff_et,
            max_trades_per_day=self._max_trades,
        )

        # Execution loop (reuse existing AsyncExecutionLoop)
        risk_gate = PreTradeRiskGate()
        router = SmartOrderRouter(
            alpaca_broker=None,
            ibkr_broker=None,
            fill_tracker=fill_tracker,
            paper=self._paper_mode,
        )
        exec_loop = AsyncExecutionLoop(
            signal_evaluator=evaluator,
            risk_gate=risk_gate,
            router=router,
        )

        # Bar counter callback
        async def count_bars(bar: BarEvent) -> None:
            self._bar_count += 1

        # ── 4. Wire the pipeline ───────────────────────────────────────────
        # Feature engine consumes from publisher via queue
        feature_engine.add_callback(position_manager.on_features)
        feature_engine.add_callback(exec_loop.on_features)

        # Register fill callback on execution loop results → position manager
        original_process = exec_loop._process_signal

        async def _instrumented_process(features):
            """Wrap execution loop to register entries with position manager."""
            result = await original_process(features)
            # If an order was placed, register entry with position manager
            if result and hasattr(result, "symbol"):
                position_manager.register_entry(
                    symbol=features.symbol,
                    price=features.close,
                    atr=features.atr,
                )
            return result

        exec_loop._process_signal = _instrumented_process

        # ── 5. Start ──────────────────────────────────────────────────────
        await live_store.start()
        await exec_loop.start()

        # Install signal handlers for graceful shutdown
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda: self._shutdown_event.set())

        if self._dry_run:
            logger.info(
                "[IntradayLoop] Dry run — skipping streaming. All components initialized."
            )
            report.session_duration_seconds = time.monotonic() - t0
            return report

        # ── 6. Create streaming adapter ────────────────────────────────────
        adapter = self._create_adapter()
        if adapter is None:
            # Paper/replay mode: use a mock adapter or return
            logger.info(
                "[IntradayLoop] No streaming adapter (paper mode) — session ready"
            )
            report.session_duration_seconds = time.monotonic() - t0
            return report

        adapter.add_callback(count_bars)
        adapter.add_callback(publisher.on_bar)

        # Store subscriber
        store_queue = publisher.subscribe("live_store")
        feat_queue = publisher.subscribe("feature_engine")

        async def _drain_to_store():
            while True:
                bar = await store_queue.get()
                if bar is None:
                    break
                await live_store.on_bar(bar)

        async def _drain_to_features():
            while True:
                bar = await feat_queue.get()
                if bar is None:
                    break
                await feature_engine.on_bar(bar)

        # ── 7. Run until shutdown ──────────────────────────────────────────
        logger.info(
            f"[IntradayLoop] Subscribing to {self._symbols} on {self._provider}"
        )

        try:
            await adapter.subscribe(self._symbols, self._timeframe)

            async with asyncio.TaskGroup() as tg:
                tg.create_task(_drain_to_store())
                tg.create_task(_drain_to_features())
                tg.create_task(self._wait_for_close())

        except* Exception as eg:
            for exc in eg.exceptions:
                logger.error(f"[IntradayLoop] Task error: {exc}")
                report.errors.append(str(exc))
        finally:
            # Ensure flatten before shutdown
            if not position_manager.is_flattened:
                await position_manager.flatten_all(reason="session_shutdown")

            await publisher.shutdown()
            await exec_loop.stop()
            await live_store.stop()
            await adapter.shutdown()

        # ── 8. Build report ────────────────────────────────────────────────
        report.bars_processed = self._bar_count
        report.trades_submitted = position_manager.trades_today
        report.positions_flattened = len(
            [
                e
                for e in position_manager.exit_log
                if e.get("reason", "").startswith("flatten")
            ]
        )
        report.realized_pnl = fill_tracker.daily_realised_pnl()
        report.session_duration_seconds = time.monotonic() - t0

        stats = exec_loop.stats()
        report.trades_filled = stats.get("orders_placed", 0)

        logger.info(
            f"[IntradayLoop] Session complete | bars={report.bars_processed} "
            f"trades={report.trades_submitted} pnl=${report.realized_pnl:.2f} "
            f"duration={report.session_duration_seconds:.0f}s"
        )
        return report

    # ── Internal helpers ────────────────────────────────────────────────────

    async def _wait_for_close(self) -> None:
        """Wait until market close (16:00 ET) or shutdown signal."""
        while not self._shutdown_event.is_set():
            now_et = datetime.now(ET).time()
            if now_et >= dtime(16, 0):
                logger.info("[IntradayLoop] Market close reached — shutting down")
                break
            # Check every 5 seconds
            try:
                await asyncio.wait_for(self._shutdown_event.wait(), timeout=5.0)
                break
            except asyncio.TimeoutError:
                continue

    def _create_adapter(self):
        """Create the streaming adapter for the configured provider."""
        if self._provider == "alpaca":
            return AlpacaStreamingAdapter()

        elif self._provider == "polygon":
            return PolygonStreamingAdapter()

        elif self._provider == "ibkr":
            return IBKRStreamingAdapter()

        elif self._provider in ("paper", "replay"):
            return None

        logger.warning(
            f"[IntradayLoop] Unknown provider '{self._provider}' — no streaming"
        )
        return None

    def _load_strategies(self) -> list[dict]:
        """Load active intraday strategies from the strategy DB."""
        try:
            ctx = create_trading_context(db_path=self._db_path or ":memory:")
            rows = ctx.db.execute(
                "SELECT * FROM strategies WHERE status IN ('live', 'forward_testing') "
                "AND (time_horizon = 'intraday' OR holding_period_days <= 1)"
            ).fetchall()
            columns = [desc[0] for desc in ctx.db.description]
            return [dict(zip(columns, row)) for row in rows]
        except Exception as exc:
            logger.warning(f"[IntradayLoop] Could not load strategies: {exc}")
            return []

    def _symbols_from_strategies(self, strategies: list[dict]) -> list[str]:
        """Extract unique symbols from strategy parameters."""
        symbols = set()
        for s in strategies:
            params = s.get("parameters", {})
            if isinstance(params, dict):
                sym = params.get("symbol") or params.get("symbols")
                if isinstance(sym, str):
                    symbols.add(sym.upper())
                elif isinstance(sym, list):
                    symbols.update(s.upper() for s in sym)
        # Fallback: default watchlist
        if not symbols:
            symbols = {"SPY", "QQQ", "AAPL", "MSFT", "NVDA"}
        return sorted(symbols)
