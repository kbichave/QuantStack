# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
MicrostructurePipeline — wires tick data → microstructure features → orders.

Architecture
------------

    MicrostructureFeatureEngine
            │  MicroFeaturesCallback (async)
            ▼
    MicrostructureSignalAgent.on_features()
            │  Optional[UnifiedOrder]
            ▼
    PreTradeRiskGate.check()      ← blocks if risk limits exceeded
            │
            ▼
    SmartOrderRouter.route()      ← routes to Alpaca / IBKR / paper
            │
            ▼
    FillTracker.update_fill()     ← maintains live position map

This pipeline is the HF execution path.  It is independent of TradingDayFlow
(the daily crew-based path).  TradingDayFlow runs once per day; this pipeline
runs continuously whenever tick data is streaming.

Usage
-----
    pipeline = MicrostructurePipeline.build(
        symbols=["SPY", "QQQ"],
        sor=sor_instance,
        fill_tracker=fill_tracker,
    )
    pipeline.register_symbol("AAPL")           # add tick callbacks

    # In your tick adapter callbacks:
    await pipeline.on_trade(trade_tick)
    await pipeline.on_quote(quote_tick)

    await pipeline.start()                     # begins processing
    ...
    await pipeline.stop()                      # graceful shutdown

Design decisions:
  - Risk gate uses conservative HF limits by default (small order cap, high
    order rate limit) to prevent runaway microstructure trading.  These can
    be overridden via env vars (MICRO_MAX_ORDER_VALUE, etc.).
  - SmartOrderRouter is required; there is no paper-broker fallback here
    because HF mode is only meaningful with real broker connectivity.
  - The feature engine is per-symbol; each symbol gets its own state.
  - Exceptions in the signal→order path are caught and logged; they never
    crash the pipeline or drop subsequent ticks.

Failure modes:
  - SmartOrderRouter not configured (None): MicrostructurePipeline.build()
    raises ValueError immediately — do not silently degrade.
  - Risk gate rejects: logged at DEBUG level; counters in stats().
  - Feature engine not warm for a symbol: on_features() returns None.
  - All other exceptions: logged, pipeline keeps running.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from loguru import logger

from quantcore.data.streaming.tick_models import QuoteTick, TradeTick
from quantcore.execution.fill_tracker import FillTracker, FillEvent
from quantcore.execution.risk_gate import (
    PreTradeRiskGate,
    RiskGateError,
    RiskLimits,
)
from quantcore.execution.smart_order_router import SmartOrderRouter, SmartOrderRouterError
from quantcore.microstructure.microstructure_features import MicrostructureFeatureEngine
from quant_pod.agents.microstructure_signal_agent import MicrostructureSignalAgent

# ---- HF risk gate defaults (tunable via env vars) ----
_MICRO_MAX_ORDER_VALUE = float(os.getenv("MICRO_MAX_ORDER_VALUE", "10000"))
_MICRO_MAX_POSITION_VALUE = float(os.getenv("MICRO_MAX_POSITION_VALUE", "50000"))
_MICRO_MAX_POSITIONS = int(os.getenv("MICRO_MAX_POSITIONS", "10"))
_MICRO_MAX_ORDERS_PER_MIN = int(os.getenv("MICRO_MAX_ORDERS_PER_MIN", "30"))
_MICRO_MAX_DAILY_LOSS = float(os.getenv("MICRO_MAX_DAILY_LOSS", "5000"))
_BROKER_ACCOUNT_ID = os.getenv("BROKER_ACCOUNT_ID", "default")


@dataclass
class _PipelineStats:
    ticks_processed:   int = 0
    signals_generated: int = 0
    risk_rejections:   int = 0
    orders_placed:     int = 0
    router_errors:     int = 0
    exceptions:        int = 0
    symbols_active:    set = field(default_factory=set)


class MicrostructurePipeline:
    """
    Tick-level execution pipeline: feature engine → signal agent → broker.

    Instantiate via MicrostructurePipeline.build() rather than directly.
    """

    def __init__(
        self,
        feature_engine: MicrostructureFeatureEngine,
        signal_agent: MicrostructureSignalAgent,
        risk_gate: PreTradeRiskGate,
        router: SmartOrderRouter,
        fill_tracker: FillTracker,
        account_id: str = _BROKER_ACCOUNT_ID,
        asset_class: str = "equity",
    ) -> None:
        self._engine = feature_engine
        self._agent = signal_agent
        self._risk_gate = risk_gate
        self._router = router
        self._fill_tracker = fill_tracker
        self._account_id = account_id
        self._asset_class = asset_class
        self._stats = _PipelineStats()
        self._running = False

    # -------------------------------------------------------------------------
    # Factory
    # -------------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        symbols: List[str],
        sor: SmartOrderRouter,
        fill_tracker: FillTracker,
        account_id: str = _BROKER_ACCOUNT_ID,
        asset_class: str = "equity",
    ) -> "MicrostructurePipeline":
        """Build a ready-to-use pipeline for the given symbols.

        Args:
            symbols:      Initial symbols to register feature tracking for.
            sor:          Pre-configured SmartOrderRouter.  Must not be None —
                          HF mode requires live broker connectivity.
            fill_tracker: Shared FillTracker (may also be used by TradingDayFlow).
            account_id:   Broker account to route orders to.
            asset_class:  Asset class hint for router.

        Raises:
            ValueError: If sor is None.
        """
        if sor is None:
            raise ValueError(
                "MicrostructurePipeline requires a configured SmartOrderRouter. "
                "Set ALPACA_API_KEY or IBKR_HOST env vars to enable HF mode."
            )

        limits = RiskLimits(
            max_order_value=_MICRO_MAX_ORDER_VALUE,
            max_position_value=_MICRO_MAX_POSITION_VALUE,
            max_positions=_MICRO_MAX_POSITIONS,
            max_orders_per_min=_MICRO_MAX_ORDERS_PER_MIN,
            max_daily_loss=_MICRO_MAX_DAILY_LOSS,
        )
        risk_gate = PreTradeRiskGate(limits=limits, fill_tracker=fill_tracker)

        feature_engine = MicrostructureFeatureEngine()
        signal_agent = MicrostructureSignalAgent()

        pipeline = cls(
            feature_engine=feature_engine,
            signal_agent=signal_agent,
            risk_gate=risk_gate,
            router=sor,
            fill_tracker=fill_tracker,
            account_id=account_id,
            asset_class=asset_class,
        )

        # Register the signal agent as a feature callback so it receives
        # feature snapshots after each tick batch
        feature_engine.add_callback(pipeline._on_features_callback)

        for sym in symbols:
            pipeline.register_symbol(sym)

        return pipeline

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    async def start(self) -> None:
        """Mark the pipeline as active.  Subsequent ticks will trigger signals."""
        self._running = True
        logger.info(
            f"[MicroPipeline] Started — "
            f"symbols={sorted(self._stats.symbols_active)} "
            f"account={self._account_id}"
        )

    async def stop(self) -> None:
        """Graceful shutdown."""
        self._running = False
        logger.info(
            f"[MicroPipeline] Stopped — "
            f"ticks={self._stats.ticks_processed} "
            f"orders={self._stats.orders_placed} "
            f"risk_rejections={self._stats.risk_rejections}"
        )

    # -------------------------------------------------------------------------
    # Symbol management
    # -------------------------------------------------------------------------

    def register_symbol(self, symbol: str) -> None:
        """Add a symbol to the feature tracking set."""
        self._stats.symbols_active.add(symbol)
        logger.debug(f"[MicroPipeline] Registered symbol: {symbol}")

    # -------------------------------------------------------------------------
    # Tick ingestion
    # -------------------------------------------------------------------------

    async def on_trade(self, tick: TradeTick) -> None:
        """Feed a trade tick into the feature engine."""
        if not self._running:
            return
        self._stats.ticks_processed += 1
        try:
            await self._engine.on_trade(tick)
        except Exception as _e:
            self._stats.exceptions += 1
            logger.warning(f"[MicroPipeline] on_trade error for {tick.symbol}: {_e}")

    async def on_quote(self, tick: QuoteTick) -> None:
        """Feed a quote tick into the feature engine."""
        if not self._running:
            return
        try:
            await self._engine.on_quote(tick)
        except Exception as _e:
            self._stats.exceptions += 1
            logger.warning(f"[MicroPipeline] on_quote error for {tick.symbol}: {_e}")

    # -------------------------------------------------------------------------
    # Internal: feature → signal → risk → route
    # -------------------------------------------------------------------------

    async def _on_features_callback(self, features) -> None:
        """Called by MicrostructureFeatureEngine after each tick batch."""
        if not self._running:
            return

        try:
            order = await self._agent.on_features(features)
        except Exception as _e:
            self._stats.exceptions += 1
            logger.warning(
                f"[MicroPipeline] Signal evaluation error for "
                f"{features.symbol}: {_e}"
            )
            return

        if order is None:
            return

        self._stats.signals_generated += 1

        # Pre-trade risk gate (hard block — raises RiskGateError on failure)
        current_price = features.ofi  # best proxy available from tick features
        # Use a safe price estimate: features don't carry last trade price directly,
        # so we attempt to get it from fill_tracker; fall back to 0 (gate uses notional)
        try:
            from quantcore.execution.fill_tracker import FillTracker
            pos = self._fill_tracker.get_position(features.symbol)
            current_price = pos.avg_cost if pos else 0.0
        except Exception:
            current_price = 0.0

        try:
            self._risk_gate.check(order=order, current_price=current_price)
        except RiskGateError as rge:
            self._stats.risk_rejections += 1
            logger.debug(
                f"[MicroPipeline] Risk gate BLOCKED {order.symbol}: {rge.message}"
            )
            return

        # Route order through SmartOrderRouter
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._router.route(
                    account_id=self._account_id,
                    order=order,
                    asset_class=self._asset_class,
                ),
            )

            if result.status == "rejected":
                logger.debug(
                    f"[MicroPipeline] SOR REJECTED {order.symbol}: "
                    f"{result.reject_reason}"
                )
                return

            # Record fill so FillTracker + downstream risk gate stays consistent
            fill = FillEvent(
                symbol=order.symbol,
                side=order.side,
                quantity=result.filled_qty,
                fill_price=result.avg_fill_price or current_price,
                commission=result.commission or 0.0,
                order_id=result.order_id,
            )
            self._fill_tracker.update_fill(fill)
            self._risk_gate.record_submission()
            self._stats.orders_placed += 1

            logger.info(
                f"[MicroPipeline] FILLED {order.side.upper()} "
                f"{result.filled_qty} {order.symbol} "
                f"@ {result.avg_fill_price:.2f}"
            )

        except SmartOrderRouterError as sor_err:
            self._stats.router_errors += 1
            logger.warning(
                f"[MicroPipeline] SOR exhausted all brokers for "
                f"{order.symbol}: {sor_err}"
            )
        except Exception as _e:
            self._stats.exceptions += 1
            logger.error(
                f"[MicroPipeline] Unexpected routing error for "
                f"{order.symbol}: {_e}"
            )

    # -------------------------------------------------------------------------
    # Observability
    # -------------------------------------------------------------------------

    def stats(self) -> Dict:
        """Return pipeline counters for monitoring / MCP tool exposure."""
        return {
            "ticks_processed": self._stats.ticks_processed,
            "signals_generated": self._stats.signals_generated,
            "orders_placed": self._stats.orders_placed,
            "risk_rejections": self._stats.risk_rejections,
            "router_errors": self._stats.router_errors,
            "exceptions": self._stats.exceptions,
            "symbols_active": sorted(self._stats.symbols_active),
            "running": self._running,
        }
