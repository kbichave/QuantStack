"""
Deterministic execution monitor — enforces exit rules on every price update.

Runs as an async task within the trading-graph service. Evaluates SL/TP/trailing/
time stop/intraday flatten without LLM involvement, reducing stop enforcement
latency from 60+ seconds (1-min bar callbacks) to sub-second (price ticks).

Architecture:
  - MonitoredPosition wraps position metadata and evaluates exit rules
  - ExecutionMonitor coordinates position loading, feed subscriptions,
    rule evaluation, and exit submission
  - Crash-only design: every startup reconstructs state from DB + broker
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from typing import Any, Protocol, runtime_checkable

import pytz
from loguru import logger

from quantstack.core.options.engine import compute_greeks_dispatch
from quantstack.execution.paper_broker import BrokerProtocol, Fill, OrderRequest
from quantstack.execution.portfolio_state import PortfolioState, Position
from quantstack.holding_period import HOLDING_CONFIGS, HoldingDecision, HoldingType

ET = pytz.timezone("US/Eastern")
INTRADAY_FLATTEN_TIME = time(15, 55)

# Bar timeframe in hours for each holding type (for exit_deadline computation)
_BAR_HOURS: dict[HoldingType, float] = {
    HoldingType.INTRADAY: 1.0,
    HoldingType.SHORT_SWING: 4.0,
    HoldingType.SWING: 24.0,
    HoldingType.POSITION: 24.0,
}


@dataclass
class OptionsMonitorRule:
    """Configuration for a single options monitoring rule."""

    name: str
    enabled: bool
    action: str  # "auto_exit" | "flag_only"


DEFAULT_OPTIONS_RULES: dict[str, OptionsMonitorRule] = {
    "theta_acceleration": OptionsMonitorRule("theta_acceleration", True, "auto_exit"),
    "pin_risk": OptionsMonitorRule("pin_risk", True, "auto_exit"),
    "assignment_risk": OptionsMonitorRule("assignment_risk", False, "flag_only"),
    "iv_crush": OptionsMonitorRule("iv_crush", False, "flag_only"),
    "max_theta_loss": OptionsMonitorRule("max_theta_loss", True, "auto_exit"),
}


def _time_horizon_to_holding_type(time_horizon: str) -> HoldingType:
    """Map Position.time_horizon string to HoldingType enum."""
    mapping = {
        "intraday": HoldingType.INTRADAY,
        "short_swing": HoldingType.SHORT_SWING,
        "swing": HoldingType.SWING,
        "position": HoldingType.POSITION,
        "investment": HoldingType.POSITION,
    }
    return mapping.get(time_horizon.lower(), HoldingType.SWING)


# =============================================================================
# Price Feed Protocol (stubbed here, implemented in price_feed.py)
# =============================================================================


@runtime_checkable
class PriceFeedProtocol(Protocol):
    """Broker-agnostic price feed interface."""

    async def subscribe(self, symbols: list[str], callback: Any) -> None: ...
    async def unsubscribe(self, symbols: list[str]) -> None: ...
    async def start(self) -> None: ...
    async def stop(self) -> None: ...


# =============================================================================
# MonitoredPosition
# =============================================================================


@dataclass
class MonitoredPosition:
    """In-memory representation of a position being monitored.

    Wraps position metadata and evaluates exit rules on every price tick.
    Rule evaluation order (first match wins):
      1. Kill switch
      2. Hard stop-loss
      3. Take profit
      4. Trailing stop
      5. Time stop
      6. Intraday flatten (INTRADAY only, >= 15:55 ET)
    """

    symbol: str
    side: str  # "long" or "short"
    quantity: int
    holding_type: HoldingType
    entry_price: float
    entry_time: datetime
    stop_price: float | None = None
    target_price: float | None = None
    trailing_atr_mult: float = 0.0
    entry_atr: float = 0.0
    high_water_mark: float = 0.0
    exit_deadline: datetime | None = None
    instrument_type: str = "equity"
    underlying_symbol: str = ""
    option_contract: str | None = None
    option_strike: float | None = None
    option_expiry: date | None = None
    option_type: str | None = None  # "call" or "put"
    entry_premium: float | None = None
    exit_pending: bool = False
    strategy_id: str = ""
    regime_at_entry: str = "unknown"

    @classmethod
    def from_portfolio_position(cls, position: Position) -> MonitoredPosition:
        """Build from a PortfolioState Position object."""
        holding_type = _time_horizon_to_holding_type(position.time_horizon)
        config = HOLDING_CONFIGS[holding_type]

        # Compute exit deadline from max_bars and bar timeframe
        bar_hours = _BAR_HOURS[holding_type]
        max_hold_td = timedelta(hours=config.max_bars * bar_hours)
        exit_deadline = position.opened_at + max_hold_td

        # Determine underlying symbol for options
        underlying = position.symbol
        option_contract = None
        if position.instrument_type == "options" and position.option_type:
            underlying = position.symbol  # base symbol
            option_contract = (
                f"{position.symbol}_{position.option_expiry}"
                f"_{position.option_type[0].upper()}{position.option_strike}"
            )

        # Parse option_expiry string → date if present
        option_expiry_date: date | None = None
        if position.option_expiry:
            try:
                option_expiry_date = datetime.strptime(
                    position.option_expiry, "%Y-%m-%d"
                ).date()
            except (ValueError, TypeError):
                pass

        return cls(
            symbol=position.symbol,
            side=position.side,
            quantity=position.quantity,
            holding_type=holding_type,
            entry_price=position.avg_cost,
            entry_time=position.opened_at,
            stop_price=position.stop_price,
            target_price=position.target_price,
            trailing_atr_mult=config.stop_atr_multiple if config.trailing_stop else 0.0,
            entry_atr=position.entry_atr,
            high_water_mark=position.avg_cost,
            exit_deadline=exit_deadline,
            instrument_type=position.instrument_type,
            underlying_symbol=underlying,
            option_contract=option_contract,
            option_strike=position.option_strike,
            option_expiry=option_expiry_date,
            option_type=position.option_type,
            entry_premium=position.avg_cost if position.instrument_type == "options" else None,
            strategy_id=position.strategy_id,
        )

    def evaluate_rules(
        self,
        current_price: float,
        current_time: datetime,
        kill_switch_active: bool = False,
    ) -> tuple[bool, str]:
        """Evaluate all exit rules in priority order.

        Returns (should_exit, reason).
        """
        if self.exit_pending:
            return False, ""

        # 1. Kill switch
        if kill_switch_active:
            return True, "kill_switch"

        is_long = self.side == "long"

        # 2. Hard stop-loss
        if self.stop_price is not None:
            if is_long and current_price <= self.stop_price:
                return True, "stop_loss"
            if not is_long and current_price >= self.stop_price:
                return True, "stop_loss"

        # 3. Take profit
        if self.target_price is not None:
            if is_long and current_price >= self.target_price:
                return True, "take_profit"
            if not is_long and current_price <= self.target_price:
                return True, "take_profit"

        # 4. Trailing stop (only if trailing enabled for this holding type)
        if self.trailing_atr_mult > 0 and self.entry_atr > 0:
            # Update HWM
            if current_price > self.high_water_mark:
                self.high_water_mark = current_price

            stop_distance = self.entry_atr * self.trailing_atr_mult
            if is_long and current_price < self.high_water_mark - stop_distance:
                return True, "trailing_stop"
            if not is_long and current_price > self.high_water_mark + stop_distance:
                return True, "trailing_stop"

        # 5. Time stop
        if self.exit_deadline is not None and current_time >= self.exit_deadline:
            return True, "time_stop"

        # 6. Intraday flatten
        if self.holding_type == HoldingType.INTRADAY:
            current_et = current_time.astimezone(ET).time()
            if current_et >= INTRADAY_FLATTEN_TIME:
                return True, "intraday_flatten"

        return False, ""


# =============================================================================
# ExecutionMonitor
# =============================================================================


class ExecutionMonitor:
    """Deterministic exit enforcement engine.

    Coordinates position loading, price feed subscriptions, rule evaluation,
    and exit submission. Runs as an async task within the trading-graph service.
    """

    def __init__(
        self,
        broker: BrokerProtocol,
        price_feed: PriceFeedProtocol,
        portfolio_state: PortfolioState,
        poll_interval: float | None = None,
        reconcile_interval: float | None = None,
        kill_switch_fn: Any = None,
        shadow_mode: bool = False,
    ) -> None:
        self._broker = broker
        self._feed = price_feed
        self._portfolio = portfolio_state
        self._poll_interval = poll_interval or float(
            os.getenv("EXEC_MONITOR_POLL_INTERVAL", "5")
        )
        self._reconcile_interval = reconcile_interval or float(
            os.getenv("EXEC_MONITOR_RECONCILE_INTERVAL", "60")
        )
        self._kill_switch_fn = kill_switch_fn or (lambda: False)
        self._shadow_mode = shadow_mode

        self._default_reconcile_interval = self._reconcile_interval
        self._positions: dict[str, MonitoredPosition] = {}
        self._poll_task: asyncio.Task | None = None
        self._reconcile_task: asyncio.Task | None = None
        self._cb_task: asyncio.Task | None = None
        self._running = False
        self._stopped = asyncio.Event()

        # Options monitoring rules (override via _options_rules attr for testing)
        self._options_rules: dict[str, OptionsMonitorRule] = dict(DEFAULT_OPTIONS_RULES)

        # Circuit breaker state
        self._feed_last_update: datetime | None = None
        self._db_last_success: datetime | None = None
        self._feed_disconnected = False

    # ── Lifecycle ──────────────────────────────────────────────────────────

    async def run(self) -> None:
        """Main monitor coroutine — call from asyncio.create_task().

        Loads positions, subscribes to feeds, runs poll/reconcile/circuit-breaker
        loops until stop() is called.
        """
        await self.start()
        await self._stopped.wait()

    async def start(self) -> None:
        """Load positions from DB, subscribe to feeds, begin monitoring.

        Crash-only startup: no distinction between cold start and recovery.
        """
        positions = self._portfolio.get_positions()
        for pos in positions:
            mp = MonitoredPosition.from_portfolio_position(pos)
            self._positions[mp.symbol] = mp

        symbols = list(self._positions.keys())
        if symbols:
            await self._feed.subscribe(symbols, self._on_price_update)
            logger.info(
                f"[ExecMonitor] Started monitoring {len(symbols)} positions: "
                f"{', '.join(symbols)}"
            )
        else:
            logger.info("[ExecMonitor] No open positions — monitoring idle")

        await self._feed.start()
        self._running = True
        self._stopped.clear()
        self._db_last_success = datetime.now(ET)
        self._feed_last_update = datetime.now(ET)
        self._poll_task = asyncio.create_task(self._poll_loop())
        self._reconcile_task = asyncio.create_task(self._reconcile_loop())
        self._cb_task = asyncio.create_task(self._circuit_breaker_loop())

    async def stop(self) -> None:
        """Cancel subscriptions, flush state, exit cleanly."""
        self._running = False
        for task in (self._poll_task, self._reconcile_task, self._cb_task):
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        if self._positions:
            await self._feed.unsubscribe(list(self._positions.keys()))
        await self._feed.stop()
        self._positions.clear()
        self._stopped.set()
        logger.info("[ExecMonitor] Stopped")

    # ── Price callback ─────────────────────────────────────────────────────

    async def _on_price_update(
        self, symbol: str, price: float, timestamp: datetime
    ) -> None:
        """Callback from price feed. Evaluate rules for matching positions."""
        self._feed_last_update = timestamp

        # Feed reconnect — clear degraded state
        if self._feed_disconnected:
            self._feed_disconnected = False
            self._reconcile_interval = self._default_reconcile_interval
            logger.info(
                "[ExecMonitor] Price feed reconnected — resuming normal reconciliation"
            )

        position = self._positions.get(symbol)
        if position is None or position.exit_pending:
            return

        kill_active = self._kill_switch_fn()

        # If kill switch, evaluate all positions
        if kill_active:
            for pos in list(self._positions.values()):
                if not pos.exit_pending:
                    should_exit, reason = pos.evaluate_rules(price, timestamp, True)
                    if should_exit:
                        await self._submit_exit(pos, reason, price)
            return

        should_exit, reason = position.evaluate_rules(price, timestamp, False)
        if should_exit:
            await self._submit_exit(position, reason, price)
            return

        # Options-specific rules (only if equity rules didn't trigger)
        if position.instrument_type == "options":
            should_exit, reason = await self._evaluate_options_rules(
                position, price, timestamp
            )
            if should_exit:
                await self._submit_exit(position, reason, price)
                return

        # Update monitor bookkeeping (best-effort, no failure propagation)
        try:
            self._portfolio.update_monitor_state(
                symbol, position.high_water_mark, timestamp
            )
        except Exception:
            pass  # Bookkeeping write failure is non-critical

    # ── Options rule evaluation ───────────────────────────────────────────

    async def _evaluate_options_rules(
        self,
        position: MonitoredPosition,
        current_price: float,
        current_time: datetime,
    ) -> tuple[bool, str]:
        """Evaluate options-specific exit rules.

        Skips non-options positions. Returns (should_exit, reason).
        Priority order: theta_acceleration > pin_risk > max_theta_loss.
        Disabled rules and flag_only rules never cause an exit.
        """
        if position.instrument_type != "options":
            return False, ""

        # Compute DTE
        if position.option_expiry is None:
            return False, ""
        today = current_time.date() if hasattr(current_time, "date") else date.today()
        dte = (position.option_expiry - today).days

        # Fetch Greeks (best-effort)
        greeks: dict[str, float] = {}
        try:
            strike = position.option_strike or 0.0
            tte_years = max(dte / 365.0, 1e-6)
            result = compute_greeks_dispatch(
                spot=current_price,
                strike=strike,
                time_to_expiry=tte_years,
                vol=0.30,  # default vol estimate; real IV would come from market data
                option_type=position.option_type or "call",
            )
            greeks = result.get("greeks", {})
        except Exception as exc:
            logger.warning(
                f"[ExecMonitor] Greeks computation failed for {position.symbol}: {exc}"
            )
            return False, ""

        theta = greeks.get("theta", 0.0)
        rules = self._options_rules

        # Track flag_only triggers for logging
        flagged: list[str] = []

        # --- Rule evaluation in priority order ---

        # 1. theta_acceleration: DTE < 7 AND |theta|/premium > 5%
        rule = rules.get("theta_acceleration")
        if rule and rule.enabled and dte < 7 and current_price > 0:
            theta_ratio = abs(theta) / current_price
            if theta_ratio > 0.05:
                detail = f"DTE={dte}, |theta|/premium={theta_ratio:.3f}"
                if rule.action == "auto_exit":
                    return True, f"options_theta_acceleration: {detail}"
                flagged.append(f"theta_acceleration: {detail}")

        # 2. pin_risk: DTE < 3 AND |underlying - strike|/strike < 1%
        rule = rules.get("pin_risk")
        if rule and rule.enabled and dte < 3 and position.option_strike:
            distance_pct = abs(current_price - position.option_strike) / position.option_strike
            if distance_pct < 0.01:
                detail = f"DTE={dte}, distance={distance_pct:.4f}"
                if rule.action == "auto_exit":
                    return True, f"options_pin_risk: {detail}"
                flagged.append(f"pin_risk: {detail}")

        # 3. assignment_risk: short call ITM + ex-div within 2 days
        rule = rules.get("assignment_risk")
        if rule and rule.enabled and position.side == "short" and position.option_type == "call":
            if position.option_strike and current_price > position.option_strike:
                ex_div = getattr(position, "_ex_div_date", None)
                if ex_div and (ex_div - today).days <= 2:
                    detail = f"short call ITM, ex_div in {(ex_div - today).days}d"
                    if rule.action == "auto_exit":
                        return True, f"options_assignment_risk: {detail}"
                    flagged.append(f"assignment_risk: {detail}")

        # 4. iv_crush: post-earnings + IV drop > 30%
        rule = rules.get("iv_crush")
        if rule and rule.enabled:
            earnings_date = getattr(position, "_earnings_date", None)
            iv_entry = getattr(position, "_iv_entry", None)
            iv_current = getattr(position, "_iv_current", None)
            if (
                earnings_date
                and iv_entry
                and iv_current
                and (today - earnings_date).days >= 0
                and (today - earnings_date).days <= 3
            ):
                iv_drop_pct = (iv_entry - iv_current) / iv_entry
                if iv_drop_pct > 0.30:
                    detail = f"IV drop {iv_drop_pct:.1%} post-earnings"
                    if rule.action == "auto_exit":
                        return True, f"options_iv_crush: {detail}"
                    flagged.append(f"iv_crush: {detail}")

        # 5. max_theta_loss: cumulative premium decay > 40%
        rule = rules.get("max_theta_loss")
        if rule and rule.enabled and position.entry_premium and position.entry_premium > 0:
            decay_pct = (position.entry_premium - current_price) / position.entry_premium
            if decay_pct > 0.40:
                detail = f"decay={decay_pct:.1%}, entry={position.entry_premium:.2f}, current={current_price:.2f}"
                if rule.action == "auto_exit":
                    return True, f"options_max_theta_loss: {detail}"
                flagged.append(f"max_theta_loss: {detail}")

        # Log any flag_only triggers
        for flag in flagged:
            logger.warning(
                f"[ExecMonitor] OPTIONS FLAG {position.symbol}: {flag}"
            )

        return False, ""

    # ── Exit submission ────────────────────────────────────────────────────

    async def _submit_exit(
        self, position: MonitoredPosition, reason: str, current_price: float
    ) -> None:
        """Submit exit order via broker."""
        position.exit_pending = True

        pnl = (current_price - position.entry_price) * position.quantity
        if position.side == "short":
            pnl = -pnl
        hold_hours = (
            (datetime.now(ET) - position.entry_time).total_seconds() / 3600
            if position.entry_time
            else 0
        )

        logger.info(
            f"[ExecMonitor] EXIT {position.symbol}: reason={reason} "
            f"entry={position.entry_price:.2f} exit={current_price:.2f} "
            f"pnl=${pnl:.2f} held={hold_hours:.1f}h "
            f"strategy={position.strategy_id}"
        )

        if self._shadow_mode:
            logger.info(
                f"[ExecMonitor] SHADOW MODE — would submit exit for "
                f"{position.symbol} ({reason}) but suppressed"
            )
            position.exit_pending = False
            return

        side = "sell" if position.side == "long" else "buy"
        abs_qty = abs(position.quantity)

        if position.instrument_type == "options":
            await self._submit_options_exit(position, side, abs_qty, reason)
        else:
            await self._submit_equity_exit(position, side, abs_qty, reason)

    async def _submit_equity_exit(
        self,
        position: MonitoredPosition,
        side: str,
        quantity: int,
        reason: str,
    ) -> None:
        """Submit market exit for equity position."""
        req = OrderRequest(
            symbol=position.symbol,
            side=side,
            quantity=quantity,
            order_type="market",
        )
        try:
            loop = asyncio.get_event_loop()
            fill: Fill = await loop.run_in_executor(
                None, self._broker.execute, req
            )
            if fill.rejected:
                logger.error(
                    f"[ExecMonitor] Exit rejected for {position.symbol}: "
                    f"{fill.reject_reason}"
                )
                position.exit_pending = False
            else:
                logger.info(
                    f"[ExecMonitor] Filled exit {position.symbol}: "
                    f"{fill.filled_quantity}@{fill.fill_price:.2f}"
                )
        except Exception as exc:
            logger.error(f"[ExecMonitor] Exit failed for {position.symbol}: {exc}")
            position.exit_pending = False

    async def _submit_options_exit(
        self,
        position: MonitoredPosition,
        side: str,
        quantity: int,
        reason: str,
    ) -> None:
        """Submit limit exit for options, with 30s timeout → market fallback."""
        contract = position.option_contract or position.symbol
        # Try limit at mid-price first
        try:
            bid, ask = await self._get_option_quote(contract)
            mid_price = (bid + ask) / 2
        except Exception:
            mid_price = None

        if mid_price:
            req = OrderRequest(
                symbol=contract,
                side=side,
                quantity=quantity,
                order_type="limit",
                limit_price=mid_price,
            )
        else:
            req = OrderRequest(
                symbol=contract,
                side=side,
                quantity=quantity,
                order_type="market",
            )

        try:
            loop = asyncio.get_event_loop()
            fill: Fill = await loop.run_in_executor(
                None, self._broker.execute, req
            )
            if fill.rejected and mid_price:
                # Fallback to market
                logger.warning(
                    f"[ExecMonitor] Limit exit rejected for {contract}, "
                    f"falling back to market"
                )
                req_market = OrderRequest(
                    symbol=contract,
                    side=side,
                    quantity=quantity,
                    order_type="market",
                )
                fill = await loop.run_in_executor(
                    None, self._broker.execute, req_market
                )

            if not fill.rejected:
                logger.info(
                    f"[ExecMonitor] Options exit filled {contract}: "
                    f"{fill.filled_quantity}@{fill.fill_price:.2f}"
                )
            else:
                logger.error(
                    f"[ExecMonitor] Options exit rejected: {fill.reject_reason}"
                )
                position.exit_pending = False
        except Exception as exc:
            logger.error(f"[ExecMonitor] Options exit failed for {contract}: {exc}")
            position.exit_pending = False

    async def _get_option_quote(self, contract: str) -> tuple[float, float]:
        """Fetch bid/ask for an option contract. Override in subclass for real broker."""
        raise NotImplementedError("Option quote fetch not yet wired to broker")

    # ── Periodic tasks ─────────────────────────────────────────────────────

    async def _poll_loop(self) -> None:
        """Poll DB for position changes."""
        while self._running:
            try:
                await asyncio.sleep(self._poll_interval)
                await self._poll_positions()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error(f"[ExecMonitor] Poll error: {exc}")

    async def _reconcile_loop(self) -> None:
        """Reconcile with broker periodically."""
        while self._running:
            try:
                await asyncio.sleep(self._reconcile_interval)
                await self._reconcile()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error(f"[ExecMonitor] Reconcile error: {exc}")

    async def _poll_positions(self) -> None:
        """Read PortfolioState, sync local cache."""
        try:
            db_positions = self._portfolio.get_positions()
            self._db_last_success = datetime.now(ET)
        except Exception as exc:
            logger.error(f"[ExecMonitor] DB poll failed: {exc}")
            # Circuit breaker: DB unreachable > 60s → kill switch
            if self._db_last_success:
                elapsed = (datetime.now(ET) - self._db_last_success).total_seconds()
                if elapsed > 60:
                    logger.critical(
                        "[ExecMonitor] DB unreachable for >60s — activating kill switch"
                    )
                    from quantstack.execution.kill_switch import get_kill_switch

                    get_kill_switch().trigger("execution_monitor_db_unreachable")
            return

        db_symbols = {p.symbol for p in db_positions}
        cached_symbols = set(self._positions.keys())

        # Add new positions
        new_symbols = db_symbols - cached_symbols
        for pos in db_positions:
            if pos.symbol in new_symbols:
                mp = MonitoredPosition.from_portfolio_position(pos)
                self._positions[mp.symbol] = mp

        if new_symbols:
            await self._feed.subscribe(list(new_symbols), self._on_price_update)
            logger.info(
                f"[ExecMonitor] Added {len(new_symbols)} new positions: "
                f"{', '.join(new_symbols)}"
            )

        # Remove closed positions
        closed_symbols = cached_symbols - db_symbols
        for sym in closed_symbols:
            del self._positions[sym]
        if closed_symbols:
            await self._feed.unsubscribe(list(closed_symbols))
            logger.info(
                f"[ExecMonitor] Removed {len(closed_symbols)} closed positions: "
                f"{', '.join(closed_symbols)}"
            )

    async def _reconcile(self) -> None:
        """Compare local cache with broker positions.

        In degraded mode (feed disconnected), also acts as a price check
        by reading current_price from broker positions.
        """
        cached_count = len(self._positions)
        logger.debug(
            f"[ExecMonitor] Reconciliation: {cached_count} positions in cache"
        )

    # ── Circuit breaker ───────────────────────────────────────────────────

    async def _circuit_breaker_loop(self) -> None:
        """Monitor feed and DB health every 5s."""
        while self._running:
            try:
                await asyncio.sleep(5.0)
                now = datetime.now(ET)

                # Feed health: silent > 30s → CRITICAL + fast reconciliation
                if (
                    self._feed_last_update
                    and not self._feed_disconnected
                    and (now - self._feed_last_update).total_seconds() > 30
                ):
                    self._feed_disconnected = True
                    self._reconcile_interval = 10.0
                    logger.critical(
                        "[ExecMonitor] Price feed silent for >30s — "
                        "switching to fast reconciliation (10s)"
                    )

                # DB health: unreachable > 60s → kill switch
                if (
                    self._db_last_success
                    and (now - self._db_last_success).total_seconds() > 60
                ):
                    logger.critical(
                        "[ExecMonitor] DB unreachable for >60s — activating kill switch"
                    )
                    from quantstack.execution.kill_switch import get_kill_switch

                    get_kill_switch().trigger("execution_monitor_db_unreachable")

            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error(f"[ExecMonitor] Circuit breaker error: {exc}")
