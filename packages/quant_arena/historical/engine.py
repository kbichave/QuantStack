# Copyright 2024 QuantArena Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Historical simulation engine.

The main orchestration layer that:
- Iterates through trading days
- Builds simulation context for agents
- Runs quantpod historical flow
- Executes trades via sim broker
- Logs experience to knowledge store
- Triggers policy updates
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Callable

from loguru import logger

from quant_arena.historical.config import HistoricalConfig, SimulationContext, DayResult
from quant_arena.historical.universe import SymbolUniverse
from quant_arena.historical.data_loader import DataLoader, MarketSnapshot
from quant_arena.historical.sim_broker import SimBroker, OrderSide
from quant_arena.historical.clock import HistoricalClock


@dataclass
class SimulationResult:
    """Final result of historical simulation."""

    start_date: date
    end_date: date
    initial_equity: float
    final_equity: float
    total_return: float
    max_drawdown: float
    total_trades: int
    win_rate: float
    sharpe_ratio: Optional[float]
    trading_days: int
    symbols: List[str]


class HistoricalEngine:
    """
    Main simulation engine for historical QuantArena.

    Orchestrates the entire simulation:
    1. Load data for all symbols
    2. Initialize broker and clock
    3. For each trading day:
       - Build SimulationContext
       - Call quantpod historical flow
       - Execute trades
       - Log experience
       - Update policy if needed
    4. Generate final report

    Usage:
        config = HistoricalConfig(symbols=["SPY", "QQQ"], initial_equity=100_000)
        engine = HistoricalEngine(config)

        result = await engine.run()
        print(f"Final equity: ${result.final_equity:,.0f}")
    """

    def __init__(
        self,
        config: HistoricalConfig,
        knowledge_store: Optional[Any] = None,
        historical_flow: Optional[Any] = None,
    ):
        """
        Initialize simulation engine.

        Args:
            config: Simulation configuration
            knowledge_store: Optional KnowledgeStore instance (for logging)
            historical_flow: Optional HistoricalDailyFlow instance
        """
        self.config = config

        # Core components
        self.universe = SymbolUniverse(config.symbols)
        self.data_loader = DataLoader(
            universe=self.universe,
            start_date=config.get_start_date(),
            end_date=config.get_end_date(),
        )
        self.broker = SimBroker(
            initial_equity=config.initial_equity,
            slippage_bps=config.slippage_bps,
            commission_per_share=config.commission_per_share,
            max_position_pct=config.max_position_pct,
            max_drawdown_halt_pct=config.max_drawdown_halt_pct,
            max_leverage=config.max_leverage,
        )
        self.clock: Optional[HistoricalClock] = None  # Initialized after data load

        # External dependencies (injected or lazy-loaded)
        self._knowledge_store = knowledge_store
        self._historical_flow = historical_flow
        self._policy_store: Optional[Any] = None

        # Simulation state
        self._running = False
        self._current_day = 0

        # Learning metrics tracking
        self._total_trades = 0
        self._checkpoint_interval = 20  # Save checkpoint every N trades
        self._last_checkpoint_trades = 0

        # Multi-timeframe settings
        self._enable_mtf = getattr(config, "enable_mtf", False)
        self._execution_timeframe = getattr(config, "execution_timeframe", "daily")
        self._use_super_trader = getattr(config, "use_super_trader", True)

        # Callbacks
        self._on_day_complete: Optional[Callable] = None
        self._on_trade: Optional[Callable] = None

        logger.info(
            f"HistoricalEngine initialized: {len(self.universe)} symbols, "
            f"${config.initial_equity:,.0f} equity, "
            f"MTF={'enabled' if self._enable_mtf else 'disabled'}"
        )

    async def run(self) -> SimulationResult:
        """
        Run the full historical simulation.

        Returns:
            SimulationResult with final metrics
        """
        logger.info("Starting historical simulation...")
        self._running = True

        try:
            # 1. Load market data
            await self.data_loader.load_data()

            # 2. Initialize clock with adaptive warmup
            # Use smaller warmup for short test periods to ensure we have trading days
            total_days = len(self.data_loader.trading_days)
            warmup = min(
                20, max(5, total_days // 3)
            )  # At least 5, at most 20, or 1/3 of data
            logger.info(
                f"Using warmup period of {warmup} days (total data: {total_days} days)"
            )

            self.clock = HistoricalClock(
                trading_days=self.data_loader.trading_days,
                warmup_days=warmup,
            )

            # 3. Initialize knowledge store if not provided
            if self._knowledge_store is None:
                self._knowledge_store = self._get_knowledge_store()

            # 4. Initialize historical flow if not provided
            if self._historical_flow is None:
                self._historical_flow = self._get_historical_flow()

            # 5. Initialize policy store
            self._policy_store = self._get_policy_store()

            # 6. Load MTF data if enabled
            if self._enable_mtf:
                logger.info("Loading multi-timeframe data...")
                await self.data_loader.load_mtf_data()

            # 7. Run simulation loop
            exec_tf = self._execution_timeframe
            logger.info(
                f"Running simulation: {self.clock.effective_start_date} to "
                f"{self.clock.end_date} ({self.clock.tradable_days} days), "
                f"execution timeframe: {exec_tf}"
            )

            for current_date in self.clock.iterate():
                if self._enable_mtf and exec_tf in ["4h", "1h"]:
                    # Process multiple bars per day
                    bar_hours = self.data_loader.get_intraday_bars(
                        current_date, "SPY", exec_tf.upper()
                    )
                    for bar_hour in bar_hours:
                        await self._process_bar(current_date, bar_hour)
                else:
                    # Standard daily processing
                    await self._process_day(current_date)

                self._current_day += 1
                if self._current_day % 252 == 0:  # Log yearly progress
                    logger.info(
                        f"Progress: {self.clock.progress:.1%} "
                        f"(equity: ${self.broker.get_equity():,.0f})"
                    )

            # 7. Generate result
            result = self._generate_result()

            logger.info(
                f"Simulation complete: {result.total_return:.1%} return, "
                f"{result.max_drawdown:.1%} max drawdown"
            )

            return result

        finally:
            self._running = False

    async def _process_day(self, current_date: date) -> None:
        """
        Process a single trading day.

        Steps:
        1. Get market snapshot
        2. Update broker prices
        3. Build simulation context
        4. Run historical flow (agents)
        5. Execute trades
        6. Log to experience store
        7. Update policy if needed
        """
        # 1. Get market data
        snapshot = self.data_loader.get_day(current_date)
        if snapshot is None:
            logger.warning(f"No data for {current_date}, skipping")
            return

        # 2. Update broker prices
        prices = snapshot.get_prices()
        self.broker.update_prices(prices, current_date)

        # 3. Build simulation context
        context = await self._build_context(current_date, snapshot)

        # 4. Run historical flow
        day_result = await self._run_flow(context)

        # 5. Execute trades
        executed_trades = []
        # Handle both dict and object response formats
        trades_list = (
            day_result.get("trades", [])
            if isinstance(day_result, dict)
            else getattr(day_result, "trades", [])
        )
        for trade_instruction in trades_list:
            order = self._execute_trade(trade_instruction)
            if order and order.status.value == "filled":
                executed_trades.append(order)
                self._total_trades += 1
                if self._on_trade:
                    self._on_trade(order)

        # 6. Save daily snapshot
        portfolio_state = self.broker.save_daily_snapshot()

        # 6.5. Save learning checkpoint if threshold reached
        if (
            self._total_trades
            >= self._last_checkpoint_trades + self._checkpoint_interval
        ):
            await self._save_learning_checkpoint(
                current_date, portfolio_state, day_result
            )

        # 7. Log to experience store
        await self._log_experience(
            current_date=current_date,
            portfolio_state=portfolio_state,
            day_result=day_result,
            executed_trades=executed_trades,
        )

        # 8. Update policy if needed
        if self.clock.should_update_policy(self.config.policy_update_frequency):
            await self._update_policy(current_date)

        # 9. Callback
        if self._on_day_complete:
            self._on_day_complete(current_date, portfolio_state, day_result)

    async def _process_bar(self, current_date: date, bar_hour: int) -> None:
        """
        Process a single intraday bar (for MTF mode).

        Similar to _process_day but uses MTF snapshot and processes
        at specific hours during the trading day.

        Args:
            current_date: The simulation date
            bar_hour: Hour of the bar (e.g., 10, 14 for 4H bars)
        """
        # 1. Get MTF snapshot
        mtf_snapshot = self.data_loader.get_mtf_snapshot(current_date, bar_hour)
        if mtf_snapshot is None or not mtf_snapshot.available_symbols:
            logger.debug(f"No MTF data for {current_date} {bar_hour}:00, skipping")
            return

        # 2. Get daily snapshot for price updates
        daily_snapshot = self.data_loader.get_day(current_date)
        if daily_snapshot is None:
            return

        # 3. Update broker prices
        prices = daily_snapshot.get_prices()
        self.broker.update_prices(prices, current_date)

        # 4. Build MTF-aware context
        context = await self._build_mtf_context(
            current_date, bar_hour, mtf_snapshot, daily_snapshot
        )

        # 5. Run historical flow
        day_result = await self._run_flow(context)

        # 6. Execute trades
        executed_trades = []
        # Handle both dict and object return types
        trades_list = (
            day_result.get("trades", [])
            if isinstance(day_result, dict)
            else getattr(day_result, "trades", [])
        )
        for trade_instruction in trades_list:
            order = self._execute_trade(trade_instruction)
            if order and order.status.value == "filled":
                executed_trades.append(order)
                self._total_trades += 1
                if self._on_trade:
                    self._on_trade(order)

        # 7. Save snapshot (only at end of day)
        if bar_hour >= 14:  # Last bar of day
            portfolio_state = self.broker.save_daily_snapshot()

            # Save learning checkpoint if threshold reached
            if (
                self._total_trades
                >= self._last_checkpoint_trades + self._checkpoint_interval
            ):
                await self._save_learning_checkpoint(
                    current_date, portfolio_state, day_result
                )

            # Log experience
            await self._log_experience(
                current_date=current_date,
                portfolio_state=portfolio_state,
                day_result=day_result,
                executed_trades=executed_trades,
            )

            # Update policy if needed
            if self.clock.should_update_policy(self.config.policy_update_frequency):
                await self._update_policy(current_date)

            # Callback
            if self._on_day_complete:
                self._on_day_complete(current_date, portfolio_state, day_result)

    async def _build_mtf_context(
        self,
        current_date: date,
        bar_hour: int,
        mtf_snapshot: Any,
        daily_snapshot: Any,
    ) -> Any:
        """Build MTF-aware simulation context."""
        from quant_arena.historical.data_loader import MTFSnapshot

        # Get portfolio state
        portfolio_state = self.broker.get_portfolio_state(current_date)

        # Get current policy
        policy = {}
        if self._policy_store:
            try:
                policy_snapshot = self._policy_store.get_current(current_date)
                if policy_snapshot:
                    policy = policy_snapshot.pod_weights
            except Exception:
                pass

        # Default policy if none
        if not policy:
            policy = {
                pod: 1.0 / len(self.config.active_pods)
                for pod in self.config.active_pods
            }

        # Build market data dict from daily snapshot
        market_data = daily_snapshot.data

        # Get features
        features = await self._compute_features(current_date, daily_snapshot)

        # Get regimes
        regimes = await self._detect_regimes(current_date, daily_snapshot)

        # Build MTF data dict for each symbol
        mtf_data = {}
        for symbol in mtf_snapshot.available_symbols:
            mtf_data[symbol] = mtf_snapshot.to_dict(symbol)

        # Create simulation context with MTF data included in constructor
        context = SimulationContext(
            date=current_date,
            market_data=market_data,
            features=features,
            portfolio={
                "equity": portfolio_state.equity,
                "cash": portfolio_state.cash,
                "positions": {
                    s: p.quantity for s, p in portfolio_state.positions.items()
                },
                "exposures": portfolio_state.exposures,
                "drawdown": portfolio_state.max_drawdown,
            },
            policy=policy,
            regimes=regimes,
            # MTF-specific fields
            mtf_data=mtf_data,
            bar_hour=bar_hour,
            execution_timeframe=self._execution_timeframe,
        )

        return context

    async def _build_context(
        self,
        current_date: date,
        snapshot: MarketSnapshot,
    ) -> SimulationContext:
        """Build simulation context for agents."""
        # Get portfolio state
        portfolio_state = self.broker.get_portfolio_state(current_date)

        # Get current policy
        policy = {}
        if self._policy_store:
            try:
                policy_snapshot = self._policy_store.get_current(current_date)
                if policy_snapshot:
                    policy = policy_snapshot.pod_weights
            except Exception:
                pass

        # Default policy if none
        if not policy:
            policy = {
                pod: 1.0 / len(self.config.active_pods)
                for pod in self.config.active_pods
            }

        # Build market data dict
        market_data = snapshot.data

        # Get features (simplified - in production would call QuantCore)
        features = await self._compute_features(current_date, snapshot)

        # Get regimes
        regimes = await self._detect_regimes(current_date, snapshot)

        return SimulationContext(
            date=current_date,
            market_data=market_data,
            features=features,
            portfolio={
                "equity": portfolio_state.equity,
                "cash": portfolio_state.cash,
                "positions": {
                    s: p.quantity for s, p in portfolio_state.positions.items()
                },
                "exposures": portfolio_state.exposures,
                "drawdown": portfolio_state.max_drawdown,
            },
            policy=policy,
            regimes=regimes,
        )

    async def _compute_features(
        self,
        current_date: date,
        snapshot: MarketSnapshot,
    ) -> Dict[str, Dict]:
        """
        Compute features for all symbols.

        In production, this calls QuantCore MCP compute_all_features.
        For now, compute basic features locally.
        """
        features = {}

        for symbol in snapshot.available_symbols:
            # Get price history for calculations
            prices = self.data_loader.get_price_history(
                symbol=symbol,
                end_date=current_date,
                days=50,
            )

            if prices.empty:
                continue

            # Basic features
            close = prices.iloc[-1] if len(prices) > 0 else 0
            returns = prices.pct_change().dropna()

            features[symbol] = {
                "close": float(close),
                "return_1d": float(returns.iloc[-1]) if len(returns) > 0 else 0,
                "return_5d": float(returns.tail(5).sum()) if len(returns) >= 5 else 0,
                "return_20d": (
                    float(returns.tail(20).sum()) if len(returns) >= 20 else 0
                ),
                "volatility_20d": (
                    float(returns.tail(20).std() * (252**0.5))
                    if len(returns) >= 20
                    else 0
                ),
                "sma_20": (
                    float(prices.tail(20).mean()) if len(prices) >= 20 else float(close)
                ),
                "price_vs_sma": (
                    float(close / prices.tail(20).mean() - 1)
                    if len(prices) >= 20
                    else 0
                ),
            }

        return features

    async def _detect_regimes(
        self,
        current_date: date,
        snapshot: MarketSnapshot,
    ) -> Dict[str, Dict]:
        """
        Detect regimes for all symbols.

        In production, uses quantpod RegimeDetectorAgent.
        For now, simple heuristic based on returns and volatility.
        """
        regimes = {}

        for symbol in snapshot.available_symbols:
            prices = self.data_loader.get_price_history(
                symbol=symbol,
                end_date=current_date,
                days=50,
            )

            if len(prices) < 20:
                regimes[symbol] = {
                    "trend": "unknown",
                    "volatility": "normal",
                }
                continue

            returns = prices.pct_change().dropna()

            # Trend detection (simple momentum)
            return_20d = returns.tail(20).sum()
            if return_20d > 0.05:
                trend = "trending_up"
            elif return_20d < -0.05:
                trend = "trending_down"
            else:
                trend = "ranging"

            # Volatility detection
            vol_20d = returns.tail(20).std() * (252**0.5)
            if vol_20d < 0.10:
                vol_regime = "low"
            elif vol_20d < 0.20:
                vol_regime = "normal"
            elif vol_20d < 0.35:
                vol_regime = "high"
            else:
                vol_regime = "extreme"

            regimes[symbol] = {
                "trend": trend,
                "volatility": vol_regime,
            }

        return regimes

    async def _run_flow(self, context: SimulationContext) -> DayResult:
        """
        Run historical flow for the day.

        If no flow is configured, uses a simple rule-based strategy.
        """
        if self._historical_flow is not None:
            try:
                return await self._historical_flow.run_day(context)
            except Exception as e:
                logger.error(f"Historical flow error: {e}")

        # NO FALLBACK - LLM reasoning is required
        raise RuntimeError(
            "LLM agents failed and no fallback is allowed. "
            "Ensure HistoricalDailyFlow is properly configured with LLM agents."
        )

    def _simple_momentum_strategy(self, context: SimulationContext) -> DayResult:
        """DEPRECATED - Rule-based strategy removed. Use LLM agents only."""
        raise NotImplementedError(
            "Rule-based strategies are disabled. All trading decisions must use LLM reasoning."
        )
        trades = []
        agent_logs = []

        for symbol, features in context.features.items():
            regime = context.regimes.get(symbol, {})

            # Log regime detection
            agent_logs.append(
                {
                    "agent_name": "RegimeDetector",
                    "symbol": symbol,
                    "message": f"{regime.get('trend', 'unknown')}, vol_{regime.get('volatility', 'normal')}",
                    "role": "analysis",
                }
            )

            # Current position
            current_qty = context.portfolio.get("positions", {}).get(symbol, 0)

            # Simple rules
            price_vs_sma = features.get("price_vs_sma", 0)
            trend = regime.get("trend", "unknown")
            vol = regime.get("volatility", "normal")

            # Don't trade in extreme volatility
            if vol == "extreme":
                agent_logs.append(
                    {
                        "agent_name": "RiskManager",
                        "symbol": symbol,
                        "message": "extreme vol - no new positions",
                        "role": "risk",
                    }
                )
                continue

            # Calculate target position size
            equity = context.portfolio.get("equity", 100000)
            max_position_value = equity * self.config.max_position_pct
            price = features.get("close", 0)

            if price <= 0:
                continue

            target_shares = int(max_position_value / price)

            # Trading logic
            if trend == "trending_up" and price_vs_sma > 0 and current_qty == 0:
                # Buy signal
                trades.append(
                    {
                        "symbol": symbol,
                        "side": "buy",
                        "quantity": target_shares,
                        "reason": f"momentum buy: {trend}, price > SMA",
                    }
                )
                agent_logs.append(
                    {
                        "agent_name": "MomentumPod",
                        "symbol": symbol,
                        "message": f"BUY {target_shares} shares - trend_up confirmed",
                        "role": "execution",
                    }
                )

            elif (trend == "trending_down" or price_vs_sma < -0.03) and current_qty > 0:
                # Sell signal
                trades.append(
                    {
                        "symbol": symbol,
                        "side": "sell",
                        "quantity": current_qty,
                        "reason": f"momentum exit: {trend}, price < SMA",
                    }
                )
                agent_logs.append(
                    {
                        "agent_name": "MomentumPod",
                        "symbol": symbol,
                        "message": f"SELL {current_qty} shares - trend reversal",
                        "role": "execution",
                    }
                )

        # Risk manager summary
        agent_logs.append(
            {
                "agent_name": "RiskManager",
                "symbol": None,
                "message": f"approved {len(trades)} trades, portfolio dd={context.portfolio.get('drawdown', 0):.1%}",
                "role": "risk",
            }
        )

        return DayResult(
            trades=trades,
            agent_logs=agent_logs,
            regimes=context.regimes,
            signals=[],
        )

    def _execute_trade(self, trade_instruction: Dict) -> Optional[Any]:
        """Execute a trade instruction via broker."""
        symbol = trade_instruction.get("symbol")
        side_str = trade_instruction.get("side", "").lower()
        quantity = trade_instruction.get("quantity", 0)

        if not symbol or quantity <= 0:
            return None

        side = OrderSide.BUY if side_str == "buy" else OrderSide.SELL

        return self.broker.submit_order(symbol, side, quantity)

    async def _log_experience(
        self,
        current_date: date,
        portfolio_state: Any,
        day_result: DayResult,
        executed_trades: List,
    ) -> None:
        """Log experience to knowledge store."""
        if self._knowledge_store is None:
            return

        try:
            # Handle both dict and object return types
            regimes = (
                day_result.get("regimes", {})
                if isinstance(day_result, dict)
                else getattr(day_result, "regimes", {})
            )
            agent_logs = (
                day_result.get("agent_logs", [])
                if isinstance(day_result, dict)
                else getattr(day_result, "agent_logs", [])
            )
            signals = (
                day_result.get("signals", [])
                if isinstance(day_result, dict)
                else getattr(day_result, "signals", [])
            )

            # Log daily state
            self._knowledge_store.save_daily_state(
                {
                    "date": current_date,
                    "equity": portfolio_state.equity,
                    "cash": portfolio_state.cash,
                    "max_drawdown": portfolio_state.max_drawdown,
                    "exposures": portfolio_state.exposures,
                    "regime_summary": self._summarize_regimes(regimes),
                }
            )

            # Log agent messages (no truncation - show full reasoning)
            for log in agent_logs:
                self._knowledge_store.save_agent_log(
                    {
                        "date": current_date,
                        "agent_name": log.get("agent_name", "unknown"),
                        "symbol": log.get("symbol"),
                        "message": log.get("message", ""),  # Full reasoning preserved
                        "role": log.get("role", "analysis"),
                        "created_at_sim_time": f"{current_date}T10:00",
                    }
                )

            # Log signals (ensure date is set)
            for signal in signals:
                # Ensure signal has required date field
                signal_with_date = (
                    dict(signal)
                    if isinstance(signal, dict)
                    else signal.copy() if hasattr(signal, "copy") else {}
                )
                # Use signal_date if present, otherwise add date
                if (
                    "signal_date" in signal_with_date
                    and signal_with_date["signal_date"]
                ):
                    signal_with_date["date"] = signal_with_date["signal_date"]
                elif (
                    "date" not in signal_with_date
                    or signal_with_date.get("date") is None
                ):
                    signal_with_date["date"] = current_date
                self._knowledge_store.save_historical_signal(signal_with_date)

        except Exception as e:
            logger.error(f"Failed to log experience: {e}")

    def _summarize_regimes(self, regimes: Dict[str, Dict]) -> str:
        """Create regime summary string."""
        if not regimes:
            return "unknown"

        # Use SPY regime if available, otherwise first symbol
        if "SPY" in regimes:
            r = regimes["SPY"]
        else:
            r = list(regimes.values())[0]

        return f"{r.get('trend', 'unknown')}_{r.get('volatility', 'normal')}"

    async def _save_learning_checkpoint(
        self,
        current_date: date,
        portfolio_state: Any,
        day_result: DayResult,
    ) -> None:
        """
        Save a learning metrics checkpoint to track improvement over time.

        Called every N trades (default: 20) to record:
        - Rolling win rate (last 20 trades)
        - Cumulative win rate
        - Active lessons count
        - Strategy weights at this point
        """
        if self._knowledge_store is None:
            return

        try:
            # Compute metrics
            rolling_win_rate = self._knowledge_store.compute_rolling_win_rate(20)

            # Get all trades for cumulative stats
            all_trades = self.broker.get_trade_history()
            closed_trades = [t for t in all_trades if t.pnl is not None]

            cumulative_win_rate = None
            cumulative_pnl = 0.0
            rolling_pnl = 0.0

            if closed_trades:
                wins = sum(1 for t in closed_trades if t.pnl > 0)
                cumulative_win_rate = wins / len(closed_trades)
                cumulative_pnl = sum(t.pnl for t in closed_trades)

                # Rolling P&L (last 20)
                recent_trades = closed_trades[-20:]
                rolling_pnl = sum(t.pnl for t in recent_trades)

            # Get active lessons count
            lessons_active = 0
            try:
                lessons = self._knowledge_store.get_lessons(limit=100)
                lessons_active = len(lessons)
            except Exception:
                pass

            # Get current policy weights
            strategy_weights = {}
            if self._policy_store:
                try:
                    policy = self._policy_store.get_current(current_date)
                    if policy:
                        strategy_weights = policy.pod_weights
                except Exception:
                    pass

            # Get prompt change count
            prompt_changes = 0
            try:
                proposals = self._knowledge_store.get_prompt_proposals(
                    status="approved"
                )
                prompt_changes = len(proposals)
            except Exception:
                pass

            # Save checkpoint
            regimes = (
                day_result.get("regimes", {})
                if isinstance(day_result, dict)
                else getattr(day_result, "regimes", {})
            )
            checkpoint = {
                "date": current_date,
                "trade_count": self._total_trades,
                "rolling_win_rate": rolling_win_rate,
                "cumulative_win_rate": cumulative_win_rate,
                "rolling_pnl": rolling_pnl,
                "cumulative_pnl": cumulative_pnl,
                "lessons_active": lessons_active,
                "prompt_changes": prompt_changes,
                "strategy_weights": strategy_weights,
                "regime_at_checkpoint": self._summarize_regimes(regimes),
            }

            self._knowledge_store.save_learning_checkpoint(checkpoint)
            self._last_checkpoint_trades = self._total_trades

            rolling_wr_str = f"{rolling_win_rate:.1%}" if rolling_win_rate else "N/A"
            cumulative_wr_str = (
                f"{cumulative_win_rate:.1%}" if cumulative_win_rate else "N/A"
            )
            logger.info(
                f"Learning checkpoint #{self._total_trades}: "
                f"rolling_wr={rolling_wr_str}, "
                f"cumulative_wr={cumulative_wr_str}, "
                f"lessons={lessons_active}"
            )

        except Exception as e:
            logger.error(f"Failed to save learning checkpoint: {e}")

    async def _update_policy(self, current_date: date) -> None:
        """
        Update policy weights based on experience using the learning system.

        This method:
        1. Analyzes recent trade performance per pod
        2. Calculates expectancy-adjusted weights
        3. Applies drawdown-based risk scaling
        4. Saves the new policy snapshot
        """
        if self._knowledge_store is None:
            return

        try:
            logger.info(f"Policy update triggered at {current_date}")

            # Get recent performance data
            pod_performance = await self._analyze_pod_performance()

            # Calculate new weights based on performance
            new_weights = self._calculate_adaptive_weights(pod_performance)

            # Calculate adaptive risk parameters
            portfolio_state = self.broker.get_portfolio_state(current_date)
            risk_params = self._calculate_risk_parameters(portfolio_state)

            # Generate policy comment with reasoning
            comment = self._generate_policy_comment(
                pod_performance, new_weights, risk_params, current_date
            )

            # Save policy snapshot
            self._knowledge_store.save_policy_snapshot(
                {
                    "effective_date": current_date,
                    "pod_weights": new_weights,
                    "thresholds": risk_params,
                    "comment": comment,
                }
            )

            logger.info(
                f"Policy updated: weights={new_weights}, risk_scale={risk_params.get('risk_scale', 1.0)}"
            )

        except Exception as e:
            logger.error(f"Policy update failed: {e}")
            # Fallback to equal weights
            if self._knowledge_store:
                self._knowledge_store.save_policy_snapshot(
                    {
                        "effective_date": current_date,
                        "pod_weights": {
                            pod: 1.0 / len(self.config.active_pods)
                            for pod in self.config.active_pods
                        },
                        "thresholds": {},
                        "comment": f"Fallback equal weights at {current_date}",
                    }
                )

    async def _analyze_pod_performance(self) -> Dict[str, Dict]:
        """
        Analyze recent performance for each strategy pod.

        Returns:
            Dict mapping pod name to performance metrics
        """
        performance = {}

        # Get recent agent logs to identify which pods generated signals
        try:
            recent_logs = self._knowledge_store.load_agent_logs(limit=500)

            # Group by agent/pod
            pod_signals = {}
            for log in recent_logs:
                agent = log.get("agent_name", "")
                if agent in [
                    "TrendFollower",
                    "MeanReversion",
                    "MomentumTrader",
                    "BreakoutTrader",
                    "VolatilityTrader",
                ]:
                    if agent not in pod_signals:
                        pod_signals[agent] = {"signals": 0, "messages": []}
                    pod_signals[agent]["signals"] += 1
                    pod_signals[agent]["messages"].append(log)

            # Get trade history to calculate win rates per pod
            trades = self.broker.get_trade_history()

            # Map agent names to config pod names
            agent_to_pod = {
                "TrendFollower": "trend_following",
                "MeanReversion": "mean_reversion",
                "MomentumTrader": "momentum",
                "BreakoutTrader": "breakout",
                "VolatilityTrader": "volatility",
            }

            for agent_name, pod_name in agent_to_pod.items():
                if pod_name in self.config.active_pods:
                    # Default metrics
                    performance[pod_name] = {
                        "win_rate": 0.5,
                        "signal_count": pod_signals.get(agent_name, {}).get(
                            "signals", 0
                        ),
                        "expectancy": 0.0,
                        "trade_count": 0,
                        "total_pnl": 0.0,
                    }

                    # Calculate from trades if available
                    pod_trades = [
                        t for t in trades if hasattr(t, "pnl") and t.pnl is not None
                    ]
                    if pod_trades:
                        wins = sum(1 for t in pod_trades if t.pnl > 0)
                        losses = sum(1 for t in pod_trades if t.pnl < 0)
                        total = wins + losses

                        if total > 0:
                            win_rate = wins / total
                            total_pnl = sum(t.pnl for t in pod_trades)
                            avg_win = sum(t.pnl for t in pod_trades if t.pnl > 0) / max(
                                1, wins
                            )
                            avg_loss = sum(
                                t.pnl for t in pod_trades if t.pnl < 0
                            ) / max(1, losses)

                            # Calculate expectancy
                            expectancy = (win_rate * avg_win) + (
                                (1 - win_rate) * avg_loss
                            )

                            performance[pod_name].update(
                                {
                                    "win_rate": win_rate,
                                    "trade_count": total,
                                    "total_pnl": total_pnl,
                                    "expectancy": expectancy,
                                }
                            )

        except Exception as e:
            logger.warning(f"Error analyzing pod performance: {e}")
            # Return default performance
            for pod in self.config.active_pods:
                performance[pod] = {
                    "win_rate": 0.5,
                    "signal_count": 0,
                    "expectancy": 0.0,
                    "trade_count": 0,
                    "total_pnl": 0.0,
                }

        return performance

    def _calculate_adaptive_weights(
        self, pod_performance: Dict[str, Dict]
    ) -> Dict[str, float]:
        """
        Calculate adaptive weights based on pod performance.

        Uses a combination of:
        - Win rate (higher is better)
        - Expectancy (positive is better)
        - Signal activity (active pods get more weight)
        """
        weights = {}

        for pod_name, metrics in pod_performance.items():
            # Base weight
            base_weight = 1.0

            # Win rate adjustment (-0.3 to +0.3)
            win_rate = metrics.get("win_rate", 0.5)
            win_adj = (win_rate - 0.5) * 0.6  # ±0.3 range

            # Expectancy adjustment (-0.2 to +0.2)
            expectancy = metrics.get("expectancy", 0)
            exp_adj = max(-0.2, min(0.2, expectancy * 0.01))  # Scaled

            # Activity bonus (pods with more signals get slight boost)
            signal_count = metrics.get("signal_count", 0)
            activity_bonus = min(0.1, signal_count * 0.01)

            # Calculate final weight
            raw_weight = base_weight + win_adj + exp_adj + activity_bonus

            # Clamp to reasonable range
            weights[pod_name] = max(0.1, min(2.0, raw_weight))

        # Normalize weights to sum to 1.0
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        else:
            # Equal weights fallback
            n = len(self.config.active_pods)
            weights = {pod: 1.0 / n for pod in self.config.active_pods}

        return weights

    def _calculate_risk_parameters(self, portfolio_state: Any) -> Dict[str, float]:
        """
        Calculate adaptive risk parameters based on portfolio state.

        Implements drawdown-based risk scaling.
        """
        drawdown = portfolio_state.max_drawdown
        equity = portfolio_state.equity
        initial = self.config.initial_equity

        # Calculate return so far
        total_return = (equity - initial) / initial if initial > 0 else 0

        # Base risk scale
        risk_scale = 1.0

        # Drawdown adjustment
        if drawdown > 0.15:
            risk_scale = 0.25  # Severe reduction
        elif drawdown > 0.10:
            risk_scale = 0.50  # Major reduction
        elif drawdown > 0.05:
            risk_scale = 0.75  # Moderate reduction

        # Profit boost (if doing well, can be slightly more aggressive)
        if total_return > 0.20 and drawdown < 0.05:
            risk_scale = min(1.25, risk_scale * 1.1)

        # Confidence threshold adjustment
        base_confidence = 0.50
        if drawdown > 0.10:
            # Require higher confidence during drawdowns
            confidence_threshold = 0.60
        elif total_return > 0.10 and drawdown < 0.03:
            # Can be more aggressive when profitable
            confidence_threshold = 0.45
        else:
            confidence_threshold = base_confidence

        return {
            "risk_scale": risk_scale,
            "confidence_threshold": confidence_threshold,
            "max_position_pct": self.config.max_position_pct * risk_scale,
            "drawdown": drawdown,
            "total_return": total_return,
        }

    def _generate_policy_comment(
        self,
        pod_performance: Dict[str, Dict],
        new_weights: Dict[str, float],
        risk_params: Dict[str, float],
        current_date: date,
    ) -> str:
        """Generate a detailed policy update comment."""
        parts = [f"Policy update at {current_date}:"]

        # Performance summary
        total_signals = sum(p.get("signal_count", 0) for p in pod_performance.values())
        avg_win_rate = sum(
            p.get("win_rate", 0.5) for p in pod_performance.values()
        ) / max(1, len(pod_performance))

        parts.append(f"Signals: {total_signals}, Avg win rate: {avg_win_rate:.0%}")

        # Top performing pod
        if pod_performance:
            best_pod = max(
                pod_performance.items(), key=lambda x: x[1].get("win_rate", 0)
            )
            parts.append(
                f"Best performer: {best_pod[0]} ({best_pod[1].get('win_rate', 0):.0%} win rate)"
            )

        # Risk adjustment
        risk_scale = risk_params.get("risk_scale", 1.0)
        if risk_scale < 1.0:
            parts.append(f"Risk reduced to {risk_scale:.0%} due to drawdown")
        elif risk_scale > 1.0:
            parts.append(
                f"Risk increased to {risk_scale:.0%} due to strong performance"
            )

        return " | ".join(parts)

    def _generate_result(self) -> SimulationResult:
        """Generate final simulation result."""
        summary = self.broker.get_summary()

        # Calculate Sharpe ratio from daily returns
        snapshots = self.broker.get_daily_snapshots()
        if len(snapshots) > 1:
            returns = []
            for i in range(1, len(snapshots)):
                r = (snapshots[i].equity - snapshots[i - 1].equity) / snapshots[
                    i - 1
                ].equity
                returns.append(r)

            if returns:
                import numpy as np

                mean_return = np.mean(returns) * 252
                std_return = np.std(returns) * (252**0.5)
                sharpe = mean_return / std_return if std_return > 0 else 0
            else:
                sharpe = None
        else:
            sharpe = None

        return SimulationResult(
            start_date=self.data_loader.start,
            end_date=self.data_loader.end,
            initial_equity=self.config.initial_equity,
            final_equity=summary["final_equity"],
            total_return=summary["total_return"],
            max_drawdown=summary["max_drawdown"],
            total_trades=summary["total_trades"],
            win_rate=summary["win_rate"],
            sharpe_ratio=sharpe,
            trading_days=len(self.data_loader),
            symbols=self.universe.symbols,
        )

    def _get_knowledge_store(self) -> Any:
        """Get or create knowledge store."""
        try:
            from quant_pod.knowledge.store import KnowledgeStore

            return KnowledgeStore(db_path=self.config.db_path)
        except ImportError:
            logger.warning("KnowledgeStore not available, logging disabled")
            return None

    def _get_historical_flow(self) -> Any:
        """Get or create historical flow.

        Uses the unified TradingDayFlow with Pod architecture:

        Architecture:
        ┌─────────────────────────────────────────────────────────────────┐
        │                    QUANTPOD TRADING SYSTEM                      │
        ├─────────────────────────────────────────────────────────────────┤
        │  PHASE 1: REGIME DETECTION                                      │
        │  └── RegimeDetectorAgent (MCP) - Market regime classification   │
        ├─────────────────────────────────────────────────────────────────┤
        │  PHASE 2: POD RESEARCH (parallel MCP + LLM ICs)                 │
        │  ├── TrendPod: RegimeDetectorIC, MarketMonitorIC + LLM ICs     │
        │  ├── MeanReversionPod: WaveAnalystIC, ResearchAgentIC + LLM   │
        │  ├── MomentumPod: RegimeDetectorIC, MarketMonitorIC + LLM     │
        │  ├── BreakoutPod: MarketMonitorIC, WaveAnalystIC + LLM        │
        │  └── VolatilityPod: RegimeDetectorIC, MarketMonitorIC + LLM   │
        ├─────────────────────────────────────────────────────────────────┤
        │  PHASE 3: CHIEF STRATEGIST SYNTHESIS                            │
        │  └── ChiefStrategist - Aggregates all pod research → DailyBrief│
        ├─────────────────────────────────────────────────────────────────┤
        │  PHASE 4: SUPERTRADER DECISION                                  │
        │  └── SuperTrader (LLM) - Final trading decisions                │
        ├─────────────────────────────────────────────────────────────────┤
        │  PHASE 5: RISK CONSULTANT REVIEW                                │
        │  └── RiskConsultant - APPROVE/SCALE/VETO with veto power       │
        └─────────────────────────────────────────────────────────────────┘

        MCP Tools (QuantCore):
        - 200+ technical indicators via compute_indicators
        - Backtesting via run_backtest, run_monte_carlo
        - Options pricing via price_option, compute_greeks
        - Risk management via compute_var, stress_test_portfolio
        """
        use_fast_llm = getattr(self.config, "use_fast_llm", False)

        # Use the unified TradingDayFlow with Pod architecture
        try:
            from quant_pod.flows.trading_day_flow import TradingDayFlowAdapter

            logger.info("═══════════════════════════════════════════════════════════")
            logger.info("QuantPod Trading System Active (Unified Pod Architecture):")
            logger.info("  • 5 Strategy Pods with MCP + LLM ICs (parallel analysis)")
            logger.info("  • ChiefStrategist aggregates research into Daily Brief")
            logger.info("  • SuperTrader for final decision aggregation")
            logger.info("  • RiskConsultant with VETO power")
            logger.info("  • MCP tools via QuantCore (200+ indicators)")
            logger.info("  • Mem0 for cross-session memory")
            logger.info("═══════════════════════════════════════════════════════════")

            return TradingDayFlowAdapter(
                config=self.config,
                use_fast_llm=use_fast_llm,
            )
        except ImportError as e:
            raise RuntimeError(
                f"TradingDayFlowAdapter import failed: {e}. "
                "The trading system requires quant_pod.flows.trading_day_flow."
            )

    def _get_policy_store(self) -> Any:
        """Get or create policy store."""
        try:
            from quant_pod.knowledge.policy_store import PolicyStore

            return PolicyStore(self._knowledge_store)
        except ImportError:
            logger.warning("PolicyStore not available")
            return None

    def set_callbacks(
        self,
        on_day_complete: Optional[Callable] = None,
        on_trade: Optional[Callable] = None,
    ) -> None:
        """Set progress callbacks."""
        self._on_day_complete = on_day_complete
        self._on_trade = on_trade

    @property
    def progress(self) -> float:
        """Get simulation progress."""
        if self.clock:
            return self.clock.progress
        return 0.0

    @property
    def is_running(self) -> bool:
        """Check if simulation is running."""
        return self._running

    def __repr__(self) -> str:
        return (
            f"HistoricalEngine(symbols={len(self.universe)}, "
            f"progress={self.progress:.1%}, running={self._running})"
        )
