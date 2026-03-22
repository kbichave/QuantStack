# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Trading Day Flow — orchestrates the full trading pipeline.

Uses SignalEngine (pure-Python collectors, no LLM) for analysis.

Flow Structure:
    @start(): detect_regime
         │
    @listen(): run_signal_analysis
         │
    @router(): route_execution
         │
    @listen("execute"): execute_trades
    @listen("hold"): log_hold_decision
         │
    finalize_day
"""

from __future__ import annotations

import asyncio
import os
import uuid
from datetime import date, datetime
from typing import Any

from loguru import logger
from pydantic import BaseModel, Field
from quantstack.core.execution.fill_tracker import FillTracker
from quantstack.core.execution.smart_order_router import (
    SmartOrderRouter,
    SmartOrderRouterError,
)
from quantstack.core.execution.tca_engine import OrderSide as TCAOrderSide
from quantstack.core.execution.tca_engine import TCAEngine
from quantstack.core.execution.unified_models import UnifiedOrder

from quantstack.agents.portfolio_optimizer_agent import PortfolioOptimizerAgent

# Import regime detector
from quantstack.agents.regime_detector import RegimeDetectorAgent

# Import audit trail
from quantstack.audit.decision_log import get_decision_log, make_trade_event
from quantstack.crewai_compat import Flow, listen, router, start
from quantstack.crews.schemas import (
    DailyBrief,
)
from quantstack.signal_engine import SignalEngine
from quantstack.execution.broker_factory import get_broker, get_broker_mode
from quantstack.execution.kill_switch import get_kill_switch
from quantstack.execution.paper_broker import OrderRequest

# Import execution layer
from quantstack.execution.portfolio_state import get_portfolio_state
from quantstack.execution.risk_gate import get_risk_gate
from quantstack.execution.signal_cache import SignalCache, TradeSignal

# Import blackboard for cross-agent memory
from quantstack.memory.blackboard import (
    get_blackboard,
    read_blackboard_context,
    write_portfolio_state,
    write_to_blackboard,
)

# =============================================================================
# FLOW STATE MODEL
# =============================================================================


class TradingDayState(BaseModel):
    """Structured state for the trading day flow."""

    # Flow metadata
    current_date: date | None = None
    symbols: list[str] = Field(default_factory=list)

    # Market context
    market_data: dict[str, dict] = Field(default_factory=dict)
    features: dict[str, dict] = Field(default_factory=dict)
    portfolio: dict[str, Any] = Field(default_factory=dict)

    # Phase 1: Regime Detection
    regimes: dict[str, dict] = Field(default_factory=dict)

    # Phase 2-4: Signal Analysis (handled by SignalEngine)
    daily_briefs: dict[str, DailyBrief] = Field(default_factory=dict)
    trade_decisions: list[dict] = Field(default_factory=list)

    # Phase 5: Execution
    approved_trades: list[dict] = Field(default_factory=list)
    executed_trades: list[dict] = Field(default_factory=list)

    # Metrics
    signal_funnel: dict[str, int] = Field(
        default_factory=lambda: {
            "symbols_analyzed": 0,
            "decisions_made": 0,
            "executed": 0,
        }
    )

    # Agent logs for UI
    agent_logs: list[dict] = Field(default_factory=list)

    # Historical context
    historical_context: dict[str, str] = Field(default_factory=dict)

    # Error tracking
    errors: list[str] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True


# =============================================================================
# TRADING DAY FLOW
# =============================================================================


class TradingDayFlow(Flow[TradingDayState]):
    """
    CrewAI Flow for orchestrating a trading day.

    Uses the hierarchical TradingCrew:
    - ICs fetch raw data and metrics
    - Pod Managers compile and coordinate
    - Assistant synthesizes into 1-pager
    - SuperTrader makes final decisions

    NO FALLBACKS - all agents must produce output or fail explicitly.
    """

    def __init__(
        self,
        signal_cache: SignalCache | None = None,
        signal_ttl_seconds: int = 900,
    ):
        """
        Initialize the trading day flow.

        Args:
            signal_cache: When provided, the flow publishes signals to this cache
                instead of executing directly.  The TickExecutor then picks up
                signals and executes them on market ticks.  When None (default),
                the flow executes trades directly (backward-compatible mode).
            signal_ttl_seconds: How long published signals remain valid (default 15 min).
        """
        self._regime_detector = None
        self._signal_engine = None
        self._session_id = str(uuid.uuid4())
        self._audit_log = get_decision_log()
        self._portfolio = get_portfolio_state()
        self._risk_gate = get_risk_gate()
        self._kill_switch = get_kill_switch()
        self._broker = get_broker()
        self._signal_cache = signal_cache
        self._signal_ttl_seconds = signal_ttl_seconds

        # TCA engine tracks arrival prices + post-fill shortfall
        self._tca = TCAEngine()
        # FillTracker maintains live position map for SOR
        self._fill_tracker = FillTracker()
        # SmartOrderRouter: Alpaca/IBKR routing; None when no broker env vars are set
        self._sor = self._build_sor()
        # Portfolio optimizer: converts per-symbol signals into MV-optimal weights
        self._portfolio_optimizer = PortfolioOptimizerAgent()

        # RL online feedback — lazy initialised; never raises on import failure
        self._rl_adapter = None
        self._rl_config = None
        try:
            from quantstack.rl.config import get_rl_config
            from quantstack.rl.online_adapter import PostTradeRLAdapter

            self._rl_config = get_rl_config()
            self._rl_adapter = PostTradeRLAdapter(self._rl_config, self._kill_switch)
        except Exception as _rl_init_err:
            logger.debug(
                f"[RL] Online adapter init skipped (non-fatal): {_rl_init_err}"
            )

        # Keep the blackboard aware of the current session
        get_blackboard().set_session(self._session_id)

        super().__init__()
        mode = "signal-publish" if signal_cache is not None else "direct-execute"
        logger.info(
            f"TradingDayFlow initialized (session={self._session_id[:8]}, "
            f"broker={get_broker_mode()}, mode={mode})"
        )

    @property
    def regime_detector(self) -> RegimeDetectorAgent:
        """Get or create regime detector."""
        if self._regime_detector is None:
            self._regime_detector = RegimeDetectorAgent()
        return self._regime_detector

    @property
    def signal_engine(self) -> SignalEngine:
        """Get or create SignalEngine instance."""
        if self._signal_engine is None:
            self._signal_engine = SignalEngine()
        return self._signal_engine

    def _build_sor(self) -> SmartOrderRouter | None:
        """Build SmartOrderRouter if Alpaca or IBKR env vars are present.

        Returns None when neither broker is configured so that _execute_directly
        falls through to the paper broker unchanged (backward-compatible).
        """
        alpaca_broker = None
        ibkr_broker = None

        if os.getenv("ALPACA_API_KEY"):
            try:
                from alpaca_mcp.client import AlpacaBrokerClient  # type: ignore

                alpaca_broker = AlpacaBrokerClient()
                logger.info("[SOR] Alpaca broker client initialised")
            except Exception as _e:
                logger.debug(f"[SOR] Alpaca init skipped: {_e}")

        if os.getenv("IBKR_HOST"):
            try:
                from ibkr_mcp.client import IBKRBrokerClient  # type: ignore

                ibkr_broker = IBKRBrokerClient()
                logger.info("[SOR] IBKR broker client initialised")
            except Exception as _e:
                logger.debug(f"[SOR] IBKR init skipped: {_e}")

        if alpaca_broker is None and ibkr_broker is None:
            logger.debug("[SOR] No live broker env vars set — direct broker mode")
            return None

        paper = os.getenv("ALPACA_PAPER", "true").lower() != "false"
        return SmartOrderRouter(
            alpaca_broker=alpaca_broker,
            ibkr_broker=ibkr_broker,
            fill_tracker=self._fill_tracker,
            paper=paper,
        )

    # =========================================================================
    # PHASE 1: REGIME DETECTION
    # =========================================================================

    @start()
    def detect_regime(self) -> dict[str, Any]:
        """
        Detect market regimes for all symbols.
        Entry point of the flow.
        """
        # Guard: no trading if kill switch is active
        self._kill_switch.guard()

        current_date = self.state.current_date or date.today()
        symbols = self.state.symbols or ["SPY"]

        # Inject current portfolio state as context so agents know existing positions
        portfolio_context = self._portfolio.as_context_string()
        portfolio_snapshot = self._portfolio.get_snapshot()
        self.state.portfolio = {
            **self.state.portfolio,
            "context": portfolio_context,
            "snapshot": portfolio_snapshot.model_dump(),
        }

        # Pin portfolio state to blackboard so agents can read it in history queries
        write_portfolio_state(portfolio_context)

        # Snapshot portfolio into knowledge store for time-series tracking
        try:
            from quantstack.knowledge.store import KnowledgeStore

            KnowledgeStore().save_portfolio_snapshot(portfolio_snapshot.model_dump())
        except Exception as _snap_err:
            logger.debug(f"Portfolio snapshot to knowledge store failed: {_snap_err}")

        logger.info("")
        logger.info(
            "╔══════════════════════════════════════════════════════════════════╗"
        )
        logger.info(
            f"║  TRADING DAY FLOW: {current_date}                                ║"
        )
        logger.info(
            f"║  Symbols: {symbols}                                              ║"
        )
        logger.info(
            "╚══════════════════════════════════════════════════════════════════╝"
        )
        logger.info("")

        logger.info("═══ PHASE 1: REGIME DETECTION ═══")

        for symbol in symbols:
            try:
                result = self.regime_detector.detect_regime(symbol)

                if result.get("success"):
                    regime = {
                        "trend": result.get("trend_regime", "unknown"),
                        "volatility": result.get("volatility_regime", "normal"),
                        "confidence": result.get("confidence", 0.5),
                        "adx": result.get("adx", 0),
                        "atr_percentile": result.get("atr_percentile", 50),
                    }
                    self.state.regimes[symbol] = regime

                    logger.info(
                        f"[REGIME] {symbol}: {regime['trend']} / {regime['volatility']} "
                        f"({regime['confidence']:.0%} conf)"
                    )
                else:
                    # No fallback - report failure
                    error_msg = result.get("error", "Unknown error")
                    logger.error(f"[REGIME] {symbol}: Detection failed - {error_msg}")
                    self.state.errors.append(f"Regime {symbol}: {error_msg}")
                    self.state.regimes[symbol] = {
                        "trend": "unknown",
                        "volatility": "unknown",
                        "confidence": 0.0,
                        "error": error_msg,
                    }

            except Exception as e:
                logger.error(f"Regime detection failed for {symbol}: {e}")
                self.state.errors.append(f"Regime {symbol}: {str(e)}")
                self.state.regimes[symbol] = {
                    "trend": "unknown",
                    "volatility": "unknown",
                    "confidence": 0.0,
                    "error": str(e),
                }

        self._log_agent(
            agent="RegimeDetector",
            symbol=None,
            message=f"Detected regimes for {len(symbols)} symbols",
            role="monitoring",
        )

        # Get historical context
        for symbol in symbols:
            self.state.historical_context[symbol] = self._get_historical_context(symbol)

        return {"regimes": self.state.regimes, "count": len(self.state.regimes)}

    # =========================================================================
    # PHASE 2-4: CREW ANALYSIS
    # =========================================================================

    @listen(detect_regime)
    def run_signal_analysis(self, regime_result: dict) -> dict[str, Any]:
        """
        Run SignalEngine for each symbol and extract trade decisions.

        SignalEngine replaces TradingCrew: pure-Python collectors, no LLM,
        2-6 seconds per symbol vs 3-5 minutes.
        """
        logger.info("═══ PHASE 2-4: SIGNAL ENGINE ANALYSIS ═══")

        loop = asyncio.new_event_loop()
        try:
            for symbol in self.state.symbols:
                regime = self.state.regimes.get(symbol, {})

                # Skip if regime detection failed completely
                if regime.get("error") and regime.get("confidence", 0) == 0:
                    logger.warning(f"Skipping {symbol} due to regime detection failure")
                    continue

                try:
                    logger.info(
                        f"[SignalEngine] Analyzing {symbol} "
                        f"[{regime.get('trend', 'unknown')}/{regime.get('volatility', 'normal')}]"
                    )

                    brief = loop.run_until_complete(
                        self.signal_engine.run(symbol, regime=regime)
                    )

                    # Map SignalBrief to a decision dict compatible with downstream
                    sym_brief = brief.symbol_briefs[0] if brief.symbol_briefs else None
                    if sym_brief is None:
                        continue

                    bias = sym_brief.consensus_bias
                    confidence = sym_brief.consensus_conviction
                    action = "hold"
                    if bias in ("bullish", "strong_bullish") and confidence >= 0.6:
                        action = "buy"
                    elif bias in ("bearish", "strong_bearish") and confidence >= 0.6:
                        action = "sell"

                    reasoning = sym_brief.market_summary or brief.market_overview

                    decision_dict = {
                        "symbol": symbol,
                        "action": action,
                        "confidence": confidence,
                        "reasoning": reasoning,
                        "bias": bias,
                        "risk_factors": sym_brief.risk_factors,
                        "position_size": "quarter" if confidence < 0.75 else "half",
                    }

                    decision_summary = (
                        f"{action.upper()} conf={confidence:.0%} bias={bias}"
                    )
                    logger.info(f"[SignalEngine] {symbol}: {decision_summary}")

                    self._log_agent(
                        agent="SignalEngine",
                        symbol=symbol,
                        message=f"Analysis complete: {decision_summary}",
                        role="analysis",
                    )

                    self.state.trade_decisions.append(decision_dict)

                    # Audit log
                    audit_event = make_trade_event(
                        session_id=self._session_id,
                        agent_name="SignalEngine",
                        agent_role="analysis",
                        symbol=symbol,
                        action=action,
                        confidence=confidence,
                        reasoning=reasoning,
                        output_structured=decision_dict,
                        portfolio_snapshot=(
                            self._portfolio.get_snapshot().model_dump()
                            if hasattr(self._portfolio.get_snapshot(), "model_dump")
                            else {}
                        ),
                    )
                    snapshot = self._portfolio.get_snapshot()
                    audit_event.portfolio_snapshot = snapshot.model_dump()
                    self._audit_log.record(audit_event)

                    # Write to blackboard
                    self._store_decision(symbol, decision_dict)

                    # If actionable, add to approved
                    if action in ["buy", "sell"]:
                        self.state.approved_trades.append(decision_dict)

                except Exception as e:
                    logger.error(f"SignalEngine analysis failed for {symbol}: {e}")
                    self.state.errors.append(f"SignalEngine {symbol}: {str(e)}")

        finally:
            loop.close()

        # Update metrics
        self.state.signal_funnel["symbols_analyzed"] = len(self.state.symbols)
        self.state.signal_funnel["decisions_made"] = len(self.state.trade_decisions)

        return {
            "decisions": len(self.state.trade_decisions),
            "approved": len(self.state.approved_trades),
        }

    # =========================================================================
    # ROUTER: EXECUTE OR HOLD
    # =========================================================================

    @router(run_signal_analysis)
    def route_execution(self, analysis_result: dict) -> str:
        """Route to execute or hold based on approved trades."""
        if self.state.approved_trades:
            logger.info(
                f"Router: {len(self.state.approved_trades)} approved trades → execute"
            )
            return "execute"
        else:
            logger.info("Router: No approved trades → hold")
            return "hold"

    # =========================================================================
    # EXECUTION ROUTES
    # =========================================================================

    @listen("execute")
    def execute_trades(self) -> dict[str, Any]:
        """
        Publish signals or execute trades directly, depending on mode.

        HF mode (signal_cache provided):
            Writes TradeSignal objects to the SignalCache with a TTL.
            The TickExecutor picks up signals and executes on market ticks.
            The slow analysis plane (this flow) never blocks on broker I/O.

        Direct mode (no signal_cache — default):
            Executes trades immediately.  Used for paper trading and backtesting.
        """
        if self._signal_cache is not None:
            return self._publish_signals()
        return self._execute_directly()

    def _publish_signals(self) -> dict[str, Any]:
        """HF mode: write signals to SignalCache for the TickExecutor."""
        logger.info("═══ PHASE 5: PUBLISHING SIGNALS → TICK EXECUTOR ═══")

        signals: list[TradeSignal] = []

        for decision in self.state.approved_trades:
            symbol = decision.get("symbol", "")
            action = decision.get("action", "hold").upper()
            confidence = float(decision.get("confidence", 0.0))

            # Map position_size string to fraction
            size_map = {"full": 0.20, "half": 0.10, "quarter": 0.05, "none": 0.0}
            position_size_pct = size_map.get(
                decision.get("position_size", "quarter"), 0.05
            )

            signal = TradeSignal.create(
                symbol=symbol,
                action=action,  # type: ignore[arg-type]
                confidence=confidence,
                position_size_pct=position_size_pct,
                stop_loss=decision.get("stop_loss"),
                take_profit=decision.get("take_profit"),
                expires_in_seconds=self._signal_ttl_seconds,
                session_id=self._session_id,
            )
            signals.append(signal)

            logger.info(
                f"[SIGNAL] Published {action} {symbol} "
                f"conf={confidence:.0%} ttl={self._signal_ttl_seconds}s"
            )

        self._signal_cache.update_batch(signals)
        self.state.signal_funnel["executed"] = len(signals)

        return {"signals_published": len(signals)}

    def _execute_directly(self) -> dict[str, Any]:
        """Direct mode: execute trades immediately (backward-compatible)."""
        logger.info("═══ PHASE 5: TRADE EXECUTION (DIRECT) ═══")

        executed = []
        # alpha_proxy: bps of expected alpha per unit confidence (tunable via env var)
        alpha_per_confidence = float(os.getenv("ALPHA_BPS_PER_UNIT_CONFIDENCE", "50"))

        # -----------------------------------------------------------------------
        # Portfolio optimization: compute MV-optimal weights for all approved
        # trades before the loop, replacing per-symbol bucket sizing.
        # Falls back to None → _calculate_quantity() path per symbol below.
        # -----------------------------------------------------------------------
        portfolio_target_weights: dict[str, float] = {}
        try:
            returns_df = self._get_returns_dataframe()
            snapshot = self._portfolio.get_snapshot()
            portfolio_equity = self.state.portfolio.get("equity", 100_000) or 100_000
            current_weights: dict[str, float] = {
                pos.symbol: pos.notional_value / portfolio_equity
                for pos in (snapshot.positions or [])
                if portfolio_equity > 0
            }
            portfolio_target_weights = self._portfolio_optimizer.optimize(
                decisions=self.state.approved_trades,
                returns_df=returns_df,
                current_weights=current_weights or None,
            )
            logger.info(
                f"[PortOpt] Target weights computed for {len(portfolio_target_weights)} symbols"
            )
        except Exception as _opt_err:
            logger.warning(
                f"[PortOpt] Optimization failed ({_opt_err}) — "
                "falling back to per-symbol bucket sizing"
            )

        for i, decision in enumerate(self.state.approved_trades):
            symbol = decision.get("symbol")
            action = decision.get("action")

            try:
                # Use portfolio-optimal weight when available, else legacy bucket
                if symbol in portfolio_target_weights:
                    quantity = self._calculate_quantity_from_weight(
                        symbol, portfolio_target_weights[symbol]
                    )
                else:
                    quantity = self._calculate_quantity(decision)

                if quantity > 0:
                    current_price = self.state.features.get(symbol, {}).get(
                        "close", 0.0
                    ) or self.state.market_data.get(symbol, {}).get("close", 0.0)
                    daily_volume = int(
                        self.state.market_data.get(symbol, {}).get("volume", 1_000_000)
                    )

                    # Risk gate check (hard stop — cannot be overridden by TCA or SOR)
                    verdict = self._risk_gate.check(
                        symbol=symbol,
                        side=action,
                        quantity=quantity,
                        current_price=current_price,
                        daily_volume=daily_volume,
                    )

                    if not verdict.approved:
                        logger.warning(
                            f"[RISK GATE] BLOCKED {action} {symbol}: {verdict.reason}"
                        )
                        rejection_event = make_trade_event(
                            session_id=self._session_id,
                            agent_name="RiskGate",
                            agent_role="risk_gate",
                            symbol=symbol,
                            action=action,
                            confidence=decision.get("confidence", 0.0),
                            reasoning=verdict.reason,
                            output_structured={
                                "violations": [v.__dict__ for v in verdict.violations]
                            },
                            risk_approved=False,
                            risk_violations=[v.description for v in verdict.violations],
                        )
                        self._audit_log.record(rejection_event)
                        self.state.errors.append(
                            f"RiskGate blocked {symbol}: {verdict.reason}"
                        )
                        continue

                    # Use approved_quantity (may be scaled down by risk gate)
                    final_qty = verdict.approved_quantity or quantity

                    # ----------------------------------------------------------
                    # TCA: record arrival price (signal-fire benchmark)
                    # ----------------------------------------------------------
                    trade_id = f"{symbol}_{self._session_id[:8]}_{i}"
                    tca_side = (
                        TCAOrderSide.BUY if action == "buy" else TCAOrderSide.SELL
                    )
                    self._tca.record_arrival(
                        trade_id=trade_id,
                        symbol=symbol,
                        side=tca_side,
                        shares=float(final_qty),
                        arrival_price=current_price,
                    )

                    # TCA: pre-trade cost forecast
                    # atr_pct from feature pipeline; default 1.5% daily vol
                    daily_vol_pct = float(
                        self.state.features.get(symbol, {}).get("atr_pct", 1.5)
                    )
                    forecast = self._tca.pre_trade(
                        trade_id=trade_id,
                        adv=float(daily_volume),
                        daily_vol_pct=daily_vol_pct,
                    )

                    # TCA: soft alpha-vs-cost check (cannot override risk gate)
                    alpha_proxy_bps = (
                        decision.get("confidence", 0.5) * alpha_per_confidence
                    )
                    should_trade, tca_reason = self._tca.alpha_vs_cost_check(
                        trade_id, alpha_proxy_bps
                    )
                    if not should_trade:
                        logger.warning(f"[TCA] Skipping {symbol}: {tca_reason}")
                        self.state.errors.append(f"TCA {symbol}: {tca_reason}")
                        continue

                    # Derive recommended execution algo from TCA forecast
                    exec_order_type = decision.get("entry_type", "market")
                    if forecast and forecast.recommended_algo.value == "LIMIT":
                        exec_order_type = "limit"
                    elif exec_order_type not in ("market", "limit"):
                        exec_order_type = "market"

                    # ----------------------------------------------------------
                    # Execution: SmartOrderRouter (if configured) else paper broker
                    # ----------------------------------------------------------
                    fill_price: float
                    filled_quantity: int
                    fill_slippage_bps: float
                    fill_commission: float
                    fill_order_id: str

                    if self._sor is not None:
                        unified_order = UnifiedOrder(
                            symbol=symbol,
                            side=action,
                            quantity=float(final_qty),
                            order_type=exec_order_type,
                            limit_price=decision.get("limit_price"),
                            client_order_id=trade_id,
                        )
                        try:
                            sor_result = self._sor.route(
                                account_id=os.getenv("BROKER_ACCOUNT_ID", "default"),
                                order=unified_order,
                                asset_class=decision.get("asset_class", "equity"),
                            )
                            if sor_result.status == "rejected":
                                logger.warning(
                                    f"[SOR] REJECTED {symbol}: {sor_result.reject_reason}"
                                )
                                self.state.errors.append(
                                    f"SOR rejected {symbol}: {sor_result.reject_reason}"
                                )
                                continue
                            fill_price = sor_result.avg_fill_price or current_price
                            filled_quantity = int(sor_result.filled_qty)
                            fill_commission = sor_result.commission or 0.0
                            fill_slippage_bps = (
                                abs(fill_price - current_price) / current_price * 10_000
                                if current_price > 0
                                else 0.0
                            )
                            fill_order_id = sor_result.order_id
                        except SmartOrderRouterError as sor_err:
                            # SOR exhausted all brokers — fall back to paper broker
                            logger.warning(
                                f"[SOR] Failed for {symbol}: {sor_err} — "
                                "falling back to paper broker"
                            )
                            fill = self._broker.execute(
                                OrderRequest(
                                    symbol=symbol,
                                    side=action,
                                    quantity=final_qty,
                                    order_type=exec_order_type,
                                    limit_price=decision.get("limit_price"),
                                    current_price=current_price,
                                    daily_volume=daily_volume,
                                )
                            )
                            if fill.rejected:
                                logger.warning(
                                    f"[BROKER] REJECTED {symbol}: {fill.reject_reason}"
                                )
                                self.state.errors.append(
                                    f"Broker rejected {symbol}: {fill.reject_reason}"
                                )
                                continue
                            fill_price = fill.fill_price
                            filled_quantity = fill.filled_quantity
                            fill_commission = fill.commission
                            fill_slippage_bps = fill.slippage_bps
                            fill_order_id = fill.order_id
                    else:
                        # Default: paper broker (original path, fully backward-compatible)
                        fill = self._broker.execute(
                            OrderRequest(
                                symbol=symbol,
                                side=action,
                                quantity=final_qty,
                                order_type=exec_order_type,
                                limit_price=decision.get("limit_price"),
                                current_price=current_price,
                                daily_volume=daily_volume,
                            )
                        )
                        if fill.rejected:
                            logger.warning(
                                f"[BROKER] REJECTED {symbol}: {fill.reject_reason}"
                            )
                            self.state.errors.append(
                                f"Broker rejected {symbol}: {fill.reject_reason}"
                            )
                            continue
                        fill_price = fill.fill_price
                        filled_quantity = fill.filled_quantity
                        fill_commission = fill.commission
                        fill_slippage_bps = fill.slippage_bps
                        fill_order_id = fill.order_id

                    # ----------------------------------------------------------
                    # TCA: record fill and compute post-trade shortfall
                    # ----------------------------------------------------------
                    self._tca.record_fill(trade_id=trade_id, fill_price=fill_price)

                    trade = {
                        "symbol": symbol,
                        "side": action,
                        "quantity": filled_quantity,
                        "fill_price": fill_price,
                        "slippage_bps": fill_slippage_bps,
                        "commission": fill_commission,
                        "reason": decision.get("reasoning", ""),
                        "confidence": decision.get("confidence", 0),
                        "stop_loss": decision.get("stop_loss"),
                        "take_profit": decision.get("take_profit"),
                        "date": str(self.state.current_date),
                        "order_id": fill_order_id,
                    }
                    executed.append(trade)

                    logger.info(
                        f"[EXECUTED] {action.upper()} {filled_quantity} {symbol} "
                        f"@ ${fill_price:.2f} "
                        f"({fill_slippage_bps:.1f} bps slippage, "
                        f"{decision.get('confidence', 0):.0%} confidence)"
                    )

            except Exception as e:
                logger.error(f"Execution failed for {symbol}: {e}")
                self.state.errors.append(f"Execution {symbol}: {str(e)}")

        self.state.executed_trades = executed
        self.state.signal_funnel["executed"] = len(executed)

        return {"executed": len(executed)}

    @listen("hold")
    def log_hold_decision(self) -> dict[str, Any]:
        """Log when no trades are executed."""
        logger.info("═══ NO TRADES - HOLDING ═══")

        self._log_agent(
            agent="FlowController",
            symbol=None,
            message="No trades executed - all positions held",
            role="system",
        )

        return {"held": True}

    # =========================================================================
    # FINALIZATION
    # =========================================================================

    @listen(execute_trades)
    def finalize_executed_day(self, execute_result: dict) -> dict[str, Any]:
        """Finalize day after execution."""
        return self._finalize_day()

    @listen(log_hold_decision)
    def finalize_hold_day(self, hold_result: dict) -> dict[str, Any]:
        """Finalize day after hold."""
        return self._finalize_day()

    def _finalize_day(self) -> dict[str, Any]:
        """Build final day result and run post-trade learning."""
        # Post-trade learning: run ExpectancyEngine on any newly closed trades
        self._run_post_trade_learning()

        # TCA session aggregate
        tca_report = self._tca.aggregate_report()
        if tca_report.get("n_trades", 0) > 0:
            logger.info(
                f"[TCA] Session execution quality: {tca_report.get('execution_quality')} | "
                f"avg IS={tca_report.get('avg_shortfall_vs_arrival_bps', 0):+.1f}bps | "
                f"favorable={tca_report.get('pct_favorable', 0):.0f}% | "
                f"total cost=${tca_report.get('total_dollar_cost', 0):+.2f}"
            )

        logger.info("")
        logger.info(
            "╔══════════════════════════════════════════════════════════════════╗"
        )
        logger.info(
            f"║  TRADING DAY COMPLETE: {self.state.current_date}                 ║"
        )
        logger.info(
            f"║  Decisions: {len(self.state.trade_decisions)}                    ║"
        )
        logger.info(
            f"║  Executed: {len(self.state.executed_trades)}                     ║"
        )
        logger.info(
            f"║  Errors: {len(self.state.errors)}                                ║"
        )
        logger.info(
            "╚══════════════════════════════════════════════════════════════════╝"
        )
        logger.info("")

        return {
            "trades": self.state.executed_trades,
            "agent_logs": self.state.agent_logs,
            "regimes": self.state.regimes,
            "signal_funnel": self.state.signal_funnel,
            "errors": self.state.errors,
            "session_id": self._session_id,
        }

    def _run_post_trade_learning(self) -> None:
        """Run ExpectancyEngine and SkillTracker updates after trade execution."""
        try:
            from quantstack.knowledge.store import KnowledgeStore
            from quantstack.learning.calibration import get_calibration_tracker
            from quantstack.learning.expectancy_engine import ExpectancyEngine
            from quantstack.learning.skill_tracker import SkillTracker

            store = KnowledgeStore()
            expectancy = ExpectancyEngine(store)
            skill_tracker = SkillTracker(store)
            calibration = get_calibration_tracker()

            result = expectancy.calculate_expectancy()
            if result.sample_size >= 5:
                logger.info(
                    f"[LEARNING] Expectancy update: "
                    f"win_rate={result.win_rate:.1%}, "
                    f"expectancy={result.expectancy:+.2f}, "
                    f"n={result.sample_size}"
                )

            # Update SkillTracker for each executed trade
            for trade in self.state.executed_trades:
                pnl = trade.get("pnl")
                confidence = trade.get("confidence", 0.5)
                if pnl is not None:
                    skill_tracker.update_agent_skill(
                        agent_id="SuperTrader",
                        prediction_correct=pnl > 0,
                        signal_pnl=pnl,
                    )
                    calibration.record(
                        agent_name="SuperTrader",
                        stated_confidence=confidence,
                        was_correct=pnl > 0,
                        symbol=trade.get("symbol"),
                        action=trade.get("side"),
                        pnl=pnl,
                    )

                # Check retraining trigger
                if skill_tracker.needs_retraining("SuperTrader"):
                    logger.warning(
                        "[LEARNING] SuperTrader win rate below threshold — consider retraining"
                    )

        except Exception as e:
            logger.warning(f"[LEARNING] Post-trade learning failed (non-fatal): {e}")

        # RL online feedback loop — push trade outcomes to OnlineRLTrainer
        # Reads pre-trade snapshots saved by RL tools to the module-level registry.
        # Runs after ExpectancyEngine/SkillTracker so it never blocks them.
        if self._rl_adapter is not None and self._rl_config is not None:
            try:
                from quantstack.rl.rl_tools import pop_pretrade_snapshot

                for trade in self.state.executed_trades:
                    # Try each tool type — one snapshot per tool per day
                    for tool_name in ("rl_position_size", "rl_execution_strategy"):
                        snapshot = pop_pretrade_snapshot(tool_name)
                        if snapshot:
                            self._rl_adapter.process_trade_outcome(trade, snapshot)
            except Exception as _rl_err:
                logger.warning(f"[RL] Online update failed (non-fatal): {_rl_err}")

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _log_agent(
        self,
        agent: str,
        symbol: str | None,
        message: str,
        role: str,
    ) -> None:
        """Add agent log entry."""
        self.state.agent_logs.append(
            {
                "date": str(self.state.current_date),
                "agent_name": agent,
                "symbol": symbol,
                "message": message,
                "role": role,
                "timestamp": datetime.now().isoformat(),
            }
        )

    def _get_historical_context(self, symbol: str) -> str:
        """Retrieve historical context from blackboard."""
        return read_blackboard_context(symbol, limit=10)

    def _store_decision(self, symbol: str, decision: dict) -> None:
        """Store trade decision to blackboard."""
        action = decision.get("action", "hold").upper()
        conf = decision.get("confidence", 0)
        reasoning = decision.get("reasoning", "")

        write_to_blackboard(
            agent="SuperTrader",
            symbol=symbol,
            message=f"""**DECISION:** {action}
**Confidence:** {conf:.0%}
**Reasoning:** {reasoning}""",
            sim_date=self.state.current_date,
        )

    def _get_returns_dataframe(self) -> pd.DataFrame | None:  # noqa: F821
        """Pull 90-day daily close returns from the quantcore DataStore.

        Returns None on any failure so the portfolio optimizer's fallback path
        (signal-proportional equal-weight) activates instead of crashing the flow.
        """
        try:
            from datetime import timedelta

            import pandas as pd
            from quantstack.config.timeframes import Timeframe
            from quantstack.data.storage import DataStore

            store = DataStore()
            symbols = self.state.symbols or []
            end_date = datetime.combine(
                self.state.current_date or date.today(), datetime.min.time()
            )
            start_date = end_date - timedelta(
                days=120
            )  # extra buffer for weekends/holidays
            returns_dict: dict[str, Any] = {}

            for sym in symbols:
                try:
                    ohlcv = store.load_ohlcv(
                        symbol=sym,
                        timeframe=Timeframe.D1,
                        start_date=start_date,
                        end_date=end_date,
                    )
                    if ohlcv is not None and len(ohlcv) >= 2:
                        returns_dict[sym] = ohlcv["close"].pct_change().dropna()
                except Exception:
                    pass  # Missing symbol data is non-fatal

            if not returns_dict:
                return None

            returns_df = pd.DataFrame(returns_dict).dropna(how="all")
            return returns_df if len(returns_df) >= 2 else None

        except Exception as _e:
            logger.debug(f"[PortOpt] Could not build returns DataFrame: {_e}")
            return None

    def _calculate_quantity_from_weight(self, symbol: str, weight: float) -> int:
        """Convert a portfolio weight fraction into an integer share count.

        Args:
            symbol: Ticker symbol.
            weight: Target portfolio weight as a fraction of NAV (0.15 = 15%).

        Returns:
            Number of shares to trade (0 if price unavailable or weight ≤ 0).
        """
        if weight <= 0.0:
            return 0

        equity = self.state.portfolio.get("equity", 100_000)
        price = self.state.features.get(symbol, {}).get(
            "close", 0.0
        ) or self.state.market_data.get(symbol, {}).get("close", 0.0)

        if price <= 0.0:
            return 0

        return int(equity * weight / price)

    def _calculate_quantity(self, decision: dict) -> int:
        """Calculate trade quantity based on decision.

        Legacy per-symbol sizing used when the portfolio optimizer is bypassed
        (e.g., single-symbol mode or optimizer failure).  Kept as a fallback
        so no caller breaks if target_weights is unavailable for a symbol.
        """
        symbol = decision.get("symbol")
        equity = self.state.portfolio.get("equity", 100_000)
        price = self.state.features.get(symbol, {}).get(
            "close", 0.0
        ) or self.state.market_data.get(symbol, {}).get("close", 0.0)

        if price <= 0:
            return 0

        size_map = {"full": 0.20, "half": 0.10, "quarter": 0.05, "none": 0.0}
        size_pct = size_map.get(decision.get("position_size", "quarter"), 0.05)

        return int(equity * size_pct / price)


# =============================================================================
# FLOW ADAPTER - Bridge to simulation engine
# =============================================================================


class TradingDayFlowAdapter:
    """Adapter to integrate TradingDayFlow with the simulation engine."""

    def __init__(
        self,
        config: Any | None = None,
        **kwargs,
    ):
        self.config = config
        self._engine = SignalEngine()
        logger.info("TradingDayFlowAdapter initialized (SignalEngine mode)")

    async def run_day(self, context: Any) -> dict[str, Any]:
        """Run the trading day using SignalEngine."""
        agent_logs: list[dict] = []
        trades: list[dict] = []

        for symbol in context.market_data.keys():
            try:
                regime = context.regimes.get(
                    symbol, {"trend": "unknown", "volatility": "normal"}
                )

                logger.info(f"[SignalEngine] Analyzing {symbol}...")
                brief = await self._engine.run(symbol, regime=regime)

                sym_brief = brief.symbol_briefs[0] if brief.symbol_briefs else None
                if sym_brief is None:
                    continue

                bias = sym_brief.consensus_bias
                confidence = sym_brief.consensus_conviction
                action = "hold"
                if bias in ("bullish", "strong_bullish") and confidence >= 0.6:
                    action = "buy"
                elif bias in ("bearish", "strong_bearish") and confidence >= 0.6:
                    action = "sell"

                reasoning = sym_brief.market_summary or brief.market_overview

                write_to_blackboard(
                    agent="SignalEngine",
                    symbol=symbol,
                    message=f"**DECISION:** {action.upper()}\n"
                    f"**Confidence:** {confidence:.0%}\n"
                    f"**Reasoning:** {reasoning}",
                    sim_date=context.date,
                )

                if action in ["buy", "sell"]:
                    decision_dict = {
                        "action": action,
                        "confidence": confidence,
                        "position_size": "quarter" if confidence < 0.75 else "half",
                    }
                    # Surface stop/take-profit from critical_levels if available
                    stop_loss = next(
                        (
                            lvl.price
                            for lvl in (sym_brief.critical_levels or [])
                            if getattr(lvl, "level_type", "") == "stop"
                        ),
                        None,
                    )
                    take_profit = next(
                        (
                            lvl.price
                            for lvl in (sym_brief.critical_levels or [])
                            if getattr(lvl, "level_type", "")
                            in ("target", "resistance")
                        ),
                        None,
                    )
                    trade = {
                        "symbol": symbol,
                        "action": action,
                        "quantity": self._calculate_quantity(
                            context, symbol, decision_dict
                        ),
                        "confidence": confidence,
                        "reasoning": reasoning,
                        "stop_loss": stop_loss,
                        "take_profit": take_profit,
                    }
                    trades.append(trade)

                agent_logs.append(
                    {
                        "agent_name": "SignalEngine",
                        "symbol": symbol,
                        "message": reasoning,
                        "role": "execution",
                    }
                )

            except Exception as e:
                logger.error(f"SignalEngine error for {symbol}: {e}")
                agent_logs.append(
                    {
                        "agent_name": "SignalEngine",
                        "symbol": symbol,
                        "message": f"Error: {str(e)}",
                        "role": "error",
                    }
                )

        return {
            "trades": trades,
            "agent_logs": agent_logs,
            "regimes": context.regimes,
            "signals": [],
            "signal_funnel": {
                "symbols_analyzed": len(context.market_data),
                "trades_generated": len(trades),
            },
        }

    def _calculate_quantity(
        self,
        context: Any,
        symbol: str,
        decision: dict[str, Any],
    ) -> int:
        portfolio = context.portfolio if hasattr(context, "portfolio") else {}
        equity = portfolio.get("equity", 100000)

        size_map = {"full": 0.20, "half": 0.10, "quarter": 0.05, "none": 0}
        position_pct = size_map.get(decision.get("position_size", "quarter"), 0.05)

        market_data = context.market_data.get(symbol, {})
        price = market_data.get("close", market_data.get("price", 100))

        if price <= 0:
            return 0

        max_position_value = equity * position_pct
        quantity = int(max_position_value / price)

        return max(1, quantity) if quantity > 0 else 0
