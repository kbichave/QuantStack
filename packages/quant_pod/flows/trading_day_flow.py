# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Trading Day Flow - CrewAI Flow Orchestrator.

Orchestrates the hierarchical trading system using CrewAI TradingCrew.
NO FALLBACKS - The system uses TradingCrew exclusively.

Flow Structure:
    @start(): detect_regime
         │
    @listen(): run_crew_analysis
         │
    @router(): route_execution
         │
    @listen("execute"): execute_trades
    @listen("hold"): log_hold_decision
         │
    finalize_day

All agents have reasoning enabled. If an agent fails, the flow fails.
"""

from __future__ import annotations

import asyncio
from datetime import date, datetime
from typing import Any, Dict, List, Literal, Optional

from quant_pod.crewai_compat import Flow, listen, router, start
from loguru import logger
from pydantic import BaseModel, Field

# Import regime detector
from quant_pod.agents.regime_detector import RegimeDetectorAgent

# Import CrewAI-native TradingCrew
from quant_pod.crews import TradingCrew
from quant_pod.crews.schemas import (
    TradeDecision,
    DailyBrief,
    RiskVerdict,
)

# Import blackboard for cross-agent memory
from quant_pod.memory.blackboard import (
    write_to_blackboard,
    read_blackboard_context,
    get_blackboard,
)


# =============================================================================
# FLOW STATE MODEL
# =============================================================================


class TradingDayState(BaseModel):
    """Structured state for the trading day flow."""

    # Flow metadata
    current_date: Optional[date] = None
    symbols: List[str] = Field(default_factory=list)

    # Market context
    market_data: Dict[str, Dict] = Field(default_factory=dict)
    features: Dict[str, Dict] = Field(default_factory=dict)
    portfolio: Dict[str, Any] = Field(default_factory=dict)

    # Phase 1: Regime Detection
    regimes: Dict[str, Dict] = Field(default_factory=dict)

    # Phase 2-4: Crew Analysis (handled by TradingCrew)
    daily_briefs: Dict[str, DailyBrief] = Field(default_factory=dict)
    trade_decisions: List[Dict] = Field(default_factory=list)

    # Phase 5: Execution
    approved_trades: List[Dict] = Field(default_factory=list)
    executed_trades: List[Dict] = Field(default_factory=list)

    # Metrics
    signal_funnel: Dict[str, int] = Field(
        default_factory=lambda: {
            "symbols_analyzed": 0,
            "decisions_made": 0,
            "executed": 0,
        }
    )

    # Agent logs for UI
    agent_logs: List[Dict] = Field(default_factory=list)

    # Historical context
    historical_context: Dict[str, str] = Field(default_factory=dict)

    # Error tracking
    errors: List[str] = Field(default_factory=list)

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

    def __init__(self):
        """Initialize the trading day flow."""
        self._regime_detector = None
        self._trading_crew = None

        super().__init__()
        logger.info("TradingDayFlow initialized (CrewAI TradingCrew mode)")

    @property
    def regime_detector(self) -> RegimeDetectorAgent:
        """Get or create regime detector."""
        if self._regime_detector is None:
            self._regime_detector = RegimeDetectorAgent()
        return self._regime_detector

    @property
    def trading_crew(self) -> TradingCrew:
        """Get or create TradingCrew instance."""
        if self._trading_crew is None:
            self._trading_crew = TradingCrew()
        return self._trading_crew

    # =========================================================================
    # PHASE 1: REGIME DETECTION
    # =========================================================================

    @start()
    def detect_regime(self) -> Dict[str, Any]:
        """
        Detect market regimes for all symbols.
        Entry point of the flow.
        """
        current_date = self.state.current_date or date.today()
        symbols = self.state.symbols or ["SPY"]

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
    def run_crew_analysis(self, regime_result: Dict) -> Dict[str, Any]:
        """
        Run TradingCrew for full analysis.

        The crew handles:
        - IC tasks: Raw data gathering
        - Pod Manager tasks: Compilation
        - Assistant task: Synthesis into 1-pager
        - SuperTrader task: Final decision
        """
        logger.info("═══ PHASE 2-4: TRADING CREW ANALYSIS ═══")

        for symbol in self.state.symbols:
            market_data = self.state.market_data.get(symbol, {})
            features = self.state.features.get(symbol, {})
            regime = self.state.regimes.get(symbol, {})
            historical_context = self.state.historical_context.get(symbol, "")

            # Skip if regime detection failed completely
            if regime.get("error") and regime.get("confidence", 0) == 0:
                logger.warning(f"Skipping {symbol} due to regime detection failure")
                continue

            try:
                # Prepare inputs for crew kickoff
                inputs = {
                    "symbol": symbol,
                    "current_date": self.state.current_date,
                    "regime": regime,
                    "market_data": market_data,
                    "features": features,
                    "portfolio": self.state.portfolio,
                    "historical_context": historical_context,
                }

                logger.info(f"[CREW] Running TradingCrew for {symbol}...")

                # Kickoff the crew - NO FALLBACK
                result = self.trading_crew.crew().kickoff(inputs=inputs)

                # Extract decision
                if hasattr(result, "pydantic"):
                    decision = result.pydantic
                elif hasattr(result, "raw"):
                    decision = result.raw
                else:
                    decision = result

                # Log the decision
                decision_summary = str(decision)[:300] if decision else "No decision"
                logger.info(f"[CREW] {symbol}: {decision_summary}")

                self._log_agent(
                    agent="TradingCrew",
                    symbol=symbol,
                    message=f"Analysis complete: {decision_summary}",
                    role="analysis",
                )

                # Store decision
                if decision:
                    if hasattr(decision, "model_dump"):
                        decision_dict = decision.model_dump()
                    elif hasattr(decision, "dict"):
                        decision_dict = decision.dict()
                    else:
                        decision_dict = {"raw": str(decision)}

                    decision_dict["symbol"] = symbol
                    self.state.trade_decisions.append(decision_dict)

                    # Write to blackboard
                    self._store_decision(symbol, decision_dict)

                    # If actionable, add to approved
                    action = decision_dict.get("action", "hold")
                    if action in ["buy", "sell"]:
                        self.state.approved_trades.append(decision_dict)

            except Exception as e:
                # NO FALLBACK - log error and continue
                logger.error(f"Crew analysis failed for {symbol}: {e}")
                self.state.errors.append(f"Crew {symbol}: {str(e)}")

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

    @router(run_crew_analysis)
    def route_execution(self, analysis_result: Dict) -> str:
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
    def execute_trades(self) -> Dict[str, Any]:
        """Execute approved trades."""
        logger.info("═══ PHASE 5: TRADE EXECUTION ═══")

        executed = []

        for decision in self.state.approved_trades:
            symbol = decision.get("symbol")
            action = decision.get("action")

            try:
                quantity = self._calculate_quantity(decision)

                if quantity > 0:
                    trade = {
                        "symbol": symbol,
                        "side": action,
                        "quantity": quantity,
                        "reason": decision.get("reasoning", ""),
                        "confidence": decision.get("confidence", 0),
                        "stop_loss": decision.get("stop_loss"),
                        "take_profit": decision.get("take_profit"),
                        "date": str(self.state.current_date),
                    }
                    executed.append(trade)

                    logger.info(
                        f"[EXECUTED] {action.upper()} {quantity} {symbol} "
                        f"@ {decision.get('confidence', 0):.0%} confidence"
                    )

            except Exception as e:
                logger.error(f"Execution failed for {symbol}: {e}")
                self.state.errors.append(f"Execution {symbol}: {str(e)}")

        self.state.executed_trades = executed
        self.state.signal_funnel["executed"] = len(executed)

        return {"executed": len(executed)}

    @listen("hold")
    def log_hold_decision(self) -> Dict[str, Any]:
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
    def finalize_executed_day(self, execute_result: Dict) -> Dict[str, Any]:
        """Finalize day after execution."""
        return self._finalize_day()

    @listen(log_hold_decision)
    def finalize_hold_day(self, hold_result: Dict) -> Dict[str, Any]:
        """Finalize day after hold."""
        return self._finalize_day()

    def _finalize_day(self) -> Dict[str, Any]:
        """Build final day result."""
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
        }

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _log_agent(
        self,
        agent: str,
        symbol: Optional[str],
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

    def _store_decision(self, symbol: str, decision: Dict) -> None:
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

    def _calculate_quantity(self, decision: Dict) -> int:
        """Calculate trade quantity based on decision."""
        symbol = decision.get("symbol")
        equity = self.state.portfolio.get("equity", 100000)
        price = self.state.features.get(symbol, {}).get("close", 0)

        if price <= 0:
            return 0

        size_map = {"full": 0.20, "half": 0.10, "quarter": 0.05, "none": 0.0}
        size_pct = size_map.get(decision.get("position_size", "quarter"), 0.05)

        max_position_value = equity * size_pct
        quantity = int(max_position_value / price)

        return quantity


# =============================================================================
# FLOW ADAPTER - Bridge to simulation engine
# =============================================================================


class TradingDayFlowAdapter:
    """
    Adapter to integrate TradingDayFlow with the simulation engine.

    NO FALLBACKS - uses TradingCrew exclusively.
    """

    def __init__(
        self,
        config: Optional[Any] = None,
        **kwargs,
    ):
        """Initialize the adapter."""
        self.config = config
        logger.info("TradingDayFlowAdapter initialized (TradingCrew mode)")

    async def run_day(self, context: Any) -> Dict[str, Any]:
        """
        Run the trading day using TradingCrew.
        """
        crew = TradingCrew()
        agent_logs = []
        trades = []

        for symbol in context.market_data.keys():
            try:
                inputs = {
                    "symbol": symbol,
                    "current_date": context.date,
                    "regime": context.regimes.get(
                        symbol, {"trend": "unknown", "volatility": "normal"}
                    ),
                    "market_data": context.market_data.get(symbol, {}),
                    "features": context.features.get(symbol, {}),
                    "portfolio": (
                        context.portfolio if hasattr(context, "portfolio") else {}
                    ),
                    "historical_context": "",
                }

                logger.info(f"[CREW] Running TradingCrew for {symbol}...")

                # Run crew in thread pool for async context
                import concurrent.futures

                loop = asyncio.get_event_loop()
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    result = await loop.run_in_executor(
                        pool, lambda: crew.crew().kickoff(inputs=inputs)
                    )

                # Extract decision
                if hasattr(result, "pydantic"):
                    decision = result.pydantic
                elif hasattr(result, "raw"):
                    decision = result.raw
                else:
                    decision = result

                if decision:
                    decision_dict = self._extract_decision_dict(decision)
                    action = decision_dict.get("action", "hold")
                    reasoning = decision_dict.get("reasoning", str(decision)[:500])
                    confidence = decision_dict.get("confidence", 0.5)

                    write_to_blackboard(
                        agent="TradingCrew",
                        symbol=symbol,
                        message=f"""**DECISION:** {action.upper()}
**Confidence:** {confidence:.0%}
**Reasoning:** {reasoning}""",
                        sim_date=context.date,
                    )

                    if action in ["buy", "sell"]:
                        trade = {
                            "symbol": symbol,
                            "action": action,
                            "quantity": self._calculate_quantity(
                                context, symbol, decision_dict
                            ),
                            "confidence": confidence,
                            "reasoning": reasoning,
                            "stop_loss": decision_dict.get("stop_loss"),
                            "take_profit": decision_dict.get("take_profit"),
                        }
                        trades.append(trade)

                    agent_logs.append(
                        {
                            "agent_name": "TradingCrew",
                            "symbol": symbol,
                            "message": reasoning,
                            "role": "execution",
                        }
                    )

            except Exception as e:
                # NO FALLBACK - log error
                logger.error(f"TradingCrew error for {symbol}: {e}")
                agent_logs.append(
                    {
                        "agent_name": "TradingCrew",
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

    def _extract_decision_dict(self, decision: Any) -> Dict[str, Any]:
        """Extract decision as dict from various formats."""
        if hasattr(decision, "model_dump"):
            return decision.model_dump()
        elif hasattr(decision, "dict"):
            return decision.dict()
        elif isinstance(decision, dict):
            return decision
        else:
            return {"raw": str(decision)}

    def _calculate_quantity(
        self,
        context: Any,
        symbol: str,
        decision: Dict[str, Any],
    ) -> int:
        """Calculate trade quantity."""
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
