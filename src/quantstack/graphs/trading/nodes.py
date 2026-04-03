"""Node functions for the Trading Graph.

12 nodes + 2 conditional routers. Agent nodes use LLM reasoning;
tool nodes are deterministic data-gathering functions.
"""

import json
import logging
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from quantstack.core.risk.safety_gate import RiskDecision, SafetyGate
from quantstack.graphs.config import AgentConfig
from quantstack.graphs.state import TradingState

logger = logging.getLogger(__name__)


def _build_system_message(config: AgentConfig) -> SystemMessage:
    return SystemMessage(content=(
        f"You are a {config.role}.\n\n"
        f"Goal: {config.goal}\n\n"
        f"Background: {config.backstory}\n\n"
        "Always respond with valid JSON."
    ))


def make_safety_check(llm: BaseChatModel, config: AgentConfig):
    """Create the safety_check node (deterministic, no retry)."""

    async def safety_check(state: TradingState) -> dict[str, Any]:
        try:
            response = await llm.ainvoke([
                _build_system_message(config),
                HumanMessage(content=(
                    f"Cycle {state['cycle_number']}: Check system status. "
                    "Is the system halted or healthy? "
                    'Return JSON: {"halted": true/false, "reason": "..."}'
                )),
            ])
            try:
                parsed = json.loads(response.content)
            except (json.JSONDecodeError, TypeError):
                parsed = {"halted": False}
            halted = parsed.get("halted", False)
            update: dict[str, Any] = {
                "decisions": [{"node": "safety_check", "halted": halted}],
            }
            if halted:
                update["errors"] = [f"System halted: {parsed.get('reason', 'unknown')}"]
            return update
        except Exception as exc:
            logger.error("safety_check failed: %s", exc)
            return {
                "errors": [f"safety_check: {exc}"],
                "decisions": [{"node": "safety_check", "halted": True, "error": str(exc)}],
            }

    return safety_check


def make_daily_plan(llm: BaseChatModel, config: AgentConfig):
    """Create the daily_plan node (agent)."""

    async def daily_plan(state: TradingState) -> dict[str, Any]:
        try:
            response = await llm.ainvoke([
                _build_system_message(config),
                HumanMessage(content=(
                    f"Cycle {state['cycle_number']}, regime: {state.get('regime', 'unknown')}.\n"
                    f"Portfolio: {json.dumps(state.get('portfolio_context', {}), default=str)}\n\n"
                    "Generate a daily trading plan. Include regime assessment, "
                    "key levels, and trading priorities. "
                    'Return JSON: {"plan": "...", "priorities": [...]}'
                )),
            ])
            try:
                parsed = json.loads(response.content)
                plan = parsed.get("plan", response.content)
            except (json.JSONDecodeError, TypeError):
                plan = response.content
            return {
                "daily_plan": plan,
                "decisions": [{"node": "daily_plan", "action": "generated"}],
            }
        except Exception as exc:
            logger.error("daily_plan failed: %s", exc)
            return {
                "daily_plan": f"Plan generation failed: {exc}",
                "errors": [f"daily_plan: {exc}"],
            }

    return daily_plan


def make_position_review(llm: BaseChatModel, config: AgentConfig):
    """Create the position_review node (agent)."""

    async def position_review(state: TradingState) -> dict[str, Any]:
        try:
            positions = state.get("portfolio_context", {}).get("positions", [])
            response = await llm.ainvoke([
                _build_system_message(config),
                HumanMessage(content=(
                    f"Review these open positions:\n{json.dumps(positions, default=str)}\n\n"
                    "For each position, recommend HOLD, TRIM, or CLOSE with reasoning. "
                    'Return JSON: [{"symbol": "...", "action": "HOLD|TRIM|CLOSE", "reason": "..."}]'
                )),
            ])
            try:
                reviews = json.loads(response.content)
                if not isinstance(reviews, list):
                    reviews = [reviews] if reviews else []
            except (json.JSONDecodeError, TypeError):
                reviews = []
            return {
                "position_reviews": reviews,
                "decisions": [{"node": "position_review", "count": len(reviews)}],
            }
        except Exception as exc:
            logger.error("position_review failed: %s", exc)
            return {
                "position_reviews": [],
                "errors": [f"position_review: {exc}"],
            }

    return position_review


def make_execute_exits(llm: BaseChatModel, config: AgentConfig):
    """Create the execute_exits node (tool, deterministic)."""

    async def execute_exits(state: TradingState) -> dict[str, Any]:
        try:
            reviews = state.get("position_reviews", [])
            exits_needed = [r for r in reviews if r.get("action") in ("TRIM", "CLOSE")]
            if not exits_needed:
                return {
                    "exit_orders": [],
                    "decisions": [{"node": "execute_exits", "action": "no_exits"}],
                }
            response = await llm.ainvoke([
                _build_system_message(config),
                HumanMessage(content=(
                    f"Execute exits for these positions:\n{json.dumps(exits_needed, default=str)}\n\n"
                    "Submit exit orders and return confirmation. "
                    'Return JSON: [{"symbol": "...", "order_id": "...", "action": "..."}]'
                )),
            ])
            try:
                orders = json.loads(response.content)
                if not isinstance(orders, list):
                    orders = [orders] if orders else []
            except (json.JSONDecodeError, TypeError):
                orders = []
            return {
                "exit_orders": orders,
                "decisions": [{"node": "execute_exits", "count": len(orders)}],
            }
        except Exception as exc:
            logger.error("execute_exits failed: %s", exc)
            return {
                "exit_orders": [],
                "errors": [f"execute_exits: {exc}"],
            }

    return execute_exits


def make_entry_scan(llm: BaseChatModel, config: AgentConfig):
    """Create the entry_scan node (agent)."""

    async def entry_scan(state: TradingState) -> dict[str, Any]:
        try:
            response = await llm.ainvoke([
                _build_system_message(config),
                HumanMessage(content=(
                    f"Regime: {state.get('regime', 'unknown')}\n"
                    f"Daily plan: {state.get('daily_plan', 'none')}\n\n"
                    "Scan for entry candidates matching current regime and strategies. "
                    'Return JSON: [{"symbol": "...", "strategy": "...", "signal_strength": ...}]'
                )),
            ])
            try:
                candidates = json.loads(response.content)
                if not isinstance(candidates, list):
                    candidates = [candidates] if candidates else []
            except (json.JSONDecodeError, TypeError):
                candidates = []
            return {
                "entry_candidates": candidates,
                "decisions": [{"node": "entry_scan", "count": len(candidates)}],
            }
        except Exception as exc:
            logger.error("entry_scan failed: %s", exc)
            return {
                "entry_candidates": [],
                "errors": [f"entry_scan: {exc}"],
            }

    return entry_scan


async def merge_parallel(state: TradingState) -> dict[str, Any]:
    """No-op join node. Convergence point for parallel branches."""
    return {}


def make_risk_sizing(llm: BaseChatModel, config: AgentConfig):
    """Create the risk_sizing node (tool+agent hybrid).

    Computes position sizes via LLM reasoning, then validates each
    candidate through SafetyGate. SafetyGate is pure Python — no LLM.
    """

    async def risk_sizing(state: TradingState) -> dict[str, Any]:
        candidates = state.get("entry_candidates", [])
        if not candidates:
            return {
                "risk_verdicts": [],
                "decisions": [{"node": "risk_sizing", "action": "no_candidates"}],
            }

        portfolio_ctx = state.get("portfolio_context", {})
        gate = SafetyGate()
        verdicts = []

        try:
            response = await llm.ainvoke([
                _build_system_message(config),
                HumanMessage(content=(
                    f"Size these entry candidates:\n{json.dumps(candidates, default=str)}\n"
                    f"Portfolio context:\n{json.dumps(portfolio_ctx, default=str)}\n\n"
                    "For each candidate, compute position size using Kelly criterion. "
                    'Return JSON: [{"symbol": "...", "recommended_size_pct": ..., '
                    '"reasoning": "...", "confidence": ...}]'
                )),
            ])
            try:
                sizing_results = json.loads(response.content)
                if not isinstance(sizing_results, list):
                    sizing_results = [sizing_results] if sizing_results else []
            except (json.JSONDecodeError, TypeError):
                sizing_results = []

            errors = []
            for sizing in sizing_results:
                decision = RiskDecision(
                    symbol=sizing.get("symbol", "UNKNOWN"),
                    recommended_size_pct=sizing.get("recommended_size_pct", 0),
                    reasoning=sizing.get("reasoning", ""),
                    confidence=sizing.get("confidence", 0),
                )
                verdict = gate.validate(decision, portfolio_ctx)
                verdict_dict = {
                    "symbol": decision.symbol,
                    "approved": verdict.approved,
                    "size_pct": decision.recommended_size_pct,
                }
                if not verdict.approved:
                    verdict_dict["violations"] = verdict.violations
                    errors.append(
                        f"Risk gate rejected {decision.symbol}: "
                        f"{', '.join(verdict.violations)}"
                    )
                verdicts.append(verdict_dict)

            update: dict[str, Any] = {
                "risk_verdicts": verdicts,
                "decisions": [{"node": "risk_sizing", "verdicts": len(verdicts)}],
            }
            if errors:
                update["errors"] = errors
            return update

        except Exception as exc:
            logger.error("risk_sizing failed: %s", exc)
            return {
                "risk_verdicts": [],
                "errors": [f"risk_sizing: {exc}"],
            }

    return risk_sizing


def make_portfolio_review(llm: BaseChatModel, config: AgentConfig):
    """Create the portfolio_review node (agent: fund_manager)."""

    async def portfolio_review(state: TradingState) -> dict[str, Any]:
        try:
            verdicts = state.get("risk_verdicts", [])
            approved = [v for v in verdicts if v.get("approved", False)]
            response = await llm.ainvoke([
                _build_system_message(config),
                HumanMessage(content=(
                    f"Review these risk-approved candidates:\n{json.dumps(approved, default=str)}\n\n"
                    "Assess portfolio-level risk: correlation, allocation, diversity. "
                    'Return JSON: [{"symbol": "...", "decision": "APPROVED|REJECTED", "reason": "..."}]'
                )),
            ])
            try:
                decisions = json.loads(response.content)
                if not isinstance(decisions, list):
                    decisions = [decisions] if decisions else []
            except (json.JSONDecodeError, TypeError):
                decisions = []
            return {
                "fund_manager_decisions": decisions,
                "decisions": [{"node": "portfolio_review", "count": len(decisions)}],
            }
        except Exception as exc:
            logger.error("portfolio_review failed: %s", exc)
            return {
                "fund_manager_decisions": [],
                "errors": [f"portfolio_review: {exc}"],
            }

    return portfolio_review


def make_options_analysis(llm: BaseChatModel, config: AgentConfig):
    """Create the options_analysis node (agent)."""

    async def options_analysis(state: TradingState) -> dict[str, Any]:
        try:
            fm_decisions = state.get("fund_manager_decisions", [])
            approved = [d for d in fm_decisions if d.get("decision") == "APPROVED"]
            if not approved:
                return {
                    "options_analysis": [],
                    "decisions": [{"node": "options_analysis", "action": "no_candidates"}],
                }
            response = await llm.ainvoke([
                _build_system_message(config),
                HumanMessage(content=(
                    f"Analyze options structures for:\n{json.dumps(approved, default=str)}\n\n"
                    "For eligible candidates, select optimal options structures. "
                    'Return JSON: [{"symbol": "...", "structure": "...", "params": {...}}]'
                )),
            ])
            try:
                analysis = json.loads(response.content)
                if not isinstance(analysis, list):
                    analysis = [analysis] if analysis else []
            except (json.JSONDecodeError, TypeError):
                analysis = []
            return {
                "options_analysis": analysis,
                "decisions": [{"node": "options_analysis", "count": len(analysis)}],
            }
        except Exception as exc:
            logger.error("options_analysis failed: %s", exc)
            return {
                "options_analysis": [],
                "errors": [f"options_analysis: {exc}"],
            }

    return options_analysis


def make_execute_entries(llm: BaseChatModel, config: AgentConfig):
    """Create the execute_entries node (tool, deterministic)."""

    async def execute_entries(state: TradingState) -> dict[str, Any]:
        try:
            fm_decisions = state.get("fund_manager_decisions", [])
            approved = [d for d in fm_decisions if d.get("decision") == "APPROVED"]
            if not approved:
                return {
                    "entry_orders": [],
                    "decisions": [{"node": "execute_entries", "action": "no_approved"}],
                }
            response = await llm.ainvoke([
                _build_system_message(config),
                HumanMessage(content=(
                    f"Execute entries for approved candidates:\n{json.dumps(approved, default=str)}\n"
                    f"Options analysis:\n{json.dumps(state.get('options_analysis', []), default=str)}\n\n"
                    "Submit entry orders and return confirmation. "
                    'Return JSON: [{"symbol": "...", "order_id": "...", "type": "..."}]'
                )),
            ])
            try:
                orders = json.loads(response.content)
                if not isinstance(orders, list):
                    orders = [orders] if orders else []
            except (json.JSONDecodeError, TypeError):
                orders = []
            return {
                "entry_orders": orders,
                "decisions": [{"node": "execute_entries", "count": len(orders)}],
            }
        except Exception as exc:
            logger.error("execute_entries failed: %s", exc)
            return {
                "entry_orders": [],
                "errors": [f"execute_entries: {exc}"],
            }

    return execute_entries


def make_reflection(llm: BaseChatModel, config: AgentConfig):
    """Create the reflection node (agent)."""

    async def reflection(state: TradingState) -> dict[str, Any]:
        try:
            response = await llm.ainvoke([
                _build_system_message(config),
                HumanMessage(content=(
                    f"Reflect on this trading cycle:\n"
                    f"Exits: {json.dumps(state.get('exit_orders', []), default=str)}\n"
                    f"Entries: {json.dumps(state.get('entry_orders', []), default=str)}\n\n"
                    "Analyze what happened and extract lessons. "
                    'Return JSON: {"reflection": "...", "lessons": [...]}'
                )),
            ])
            try:
                parsed = json.loads(response.content)
                text = parsed.get("reflection", response.content)
            except (json.JSONDecodeError, TypeError):
                text = response.content
            return {
                "reflection": text,
                "decisions": [{"node": "reflection", "action": "completed"}],
            }
        except Exception as exc:
            logger.error("reflection failed: %s", exc)
            return {
                "reflection": f"Reflection failed: {exc}",
                "errors": [f"reflection: {exc}"],
            }

    return reflection
