"""Node functions for the Trading Graph.

12 nodes + 2 conditional routers. Agent nodes use LLM reasoning with
tool access; tool nodes are deterministic data-gathering functions.

Each agent node uses the shared run_agent() executor which handles the
tool-calling loop: LLM decides to call tools → tools execute →
results fed back → LLM produces final answer.
"""

import json
import logging
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool

from quantstack.core.risk.safety_gate import RiskDecision, SafetyGate
from quantstack.graphs.agent_executor import parse_json_response, run_agent
from quantstack.graphs.config import AgentConfig
from quantstack.graphs.state import TradingState
from quantstack.runners import is_market_hours

logger = logging.getLogger(__name__)


def make_market_intel(llm: BaseChatModel, config: AgentConfig, tools: list[BaseTool] | None = None):
    """Create the market_intel node (agent with tool access).

    Runs pre-market (8:30-9:30 AM ET) to gather macro news, analyst actions,
    and position-specific alerts. Also triggered by MARKET_MOVE events
    (>2% intraday index move). Outside the pre-market window on normal days,
    returns empty context (no-op).
    """
    tools = tools or []

    async def market_intel(state: TradingState) -> dict[str, Any]:
        import zoneinfo
        from datetime import datetime

        try:
            et = zoneinfo.ZoneInfo("America/New_York")
            now_et = datetime.now(et)
            hour, minute = now_et.hour, now_et.minute
            current_minutes = hour * 60 + minute

            # Pre-market window: 8:30-9:30 AM ET
            pre_market_start = 8 * 60 + 30
            pre_market_end = 9 * 60 + 30

            # Check for event bus trigger (MARKET_MOVE with magnitude > 2%)
            event_triggered = False
            portfolio_ctx = state.get("portfolio_context", {})
            if portfolio_ctx.get("market_move_trigger"):
                event_triggered = True

            in_window = pre_market_start <= current_minutes <= pre_market_end
            if not in_window and not event_triggered:
                return {"market_context": {}}

            mode = "event_triggered" if event_triggered else "morning_briefing"
            prompt = (
                f"Mode: {mode}\n"
                f"Cycle {state['cycle_number']}, regime: {state.get('regime', 'unknown')}.\n"
                f"Portfolio: {json.dumps(portfolio_ctx, default=str)}\n\n"
                "Gather pre-market intelligence:\n"
                "1. Search for major overnight macro news, Fed announcements, geopolitical events\n"
                "2. Check sector news for held positions\n"
                "3. Identify any analyst rating changes or earnings surprises\n"
                "4. Assess market sentiment and key risk factors\n\n"
                "Return structured JSON:\n"
                '{"headlines": [...], "risk_alerts": [...], "event_calendar": [...], '
                '"sector_news": {...}, "sentiment": "bullish|neutral|bearish"}'
            )
            text = await run_agent(llm, tools, config, prompt)
            parsed = parse_json_response(text, {})
            return {
                "market_context": parsed,
                "decisions": [{"node": "market_intel", "mode": mode}],
            }
        except Exception as exc:
            logger.error("market_intel failed: %s", exc)
            return {
                "market_context": {},
                "errors": [f"market_intel: {exc}"],
            }

    return market_intel


def make_earnings_analysis(llm: BaseChatModel, config: AgentConfig, tools: list[BaseTool] | None = None):
    """Create the earnings_analysis node (agent with tool access).

    Triggered when plan_day detects positions or watchlist symbols with
    earnings within 14 days. Analyzes historical beat rate, estimate
    revisions, IV premium ratio, and recommends hold/exit/hedge.
    """
    tools = tools or []

    async def earnings_analysis(state: TradingState) -> dict[str, Any]:
        earnings_symbols = state.get("earnings_symbols", [])
        if not earnings_symbols:
            return {"earnings_analysis": {}}

        try:
            prompt = (
                f"Symbols with earnings within 14 days: {earnings_symbols}\n\n"
                "For each symbol, analyze:\n"
                "1. Historical beat rate from last 4-8 quarters\n"
                "2. Analyst estimate revision direction (up/down/flat)\n"
                "3. IV premium ratio (implied move / expected move from history)\n"
                "4. Current position status (if held)\n\n"
                "For held positions: recommend HOLD_THROUGH, EXIT_BEFORE, or HEDGE_WITH_OPTIONS\n"
                "For entry candidates: assess whether earnings creates opportunity or risk\n\n"
                "Return JSON:\n"
                '{"analyses": [{"symbol": "...", "beat_rate": ..., '
                '"iv_premium_ratio": ..., "recommendation": "...", '
                '"options_suggestion": "...", "reasoning": "..."}]}'
            )
            text = await run_agent(llm, tools, config, prompt)
            parsed = parse_json_response(text, {"analyses": []})
            return {
                "earnings_analysis": parsed,
                "decisions": [{"node": "earnings_analysis", "count": len(earnings_symbols)}],
            }
        except Exception as exc:
            logger.error("earnings_analysis failed: %s", exc)
            return {
                "earnings_analysis": {},
                "errors": [f"earnings_analysis: {exc}"],
            }

    return earnings_analysis


def make_data_refresh():
    """Create the data_refresh node (deterministic, no LLM).

    Runs scheduled_refresh.run_intraday_refresh() every cycle during market
    hours.  Outside market hours this is a no-op (the trading graph is paused
    anyway, but we guard defensively).
    """

    async def data_refresh(state: TradingState) -> dict[str, Any]:
        if not is_market_hours():
            return {
                "data_refresh_summary": {"skipped": True, "reason": "outside_market_hours"},
            }

        try:
            from quantstack.data.scheduled_refresh import run_intraday_refresh

            report = await run_intraday_refresh()
            summary = {
                "mode": report.mode,
                "symbols_refreshed": report.symbols_refreshed,
                "api_calls": report.api_calls,
                "errors": report.errors,
                "elapsed_seconds": round(report.elapsed_seconds, 1),
            }
            if report.errors:
                return {
                    "data_refresh_summary": summary,
                    "errors": [f"data_refresh: {len(report.errors)} errors"],
                }
            return {"data_refresh_summary": summary}
        except Exception as exc:
            logger.error("data_refresh failed: %s", exc)
            return {
                "data_refresh_summary": {"error": str(exc)},
                "errors": [f"data_refresh: {exc}"],
            }

    return data_refresh


def make_safety_check(llm: BaseChatModel, config: AgentConfig):
    """Create the safety_check node (deterministic, no retry, no tools)."""

    async def safety_check(state: TradingState) -> dict[str, Any]:
        try:
            response = await llm.ainvoke([
                SystemMessage(content=(
                    f"You are a {config.role}.\nGoal: {config.goal}\n"
                    "Always respond with valid JSON."
                )),
                HumanMessage(content=(
                    f"Cycle {state['cycle_number']}: Check system status. "
                    "Is the system halted or healthy? "
                    'Return JSON: {"halted": true/false, "reason": "..."}'
                )),
            ])
            parsed = parse_json_response(response.content, {"halted": False})
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


def make_daily_plan(llm: BaseChatModel, config: AgentConfig, tools: list[BaseTool] | None = None):
    """Create the daily_plan node (agent with tool access)."""
    tools = tools or []

    async def daily_plan(state: TradingState) -> dict[str, Any]:
        try:
            # Include market intelligence if available
            market_intel_section = ""
            market_ctx = state.get("market_context", {})
            if market_ctx:
                market_intel_section = (
                    f"\n--- Market Intelligence Briefing ---\n"
                    f"{json.dumps(market_ctx, default=str)}\n"
                    f"--- End Briefing ---\n\n"
                )

            prompt = (
                f"Cycle {state['cycle_number']}, regime: {state.get('regime', 'unknown')}.\n"
                f"Portfolio: {json.dumps(state.get('portfolio_context', {}), default=str)}\n"
                f"{market_intel_section}"
                "First, use your tools to gather real data:\n"
                "1. Fetch the current portfolio state\n"
                "2. Get signal briefs for key symbols in the portfolio and watchlist\n"
                "3. Search the knowledge base for recent lessons\n\n"
                "Then generate a daily trading plan with:\n"
                "- Regime assessment based on actual data\n"
                "- Ranked entry candidates with real signal scores\n"
                "- Exit recommendations for existing positions\n"
                "- Key events calendar\n"
                "- Flag any symbols with earnings within 14 days\n\n"
                'Return final JSON: {"plan": "...", "priorities": [...], '
                '"entry_candidates": [...], "exit_recommendations": [...], '
                '"earnings_within_14d": ["SYMBOL", ...]}'
            )
            text = await run_agent(llm, tools, config, prompt)
            parsed = parse_json_response(text, {"plan": text})

            # Extract earnings symbols from the plan
            earnings_symbols = parsed.get("earnings_within_14d", [])
            if not isinstance(earnings_symbols, list):
                earnings_symbols = []

            update: dict[str, Any] = {
                "daily_plan": parsed.get("plan", text),
                "earnings_symbols": earnings_symbols,
                "decisions": [{"node": "daily_plan", "action": "generated"}],
            }
            return update
        except Exception as exc:
            logger.error("daily_plan failed: %s", exc)
            return {
                "daily_plan": f"Plan generation failed: {exc}",
                "errors": [f"daily_plan: {exc}"],
            }

    return daily_plan


def make_position_review(llm: BaseChatModel, config: AgentConfig, tools: list[BaseTool] | None = None):
    """Create the position_review node (agent with tool access)."""
    tools = tools or []

    async def position_review(state: TradingState) -> dict[str, Any]:
        try:
            prompt = (
                "Review all open positions.\n\n"
                "First, use your tools to:\n"
                "1. Fetch the current portfolio with all positions\n"
                "2. Get signal briefs for each position's symbol\n"
                "3. Search knowledge base for lessons on these symbols\n\n"
                "Then for each position, recommend HOLD, TRIM, TIGHTEN, or CLOSE:\n"
                "- Hard exits: options DTE <= 2, loss > 2x stop distance\n"
                "- Hold: thesis intact, regime unchanged\n"
                "- Tighten: profitable > 1x ATR, regime weakening\n"
                "- Close: regime flipped, target reached 75%+, holding period exceeded\n\n"
                'Return JSON: [{"symbol": "...", "action": "HOLD|TRIM|TIGHTEN|CLOSE", '
                '"reason": "...", "urgency": "low|medium|high"}]'
            )
            text = await run_agent(llm, tools, config, prompt)
            reviews = parse_json_response(text, [])
            if not isinstance(reviews, list):
                reviews = [reviews] if reviews else []
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


def make_execute_exits(llm: BaseChatModel, config: AgentConfig, tools: list[BaseTool] | None = None):
    """Create the execute_exits node (tool, deterministic)."""
    tools = tools or []

    async def execute_exits(state: TradingState) -> dict[str, Any]:
        try:
            reviews = state.get("position_reviews", [])
            exits_needed = [r for r in reviews if r.get("action") in ("TRIM", "CLOSE")]
            if not exits_needed:
                return {
                    "exit_orders": [],
                    "decisions": [{"node": "execute_exits", "action": "no_exits"}],
                }
            prompt = (
                f"Execute exits for these positions:\n{json.dumps(exits_needed, default=str)}\n\n"
                "Use the execute_order tool for each exit. "
                'Return JSON: [{"symbol": "...", "order_id": "...", "action": "..."}]'
            )
            text = await run_agent(llm, tools, config, prompt)
            orders = parse_json_response(text, [])
            if not isinstance(orders, list):
                orders = [orders] if orders else []
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


def make_entry_scan(llm: BaseChatModel, config: AgentConfig, tools: list[BaseTool] | None = None):
    """Create the entry_scan node (agent with tool access)."""
    tools = tools or []

    async def entry_scan(state: TradingState) -> dict[str, Any]:
        try:
            prompt = (
                f"Regime: {state.get('regime', 'unknown')}\n"
                f"Daily plan: {state.get('daily_plan', 'none')}\n\n"
                "Use your tools to find entry candidates:\n"
                "1. Get multi-symbol signal briefs for potential candidates\n"
                "2. Fetch fundamentals for promising symbols\n"
                "3. Search knowledge base for lessons on candidate strategies\n\n"
                "Run a structured bull/bear debate for each candidate.\n"
                'Return JSON: [{"symbol": "...", "strategy": "...", "signal_strength": ..., '
                '"verdict": "ENTER|SKIP", "reasoning": "..."}]'
            )
            text = await run_agent(llm, tools, config, prompt)
            candidates = parse_json_response(text, [])
            if not isinstance(candidates, list):
                candidates = [candidates] if candidates else []
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


def make_risk_sizing(llm: BaseChatModel, config: AgentConfig, tools: list[BaseTool] | None = None):
    """Create the risk_sizing node (tool+agent hybrid).

    Computes position sizes via LLM reasoning with tool access,
    then validates each candidate through SafetyGate (pure Python).
    """
    tools = tools or []

    async def risk_sizing(state: TradingState) -> dict[str, Any]:
        candidates = state.get("entry_candidates", [])
        if not candidates:
            return {
                "risk_verdicts": [],
                "decisions": [{"node": "risk_sizing", "action": "no_candidates"}],
            }

        portfolio_ctx = state.get("portfolio_context", {})
        gate = SafetyGate()

        try:
            prompt = (
                f"Size these entry candidates:\n{json.dumps(candidates, default=str)}\n"
                f"Portfolio context:\n{json.dumps(portfolio_ctx, default=str)}\n\n"
                "Use your tools to:\n"
                "1. Fetch the current portfolio state\n"
                "2. Compute risk metrics for each candidate\n"
                "3. Compute position sizes using Kelly criterion\n\n"
                'Return JSON: [{"symbol": "...", "recommended_size_pct": ..., '
                '"reasoning": "...", "confidence": ...}]'
            )
            text = await run_agent(llm, tools, config, prompt)
            sizing_results = parse_json_response(text, [])
            if not isinstance(sizing_results, list):
                sizing_results = [sizing_results] if sizing_results else []

            verdicts = []
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


def make_portfolio_construction():
    """Create the portfolio_construction node (deterministic, no LLM).

    Reads approved candidates from risk_sizing and current positions from
    portfolio_context. Runs the optimizer. Computes delta trades. Passes
    the optimized trade list to portfolio_review.
    """
    import numpy as np

    from quantstack.portfolio.optimizer import optimize_portfolio

    async def portfolio_construction(state: TradingState) -> dict[str, Any]:
        try:
            verdicts = state.get("risk_verdicts", [])
            approved = [v for v in verdicts if v.get("approved", False)]

            if not approved:
                return {
                    "portfolio_target_weights": {},
                    "decisions": [{"node": "portfolio_construction", "action": "no_candidates"}],
                }

            portfolio_ctx = state.get("portfolio_context", {})
            positions = portfolio_ctx.get("positions", [])

            # Build symbol list from candidates + existing positions
            candidate_symbols = [c.get("symbol", "") for c in approved]
            position_symbols = [p.get("symbol", "") for p in positions]
            all_symbols = list(dict.fromkeys(candidate_symbols + position_symbols))

            if not all_symbols:
                return {
                    "portfolio_target_weights": {},
                    "decisions": [{"node": "portfolio_construction", "action": "no_symbols"}],
                }

            n = len(all_symbols)

            # Build current weights (from positions)
            total_equity = portfolio_ctx.get("total_equity", 100_000)
            current_weights = np.zeros(n)
            for p in positions:
                sym = p.get("symbol", "")
                if sym in all_symbols:
                    idx = all_symbols.index(sym)
                    mv = abs(p.get("quantity", 0) * p.get("current_price", 0))
                    current_weights[idx] = mv / total_equity if total_equity > 0 else 0

            # Build alpha signals from candidate conviction
            alpha_signals = np.zeros(n)
            for c in approved:
                sym = c.get("symbol", "")
                if sym in all_symbols:
                    idx = all_symbols.index(sym)
                    alpha_signals[idx] = c.get("conviction", 0.5)

            # Simple diagonal covariance (use actual returns in production)
            cov_matrix = np.eye(n) * 0.02

            sector_map = {sym: "default" for sym in all_symbols}
            strategy_map = {sym: "default" for sym in all_symbols}

            target_weights, meta = optimize_portfolio(
                cov_matrix, alpha_signals, current_weights,
                sector_map, strategy_map,
            )

            # Build weight dict
            weight_dict = {sym: float(target_weights[i]) for i, sym in enumerate(all_symbols)}

            return {
                "portfolio_target_weights": weight_dict,
                "decisions": [{
                    "node": "portfolio_construction",
                    "action": "optimized",
                    "n_assets": n,
                    "turnover": meta.get("turnover", 0),
                    "feasible": meta.get("feasible", True),
                }],
            }
        except Exception as exc:
            logger.error("portfolio_construction failed: %s", exc)
            return {
                "portfolio_target_weights": {},
                "errors": [f"portfolio_construction: {exc}"],
            }

    return portfolio_construction


def make_portfolio_review(llm: BaseChatModel, config: AgentConfig, tools: list[BaseTool] | None = None):
    """Create the portfolio_review node (agent: fund_manager with tool access)."""
    tools = tools or []

    async def portfolio_review(state: TradingState) -> dict[str, Any]:
        try:
            verdicts = state.get("risk_verdicts", [])
            approved = [v for v in verdicts if v.get("approved", False)]
            prompt = (
                f"Review these risk-approved candidates:\n{json.dumps(approved, default=str)}\n\n"
                "Use your tools to:\n"
                "1. Fetch the full portfolio for correlation analysis\n"
                "2. Compute risk metrics for the proposed additions\n"
                "3. Search knowledge base for past experiences with these symbols\n\n"
                "Assess portfolio-level risk: correlation, allocation, diversity.\n"
                'Return JSON: [{"symbol": "...", "decision": "APPROVED|REJECTED", "reason": "..."}]'
            )
            text = await run_agent(llm, tools, config, prompt)
            decisions = parse_json_response(text, [])
            if not isinstance(decisions, list):
                decisions = [decisions] if decisions else []
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


def make_options_analysis(llm: BaseChatModel, config: AgentConfig, tools: list[BaseTool] | None = None):
    """Create the options_analysis node (agent with tool access)."""
    tools = tools or []

    async def options_analysis(state: TradingState) -> dict[str, Any]:
        try:
            fm_decisions = state.get("fund_manager_decisions", [])
            approved = [d for d in fm_decisions if d.get("decision") == "APPROVED"]
            if not approved:
                return {
                    "options_analysis": [],
                    "decisions": [{"node": "options_analysis", "action": "no_candidates"}],
                }
            # Include earnings analysis context if available
            earnings_ctx = state.get("earnings_analysis", {})
            earnings_section = ""
            if earnings_ctx:
                earnings_section = (
                    f"\n--- Earnings Context ---\n"
                    f"{json.dumps(earnings_ctx, default=str)}\n"
                    "Adjust structure selection: prefer iron condors when IV premium ratio > 1.5, "
                    "directional options when IV premium < 1.2 with conviction.\n"
                    f"--- End Earnings ---\n\n"
                )

            prompt = (
                f"Analyze options structures for:\n{json.dumps(approved, default=str)}\n"
                f"{earnings_section}"
                "Use your tools to:\n"
                "1. Fetch options chains for each symbol\n"
                "2. Compute Greeks for candidate structures\n"
                "3. Search knowledge base for options lessons\n\n"
                "Select optimal options structures based on IV rank, regime, direction.\n"
                'Return JSON: [{"symbol": "...", "structure": "...", "legs": [...], '
                '"max_profit": ..., "max_loss": ..., "reasoning": "..."}]'
            )
            text = await run_agent(llm, tools, config, prompt)
            analysis = parse_json_response(text, [])
            if not isinstance(analysis, list):
                analysis = [analysis] if analysis else []
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


def make_execute_entries(llm: BaseChatModel, config: AgentConfig, tools: list[BaseTool] | None = None):
    """Create the execute_entries node (tool, deterministic)."""
    tools = tools or []

    async def execute_entries(state: TradingState) -> dict[str, Any]:
        try:
            fm_decisions = state.get("fund_manager_decisions", [])
            approved = [d for d in fm_decisions if d.get("decision") == "APPROVED"]
            if not approved:
                return {
                    "entry_orders": [],
                    "decisions": [{"node": "execute_entries", "action": "no_approved"}],
                }
            prompt = (
                f"Execute entries for approved candidates:\n{json.dumps(approved, default=str)}\n"
                f"Options analysis:\n{json.dumps(state.get('options_analysis', []), default=str)}\n\n"
                "Use the execute_order tool for each entry.\n"
                'Return JSON: [{"symbol": "...", "order_id": "...", "type": "..."}]'
            )
            text = await run_agent(llm, tools, config, prompt)
            orders = parse_json_response(text, [])
            if not isinstance(orders, list):
                orders = [orders] if orders else []
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


def make_reflection(llm: BaseChatModel, config: AgentConfig, tools: list[BaseTool] | None = None):
    """Create the reflection node (agent with tool access)."""
    tools = tools or []

    async def reflection(state: TradingState) -> dict[str, Any]:
        try:
            prompt = (
                f"Reflect on this trading cycle:\n"
                f"Exits: {json.dumps(state.get('exit_orders', []), default=str)}\n"
                f"Entries: {json.dumps(state.get('entry_orders', []), default=str)}\n\n"
                "Use your tools to:\n"
                "1. Fetch portfolio to see current state after this cycle\n"
                "2. Search knowledge base for similar past outcomes\n\n"
                "Analyze what happened and extract lessons.\n"
                'Return JSON: {"reflection": "...", "lessons": [...]}'
            )
            text = await run_agent(llm, tools, config, prompt)
            parsed = parse_json_response(text, {"reflection": text})
            return {
                "reflection": parsed.get("reflection", text),
                "decisions": [{"node": "reflection", "action": "completed"}],
            }
        except Exception as exc:
            logger.error("reflection failed: %s", exc)
            return {
                "reflection": f"Reflection failed: {exc}",
                "errors": [f"reflection: {exc}"],
            }

    return reflection
