"""Node functions for the Research Graph.

8 nodes + 1 conditional router. Agent nodes use LLM reasoning with
tool access to gather real data before making decisions.
"""

import logging
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from quantstack.graphs.agent_executor import parse_json_response, run_agent
from quantstack.graphs.config import AgentConfig
from quantstack.graphs.state import ResearchState

logger = logging.getLogger(__name__)


def make_context_load(llm: BaseChatModel, config: AgentConfig, tools: list[BaseTool] | None = None):
    """Create the context_load node (data gathering with tools)."""
    tools = tools or []

    async def context_load(state: ResearchState) -> dict[str, Any]:
        # Poll for IDEAS_DISCOVERED events from community intel scans
        community_ideas_section = ""
        try:
            from quantstack.coordination.event_bus import EventBus, EventType
            from quantstack.db import db_conn

            with db_conn() as conn:
                bus = EventBus(conn)
                events = bus.poll(
                    "research_graph",
                    event_types=[EventType.IDEAS_DISCOVERED],
                )
                if events:
                    all_ideas = []
                    for evt in events:
                        ideas = evt.payload.get("ideas", [])
                        all_ideas.extend(ideas)
                    if all_ideas:
                        import json as _json

                        community_ideas_section = (
                            f"\n--- Community Intelligence ({len(all_ideas)} ideas) ---\n"
                            f"{_json.dumps(all_ideas[:10], default=str)}\n"
                            "Consider these external ideas when selecting research priorities.\n"
                            "--- End Community Intel ---\n"
                        )
        except Exception as poll_exc:
            logger.debug("Failed to poll IDEAS_DISCOVERED events: %s", poll_exc)

        try:
            prompt = (
                f"Cycle {state['cycle_number']}, regime: {state.get('regime', 'unknown')}.\n"
                f"{community_ideas_section}\n"
                "Use your tools to gather the current research context:\n"
                "1. Search the knowledge base for recent session findings and NEGATIVE results "
                "(to avoid retesting failed hypotheses)\n"
                "2. Use fetch_market_data for a few INDIVIDUAL STOCKS (not just indices) to see "
                "which symbols have data available. Try 2-3 stocks you haven't researched recently.\n"
                "3. Get signal briefs for the symbols that have data.\n"
                "4. Fetch the strategy registry to identify portfolio gaps.\n\n"
                "IMPORTANT: Diversify research across individual stocks, not just ETFs/indices. "
                "Explore momentum, quality, value, sector rotation, and trend-following — not just "
                "mean-reversion. If regime is 'unknown', use price action and trend analysis instead.\n\n"
                "Summarize: market regime, portfolio gaps, recent findings, what to research next.\n"
                'Return JSON: {"summary": "...", "regime": "...", "gaps": [...], "priorities": [...]}'
            )
            text = await run_agent(llm, tools, config, prompt)
            parsed = parse_json_response(text, {"summary": text})
            return {
                "context_summary": parsed.get("summary", text),
                "decisions": [{"node": "context_load", "action": "loaded_context"}],
            }
        except Exception as exc:
            logger.error("context_load failed: %s", exc)
            return {
                "context_summary": f"Context load failed: {exc}",
                "errors": [f"context_load: {exc}"],
            }

    return context_load


def make_domain_selection(llm: BaseChatModel, config: AgentConfig, tools: list[BaseTool] | None = None):
    """Create the domain_selection node (agent with tool access)."""
    tools = tools or []

    async def domain_selection(state: ResearchState) -> dict[str, Any]:
        try:
            prompt = (
                f"Context: {state.get('context_summary', 'no context')}\n\n"
                "Based on the context, select a research domain and symbols.\n"
                "Use your tools to:\n"
                "1. Search knowledge base for what domains need more coverage\n"
                "2. Use fetch_market_data to check which individual stocks have data available. "
                "Try stocks you haven't researched in recent cycles. Avoid indices (SPY/QQQ/IWM) "
                "for strategy research — focus on individual stocks.\n"
                "3. Get signal briefs for the stocks that have data.\n\n"
                "Select domain: 'swing', 'investment', or 'options'.\n"
                "Select 1-3 individual stocks that have data and show interesting setups.\n\n"
                "Strategy types to explore (rotate through these):\n"
                "- Momentum: 20-day breakouts, relative strength, earnings acceleration\n"
                "- Quality/Value: ROE improvement, P/E compression, buyback yield\n"
                "- Sector rotation: energy vs tech, cyclical vs defensive\n"
                "- Trend-following: moving average crossovers, ADX-based entries\n"
                "- Event-driven: earnings drift, analyst revision, institutional accumulation\n\n"
                'Return JSON: {"domain": "...", "symbols": ["..."], "reasoning": "..."}'
            )
            text = await run_agent(llm, tools, config, prompt)
            parsed = parse_json_response(text, {"domain": "swing", "symbols": []})
            return {
                "selected_domain": parsed.get("domain", "swing"),
                "selected_symbols": parsed.get("symbols", []),
                "decisions": [{"node": "domain_selection", "domain": parsed.get("domain"), "reasoning": parsed.get("reasoning")}],
            }
        except Exception as exc:
            logger.error("domain_selection failed: %s", exc)
            return {
                "selected_domain": "swing",
                "selected_symbols": [],
                "errors": [f"domain_selection: {exc}"],
            }

    return domain_selection


def make_hypothesis_generation(llm: BaseChatModel, config: AgentConfig, tools: list[BaseTool] | None = None):
    """Create the hypothesis_generation node (agent with tool access)."""
    tools = tools or []

    async def hypothesis_generation(state: ResearchState) -> dict[str, Any]:
        try:
            prompt = (
                f"Domain: {state.get('selected_domain', 'swing')}\n"
                f"Symbols: {state.get('selected_symbols', [])}\n\n"
                "Use your tools to:\n"
                "1. Fetch market data for the selected symbols\n"
                "2. Compute features (indicators, factors) to identify patterns\n"
                "3. Search knowledge base for past hypotheses on these symbols (avoid retesting dead ends)\n\n"
                "Generate a testable trading hypothesis with:\n"
                "- Directional prediction\n"
                "- Economic mechanism\n"
                "- Expected effect size\n"
                "- Falsification criteria\n"
                "- Signals to check\n"
                'Return JSON: {"hypothesis": "...", "mechanism": "...", "signals_to_check": [...], '
                '"expected_effect": "...", "falsification": "..."}'
            )
            text = await run_agent(llm, tools, config, prompt)
            parsed = parse_json_response(text, {"hypothesis": text})
            return {
                "hypothesis": parsed.get("hypothesis", text),
                "decisions": [{"node": "hypothesis_generation", "hypothesis": parsed.get("hypothesis")}],
            }
        except Exception as exc:
            logger.error("hypothesis_generation failed: %s", exc)
            return {"hypothesis": "", "errors": [f"hypothesis_generation: {exc}"]}

    return hypothesis_generation


def make_signal_validation(llm: BaseChatModel, config: AgentConfig, tools: list[BaseTool] | None = None):
    """Create the signal_validation node (tool-heavy)."""
    tools = tools or []

    async def signal_validation(state: ResearchState) -> dict[str, Any]:
        try:
            prompt = (
                f"Hypothesis: {state.get('hypothesis', '')}\n"
                f"Symbols: {state.get('selected_symbols', [])}\n\n"
                "Validate this hypothesis using real data:\n"
                "1. Get signal briefs for the target symbols\n"
                "2. Compute features relevant to the hypothesis\n"
                "3. Check for signal confluence across indicators\n\n"
                "Validation criteria (pass if MAJORITY met):\n"
                "- IC > 0.02 OR strong directional trend in price data\n"
                "- Signal half-life > intended holding period\n"
                "- At least 2 confirming indicators from: price trend, volume, momentum, "
                "mean-reversion, relative strength, fundamentals\n\n"
                "IMPORTANT: If regime is 'unknown' or regime confidence is 0%, do NOT reject "
                "solely for regime reasons. Use price action and technical indicators as regime "
                "proxies instead. A hypothesis can still pass if the price data and technical "
                "signals support it, even without a formal regime classification.\n\n"
                "Be constructive: if 2 of 3 criteria pass, approve for backtesting. "
                "The backtest will be the definitive test.\n\n"
                'Return JSON: {"passed": true/false, "signals": [...], "ic": ..., "reason": "..."}'
            )
            text = await run_agent(llm, tools, config, prompt)
            result = parse_json_response(text, {"passed": False, "reason": "Failed to parse"})
            decisions = [{"node": "signal_validation", "passed": result.get("passed", False)}]
            state_update: dict[str, Any] = {"validation_result": result, "decisions": decisions}
            if not result.get("passed", False):
                state_update["errors"] = [f"Signal validation failed: {result.get('reason', 'unknown')}"]
            return state_update
        except Exception as exc:
            logger.error("signal_validation failed: %s", exc)
            return {
                "validation_result": {"passed": False, "reason": str(exc)},
                "errors": [f"signal_validation: {exc}"],
            }

    return signal_validation


def route_after_validation(state: ResearchState) -> str:
    """Conditional router after signal_validation."""
    from langgraph.graph import END
    if state.get("validation_result", {}).get("passed", False):
        return "backtest_validation"
    return END


def make_backtest_validation(llm: BaseChatModel, config: AgentConfig, tools: list[BaseTool] | None = None):
    """Create the backtest_validation node (tool-heavy)."""
    tools = tools or []

    async def backtest_validation(state: ResearchState) -> dict[str, Any]:
        try:
            prompt = (
                f"Hypothesis: {state.get('hypothesis', '')}\n"
                f"Symbols: {state.get('selected_symbols', [])}\n"
                f"Validation: {state.get('validation_result', {})}\n\n"
                "Run a backtest for this validated hypothesis:\n"
                "1. Fetch market data for the symbols\n"
                "2. Compute features needed for the strategy\n"
                "3. Run the backtest with the strategy parameters\n\n"
                "Evaluate: Sharpe > 0.5, trades > 20, PF > 1.2, max DD < 20%.\n"
                'Return JSON: {"backtest_id": "...", "sharpe": ..., "win_rate": ..., '
                '"max_dd": ..., "trades": ..., "profit_factor": ..., "passed": true/false}'
            )
            text = await run_agent(llm, tools, config, prompt)
            parsed = parse_json_response(text, {"backtest_id": "bt-unknown"})
            return {
                "backtest_id": parsed.get("backtest_id", "bt-unknown"),
                "decisions": [{"node": "backtest_validation", "metrics": parsed}],
            }
        except Exception as exc:
            logger.error("backtest_validation failed: %s", exc)
            return {"backtest_id": "", "errors": [f"backtest_validation: {exc}"]}

    return backtest_validation


def make_ml_experiment(llm: BaseChatModel, config: AgentConfig, tools: list[BaseTool] | None = None):
    """Create the ml_experiment node (agent with ML tools)."""
    tools = tools or []

    async def ml_experiment(state: ResearchState) -> dict[str, Any]:
        try:
            prompt = (
                f"Hypothesis: {state.get('hypothesis', '')}\n"
                f"Backtest: {state.get('backtest_id', '')}\n"
                f"Symbols: {state.get('selected_symbols', [])}\n\n"
                "Design and run an ML experiment:\n"
                "1. Fetch market data for training\n"
                "2. Compute features for the ML model\n"
                "3. Train a model (LightGBM, XGBoost, or ensemble)\n"
                "4. Search knowledge base for past experiment results\n\n"
                "Follow the Feature Quality Protocol: stationarity, redundancy, stability.\n"
                'Return JSON: {"experiment_id": "...", "model_type": "...", "ic": ..., '
                '"accuracy": ..., "feature_importance": [...], "passed": true/false}'
            )
            text = await run_agent(llm, tools, config, prompt)
            parsed = parse_json_response(text, {"experiment_id": "exp-unknown"})
            return {
                "ml_experiment_id": parsed.get("experiment_id", "exp-unknown"),
                "decisions": [{"node": "ml_experiment", "results": parsed}],
            }
        except Exception as exc:
            logger.error("ml_experiment failed: %s", exc)
            return {"ml_experiment_id": "", "errors": [f"ml_experiment: {exc}"]}

    return ml_experiment


def make_strategy_registration(llm: BaseChatModel, config: AgentConfig, tools: list[BaseTool] | None = None):
    """Create the strategy_registration node (tool)."""
    tools = tools or []

    async def strategy_registration(state: ResearchState) -> dict[str, Any]:
        try:
            prompt = (
                f"Register strategy from hypothesis: {state.get('hypothesis', '')}\n"
                f"Backtest: {state.get('backtest_id', '')}\n"
                f"ML experiment: {state.get('ml_experiment_id', '')}\n\n"
                "Use fetch_strategy_registry to check current registry for duplicates.\n"
                "Then register this strategy with status 'paper_ready'.\n"
                'Return JSON: {"strategy_id": "...", "status": "paper_ready", "reasoning": "..."}'
            )
            text = await run_agent(llm, tools, config, prompt)
            parsed = parse_json_response(text, {"strategy_id": "strat-unknown"})
            return {
                "registered_strategy_id": parsed.get("strategy_id", "strat-unknown"),
                "decisions": [{"node": "strategy_registration", "strategy_id": parsed.get("strategy_id")}],
            }
        except Exception as exc:
            logger.error("strategy_registration failed: %s", exc)
            return {"registered_strategy_id": "", "errors": [f"strategy_registration: {exc}"]}

    return strategy_registration


def make_knowledge_update(llm: BaseChatModel, config: AgentConfig, tools: list[BaseTool] | None = None):
    """Create the knowledge_update node (tool)."""
    tools = tools or []

    async def knowledge_update(state: ResearchState) -> dict[str, Any]:
        try:
            prompt = (
                f"Update knowledge base with research findings:\n"
                f"Hypothesis: {state.get('hypothesis', '')}\n"
                f"Strategy: {state.get('registered_strategy_id', '')}\n"
                f"Validation: {state.get('validation_result', {})}\n\n"
                "Use search_knowledge_base to check for existing entries.\n"
                "Summarize what was learned (both positive and negative results).\n"
                'Return JSON: {"updated": true, "summary": "..."}'
            )
            text = await run_agent(llm, tools, config, prompt)
            return {
                "decisions": [{"node": "knowledge_update", "action": "updated"}],
            }
        except Exception as exc:
            logger.error("knowledge_update failed: %s", exc)
            return {"errors": [f"knowledge_update: {exc}"]}

    return knowledge_update
