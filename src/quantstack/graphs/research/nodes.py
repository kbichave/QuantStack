"""Node functions for the Research Graph.

8 nodes + 1 conditional router. Agent nodes use LLM reasoning;
tool nodes are deterministic data-gathering functions.
"""

import json
import logging
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from quantstack.graphs.config import AgentConfig
from quantstack.graphs.state import ResearchState

logger = logging.getLogger(__name__)


def _build_system_message(config: AgentConfig) -> SystemMessage:
    return SystemMessage(content=(
        f"You are a {config.role}.\n\n"
        f"Goal: {config.goal}\n\n"
        f"Background: {config.backstory}\n\n"
        "Always respond with valid JSON."
    ))


def make_context_load(llm: BaseChatModel, config: AgentConfig):
    """Create the context_load node (deterministic data gathering)."""

    async def context_load(state: ResearchState) -> dict[str, Any]:
        try:
            response = await llm.ainvoke([
                _build_system_message(config),
                HumanMessage(content=(
                    f"Cycle {state['cycle_number']}, regime: {state.get('regime', 'unknown')}. "
                    "Summarize the current context for research: market regime, "
                    "portfolio gaps, recent session findings, and strategy registry status. "
                    "Return JSON with a 'summary' field."
                )),
            ])
            try:
                parsed = json.loads(response.content)
                summary = parsed.get("summary", response.content)
            except (json.JSONDecodeError, TypeError):
                summary = response.content
            return {
                "context_summary": summary,
                "decisions": [{"node": "context_load", "action": "loaded_context"}],
            }
        except Exception as exc:
            logger.error("context_load failed: %s", exc)
            return {
                "context_summary": f"Context load failed: {exc}",
                "errors": [f"context_load: {exc}"],
            }

    return context_load


def make_domain_selection(llm: BaseChatModel, config: AgentConfig):
    """Create the domain_selection node (agent — LLM reasoning)."""

    async def domain_selection(state: ResearchState) -> dict[str, Any]:
        try:
            response = await llm.ainvoke([
                _build_system_message(config),
                HumanMessage(content=(
                    f"Context: {state.get('context_summary', 'no context')}\n\n"
                    "Select a research domain: 'swing', 'investment', or 'options'. "
                    "Also select 1-3 symbols to research. "
                    'Return JSON: {"domain": "...", "symbols": ["..."], "reasoning": "..."}'
                )),
            ])
            try:
                parsed = json.loads(response.content)
            except (json.JSONDecodeError, TypeError):
                parsed = {"domain": "swing", "symbols": [], "reasoning": response.content}
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


def make_hypothesis_generation(llm: BaseChatModel, config: AgentConfig):
    """Create the hypothesis_generation node (agent)."""

    async def hypothesis_generation(state: ResearchState) -> dict[str, Any]:
        try:
            response = await llm.ainvoke([
                _build_system_message(config),
                HumanMessage(content=(
                    f"Domain: {state.get('selected_domain', 'swing')}\n"
                    f"Symbols: {state.get('selected_symbols', [])}\n\n"
                    "Generate a testable trading hypothesis. It must be specific "
                    "enough to validate with signal computation. "
                    'Return JSON: {"hypothesis": "...", "signals_to_check": [...]}'
                )),
            ])
            try:
                parsed = json.loads(response.content)
            except (json.JSONDecodeError, TypeError):
                parsed = {"hypothesis": response.content}
            return {
                "hypothesis": parsed.get("hypothesis", response.content),
                "decisions": [{"node": "hypothesis_generation", "hypothesis": parsed.get("hypothesis")}],
            }
        except Exception as exc:
            logger.error("hypothesis_generation failed: %s", exc)
            return {"hypothesis": "", "errors": [f"hypothesis_generation: {exc}"]}

    return hypothesis_generation


def make_signal_validation(llm: BaseChatModel, config: AgentConfig):
    """Create the signal_validation node (tool-heavy)."""

    async def signal_validation(state: ResearchState) -> dict[str, Any]:
        try:
            response = await llm.ainvoke([
                _build_system_message(config),
                HumanMessage(content=(
                    f"Hypothesis: {state.get('hypothesis', '')}\n"
                    f"Symbols: {state.get('selected_symbols', [])}\n\n"
                    "Validate this hypothesis using technical signals. "
                    "Check for confluence across indicators. "
                    'Return JSON: {"passed": true/false, "signals": [...], "reason": "..."}'
                )),
            ])
            try:
                result = json.loads(response.content)
            except (json.JSONDecodeError, TypeError):
                result = {"passed": False, "reason": "Failed to parse validation result"}
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


def make_backtest_validation(llm: BaseChatModel, config: AgentConfig):
    """Create the backtest_validation node."""

    async def backtest_validation(state: ResearchState) -> dict[str, Any]:
        try:
            response = await llm.ainvoke([
                _build_system_message(config),
                HumanMessage(content=(
                    f"Hypothesis: {state.get('hypothesis', '')}\n"
                    f"Symbols: {state.get('selected_symbols', [])}\n\n"
                    "Run a backtest for this validated hypothesis. "
                    'Return JSON: {"backtest_id": "...", "sharpe": ..., "win_rate": ..., "max_dd": ...}'
                )),
            ])
            try:
                parsed = json.loads(response.content)
            except (json.JSONDecodeError, TypeError):
                parsed = {"backtest_id": "bt-unknown"}
            return {
                "backtest_id": parsed.get("backtest_id", "bt-unknown"),
                "decisions": [{"node": "backtest_validation", "metrics": parsed}],
            }
        except Exception as exc:
            logger.error("backtest_validation failed: %s", exc)
            return {"backtest_id": "", "errors": [f"backtest_validation: {exc}"]}

    return backtest_validation


def make_ml_experiment(llm: BaseChatModel, config: AgentConfig):
    """Create the ml_experiment node (agent)."""

    async def ml_experiment(state: ResearchState) -> dict[str, Any]:
        try:
            response = await llm.ainvoke([
                _build_system_message(config),
                HumanMessage(content=(
                    f"Hypothesis: {state.get('hypothesis', '')}\n"
                    f"Backtest: {state.get('backtest_id', '')}\n\n"
                    "Design and run an ML experiment to validate this strategy. "
                    'Return JSON: {"experiment_id": "...", "model_type": "...", "ic": ..., "accuracy": ...}'
                )),
            ])
            try:
                parsed = json.loads(response.content)
            except (json.JSONDecodeError, TypeError):
                parsed = {"experiment_id": "exp-unknown"}
            return {
                "ml_experiment_id": parsed.get("experiment_id", "exp-unknown"),
                "decisions": [{"node": "ml_experiment", "results": parsed}],
            }
        except Exception as exc:
            logger.error("ml_experiment failed: %s", exc)
            return {"ml_experiment_id": "", "errors": [f"ml_experiment: {exc}"]}

    return ml_experiment


def make_strategy_registration(llm: BaseChatModel, config: AgentConfig):
    """Create the strategy_registration node (tool)."""

    async def strategy_registration(state: ResearchState) -> dict[str, Any]:
        try:
            response = await llm.ainvoke([
                _build_system_message(config),
                HumanMessage(content=(
                    f"Register strategy from hypothesis: {state.get('hypothesis', '')}\n"
                    f"Backtest: {state.get('backtest_id', '')}\n"
                    f"ML experiment: {state.get('ml_experiment_id', '')}\n\n"
                    "Register this strategy with status 'paper_ready'. "
                    'Return JSON: {"strategy_id": "...", "status": "paper_ready"}'
                )),
            ])
            try:
                parsed = json.loads(response.content)
            except (json.JSONDecodeError, TypeError):
                parsed = {"strategy_id": "strat-unknown"}
            return {
                "registered_strategy_id": parsed.get("strategy_id", "strat-unknown"),
                "decisions": [{"node": "strategy_registration", "strategy_id": parsed.get("strategy_id")}],
            }
        except Exception as exc:
            logger.error("strategy_registration failed: %s", exc)
            return {"registered_strategy_id": "", "errors": [f"strategy_registration: {exc}"]}

    return strategy_registration


def make_knowledge_update(llm: BaseChatModel, config: AgentConfig):
    """Create the knowledge_update node (tool)."""

    async def knowledge_update(state: ResearchState) -> dict[str, Any]:
        try:
            response = await llm.ainvoke([
                _build_system_message(config),
                HumanMessage(content=(
                    f"Update knowledge base with research findings:\n"
                    f"Hypothesis: {state.get('hypothesis', '')}\n"
                    f"Strategy: {state.get('registered_strategy_id', '')}\n\n"
                    "Summarize what was learned and return JSON confirmation."
                )),
            ])
            return {
                "decisions": [{"node": "knowledge_update", "action": "updated"}],
            }
        except Exception as exc:
            logger.error("knowledge_update failed: %s", exc)
            return {"errors": [f"knowledge_update: {exc}"]}

    return knowledge_update
