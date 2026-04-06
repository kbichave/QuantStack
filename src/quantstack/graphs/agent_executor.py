"""Shared tool-calling agent executor for LangGraph nodes.

Runs a tool-calling loop: LLM decides to call tools -> tools execute ->
results fed back -> LLM continues until it produces a final text response
(no more tool calls).

Supports three tool loading strategies:
- Anthropic native: BM25 tool search with defer_loading + server_tool_use filtering
- Bigtool: langgraph-bigtool with pgvector semantic search (any provider)
- Full loading: all tools bound upfront (backward compat)
"""

import json
import logging
import time
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool

from quantstack.graphs.config import AgentConfig
from quantstack.observability.instrumentation import log_llm_call, log_tool_call

logger = logging.getLogger(__name__)

MAX_TOOL_ROUNDS = 10  # Safety cap to prevent infinite tool-calling loops
MAX_TOOL_RESULT_CHARS = 4000  # Truncate individual tool results beyond this
MAX_MESSAGE_CHARS = 150_000  # Rough char budget (~37k tokens) before pruning old messages

# Agent-to-graph mapping (used for system prompt categories and event routing)
_AGENT_TEAMS = {
    "quant_researcher": "research", "ml_scientist": "research",
    "strategy_rd": "research", "equity_swing_researcher": "research",
    "equity_investment_researcher": "research", "options_researcher": "research",
    "execution_researcher": "research", "community_intel": "research",
    "position_monitor": "trading", "entry_scanner": "trading",
    "exit_manager": "trading", "risk_assessor": "trading",
    "daily_planner": "trading", "fund_manager": "trading",
    "options_analyst": "trading", "trade_debater": "trading",
    "trade_reflector": "trading", "execution_manager": "trading",
    "risk_analyst": "trading", "executor": "trading",
    "market_intel": "trading", "earnings_analyst": "trading",
    "health_monitor": "supervisor", "diagnostician": "supervisor",
    "self_healer": "supervisor", "strategy_promoter": "supervisor",
    "scheduler": "supervisor",
}

# Per-graph tool category descriptions for system prompt injection
_TOOL_CATEGORIES = {
    "trading": (
        "\nYou have access to tools across these categories:\n"
        "- Signal & Analysis: signal briefs, multi-symbol briefs, technical indicators, sentiment\n"
        "- Data: market data, fundamentals, earnings, analyst estimates, company facts\n"
        "- Portfolio: positions, equity curve, daily P&L, strategy performance, fill history\n"
        "- Risk: risk metrics, VaR, drawdown, stress testing, position sizing, risk limits, GARCH\n"
        "- Execution: order submission, options execution, position closing, broker connectivity\n"
        "- Options: options chain, Greeks, IV surface, option pricing, structure scoring\n"
        "- Market Intel: web search, credit market signals, institutional accumulation, breadth\n"
        "- Monitoring: position monitor, exit signals, stop updates, alert management\n"
        "- Knowledge: knowledge base search, audit trail, decision history\n"
        "\nUse the tool search to find specific tools when needed.\n"
    ),
    "research": (
        "\nYou have access to tools across these categories:\n"
        "- Signal & Analysis: signal briefs, technical indicators, institutional accumulation, market breadth\n"
        "- Data: market data, fundamentals, earnings, analyst estimates, company facts\n"
        "- Features: feature computation, feature matrix, information coefficient, ADF test\n"
        "- ML Training: model training, prediction, drift detection, purged CV\n"
        "- Backtesting: backtest execution, walk-forward, Monte Carlo, alpha decay\n"
        "- Validation: deflated Sharpe ratio, probability of overfitting, lookahead bias, leakage detection\n"
        "- Options: options chain, Greeks, IV surface, option pricing, structure analysis\n"
        "- FinRL: environment creation, model training, ensemble training, evaluation, promotion\n"
        "- Strategy: registry, registration, gaps, regime strategies, screening\n"
        "- Knowledge: knowledge base search\n"
        "\nUse the tool search to find specific tools when needed.\n"
    ),
    "supervisor": (
        "\nYou have access to tools across these categories:\n"
        "- System Health: heartbeat checks, service reachability, broker connectivity\n"
        "- Data Freshness: symbol coverage, API rate limit status, intraday status\n"
        "- Strategy Lifecycle: strategy registry, performance metrics, promotion, retirement\n"
        "- Knowledge: knowledge base search, audit trail, decision history\n"
        "- Data Acquisition: historical data fetch, ticker registration\n"
        "\nUse the tool search to find specific tools when needed.\n"
    ),
}


def _truncate_tool_result(result: str, max_chars: int = MAX_TOOL_RESULT_CHARS) -> str:
    """Truncate long tool results to stay within token budget."""
    if len(result) <= max_chars:
        return result
    return result[:max_chars] + f"\n... [truncated, {len(result)} total chars]"


def _estimate_message_chars(messages: list) -> int:
    """Rough character count across all messages."""
    total = 0
    for m in messages:
        content = m.content if hasattr(m, "content") else ""
        total += len(content) if isinstance(content, str) else 0
    return total


def _prune_messages(messages: list) -> list:
    """Drop old tool round pairs (AIMessage + ToolMessages) if messages are too long.

    Keeps: system message (index 0), user message (index 1), and the most recent
    tool rounds. Drops the oldest tool rounds first.
    """
    if _estimate_message_chars(messages) <= MAX_MESSAGE_CHARS:
        return messages

    # Keep system + user, prune from the middle
    kept = messages[:2]
    # Find tool round boundaries (AIMessage with tool_calls followed by ToolMessages)
    rounds = []
    current_round = []
    for msg in messages[2:]:
        current_round.append(msg)
        if isinstance(msg, (HumanMessage, AIMessage)) and not getattr(msg, "tool_calls", None):
            rounds.append(current_round)
            current_round = []
        elif isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            # Start of a new tool round -- flush previous if it was just tool results
            pass
    if current_round:
        rounds.append(current_round)

    # Drop oldest rounds until under budget
    while rounds and _estimate_message_chars(kept + [m for r in rounds for m in r]) > MAX_MESSAGE_CHARS:
        dropped = rounds.pop(0)
        logger.debug("Pruned %d messages from agent context to stay under token limit", len(dropped))

    for r in rounds:
        kept.extend(r)
    return kept


def _is_server_tool_call(tool_call: dict) -> bool:
    """Check if a tool_call entry is a server-side tool (tool search) that should be skipped."""
    name = tool_call.get("name", "")
    return name.startswith("tool_search_tool")


def build_system_message(
    config: AgentConfig,
    graph_name: str | None = None,
) -> SystemMessage:
    """Build a system message from an agent config.

    When tool search is active (config.always_loaded_tools is non-empty),
    appends tool category guidance tailored to the agent's graph.
    """
    if graph_name is None:
        graph_name = _AGENT_TEAMS.get(config.name, "other")

    base = (
        f"You are a {config.role}.\n\n"
        f"Goal: {config.goal}\n\n"
        f"Background: {config.backstory}\n\n"
        "You have access to tools. Use them to gather real data before reasoning.\n"
        "When you have enough information, respond with your final answer as valid JSON.\n"
        "Do NOT fabricate data — always call the appropriate tool first."
    )

    if config.always_loaded_tools:
        categories = _TOOL_CATEGORIES.get(graph_name, "")
        if categories:
            base += categories

    return SystemMessage(content=base)


async def run_agent(
    llm: BaseChatModel,
    tools: list[BaseTool],
    config: AgentConfig,
    user_message: str,
    max_rounds: int = MAX_TOOL_ROUNDS,
    _skip_bigtool: bool = False,
) -> str:
    """Run a tool-calling agent loop.

    Automatically routes to bigtool executor when:
    - config.always_loaded_tools is set (deferred loading desired)
    - LLM is not ChatAnthropic (no native defer_loading support)

    Otherwise runs the standard ReAct loop with tool search filtering.

    Returns the final text response from the LLM.
    """
    # Auto-detect bigtool mode: always_loaded_tools configured + non-Anthropic LLM
    if (
        not _skip_bigtool
        and config.always_loaded_tools
        and type(llm).__name__ != "ChatAnthropic"
        and tools
    ):
        return await run_agent_bigtool(llm, tools, config, user_message)

    from quantstack.dashboard.events import publish_event

    tool_map = {t.name: t for t in tools}
    messages = [
        build_system_message(config),
        HumanMessage(content=user_message),
    ]

    graph_name = _AGENT_TEAMS.get(config.name, "other")

    publish_event(
        graph_name=graph_name,
        node_name=config.name,
        agent_name=config.name,
        event_type="agent_start",
        content=user_message[:500],
    )

    model_name = getattr(llm, "model_id", "") or getattr(llm, "model_name", "") or type(llm).__name__

    for round_num in range(max_rounds):
        t0 = time.monotonic()
        try:
            response: AIMessage = await llm.ainvoke(messages)
        except Exception as exc:
            if round_num == 0:
                raise  # First-round failures indicate config problems
            logger.warning(
                "Agent %s API error at round %d: %s",
                config.name, round_num, exc,
            )
            return json.dumps({
                "error": "agent_executor_mid_conversation_failure",
                "agent": config.name,
                "round": round_num,
                "detail": str(exc),
            })
        llm_dur = time.monotonic() - t0

        # Log LLM call to Langfuse
        usage = {}
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            um = response.usage_metadata
            usage = {"input": um.get("input_tokens", 0), "output": um.get("output_tokens", 0)}
        log_llm_call(
            agent_name=config.name,
            model_name=model_name,
            input_messages=messages,
            output_content=response.content or "",
            duration_seconds=llm_dur,
            tool_calls=response.tool_calls if response.tool_calls else None,
            usage=usage,
        )

        messages.append(response)

        # If no tool calls, we have our final answer
        if not response.tool_calls:
            publish_event(
                graph_name=graph_name,
                node_name=config.name,
                agent_name=config.name,
                event_type="agent_response",
                content=response.content[:500] if response.content else "",
            )
            return response.content

        # Execute each tool call
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_id = tool_call["id"]

            # Skip server-side tool blocks (tool search results)
            if _is_server_tool_call(tool_call):
                logger.debug(
                    "Skipping server-side tool call '%s' in agent %s",
                    tool_name, config.name,
                )
                continue

            tool_t0 = time.monotonic()
            tool_ok = True
            if tool_name not in tool_map:
                result = json.dumps({"error": f"Unknown tool: {tool_name}"})
                tool_ok = False
            else:
                try:
                    result = await tool_map[tool_name].ainvoke(tool_args)
                    if not isinstance(result, str):
                        result = json.dumps(result, default=str)
                except Exception as exc:
                    logger.warning(
                        "Tool %s failed in agent %s: %s",
                        tool_name, config.name, exc,
                    )
                    result = json.dumps({"error": str(exc)})
                    tool_ok = False
            log_tool_call(
                agent_name=config.name,
                tool_name=tool_name,
                tool_args=tool_args,
                result=result,
                duration_seconds=time.monotonic() - tool_t0,
                success=tool_ok,
            )

            publish_event(
                graph_name=graph_name,
                node_name=config.name,
                agent_name=config.name,
                event_type="tool_call",
                content=f"{tool_name}({json.dumps(tool_args, default=str)[:200]})",
                metadata={"tool": tool_name, "success": "error" not in result[:50]},
            )

            messages.append(ToolMessage(
                content=_truncate_tool_result(result),
                tool_call_id=tool_id,
            ))

        # Prune old messages if conversation is getting too long
        messages = _prune_messages(messages)

    # Exhausted rounds -- ask LLM to produce final answer without tools
    logger.warning(
        "Agent %s hit max tool rounds (%d), forcing final answer",
        config.name, max_rounds,
    )
    messages.append(HumanMessage(
        content="You've used all available tool rounds. Produce your final JSON answer now "
                "using the data you've already gathered."
    ))
    # Call without tools to force a text response
    final = await llm.ainvoke(messages)
    publish_event(
        graph_name=graph_name,
        node_name=config.name,
        agent_name=config.name,
        event_type="agent_response",
        content=final.content[:500] if final.content else "",
    )
    return final.content


async def run_agent_bigtool(
    llm: BaseChatModel,
    tools: list[BaseTool],
    config: AgentConfig,
    user_message: str,
) -> str:
    """Run an agent using langgraph-bigtool for dynamic tool retrieval.

    Used for non-Anthropic providers (Bedrock, OpenAI) when always_loaded_tools
    is configured. Tools are discovered via pgvector semantic search using
    ollama mxbai-embed-large embeddings.

    Falls back to run_agent() with all tools if bigtool store is unavailable.
    """
    from quantstack.dashboard.events import publish_event
    from quantstack.graphs.bigtool_store import TOOL_NAMESPACE, ensure_tool_store_populated

    graph_name = _AGENT_TEAMS.get(config.name, "other")

    publish_event(
        graph_name=graph_name,
        node_name=config.name,
        agent_name=config.name,
        event_type="agent_start",
        content=f"[bigtool] {user_message[:450]}",
    )

    store = ensure_tool_store_populated()
    if store is None:
        logger.warning(
            "Bigtool store unavailable for agent '%s' — falling back to full tool loading",
            config.name,
        )
        bound_llm = llm.bind_tools(tools)
        return await run_agent(bound_llm, tools, config, user_message, _skip_bigtool=True)

    # Build tool registry: tool_name -> BaseTool (bigtool needs string keys)
    tool_registry = {t.name: t for t in tools}

    try:
        from langgraph_bigtool import create_agent
        from langgraph.prebuilt import InjectedStore, ToolNode
        from langgraph.store.base import BaseStore
        from typing import Annotated

        # langgraph >=1.1 renamed inject_tool_args to _inject_tool_args.
        # bigtool 0.0.3 still calls the public name. Patch if needed.
        if not hasattr(ToolNode, "inject_tool_args") and hasattr(ToolNode, "_inject_tool_args"):
            ToolNode.inject_tool_args = ToolNode._inject_tool_args

        agent_tool_names = set(tool_registry.keys())

        def retrieve_tools_for_agent(
            query: str,
            *,
            store: Annotated[BaseStore, InjectedStore],
        ) -> list[str]:
            """Retrieve tools via semantic search, filtered to this agent's registry."""
            results = store.search(TOOL_NAMESPACE, query=query, limit=10)
            return [r.key for r in results if r.key in agent_tool_names][:5]

        async def aretrieve_tools_for_agent(
            query: str,
            *,
            store: Annotated[BaseStore, InjectedStore],
        ) -> list[str]:
            """Async retrieve tools via semantic search, filtered to this agent's registry."""
            results = await store.asearch(TOOL_NAMESPACE, query=query, limit=10)
            return [r.key for r in results if r.key in agent_tool_names][:5]

        builder = create_agent(
            llm,
            tool_registry,
            namespace_prefix=TOOL_NAMESPACE,
            retrieve_tools_function=retrieve_tools_for_agent,
            retrieve_tools_coroutine=aretrieve_tools_for_agent,
        )
        agent = builder.compile(store=store)
        logger.info(
            "Bigtool agent compiled for '%s' with %d tools in registry",
            config.name, len(tool_registry),
        )

        # Build system + user messages
        sys_msg = build_system_message(config)
        # Explicitly pass store in config — required when this subgraph runs
        # inside another LangGraph node (parent config propagates store=None)
        result = await agent.ainvoke(
            {"messages": [sys_msg, HumanMessage(content=user_message)]},
            config={"configurable": {"__pregel_store": store}},
        )

        # Extract final response from messages
        final_messages = result.get("messages", [])
        for msg in reversed(final_messages):
            if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
                publish_event(
                    graph_name=graph_name,
                    node_name=config.name,
                    agent_name=config.name,
                    event_type="agent_response",
                    content=msg.content[:500],
                )
                return msg.content

        # No clean text response found — return last message content
        if final_messages:
            last = final_messages[-1]
            content = last.content if hasattr(last, "content") else str(last)
            return content
        return json.dumps({"error": "bigtool agent produced no response"})

    except Exception as exc:
        logger.error(
            "Bigtool agent failed for '%s': %s — falling back to full tool loading",
            config.name, exc, exc_info=True,
        )
        bound_llm = llm.bind_tools(tools)
        return await run_agent(bound_llm, tools, config, user_message, _skip_bigtool=True)


def parse_json_response(text: str, fallback: dict | list | None = None) -> dict | list:
    """Parse a JSON object or array from LLM response, handling common issues.

    When *fallback* is a list, the function accepts JSON arrays as valid results.
    When *fallback* is a dict (or None), only JSON objects are accepted.
    Returns fallback if parsing fails entirely.
    """
    accept_list = isinstance(fallback, list)
    if not text:
        return fallback if fallback is not None else {}

    def _try_parse(s: str) -> dict | list | None:
        try:
            val = json.loads(s)
            if isinstance(val, dict):
                return val
            if isinstance(val, list) and accept_list:
                return val
            return None
        except (json.JSONDecodeError, TypeError):
            return None

    # Try direct parse first
    result = _try_parse(text)
    if result is not None:
        return result

    # Try to find a JSON object in the text (prefer {} over [])
    start = text.find("{")
    if start != -1:
        end = text.rfind("}")
        if end > start:
            result = _try_parse(text[start:end + 1])
            if result is not None:
                return result

    # If accepting lists, try array extraction
    if accept_list:
        start = text.find("[")
        if start != -1:
            end = text.rfind("]")
            if end > start:
                result = _try_parse(text[start:end + 1])
                if result is not None:
                    return result

    logger.debug("Failed to parse JSON from response: %.200s", text)
    return fallback if fallback is not None else {}
