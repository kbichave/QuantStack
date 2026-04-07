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
from quantstack.observability.cost_queries import TokenBudgetTracker
from quantstack.observability.instrumentation import log_llm_call, log_tool_call
from quantstack.observability.tracing import trace_prompt_cache_metrics

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


PRIORITY_P0 = "P0"  # Never pruned or summarized
PRIORITY_P1 = "P1"  # Summarized when over budget
PRIORITY_P2 = "P2"  # Pruned first (oldest-first)
PRIORITY_P3 = "P3"  # Excluded from LLM context entirely

# Message type override patterns: always P0 regardless of source agent
_P0_TYPE_PATTERNS = frozenset({
    "risk_gate", "kill_switch", "position_state", "portfolio_context",
    "blocking_node_error",
})

SUMMARIZE_TRUNCATE_CHARS = 500
SUMMARIZE_TIMEOUT_S = 2.0


def _get_message_priority(msg) -> str:
    """Read priority tier from message metadata, defaulting to P2."""
    metadata = getattr(msg, "metadata", None) or {}
    if isinstance(metadata, dict):
        return metadata.get("priority_tier", PRIORITY_P2)
    return PRIORITY_P2


def tag_message_priority(
    msg,
    agent_priority: str = PRIORITY_P2,
    message_type: str = "",
) -> None:
    """Tag a message with its priority tier in metadata.

    Type overrides (risk gate, kill switch, etc.) take precedence
    over the agent's configured default.
    """
    priority = agent_priority
    if message_type and any(pat in message_type for pat in _P0_TYPE_PATTERNS):
        priority = PRIORITY_P0
    if not hasattr(msg, "metadata") or msg.metadata is None:
        msg.metadata = {}
    if isinstance(msg.metadata, dict):
        msg.metadata["priority_tier"] = priority


def _truncate_content(content: str, max_chars: int = SUMMARIZE_TRUNCATE_CHARS) -> str:
    """Truncate content with a suffix marker."""
    if len(content) <= max_chars:
        return content
    return content[:max_chars] + " [truncated]"


async def _summarize_message(content: str) -> str:
    """Summarize message content using Haiku with 2s timeout, falling back to truncation."""
    import asyncio
    try:
        from quantstack.llm.routing import get_llm
        llm = get_llm("light")
        prompt = (
            "Summarize the following agent output in 2-3 sentences, preserving "
            "any numerical values, ticker symbols, and directional signals:\n\n"
            + content
        )
        result = await asyncio.wait_for(
            llm.ainvoke(prompt),
            timeout=SUMMARIZE_TIMEOUT_S,
        )
        return result.content if hasattr(result, "content") else str(result)
    except Exception:
        return _truncate_content(content)


def _prune_messages(messages: list) -> list:
    """Priority-aware message pruning.

    Pruning order:
      1. Remove P3 messages (should already be excluded, safety sweep)
      2. Remove P2 messages oldest-first
      3. Truncate P1 messages (sync fallback; async summarization done externally)
      4. P0 messages are never touched

    Keeps system message (index 0) and user message (index 1) unconditionally.
    """
    if _estimate_message_chars(messages) <= MAX_MESSAGE_CHARS:
        return messages

    # Preserve system + user messages
    header = messages[:2]
    body = messages[2:]

    # Phase 1: Remove P3 messages
    body = [m for m in body if _get_message_priority(m) != PRIORITY_P3]
    if _estimate_message_chars(header + body) <= MAX_MESSAGE_CHARS:
        return header + body

    # Phase 2: Remove P2 messages oldest-first
    p2_indices = [i for i, m in enumerate(body) if _get_message_priority(m) == PRIORITY_P2]
    removed: set[int] = set()
    for idx in p2_indices:
        removed.add(idx)
        remaining = header + [m for i, m in enumerate(body) if i not in removed]
        if _estimate_message_chars(remaining) <= MAX_MESSAGE_CHARS:
            break

    if removed:
        body = [m for i, m in enumerate(body) if i not in removed]
        logger.debug("Pruned %d P2 messages from agent context", len(removed))

    if _estimate_message_chars(header + body) <= MAX_MESSAGE_CHARS:
        return header + body

    # Phase 3: Truncate P1 messages (sync fallback — no LLM call in sync path)
    for i, msg in enumerate(body):
        if _get_message_priority(msg) == PRIORITY_P1:
            content = msg.content if hasattr(msg, "content") and isinstance(msg.content, str) else ""
            if len(content) > SUMMARIZE_TRUNCATE_CHARS:
                new_msg = type(msg)(
                    content=_truncate_content(content),
                    **({"tool_call_id": msg.tool_call_id} if isinstance(msg, ToolMessage) else {}),
                )
                if hasattr(msg, "metadata") and isinstance(msg.metadata, dict):
                    new_msg.metadata = {**msg.metadata, "summarized": True}
                body[i] = new_msg
        if _estimate_message_chars(header + body) <= MAX_MESSAGE_CHARS:
            break

    # P0 messages survive unconditionally
    return header + body


def _is_server_tool_call(tool_call: dict) -> bool:
    """Check if a tool_call entry is a server-side tool (tool search) that should be skipped."""
    name = tool_call.get("name", "")
    return name.startswith("tool_search_tool")


def _detect_provider(llm) -> str:
    """Infer provider name from LLM class for cache_control decisions."""
    cls_name = type(llm).__name__
    if cls_name == "ChatAnthropic":
        return "anthropic"
    if cls_name == "ChatBedrock":
        return "bedrock"
    if "openai" in cls_name.lower():
        return "openai"
    return "other"


def build_system_message(
    config: AgentConfig,
    graph_name: str | None = None,
    provider: str | None = None,
) -> SystemMessage:
    """Build a system message from an agent config.

    When tool search is active (config.always_loaded_tools is non-empty),
    appends tool category guidance tailored to the agent's graph.

    For Anthropic/Bedrock providers, returns structured content with
    cache_control breakpoint to enable prompt caching (90% cost reduction).
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

    if provider in ("anthropic", "bedrock"):
        return SystemMessage(content=[{
            "type": "text",
            "text": base,
            "cache_control": {"type": "ephemeral"},
        }])

    return SystemMessage(content=base)


async def run_agent(
    llm: BaseChatModel,
    tools: list[BaseTool],
    config: AgentConfig,
    user_message: str,
    max_rounds: int = MAX_TOOL_ROUNDS,
    _skip_bigtool: bool = False,
    blocked_tools: frozenset[str] = frozenset(),
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
    provider = _detect_provider(llm)
    messages = [
        build_system_message(config, provider=provider),
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

    budget_tracker = TokenBudgetTracker(max_tokens=config.max_tokens_budget)

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

        # Track token usage for budget enforcement
        usage = {}
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            um = response.usage_metadata
            usage = {"input": um.get("input_tokens", 0), "output": um.get("output_tokens", 0)}
            budget_tracker.add_usage(
                input_tokens=um.get("input_tokens", 0),
                output_tokens=um.get("output_tokens", 0),
            )

        # Budget enforcement: stop agent if token budget exceeded
        if budget_tracker.budget_exceeded:
            logger.warning(
                "Agent %s budget exceeded: %d tokens > %d limit. Returning partial result.",
                config.name, budget_tracker.total_tokens, config.max_tokens_budget,
            )
            return json.dumps({
                "budget_exceeded": True,
                "agent": config.name,
                "total_tokens": budget_tracker.total_tokens,
                "max_tokens_budget": config.max_tokens_budget,
                "partial_content": response.content[:1000] if response.content else "",
            })

        # Log LLM call to Langfuse
        log_llm_call(
            agent_name=config.name,
            model_name=model_name,
            input_messages=messages,
            output_content=response.content or "",
            duration_seconds=llm_dur,
            tool_calls=response.tool_calls if response.tool_calls else None,
            usage=usage,
        )

        # Log prompt cache metrics when available (Anthropic/Bedrock only)
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            um = response.usage_metadata
            cache_read = um.get("cache_read_input_tokens", 0)
            cache_creation = um.get("cache_creation_input_tokens", 0)
            if cache_read or cache_creation:
                trace_prompt_cache_metrics(
                    agent_name=config.name,
                    model_name=model_name,
                    cache_read_tokens=cache_read,
                    cache_creation_tokens=cache_creation,
                    total_input_tokens=um.get("input_tokens", 0),
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

            # Tool access control guard (section-08)
            if tool_name in blocked_tools:
                result = json.dumps({
                    "error": "tool_access_denied",
                    "tool": tool_name,
                    "message": f"Tool '{tool_name}' is not available in this graph context.",
                })
                tool_ok = False
                logger.warning(
                    "SECURITY: agent %s attempted blocked tool %s",
                    config.name, tool_name,
                )
                messages.append(ToolMessage(content=result, tool_call_id=tool_id))
                continue

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
        sys_msg = build_system_message(config, provider=_detect_provider(llm))
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


def parse_json_response(
    text: str,
    fallback: dict | list | None = None,
    *,
    agent_name: str = "",
    graph_name: str = "",
    run_id: str = "",
    model_used: str = "",
    prompt_text: str = "",
) -> dict | list:
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

    # Write to Dead Letter Queue if context is available
    if agent_name:
        try:
            import hashlib
            from quantstack.db import db_conn as _dlq_db_conn

            prompt_hash = hashlib.sha256(prompt_text.encode()).hexdigest()[:16] if prompt_text else ""
            input_summary = prompt_text[:500] if prompt_text else ""
            with _dlq_db_conn() as conn:
                conn.execute(
                    "INSERT INTO agent_dlq "
                    "(agent_name, graph_name, run_id, input_summary, raw_output, "
                    "error_type, error_detail, prompt_hash, model_used) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    [agent_name, graph_name, run_id, input_summary,
                     text[:10000], "parse_error", f"Failed to extract JSON from response",
                     prompt_hash, model_used],
                )
        except Exception as dlq_exc:
            logger.debug("DLQ write failed: %s", dlq_exc)

    return fallback if fallback is not None else {}


def parse_and_validate(
    raw_output: str,
    fallback: dict | list | None = None,
    *,
    output_schema: type | None = None,
    agent_name: str = "",
    graph_name: str = "",
    run_id: str = "",
    model_used: str = "",
    prompt_text: str = "",
) -> tuple[dict | list, bool]:
    """Parse LLM output as JSON and optionally validate against a Pydantic schema.

    Returns (parsed_result, was_retried) tuple. The was_retried flag is always
    False in this implementation — retry logic requires the LLM conversation
    context and is handled at the executor level.

    If output_schema is provided:
      1. Parse JSON via parse_json_response()
      2. Validate against schema via model_validate()
      3. On validation failure, return (fallback, False) and log to DLQ

    If output_schema is None, behaves identically to parse_json_response().
    """
    parsed = parse_json_response(
        raw_output,
        fallback,
        agent_name=agent_name,
        graph_name=graph_name,
        run_id=run_id,
        model_used=model_used,
        prompt_text=prompt_text,
    )

    if output_schema is None or parsed is fallback:
        return parsed, False

    try:
        validated = output_schema.model_validate(parsed)
        return validated.model_dump(), False
    except Exception as exc:
        logger.warning(
            "Schema validation failed for %s: %s", agent_name or "unknown", exc
        )
        # Write validation failure to DLQ
        if agent_name:
            try:
                import hashlib
                from quantstack.db import db_conn as _dlq_db_conn

                prompt_hash = (
                    hashlib.sha256(prompt_text.encode()).hexdigest()[:16]
                    if prompt_text
                    else ""
                )
                with _dlq_db_conn() as conn:
                    conn.execute(
                        "INSERT INTO agent_dlq "
                        "(agent_name, graph_name, run_id, input_summary, raw_output, "
                        "error_type, error_detail, prompt_hash, model_used) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        [
                            agent_name,
                            graph_name,
                            run_id,
                            prompt_text[:500] if prompt_text else "",
                            raw_output[:10000],
                            "schema_validation_error",
                            str(exc)[:2000],
                            prompt_hash,
                            model_used,
                        ],
                    )
            except Exception as dlq_exc:
                logger.debug("DLQ write failed: %s", dlq_exc)

        return fallback if fallback is not None else {}, False
