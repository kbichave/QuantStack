"""Shared tool-calling agent executor for LangGraph nodes.

Runs a tool-calling loop: LLM decides to call tools → tools execute →
results fed back → LLM continues until it produces a final text response
(no more tool calls).

This replaces the previous pattern where nodes just did a single LLM chat
call with no tool access.
"""

import json
import logging
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

logger = logging.getLogger(__name__)

MAX_TOOL_ROUNDS = 10  # Safety cap to prevent infinite tool-calling loops
MAX_TOOL_RESULT_CHARS = 4000  # Truncate individual tool results beyond this
MAX_MESSAGE_CHARS = 150_000  # Rough char budget (~37k tokens) before pruning old messages


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
            # Start of a new tool round — flush previous if it was just tool results
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


def build_system_message(config: AgentConfig) -> SystemMessage:
    """Build a system message from an agent config."""
    return SystemMessage(content=(
        f"You are a {config.role}.\n\n"
        f"Goal: {config.goal}\n\n"
        f"Background: {config.backstory}\n\n"
        "You have access to tools. Use them to gather real data before reasoning.\n"
        "When you have enough information, respond with your final answer as valid JSON.\n"
        "Do NOT fabricate data — always call the appropriate tool first."
    ))


async def run_agent(
    llm: BaseChatModel,
    tools: list[BaseTool],
    config: AgentConfig,
    user_message: str,
    max_rounds: int = MAX_TOOL_ROUNDS,
) -> str:
    """Run a tool-calling agent loop.

    Returns the final text response from the LLM after it finishes
    calling tools and produces its answer.

    Args:
        llm: Chat model (already has tools bound if any).
        tools: List of tool objects for execution.
        config: Agent config for system message.
        user_message: The task/prompt for this node.
        max_rounds: Max tool-calling rounds before forcing a text response.

    Returns:
        The final text content from the LLM.
    """
    from quantstack.dashboard.events import publish_event

    tool_map = {t.name: t for t in tools}
    messages = [
        build_system_message(config),
        HumanMessage(content=user_message),
    ]

    # Infer graph team from agent name
    _AGENT_TEAMS = {
        "quant_researcher": "research", "ml_scientist": "research",
        "position_monitor": "trading", "entry_scanner": "trading",
        "exit_manager": "trading", "risk_assessor": "trading",
        "daily_planner": "trading", "fund_manager": "trading",
        "options_analyst": "trading", "trade_debater": "trading",
        "reflector": "trading", "execution_manager": "trading",
        "health_monitor": "supervisor", "diagnostician": "supervisor",
        "self_healer": "supervisor", "strategy_promoter": "supervisor",
        "scheduler": "supervisor",
    }
    graph_name = _AGENT_TEAMS.get(config.name, "other")

    publish_event(
        graph_name=graph_name,
        node_name=config.name,
        agent_name=config.name,
        event_type="agent_start",
        content=user_message[:500],
    )

    for round_num in range(max_rounds):
        response: AIMessage = await llm.ainvoke(messages)
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

            if tool_name not in tool_map:
                result = json.dumps({"error": f"Unknown tool: {tool_name}"})
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

    # Exhausted rounds — ask LLM to produce final answer without tools
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


def parse_json_response(text: str, fallback: dict | None = None) -> dict:
    """Parse a JSON **object** from LLM response, handling common issues.

    Always returns a dict.  If the LLM response contains a JSON array or a
    non-dict value, the fallback dict is returned instead (callers rely on
    ``.get()``).  Returns fallback if parsing fails entirely.
    """
    if not text:
        return fallback or {}

    def _try_parse(s: str) -> dict | None:
        try:
            val = json.loads(s)
            return val if isinstance(val, dict) else None
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

    logger.debug("Failed to parse JSON from response: %.200s", text)
    return fallback or {}
