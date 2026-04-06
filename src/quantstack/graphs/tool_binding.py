"""Consolidated tool binding for LangGraph agents.

Replaces the duplicated _bind_tools_to_llm() that was in each graph file.
Supports three modes:
- Anthropic direct API: server-side BM25 tool search with defer_loading
- Bedrock/other providers: langgraph-bigtool with pgvector semantic search
- No always_loaded_tools: backward-compatible full loading
"""

import logging
from typing import Any

from langchain_core.tools import BaseTool

from quantstack.graphs.config import AgentConfig
from quantstack.tools.registry import get_tools_for_agent, get_tools_for_agent_with_search

logger = logging.getLogger(__name__)


def _supports_native_deferred_loading(llm: Any) -> bool:
    """Check if the LLM provider supports Anthropic's defer_loading beta."""
    return type(llm).__name__ == "ChatAnthropic"


def bind_tools_to_llm(
    llm: Any,
    config: AgentConfig,
) -> tuple[Any, list[BaseTool], bool]:
    """Bind tools from agent config to the LLM.

    Three paths:
    1. Anthropic direct API + always_loaded_tools: server-side BM25 with defer_loading
    2. Other providers + always_loaded_tools: bigtool mode (signaled via return)
    3. No always_loaded_tools: full loading (backward compat)

    Returns:
        (bound_llm, tool_list, fallback_mode)
        - bound_llm: LLM with tools bound (or unbound LLM in bigtool mode)
        - tool_list: All configured tools for execution
        - fallback_mode: True if deferred loading failed and we fell back
    """
    if not config.tools:
        return llm, [], False

    # Path 1: Anthropic direct API — server-side BM25 tool search
    if config.always_loaded_tools and _supports_native_deferred_loading(llm):
        try:
            tools_for_api, tools_for_execution = get_tools_for_agent_with_search(
                config.tools, config.always_loaded_tools,
            )
            bound_llm = llm.bind(tools=tools_for_api)
            n_deferred = sum(
                1 for t in tools_for_api
                if isinstance(t, dict) and t.get("defer_loading") is True
            )
            logger.info(
                "Agent '%s' bound %d tools (%d deferred, %d always-loaded) [anthropic-native]",
                config.name, len(tools_for_execution), n_deferred,
                len(tools_for_execution) - n_deferred,
            )
            return bound_llm, tools_for_execution, False
        except Exception as exc:
            logger.warning(
                "Deferred loading failed for agent '%s': %s — falling back to full loading",
                config.name, exc,
            )
            return _bind_all_tools(llm, config, fallback=True)

    # Path 2: Non-Anthropic provider with always_loaded_tools — use bigtool
    # Don't bind tools here; run_agent() will use the bigtool subgraph.
    if config.always_loaded_tools:
        tools_for_execution = get_tools_for_agent(config.tools)
        logger.info(
            "Agent '%s' configured for bigtool mode (%d tools, %d always-loaded) [%s]",
            config.name, len(config.tools), len(config.always_loaded_tools),
            type(llm).__name__,
        )
        return llm, tools_for_execution, False

    # Path 3: No always_loaded_tools — full loading
    return _bind_all_tools(llm, config, fallback=False)


def _bind_all_tools(
    llm: Any,
    config: AgentConfig,
    fallback: bool,
) -> tuple[Any, list[BaseTool], bool]:
    """Backward-compatible path: load all configured tools without deferred loading."""
    try:
        tools = get_tools_for_agent(config.tools)
        bound_llm = llm.bind_tools(tools)
        logger.info(
            "Agent '%s' bound %d tools: %s",
            config.name, len(tools), [t.name for t in tools],
        )
        return bound_llm, tools, fallback
    except (KeyError, Exception) as exc:
        logger.warning(
            "Failed to bind tools for agent '%s': %s — running without tools",
            config.name, exc,
        )
        return llm, [], fallback
