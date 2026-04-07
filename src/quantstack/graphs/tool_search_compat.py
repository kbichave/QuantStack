"""Tool Search compatibility validation.

Provides a startup assertion that verifies the installed langchain-anthropic
supports defer_loading and the BM25 tool search tool type.

Implementation note: langchain-anthropic 0.3.22's bind_tools() does NOT
recognize the tool_search_bm25 type as a builtin tool (its _is_builtin_tool
only knows text_editor_, computer_, bash_, web_search_, etc.). However,
llm.bind(tools=...) passes tool dicts through without conversion, and
convert_to_anthropic_tool() preserves the defer_loading field on regular tools.

The tool search path uses llm.bind(tools=formatted_tools) directly,
bypassing bind_tools(). This avoids requiring a cascading upgrade to
langchain-anthropic 1.x.

Called at graph initialization before any agent uses tool search.
"""

import logging

from langchain_anthropic.chat_models import convert_to_anthropic_tool

from quantstack.llm.provider import get_chat_model
from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)

TOOL_SEARCH_TOOL = {
    "type": "tool_search_bm25_2025_04_15",
    "name": "tool_search_tool",
    "max_results": 5,
}

_SAMPLE_DEFERRED_TOOL = {
    "name": "_compat_check",
    "description": "Compatibility check tool -- never actually called.",
    "input_schema": {
        "type": "object",
        "properties": {"x": {"type": "string"}},
    },
    "defer_loading": True,
}


def tool_to_anthropic_dict(tool_obj: BaseTool, defer: bool = False) -> dict:
    """Convert a LangChain BaseTool to an Anthropic API tool dict.

    Uses langchain-anthropic's convert_to_anthropic_tool for correct
    schema formatting, then optionally adds the defer_loading flag.
    """
    result = convert_to_anthropic_tool(tool_obj)
    if defer:
        result["defer_loading"] = True
    return result


def validate_tool_search_support() -> None:
    """Assert that the installed langchain-anthropic supports tool search features.

    Uses llm.bind(tools=...) which passes dicts through without conversion,
    since bind_tools() doesn't recognize tool_search_bm25 as a builtin type.

    Raises RuntimeError if:
    - convert_to_anthropic_tool rejects defer_loading field
    - llm.bind() rejects the tool dict format
    """
    # Step 1: Verify convert_to_anthropic_tool preserves defer_loading
    try:
        converted = convert_to_anthropic_tool(_SAMPLE_DEFERRED_TOOL)
        if "defer_loading" not in converted:
            raise RuntimeError(
                "langchain-anthropic's convert_to_anthropic_tool strips defer_loading. "
                "Upgrade to a version that preserves extra fields on tool dicts."
            )
    except (TypeError, KeyError) as exc:
        raise RuntimeError(
            f"langchain-anthropic does not support defer_loading on tool dicts. "
            f"Error: {exc}"
        ) from exc

    # Step 2: Verify llm.bind() accepts both deferred tools and tool search type
    llm = get_chat_model("heavy")
    try:
        llm.bind(tools=[_SAMPLE_DEFERRED_TOOL, TOOL_SEARCH_TOOL])
    except Exception as exc:
        raise RuntimeError(
            f"llm.bind() rejected tool search tool dicts: {exc}"
        ) from exc

    logger.info("Tool search compatibility validated: defer_loading and BM25 search supported")
