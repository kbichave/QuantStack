"""Manual validation script: test server_tool_use handling with deferred tools.

Makes a live API call with a deferred tool and the BM25 search tool to verify
that langchain-anthropic properly strips server_tool_use from AIMessage.tool_calls.

Usage:
    uv run python scripts/validate_server_tool_use.py

Requires ANTHROPIC_API_KEY or AWS Bedrock credentials.
"""

import asyncio
import json
from pathlib import Path

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage


# BM25 tool search definition (Anthropic built-in)
TOOL_SEARCH_TOOL = {
    "type": "tool_search_bm25_2025_04_15",
    "name": "tool_search_tool",
    "max_results": 5,
}

# A simple deferred tool in Anthropic API format
DEFERRED_TOOL = {
    "name": "signal_brief",
    "description": (
        "Generate a trading signal brief for a given ticker symbol. "
        "Includes technical indicators, regime classification, and signal strength."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "symbol": {
                "type": "string",
                "description": "Ticker symbol (e.g., AAPL, TSLA)",
            }
        },
        "required": ["symbol"],
    },
    "defer_loading": True,
}


async def main():
    llm = ChatAnthropic(model="claude-sonnet-4-20250514")

    # Bind tools: one deferred + the BM25 search tool
    bound = llm.bind_tools([DEFERRED_TOOL, TOOL_SEARCH_TOOL])

    prompt = "What is the current signal brief for AAPL?"
    print(f"Prompt: {prompt}")
    print()

    response = await bound.ainvoke([HumanMessage(content=prompt)])

    # Inspect response
    finding = {
        "langchain_anthropic_version": None,
        "tool_calls": [],
        "content_block_types": [],
        "server_tool_use_in_tool_calls": False,
        "server_tool_use_in_content": False,
        "additional_kwargs_keys": list(response.additional_kwargs.keys()),
    }

    try:
        from importlib.metadata import version
        finding["langchain_anthropic_version"] = version("langchain-anthropic")
    except Exception:
        pass

    # Check tool_calls
    for tc in response.tool_calls:
        finding["tool_calls"].append({
            "name": tc.get("name"),
            "type": tc.get("type", "unknown"),
        })
        if tc.get("name", "").startswith("tool_search_tool"):
            finding["server_tool_use_in_tool_calls"] = True

    # Check content blocks
    if isinstance(response.content, list):
        for block in response.content:
            if isinstance(block, dict):
                block_type = block.get("type", "unknown")
                finding["content_block_types"].append(block_type)
                if block_type == "server_tool_use":
                    finding["server_tool_use_in_content"] = True

    # Write findings
    fixture_path = Path("tests/_fixtures/server_tool_use_shape.json")
    fixture_path.parent.mkdir(parents=True, exist_ok=True)
    fixture_path.write_text(json.dumps(finding, indent=2) + "\n")

    print("=== Response ===")
    print(f"tool_calls: {response.tool_calls}")
    print(f"content types: {finding['content_block_types']}")
    print(f"additional_kwargs keys: {finding['additional_kwargs_keys']}")
    print()
    print(f"server_tool_use in tool_calls: {finding['server_tool_use_in_tool_calls']}")
    print(f"server_tool_use in content: {finding['server_tool_use_in_content']}")
    print()
    print(f"Findings written to {fixture_path}")


if __name__ == "__main__":
    asyncio.run(main())
