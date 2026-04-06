"""Measure per-agent tool schema token overhead.

Loads all agent configs across all three graphs, resolves their tool sets,
serializes schemas to the Anthropic API format, and estimates token counts.

Usage:
    uv run python scripts/measure_tool_token_overhead.py
"""

import json
from pathlib import Path

from quantstack.graphs.config import load_agent_configs
from quantstack.tools.registry import TOOL_REGISTRY, get_tools_for_agent

GRAPH_CONFIGS = {
    "trading": Path("src/quantstack/graphs/trading/config/agents.yaml"),
    "research": Path("src/quantstack/graphs/research/config/agents.yaml"),
    "supervisor": Path("src/quantstack/graphs/supervisor/config/agents.yaml"),
}

# Rough token estimation: ~4 chars per token for JSON schema text
CHARS_PER_TOKEN = 4


def tool_to_anthropic_schema(tool) -> dict:
    """Convert a LangChain tool to the Anthropic API tool schema format."""
    schema = tool.get_input_schema().model_json_schema()
    # Remove internal keys that aren't sent to the API
    schema.pop("title", None)
    return {
        "name": tool.name,
        "description": tool.description or "",
        "input_schema": schema,
    }


def estimate_tokens(text: str) -> int:
    """Rough token estimate from character count."""
    return len(text) // CHARS_PER_TOKEN


def main():
    print(f"Total tools in registry: {len(TOOL_REGISTRY)}")
    print()

    max_overhead = 0
    rows = []

    for graph_name, yaml_path in GRAPH_CONFIGS.items():
        configs = load_agent_configs(yaml_path)
        for agent_name, config in configs.items():
            if not config.tools:
                rows.append((agent_name, graph_name, 0, 0))
                continue

            try:
                tools = get_tools_for_agent(config.tools)
            except KeyError as e:
                print(f"  WARNING: {agent_name} has invalid tool ref: {e}")
                continue

            schemas = [tool_to_anthropic_schema(t) for t in tools]
            schema_json = json.dumps(schemas, indent=2)
            token_est = estimate_tokens(schema_json)
            max_overhead = max(max_overhead, token_est)
            rows.append((agent_name, graph_name, len(tools), token_est))

    # Sort by token overhead descending
    rows.sort(key=lambda r: r[3], reverse=True)

    # Print table
    print(f"{'Agent':<30} {'Graph':<12} {'# Tools':>8} {'Schema Tokens':>14}")
    print("-" * 68)
    for agent_name, graph_name, n_tools, tokens in rows:
        print(f"{agent_name:<30} {graph_name:<12} {n_tools:>8} {tokens:>14,}")

    print()
    print(f"Max per-agent tool overhead: ~{max_overhead:,} tokens")
    print()

    if max_overhead < 5000:
        print("GO/NO-GO: MARGINAL ROI")
        print(
            f"Highest per-agent tool overhead is ~{max_overhead:,} tokens (<5K threshold)."
        )
        print("Tool search deferred loading may not provide meaningful savings.")
        print("Consider whether the complexity is worth the marginal token reduction.")
    else:
        print("GO/NO-GO: PROCEED")
        print(
            f"Highest per-agent tool overhead is ~{max_overhead:,} tokens (>5K threshold)."
        )
        print("Deferred loading should provide meaningful token savings.")


if __name__ == "__main__":
    main()
