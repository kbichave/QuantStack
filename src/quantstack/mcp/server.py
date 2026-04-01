# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
QuantStack MCP Server — monolith entry point.

Registers ALL tool modules on a single FastMCP server using explicit
tool collection (no monkey-patching, no import-time side effects).

Usage:
    quantstack-mcp   (via pyproject.toml entry point)
    python -m quantstack.mcp.server
"""

import importlib
import sys

from loguru import logger

from quantstack.config.settings import get_settings
from quantstack.mcp._app import mcp

# All tool modules to register on the monolith server.
_ALL_TOOL_MODULES = [
    "analysis",
    "strategy",
    "backtesting",
    "execution",
    "decoder",
    "meta",
    "ml",
    "learning",
    "attribution",
    "alerts",
    "cross_domain",
    "finrl_tools",
    "feedback",
    "signal",
    "intraday",
    "portfolio",
    "nlp",
    "coordination",
    "capitulation",
    "institutional_accumulation",
    "macro_signals",
    "qc_data",
    "qc_indicators",
    "qc_backtesting",
    "qc_research",
    "qc_options",
    "qc_risk",
    "qc_market",
    "options_execution",
]


def _register_all_tools() -> None:
    """Import all tool modules and register their tools on the mcp singleton."""
    settings = get_settings()

    # Conditional modules — only load if API keys configured
    conditional_modules: list[str] = []
    if getattr(getattr(settings, "financial_datasets", None), "api_key", None):
        conditional_modules.append("qc_fundamentals")
    if settings.alpha_vantage_api_key:
        conditional_modules.extend(["qc_fundamentals_av", "qc_acquisition"])

    for mod_name in _ALL_TOOL_MODULES + conditional_modules:
        fqn = f"quantstack.mcp.tools.{mod_name}"
        try:
            mod = importlib.import_module(fqn)
            for tool in getattr(mod, "TOOLS", []):
                mcp.tool(tool.fn, name=tool.name, description=tool.description)
        except Exception as exc:
            logger.error(f"[server] {fqn!r} failed to load: {exc}")


_register_all_tools()


# =============================================================================
# Entry Point
# =============================================================================


def main():
    """Run the QuantStack MCP server."""
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
        level="INFO",
    )
    mcp.run()


if __name__ == "__main__":
    main()
