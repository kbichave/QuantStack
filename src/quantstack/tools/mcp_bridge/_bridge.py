# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""MCPBridge class and async helper for CrewAI-to-MCP communication."""

import asyncio
from typing import Any

from loguru import logger


class MCPBridge:
    """
    Bridge between CrewAI agents and MCP servers.

    Handles in-process communication with:
    - QuantCore MCP (technical analysis, backtesting, options, risk)
    - eTrade MCP (trading, account management)
    """

    def __init__(self):
        """Initialize MCP bridge."""
        self._quantcore_available = False
        self._etrade_available = False
        self._check_servers()
        # Lazy-init validator to avoid circular imports at module load time
        self._validator = None

    def _get_validator(self):
        """Lazily load the MCP response validator."""
        if self._validator is None:
            from quantstack.guardrails.mcp_response_validator import get_mcp_validator

            self._validator = get_mcp_validator()
        return self._validator

    def _validate_response(
        self, tool_name: str, result: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Validate a raw MCP tool response before returning it to agent context.

        Routes to the appropriate typed validator based on tool name patterns.
        Returns the original result (or an error dict) after logging violations.
        Validation is non-blocking: a failed validation logs a warning and
        returns {"error": ..., "validation_failed": True} so callers can
        decide to fall back or reject the trade.
        """
        if not isinstance(result, dict) or "error" in result:
            # Pass-through errors without re-validating
            return result

        validator = self._get_validator()

        # Route to typed validators based on tool name
        name_lower = tool_name.lower()

        if "quote" in name_lower:
            vr = validator.validate_quote_response(result)
        elif any(x in name_lower for x in ("option", "greeks", "implied_vol")):
            vr = validator.validate_options_response(result)
        elif any(x in name_lower for x in ("position", "balance", "account")):
            vr = validator.validate_portfolio_response(result)
        elif any(x in name_lower for x in ("ohlcv", "market_data", "snapshot")):
            vr = validator.validate_ohlcv_response(
                result, symbol=result.get("symbol", "UNKNOWN")
            )
        else:
            vr = validator.validate_generic_response(result, tool_name)

        if not vr.is_valid:
            logger.warning(
                f"[MCPBridge] Validation FAILED for tool={tool_name}: "
                f"{[str(v) for v in vr.violations]}"
            )
            return {
                "error": f"MCP response validation failed for {tool_name}",
                "validation_failed": True,
                "violations": [str(v) for v in vr.violations],
                "original_response": result,
            }

        return result

    def _check_servers(self) -> None:
        """Check which tool clients are available."""
        try:
            from quantstack.mcp.server import mcp as quantcore_mcp  # noqa: F401

            self._quantcore_available = True
            logger.info("QuantCore MCP server available")
        except ImportError:
            logger.warning("QuantCore MCP server not available")

        try:
            from etrade_mcp.server import mcp as etrade_mcp  # noqa: F401

            self._etrade_available = True
            logger.info("eTrade MCP server available")
        except ImportError:
            logger.warning("eTrade MCP server not available")

    async def call_quantcore(self, tool_name: str, **kwargs) -> dict[str, Any]:
        """Call a QuantCore MCP tool."""
        if not self._quantcore_available:
            return {"error": "QuantCore MCP not available"}

        try:
            from quantstack.mcp.server import mcp as quantcore_mcp

            tool_func = getattr(quantcore_mcp, tool_name, None)
            if tool_func is None:
                return {"error": f"Tool {tool_name} not found in QuantCore MCP"}
            result = await tool_func(**kwargs)
            return self._validate_response(tool_name, result)
        except Exception as e:
            logger.error(f"QuantCore MCP call failed: {e}")
            return {"error": str(e)}

    async def call_etrade(self, tool_name: str, **kwargs) -> dict[str, Any]:
        """Call an eTrade MCP tool in-process (same pattern as call_quantcore)."""
        if not self._etrade_available:
            return {"error": "eTrade MCP not available"}

        try:
            from etrade_mcp.server import mcp as etrade_mcp

            tool_func = getattr(etrade_mcp, tool_name, None)
            if tool_func is None:
                return {"error": f"Tool {tool_name} not found in eTrade MCP"}
            result = await tool_func(**kwargs)
            return self._validate_response(tool_name, result)
        except Exception as e:
            logger.error(f"eTrade MCP call failed ({tool_name}): {e}")
            return {"error": str(e)}


# Global bridge instance
_bridge: MCPBridge | None = None


def get_bridge() -> MCPBridge:
    """Get or create the global MCP bridge instance."""
    global _bridge
    if _bridge is None:
        _bridge = MCPBridge()
    return _bridge


def _run_async(coro):
    """Helper to run async coroutine synchronously."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)
