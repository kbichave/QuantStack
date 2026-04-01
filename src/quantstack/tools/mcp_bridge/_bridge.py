# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""MCPBridge class and async helper for CrewAI-to-MCP communication."""

import asyncio
import concurrent.futures
import importlib.util
from typing import Any

from loguru import logger

try:
    from etrade_mcp.server import mcp as etrade_mcp
except ImportError:
    etrade_mcp = None

from quantstack.guardrails.mcp_response_validator import get_mcp_validator

try:
    from quantstack.mcp.server import mcp as quantcore_mcp
except ImportError:
    quantcore_mcp = None


class MCPBridge:
    """
    Bridge between CrewAI agents and MCP servers.

    Handles in-process communication with:
    - QuantCore MCP (technical analysis, backtesting, options, risk)
    - eTrade MCP (trading, account management)

    Accepts MCP server objects via constructor to avoid upward imports
    (tools L10 → mcp L11). Callers at the orchestration layer wire the
    servers in.
    """

    def __init__(
        self,
        quantcore_mcp: Any | None = None,
        etrade_mcp: Any | None = None,
    ):
        """Initialize MCP bridge with optional server references.

        Args:
            quantcore_mcp: The QuantCore MCP server object (has tool functions as attrs).
            etrade_mcp: The eTrade MCP server object.
        """
        self._quantcore_mcp = quantcore_mcp
        self._etrade_mcp = etrade_mcp
        self._quantcore_available = quantcore_mcp is not None
        self._etrade_available = etrade_mcp is not None
        # Lazy-init validator to avoid circular imports at module load time
        self._validator = None

    def _get_validator(self):
        """Lazily load the MCP response validator."""
        if self._validator is None:
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

        try:
            validator = self._get_validator()
            if validator is None:
                return result
            return validator.validate(tool_name, result)
        except Exception as exc:
            logger.warning(f"Validation failed for {tool_name}: {exc}")
            return {
                "error": f"Validation failed: {exc}",
                "validation_failed": True,
                "original_response": result,
            }

    async def call_quantcore(self, tool_name: str, **kwargs) -> dict[str, Any]:
        """Call a QuantCore MCP tool."""
        if not self._quantcore_available:
            return {"error": "QuantCore MCP not available"}

        try:
            tool_func = getattr(self._quantcore_mcp, tool_name, None)
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
            tool_func = getattr(self._etrade_mcp, tool_name, None)
            if tool_func is None:
                return {"error": f"Tool {tool_name} not found in eTrade MCP"}
            result = await tool_func(**kwargs)
            return self._validate_response(tool_name, result)
        except Exception as e:
            logger.error(f"eTrade MCP call failed: {e}")
            return {"error": str(e)}

    async def call_tool(self, server: str, tool_name: str, **kwargs) -> dict[str, Any]:
        """
        Unified tool calling interface.

        Args:
            server: Which MCP server to use ("quantcore" or "etrade")
            tool_name: Name of the tool to call
            **kwargs: Tool arguments
        """
        if server == "quantcore":
            return await self.call_quantcore(tool_name, **kwargs)
        elif server == "etrade":
            return await self.call_etrade(tool_name, **kwargs)
        else:
            return {"error": f"Unknown MCP server: {server}"}


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

_bridge: MCPBridge | None = None


def get_bridge() -> MCPBridge:
    """Get or create the global MCP bridge instance.

    Auto-discovers available MCP servers at first call. Callers above L10
    (autonomous, mcp layers) can import the servers and pass them here.
    """
    global _bridge
    if _bridge is None:
        # Auto-discover servers — the import happens here at L10+,
        # only when a caller actually needs the bridge.
        quantcore = None
        etrade = None

        if importlib.util.find_spec("quantstack.mcp.server") is not None:
            quantcore = quantcore_mcp
            logger.info("QuantCore MCP server available")

        if importlib.util.find_spec("etrade_mcp.server") is not None:
            etrade = etrade_mcp
            logger.info("eTrade MCP server available")

        _bridge = MCPBridge(quantcore_mcp=quantcore, etrade_mcp=etrade)
    return _bridge


def _run_async(coro):
    """Helper to run async coroutine synchronously."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)
