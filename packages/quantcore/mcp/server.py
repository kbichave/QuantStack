# Copyright 2024 QuantCore Contributors
# SPDX-License-Identifier: Apache-2.0

"""
QuantCore MCP Server — slim hub.

Defines the FastMCP ``mcp`` singleton and ``lifespan``, then imports tool
modules so their ``@mcp.tool()`` decorators register automatically.

Usage:
    python -m quantcore.mcp.server
"""

import sys
from contextlib import asynccontextmanager
from typing import Any

from fastmcp import FastMCP
from loguru import logger

from quantcore.config.settings import get_settings
from quantcore.mcp._helpers import ServerContext, set_shared_reader


# =============================================================================
# Lifespan
# =============================================================================


@asynccontextmanager
async def lifespan(server: FastMCP):
    """Initialize and cleanup server resources."""
    logger.info("QuantCore MCP Server starting...")

    settings = get_settings()
    ctx = ServerContext(settings=settings)

    from quantcore.data.storage import DataStore
    from quantcore.features.factory import MultiTimeframeFeatureFactory

    # Short-lived write to ensure schema exists, then release lock.
    try:
        writer = DataStore()
        writer.close()
        logger.info("DuckDB schema initialized via short-lived write connection.")
    except RuntimeError as exc:
        logger.warning(
            f"DuckDB write lock conflict during schema init — OK, another process owns it. {exc}"
        )

    try:
        ctx.data_store = DataStore(read_only=True)
        set_shared_reader(ctx.data_store)
        logger.info("QuantCore DataStore opened read-only (no lock contention).")
    except Exception as ro_exc:
        logger.error(f"Read-only DataStore failed: {ro_exc}. DataStore unavailable.")
        ctx.data_store = None

    ctx.feature_factory = MultiTimeframeFeatureFactory(
        include_rrg=False,
        include_waves=True,
        include_technical_indicators=True,
    )

    from quantcore.data.registry import DataProviderRegistry

    ctx.data_registry = DataProviderRegistry.from_settings(settings)

    server.context = ctx
    logger.info("QuantCore MCP Server initialized")

    yield

    # Cleanup
    if ctx.data_store:
        ctx.data_store.close()
    logger.info("QuantCore MCP Server stopped")


# =============================================================================
# FastMCP Singleton
# =============================================================================

mcp = FastMCP(
    name="QuantCore Trading Platform",
    instructions="Quantitative trading research platform with 200+ technical indicators, "
    "backtesting, options pricing, and ML integration.",
    lifespan=lifespan,
)


# =============================================================================
# Tool Registration — importing each module triggers @mcp.tool() registration
# =============================================================================

import quantcore.mcp.tools.data  # noqa: E402, F401
import quantcore.mcp.tools.indicators  # noqa: E402, F401
import quantcore.mcp.tools.backtesting  # noqa: E402, F401
import quantcore.mcp.tools.research  # noqa: E402, F401
import quantcore.mcp.tools.options  # noqa: E402, F401
import quantcore.mcp.tools.risk  # noqa: E402, F401
import quantcore.mcp.tools.market  # noqa: E402, F401
import quantcore.mcp.tools.fundamentals  # noqa: E402, F401
import quantcore.mcp.resources  # noqa: E402, F401


# =============================================================================
# Entry Point
# =============================================================================


def main():
    """Run the MCP server."""
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
        level="INFO",
    )
    mcp.run()


if __name__ == "__main__":
    main()
