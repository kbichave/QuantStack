# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
FastMCP singleton + lifespan — isolated from tool registration.

Tools import ``mcp`` from here. ``server.py`` imports ``mcp`` from here and
then registers tools by importing each tool module. This breaks the cycle:

    server.py  →  tools/*  →  server.py   (BEFORE: circular)
    server.py  →  tools/*  →  _app.py     (AFTER:  no cycle)

Nothing that _app.py imports at module level may import from server.py or
from any tool module.
"""

import sys
from contextlib import asynccontextmanager

from fastmcp import FastMCP
from loguru import logger

from quantstack.config.settings import get_settings
from quantstack.context import create_trading_context
from quantstack.core.features.factory import MultiTimeframeFeatureFactory
from quantstack.data.pg_storage import PgDataStore
from quantstack.data.registry import DataProviderRegistry
from quantstack.mcp._helpers import ServerContext, set_shared_reader
from quantstack.mcp._state import set_ctx


# =============================================================================
# Lifespan
# =============================================================================


@asynccontextmanager
async def lifespan(server: FastMCP):
    """Initialize TradingContext + research infrastructure on startup."""
    logger.info("QuantPod MCP Server starting...")

    ctx = create_trading_context()
    set_ctx(ctx)
    logger.info(f"Trading context initialized | session={ctx.session_id}")

    settings = get_settings()
    research_ctx = ServerContext(settings=settings)

    research_ctx.data_store = PgDataStore()
    set_shared_reader(research_ctx.data_store)
    logger.info("PgDataStore initialized for research tools.")

    research_ctx.feature_factory = MultiTimeframeFeatureFactory(
        include_rrg=False,
        include_waves=True,
        include_technical_indicators=True,
    )

    research_ctx.data_registry = DataProviderRegistry.from_settings(settings)
    server.context = research_ctx
    logger.info("Research infrastructure initialized")

    yield

    logger.info("QuantPod MCP Server stopped")


# =============================================================================
# FastMCP singleton
# =============================================================================

mcp = FastMCP(
    name="QuantPod",
    instructions=(
        "QuantPod MCP server — unified quantitative trading platform. "
        "Research tools (200+ indicators, backtesting, options pricing, ML) "
        "and operational tools (signals, execution, portfolio, strategy management). "
        "Use get_signal_brief for analysis, get_portfolio_state for holdings, "
        "get_regime for market conditions."
    ),
    lifespan=lifespan,
)
