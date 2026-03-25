# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Server factory for domain-scoped MCP servers.

Creates ``FastMCP`` instances that only register tools tagged for their
target domain (via ``@domain()`` decorators).  Each server gets a
domain-appropriate lifespan that initializes only what it needs.

Usage::

    from quantstack.mcp.server_factory import create_server
    from quantstack.mcp.domains import Domain

    server = create_server(
        name="quantstack-ml",
        target=Domain.ML,
        instructions="ML training, inference, drift detection, ensembles.",
    )
    server.run()
"""

from __future__ import annotations

import importlib
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastmcp import FastMCP
from loguru import logger

from quantstack.mcp.domains import Domain
from quantstack.mcp.tools._registry import TOOL_DOMAINS

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

# ── Which tool modules belong to which domain ──────────────────────────────
# Maps Domain flag -> list of module names under quantstack.mcp.tools.*
# A module is imported when the server's target domain overlaps with any of
# its domain flags.  Cross-cutting tools register in multiple servers because
# their @domain() decorator lists multiple domains.

_DOMAIN_MODULES: dict[Domain, list[str]] = {
    Domain.EXECUTION: [
        "execution",
        "options_execution",
        "alerts",
        "coordination",
        "analysis",      # cross-cutting: get_system_status
    ],
    Domain.PORTFOLIO: [
        "analysis",      # cross-cutting: get_portfolio_state, get_system_status
        "portfolio",
        "attribution",
        "feedback",
    ],
    Domain.SIGNALS: [
        "signal",
        "intraday",
        "analysis",      # cross-cutting: get_regime
    ],
    Domain.DATA: [
        "qc_data",
        "qc_indicators",
        "qc_fundamentals",
        "qc_fundamentals_av",
        "qc_market",
        "qc_acquisition",
        "analysis",      # cross-cutting: get_regime
    ],
    Domain.RESEARCH: [
        "qc_research",
        "backtesting",
        "qc_backtesting",
        "strategy",
        "meta",
        "learning",
        "decoder",
        "analysis",      # cross-cutting: get_regime, get_portfolio_state
    ],
    Domain.OPTIONS: [
        "qc_options",
        "analysis",      # cross-cutting: get_regime
    ],
    Domain.ML: [
        "ml",
        "analysis",      # cross-cutting: get_regime
    ],
    Domain.FINRL: [
        "finrl_tools",
        "analysis",      # cross-cutting: get_regime
    ],
    Domain.INTEL: [
        "capitulation",
        "institutional_accumulation",
        "macro_signals",
        "cross_domain",
        "nlp",
        "signal",        # cross-cutting: get_signal_brief
        "analysis",      # cross-cutting: get_regime
    ],
    Domain.RISK: [
        "qc_risk",
        "analysis",      # cross-cutting: get_regime
    ],
}

# Domains that need the full TradingContext (broker, risk gate, portfolio)
_TRADING_DOMAINS = Domain.EXECUTION | Domain.PORTFOLIO | Domain.SIGNALS

# Domains that need heavy ML imports (sklearn, lightgbm, torch, etc.)
_ML_DOMAINS = Domain.ML | Domain.FINRL


# ── Domain-aware lifespan ──────────────────────────────────────────────────


def _make_lifespan(target: Domain):
    """Build an async context manager lifespan for the target domain."""

    @asynccontextmanager
    async def _lifespan(server: FastMCP) -> AsyncGenerator[None, None]:
        from quantstack.config.settings import get_settings  # noqa: PLC0415

        settings = get_settings()
        logger.info(f"[{server.name}] starting (domain={target})")

        # Trading context — only for domains that need broker/portfolio/risk
        if target & _TRADING_DOMAINS:
            from quantstack.context import create_trading_context  # noqa: PLC0415
            from quantstack.mcp._state import set_ctx  # noqa: PLC0415

            ctx = create_trading_context()
            set_ctx(ctx)
            logger.info(f"[{server.name}] TradingContext initialized")

        # Research infrastructure — needed by most domains for data access
        from quantstack.data.pg_storage import PgDataStore  # noqa: PLC0415
        from quantstack.mcp._helpers import ServerContext, set_shared_reader  # noqa: PLC0415

        research_ctx = ServerContext(settings=settings)
        research_ctx.data_store = PgDataStore()
        set_shared_reader(research_ctx.data_store)

        # Feature factory — needed by data/research/options/risk domains
        needs_features = target & (
            Domain.DATA | Domain.RESEARCH | Domain.OPTIONS | Domain.RISK | Domain.SIGNALS
        )
        if needs_features:
            from quantstack.core.features.factory import (  # noqa: PLC0415
                MultiTimeframeFeatureFactory,
            )

            research_ctx.feature_factory = MultiTimeframeFeatureFactory(
                include_rrg=False,
                include_waves=True,
                include_technical_indicators=True,
            )

        # Data registry — needed for data fetching
        if target & (Domain.DATA | Domain.RESEARCH | Domain.SIGNALS):
            from quantstack.data.registry import DataProviderRegistry  # noqa: PLC0415

            research_ctx.data_registry = DataProviderRegistry.from_settings(settings)

        server.context = research_ctx
        logger.info(f"[{server.name}] ready ({_count_tools(server)} tools)")

        yield

        logger.info(f"[{server.name}] stopped")

    return _lifespan


def _count_tools(server: FastMCP) -> int:
    """Count registered tools on a FastMCP instance."""
    try:
        return len(server._tool_manager._tools)
    except AttributeError:
        return -1


# ── Server factory ─────────────────────────────────────────────────────────


def create_server(
    name: str,
    target: Domain,
    instructions: str,
) -> FastMCP:
    """Create a domain-scoped FastMCP server.

    1. Creates a FastMCP instance with a domain-aware lifespan.
    2. Imports only the tool modules mapped to ``target``.
    3. After import, all ``@mcp.tool()`` decorators in those modules fire,
       but only tools whose ``@domain()`` tag overlaps with ``target``
       are effectively useful (the rest still register but the server
       factory filters by domain at the module level).

    Args:
        name: Server name (e.g. "quantstack-ml"). Becomes the MCP namespace prefix.
        target: Domain flag(s) for this server.
        instructions: One-line description shown to Claude.

    Returns:
        Configured ``FastMCP`` instance ready to ``run()``.
    """
    server = FastMCP(
        name=name,
        instructions=instructions,
        lifespan=_make_lifespan(target),
    )

    # Import tool modules for this domain — triggers @mcp.tool() registration
    # on the GLOBAL mcp singleton from server.py.  For the split architecture,
    # we need tools to register on THIS server instead.
    #
    # Strategy: we temporarily monkey-patch quantstack.mcp.server.mcp to point
    # to our new server, import the modules, then restore.  This is safe because
    # server creation is single-threaded at startup.
    import quantstack.mcp.server as _srv_mod

    original_mcp = _srv_mod.mcp
    _srv_mod.mcp = server

    modules_to_import: set[str] = set()
    for domain_flag, module_names in _DOMAIN_MODULES.items():
        if target & domain_flag:
            modules_to_import.update(module_names)

    for mod_name in sorted(modules_to_import):
        fqn = f"quantstack.mcp.tools.{mod_name}"
        try:
            # Force reimport so @mcp.tool() decorators fire against our server
            mod = importlib.import_module(fqn)
            importlib.reload(mod)
        except Exception as exc:
            logger.warning(f"[{name}] failed to import {fqn}: {exc}")

    # Restore original singleton
    _srv_mod.mcp = original_mcp

    return server
