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

import atexit
import importlib
import os
import signal
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from fastmcp import FastMCP
from loguru import logger

from quantstack.config.settings import get_settings
from quantstack.context import create_trading_context
from quantstack.core.features.factory import MultiTimeframeFeatureFactory
from quantstack.data.pg_storage import PgDataStore
from quantstack.data.registry import DataProviderRegistry
from quantstack.mcp._helpers import ServerContext, set_shared_reader
from quantstack.mcp._state import set_ctx
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
    Domain.COORDINATION: [
        "analysis",      # get_system_status, get_portfolio_state, get_regime
        "signal",        # get_signal_brief
        "coordination",  # record_heartbeat, publish_event, poll_events
        "execution",     # execute_trade
    ],
}

# All servers need TradingContext: analysis.py (cross-cutting) uses live_db_or_error
# for get_portfolio_state / get_recent_decisions / get_system_status, and domain-specific
# tools (strategy, backtesting, ml, cross_domain, finrl_tools, etc.) write to the DB.
_TRADING_DOMAINS = (
    Domain.EXECUTION | Domain.PORTFOLIO | Domain.SIGNALS
    | Domain.DATA | Domain.RESEARCH | Domain.OPTIONS
    | Domain.ML | Domain.FINRL | Domain.INTEL | Domain.RISK
    | Domain.COORDINATION
)

# Domains that need heavy ML imports (sklearn, lightgbm, torch, etc.)
_ML_DOMAINS = Domain.ML | Domain.FINRL

# COORDINATION domain allowlist — only these tools are registered.
# The coordination server loads modules from multiple domains but exposes
# a curated subset for hot-path orchestration.  All other computation
# happens via Python imports in Bash (see prompts/reference/python_toolkit.md).
_COORDINATION_TOOLS: frozenset[str] = frozenset({
    "get_system_status",
    "get_portfolio_state",
    "get_regime",
    "get_signal_brief",
    "record_heartbeat",
    "publish_event",
    "poll_events",
    "execute_trade",
})


# ── Domain-aware lifespan ──────────────────────────────────────────────────


def _make_lifespan(target: Domain):
    """Build an async context manager lifespan for the target domain."""

    @asynccontextmanager
    async def _lifespan(server: FastMCP) -> AsyncGenerator[None, None]:
        settings = get_settings()
        logger.info(f"[{server.name}] starting (domain={target})")

        # Trading context — only for domains that need broker/portfolio/risk
        if target & _TRADING_DOMAINS:
            ctx = create_trading_context()
            set_ctx(ctx)
            logger.info(f"[{server.name}] TradingContext initialized")

        # Research infrastructure — needed by most domains for data access
        research_ctx = ServerContext(settings=settings)
        research_ctx.data_store = PgDataStore()
        set_shared_reader(research_ctx.data_store)

        # Feature factory — needed by data/research/options/risk domains
        needs_features = target & (
            Domain.DATA | Domain.RESEARCH | Domain.OPTIONS | Domain.RISK | Domain.SIGNALS
        )
        if needs_features:
            research_ctx.feature_factory = MultiTimeframeFeatureFactory(
                include_rrg=False,
                include_waves=True,
                include_technical_indicators=True,
            )

        # Data registry — needed for data fetching
        if target & (Domain.DATA | Domain.RESEARCH | Domain.SIGNALS):
            research_ctx.data_registry = DataProviderRegistry.from_settings(settings)

        server.context = research_ctx
        logger.info(f"[{server.name}] ready ({_count_tools(server)} tools)")

        yield

        # Clean up DB pool so connections are returned to PostgreSQL
        from quantstack.db import reset_pg_pool

        reset_pg_pool()
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
    2. Imports tool modules mapped to ``target``.
    3. Collects ``TOOLS`` from each module and registers only those
       whose ``@domain()`` tag overlaps with ``target``.

    No monkey-patching, no global state mutation. Import failures
    are isolated — a broken module doesn't corrupt the server.

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

    # Collect tool modules for this domain and register explicitly.
    # No monkey-patching, no global state mutation, no import-time side effects.
    modules_to_import: set[str] = set()
    for domain_flag, module_names in _DOMAIN_MODULES.items():
        if target & domain_flag:
            modules_to_import.update(module_names)

    failed_modules: list[str] = []
    registered_count = 0

    for mod_name in sorted(modules_to_import):
        fqn = f"quantstack.mcp.tools.{mod_name}"
        try:
            mod = importlib.import_module(fqn)
            for tool in getattr(mod, "TOOLS", []):
                # COORDINATION uses an explicit allowlist (tools from multiple domains)
                if target & Domain.COORDINATION:
                    if tool.name not in _COORDINATION_TOOLS:
                        continue
                else:
                    # Standard domain filter: only register tools tagged for this server
                    tool_domain = TOOL_DOMAINS.get(tool.name)
                    if tool_domain is not None and not (tool_domain & target):
                        continue
                server.tool(tool.fn, name=tool.name, description=tool.description)
                registered_count += 1
        except Exception as exc:
            failed_modules.append(mod_name)
            logger.error(
                f"[{name}] tool module {fqn!r} failed to load — "
                f"its tools will NOT be available: {exc}"
            )

    if failed_modules:
        logger.error(
            f"[{name}] {len(failed_modules)} module(s) failed to load: "
            f"{failed_modules}. Check imports in those modules."
        )

    if registered_count == 0:
        logger.error(
            f"[{name}] no tools registered after loading "
            f"{len(modules_to_import)} module(s). "
            f"Server will start but all tool calls will fail."
        )

    return server


# ── PID lockfile — prevents stale orphan processes ─────────────────────────


_PID_DIR = Path("/tmp")


def _kill_stale_and_lock(name: str) -> None:
    """Kill any previously running instance of this server and claim the PID slot.

    stdio MCP servers become orphaned when Claude Code disconnects: the stdio
    pipe closes but the OS process keeps running.  On the next startup Claude
    Code spawns a fresh copy without killing the old one, which piles up stale
    processes over time.

    This runs before ``server.run()`` so the old instance is gone before the
    new one begins serving requests.
    """
    pid_file = _PID_DIR / f"quantstack_{name.replace('-', '_')}.pid"

    if pid_file.exists():
        try:
            old_pid = int(pid_file.read_text().strip())
            if old_pid != os.getpid():
                # Verify the PID is actually a quantstack process before killing.
                # PID reuse on macOS can cause us to kill an unrelated process.
                try:
                    import subprocess

                    result = subprocess.run(
                        ["ps", "-p", str(old_pid), "-o", "command="],
                        capture_output=True,
                        text=True,
                        timeout=2,
                    )
                    if "quantstack" in result.stdout:
                        os.kill(old_pid, signal.SIGTERM)
                        time.sleep(0.3)
                    else:
                        logger.debug(
                            f"[{name}] PID {old_pid} is not a quantstack process, skipping kill"
                        )
                except subprocess.TimeoutExpired:
                    pass
        except (ProcessLookupError, ValueError, OSError):
            pass  # already gone or bad file — ignore

    pid_file.write_text(str(os.getpid()))
    atexit.register(lambda: pid_file.unlink(missing_ok=True))
    logger.debug(f"[{name}] PID {os.getpid()} registered, stale instance cleared")


def run_server(server: "FastMCP") -> None:
    """Kill any stale predecessor, then start the MCP server.

    Call this instead of ``server.run()`` in each domain server's ``main()``.
    """
    _kill_stale_and_lock(server.name)
    server.run()
