# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
QuantPod MCP Server — slim hub.

Defines the FastMCP ``mcp`` singleton and ``lifespan``, then imports tool
modules so their ``@mcp.tool()`` decorators register automatically.

Usage:
    quantpod-mcp   (via pyproject.toml entry point)
    python -m quant_pod.mcp.server
"""

import functools
import sys
from contextlib import asynccontextmanager
from typing import Any

from fastmcp import FastMCP
from loguru import logger

from quantstack.config.settings import get_settings
from quantstack.context import TradingContext, create_trading_context
from quantstack.core.features.factory import MultiTimeframeFeatureFactory
from quantstack.data.registry import DataProviderRegistry
from quantstack.data.storage import DataStore
from quantstack.mcp._helpers import ServerContext, set_shared_reader
from quantstack.mcp._state import auto_release_db, set_ctx, set_degraded


# =============================================================================
# Lifespan
# =============================================================================


@asynccontextmanager
async def lifespan(server: FastMCP):
    """Initialize TradingContext + research infrastructure on startup."""
    logger.info("QuantPod MCP Server starting...")

    # --- Trading context (execution, portfolio, signals) ---
    try:
        ctx = create_trading_context()
        set_ctx(ctx)
        set_degraded(False)
        logger.info(f"Trading context initialized | session={ctx.session_id}")
    except RuntimeError as exc:
        msg = str(exc)
        if "lock" in msg.lower():
            logger.warning(f"[MCP] DB lock conflict — degraded mode. {msg}")
            ctx = create_trading_context(db_path=":memory:")
            set_ctx(ctx)
            set_degraded(True, msg)
        else:
            raise

    # --- Research infrastructure (DataStore, FeatureFactory, DataRegistry) ---
    settings = get_settings()
    research_ctx = ServerContext(settings=settings)

    try:
        writer = DataStore()
        writer.close()
    except RuntimeError as exc:
        logger.warning(f"DuckDB write lock conflict during schema init — OK. {exc}")

    try:
        research_ctx.data_store = DataStore(read_only=True)
        set_shared_reader(research_ctx.data_store)
        logger.info("DataStore opened read-only for research tools.")
    except Exception as ro_exc:
        logger.error(f"Read-only DataStore failed: {ro_exc}.")
        research_ctx.data_store = None

    research_ctx.feature_factory = MultiTimeframeFeatureFactory(
        include_rrg=False,
        include_waves=True,
        include_technical_indicators=True,
    )

    research_ctx.data_registry = DataProviderRegistry.from_settings(settings)
    server.context = research_ctx
    logger.info("Research infrastructure initialized")

    yield

    if research_ctx.data_store:
        research_ctx.data_store.close()
    logger.info("QuantPod MCP Server stopped")


# =============================================================================
# FastMCP Singleton — patched to auto-release DB after every tool call
# =============================================================================

_FastMCP = FastMCP


class _PatchedFastMCP(_FastMCP):
    """FastMCP subclass that wraps every tool with ``auto_release_db``.

    After each tool function returns, the DuckDB file lock is released so
    other processes can access the database between tool calls.
    """

    def tool(self, name_or_fn=None, **kwargs):
        # Case 1: @mcp.tool  (no parens, fn passed directly)
        if callable(name_or_fn):
            fn = name_or_fn
            wrapped = auto_release_db(fn)
            functools.update_wrapper(wrapped, fn)
            super().tool(wrapped, **kwargs)
            return wrapped  # return the callable, not the FunctionTool

        # Case 2: @mcp.tool() or @mcp.tool("name") — returns a decorator
        decorator = super().tool(name_or_fn, **kwargs)

        def patched_decorator(fn):
            wrapped = auto_release_db(fn)
            functools.update_wrapper(wrapped, fn)
            decorator(wrapped)
            return wrapped  # return the callable, not the FunctionTool

        return patched_decorator


mcp = _PatchedFastMCP(
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


# =============================================================================
# Tool Registration — importing each module triggers @mcp.tool() registration
# =============================================================================

from quantstack.mcp.tools.analysis import (  # noqa: E402, F401
    get_portfolio_state,
    get_regime,
    get_recent_decisions,
    get_system_status,
)
from quantstack.mcp.tools.strategy import (  # noqa: E402, F401
    register_strategy,
    list_strategies,
    get_strategy,
    update_strategy,
)
from quantstack.mcp.tools.backtesting import (  # noqa: E402, F401
    run_backtest,
    run_backtest_mtf,
    run_walkforward,
    run_walkforward_mtf,
    walk_forward_sparse_signal,
    run_backtest_options,
)
from quantstack.mcp.tools.execution import (  # noqa: E402, F401
    execute_trade,
    close_position,
    cancel_order,
    get_fills,
    get_risk_metrics,
    get_audit_trail,
)
from quantstack.mcp.tools.decoder import (  # noqa: E402, F401
    decode_strategy,
    decode_from_trades,
)
from quantstack.mcp.tools.meta import (  # noqa: E402, F401
    get_regime_strategies,
    set_regime_allocation,
    resolve_portfolio_conflicts,
    get_strategy_gaps,
    promote_draft_strategies,
    check_strategy_rules,
)
from quantstack.mcp.tools.ml import (  # noqa: E402, F401
    train_ml_model,
    get_ml_model_status,
    predict_ml_signal,
)
from quantstack.mcp.tools.learning import (  # noqa: E402, F401
    promote_strategy,
    retire_strategy,
    get_strategy_performance,
    validate_strategy,
    update_regime_matrix_from_performance,
)
from quantstack.mcp.tools.attribution import (  # noqa: E402, F401
    get_daily_equity,
    get_strategy_pnl,
)
import quantstack.mcp.tools.finrl_tools  # noqa: E402, F401
from quantstack.mcp.tools.feedback import (  # noqa: E402, F401
    get_fill_quality,
    get_position_monitor,
)
from quantstack.mcp.tools.signal import (  # noqa: E402, F401
    get_signal_brief,
    run_multi_signal_brief,
)
from quantstack.mcp.tools.intraday import (  # noqa: E402, F401
    get_intraday_status,
    get_tca_report,
    get_algo_recommendation,
)
from quantstack.mcp.tools.portfolio import (  # noqa: E402, F401
    optimize_portfolio,
    compute_hrp_weights,
)
from quantstack.mcp.tools.nlp import analyze_text_sentiment  # noqa: E402, F401
import quantstack.mcp.tools.coordination  # noqa: E402, F401

# --- Research tools (formerly quantcore MCP) ---
import quantstack.mcp.tools.qc_data  # noqa: E402, F401
import quantstack.mcp.tools.qc_indicators  # noqa: E402, F401
import quantstack.mcp.tools.qc_backtesting  # noqa: E402, F401
import quantstack.mcp.tools.qc_research  # noqa: E402, F401
import quantstack.mcp.tools.qc_options  # noqa: E402, F401
import quantstack.mcp.tools.qc_risk  # noqa: E402, F401
import quantstack.mcp.tools.qc_market  # noqa: E402, F401

# Conditional research tools — only register if API keys configured
_settings = get_settings()
if getattr(getattr(_settings, "financial_datasets", None), "api_key", None):
    import quantstack.mcp.tools.qc_fundamentals  # noqa: E402, F401
if _settings.alpha_vantage_api_key:
    import quantstack.mcp.tools.qc_fundamentals_av  # noqa: E402, F401
    import quantstack.mcp.tools.qc_acquisition  # noqa: E402, F401


# =============================================================================
# Entry Point
# =============================================================================


def main():
    """Run the QuantPod MCP server."""
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
        level="INFO",
    )
    mcp.run()


if __name__ == "__main__":
    main()
