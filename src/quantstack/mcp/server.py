# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
QuantPod MCP Server — slim hub.

Defines the FastMCP ``mcp`` singleton and ``lifespan``, then imports tool
modules so their ``@mcp.tool()`` decorators register automatically.

No ``_PatchedFastMCP`` or ``auto_release_db`` — those wrappers existed solely
to release DuckDB's exclusive file lock between tool calls.  PostgreSQL handles
concurrent access natively so no per-tool lock management is needed.

Usage:
    quantpod-mcp   (via pyproject.toml entry point)
    python -m quantstack.mcp.server
"""

import sys
from contextlib import asynccontextmanager

from fastmcp import FastMCP
from loguru import logger

from quantstack.config.settings import get_settings
# context (~1.1s) and features.factory (~0.8s) deferred to lifespan — only needed after handshake.
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

    from quantstack.context import create_trading_context  # noqa: PLC0415
    from quantstack.core.features.factory import MultiTimeframeFeatureFactory  # noqa: PLC0415

    # PostgreSQL never holds exclusive file locks — startup always succeeds
    # as long as the PostgreSQL daemon is running.
    ctx = create_trading_context()
    set_ctx(ctx)
    logger.info(f"Trading context initialized | session={ctx.session_id}")

    # --- Research infrastructure (PgDataStore, FeatureFactory, DataRegistry) ---
    settings = get_settings()
    research_ctx = ServerContext(settings=settings)

    # PgDataStore is stateless — no persistent connection to open or fail.
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
# FastMCP Singleton
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
from quantstack.mcp.tools.alerts import (  # noqa: E402, F401
    create_equity_alert,
    get_equity_alerts,
    update_alert_status,
    create_exit_signal,
    add_alert_update,
)
from quantstack.mcp.tools.cross_domain import (  # noqa: E402, F401
    get_cross_domain_intel,
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

# --- Institutional-grade bottom detection tools ---
import quantstack.mcp.tools.capitulation  # noqa: E402, F401
import quantstack.mcp.tools.institutional_accumulation  # noqa: E402, F401
import quantstack.mcp.tools.macro_signals  # noqa: E402, F401

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
