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

import sys
from contextlib import asynccontextmanager
from typing import Any

from fastmcp import FastMCP
from loguru import logger

from quant_pod.context import TradingContext, create_trading_context
from quant_pod.mcp._state import set_ctx, set_degraded


# =============================================================================
# Lifespan
# =============================================================================


@asynccontextmanager
async def lifespan(server: FastMCP):
    """Initialize TradingContext on startup, cleanup on shutdown."""
    logger.info("QuantPod MCP Server starting...")
    try:
        ctx = create_trading_context()
        set_ctx(ctx)
        set_degraded(False)
        logger.info(f"QuantPod MCP Server initialized | session={ctx.session_id}")
    except RuntimeError as exc:
        msg = str(exc)
        if "locked by a running process" in msg or "Stale lock" in msg:
            logger.warning(
                f"[MCP] DB lock conflict — starting in degraded mode. "
                f"Analysis tools work; portfolio/execution tools unavailable. "
                f"Reason: {msg}"
            )
            ctx = create_trading_context(db_path=":memory:")
            set_ctx(ctx)
            set_degraded(True, msg)
        else:
            raise
    yield
    logger.info("QuantPod MCP Server stopped")


# =============================================================================
# FastMCP Singleton
# =============================================================================

mcp = FastMCP(
    name="QuantPod Trading Intelligence",
    instructions=(
        "QuantPod MCP server — the operational interface for the autonomous "
        "trading intelligence system.  Use get_signal_brief for fast deterministic "
        "analysis, get_portfolio_state to inspect holdings, and get_regime "
        "to classify current market conditions."
    ),
    lifespan=lifespan,
)


# =============================================================================
# Tool Registration — importing each module triggers @mcp.tool() registration
# =============================================================================

from quant_pod.mcp.tools.analysis import (  # noqa: E402, F401
    get_portfolio_state, get_regime,
    get_recent_decisions, get_system_status,
)
from quant_pod.mcp.tools.strategy import (  # noqa: E402, F401
    register_strategy, list_strategies, get_strategy, update_strategy,
)
from quant_pod.mcp.tools.backtesting import (  # noqa: E402, F401
    run_backtest, run_backtest_mtf, run_walkforward,
    run_walkforward_mtf, walk_forward_sparse_signal, run_backtest_options,
    _generate_signals_from_rules, _evaluate_rule, _fetch_price_data,
)
from quant_pod.mcp.tools.execution import (  # noqa: E402, F401
    execute_trade, close_position, cancel_order,
    get_fills, get_risk_metrics, get_audit_trail,
)
from quant_pod.mcp.tools.decoder import (  # noqa: E402, F401
    decode_strategy, decode_from_trades,
)
from quant_pod.mcp.tools.meta import (  # noqa: E402, F401
    get_regime_strategies, set_regime_allocation,
    resolve_portfolio_conflicts,
    get_strategy_gaps, promote_draft_strategies,
    check_strategy_rules,
)
from quant_pod.mcp.tools.ml import (  # noqa: E402, F401
    train_ml_model, get_ml_model_status, predict_ml_signal,
)
from quant_pod.mcp.tools.learning import (  # noqa: E402, F401
    get_rl_status, get_rl_recommendation, promote_strategy,
    retire_strategy, get_strategy_performance, validate_strategy,
    update_regime_matrix_from_performance,
)
from quant_pod.mcp.tools.feedback import (  # noqa: E402, F401
    get_fill_quality, get_position_monitor,
)
from quant_pod.mcp.tools.signal import (  # noqa: E402, F401
    get_signal_brief, run_multi_signal_brief,
)
from quant_pod.mcp.tools.intraday import (  # noqa: E402, F401
    get_intraday_status, get_tca_report, get_algo_recommendation,
)
from quant_pod.mcp.tools.portfolio import (  # noqa: E402, F401
    optimize_portfolio, compute_hrp_weights,
)
from quant_pod.mcp.tools.nlp import analyze_text_sentiment  # noqa: E402, F401


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
