"""Data functions called directly by graph nodes."""

import asyncio
from typing import Any

from loguru import logger

from quantstack.agents.regime_detector import RegimeDetectorAgent
from quantstack.tools._state import live_db_or_error, _serialize


async def get_regime(symbol: str = "SPY") -> dict[str, Any]:
    """Get current market regime for a symbol.

    Called by load_context nodes to populate state.regime.
    """
    try:
        detector = RegimeDetectorAgent(symbols=[symbol])
        result = await asyncio.get_event_loop().run_in_executor(
            None, detector.detect_regime, symbol
        )
        return result
    except Exception as e:
        logger.error(f"get_regime({symbol}) failed: {e}")
        return {"success": False, "symbol": symbol, "error": str(e)}


async def get_portfolio_state() -> dict[str, Any]:
    """Get current portfolio state.

    Called by load_context nodes to populate state.portfolio_context.
    """
    ctx, err = live_db_or_error()
    if err:
        return err
    try:
        snapshot = ctx.portfolio.get_snapshot()
        positions = ctx.portfolio.get_positions()
        context_str = ctx.portfolio.as_context_string()
        return {
            "success": True,
            "snapshot": _serialize(snapshot),
            "positions": [_serialize(p) for p in positions],
            "context_string": context_str,
        }
    except Exception as e:
        logger.error(f"get_portfolio_state failed: {e}")
        return {"success": False, "error": str(e)}
