# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
QuantPod MCP Server — exposes the trading system to Claude Code.

Phase 1 tools (read-only):
  - run_analysis      — run TradingCrew in stop-at-assistant mode, return DailyBrief
  - get_portfolio_state — current positions, cash, equity
  - get_regime        — market regime classification for a symbol
  - get_recent_decisions — recent audit trail entries
  - get_system_status — kill switch, risk halt, broker mode

Usage:
    quantpod-mcp   (via pyproject.toml entry point)
    python -m quant_pod.mcp.server
"""

import asyncio
import sys
import time
from contextlib import asynccontextmanager
from typing import Any

from fastmcp import FastMCP
from loguru import logger

from quant_pod.context import TradingContext, create_trading_context
from quant_pod.execution.broker_factory import get_broker_mode

# =============================================================================
# Server State
# =============================================================================

_ctx: TradingContext | None = None

# When the DB write lock is held by another process, the server starts in
# degraded mode using an in-memory context.  Tools that need persistent DB
# state return a structured error; analysis tools (no DB dependency) work normally.
_degraded_mode: bool = False
_degraded_reason: str = ""

# ---------------------------------------------------------------------------
# IC Output Cache — keyed "{symbol}::{ic_name}", 30-minute TTL.
# Populated by run_analysis after every full crew run, and by run_ic/run_pod.
# ---------------------------------------------------------------------------
_ic_output_cache: dict[str, Any] = {}
_IC_CACHE_TTL_SECS = 1800  # 30 minutes


def _ic_cache_set(symbol: str, ic_name: str, output: str) -> None:
    """Cache raw IC output text for later retrieval."""
    _ic_output_cache[f"{symbol}::{ic_name}"] = {
        "output": output,
        "ts": time.monotonic(),
    }


def _ic_cache_get(symbol: str, ic_name: str) -> str | None:
    """Retrieve cached IC output, returning None if absent or expired."""
    entry = _ic_output_cache.get(f"{symbol}::{ic_name}")
    if not entry:
        return None
    if time.monotonic() - entry["ts"] > _IC_CACHE_TTL_SECS:
        del _ic_output_cache[f"{symbol}::{ic_name}"]
        return None
    return entry["output"]


def _populate_ic_cache_from_result(symbol: str, result: Any) -> None:
    """Extract and cache per-IC outputs from a full crew result (best-effort)."""
    try:
        if not hasattr(result, "tasks_output") or not result.tasks_output:
            return
        from quant_pod.crews.trading_crew import IC_AGENT_ORDER

        for i, ic_name in enumerate(IC_AGENT_ORDER):
            if i >= len(result.tasks_output):
                break
            task_out = result.tasks_output[i]
            raw = task_out.raw if hasattr(task_out, "raw") else str(task_out)
            if raw:
                _ic_cache_set(symbol, ic_name, raw)
    except Exception as exc:
        logger.debug(f"[quantpod_mcp] IC cache population failed (non-critical): {exc}")


def _require_ctx() -> TradingContext:
    """Get the trading context, raising if the server hasn't started."""
    if _ctx is None:
        raise RuntimeError("QuantPod MCP server not initialized — call lifespan first")
    return _ctx


def _require_live_db() -> TradingContext:
    """
    Get the trading context, raising if in degraded mode (DB locked).

    Call this from any tool that reads or writes persistent DB state
    (portfolio, fills, audit trail, strategies, regime matrix).

    Tools that are purely computational — run_analysis, get_regime,
    get_system_status — should call _require_ctx() instead so they
    continue to work even when the write lock is held by another process.
    """
    if _degraded_mode:
        raise RuntimeError(
            f"QuantPod is running in degraded mode — the persistent DB is locked. "
            f"Portfolio state and trade execution are unavailable. "
            f"Analysis tools (run_analysis, get_regime) still work. "
            f"Reason: {_degraded_reason}"
        )
    return _require_ctx()


def _live_db_or_error() -> tuple["TradingContext | None", "dict | None"]:
    """
    Convenience wrapper for tool handlers that need a live DB connection.

    Returns (ctx, None) on success, or (None, error_dict) when in degraded
    mode.  Avoids a try/except at every call site:

        ctx, err = _live_db_or_error()
        if err:
            return err
        # proceed with ctx
    """
    try:
        return _require_live_db(), None
    except RuntimeError as exc:
        return None, {"success": False, "error": str(exc), "degraded_mode": True}


# =============================================================================
# Lifespan
# =============================================================================


@asynccontextmanager
async def lifespan(server: FastMCP):
    """Initialize TradingContext on startup, cleanup on shutdown."""
    global _ctx, _degraded_mode, _degraded_reason
    logger.info("QuantPod MCP Server starting...")
    try:
        _ctx = create_trading_context()
        _degraded_mode = False
        _degraded_reason = ""
        logger.info(f"QuantPod MCP Server initialized | session={_ctx.session_id}")
    except RuntimeError as exc:
        msg = str(exc)
        if "locked by a running process" in msg or "Stale lock" in msg:
            logger.warning(
                f"[MCP] DB lock conflict — starting in degraded mode. "
                f"Analysis tools work; portfolio/execution tools unavailable. "
                f"Reason: {msg}"
            )
            _ctx = create_trading_context(db_path=":memory:")
            _degraded_mode = True
            _degraded_reason = msg
        else:
            raise  # Non-lock error (migration failure etc.) — crash loudly
    yield
    logger.info("QuantPod MCP Server stopped")


# =============================================================================
# Server
# =============================================================================

mcp = FastMCP(
    name="QuantPod Trading Intelligence",
    instructions=(
        "QuantPod MCP server — the operational interface for the autonomous "
        "trading intelligence system.  Use run_analysis to commission crew "
        "analysis, get_portfolio_state to inspect holdings, and get_regime "
        "to classify current market conditions."
    ),
    lifespan=lifespan,
)


# =============================================================================
# Serialization Helpers
# =============================================================================


def _serialize(obj: Any) -> Any:
    """Convert Pydantic models, dataclasses, and datetime to JSON-safe dicts."""
    if obj is None:
        return None
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    if hasattr(obj, "__dataclass_fields__"):
        from dataclasses import asdict

        return asdict(obj)
    return obj


# =============================================================================
# TOOL 1: run_analysis
# =============================================================================


@mcp.tool()
async def run_analysis(
    symbol: str,
    regime: dict[str, Any] | None = None,
    include_historical_context: bool = True,
) -> dict[str, Any]:
    """
    Run TradingCrew analysis for a symbol and return a DailyBrief.

    The crew runs all ICs (data, technicals, quant, risk, market monitor),
    Pod Managers compile their findings, and the Trading Assistant synthesizes
    a structured DailyBrief.  The SuperTrader is NOT invoked — Claude Code
    acts as the decision maker.

    Args:
        symbol: Ticker symbol (e.g., "SPY", "AAPL").
        regime: Pre-computed regime dict.  If None, regime is detected
                automatically using ADX/ATR indicators.
        include_historical_context: Whether to load blackboard history
                                    as context for the crew.

    Returns:
        Dict with keys: success, daily_brief, regime_used, elapsed_seconds, error.
    """
    ctx = _require_ctx()
    start = time.monotonic()

    try:
        # 1. Detect regime if not provided
        if regime is None:
            from quant_pod.agents.regime_detector import RegimeDetectorAgent

            detector = RegimeDetectorAgent(symbols=[symbol])
            regime_result = await asyncio.get_event_loop().run_in_executor(
                None, detector.detect_regime, symbol
            )
            if regime_result.get("success"):
                regime = {
                    "trend": regime_result.get("trend_regime", "unknown"),
                    "volatility": regime_result.get("volatility_regime", "normal"),
                    "confidence": regime_result.get("confidence", 0.5),
                }
            else:
                regime = {"trend": "unknown", "volatility": "normal", "confidence": 0.5}

        # 2. Build portfolio context
        portfolio = _serialize(ctx.portfolio.get_snapshot())

        # 3. Load historical context from blackboard
        historical_context = ""
        if include_historical_context:
            historical_context = ctx.blackboard.read_as_context(symbol=symbol, limit=10)

        # 4. Run crew in stop-at-assistant mode (sync call in thread pool)
        from quant_pod.crews.trading_crew import run_analysis_only

        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: run_analysis_only(
                symbol=symbol,
                regime=regime,
                portfolio=portfolio,
                historical_context=historical_context,
            ),
        )

        # 5. Extract DailyBrief from crew result
        brief = None
        if hasattr(result, "pydantic") and result.pydantic is not None:
            brief = _serialize(result.pydantic)
        elif hasattr(result, "json_dict") and result.json_dict is not None:
            brief = result.json_dict
        elif isinstance(result, dict):
            brief = result
        else:
            brief = {"raw_output": str(result)}

        # 6. Populate per-IC output cache (non-blocking best-effort)
        _populate_ic_cache_from_result(symbol, result)

        elapsed = time.monotonic() - start
        return {
            "success": True,
            "daily_brief": brief,
            "regime_used": regime,
            "elapsed_seconds": round(elapsed, 2),
        }

    except Exception as e:
        elapsed = time.monotonic() - start
        logger.error(f"[quantpod_mcp] run_analysis({symbol}) failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "regime_used": regime or {},
            "elapsed_seconds": round(elapsed, 2),
        }


# =============================================================================
# TOOL 2: get_portfolio_state
# =============================================================================


@mcp.tool()
async def get_portfolio_state() -> dict[str, Any]:
    """
    Return the current portfolio state: positions, cash, equity, and P&L.

    Returns:
        Dict with keys: snapshot, positions, context_string.
        - snapshot: cash, positions_value, total_equity, daily_pnl, etc.
        - positions: list of open positions with symbol, quantity, avg_cost, etc.
        - context_string: human-readable markdown summary.
    """
    ctx, err = _live_db_or_error()
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
        logger.error(f"[quantpod_mcp] get_portfolio_state failed: {e}")
        return {"success": False, "error": str(e)}


# =============================================================================
# TOOL 3: get_regime
# =============================================================================


@mcp.tool()
async def get_regime(symbol: str) -> dict[str, Any]:
    """
    Detect the current market regime for a symbol.

    Uses ADX for trend strength/direction and ATR percentile for volatility.
    Deterministic — no LLM calls.

    Args:
        symbol: Ticker symbol (e.g., "SPY").

    Returns:
        Dict with keys: success, symbol, trend_regime, volatility_regime,
        confidence, adx, atr, atr_percentile, error.
    """
    try:
        from quant_pod.agents.regime_detector import RegimeDetectorAgent

        detector = RegimeDetectorAgent(symbols=[symbol])
        result = await asyncio.get_event_loop().run_in_executor(
            None, detector.detect_regime, symbol
        )
        return result
    except Exception as e:
        logger.error(f"[quantpod_mcp] get_regime({symbol}) failed: {e}")
        return {"success": False, "symbol": symbol, "error": str(e)}


# =============================================================================
# TOOL 4: get_recent_decisions
# =============================================================================


@mcp.tool()
async def get_recent_decisions(
    symbol: str | None = None,
    limit: int = 20,
) -> dict[str, Any]:
    """
    Query recent audit trail entries.

    Args:
        symbol: Filter by ticker symbol.  None returns all symbols.
        limit: Maximum number of entries to return.

    Returns:
        Dict with keys: decisions (list of summaries), total.
    """
    ctx, err = _live_db_or_error()
    if err:
        return err
    try:
        from quant_pod.audit.models import AuditQuery

        query = AuditQuery(symbol=symbol or "", limit=limit)
        events = ctx.audit.query(query)
        summaries = [
            {
                "event_id": e.event_id,
                "event_type": e.event_type,
                "agent_name": e.agent_name,
                "symbol": e.symbol,
                "action": e.action,
                "confidence": e.confidence,
                "output_summary": e.output_summary[:200] if e.output_summary else "",
                "created_at": e.created_at.isoformat() if e.created_at else None,
            }
            for e in events
        ]
        return {"success": True, "decisions": summaries, "total": len(summaries)}
    except Exception as e:
        logger.error(f"[quantpod_mcp] get_recent_decisions failed: {e}")
        return {"success": False, "error": str(e), "decisions": [], "total": 0}


# =============================================================================
# TOOL 5: get_system_status
# =============================================================================


@mcp.tool()
async def get_system_status() -> dict[str, Any]:
    """
    Return system health: kill switch state, risk halt, broker mode, session ID.

    Returns:
        Dict with keys: kill_switch_active, kill_switch_reason, risk_halted,
        broker_mode, session_id.
    """
    ctx = _require_ctx()
    try:
        ks_status = ctx.kill_switch.status()
        return {
            "success": True,
            "kill_switch_active": ks_status.active,
            "kill_switch_reason": ks_status.reason,
            "risk_halted": ctx.risk_gate.is_halted(),
            "broker_mode": get_broker_mode(),
            "session_id": ctx.session_id,
        }
    except Exception as e:
        logger.error(f"[quantpod_mcp] get_system_status failed: {e}")
        return {"success": False, "error": str(e)}


# =============================================================================
# PHASE 2: Strategy Registry CRUD
# =============================================================================


@mcp.tool()
async def register_strategy(
    name: str,
    parameters: dict[str, Any],
    entry_rules: list[dict[str, Any]],
    exit_rules: list[dict[str, Any]],
    description: str = "",
    asset_class: str = "equities",
    regime_affinity: dict[str, float] | None = None,
    risk_params: dict[str, Any] | None = None,
    source: str = "manual",
) -> dict[str, Any]:
    """
    Register a new strategy in the persistent catalog.

    Args:
        name: Unique human-readable name.
        parameters: Indicator settings (e.g., {"rsi_period": 14}).
        entry_rules: List of rule dicts (e.g., [{"indicator": "rsi_14", "condition": "crosses_below", "value": 30}]).
        exit_rules: List of rule dicts for exit conditions.
        description: Free-text strategy description.
        asset_class: "equities", "options", "futures", "fx_crypto".
        regime_affinity: Map of regime → suitability score (0-1).
        risk_params: Sizing/stop config (e.g., {"stop_loss_atr": 2.0}).
        source: "manual", "decoded", "workshop", "generated".

    Returns:
        Dict with strategy_id and status.
    """
    import json
    import uuid

    ctx, err = _live_db_or_error()
    if err:
        return err
    strategy_id = f"strat_{uuid.uuid4().hex[:12]}"

    try:
        ctx.db.execute(
            """
            INSERT INTO strategies
                (strategy_id, name, description, asset_class, regime_affinity,
                 parameters, entry_rules, exit_rules, risk_params, status, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'draft', ?)
            """,
            [
                strategy_id,
                name,
                description,
                asset_class,
                json.dumps(regime_affinity or {}),
                json.dumps(parameters),
                json.dumps(entry_rules),
                json.dumps(exit_rules),
                json.dumps(risk_params or {}),
                source,
            ],
        )
        logger.info(f"[quantpod_mcp] Registered strategy {strategy_id}: {name}")
        return {"success": True, "strategy_id": strategy_id, "status": "draft"}
    except Exception as e:
        logger.error(f"[quantpod_mcp] register_strategy failed: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool()
async def list_strategies(
    status: str | None = None,
    asset_class: str | None = None,
) -> dict[str, Any]:
    """
    List strategies from the registry, optionally filtered.

    Args:
        status: Filter by status (draft, backtested, forward_testing, live, failed, retired).
        asset_class: Filter by asset class.

    Returns:
        Dict with list of strategy summaries.
    """
    ctx, err = _live_db_or_error()
    if err:
        return err
    try:
        conditions = []
        params = []
        if status:
            conditions.append("status = ?")
            params.append(status)
        if asset_class:
            conditions.append("asset_class = ?")
            params.append(asset_class)

        where = f" WHERE {' AND '.join(conditions)}" if conditions else ""
        rows = ctx.db.execute(
            f"SELECT strategy_id, name, description, asset_class, status, source, "
            f"created_at, updated_at FROM strategies{where} ORDER BY updated_at DESC",
            params,
        ).fetchall()

        strategies = [
            {
                "strategy_id": r[0],
                "name": r[1],
                "description": r[2],
                "asset_class": r[3],
                "status": r[4],
                "source": r[5],
                "created_at": str(r[6]) if r[6] else None,
                "updated_at": str(r[7]) if r[7] else None,
            }
            for r in rows
        ]
        return {"success": True, "strategies": strategies, "total": len(strategies)}
    except Exception as e:
        logger.error(f"[quantpod_mcp] list_strategies failed: {e}")
        return {"success": False, "error": str(e), "strategies": [], "total": 0}


@mcp.tool()
async def get_strategy(
    strategy_id: str | None = None,
    name: str | None = None,
) -> dict[str, Any]:
    """
    Get full strategy details by ID or name.

    Args:
        strategy_id: Strategy UUID.
        name: Strategy name (used if strategy_id is None).

    Returns:
        Full strategy record.
    """
    import json as _json

    ctx, err = _live_db_or_error()
    if err:
        return err
    try:
        if strategy_id:
            row = ctx.db.execute(
                "SELECT * FROM strategies WHERE strategy_id = ?", [strategy_id]
            ).fetchone()
        elif name:
            row = ctx.db.execute("SELECT * FROM strategies WHERE name = ?", [name]).fetchone()
        else:
            return {"success": False, "error": "Provide strategy_id or name"}

        if row is None:
            return {"success": False, "error": "Strategy not found"}

        cols = [
            "strategy_id",
            "name",
            "description",
            "asset_class",
            "regime_affinity",
            "parameters",
            "entry_rules",
            "exit_rules",
            "risk_params",
            "backtest_summary",
            "walkforward_summary",
            "status",
            "source",
            "created_at",
            "updated_at",
            "created_by",
        ]
        record = {}
        for i, col in enumerate(cols):
            val = row[i]
            if isinstance(val, str) and col in (
                "regime_affinity",
                "parameters",
                "entry_rules",
                "exit_rules",
                "risk_params",
                "backtest_summary",
                "walkforward_summary",
            ):
                try:
                    val = _json.loads(val)
                except (ValueError, TypeError):
                    pass
            if col in ("created_at", "updated_at") and val is not None:
                val = str(val)
            record[col] = val

        return {"success": True, "strategy": record}
    except Exception as e:
        logger.error(f"[quantpod_mcp] get_strategy failed: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool()
async def update_strategy(
    strategy_id: str,
    status: str | None = None,
    description: str | None = None,
    parameters: dict[str, Any] | None = None,
    entry_rules: list[dict[str, Any]] | None = None,
    exit_rules: list[dict[str, Any]] | None = None,
    risk_params: dict[str, Any] | None = None,
    regime_affinity: dict[str, float] | None = None,
    backtest_summary: dict[str, Any] | None = None,
    walkforward_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Update fields of an existing strategy.

    Only non-None fields are updated. Pass the fields you want to change.

    Args:
        strategy_id: Strategy to update.
        status: New status (draft, backtested, forward_testing, live, failed, retired).
        description: Updated description.
        parameters: Updated indicator parameters.
        entry_rules: Updated entry rules.
        exit_rules: Updated exit rules.
        risk_params: Updated risk parameters.
        regime_affinity: Updated regime affinity map.
        backtest_summary: Backtest results to store.
        walkforward_summary: Walk-forward results to store.

    Returns:
        Updated strategy record.
    """
    import json as _json

    ctx, err = _live_db_or_error()
    if err:
        return err
    try:
        sets = []
        params = []
        field_map = {
            "status": status,
            "description": description,
        }
        json_fields = {
            "parameters": parameters,
            "entry_rules": entry_rules,
            "exit_rules": exit_rules,
            "risk_params": risk_params,
            "regime_affinity": regime_affinity,
            "backtest_summary": backtest_summary,
            "walkforward_summary": walkforward_summary,
        }

        for col, val in field_map.items():
            if val is not None:
                sets.append(f"{col} = ?")
                params.append(val)

        for col, val in json_fields.items():
            if val is not None:
                sets.append(f"{col} = ?")
                params.append(_json.dumps(val))

        if not sets:
            return {"success": False, "error": "No fields to update"}

        sets.append("updated_at = CURRENT_TIMESTAMP")
        params.append(strategy_id)

        ctx.db.execute(
            f"UPDATE strategies SET {', '.join(sets)} WHERE strategy_id = ?",
            params,
        )
        logger.info(f"[quantpod_mcp] Updated strategy {strategy_id}")

        # Return the updated record
        return await get_strategy.fn(strategy_id=strategy_id)
    except Exception as e:
        logger.error(f"[quantpod_mcp] update_strategy failed: {e}")
        return {"success": False, "error": str(e)}


# =============================================================================
# PHASE 2: Backtesting Tools
# =============================================================================


def _generate_signals_from_rules(
    price_data: "pd.DataFrame",  # noqa: F821
    entry_rules: list[dict[str, Any]],
    exit_rules: list[dict[str, Any]],
    parameters: dict[str, Any],
) -> "pd.DataFrame":  # noqa: F821
    """
    Generate a signals DataFrame from strategy rules + price data.

    This is a vectorized signal generator that interprets rule dicts into
    indicator computations and condition checks.  Supports common patterns:
      - SMA crossovers
      - RSI overbought/oversold
      - Breakout (close > N-bar high)
      - Mean reversion (z-score)

    Returns DataFrame with 'signal' (0/1) and 'signal_direction' (LONG/SHORT/NONE).
    """
    import pandas as pd

    df = price_data.copy()
    len(df)

    # Pre-compute common indicators based on parameters
    close = df["close"]
    high = df["high"]
    low = df["low"]

    # SMA
    for key, val in parameters.items():
        if key.startswith("sma_") and key != "sma_fast" and key != "sma_slow":
            period = int(val)
            df[f"sma_{period}"] = close.rolling(period).mean()

    sma_fast_p = parameters.get("sma_fast", parameters.get("sma_fast_period", 10))
    sma_slow_p = parameters.get("sma_slow", parameters.get("sma_slow_period", 50))
    df["sma_fast"] = close.rolling(int(sma_fast_p)).mean()
    df["sma_slow"] = close.rolling(int(sma_slow_p)).mean()

    # RSI
    rsi_period = int(parameters.get("rsi_period", 14))
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(rsi_period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(rsi_period).mean()
    rs = gain / (loss + 1e-10)
    df["rsi"] = 100 - (100 / (1 + rs))

    # ATR
    atr_period = int(parameters.get("atr_period", 14))
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["atr"] = tr.ewm(span=atr_period, adjust=False).mean()

    # Bollinger Bands
    bb_period = int(parameters.get("bb_period", 20))
    bb_std = float(parameters.get("bb_std", 2.0))
    bb_ma = close.rolling(bb_period).mean()
    bb_sd = close.rolling(bb_period).std()
    df["bb_upper"] = bb_ma + bb_std * bb_sd
    df["bb_lower"] = bb_ma - bb_std * bb_sd
    df["bb_pct"] = (close - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-10)

    # Breakout levels
    lookback = int(parameters.get("breakout_period", 20))
    df["high_n"] = high.rolling(lookback).max()
    df["low_n"] = low.rolling(lookback).min()

    # Z-score of close relative to SMA
    zscore_period = int(parameters.get("zscore_period", 20))
    zscore_ma = close.rolling(zscore_period).mean()
    zscore_sd = close.rolling(zscore_period).std()
    df["zscore"] = (close - zscore_ma) / (zscore_sd + 1e-10)

    # Evaluate rules
    entry_long = pd.Series(False, index=df.index)
    entry_short = pd.Series(False, index=df.index)
    exit_signal = pd.Series(False, index=df.index)

    for rule in entry_rules:
        cond = _evaluate_rule(df, rule, parameters)
        direction = rule.get("direction", "long").lower()
        if direction == "long":
            entry_long = entry_long | cond
        elif direction == "short":
            entry_short = entry_short | cond

    for rule in exit_rules:
        cond = _evaluate_rule(df, rule, parameters)
        exit_signal = exit_signal | cond

    # Build signal / direction columns
    signals = pd.DataFrame(index=df.index)
    signals["signal"] = 0
    signals["signal_direction"] = "NONE"

    signals.loc[entry_long, "signal"] = 1
    signals.loc[entry_long, "signal_direction"] = "LONG"
    signals.loc[entry_short, "signal"] = 1
    signals.loc[entry_short, "signal_direction"] = "SHORT"
    signals.loc[exit_signal, "signal"] = 0
    signals.loc[exit_signal, "signal_direction"] = "NONE"

    return signals


def _evaluate_rule(
    df: "pd.DataFrame",  # noqa: F821
    rule: dict[str, Any],
    parameters: dict[str, Any],
) -> "pd.Series":  # noqa: F821
    """Evaluate a single rule dict against the indicator DataFrame."""
    import pandas as pd

    indicator = rule.get("indicator", "")
    condition = rule.get("condition", "")
    value = rule.get("value")

    # Resolve indicator column
    if indicator in df.columns:
        series = df[indicator]
    elif indicator == "close":
        series = df["close"]
    elif indicator == "sma_crossover":
        # Special: fast > slow
        if condition == "crosses_above":
            prev_fast = df["sma_fast"].shift(1)
            prev_slow = df["sma_slow"].shift(1)
            return (prev_fast <= prev_slow) & (df["sma_fast"] > df["sma_slow"])
        elif condition == "crosses_below":
            prev_fast = df["sma_fast"].shift(1)
            prev_slow = df["sma_slow"].shift(1)
            return (prev_fast >= prev_slow) & (df["sma_fast"] < df["sma_slow"])
        return pd.Series(False, index=df.index)
    elif indicator == "breakout":
        if condition == "above":
            return df["close"] > df["high_n"].shift(1)
        elif condition == "below":
            return df["close"] < df["low_n"].shift(1)
        return pd.Series(False, index=df.index)
    else:
        return pd.Series(False, index=df.index)

    # Evaluate condition
    if value is None:
        return pd.Series(False, index=df.index)

    value = float(value)
    if condition == "above" or condition == "greater_than":
        return series > value
    elif condition == "below" or condition == "less_than":
        return series < value
    elif condition == "crosses_above":
        return (series.shift(1) <= value) & (series > value)
    elif condition == "crosses_below":
        return (series.shift(1) >= value) & (series < value)
    elif condition == "between":
        upper = float(rule.get("upper", value))
        lower = float(rule.get("lower", 0))
        return (series >= lower) & (series <= upper)
    else:
        return pd.Series(False, index=df.index)


def _fetch_price_data(
    symbol: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> "pd.DataFrame":  # noqa: F821
    """Fetch OHLCV price data from QuantCore DataStore or provider."""
    try:
        from quantcore.data.storage import DataStore

        store = DataStore()
        df = store.load(symbol)
        if df is not None and not df.empty:
            if start_date:
                df = df[df.index >= start_date]
            if end_date:
                df = df[df.index <= end_date]
            return df
    except Exception:
        pass

    # Fallback: try fetching from provider
    try:
        from quantcore.data.providers import get_data_provider

        provider = get_data_provider()
        df = provider.fetch_ohlcv(symbol, interval="daily")
        if df is not None and not df.empty:
            if start_date:
                df = df[df.index >= start_date]
            if end_date:
                df = df[df.index <= end_date]
            return df
    except Exception:
        pass

    return None


@mcp.tool()
async def run_backtest(
    strategy_id: str,
    symbol: str,
    start_date: str | None = None,
    end_date: str | None = None,
    initial_capital: float = 100_000.0,
    position_size_pct: float = 0.10,
    commission: float = 1.0,
    slippage_pct: float = 0.001,
) -> dict[str, Any]:
    """
    Backtest a registered strategy against historical price data.

    Fetches the strategy from the registry, generates entry/exit signals from
    its rules, then runs the BacktestEngine.  Results are stored back on the
    strategy record as backtest_summary and status is updated to 'backtested'.

    Args:
        strategy_id: Strategy to backtest.
        symbol: Ticker symbol for price data.
        start_date: Start date (YYYY-MM-DD). None = earliest available.
        end_date: End date (YYYY-MM-DD). None = latest available.
        initial_capital: Starting capital.
        position_size_pct: Fraction of capital per trade.
        commission: Commission per trade in dollars.
        slippage_pct: Slippage as fraction of price.

    Returns:
        BacktestResult dict with metrics.
    """
    import json as _json

    ctx, err = _live_db_or_error()
    if err:
        return err

    try:
        # 1. Load strategy
        strat_result = await get_strategy.fn(strategy_id=strategy_id)
        if not strat_result.get("success"):
            return {"success": False, "error": strat_result.get("error", "Strategy not found")}
        strat = strat_result["strategy"]

        # 2. Fetch price data
        price_data = await asyncio.get_event_loop().run_in_executor(
            None, _fetch_price_data, symbol, start_date, end_date
        )
        if price_data is None or price_data.empty:
            return {"success": False, "error": f"No price data available for {symbol}"}

        # 3. Generate signals from rules
        entry_rules = strat.get("entry_rules", [])
        exit_rules = strat.get("exit_rules", [])
        parameters = strat.get("parameters", {})

        if not entry_rules:
            return {"success": False, "error": "Strategy has no entry_rules"}

        signals = _generate_signals_from_rules(price_data, entry_rules, exit_rules, parameters)

        # 4. Run backtest
        from quantcore.backtesting.engine import BacktestConfig, BacktestEngine

        config = BacktestConfig(
            initial_capital=initial_capital,
            position_size_pct=position_size_pct,
            commission_per_trade=commission,
            slippage_pct=slippage_pct,
        )
        engine = BacktestEngine(config=config)
        result = engine.run(signals, price_data)

        # 5. Compute additional metrics
        import numpy as np

        calmar = 0.0
        if result.max_drawdown > 0:
            calmar = (result.total_return / 100.0) / (result.max_drawdown / 100.0)

        avg_pnl = 0.0
        if result.trades:
            avg_pnl = np.mean([t["pnl"] for t in result.trades])

        summary = {
            "symbol": symbol,
            "total_trades": result.total_trades,
            "win_rate": round(result.win_rate, 2),
            "sharpe_ratio": round(result.sharpe_ratio, 4),
            "max_drawdown": round(result.max_drawdown, 2),
            "total_return_pct": round(result.total_return, 2),
            "profit_factor": round(result.profit_factor, 4),
            "calmar_ratio": round(calmar, 4),
            "avg_trade_pnl": round(avg_pnl, 2),
            "start_date": str(price_data.index[0].date())
            if hasattr(price_data.index[0], "date")
            else str(price_data.index[0]),
            "end_date": str(price_data.index[-1].date())
            if hasattr(price_data.index[-1], "date")
            else str(price_data.index[-1]),
            "bars_tested": len(price_data),
        }

        # 6. Persist summary on strategy record
        ctx.db.execute(
            "UPDATE strategies SET backtest_summary = ?, status = CASE WHEN status = 'draft' THEN 'backtested' ELSE status END, updated_at = CURRENT_TIMESTAMP WHERE strategy_id = ?",
            [_json.dumps(summary), strategy_id],
        )

        return {"success": True, **summary, "strategy_id": strategy_id}

    except Exception as e:
        logger.error(f"[quantpod_mcp] run_backtest failed: {e}")
        return {"success": False, "error": str(e), "strategy_id": strategy_id, "symbol": symbol}


@mcp.tool()
async def run_walkforward(
    strategy_id: str,
    symbol: str,
    n_splits: int = 5,
    test_size: int = 252,
    min_train_size: int = 504,
    gap: int = 0,
    expanding: bool = True,
    initial_capital: float = 100_000.0,
    position_size_pct: float = 0.10,
) -> dict[str, Any]:
    """
    Walk-forward validation of a registered strategy.

    Splits price data into successive train/test folds.  On each fold, signals
    are generated from the strategy rules and a backtest is run on the test
    period.  Returns per-fold IS/OOS metrics and aggregate statistics.

    Args:
        strategy_id: Strategy to validate.
        symbol: Ticker symbol.
        n_splits: Number of walk-forward folds.
        test_size: Bars per test fold.
        min_train_size: Minimum training bars.
        gap: Embargo bars between train and test.
        expanding: Expanding (True) or rolling (False) window.
        initial_capital: Starting capital per fold.
        position_size_pct: Position size fraction.

    Returns:
        WalkForwardResult dict with per-fold and aggregate metrics.
    """
    import json as _json

    import numpy as np

    ctx, err = _live_db_or_error()
    if err:
        return err

    try:
        # 1. Load strategy
        strat_result = await get_strategy.fn(strategy_id=strategy_id)
        if not strat_result.get("success"):
            return {"success": False, "error": strat_result.get("error", "Strategy not found")}
        strat = strat_result["strategy"]

        entry_rules = strat.get("entry_rules", [])
        exit_rules = strat.get("exit_rules", [])
        parameters = strat.get("parameters", {})

        if not entry_rules:
            return {"success": False, "error": "Strategy has no entry_rules"}

        # 2. Fetch price data
        price_data = await asyncio.get_event_loop().run_in_executor(
            None, _fetch_price_data, symbol, None, None
        )
        if price_data is None or price_data.empty:
            return {"success": False, "error": f"No price data for {symbol}"}

        n = len(price_data)
        total_needed = min_train_size + n_splits * test_size
        if n < total_needed:
            return {
                "success": False,
                "error": f"Insufficient data: need {total_needed} bars, have {n}",
            }

        # 3. Walk-forward splits
        from quantcore.backtesting.engine import BacktestConfig, BacktestEngine

        config = BacktestConfig(
            initial_capital=initial_capital,
            position_size_pct=position_size_pct,
        )

        fold_results = []
        first_test_start = min_train_size + gap

        for i in range(n_splits):
            test_start = first_test_start + i * test_size
            test_end = min(test_start + test_size, n)
            if expanding:
                train_start = 0
            else:
                train_start = max(0, test_start - gap - min_train_size)
            train_end = test_start - gap

            train_data = price_data.iloc[train_start:train_end]
            test_data = price_data.iloc[test_start:test_end]

            # Generate signals for each period
            train_signals = _generate_signals_from_rules(
                train_data, entry_rules, exit_rules, parameters
            )
            test_signals = _generate_signals_from_rules(
                test_data, entry_rules, exit_rules, parameters
            )

            # Run backtests
            engine_is = BacktestEngine(config=config)
            result_is = engine_is.run(train_signals, train_data)

            engine_oos = BacktestEngine(config=config)
            result_oos = engine_oos.run(test_signals, test_data)

            fold_results.append(
                {
                    "fold": i + 1,
                    "train_bars": len(train_data),
                    "test_bars": len(test_data),
                    "is_sharpe": round(result_is.sharpe_ratio, 4),
                    "is_return_pct": round(result_is.total_return, 2),
                    "is_trades": result_is.total_trades,
                    "oos_sharpe": round(result_oos.sharpe_ratio, 4),
                    "oos_return_pct": round(result_oos.total_return, 2),
                    "oos_trades": result_oos.total_trades,
                    "oos_max_dd": round(result_oos.max_drawdown, 2),
                }
            )

        # 4. Aggregate
        is_sharpes = [f["is_sharpe"] for f in fold_results]
        oos_sharpes = [f["oos_sharpe"] for f in fold_results]
        is_mean = float(np.mean(is_sharpes)) if is_sharpes else 0.0
        oos_mean = float(np.mean(oos_sharpes)) if oos_sharpes else 0.0
        is_std = float(np.std(is_sharpes)) if is_sharpes else 0.0
        oos_std = float(np.std(oos_sharpes)) if oos_sharpes else 0.0
        overfit_ratio = is_mean / oos_mean if oos_mean != 0 else float("inf")
        oos_positive = sum(1 for s in oos_sharpes if s > 0)
        degradation = ((is_mean - oos_mean) / abs(is_mean) * 100) if is_mean != 0 else 0.0

        summary = {
            "symbol": symbol,
            "n_folds": n_splits,
            "is_sharpe_mean": round(is_mean, 4),
            "oos_sharpe_mean": round(oos_mean, 4),
            "is_sharpe_std": round(is_std, 4),
            "oos_sharpe_std": round(oos_std, 4),
            "overfit_ratio": round(overfit_ratio, 4),
            "oos_positive_folds": oos_positive,
            "oos_degradation_pct": round(degradation, 2),
            "fold_results": fold_results,
        }

        # 5. Persist
        ctx.db.execute(
            "UPDATE strategies SET walkforward_summary = ?, updated_at = CURRENT_TIMESTAMP WHERE strategy_id = ?",
            [_json.dumps(summary), strategy_id],
        )

        return {"success": True, "strategy_id": strategy_id, **summary}

    except Exception as e:
        logger.error(f"[quantpod_mcp] run_walkforward failed: {e}")
        return {"success": False, "error": str(e), "strategy_id": strategy_id, "symbol": symbol}


# =============================================================================
# PHASE 3: Execution Tools
# =============================================================================


def _calc_quantity_from_size(position_size: str, equity: float, current_price: float) -> int:
    """Convert a position_size label ('full', 'half', 'quarter') to shares."""
    fractions = {"full": 0.10, "half": 0.05, "quarter": 0.025}
    frac = fractions.get(position_size, 0.025)
    if current_price <= 0:
        return 0
    return max(1, int((equity * frac) / current_price))


@mcp.tool()
async def execute_trade(
    symbol: str,
    action: str,
    reasoning: str,
    confidence: float,
    quantity: int | None = None,
    position_size: str = "quarter",
    order_type: str = "market",
    limit_price: float | None = None,
    strategy_id: str | None = None,
    paper_mode: bool = True,
) -> dict[str, Any]:
    """
    Execute a trade through the risk gate and broker.

    The risk gate is NEVER bypassed. reasoning and confidence are REQUIRED
    to ensure every trade has an audit trail.

    Args:
        symbol: Ticker symbol.
        action: "buy" or "sell".
        reasoning: REQUIRED. Why you are making this trade.
        confidence: REQUIRED. 0-1 confidence score.
        quantity: Number of shares. Auto-calculated from position_size if None.
        position_size: "full", "half", or "quarter" (used if quantity is None).
        order_type: "market" or "limit".
        limit_price: Required for limit orders.
        strategy_id: Links trade to a registered strategy.
        paper_mode: Must be explicitly False for live trading.

    Returns:
        Dict with fill details or rejection reason.
    """
    import os
    import uuid as _uuid

    ctx, err = _live_db_or_error()
    if err:
        return err

    try:
        # 1. Kill switch guard
        ctx.kill_switch.guard()

        # 2. Paper/live mode check
        if not paper_mode:
            use_real = os.getenv("USE_REAL_TRADING", "false").strip().lower()
            if use_real not in ("true", "1", "yes"):
                return {
                    "success": False,
                    "error": (
                        "Live trading rejected: paper_mode=False but "
                        "USE_REAL_TRADING env var is not 'true'. "
                        "Set USE_REAL_TRADING=true to enable live trading."
                    ),
                    "broker_mode": get_broker_mode(),
                }

        # 3. Get current price (from portfolio position or default)
        snapshot = ctx.portfolio.get_snapshot()
        pos = ctx.portfolio.get_position(symbol)
        current_price = pos.current_price if pos and pos.current_price > 0 else 0.0

        # If no current price from portfolio, we need one for risk calculations
        if current_price <= 0:
            return {
                "success": False,
                "error": (
                    f"No current price available for {symbol}. "
                    "Run get_portfolio_state or run_analysis first to populate prices."
                ),
            }

        # 4. Calculate quantity if not explicit
        if quantity is None or quantity <= 0:
            quantity = _calc_quantity_from_size(position_size, snapshot.total_equity, current_price)

        # 5. Risk gate check — SACRED, NEVER BYPASSED
        daily_volume = 1_000_000  # Default for paper mode
        verdict = ctx.risk_gate.check(
            symbol=symbol,
            side=action,
            quantity=quantity,
            current_price=current_price,
            daily_volume=daily_volume,
        )

        if not verdict.approved:
            violations = [v.description for v in verdict.violations]

            # Log the rejection
            from quant_pod.audit.models import DecisionEvent

            ctx.audit.record(
                DecisionEvent(
                    event_id=str(_uuid.uuid4()),
                    session_id=ctx.session_id,
                    event_type="risk_rejection",
                    agent_name="ClaudeCode",
                    agent_role="strategic_brain",
                    symbol=symbol,
                    action=action,
                    confidence=confidence,
                    output_summary=f"REJECTED: {'; '.join(violations)}",
                    risk_approved=False,
                    risk_violations=violations,
                    portfolio_snapshot=_serialize(snapshot) or {},
                )
            )

            return {
                "success": False,
                "risk_approved": False,
                "risk_violations": violations,
                "error": f"Risk gate rejected: {'; '.join(violations)}",
                "broker_mode": get_broker_mode(),
            }

        # Use approved_quantity (may have been scaled down)
        approved_qty = verdict.approved_quantity or quantity

        # 6. Execute via broker
        from quant_pod.execution.paper_broker import OrderRequest

        order = OrderRequest(
            symbol=symbol,
            side=action,
            quantity=approved_qty,
            order_type=order_type,
            limit_price=limit_price,
            current_price=current_price,
            daily_volume=daily_volume,
        )

        fill = ctx.broker.execute(order)

        # 7. Log to audit trail
        from quant_pod.audit.models import DecisionEvent

        ctx.audit.record(
            DecisionEvent(
                event_id=str(_uuid.uuid4()),
                session_id=ctx.session_id,
                event_type="execution",
                agent_name="ClaudeCode",
                agent_role="strategic_brain",
                symbol=symbol,
                action=action,
                confidence=confidence,
                output_summary=(
                    f"{'FILLED' if not fill.rejected else 'REJECTED'}: "
                    f"{action.upper()} {fill.filled_quantity} {symbol} "
                    f"@ ${fill.fill_price:.4f} | reasoning: {reasoning[:200]}"
                ),
                output_structured={
                    "order_id": fill.order_id,
                    "fill_price": fill.fill_price,
                    "filled_quantity": fill.filled_quantity,
                    "slippage_bps": fill.slippage_bps,
                    "strategy_id": strategy_id,
                    "paper_mode": paper_mode,
                },
                risk_approved=True,
                portfolio_snapshot=_serialize(snapshot) or {},
            )
        )

        return {
            "success": not fill.rejected,
            "order_id": fill.order_id,
            "fill_price": fill.fill_price,
            "filled_quantity": fill.filled_quantity,
            "slippage_bps": round(fill.slippage_bps, 2),
            "commission": round(fill.commission, 4),
            "risk_approved": True,
            "risk_violations": [],
            "broker_mode": get_broker_mode(),
            "error": fill.reject_reason if fill.rejected else None,
        }

    except RuntimeError as e:
        # Kill switch or other runtime guard
        return {"success": False, "error": str(e), "broker_mode": get_broker_mode()}
    except Exception as e:
        logger.error(f"[quantpod_mcp] execute_trade failed: {e}")
        return {"success": False, "error": str(e), "broker_mode": get_broker_mode()}


@mcp.tool()
async def close_position(
    symbol: str,
    reasoning: str,
    quantity: int | None = None,
) -> dict[str, Any]:
    """
    Close an open position (all or partial).

    Infers the correct action (sell for longs, buy for shorts) from
    the current position side. Routes through risk gate and audit.

    Args:
        symbol: Ticker symbol of the position to close.
        reasoning: REQUIRED. Why you are closing.
        quantity: Shares to close. None = close all.

    Returns:
        Dict with fill details or error.
    """
    ctx, err = _live_db_or_error()
    if err:
        return err
    try:
        pos = ctx.portfolio.get_position(symbol)
        if pos is None:
            return {"success": False, "error": f"No open position for {symbol}"}

        close_qty = quantity or abs(pos.quantity)
        close_action = "sell" if pos.side == "long" else "buy"

        return await execute_trade.fn(
            symbol=symbol,
            action=close_action,
            reasoning=f"CLOSE: {reasoning}",
            confidence=1.0,
            quantity=close_qty,
            order_type="market",
            paper_mode=True,
        )
    except Exception as e:
        logger.error(f"[quantpod_mcp] close_position failed: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool()
async def cancel_order(order_id: str) -> dict[str, Any]:
    """
    Cancel an open order by ID.

    Note: PaperBroker executes all orders immediately, so cancellation
    is only meaningful for limit orders that were not filled.

    Args:
        order_id: The order ID to cancel.

    Returns:
        Dict with success status.
    """
    return {
        "success": True,
        "message": (
            f"Order {order_id} cancel acknowledged. "
            "Note: PaperBroker fills all market orders immediately; "
            "unfilled limit orders are not persisted."
        ),
    }


@mcp.tool()
async def get_fills(
    symbol: str | None = None,
    limit: int = 20,
) -> dict[str, Any]:
    """
    Get recent trade fills.

    Args:
        symbol: Filter by symbol. None = all symbols.
        limit: Maximum fills to return.

    Returns:
        Dict with list of fill records.
    """
    ctx, err = _live_db_or_error()
    if err:
        return err
    try:
        fills = ctx.broker.get_fills(symbol=symbol, limit=limit)
        return {
            "success": True,
            "fills": [_serialize(f) for f in fills],
            "total": len(fills),
        }
    except Exception as e:
        logger.error(f"[quantpod_mcp] get_fills failed: {e}")
        return {"success": False, "error": str(e), "fills": [], "total": 0}


@mcp.tool()
async def get_risk_metrics() -> dict[str, Any]:
    """
    Get current risk exposure, drawdown, and limits headroom.

    Returns:
        Dict with cash, equity, exposure, daily loss, and all limit values.
    """
    ctx, err = _live_db_or_error()
    if err:
        return err
    try:
        snapshot = ctx.portfolio.get_snapshot()
        positions = ctx.portfolio.get_positions()
        limits = ctx.risk_gate.limits

        gross_exposure = sum(abs(p.quantity) * p.current_price for p in positions)
        equity = snapshot.total_equity or 1.0
        gross_pct = gross_exposure / equity
        daily_loss_pct = abs(min(0, snapshot.daily_pnl)) / equity if equity > 0 else 0.0

        return {
            "success": True,
            "cash": round(snapshot.cash, 2),
            "total_equity": round(snapshot.total_equity, 2),
            "positions_value": round(snapshot.positions_value, 2),
            "position_count": snapshot.position_count,
            "daily_pnl": round(snapshot.daily_pnl, 2),
            "daily_loss_pct": round(daily_loss_pct * 100, 2),
            "daily_loss_limit_pct": round(limits.daily_loss_limit_pct * 100, 2),
            "daily_headroom_pct": round((limits.daily_loss_limit_pct - daily_loss_pct) * 100, 2),
            "gross_exposure": round(gross_exposure, 2),
            "gross_exposure_pct": round(gross_pct * 100, 2),
            "max_gross_exposure_pct": round(limits.max_gross_exposure_pct * 100, 2),
            "largest_position_pct": round(snapshot.largest_position_pct * 100, 2),
            "max_position_pct": round(limits.max_position_pct * 100, 2),
            "kill_switch_active": ctx.kill_switch.is_active(),
            "risk_halted": ctx.risk_gate.is_halted(),
        }
    except Exception as e:
        logger.error(f"[quantpod_mcp] get_risk_metrics failed: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool()
async def get_audit_trail(
    session_id: str | None = None,
    symbol: str | None = None,
    limit: int = 50,
) -> dict[str, Any]:
    """
    Query the decision audit trail.

    Args:
        session_id: Filter by session. None = current session.
        symbol: Filter by symbol.
        limit: Maximum entries.

    Returns:
        Dict with list of audit events.
    """
    ctx, err = _live_db_or_error()
    if err:
        return err
    try:
        from quant_pod.audit.models import AuditQuery

        query = AuditQuery(
            session_id=session_id or "",
            symbol=symbol or "",
            limit=limit,
        )
        events = ctx.audit.query(query)
        return {
            "success": True,
            "events": [
                {
                    "event_id": e.event_id,
                    "event_type": e.event_type,
                    "agent_name": e.agent_name,
                    "symbol": e.symbol,
                    "action": e.action,
                    "confidence": e.confidence,
                    "risk_approved": e.risk_approved,
                    "output_summary": e.output_summary[:300] if e.output_summary else "",
                    "created_at": e.created_at.isoformat() if e.created_at else None,
                }
                for e in events
            ],
            "total": len(events),
        }
    except Exception as e:
        logger.error(f"[quantpod_mcp] get_audit_trail failed: {e}")
        return {"success": False, "error": str(e), "events": [], "total": 0}


# =============================================================================
# PHASE 4: Decoder Tools
# =============================================================================


@mcp.tool()
async def decode_strategy(
    signals: list[dict[str, Any]],
    source_name: str = "unknown",
    strategy_name: str | None = None,
) -> dict[str, Any]:
    """
    Reverse-engineer a trading strategy from historical trade signals.

    Analyzes entry patterns (timing, direction bias), exit patterns (holding
    period, target vs time-based), sizing patterns, and regime affinity.

    Args:
        signals: List of trade signal dicts. Each must have:
            symbol, direction, entry_time, entry_price, exit_time, exit_price.
            Optional: size, notes.
        source_name: Name of the signal source (e.g., "discord_trader_x").
        strategy_name: If provided, auto-registers the decoded strategy.

    Returns:
        DecodedStrategy with entry_trigger, exit_trigger, timing_pattern,
        win_rate, regime_affinity, edge_hypothesis, and per-IC analysis.
    """
    try:
        from quant_pod.crews.decoder_crew import decode_signals

        result = await asyncio.get_event_loop().run_in_executor(
            None, decode_signals, signals, source_name
        )

        if not result.get("success"):
            return result

        # Auto-register if strategy_name provided
        if strategy_name and result.get("decoded_strategy"):
            decoded = result["decoded_strategy"]
            reg_result = await register_strategy.fn(
                name=strategy_name,
                description=decoded.get("edge_hypothesis", ""),
                parameters={},
                entry_rules=[{"decoded_trigger": decoded.get("entry_trigger", "")}],
                exit_rules=[{"decoded_trigger": decoded.get("exit_trigger", "")}],
                regime_affinity=decoded.get("regime_affinity", {}),
                source="decoded",
            )
            result["registered"] = reg_result

        return result
    except Exception as e:
        logger.error(f"[quantpod_mcp] decode_strategy failed: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool()
async def decode_from_trades(
    source: str = "closed_trades",
    symbol: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    source_name: str = "self",
) -> dict[str, Any]:
    """
    Decode strategy patterns from the system's own trade history.

    Pulls trades from the closed_trades or fills table and feeds them
    to the decoder.

    Args:
        source: "closed_trades" or "fills".
        symbol: Filter by symbol. None = all.
        start_date: Start date filter (YYYY-MM-DD).
        end_date: End date filter (YYYY-MM-DD).
        source_name: Label for the decoded source.

    Returns:
        DecodedStrategy from historical trades.
    """
    ctx, err = _live_db_or_error()
    if err:
        return err
    try:
        if source == "closed_trades":
            query = "SELECT symbol, side, entry_price, exit_price, opened_at, closed_at FROM closed_trades"
        elif source == "fills":
            query = "SELECT symbol, side, fill_price, fill_price, filled_at, filled_at FROM fills WHERE rejected = FALSE"
        else:
            return {
                "success": False,
                "error": f"Unknown source: {source}. Use 'closed_trades' or 'fills'.",
            }

        conditions = []
        params = []
        if symbol:
            conditions.append("symbol = ?")
            params.append(symbol)
        if start_date:
            if source == "closed_trades":
                conditions.append("closed_at >= ?")
            else:
                conditions.append("filled_at >= ?")
            params.append(start_date)
        if end_date:
            if source == "closed_trades":
                conditions.append("closed_at <= ?")
            else:
                conditions.append("filled_at <= ?")
            params.append(end_date)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY 5"  # Order by entry/fill time

        rows = ctx.db.execute(query, params).fetchall()

        if not rows:
            return {"success": False, "error": "No trades found matching filters"}

        # Convert to signal format
        signals = []
        for r in rows:
            signals.append(
                {
                    "symbol": r[0],
                    "direction": "long" if r[1] == "long" else "short",
                    "entry_price": r[2],
                    "exit_price": r[3],
                    "entry_time": str(r[4]),
                    "exit_time": str(r[5]),
                }
            )

        return await decode_strategy.fn(
            signals=signals,
            source_name=source_name,
        )
    except Exception as e:
        logger.error(f"[quantpod_mcp] decode_from_trades failed: {e}")
        return {"success": False, "error": str(e)}


# =============================================================================
# PHASE 5: Meta Orchestration Tools
# =============================================================================


@mcp.tool()
async def get_regime_strategies(regime: str) -> dict[str, Any]:
    """
    Get strategy allocations for a given regime from the matrix.

    Args:
        regime: Regime label (e.g., "trending_up", "ranging").

    Returns:
        Dict with list of (strategy_id, allocation_pct, confidence).
    """
    ctx, err = _live_db_or_error()
    if err:
        return err
    try:
        rows = ctx.db.execute(
            "SELECT strategy_id, allocation_pct, confidence, last_updated "
            "FROM regime_strategy_matrix WHERE regime = ? ORDER BY allocation_pct DESC",
            [regime],
        ).fetchall()

        allocations = [
            {
                "strategy_id": r[0],
                "allocation_pct": r[1],
                "confidence": r[2],
                "last_updated": str(r[3]) if r[3] else None,
            }
            for r in rows
        ]
        return {
            "success": True,
            "regime": regime,
            "allocations": allocations,
            "total": len(allocations),
        }
    except Exception as e:
        logger.error(f"[quantpod_mcp] get_regime_strategies failed: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool()
async def set_regime_allocation(
    regime: str,
    allocations: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Set or update strategy allocations for a regime.

    Upserts into the regime_strategy_matrix. This is how /reflect updates
    the matrix based on accumulated performance data.

    Args:
        regime: Regime label.
        allocations: List of dicts with strategy_id, allocation_pct, confidence (optional).

    Returns:
        Confirmation with the updated allocations.
    """
    ctx, err = _live_db_or_error()
    if err:
        return err
    try:
        # Validate total allocation <= 1.0
        total = sum(a.get("allocation_pct", 0) for a in allocations)
        if total > 1.0:
            return {
                "success": False,
                "error": f"Total allocation {total:.0%} exceeds 100%. Reduce allocations.",
            }

        for alloc in allocations:
            strategy_id = alloc.get("strategy_id")
            allocation_pct = alloc.get("allocation_pct", 0)
            confidence = alloc.get("confidence", 0.5)

            if not strategy_id:
                continue

            # Upsert: try update, then insert
            ctx.db.execute(
                "UPDATE regime_strategy_matrix "
                "SET allocation_pct = ?, confidence = ?, last_updated = CURRENT_TIMESTAMP "
                "WHERE regime = ? AND strategy_id = ?",
                [allocation_pct, confidence, regime, strategy_id],
            ).fetchone()

            # Check if row existed
            exists = ctx.db.execute(
                "SELECT 1 FROM regime_strategy_matrix WHERE regime = ? AND strategy_id = ?",
                [regime, strategy_id],
            ).fetchone()

            if not exists:
                ctx.db.execute(
                    "INSERT INTO regime_strategy_matrix (regime, strategy_id, allocation_pct, confidence) "
                    "VALUES (?, ?, ?, ?)",
                    [regime, strategy_id, allocation_pct, confidence],
                )

        logger.info(
            f"[quantpod_mcp] Updated regime matrix for '{regime}': {len(allocations)} strategies"
        )
        return await get_regime_strategies.fn(regime)
    except Exception as e:
        logger.error(f"[quantpod_mcp] set_regime_allocation failed: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool()
async def run_multi_analysis(
    symbols: list[str],
) -> dict[str, Any]:
    """
    Run TradingCrew analysis for multiple symbols.

    Runs run_analysis sequentially for each symbol and collects all DailyBriefs.

    Args:
        symbols: List of ticker symbols to analyze.

    Returns:
        Dict with list of per-symbol results.
    """
    results = []
    for symbol in symbols:
        result = await run_analysis.fn(symbol=symbol)
        results.append({"symbol": symbol, **result})

    successes = sum(1 for r in results if r.get("success"))
    return {
        "success": successes > 0,
        "results": results,
        "symbols_analyzed": len(symbols),
        "symbols_succeeded": successes,
        "symbols_failed": len(symbols) - successes,
    }


@mcp.tool()
async def resolve_portfolio_conflicts(
    proposed_trades: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Resolve signal conflicts across multiple strategies for the same symbols.

    Rules:
      - Same symbol, different directions: high confidence wins, or SKIP if both high
      - Same symbol, same direction: merge with conservative sizing

    Args:
        proposed_trades: List of trade dicts, each with:
            symbol, action, confidence, strategy_id, capital_pct.

    Returns:
        Dict with resolved_trades, resolutions, conflicts_count.
    """
    try:
        from quant_pod.mcp.allocation import resolve_conflicts

        result = resolve_conflicts(proposed_trades)
        return {"success": True, **result}
    except Exception as e:
        logger.error(f"[quantpod_mcp] resolve_portfolio_conflicts failed: {e}")
        return {"success": False, "error": str(e)}


# =============================================================================
# PHASE 6: Learning Loop — RL, Lifecycle, Performance
# =============================================================================


@mcp.tool()
async def get_rl_status() -> dict[str, Any]:
    """
    Get RL model status: which models are enabled, shadow vs live, config.

    Returns:
        Dict with RL config flags, shadow mode state, and agent statuses.
    """
    try:
        from quantcore.rl.config import get_rl_config

        cfg = get_rl_config()
        return {
            "success": True,
            "config_version": cfg.config_version,
            "shadow_mode_enabled": cfg.shadow_mode_enabled,
            "agents": {
                "execution_rl": {
                    "enabled": cfg.enable_execution_rl,
                    "shadow": cfg.execution_shadow,
                },
                "sizing_rl": {"enabled": cfg.enable_sizing_rl, "shadow": cfg.sizing_shadow},
                "meta_rl": {"enabled": cfg.enable_meta_rl, "shadow": cfg.meta_shadow},
                "spread_rl": {"enabled": cfg.enable_spread_rl},
            },
        }
    except Exception as e:
        logger.error(f"[quantpod_mcp] get_rl_status failed: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool()
async def get_rl_recommendation(
    symbol: str,
    direction: str,
    signal_confidence: float = 0.5,
    regime: str = "normal",
    current_drawdown: float = 0.0,
) -> dict[str, Any]:
    """
    Get RL-recommended position size adjustment for a trade.

    Claude reads this as INPUT to its decision, not a directive.
    All RL agents start in shadow mode — output is advisory only.

    Args:
        symbol: Ticker symbol.
        direction: "LONG" or "SHORT".
        signal_confidence: Signal confidence (0-1).
        regime: Current market regime label.
        current_drawdown: Current portfolio drawdown fraction.

    Returns:
        Dict with RL recommendations (tagged as shadow if applicable).
    """
    try:
        from quantcore.rl.config import get_rl_config
        from quantcore.rl.rl_tools import RLPositionSizeTool

        cfg = get_rl_config()
        if not cfg.enable_sizing_rl:
            return {"success": True, "recommendation": None, "note": "Sizing RL disabled in config"}

        tool = RLPositionSizeTool()
        result_str = tool._run(
            signal_confidence=signal_confidence,
            signal_direction=direction,
            regime=regime,
            current_drawdown=current_drawdown,
            current_position_pct=0.0,
            portfolio_heat=0.0,
            recent_win_rate=0.5,
            atr_percentile=50.0,
        )

        import json as _json

        try:
            result = _json.loads(result_str) if isinstance(result_str, str) else result_str
        except (ValueError, TypeError):
            result = {"raw": str(result_str)}

        return {
            "success": True,
            "symbol": symbol,
            "recommendation": result,
            "shadow_mode": cfg.shadow_mode_enabled,
            "note": "[SHADOW — advisory only]" if cfg.shadow_mode_enabled else "LIVE",
        }
    except Exception as e:
        logger.warning(f"[quantpod_mcp] get_rl_recommendation failed (graceful): {e}")
        return {"success": True, "recommendation": None, "note": f"RL unavailable: {e}"}


@mcp.tool()
async def promote_strategy(
    strategy_id: str,
    evidence: str,
) -> dict[str, Any]:
    """
    Promote a strategy to "live" status after validation.

    Validates:
      - Backtest exists with positive Sharpe
      - Walk-forward OOS passes (if available)
      - Current status is "forward_testing"
    The evidence field documents why promotion is justified — required.

    Args:
        strategy_id: Strategy to promote.
        evidence: REQUIRED. Justification for promotion.

    Returns:
        Success with updated record, or rejection with failed criteria.
    """

    ctx, err = _live_db_or_error()
    if err:
        return err
    try:
        strat_result = await get_strategy.fn(strategy_id=strategy_id)
        if not strat_result.get("success"):
            return {"success": False, "error": "Strategy not found"}

        strat = strat_result["strategy"]
        failures = []

        # Check current status
        if strat.get("status") != "forward_testing":
            failures.append(f"Status is '{strat.get('status')}', expected 'forward_testing'")

        # Check backtest exists with positive Sharpe
        bt = strat.get("backtest_summary") or {}
        if not bt:
            failures.append("No backtest_summary — run backtest first")
        elif bt.get("sharpe_ratio", 0) <= 0:
            failures.append(f"Backtest Sharpe {bt.get('sharpe_ratio', 0):.2f} <= 0")

        # Check walk-forward if available
        wf = strat.get("walkforward_summary") or {}
        if wf:
            oos_sharpe = wf.get("oos_sharpe_mean", 0)
            if oos_sharpe <= 0:
                failures.append(f"OOS Sharpe {oos_sharpe:.2f} <= 0")

        if failures:
            return {
                "success": False,
                "error": "Promotion criteria not met",
                "failures": failures,
                "strategy_id": strategy_id,
            }

        # Promote
        ctx.db.execute(
            "UPDATE strategies SET status = 'live', updated_at = CURRENT_TIMESTAMP WHERE strategy_id = ?",
            [strategy_id],
        )
        logger.info(f"[quantpod_mcp] Promoted strategy {strategy_id} to LIVE: {evidence}")

        return {
            "success": True,
            "strategy_id": strategy_id,
            "new_status": "live",
            "evidence": evidence,
        }
    except Exception as e:
        logger.error(f"[quantpod_mcp] promote_strategy failed: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool()
async def retire_strategy(
    strategy_id: str,
    reason: str,
) -> dict[str, Any]:
    """
    Retire a strategy and remove it from the regime-strategy matrix.

    The reason field is required — retirement reasons are learning data
    for /reflect sessions.

    Args:
        strategy_id: Strategy to retire.
        reason: REQUIRED. Why this strategy is being retired.

    Returns:
        Confirmation with the retirement details.
    """
    ctx, err = _live_db_or_error()
    if err:
        return err
    try:
        # Update status
        ctx.db.execute(
            "UPDATE strategies SET status = 'retired', updated_at = CURRENT_TIMESTAMP WHERE strategy_id = ?",
            [strategy_id],
        )

        # Remove from regime matrix
        ctx.db.execute(
            "DELETE FROM regime_strategy_matrix WHERE strategy_id = ?",
            [strategy_id],
        )

        logger.info(f"[quantpod_mcp] Retired strategy {strategy_id}: {reason}")
        return {
            "success": True,
            "strategy_id": strategy_id,
            "new_status": "retired",
            "reason": reason,
            "removed_from_matrix": True,
        }
    except Exception as e:
        logger.error(f"[quantpod_mcp] retire_strategy failed: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool()
async def get_strategy_performance(
    strategy_id: str,
    lookback_days: int = 30,
) -> dict[str, Any]:
    """
    Compute live performance metrics for a strategy over a lookback period.

    Queries closed_trades linked to this strategy (by session correlation),
    computes win rate, average win/loss, Sharpe approximation, and compares
    against the registered backtest_summary.

    Args:
        strategy_id: Strategy to evaluate.
        lookback_days: Number of days to look back.

    Returns:
        Dict with live metrics, backtest comparison, and degradation flag.
    """

    ctx, err = _live_db_or_error()
    if err:
        return err
    try:
        # Get strategy record for backtest comparison
        strat_result = await get_strategy.fn(strategy_id=strategy_id)
        if not strat_result.get("success"):
            return {"success": False, "error": "Strategy not found"}

        strat = strat_result["strategy"]
        bt = strat.get("backtest_summary") or {}

        # Query closed trades in lookback period
        from datetime import datetime as _dt
        from datetime import timedelta as _td

        cutoff = _dt.now() - _td(days=lookback_days)
        rows = ctx.db.execute(
            """
            SELECT realized_pnl, closed_at, holding_days
            FROM closed_trades
            WHERE closed_at >= ?
            ORDER BY closed_at
            """,
            [cutoff],
        ).fetchall()

        if not rows:
            return {
                "success": True,
                "strategy_id": strategy_id,
                "lookback_days": lookback_days,
                "total_trades": 0,
                "note": "No closed trades in lookback period",
            }

        pnls = [float(r[0]) for r in rows]
        total_trades = len(pnls)
        winners = sum(1 for p in pnls if p > 0)
        win_rate = winners / total_trades * 100
        avg_win = sum(p for p in pnls if p > 0) / max(1, winners)
        avg_loss = sum(p for p in pnls if p < 0) / max(1, total_trades - winners)
        total_pnl = sum(pnls)

        # Simple Sharpe approximation
        import numpy as np

        pnl_arr = np.array(pnls)
        live_sharpe = (
            float(np.mean(pnl_arr) / (np.std(pnl_arr) + 1e-10) * np.sqrt(252))
            if len(pnl_arr) > 1
            else 0.0
        )

        # Compare to backtest
        bt_sharpe = bt.get("sharpe_ratio", 0)
        degradation_pct = 0.0
        if bt_sharpe > 0:
            degradation_pct = (bt_sharpe - live_sharpe) / bt_sharpe * 100

        return {
            "success": True,
            "strategy_id": strategy_id,
            "lookback_days": lookback_days,
            "total_trades": total_trades,
            "win_rate": round(win_rate, 2),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "total_pnl": round(total_pnl, 2),
            "live_sharpe": round(live_sharpe, 4),
            "backtest_sharpe": round(bt_sharpe, 4),
            "degradation_pct": round(degradation_pct, 2),
            "degraded": degradation_pct > 30,
        }
    except Exception as e:
        logger.error(f"[quantpod_mcp] get_strategy_performance failed: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool()
async def validate_strategy(strategy_id: str) -> dict[str, Any]:
    """
    Re-validate a strategy by comparing current backtest to registered summary.

    Runs a fresh backtest (if price data is available) and compares metrics
    to the stored backtest_summary. Flags significant degradation.

    Args:
        strategy_id: Strategy to validate.

    Returns:
        Dict with still_valid flag, current vs historical metrics, degradation.
    """
    try:
        strat_result = await get_strategy.fn(strategy_id=strategy_id)
        if not strat_result.get("success"):
            return {"success": False, "error": "Strategy not found"}

        strat = strat_result["strategy"]
        bt = strat.get("backtest_summary") or {}

        if not bt:
            return {
                "success": True,
                "strategy_id": strategy_id,
                "still_valid": False,
                "reason": "No backtest_summary to compare against",
            }

        symbol = bt.get("symbol", "SPY")
        bt_sharpe = bt.get("sharpe_ratio", 0)
        bt_dd = bt.get("max_drawdown", 0)

        # Re-run backtest on recent data
        fresh = await run_backtest.fn(
            strategy_id=strategy_id,
            symbol=symbol,
        )

        if not fresh.get("success"):
            return {
                "success": True,
                "strategy_id": strategy_id,
                "still_valid": None,
                "reason": f"Could not re-run backtest: {fresh.get('error', 'unknown')}",
            }

        fresh_sharpe = fresh.get("sharpe_ratio", 0)
        fresh_dd = fresh.get("max_drawdown", 0)

        sharpe_degradation = 0.0
        if bt_sharpe > 0:
            sharpe_degradation = (bt_sharpe - fresh_sharpe) / bt_sharpe * 100

        still_valid = sharpe_degradation < 30 and fresh_sharpe > 0

        return {
            "success": True,
            "strategy_id": strategy_id,
            "still_valid": still_valid,
            "original_sharpe": round(bt_sharpe, 4),
            "fresh_sharpe": round(fresh_sharpe, 4),
            "sharpe_degradation_pct": round(sharpe_degradation, 2),
            "original_max_dd": round(bt_dd, 2),
            "fresh_max_dd": round(fresh_dd, 2),
        }
    except Exception as e:
        logger.error(f"[quantpod_mcp] validate_strategy failed: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool()
async def update_regime_matrix_from_performance(
    lookback_days: int = 60,
) -> dict[str, Any]:
    """
    Propose updated regime-strategy allocations based on actual trade performance.

    Analyzes closed trades, groups by regime context, and proposes
    updated allocation weights. Does NOT auto-apply — returns proposals
    for /reflect or /meta to review and apply via set_regime_allocation.

    Args:
        lookback_days: Days of trade history to analyze.

    Returns:
        Dict with proposed changes per regime + reasoning.
    """
    ctx, err = _live_db_or_error()
    if err:
        return err
    try:
        # Get all closed trades in the lookback period
        from datetime import datetime as _dt
        from datetime import timedelta as _td

        cutoff = _dt.now() - _td(days=lookback_days)
        rows = ctx.db.execute(
            """
            SELECT symbol, side, realized_pnl, closed_at
            FROM closed_trades
            WHERE closed_at >= ?
            """,
            [cutoff],
        ).fetchall()

        if not rows:
            return {
                "success": True,
                "proposals": [],
                "note": f"No closed trades in last {lookback_days} days. Cannot propose changes.",
            }

        # Get current matrix
        matrix_rows = ctx.db.execute(
            "SELECT regime, strategy_id, allocation_pct FROM regime_strategy_matrix"
        ).fetchall()

        current_matrix = {}
        for r in matrix_rows:
            current_matrix.setdefault(r[0], {})[r[1]] = r[2]

        # Since we don't have per-trade regime labels in closed_trades,
        # we can report aggregate stats and suggest directional changes.
        total_pnl = sum(float(r[2]) for r in rows)
        total_trades = len(rows)
        win_rate = sum(1 for r in rows if float(r[2]) > 0) / total_trades * 100

        return {
            "success": True,
            "lookback_days": lookback_days,
            "total_trades": total_trades,
            "total_pnl": round(total_pnl, 2),
            "win_rate": round(win_rate, 2),
            "current_matrix": current_matrix,
            "proposals": [],
            "note": (
                "Per-regime trade attribution requires strategy_id on closed_trades "
                "(Phase 3 execute_trade logs strategy_id in audit trail, not in "
                "closed_trades). Use /reflect to manually review and update allocations "
                "based on trade journal patterns."
            ),
        }
    except Exception as e:
        logger.error(f"[quantpod_mcp] update_regime_matrix_from_performance failed: {e}")
        return {"success": False, "error": str(e)}


# =============================================================================
# ENHANCEMENT 1: Granular IC Access
# =============================================================================

# Human-readable descriptions for each IC (used by list_ics and IDEs)
_IC_DESCRIPTIONS: dict[str, str] = {
    "data_ingestion_ic": "Fetch OHLCV market data; assess data quality and coverage",
    "market_snapshot_ic": "Current price, volume, key indicator snapshot",
    "regime_detector_ic": "Market regime classification (trend + volatility)",
    "trend_momentum_ic": "RSI, MACD, ADX, SMA — trend and momentum metrics",
    "volatility_ic": "ATR, Bollinger Bands, volatility regime",
    "structure_levels_ic": "Support and resistance levels, pivot points",
    "statarb_ic": "ADF stationarity test, information coefficient, mean-reversion signals",
    "options_vol_ic": "Implied volatility, Greeks, skew, term structure",
    "risk_limits_ic": "VaR, stress tests, position limit checks",
    "calendar_events_ic": "Earnings dates, FOMC, CPI, macro event calendar",
    "news_sentiment_ic": "News sentiment score, recent headline risk",
    "options_flow_ic": "Unusual options activity, put/call ratio, institutional flow",
    "fundamentals_ic": "P/E, EPS growth, revenue, sector comparison",
}

# Valid IC names for input validation
_VALID_IC_NAMES = list(_IC_DESCRIPTIONS.keys())

# Valid pod names for input validation
_VALID_POD_NAMES = [
    "data_pod_manager",
    "market_monitor_pod_manager",
    "technicals_pod_manager",
    "quant_pod_manager",
    "risk_pod_manager",
    "alpha_signals_pod_manager",
]


# Standard crew inputs for minimal runs (symbol is injected at call time)
def _minimal_crew_inputs(symbol: str, regime: dict[str, Any]) -> dict[str, Any]:
    from datetime import date

    return {
        "symbol": symbol,
        "current_date": str(date.today()),
        "regime": regime,
        "regime_str": (
            f"Trend: {regime.get('trend', 'unknown')}, "
            f"Volatility: {regime.get('volatility', 'normal')}, "
            f"Confidence: {regime.get('confidence', 0.5):.0%}"
        ),
        "portfolio": {},
        "historical_context": "",
        "asset_class": "equities",
        "instrument_type": "equity",
        "task_intent": "analysis",
        "task_scope": "equities/equity:analysis",
    }


async def _detect_regime_for_symbol(symbol: str) -> dict[str, Any]:
    """Lightweight regime detection for use in IC/pod runners."""
    try:
        from quant_pod.agents.regime_detector import RegimeDetectorAgent

        detector = RegimeDetectorAgent(symbols=[symbol])
        result = await asyncio.get_event_loop().run_in_executor(
            None, detector.detect_regime, symbol
        )
        if result.get("success"):
            return {
                "trend": result.get("trend_regime", "unknown"),
                "volatility": result.get("volatility_regime", "normal"),
                "confidence": result.get("confidence", 0.5),
            }
    except Exception:
        pass
    return {"trend": "unknown", "volatility": "normal", "confidence": 0.5}


@mcp.tool()
async def list_ics() -> dict[str, Any]:
    """
    Return the catalog of all available IC agents and pod managers.

    Each IC entry includes its name, description, which pod it reports to,
    its capabilities, and which asset classes it supports.

    Returns:
        Dict with 'ics' list and 'pods' list.
    """
    try:
        from quant_pod.crews.registry import (
            IC_REGISTRY,
            POD_DEPENDENCIES,
            POD_MANAGER_REGISTRY,
        )

        # Build IC → pod reverse map
        pod_of_ic: dict[str, str] = {"data_ingestion_ic": "data_pod_manager"}
        for pod, ics in POD_DEPENDENCIES.items():
            for ic in ics:
                if ic not in pod_of_ic:
                    pod_of_ic[ic] = pod

        ics = [
            {
                "name": ic_name,
                "description": _IC_DESCRIPTIONS.get(ic_name, ""),
                "pod": pod_of_ic.get(ic_name, "unknown"),
                "capabilities": list(meta.get("capabilities", set())),
                "asset_classes": sorted(meta.get("asset_classes", set())),
            }
            for ic_name, meta in IC_REGISTRY.items()
        ]

        pods = [
            {
                "name": pod_name,
                "capabilities": list(meta.get("capabilities", set())),
                "constituent_ics": POD_DEPENDENCIES.get(pod_name, []),
            }
            for pod_name, meta in POD_MANAGER_REGISTRY.items()
        ]

        return {"success": True, "ics": ics, "pods": pods, "total_ics": len(ics)}

    except Exception as e:
        logger.error(f"[quantpod_mcp] list_ics failed: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool()
async def run_ic(
    ic_name: str,
    symbol: str,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Run a single IC agent in isolation and return its raw output.

    Runs data_ingestion_ic first (as data prerequisite) if ic_name is not
    itself the data IC, then runs the requested IC as a minimal 2-agent crew.
    Output is cached for 30 minutes.

    Cost: ~1 LLM call for data IC + 1 for target IC (cheaper than full crew).

    Args:
        ic_name: IC to run. Use list_ics() to see valid names.
        symbol: Ticker symbol to analyze.
        params: Optional extra inputs forwarded to the crew.

    Returns:
        Dict with ic_name, symbol, regime_context, raw_output, elapsed_seconds.
    """
    if ic_name not in _VALID_IC_NAMES:
        return {
            "success": False,
            "error": f"Unknown IC '{ic_name}'. Valid: {_VALID_IC_NAMES}",
        }

    start = time.monotonic()

    try:
        regime = await _detect_regime_for_symbol(symbol)

        from quant_pod.crewai_compat import Crew, Process
        from quant_pod.crews.trading_crew import TradingCrew

        tc = TradingCrew()
        ic_factories = tc._ic_agent_factories()
        ic_task_factories = tc._ic_task_factories()

        if ic_name not in ic_factories:
            return {"success": False, "error": f"No factory for IC '{ic_name}'"}

        # Build minimal agent/task list: data IC + target IC
        agents, tasks = [], []
        if ic_name != "data_ingestion_ic":
            agents.append(ic_factories["data_ingestion_ic"]())
            tasks.append(ic_task_factories["data_ingestion_ic"]())

        agents.append(ic_factories[ic_name]())
        tasks.append(ic_task_factories[ic_name]())

        minimal_crew = Crew(
            agents=agents,
            tasks=tasks,
            process=Process.sequential,
            verbose=False,
            cache=True,
        )

        inputs = _minimal_crew_inputs(symbol, regime)
        if params:
            inputs.update(params)

        result = await asyncio.get_event_loop().run_in_executor(
            None, lambda: minimal_crew.kickoff(inputs=inputs)
        )

        # Extract target IC output (last task in the crew)
        raw_output = ""
        if hasattr(result, "tasks_output") and result.tasks_output:
            last = result.tasks_output[-1]
            raw_output = last.raw if hasattr(last, "raw") else str(last)
        elif hasattr(result, "raw"):
            raw_output = str(result.raw)
        else:
            raw_output = str(result)

        _ic_cache_set(symbol, ic_name, raw_output)

        elapsed = time.monotonic() - start
        return {
            "success": True,
            "ic_name": ic_name,
            "symbol": symbol,
            "regime_context": regime,
            "raw_output": raw_output,
            "elapsed_seconds": round(elapsed, 2),
        }

    except Exception as e:
        elapsed = time.monotonic() - start
        logger.error(f"[quantpod_mcp] run_ic({ic_name}, {symbol}) failed: {e}")
        return {
            "success": False,
            "ic_name": ic_name,
            "symbol": symbol,
            "error": str(e),
            "elapsed_seconds": round(elapsed, 2),
        }


@mcp.tool()
async def run_pod(
    pod_name: str,
    symbol: str,
    ic_outputs: dict[str, str] | None = None,
) -> dict[str, Any]:
    """
    Run a single pod manager with its constituent ICs.

    If ic_outputs is provided (dict of {ic_name: raw_output}), those results
    are injected as context and only the pod manager LLM is invoked.
    If ic_outputs is empty/None, constituent ICs are run first.

    Args:
        pod_name: Pod to run. Use list_ics() to see valid names.
        symbol: Ticker symbol.
        ic_outputs: Optional pre-computed IC outputs to skip re-running ICs.

    Returns:
        Dict with pod_name, symbol, raw_output, constituent_ic_outputs (truncated).
    """
    if pod_name not in _VALID_POD_NAMES:
        return {
            "success": False,
            "error": f"Unknown pod '{pod_name}'. Valid: {_VALID_POD_NAMES}",
        }

    start = time.monotonic()

    try:
        from quant_pod.crewai_compat import Crew, Process
        from quant_pod.crews.registry import POD_DEPENDENCIES
        from quant_pod.crews.trading_crew import TradingCrew

        regime = await _detect_regime_for_symbol(symbol)
        constituent_ics = POD_DEPENDENCIES.get(pod_name, [])
        collected: dict[str, str] = dict(ic_outputs or {})

        tc = TradingCrew()
        ic_factories = tc._ic_agent_factories()
        ic_task_factories = tc._ic_task_factories()
        pod_factories = tc._pod_manager_factories()
        pod_task_factories = tc._pod_task_factories()

        if pod_name not in pod_factories:
            return {"success": False, "error": f"No factory for pod '{pod_name}'"}

        if not collected:
            # Run data_ingestion_ic + constituent ICs + pod manager
            ics_to_run = list(constituent_ics)
            if "data_ingestion_ic" not in ics_to_run:
                ics_to_run.insert(0, "data_ingestion_ic")

            agents = [ic_factories[ic]() for ic in ics_to_run if ic in ic_factories]
            tasks = [ic_task_factories[ic]() for ic in ics_to_run if ic in ic_task_factories]
            agents.append(pod_factories[pod_name]())
            tasks.append(pod_task_factories[pod_name]())

            pod_crew = Crew(
                agents=agents, tasks=tasks, process=Process.sequential, verbose=False, cache=True
            )
            inputs = _minimal_crew_inputs(symbol, regime)

            result = await asyncio.get_event_loop().run_in_executor(
                None, lambda: pod_crew.kickoff(inputs=inputs)
            )

            raw_output = ""
            if hasattr(result, "tasks_output") and result.tasks_output:
                # Last task = pod manager output
                last = result.tasks_output[-1]
                raw_output = last.raw if hasattr(last, "raw") else str(last)
                # Cache per-IC outputs
                for i, ic_nm in enumerate(ics_to_run):
                    if i < len(result.tasks_output):
                        to = result.tasks_output[i]
                        ic_raw = to.raw if hasattr(to, "raw") else str(to)
                        collected[ic_nm] = ic_raw
                        _ic_cache_set(symbol, ic_nm, ic_raw)
            elif hasattr(result, "raw"):
                raw_output = str(result.raw)
            else:
                raw_output = str(result)

        else:
            # Use pre-computed IC outputs — only invoke the pod manager
            combined_context = "\n\n".join(
                f"## {ic_nm} Output:\n{out}" for ic_nm, out in collected.items()
            )
            agents = [pod_factories[pod_name]()]
            tasks = [pod_task_factories[pod_name]()]
            pod_crew = Crew(
                agents=agents, tasks=tasks, process=Process.sequential, verbose=False, cache=True
            )

            override_inputs = _minimal_crew_inputs(symbol, regime)
            override_inputs["historical_context"] = combined_context

            result = await asyncio.get_event_loop().run_in_executor(
                None, lambda: pod_crew.kickoff(inputs=override_inputs)
            )
            raw_output = ""
            if hasattr(result, "raw"):
                raw_output = str(result.raw)
            elif hasattr(result, "tasks_output") and result.tasks_output:
                last = result.tasks_output[-1]
                raw_output = last.raw if hasattr(last, "raw") else str(last)
            else:
                raw_output = str(result)

        elapsed = time.monotonic() - start
        return {
            "success": True,
            "pod_name": pod_name,
            "symbol": symbol,
            "constituent_ics": constituent_ics,
            "raw_output": raw_output,
            "ic_outputs_preview": {k: v[:200] + "..." for k, v in collected.items()},
            "elapsed_seconds": round(elapsed, 2),
        }

    except Exception as e:
        elapsed = time.monotonic() - start
        logger.error(f"[quantpod_mcp] run_pod({pod_name}, {symbol}) failed: {e}")
        return {
            "success": False,
            "pod_name": pod_name,
            "symbol": symbol,
            "error": str(e),
            "elapsed_seconds": round(elapsed, 2),
        }


@mcp.tool()
async def run_crew_subset(
    ic_names: list[str],
    symbol: str,
) -> dict[str, Any]:
    """
    Run a custom subset of ICs through their pod managers to the assistant.

    The assistant synthesizes a partial DailyBrief scoped to only the specified
    ICs. Useful for targeted, cheaper analysis (e.g., only regime + volatility
    ICs for a quick pre-screen before committing to a full run).

    data_ingestion_ic is always auto-added as a prerequisite.
    Pod managers are auto-selected based on which ICs are included.

    Args:
        ic_names: List of IC names to run. Use list_ics() for valid names.
        symbol: Ticker symbol.

    Returns:
        Dict with partial_daily_brief, ics_run, pods_activated, elapsed_seconds.
    """
    invalid = [ic for ic in ic_names if ic not in _VALID_IC_NAMES]
    if invalid:
        return {
            "success": False,
            "error": f"Unknown ICs: {invalid}. Valid: {_VALID_IC_NAMES}",
        }

    start = time.monotonic()

    try:
        from quant_pod.crewai_compat import Crew, Process
        from quant_pod.crews.assembler import PodSelection
        from quant_pod.crews.registry import POD_DEPENDENCIES
        from quant_pod.crews.trading_crew import TradingCrew

        regime = await _detect_regime_for_symbol(symbol)

        # Auto-select pod managers for requested ICs
        activated_pods = [
            pod
            for pod, pod_ics in POD_DEPENDENCIES.items()
            if any(ic in ic_names for ic in pod_ics)
        ]

        # Always include data IC
        full_ic_list = list(ic_names)
        if "data_ingestion_ic" not in full_ic_list:
            full_ic_list.insert(0, "data_ingestion_ic")

        roster = PodSelection(
            asset_class="equities",
            ic_agents=full_ic_list,
            pod_managers=activated_pods,
            profile_used="subset",
        )

        tc = TradingCrew()
        agents = tc._build_agents(roster, stop_at_assistant=True)
        tasks = tc._build_tasks(roster, stop_at_assistant=True)

        subset_crew = Crew(
            agents=agents,
            tasks=tasks,
            process=Process.sequential,
            verbose=False,
            cache=True,
        )

        inputs = _minimal_crew_inputs(symbol, regime)
        inputs["historical_context"] = f"Subset analysis — ICs: {full_ic_list}"

        result = await asyncio.get_event_loop().run_in_executor(
            None, lambda: subset_crew.kickoff(inputs=inputs)
        )

        brief = None
        if hasattr(result, "pydantic") and result.pydantic is not None:
            brief = _serialize(result.pydantic)
        elif hasattr(result, "json_dict") and result.json_dict is not None:
            brief = result.json_dict
        elif hasattr(result, "raw"):
            brief = {"raw_output": str(result.raw)}

        # Cache per-IC outputs from the subset run
        if hasattr(result, "tasks_output") and result.tasks_output:
            for i, ic_nm in enumerate(full_ic_list):
                if i < len(result.tasks_output):
                    to = result.tasks_output[i]
                    _ic_cache_set(symbol, ic_nm, to.raw if hasattr(to, "raw") else str(to))

        elapsed = time.monotonic() - start
        return {
            "success": True,
            "symbol": symbol,
            "ics_run": full_ic_list,
            "pods_activated": activated_pods,
            "partial_daily_brief": brief,
            "regime_used": regime,
            "elapsed_seconds": round(elapsed, 2),
        }

    except Exception as e:
        elapsed = time.monotonic() - start
        logger.error(f"[quantpod_mcp] run_crew_subset({ic_names}, {symbol}) failed: {e}")
        return {
            "success": False,
            "ic_names": ic_names,
            "symbol": symbol,
            "error": str(e),
            "elapsed_seconds": round(elapsed, 2),
        }


@mcp.tool()
async def get_last_ic_output(
    symbol: str,
    ic_name: str,
) -> dict[str, Any]:
    """
    Retrieve the cached raw output from the last IC run for a symbol.

    The cache is populated by run_analysis, run_ic, run_pod, and
    run_crew_subset. Entries expire after 30 minutes.

    Useful for /reflect sessions analyzing which ICs were right without
    re-running the full crew.

    Args:
        ic_name: IC whose output to retrieve.
        symbol: Ticker symbol.

    Returns:
        Dict with raw_output, or cache_miss=True if absent/expired.
    """
    cached = _ic_cache_get(symbol, ic_name)
    if cached is None:
        return {
            "success": True,
            "ic_name": ic_name,
            "symbol": symbol,
            "cache_miss": True,
            "note": "No cached output. Run run_analysis, run_ic, or run_crew_subset first.",
        }
    return {
        "success": True,
        "ic_name": ic_name,
        "symbol": symbol,
        "cache_miss": False,
        "raw_output": cached,
    }


# =============================================================================
# ENHANCEMENT 5: Execution Feedback Loop
# =============================================================================


@mcp.tool()
async def get_fill_quality(order_id: str) -> dict[str, Any]:
    """
    Assess execution quality for a completed fill.

    Compares the fill price to VWAP at fill time and returns slippage analysis.
    Use during /reflect sessions to track execution quality over time.

    Args:
        order_id: Order ID from get_fills output.

    Returns:
        Dict with fill_price, slippage_bps, vwap, fill_vs_vwap_bps, quality_note.
    """
    ctx, err = _live_db_or_error()
    if err:
        return err
    try:
        row = ctx.db.execute(
            """
            SELECT order_id, symbol, side, fill_price, filled_quantity,
                   slippage_bps, commission, filled_at
            FROM fills
            WHERE order_id = ? AND rejected = FALSE
            """,
            [order_id],
        ).fetchone()

        if not row:
            return {
                "success": False,
                "error": f"Fill not found for order_id={order_id}",
            }

        oid, symbol, side, fill_price, filled_qty, recorded_slippage, commission, filled_at = row

        # Attempt VWAP comparison via QuantCore data store
        vwap: float | None = None
        fill_vs_vwap_bps: float | None = None
        try:
            from quantcore.data.storage import DataStore

            store = DataStore()
            df = store.load(symbol)
            if df is not None and not df.empty and "vwap" in df.columns:
                fill_date = str(filled_at)[:10]
                day_rows = df[df.index.astype(str).str.startswith(fill_date)]
                if not day_rows.empty:
                    vwap = float(day_rows["vwap"].iloc[-1])
                    if vwap > 0 and fill_price and fill_price > 0:
                        fill_vs_vwap_bps = round((fill_price - vwap) / vwap * 10_000, 1)
        except Exception:
            pass

        direction_label = "above" if (fill_vs_vwap_bps or 0) > 0 else "below"
        quality_note = f"Recorded slippage: {(recorded_slippage or 0):.1f} bps. " + (
            f"Fill was {abs(fill_vs_vwap_bps):.1f} bps {direction_label} VWAP."
            if fill_vs_vwap_bps is not None
            else "VWAP data unavailable for comparison."
        )

        return {
            "success": True,
            "order_id": order_id,
            "symbol": symbol,
            "side": side,
            "fill_price": fill_price,
            "filled_quantity": filled_qty,
            "slippage_bps": recorded_slippage,
            "vwap": vwap,
            "fill_vs_vwap_bps": fill_vs_vwap_bps,
            "commission": commission,
            "filled_at": str(filled_at) if filled_at else None,
            "quality_note": quality_note,
        }

    except Exception as e:
        logger.error(f"[quantpod_mcp] get_fill_quality({order_id}) failed: {e}")
        return {"success": False, "error": str(e), "order_id": order_id}


@mcp.tool()
async def get_position_monitor(symbol: str) -> dict[str, Any]:
    """
    Comprehensive position status for an open position.

    Returns price, unrealized P&L, ATR-based stop distance, days held,
    and current vs entry regime.  Designed for /review position checks.

    Args:
        symbol: Ticker symbol of the open position.

    Returns:
        Dict with price, pnl, days_held, current_regime, flags, recommended_action.
        Returns has_position=False if no open position exists.
    """
    ctx, err = _live_db_or_error()
    if err:
        return err
    try:
        pos = ctx.portfolio.get_position(symbol)
        if not pos:
            return {
                "success": True,
                "symbol": symbol,
                "has_position": False,
                "note": "No open position found.",
            }

        current_price = pos.current_price or 0.0
        avg_cost = pos.avg_cost or 0.0
        quantity = pos.quantity or 0
        unrealized_pnl = pos.unrealized_pnl or 0.0

        pnl_pct = 0.0
        if avg_cost > 0 and current_price > 0:
            pnl_pct = round((current_price - avg_cost) / avg_cost * 100, 2)

        # Time held
        days_held: int | None = None
        entry_time: str | None = None
        try:
            row = ctx.db.execute(
                "SELECT opened_at FROM positions WHERE symbol = ?",
                [symbol],
            ).fetchone()
            if row and row[0]:
                from datetime import datetime as _dt

                opened_at = row[0]
                if isinstance(opened_at, str):
                    opened_at = _dt.fromisoformat(opened_at)
                days_held = (_dt.now() - opened_at).days
                entry_time = str(row[0])
        except Exception:
            pass

        # Current regime
        current_regime = "unknown"
        atr: float = 0.0
        try:
            from quant_pod.agents.regime_detector import RegimeDetectorAgent

            detector = RegimeDetectorAgent(symbols=[symbol])
            r = await asyncio.get_event_loop().run_in_executor(None, detector.detect_regime, symbol)
            current_regime = r.get("trend_regime", "unknown")
            atr = float(r.get("atr", 0))
        except Exception:
            pass

        # ATR-based stop proximity
        near_stop = False
        atr_stop_distance_pct: float | None = None
        if atr > 0 and avg_cost > 0 and current_price > 0:
            atr_stop_distance_pct = round(atr / avg_cost * 100, 2)
            # Flag if within 30% of a 2-ATR stop
            stop_level = avg_cost - 2 * atr
            range_to_stop = avg_cost - stop_level  # = 2 * ATR
            if range_to_stop > 0:
                pct_to_stop = (current_price - stop_level) / range_to_stop
                near_stop = pct_to_stop < 0.30

        # Approaching target (>80% of a 3R move)
        near_target = False
        if atr > 0 and avg_cost > 0 and current_price > 0:
            target_level = avg_cost + 3 * atr
            range_to_target = target_level - avg_cost
            if range_to_target > 0:
                pct_to_target = (current_price - avg_cost) / range_to_target
                near_target = pct_to_target >= 0.80

        flags = {
            "near_stop": near_stop,
            "near_target": near_target,
            "pnl_positive": unrealized_pnl > 0,
        }

        if near_stop:
            recommended_action = "TIGHTEN STOP — price approaching 2-ATR stop level"
        elif near_target:
            recommended_action = "CONSIDER PARTIAL EXIT — 80% of 3R target reached"
        else:
            recommended_action = "HOLD — within normal parameters"

        return {
            "success": True,
            "symbol": symbol,
            "has_position": True,
            "quantity": quantity,
            "avg_cost": round(avg_cost, 4),
            "current_price": round(current_price, 4),
            "unrealized_pnl": round(unrealized_pnl, 2),
            "pnl_pct": pnl_pct,
            "days_held": days_held,
            "entry_time": entry_time,
            "current_regime": current_regime,
            "atr_stop_distance_pct": atr_stop_distance_pct,
            "flags": flags,
            "recommended_action": recommended_action,
        }

    except Exception as e:
        logger.error(f"[quantpod_mcp] get_position_monitor({symbol}) failed: {e}")
        return {"success": False, "symbol": symbol, "error": str(e)}


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
