"""Meta orchestration tools for LangGraph agents."""

import json
from typing import Annotated, Any

from langchain_core.tools import tool
from loguru import logger
from pydantic import Field

from quantstack.db import db_conn


def _get_breaker_status(strategy_id: str) -> str:
    """Get breaker status label for a strategy. Fail-safe."""
    try:
        from quantstack.execution.strategy_breaker import StrategyBreaker
        breaker = StrategyBreaker()
        factor = breaker.get_scale_factor(strategy_id)
        if factor == 0.0:
            return "TRIPPED"
        if factor < 1.0:
            return "SCALED"
        return "ACTIVE"
    except Exception as exc:
        logger.warning(f"[meta_tools] Breaker check failed for {strategy_id}: {exc}")
        return "UNKNOWN"


@tool
async def get_regime_strategies(
    regime: Annotated[str, Field(description="Market regime label such as 'trending_up', 'trending_down', 'ranging', or 'unknown'. Determines which strategy set is returned.")],
) -> str:
    """Retrieves strategy allocations for a given market regime from the regime-strategy matrix. Use when you need to look up which trading strategies are active, their allocation percentages, and confidence scores for a specific regime. Returns a JSON list of (strategy_id, allocation_pct, confidence) tuples. Covers regime mapping, strategy selection, portfolio allocation, and market condition routing."""
    try:
        with db_conn() as conn:
            rows = conn.fetchall(
                "SELECT strategy_id, name, status, regime_affinity "
                "FROM strategies WHERE status != 'retired'"
            )
    except Exception as exc:
        logger.warning(f"[meta_tools] get_regime_strategies DB query failed: {exc}")
        return json.dumps({"error": "Failed to query strategies", "strategies": []})

    # Filter by regime affinity and sort descending
    strategies = []
    for row in rows:
        affinity_data = row.get("regime_affinity") or {}
        if isinstance(affinity_data, str):
            try:
                affinity_data = json.loads(affinity_data)
            except (json.JSONDecodeError, TypeError):
                affinity_data = {}
        affinity_score = affinity_data.get(regime, 0)
        if affinity_score <= 0:
            continue
        strategies.append({
            "strategy_id": row["strategy_id"],
            "name": row.get("name", ""),
            "affinity": affinity_score,
            "breaker_status": _get_breaker_status(row["strategy_id"]),
        })

    strategies.sort(key=lambda s: s["affinity"], reverse=True)
    return json.dumps({"strategies": strategies}, default=str)


@tool
async def set_regime_allocation(
    regime: Annotated[str, Field(description="Market regime label to update, e.g. 'trending_up', 'trending_down', 'ranging', 'unknown'.")],
    allocations: Annotated[list[dict[str, Any]], Field(description="List of allocation dicts, each containing 'strategy_id' (str), 'allocation_pct' (float 0-100), and optionally 'confidence' (float 0-1).")],
) -> str:
    """Sets or updates strategy allocations for a specific market regime by upserting into the regime-strategy matrix. Use when reflecting on performance data to rebalance which strategies are active per regime. Provides regime rotation, allocation adjustment, strategy weighting, and portfolio rebalancing. Returns JSON with the updated allocation entries."""
    result = {"error": "Tool pending implementation", "status": "not_available"}
    return json.dumps(result, default=str)


@tool
async def resolve_portfolio_conflicts(
    proposed_trades: Annotated[list[dict[str, Any]], Field(description="List of proposed trade dicts, each containing 'symbol', 'action' (buy/sell), 'confidence' (float 0-1), 'strategy_id', and 'capital_pct' (float). Conflicting signals on the same symbol are resolved automatically.")],
) -> str:
    """Resolves signal conflicts across multiple strategies proposing trades on the same symbols. Use when aggregating trade proposals from different strategies before execution. Provides conflict resolution, signal merging, position deduplication, and conservative sizing for overlapping orders. Returns JSON with resolved_trades, per-conflict resolutions, and conflicts_count."""
    result = {"error": "Tool pending implementation", "status": "not_available"}
    return json.dumps(result, default=str)


@tool
async def get_strategy_gaps() -> str:
    """Analyzes the strategy registry for coverage gaps across all market regimes. Use when identifying which regimes lack live or forward-testing strategies, have degraded Sharpe ratios below 0.3, or suffer concentration risk with only one strategy. Provides gap analysis, coverage audit, regime exposure mapping, and research prioritization. Returns JSON with critical gaps, coverage_summary, and per-regime strategy counts."""
    result = {"error": "Tool pending implementation", "status": "not_available"}
    return json.dumps(result, default=str)


@tool
async def promote_draft_strategies(
    min_oos_sharpe: Annotated[float, Field(default=0.6, description="Minimum out-of-sample Sharpe ratio required for promotion. Strategies below this threshold are rejected.")],
    max_overfit_ratio: Annotated[float, Field(default=2.0, description="Maximum in-sample/out-of-sample Sharpe ratio. Values above this indicate overfitting and block promotion.")],
    max_age_days: Annotated[int, Field(default=14, description="Maximum age in days for draft strategies. Drafts older than this are automatically retired instead of promoted.")],
) -> str:
    """Evaluates all draft strategies for auto-promotion to forward_testing status using walk-forward validation gates. Use when running the strategy lifecycle pipeline to advance promising drafts. Computes OOS Sharpe, overfit ratio, walk-forward degradation, and staleness checks. Provides strategy promotion, lifecycle management, backtest validation, and overfitting detection. Returns JSON with promoted, rejected, and retired strategy lists. Never promotes directly to live."""
    result = {"error": "Tool pending implementation", "status": "not_available"}
    return json.dumps(result, default=str)


@tool
async def check_strategy_rules(
    symbol: Annotated[str, Field(description="Ticker symbol to evaluate against current market data, e.g. 'AAPL', 'SPY'.")],
    strategy_id: Annotated[str, Field(description="Strategy identifier whose entry/exit rules are loaded from the database and evaluated.")],
) -> str:
    """Evaluates a strategy's entry and exit rules against current live market data using the same FeatureEnricher and rule evaluator as backtesting to avoid train/serve skew. Use when checking whether a strategy's signals are triggered for a specific ticker right now. Computes technical indicators, enriched features on 252 daily bars, and returns per-rule evaluation details. Provides signal checking, rule evaluation, entry detection, exit detection, and trade trigger analysis. Returns JSON with entry_triggered, exit_triggered, and per-rule pass/fail details."""
    result = {"error": "Tool pending implementation", "status": "not_available"}
    return json.dumps(result, default=str)
