"""Meta orchestration tools for LangGraph agents."""

import json
from typing import Any

from langchain_core.tools import tool


@tool
async def get_regime_strategies(regime: str) -> str:
    """Get strategy allocations for a given regime from the matrix.

    Args:
        regime: Regime label (e.g., "trending_up", "ranging").

    Returns JSON with list of (strategy_id, allocation_pct, confidence).
    """
    result = {"error": "Tool pending implementation", "status": "not_available"}
    return json.dumps(result, default=str)


@tool
async def set_regime_allocation(
    regime: str,
    allocations: list[dict[str, Any]],
) -> str:
    """Set or update strategy allocations for a regime.

    Upserts into the regime_strategy_matrix. This is how /reflect updates
    the matrix based on accumulated performance data.

    Args:
        regime: Regime label.
        allocations: List of dicts with strategy_id, allocation_pct, confidence (optional).

    Returns JSON with the updated allocations.
    """
    result = {"error": "Tool pending implementation", "status": "not_available"}
    return json.dumps(result, default=str)


@tool
async def resolve_portfolio_conflicts(
    proposed_trades: list[dict[str, Any]],
) -> str:
    """Resolve signal conflicts across multiple strategies for the same symbols.

    Rules:
      - Same symbol, different directions: high confidence wins, or SKIP if both high
      - Same symbol, same direction: merge with conservative sizing

    Args:
        proposed_trades: List of trade dicts, each with:
            symbol, action, confidence, strategy_id, capital_pct.

    Returns JSON with resolved_trades, resolutions, conflicts_count.
    """
    result = {"error": "Tool pending implementation", "status": "not_available"}
    return json.dumps(result, default=str)


@tool
async def get_strategy_gaps() -> str:
    """Analyze the strategy registry for coverage gaps.

    Identifies regimes where:
    - No live or forward_testing strategy exists (critical)
    - The best strategy has trailing Sharpe < 0.3 (degraded)
    - Only one strategy covers the regime (concentration risk)

    Used by the Strategy Factory loop to target research at the regimes
    that need it most.

    Returns JSON with gaps, coverage_summary, and strategy counts.
    """
    result = {"error": "Tool pending implementation", "status": "not_available"}
    return json.dumps(result, default=str)


@tool
async def promote_draft_strategies(
    min_oos_sharpe: float = 0.6,
    max_overfit_ratio: float = 2.0,
    max_age_days: int = 14,
) -> str:
    """Evaluate all draft strategies for auto-promotion to forward_testing.

    Criteria (ALL must pass):
    1. OOS Sharpe mean >= min_oos_sharpe (from walk-forward summary)
    2. Overfit ratio < max_overfit_ratio
    3. Walk-forward degradation < 30%
    4. Created within max_age_days (stale drafts are retired instead)

    NEVER promotes to live. Only draft -> forward_testing.

    Args:
        min_oos_sharpe: Minimum OOS Sharpe for promotion (default 0.6).
        max_overfit_ratio: Maximum IS/OOS Sharpe ratio (default 2.0).
        max_age_days: Drafts older than this are retired (default 14).

    Returns JSON with promoted, rejected, and retired strategy lists.
    """
    result = {"error": "Tool pending implementation", "status": "not_available"}
    return json.dumps(result, default=str)


@tool
async def check_strategy_rules(symbol: str, strategy_id: str) -> str:
    """Evaluate a strategy's entry/exit rules against CURRENT market data.

    Uses the same FeatureEnricher and rule evaluator as backtesting -- no
    train/serve skew. Loads the latest 252 daily bars, computes technical
    indicators + enriched features, then evaluates each rule on the latest bar.

    Args:
        symbol: Ticker symbol to evaluate.
        strategy_id: Strategy to check (loads entry/exit rules from DB).

    Returns JSON with entry_triggered, exit_triggered, and per-rule details.
    """
    result = {"error": "Tool pending implementation", "status": "not_available"}
    return json.dumps(result, default=str)
