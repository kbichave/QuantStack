"""Portfolio optimization MCP tools.

Tools:
  - optimize_portfolio   — multi-method portfolio optimization (HRP, min-var, risk parity, max Sharpe, equal weight)
  - compute_hrp_weights  — dedicated HRP with cluster tree and full risk decomposition
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.optimize import minimize

from quantstack.config.settings import get_settings
from quantstack.config.timeframes import Timeframe
from quantstack.data.base import AssetClass
from quantstack.data.registry import DataProviderRegistry
from quantstack.data.storage import DataStore  # noqa: F401
from quantstack.mcp._helpers import _get_reader
from quantstack.mcp.server import mcp
from quantstack.mcp.domains import Domain
from quantstack.mcp.tools._registry import domain



# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def _load_returns(symbols: list[str], lookback_days: int) -> pd.DataFrame:
    """Load daily close prices for *symbols* and return a log-returns DataFrame.

    Uses the same resolution chain as the backtesting tools:
      1. Local DuckDB cache (read-only)
      2. Provider registry (DATA_PROVIDER_PRIORITY)

    Raises ValueError when fewer than 2 symbols have sufficient data.
    """
    end_dt = datetime.now()
    start_dt = end_dt - timedelta(
        days=int(lookback_days * 1.5)
    )  # overshoot for weekends/holidays

    closes: dict[str, pd.Series] = {}

    for symbol in symbols:
        df: pd.DataFrame | None = None

        # 1. DuckDB cache
        try:
            store = _get_reader()
            df = store.load_ohlcv(symbol, Timeframe.D1)
        except Exception as exc:
            logger.debug(f"DuckDB miss for {symbol}: {exc}")

        # 2. Provider registry fallback
        if df is None or df.empty:
            try:
                settings = get_settings()
                registry = DataProviderRegistry.from_settings(settings)
                df = registry.fetch_ohlcv(
                    symbol, AssetClass.EQUITY, Timeframe.D1, start_dt, end_dt
                )
            except Exception as exc:
                logger.warning(f"Provider fetch failed for {symbol}: {exc}")

        if df is not None and not df.empty:
            col = "close" if "close" in df.columns else "Close"
            series = df[col].dropna()
            if len(series) >= lookback_days // 2:
                closes[symbol] = series

    if len(closes) < 2:
        raise ValueError(
            f"Need price data for at least 2 symbols; got {list(closes.keys())}"
        )

    prices = pd.DataFrame(closes).dropna().tail(lookback_days)
    returns = np.log(prices / prices.shift(1)).dropna()

    if len(returns) < 20:
        raise ValueError(
            f"Only {len(returns)} overlapping return observations — need at least 20"
        )

    return returns


# ---------------------------------------------------------------------------
# HRP internals (López de Prado, 2016)
# ---------------------------------------------------------------------------


def _hrp_correlation_distance(corr: pd.DataFrame) -> np.ndarray:
    """Distance matrix: d_ij = sqrt(0.5 * (1 - rho_ij))."""
    dist = np.sqrt(0.5 * (1.0 - corr.values))
    np.fill_diagonal(dist, 0.0)
    return dist


def _hrp_quasi_diag(link: np.ndarray) -> list[int]:
    """Reorder assets using the dendrogram leaf order (quasi-diagonalization)."""
    return list(leaves_list(link))


def _hrp_recursive_bisection(
    cov: pd.DataFrame,
    sorted_idx: list[int],
) -> pd.Series:
    """Allocate by inverse variance via recursive bisection of the sorted index."""
    weights = pd.Series(1.0, index=sorted_idx)
    cluster_items = [sorted_idx]

    while cluster_items:
        next_level: list[list[int]] = []
        for subset in cluster_items:
            if len(subset) <= 1:
                continue
            mid = len(subset) // 2
            left, right = subset[:mid], subset[mid:]

            left_var = _cluster_variance(cov, left)
            right_var = _cluster_variance(cov, right)
            alloc_left = 1.0 - left_var / (left_var + right_var)

            for i in left:
                weights[i] *= alloc_left
            for i in right:
                weights[i] *= 1.0 - alloc_left

            next_level.extend([left, right])
        cluster_items = next_level

    return weights


def _cluster_variance(cov: pd.DataFrame, indices: list[int]) -> float:
    """Inverse-variance–weighted portfolio variance for a cluster."""
    sub_cov = cov.iloc[indices, indices].values
    diag = np.diag(sub_cov)
    if np.any(diag < 1e-9):
        raise ValueError(f"Degenerate covariance matrix: zero/near-zero variance in cluster {indices}")
    inv_diag = 1.0 / diag
    inv_diag /= inv_diag.sum()
    return float(inv_diag @ sub_cov @ inv_diag)


def _run_hrp(returns: pd.DataFrame) -> tuple[dict[str, float], np.ndarray, list[int]]:
    """Full HRP pipeline.  Returns (weights_dict, linkage_matrix, leaf_order)."""
    corr = returns.corr()
    cov = returns.cov()
    dist = _hrp_correlation_distance(corr)

    # Condensed distance for scipy (upper-triangle, row-major)
    n = len(dist)
    condensed = dist[np.triu_indices(n, k=1)]
    link = linkage(condensed, method="ward")

    order = _hrp_quasi_diag(link)
    raw_weights = _hrp_recursive_bisection(cov, order)

    # Map integer indices back to symbol names
    symbols = returns.columns.tolist()
    weights = {symbols[i]: float(raw_weights[i]) for i in order}
    total = sum(weights.values())
    weights = {s: w / total for s, w in weights.items()}

    return weights, link, order


# ---------------------------------------------------------------------------
# Other optimizers
# ---------------------------------------------------------------------------


def _min_variance(returns: pd.DataFrame) -> dict[str, float]:
    cov = returns.cov().values
    n = cov.shape[0]
    x0 = np.ones(n) / n

    result = minimize(
        fun=lambda w: float(w @ cov @ w),
        x0=x0,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n,
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1.0},
    )
    w = result.x / result.x.sum()
    return dict(zip(returns.columns, w.tolist()))


def _risk_parity(returns: pd.DataFrame) -> dict[str, float]:
    """Equal risk contribution — each asset contributes the same marginal variance."""
    cov = returns.cov().values
    n = cov.shape[0]
    x0 = np.ones(n) / n
    target_rc = 1.0 / n

    def objective(w: np.ndarray) -> float:
        port_var = w @ cov @ w
        marginal = cov @ w
        rc = w * marginal / port_var
        return float(((rc - target_rc) ** 2).sum())

    result = minimize(
        fun=objective,
        x0=x0,
        method="SLSQP",
        bounds=[(1e-6, 1.0)] * n,
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1.0},
    )
    w = result.x / result.x.sum()
    return dict(zip(returns.columns, w.tolist()))


def _max_sharpe(returns: pd.DataFrame, risk_free_rate: float) -> dict[str, float]:
    mu = returns.mean().values * 252
    cov = returns.cov().values * 252
    n = cov.shape[0]
    x0 = np.ones(n) / n
    rf = risk_free_rate

    def neg_sharpe(w: np.ndarray) -> float:
        port_ret = w @ mu
        port_vol = np.sqrt(w @ cov @ w)
        return -(port_ret - rf) / max(port_vol, 1e-10)

    result = minimize(
        fun=neg_sharpe,
        x0=x0,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n,
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1.0},
    )
    w = result.x / result.x.sum()
    return dict(zip(returns.columns, w.tolist()))


# ---------------------------------------------------------------------------
# Portfolio statistics
# ---------------------------------------------------------------------------


def _portfolio_stats(
    weights: dict[str, float],
    returns: pd.DataFrame,
    risk_free_rate: float,
) -> dict[str, float]:
    """Compute expected return, volatility, Sharpe, diversification ratio, and risk contributions."""
    syms = list(weights.keys())
    w = np.array([weights[s] for s in syms])
    mu = returns[syms].mean().values * 252
    cov = returns[syms].cov().values * 252

    port_ret = float(w @ mu)
    port_vol = float(np.sqrt(w @ cov @ w))
    sharpe = (port_ret - risk_free_rate) / max(port_vol, 1e-10)

    # Diversification ratio = weighted avg of individual vols / portfolio vol
    ind_vols = np.sqrt(np.diag(cov))
    div_ratio = float((w @ ind_vols) / max(port_vol, 1e-10))

    # Risk contributions (% of portfolio variance)
    marginal = cov @ w
    rc = w * marginal
    rc_total = rc.sum()
    risk_contributions = {
        s: round(float(rc[i] / rc_total) * 100, 2) for i, s in enumerate(syms)
    }

    return {
        "expected_return": round(port_ret, 4),
        "expected_volatility": round(port_vol, 4),
        "sharpe_ratio": round(sharpe, 4),
        "diversification_ratio": round(div_ratio, 4),
        "risk_contributions": risk_contributions,
    }


# ---------------------------------------------------------------------------
# MCP Tools
# ---------------------------------------------------------------------------

_VALID_METHODS = {"hrp", "min_variance", "risk_parity", "max_sharpe", "equal_weight"}


@domain(Domain.PORTFOLIO)
@mcp.tool()
async def optimize_portfolio(
    symbols: list[str],
    method: str = "hrp",
    lookback_days: int = 252,
    risk_free_rate: float = 0.05,
) -> dict[str, Any]:
    """
    Optimize portfolio weights across multiple assets.

    Args:
        symbols: List of ticker symbols (min 2).
        method: Optimization method — "hrp" | "min_variance" | "risk_parity" | "max_sharpe" | "equal_weight".
        lookback_days: Number of trading days for return estimation.
        risk_free_rate: Annualized risk-free rate for Sharpe calculation.

    Returns:
        Dict with weights, expected return/volatility, Sharpe ratio,
        diversification ratio, and per-asset risk contributions.
    """
    if method not in _VALID_METHODS:
        return {
            "success": False,
            "error": f"Unknown method '{method}'. Choose from: {sorted(_VALID_METHODS)}",
        }
    if len(symbols) < 2:
        return {
            "success": False,
            "error": "Need at least 2 symbols for portfolio optimization",
        }

    def _compute() -> dict[str, Any]:
        returns = _load_returns(symbols, lookback_days)

        if method == "hrp":
            weights, _, _ = _run_hrp(returns)
        elif method == "min_variance":
            weights = _min_variance(returns)
        elif method == "risk_parity":
            weights = _risk_parity(returns)
        elif method == "max_sharpe":
            weights = _max_sharpe(returns, risk_free_rate)
        else:  # equal_weight
            n = len(returns.columns)
            weights = {s: round(1.0 / n, 6) for s in returns.columns}

        # Round weights for readability
        weights = {s: round(w, 6) for s, w in weights.items()}

        stats = _portfolio_stats(weights, returns, risk_free_rate)
        return {
            "success": True,
            "method": method,
            "lookback_days": len(returns),
            "symbols_used": list(weights.keys()),
            "weights": weights,
            **stats,
        }

    try:
        return await asyncio.to_thread(_compute)
    except Exception as exc:
        logger.error(f"optimize_portfolio failed: {exc}")
        return {"success": False, "error": str(exc)}


@domain(Domain.PORTFOLIO)
@mcp.tool()
async def compute_hrp_weights(
    symbols: list[str],
    lookback_days: int = 252,
) -> dict[str, Any]:
    """
    Compute Hierarchical Risk Parity weights with full cluster decomposition.

    Uses the López de Prado (2016) algorithm:
      1. Compute correlation-based distance matrix  d = sqrt(0.5*(1 - rho))
      2. Ward linkage clustering
      3. Quasi-diagonalize the covariance matrix via dendrogram leaf order
      4. Recursive bisection — allocate by inverse variance at each split

    Args:
        symbols: List of ticker symbols (min 2).
        lookback_days: Number of trading days for covariance estimation.

    Returns:
        Dict with weights, cluster linkage matrix (for visualization),
        leaf order, correlation summary, and per-asset risk contributions.
    """
    if len(symbols) < 2:
        return {"success": False, "error": "Need at least 2 symbols for HRP"}

    def _compute() -> dict[str, Any]:
        returns = _load_returns(symbols, lookback_days)
        weights, link, order = _run_hrp(returns)
        weights = {s: round(w, 6) for s, w in weights.items()}

        corr = returns.corr()
        cov = returns.cov() * 252

        # Build cluster merge list for the caller (readable form of linkage matrix)
        syms = returns.columns.tolist()
        cluster_merges: list[dict[str, Any]] = []
        for row in link:
            left_idx, right_idx, dist, count = (
                int(row[0]),
                int(row[1]),
                float(row[2]),
                int(row[3]),
            )
            left_label = (
                syms[left_idx] if left_idx < len(syms) else f"cluster_{left_idx}"
            )
            right_label = (
                syms[right_idx] if right_idx < len(syms) else f"cluster_{right_idx}"
            )
            cluster_merges.append(
                {
                    "left": left_label,
                    "right": right_label,
                    "distance": round(dist, 4),
                    "size": count,
                }
            )

        leaf_order = [syms[i] for i in order]

        # Per-asset annualized vol
        ann_vols = {s: round(float(np.sqrt(cov.loc[s, s])), 4) for s in syms}

        # Pairwise correlation summary (top/bottom 3)
        pairs: list[tuple[float, str, str]] = []
        for i, s1 in enumerate(syms):
            for j, s2 in enumerate(syms):
                if i < j:
                    pairs.append((float(corr.loc[s1, s2]), s1, s2))
        pairs.sort()
        lowest_corr = [
            {"pair": f"{s1}/{s2}", "correlation": round(c, 4)}
            for c, s1, s2 in pairs[:3]
        ]
        highest_corr = [
            {"pair": f"{s1}/{s2}", "correlation": round(c, 4)}
            for c, s1, s2 in pairs[-3:]
        ]

        stats = _portfolio_stats(weights, returns, risk_free_rate=0.05)

        return {
            "success": True,
            "method": "hrp",
            "lookback_days": len(returns),
            "symbols_used": leaf_order,
            "weights": weights,
            "leaf_order": leaf_order,
            "cluster_merges": cluster_merges,
            "annualized_vols": ann_vols,
            "lowest_correlations": lowest_corr,
            "highest_correlations": highest_corr,
            **stats,
        }

    try:
        return await asyncio.to_thread(_compute)
    except Exception as exc:
        logger.error(f"compute_hrp_weights failed: {exc}")
        return {"success": False, "error": str(exc)}
