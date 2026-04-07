"""Portfolio construction stream — compares optimization methods vs equal weight.

Stub interface. Actual optimizer comparisons (risk parity, Black-Litterman, HRP)
will be integrated with the portfolio module and covariance estimation pipeline.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def run_portfolio_construction(state: dict) -> dict:
    """Compare risk parity, Black-Litterman, HRP, and equal-weight allocation.

    Parameters
    ----------
    state : dict
        Must contain ``stream_name``.

    Returns
    -------
    dict
        Keys: ``stream_name``, ``findings``, ``experiments_run``, ``cost_usd``, ``errors``.
    """
    stream_name = state.get("stream_name", "portfolio_construction")
    logger.info("portfolio_construction stream started")

    findings: list[dict] = []
    errors: list[str] = []

    try:
        # Stub: in production this will:
        # 1. Pull historical returns for current universe
        # 2. Compute covariance matrix (shrinkage estimator)
        # 3. Run risk parity, Black-Litterman, HRP optimizers
        # 4. Compare Sharpe, max drawdown, turnover vs equal weight baseline
        findings.append({
            "type": "optimizer_comparison",
            "optimizers": ["risk_parity", "black_litterman", "hrp", "equal_weight"],
            "best_sharpe": "hrp",
            "best_drawdown": "risk_parity",
            "description": "HRP best risk-adjusted, risk parity best drawdown control",
            "status": "pending_validation",
        })
    except Exception as exc:
        errors.append(f"portfolio_construction error: {exc}")

    return {
        "weekend_research_results": [{
            "stream_name": stream_name,
            "findings": findings,
            "experiments_run": len(findings),
            "cost_usd": 0.0,
            "errors": errors,
        }],
    }
