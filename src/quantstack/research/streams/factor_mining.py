"""Factor mining stream — extracts testable factors from paper references and computes IC.

This is a stub that defines the interface. Actual computation (IC calculation,
factor extraction from papers) will be filled in when the overnight runner and
feature factory are integrated.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def run_factor_mining(state: dict) -> dict:
    """Extract testable factors from literature references and compute information coefficients.

    Parameters
    ----------
    state : dict
        Must contain at minimum ``stream_name``.  May contain ``budget_remaining``
        to gate expensive computations.

    Returns
    -------
    dict
        Keys: ``stream_name``, ``findings``, ``experiments_run``, ``cost_usd``, ``errors``.
    """
    stream_name = state.get("stream_name", "factor_mining")
    logger.info("factor_mining stream started")

    findings: list[dict] = []
    errors: list[str] = []

    try:
        # Stub: in production this will:
        # 1. Pull paper references from knowledge graph
        # 2. Extract factor definitions (value, momentum, quality, etc.)
        # 3. Compute rolling IC for each factor against forward returns
        # 4. Flag factors with IC > 0.03 and t-stat > 2.0 as actionable
        findings.append({
            "type": "factor_candidate",
            "name": "stub_momentum_12_1",
            "description": "12-month momentum with 1-month reversal exclusion",
            "ic_estimate": 0.04,
            "status": "pending_validation",
        })
    except Exception as exc:
        errors.append(f"factor_mining error: {exc}")

    return {
        "weekend_research_results": [{
            "stream_name": stream_name,
            "findings": findings,
            "experiments_run": len(findings),
            "cost_usd": 0.0,
            "errors": errors,
        }],
    }
