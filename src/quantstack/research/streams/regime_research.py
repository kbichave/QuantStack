"""Regime research stream — labels market regimes and tests regime-conditional allocation.

Stub interface. Actual regime labelling (HMM, threshold-based, etc.) and conditional
allocation backtests will be integrated with the overnight runner.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def run_regime_research(state: dict) -> dict:
    """Label historical regimes and backtest regime-conditional allocation rules.

    Parameters
    ----------
    state : dict
        Must contain ``stream_name``.

    Returns
    -------
    dict
        Keys: ``stream_name``, ``findings``, ``experiments_run``, ``cost_usd``, ``errors``.
    """
    stream_name = state.get("stream_name", "regime_research")
    logger.info("regime_research stream started")

    findings: list[dict] = []
    errors: list[str] = []

    try:
        # Stub: in production this will:
        # 1. Fit HMM on SPY returns to label trending_up/trending_down/ranging
        # 2. Test allocation shifts per regime (e.g. 60/40 trending vs 20/80 ranging)
        # 3. Compare Sharpe across regime-aware vs static allocation
        findings.append({
            "type": "regime_label",
            "method": "hmm_2state",
            "description": "2-state HMM on SPY daily returns",
            "regimes_identified": ["trending", "mean_reverting"],
            "status": "pending_validation",
        })
    except Exception as exc:
        errors.append(f"regime_research error: {exc}")

    return {
        "weekend_research_results": [{
            "stream_name": stream_name,
            "findings": findings,
            "experiments_run": len(findings),
            "cost_usd": 0.0,
            "errors": errors,
        }],
    }
