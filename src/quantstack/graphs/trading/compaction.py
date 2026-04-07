"""Deterministic compaction nodes for Trading graph merge points.

These nodes replace the no-op merge_parallel and merge_pre_execution
nodes. They extract typed fields from accumulated state and produce
compact Pydantic briefs that downstream nodes consume.

No LLM calls — pure Python extraction. Faster, cheaper, and more
reliable than LLM-based summarization.
"""

import logging
from typing import Any

from quantstack.graphs.state import TradingState

from .briefs import ParallelMergeBrief, PreExecutionBrief

logger = logging.getLogger(__name__)


def compact_parallel(state: TradingState) -> dict[str, Any]:
    """Produce a ParallelMergeBrief from the exits + entries convergence.

    Extracts:
    - exit_orders → exits
    - entry_candidates → entries (with signal_strength validation)
    - earnings_analysis → earnings_flags
    - regime → regime
    - position_reviews → risks (any flagged items)

    On extraction failure, returns a degraded brief with empty fields
    and compaction_degraded=True rather than crashing the pipeline.
    """
    try:
        exits = state.exit_orders if state.exit_orders else []
        entries = state.entry_candidates if state.entry_candidates else []
        regime = state.regime or ""
        earnings_flags = state.earnings_analysis if state.earnings_analysis else {}

        # Extract risk flags from position reviews (items flagged for concern)
        risks = []
        for review in (state.position_reviews or []):
            if review.get("action") in ("flag", "warn", "close"):
                risks.append({
                    "risk_type": "position_review",
                    "severity": review.get("urgency", "medium"),
                    "detail": review.get("reason", ""),
                })

        brief = ParallelMergeBrief(
            exits=exits,
            entries=entries,
            risks=risks,
            regime=regime,
            earnings_flags=earnings_flags,
        )
    except Exception as exc:
        logger.warning("compact_parallel degraded: %s", exc)
        brief = ParallelMergeBrief(compaction_degraded=True)

    return {"parallel_brief": brief}


def compact_pre_execution(state: TradingState) -> dict[str, Any]:
    """Produce a PreExecutionBrief from portfolio_review + options convergence.

    Extracts:
    - fund_manager_decisions → approved / rejected (split by decision field)
    - options_analysis → options_specs
    - risk_verdicts → risk_checks (aggregated into a dict)

    On extraction failure, returns a degraded brief with empty fields.
    """
    try:
        approved = []
        rejected = []
        for decision in (state.fund_manager_decisions or []):
            if decision.get("decision") == "APPROVED":
                approved.append(decision)
            else:
                rejected.append(decision)

        options_specs = state.options_analysis if state.options_analysis else []

        # Aggregate risk verdicts into a lookup dict
        risk_checks = {}
        for verdict in (state.risk_verdicts or []):
            symbol = verdict.get("symbol", "unknown")
            risk_checks[symbol] = verdict

        brief = PreExecutionBrief(
            approved=approved,
            rejected=rejected,
            options_specs=options_specs,
            risk_checks=risk_checks,
        )
    except Exception as exc:
        logger.warning("compact_pre_execution degraded: %s", exc)
        brief = PreExecutionBrief(
            approved=[], rejected=[], compaction_degraded=True,
        )

    return {"pre_execution_brief": brief}
