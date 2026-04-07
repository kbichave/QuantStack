"""Pydantic brief schemas for context compaction at merge points.

These schemas define the compact, typed representations that replace
raw unstructured state at graph merge points. Downstream nodes read
briefs instead of full accumulated state, reducing context size by 40%+.

Two briefs:
- ParallelMergeBrief: after exits + entries/earnings converge
- PreExecutionBrief: after portfolio_review + analyze_options converge
"""

from pydantic import BaseModel, Field


class EntryCandidate(BaseModel):
    """Compact representation of an entry candidate from entry_scan."""
    symbol: str
    signal_strength: float = Field(ge=0.0, le=1.0)
    thesis: str
    ewf_bias: str = ""


class ParallelMergeBrief(BaseModel):
    """Brief produced at the merge_parallel convergence point.

    Summarizes the outputs of execute_exits, entry_scan, and
    (optionally) earnings_analysis into a typed structure.
    """
    exits: list[dict] = []
    entries: list[dict] = []
    risks: list[dict] = []
    regime: str = ""
    earnings_flags: dict = {}
    compaction_degraded: bool = False


class PreExecutionBrief(BaseModel):
    """Brief produced at the merge_pre_execution convergence point.

    Summarizes fund_manager_decisions (approved/rejected) and
    options_analysis into a typed structure for execute_entries.
    """
    approved: list[dict]
    rejected: list[dict]
    options_specs: list[dict] = []
    risk_checks: dict = {}
    compaction_degraded: bool = False
