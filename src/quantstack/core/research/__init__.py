"""
Research module for quantitative analysis and signal evaluation.

This module provides tools for:
- Signal quality diagnostics (IC, alpha decay)
- Cost-adjusted performance analysis
- Walkforward optimization
- Statistical tests for alpha validation
"""

from quantstack.core.research.quant_metrics import (
    QuantResearchReport,
    compute_cost_adjusted_returns,
    run_alpha_decay_analysis,
    run_signal_diagnostics,
)

__all__ = [
    "run_signal_diagnostics",
    "run_alpha_decay_analysis",
    "compute_cost_adjusted_returns",
    "QuantResearchReport",
]
