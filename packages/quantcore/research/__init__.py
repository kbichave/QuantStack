"""
Research module for quantitative analysis and signal evaluation.

This module provides tools for:
- Signal quality diagnostics (IC, alpha decay)
- Cost-adjusted performance analysis
- Walkforward optimization
- Statistical tests for alpha validation
"""

from quantcore.research.quant_metrics import (
    run_signal_diagnostics,
    run_alpha_decay_analysis,
    compute_cost_adjusted_returns,
    QuantResearchReport,
)

__all__ = [
    "run_signal_diagnostics",
    "run_alpha_decay_analysis",
    "compute_cost_adjusted_returns",
    "QuantResearchReport",
]
