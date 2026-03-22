# Copyright 2024 QuantCore Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Analytics adapters for external quant libraries.

Provides unified interfaces to:
- ffn: Portfolio analytics, performance metrics, tear sheets
"""

from quantstack.core.analytics.adapters.ffn_adapter import (
    compute_factor_stats_ffn,
    compute_portfolio_stats_ffn,
    generate_tearsheet_data,
)

__all__ = [
    "compute_portfolio_stats_ffn",
    "compute_factor_stats_ffn",
    "generate_tearsheet_data",
]
