# Copyright 2024 QuantCore Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Analytics module for portfolio and performance analysis.
"""

from quantcore.analytics.adapters import (
    compute_portfolio_stats_ffn,
    compute_factor_stats_ffn,
    generate_tearsheet_data,
)

__all__ = [
    "compute_portfolio_stats_ffn",
    "compute_factor_stats_ffn",
    "generate_tearsheet_data",
]
