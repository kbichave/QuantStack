# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
CrewAI Flows for QuantPod orchestration.

The main flow is TradingDayFlow which orchestrates the hierarchical TradingCrew:

Hierarchy:
- SuperTrader: Final decision maker (consumes only 1-pager)
- Assistant: Synthesizes pod outputs into 1-pager
- Pod Managers: Coordinate and compile IC outputs
- ICs: Fetch data, compute metrics, return raw findings

NO FALLBACKS - The system requires TradingCrew to function.
"""

from quant_pod.flows.trading_day_flow import (
    TradingDayFlow,
    TradingDayState,
    TradingDayFlowAdapter,
)

__all__ = [
    "TradingDayFlow",
    "TradingDayState",
    "TradingDayFlowAdapter",
]
