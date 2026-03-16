# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Compliance audit trail for QuantPod.

Records every agent decision append-only in DuckDB for regulatory review.
"""

from quant_pod.audit.models import AuditQuery, DecisionEvent, ToolCall
from quant_pod.audit.decision_log import (
    DecisionLog,
    get_decision_log,
    make_trade_event,
    make_analysis_event,
)

__all__ = [
    "AuditQuery",
    "DecisionEvent",
    "ToolCall",
    "DecisionLog",
    "get_decision_log",
    "make_trade_event",
    "make_analysis_event",
]
