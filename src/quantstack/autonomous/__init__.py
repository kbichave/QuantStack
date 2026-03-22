# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
AutonomousRunner — fully deterministic unattended trading loop.

Runs the full analysis + execution pipeline without an active Claude Code session.
Uses SignalEngine for analysis and DecisionRouter for fully deterministic routing.
No LLM calls in the execution path (v1.1).

Usage (CLI):
    python -m quant_pod.autonomous.runner --symbols XOM MSFT SPY
    python -m quant_pod.autonomous.runner --dry-run           # no order submission
    python -m quant_pod.autonomous.runner --paper-only        # force paper mode

Usage (code):
    from quantstack.autonomous.runner import AutonomousRunner

    report = await AutonomousRunner().run(symbols=["XOM", "MSFT"])
"""

from quantstack.autonomous.decision import DecisionPath, DecisionRouter, RouteContext
from quantstack.autonomous.runner import AutonomousRunner

__all__ = ["AutonomousRunner", "DecisionRouter", "DecisionPath", "RouteContext"]
