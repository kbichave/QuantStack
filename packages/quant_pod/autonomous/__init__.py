# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
AutonomousRunner — unattended trading loop.

Runs the full analysis + execution pipeline without an active Claude Code session.
Uses SignalEngine for analysis, DecisionRouter to choose execution path,
and GroqPM for non-routine PM synthesis.

Usage (CLI):
    python -m quant_pod.autonomous.runner --symbols XOM MSFT SPY
    python -m quant_pod.autonomous.runner --dry-run           # no order submission
    python -m quant_pod.autonomous.runner --paper-only        # force paper mode

Usage (code):
    from quant_pod.autonomous.runner import AutonomousRunner

    report = await AutonomousRunner().run(symbols=["XOM", "MSFT"])
"""

from quant_pod.autonomous.runner import AutonomousRunner
from quant_pod.autonomous.decision import DecisionPath, DecisionRouter

__all__ = ["AutonomousRunner", "DecisionRouter", "DecisionPath"]
