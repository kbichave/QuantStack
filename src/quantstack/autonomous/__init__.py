# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Autonomous trading components — debate, routing, screening, strategy lifecycle.

The Python runner (runner.py) and research orchestrator (research_orchestrator.py)
have been removed. All trading and research execution is handled by the Ralph loops
in prompts/trading_loop.md and prompts/research_loop.md.
"""

from quantstack.autonomous.decision import DecisionPath, DecisionRouter, RouteContext

__all__ = ["DecisionRouter", "DecisionPath", "RouteContext"]
