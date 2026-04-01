# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
SignalEngine — pure-Python replacement for the TradingCrew analysis layer.

Replaces 13 Ollama LLM IC agents + 6 pod managers + trading assistant
with deterministic Python collectors and a rule-based synthesizer.

Total wall-clock time: 2–6 seconds vs 3–5 minutes with TradingCrew.
No LLM calls. No Ollama dependency. No CrewAI.

Usage:
    from quantstack.signal_engine import SignalEngine

    brief = await SignalEngine().run("XOM")
    # brief is a SignalBrief — backward-compatible superset of DailyBrief
"""

from quantstack.signal_engine.brief import SignalBrief
from quantstack.signal_engine.engine import SignalEngine

__all__ = ["SignalEngine", "SignalBrief"]
