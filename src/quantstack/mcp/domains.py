# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Domain taxonomy for MCP tool registration.

Each domain maps to a cohesive server that agents can discover by namespace
prefix (e.g. ``mcp__quantstack-ml__train_ml_model``).  Tools are tagged with
one or more domains via the ``@domain()`` decorator from ``_registry.py``.

The server factory (``server_factory.py``) uses these tags to register only
the tools that belong to a given domain's server process.
"""

from __future__ import annotations

from enum import Flag, auto


class Domain(Flag):
    """MCP server domain — each value maps to one ``quantstack-*`` process."""

    EXECUTION = auto()   # trade execution, order mgmt, alerts, coordination
    PORTFOLIO = auto()   # portfolio state, optimization, attribution, feedback
    SIGNALS = auto()     # signal briefs, regime, intraday, TCA
    DATA = auto()        # OHLCV, indicators, fundamentals, microstructure
    RESEARCH = auto()    # backtesting, walk-forward, strategy lifecycle, validation
    OPTIONS = auto()     # Greeks, IV surface, vol forecasting, pricing
    ML = auto()          # supervised training, inference, drift, ensembles
    FINRL = auto()       # DRL training, evaluation, promotion, screening
    INTEL = auto()       # capitulation, accumulation, macro, NLP, cross-domain
    RISK = auto()        # VaR, stress testing, drawdown, Sortino/Calmar
    COORDINATION = auto()  # thin orchestrator: status, portfolio, regime, signals, heartbeat, events, execute
