# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
CrewAI-native Trading Crew Module.

This module implements the trading system using CrewAI's native Agent/Task/Crew
architecture with JSON-based prompt configuration.

Components:
- TradingCrew: Main crew orchestrating all trading agents
- prompts/*.json: Agent definitions (role, goal, backstory, settings)
- config/tasks.yaml: Task definitions (description, expected_output, context)
- tools.py: MCP QuantCore tool integration
- schemas.py: Pydantic output schemas

Hierarchy:
    SuperTrader → Assistant → Pod Managers → ICs

Usage:
    from quant_pod.crews import TradingCrew, list_available_agents

    # See available agents
    agents = list_available_agents()

    # Run analysis
    crew = TradingCrew()
    result = crew.crew().kickoff(inputs={
        "symbol": "SPY",
        "current_date": date.today(),
        "regime": {"trend": "bullish", "volatility": "normal", "confidence": 0.8},
        "portfolio": {...},
    })
"""

# Import schemas first (no dependencies on other crew modules)
from quant_pod.crews.schemas import (
    AnalysisNote,
    PodResearchNote,
    DailyBrief,
    SymbolBrief,
    TradeDecision,
    RiskVerdict,
    KeyLevel,
    TaskEnvelope,
)

# Import crew class (depends on schemas)
from quant_pod.crews.trading_crew import (
    TradingCrew,
    create_trading_crew,
    run_trading_analysis,
    list_available_agents,
)

__all__ = [
    # Main crew
    "TradingCrew",
    "create_trading_crew",
    "run_trading_analysis",
    "list_available_agents",
    # Schemas
    "AnalysisNote",
    "PodResearchNote",
    "DailyBrief",
    "SymbolBrief",
    "TradeDecision",
    "RiskVerdict",
    "KeyLevel",
    "TaskEnvelope",
]
