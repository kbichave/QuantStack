"""Pydantic output schemas for all 21 LLM agents.

Each model defines the expected output shape for one agent. Used by
parse_and_validate() for structured validation and retry prompting.

Fail-safe fallback principle: on parse failure, safety-critical agents
fail CLOSED (halt/reject/empty), never fail OPEN (approve/proceed).
"""

from quantstack.graphs.schemas.trading import (
    DailyPlanOutput,
    EarningsAnalysisOutput,
    EntrySignalOutput,
    ExecutionOrderOutput,
    ExitDecisionOutput,
    MarketIntelOutput,
    OptionsAnalysisOutput,
    PositionReviewOutput,
    SafetyCheckOutput,
    TradeDebateOutput,
    TradeReflectionOutput,
)
from quantstack.graphs.schemas.research import (
    BacktestResultOutput,
    CommunityIntelOutput,
    DomainResearchOutput,
    ExperimentResultOutput,
    HypothesisCritiqueOutput,
    HypothesisOutput,
    QuantResearchOutput,
    StrategyRegistrationOutput,
    ValidationResultOutput,
)
from quantstack.graphs.schemas.supervisor import (
    DiagnosticOutput,
    HealthCheckOutput,
    PortfolioRiskOutput,
    RecoveryActionOutput,
    StrategyPromoterOutput,
)

# Registry mapping agent name -> output schema.
# Agent names match keys in graphs/*/config/agents.yaml.
AGENT_OUTPUT_SCHEMAS: dict[str, type] = {
    # Trading graph (10 agents)
    "market_intel": MarketIntelOutput,
    "daily_planner": DailyPlanOutput,
    "safety_check": SafetyCheckOutput,
    "position_monitor": PositionReviewOutput,
    "exit_evaluator": ExitDecisionOutput,
    "trade_debater": TradeDebateOutput,
    "fund_manager": EntrySignalOutput,
    "options_analyst": OptionsAnalysisOutput,
    "earnings_analyst": EarningsAnalysisOutput,
    "executor": ExecutionOrderOutput,
    "trade_reflector": TradeReflectionOutput,
    # Research graph (7 agents)
    "quant_researcher": QuantResearchOutput,
    "ml_scientist": ExperimentResultOutput,
    "strategy_rd": StrategyRegistrationOutput,
    "hypothesis_critic": HypothesisCritiqueOutput,
    "community_intel": CommunityIntelOutput,
    "domain_researcher": DomainResearchOutput,
    "execution_researcher": BacktestResultOutput,
    # Supervisor graph (4 agents)
    "health_monitor": HealthCheckOutput,
    "self_healer": RecoveryActionOutput,
    "portfolio_risk_monitor": PortfolioRiskOutput,
    "strategy_promoter": StrategyPromoterOutput,
}

# Fail-safe fallback values for each agent.
# CRITICAL: Safety agents fail CLOSED (halted=True, reject, empty).
AGENT_FALLBACKS: dict[str, dict | list] = {
    # Trading — safety-critical fallbacks
    "market_intel": {},
    "daily_planner": {"plan": "parse_failure — no plan generated"},
    "safety_check": {"halted": True, "reason": "parse_failure"},  # FAIL CLOSED
    "position_monitor": {"analyses": []},  # No exits triggered on failure
    "exit_evaluator": [],  # No exits on failure
    "trade_debater": [],  # No recommendations on failure
    "fund_manager": [],  # No entries on failure
    "options_analyst": [],  # No analysis on failure
    "earnings_analyst": [],  # No analysis on failure
    "executor": [],  # No orders on failure
    "trade_reflector": {"reflection": "parse_failure"},
    # Research — conservative fallbacks
    "quant_researcher": {"summary": "parse_failure"},
    "ml_scientist": {"experiment_id": "exp-unknown"},
    "strategy_rd": {"strategy_id": "strat-unknown"},
    "hypothesis_critic": {"confidence": 0.0, "critique": "parse_failure"},
    "community_intel": {"ideas": []},
    "domain_researcher": {"domain": "swing", "symbols": []},
    "execution_researcher": {"backtest_id": "bt-unknown"},
    # Supervisor — conservative fallbacks
    "health_monitor": {"overall": "unknown"},
    "self_healer": [],
    "portfolio_risk_monitor": [],
    "strategy_promoter": [],
}

__all__ = [
    "AGENT_OUTPUT_SCHEMAS",
    "AGENT_FALLBACKS",
    # Trading
    "MarketIntelOutput",
    "DailyPlanOutput",
    "SafetyCheckOutput",
    "PositionReviewOutput",
    "ExitDecisionOutput",
    "TradeDebateOutput",
    "EntrySignalOutput",
    "OptionsAnalysisOutput",
    "EarningsAnalysisOutput",
    "ExecutionOrderOutput",
    "TradeReflectionOutput",
    # Research
    "QuantResearchOutput",
    "ExperimentResultOutput",
    "StrategyRegistrationOutput",
    "HypothesisCritiqueOutput",
    "CommunityIntelOutput",
    "DomainResearchOutput",
    "BacktestResultOutput",
    "HypothesisOutput",
    "ValidationResultOutput",
    # Supervisor
    "HealthCheckOutput",
    "DiagnosticOutput",
    "RecoveryActionOutput",
    "PortfolioRiskOutput",
    "StrategyPromoterOutput",
]
