# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Execution layer — sits between agent recommendations and broker.

Components (apply in order on every trade):
  1. PortfolioState  — persistent positions/cash across sessions
  2. RiskGate        — hard stop enforcement (cannot be overridden by agent)
  3. KillSwitch      — emergency halt all agents and positions
  4. Broker          — PaperBroker (default) or EtradeBroker (USE_REAL_TRADING=true)

Broker selection:
  USE_REAL_TRADING=false (default) → PaperBroker (local simulation)
  USE_REAL_TRADING=true            → EtradeBroker (eTrade API, paper or live account)
"""

from quant_pod.execution.broker_factory import get_broker, get_broker_mode
from quant_pod.execution.kill_switch import KillSwitch, get_kill_switch
from quant_pod.execution.order_lifecycle import (
    ExecAlgoOMS,
    Order,
    OrderLifecycle,
    OrderStatus,
    get_order_lifecycle,
)
from quant_pod.execution.paper_broker import Fill, OrderRequest, PaperBroker, get_paper_broker
from quant_pod.execution.portfolio_state import (
    ClosedTrade,
    PortfolioSnapshot,
    PortfolioState,
    Position,
    get_portfolio_state,
)
from quant_pod.execution.risk_gate import RiskGate, RiskViolation, get_risk_gate
from quant_pod.execution.strategy_breaker import BreakerConfig, BreakerState, StrategyBreaker

__all__ = [
    "PortfolioState",
    "PortfolioSnapshot",
    "Position",
    "ClosedTrade",
    "get_portfolio_state",
    "RiskGate",
    "RiskViolation",
    "get_risk_gate",
    "KillSwitch",
    "get_kill_switch",
    "PaperBroker",
    "Fill",
    "OrderRequest",
    "get_paper_broker",
    "get_broker",
    "get_broker_mode",
    # OMS
    "OrderLifecycle",
    "Order",
    "OrderStatus",
    "ExecAlgoOMS",
    "get_order_lifecycle",
    # Strategy breakers
    "StrategyBreaker",
    "BreakerConfig",
    "BreakerState",
]
