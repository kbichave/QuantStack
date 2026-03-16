"""Execution modeling, cost analysis, and broker abstraction module."""

from quantcore.execution.slippage import (
    SlippageModel,
    VolumeSlippageModel,
    VolatilitySlippageModel,
)
from quantcore.execution.costs import (
    TransactionCostModel,
    ImplementationShortfall,
    CostAwareLabeler,
)
from quantcore.execution.paper_trading_enhanced import (
    EnhancedPaperTradingEngine,
    EnhancedPaperOrder,
    EnhancedPaperPosition,
    OrderBookState,
    ExecutionQualityMetrics,
)
from quantcore.execution.broker import (
    BrokerInterface,
    BrokerError,
    BrokerConnectionError,
    BrokerOrderError,
    BrokerAuthError,
)
from quantcore.execution.unified_models import (
    UnifiedAccount,
    UnifiedBalance,
    UnifiedPosition,
    UnifiedQuote,
    UnifiedOrder,
    UnifiedOrderPreview,
    UnifiedOrderResult,
)
from quantcore.execution.kill_switch import KillSwitch, KillSwitchError
from quantcore.execution.fill_tracker import FillTracker, FillEvent, LivePosition
from quantcore.execution.risk_gate import PreTradeRiskGate, RiskLimits, RiskGateError
from quantcore.execution.smart_order_router import SmartOrderRouter, SmartOrderRouterError
from quantcore.execution.async_execution_loop import AsyncExecutionLoop

__all__ = [
    # Slippage models
    "SlippageModel",
    "VolumeSlippageModel",
    "VolatilitySlippageModel",
    # Cost models
    "TransactionCostModel",
    "ImplementationShortfall",
    "CostAwareLabeler",
    # Enhanced paper trading
    "EnhancedPaperTradingEngine",
    "EnhancedPaperOrder",
    "EnhancedPaperPosition",
    "OrderBookState",
    "ExecutionQualityMetrics",
    # Broker abstraction
    "BrokerInterface",
    "BrokerError",
    "BrokerConnectionError",
    "BrokerOrderError",
    "BrokerAuthError",
    # Unified models
    "UnifiedAccount",
    "UnifiedBalance",
    "UnifiedPosition",
    "UnifiedQuote",
    "UnifiedOrder",
    "UnifiedOrderPreview",
    "UnifiedOrderResult",
    # Phase 5 — execution engine
    "KillSwitch",
    "KillSwitchError",
    "FillTracker",
    "FillEvent",
    "LivePosition",
    "PreTradeRiskGate",
    "RiskLimits",
    "RiskGateError",
    "SmartOrderRouter",
    "SmartOrderRouterError",
    "AsyncExecutionLoop",
]
