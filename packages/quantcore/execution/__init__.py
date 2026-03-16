"""Execution modeling, cost analysis, and broker abstraction module."""

from quantcore.execution.async_execution_loop import AsyncExecutionLoop
from quantcore.execution.broker import (
    BrokerAuthError,
    BrokerConnectionError,
    BrokerError,
    BrokerInterface,
    BrokerOrderError,
)
from quantcore.execution.costs import (
    CostAwareLabeler,
    ImplementationShortfall,
    TransactionCostModel,
)
from quantcore.execution.fill_tracker import FillEvent, FillTracker, LivePosition
from quantcore.execution.kill_switch import KillSwitch, KillSwitchError
from quantcore.execution.paper_trading_enhanced import (
    EnhancedPaperOrder,
    EnhancedPaperPosition,
    EnhancedPaperTradingEngine,
    ExecutionQualityMetrics,
    OrderBookState,
)
from quantcore.execution.risk_gate import PreTradeRiskGate, RiskGateError, RiskLimits
from quantcore.execution.slippage import (
    SlippageModel,
    VolatilitySlippageModel,
    VolumeSlippageModel,
)
from quantcore.execution.smart_order_router import SmartOrderRouter, SmartOrderRouterError
from quantcore.execution.unified_models import (
    UnifiedAccount,
    UnifiedBalance,
    UnifiedOrder,
    UnifiedOrderPreview,
    UnifiedOrderResult,
    UnifiedPosition,
    UnifiedQuote,
)

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
