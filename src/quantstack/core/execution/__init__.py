"""Execution modeling, cost analysis, and broker abstraction module."""

from quantstack.core.execution.async_execution_loop import AsyncExecutionLoop
from quantstack.core.execution.broker import (
    BrokerAuthError,
    BrokerConnectionError,
    BrokerError,
    BrokerInterface,
    BrokerOrderError,
)
from quantstack.core.execution.costs import (
    CostAwareLabeler,
    ImplementationShortfall,
    TransactionCostModel,
)
from quantstack.core.execution.fill_tracker import FillEvent, FillTracker, LivePosition
from quantstack.core.execution.kill_switch import KillSwitch, KillSwitchError
from quantstack.core.execution.paper_trading_enhanced import (
    EnhancedPaperOrder,
    EnhancedPaperPosition,
    EnhancedPaperTradingEngine,
    ExecutionQualityMetrics,
    OrderBookState,
)
from quantstack.core.execution.risk_gate import (
    PreTradeRiskGate,
    RiskGateError,
    RiskLimits,
)
from quantstack.core.execution.slippage import (
    SlippageModel,
    VolatilitySlippageModel,
    VolumeSlippageModel,
)
from quantstack.core.execution.smart_order_router import (
    SmartOrderRouter,
    SmartOrderRouterError,
)
from quantstack.core.execution.unified_models import (
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
