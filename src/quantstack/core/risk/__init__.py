"""Risk management module."""

from quantstack.core.risk.controls import (
    DrawdownProtection,
    ExposureManager,
    RiskController,
)
from quantstack.core.risk.position_sizing import ATRPositionSizer, PositionSizer
from quantstack.core.risk.span_margin import (
    MarginBreakdown,
    MarginTier,
    SPANMarginCalculator,
    calculate_span_margin,
)
from quantstack.core.risk.safety_gate import (
    RiskDecision,
    RiskVerdict,
    SafetyGate,
    SafetyGateLimits,
)
from quantstack.core.risk.stress_testing import (
    STRESS_SCENARIOS,
    MonteCarloSimulator,
    PortfolioStressResult,
    PortfolioStressTester,
    StressResult,
    VaRCalculator,
    VaRResult,
)

__all__ = [
    # Position sizing
    "PositionSizer",
    "ATRPositionSizer",
    # Risk controls
    "ExposureManager",
    "DrawdownProtection",
    "RiskController",
    # Stress testing
    "MonteCarloSimulator",
    "VaRCalculator",
    "VaRResult",
    "PortfolioStressTester",
    "STRESS_SCENARIOS",
    "StressResult",
    "PortfolioStressResult",
    # Safety gate
    "SafetyGate",
    "SafetyGateLimits",
    "RiskDecision",
    "RiskVerdict",
    # SPAN margin
    "SPANMarginCalculator",
    "MarginBreakdown",
    "MarginTier",
    "calculate_span_margin",
]
