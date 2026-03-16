"""Risk management module."""

from quantcore.risk.controls import (
    DrawdownProtection,
    ExposureManager,
    RiskController,
)
from quantcore.risk.position_sizing import ATRPositionSizer, PositionSizer
from quantcore.risk.span_margin import (
    MarginBreakdown,
    MarginTier,
    SPANMarginCalculator,
    calculate_span_margin,
)
from quantcore.risk.stress_testing import (
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
    # SPAN margin
    "SPANMarginCalculator",
    "MarginBreakdown",
    "MarginTier",
    "calculate_span_margin",
]
