"""Risk management module."""

from quantcore.risk.position_sizing import PositionSizer, ATRPositionSizer
from quantcore.risk.controls import (
    ExposureManager,
    DrawdownProtection,
    RiskController,
)
from quantcore.risk.stress_testing import (
    MonteCarloSimulator,
    VaRCalculator,
    VaRResult,
    PortfolioStressTester,
    STRESS_SCENARIOS,
    StressResult,
    PortfolioStressResult,
)
from quantcore.risk.span_margin import (
    SPANMarginCalculator,
    MarginBreakdown,
    MarginTier,
    calculate_span_margin,
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
