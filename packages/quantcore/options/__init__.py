"""
Options trading module.

Provides:
- Options domain models (contracts, legs, positions)
- Black-Scholes pricing and Greeks (with dividend adjustment)
- IV surface interpolation
- Bid-ask based slippage model
- Contract selection logic
"""

from quantcore.options.models import (
    OptionType,
    OptionContract,
    OptionLeg,
    OptionsPosition,
    VerticalSpread,
)
from quantcore.options.pricing import (
    black_scholes_price,
    black_scholes_greeks,
    black_scholes_merton,
    implied_volatility,
    price_with_symbol,
    greeks_with_symbol,
    get_dividend_yield,
    DEFAULT_DIVIDEND_YIELDS,
)
from quantcore.options.contract_selector import (
    ContractSelector,
    VolRegime,
    TrendRegime,
)
from quantcore.options.iv_surface import (
    IVSurface,
    IVSurfaceMetrics,
    build_iv_surface_from_chain,
    extract_iv_features,
)
from quantcore.options.slippage import (
    SpreadBasedSlippage,
    ParametricSlippage,
    SlippageEstimate,
    ExecutionUrgency,
)

__all__ = [
    # Models
    "OptionType",
    "OptionContract",
    "OptionLeg",
    "OptionsPosition",
    "VerticalSpread",
    # Pricing
    "black_scholes_price",
    "black_scholes_greeks",
    "black_scholes_merton",
    "implied_volatility",
    "price_with_symbol",
    "greeks_with_symbol",
    "get_dividend_yield",
    "DEFAULT_DIVIDEND_YIELDS",
    # IV Surface
    "IVSurface",
    "IVSurfaceMetrics",
    "build_iv_surface_from_chain",
    "extract_iv_features",
    # Slippage
    "SpreadBasedSlippage",
    "ParametricSlippage",
    "SlippageEstimate",
    "ExecutionUrgency",
    # Selection
    "ContractSelector",
    "VolRegime",
    "TrendRegime",
]
