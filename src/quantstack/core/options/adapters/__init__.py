# Copyright 2024 QuantCore Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Options adapters for external quant libraries.

Provides unified interfaces to:
- vollib: Black-Scholes pricing, IV, Greeks
- pysabr: SABR volatility surface fitting
- financepy: American options, exotics
- quantsbin: Payoff analysis, option structures
"""

from quantstack.core.options.adapters.financepy_adapter import (
    price_american_option,
    price_vanilla_financepy,
)
from quantstack.core.options.adapters.pysabr_adapter import (
    fit_sabr_surface,
    get_sabr_smile,
    interpolate_sabr_vol,
)
from quantstack.core.options.adapters.quantsbin_adapter import (
    analyze_structure_quantsbin,
)
from quantstack.core.options.adapters.vollib_adapter import (
    bs_price_vollib,
    greeks_vollib,
    implied_vol_vollib,
)

__all__ = [
    # vollib
    "bs_price_vollib",
    "implied_vol_vollib",
    "greeks_vollib",
    # pysabr
    "fit_sabr_surface",
    "interpolate_sabr_vol",
    "get_sabr_smile",
    # financepy
    "price_vanilla_financepy",
    "price_american_option",
    # quantsbin
    "analyze_structure_quantsbin",
]
