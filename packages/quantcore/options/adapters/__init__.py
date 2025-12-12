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

from quantcore.options.adapters.vollib_adapter import (
    bs_price_vollib,
    implied_vol_vollib,
    greeks_vollib,
)
from quantcore.options.adapters.pysabr_adapter import (
    fit_sabr_surface,
    interpolate_sabr_vol,
    get_sabr_smile,
)
from quantcore.options.adapters.financepy_adapter import (
    price_vanilla_financepy,
    price_american_option,
)
from quantcore.options.adapters.quantsbin_adapter import (
    analyze_structure_quantsbin,
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
