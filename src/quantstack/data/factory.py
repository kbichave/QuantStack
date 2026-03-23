# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Data provider factory — resolves the active DataProvider at runtime.

Separated from provider.py to break the factory-subclass circular dependency
(provider.py defines the ABC, subclasses import it, factory imports subclasses).
"""

from __future__ import annotations

import os

from quantstack.data.adapters.alphavantage import AlphaVantageAdapter
from quantstack.data.polygon import PolygonProvider
from quantstack.data.provider import DataProvider

_active_provider: DataProvider | None = None


def get_provider() -> DataProvider:
    """
    Return the active data provider.

    Priority:
      1. DATA_PROVIDER env var ("polygon" or "alphavantage")
      2. Auto-detect based on available API keys
         - POLYGON_API_KEY set → PolygonProvider
         - ALPHA_VANTAGE_API_KEY set → AlphaVantageProvider
      3. Raise if neither key is set
    """
    global _active_provider
    if _active_provider is not None:
        return _active_provider

    pref = os.getenv("DATA_PROVIDER", "").lower()

    if pref == "polygon" or (not pref and os.getenv("POLYGON_API_KEY")):
        _active_provider = PolygonProvider()
    elif pref == "alphavantage" or (not pref and os.getenv("ALPHA_VANTAGE_API_KEY")):
        _active_provider = AlphaVantageAdapter()
    else:
        raise RuntimeError(
            "No data provider configured. Set POLYGON_API_KEY (recommended) "
            "or ALPHA_VANTAGE_API_KEY, or DATA_PROVIDER env var."
        )

    return _active_provider


def set_provider(provider: DataProvider) -> None:
    """Override the active provider (useful for testing)."""
    global _active_provider
    _active_provider = provider
