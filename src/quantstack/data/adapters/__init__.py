"""
Data source adapters — concrete AssetClassAdapter implementations.

Each adapter wraps one external data provider and translates its API
into the standard OHLCV DataFrame contract expected by DataProviderRegistry.

Adapters that require optional broker packages (IBKR) are loaded only
when the matching env vars are configured. This avoids importing heavy
SDKs that aren't needed.
"""

import os

from quantstack.data.adapters.alpaca import AlpacaAdapter
from quantstack.data.adapters.alphavantage import AlphaVantageAdapter
from quantstack.data.adapters.financial_datasets import FinancialDatasetsAdapter
from quantstack.data.adapters.polygon_adapter import PolygonAdapter

IBKRDataAdapter = None  # type: ignore[assignment,misc]

if os.environ.get("IBKR_HOST"):
    from quantstack.data.adapters.ibkr import IBKRDataAdapter  # type: ignore[assignment,no-redef]

__all__ = [
    "AlphaVantageAdapter",
    "AlpacaAdapter",
    "FinancialDatasetsAdapter",
    "IBKRDataAdapter",
    "PolygonAdapter",
]
