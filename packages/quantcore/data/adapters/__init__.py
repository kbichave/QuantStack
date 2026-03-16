"""
Data source adapters — concrete AssetClassAdapter implementations.

Each adapter wraps one external data provider and translates its API
into the standard OHLCV DataFrame contract expected by DataProviderRegistry.

Importing an adapter that requires an optional dependency (alpaca-py,
polygon-api-client, ib_insync) will raise ImportError with a clear message
if the package is not installed.  Install the matching extra:

    uv pip install -e ".[alpaca]"
    uv pip install -e ".[polygon]"
    uv pip install -e ".[ibkr]"
"""

from quantcore.data.adapters.alpaca import AlpacaAdapter
from quantcore.data.adapters.alphavantage import AlphaVantageAdapter
from quantcore.data.adapters.financial_datasets import FinancialDatasetsAdapter
from quantcore.data.adapters.ibkr import IBKRDataAdapter
from quantcore.data.adapters.polygon_adapter import PolygonAdapter

__all__ = [
    "AlphaVantageAdapter",
    "AlpacaAdapter",
    "FinancialDatasetsAdapter",
    "IBKRDataAdapter",
    "PolygonAdapter",
]
