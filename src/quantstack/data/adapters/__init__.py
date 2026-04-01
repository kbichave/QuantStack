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

from quantstack.data.adapters.alpaca import AlpacaAdapter
from quantstack.data.adapters.alphavantage import AlphaVantageAdapter
from quantstack.data.adapters.financial_datasets import FinancialDatasetsAdapter
from quantstack.data.adapters.polygon_adapter import PolygonAdapter

try:
    from quantstack.data.adapters.ibkr import IBKRDataAdapter
except ImportError:
    IBKRDataAdapter = None  # type: ignore[assignment,misc]

__all__ = [
    "AlphaVantageAdapter",
    "AlpacaAdapter",
    "FinancialDatasetsAdapter",
    "IBKRDataAdapter",
    "PolygonAdapter",
]
