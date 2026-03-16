"""
DataProvider enum — stable identity keys for each market-data source.

Kept in its own file to avoid circular imports: base.py imports this,
all adapters import this, and the registry imports this.  No other
quantcore modules are imported here.
"""

from enum import Enum


class DataProvider(str, Enum):
    """Identifies which external data source an adapter wraps.

    Using ``str`` as a mixin means the values can be used directly as
    dict keys, stored in config files, and compared with plain strings
    (e.g. ``DataProvider.ALPACA == "alpaca"``).
    """

    ALPHA_VANTAGE = "alpha_vantage"
    ALPACA = "alpaca"
    POLYGON = "polygon"
    IBKR = "ibkr"
