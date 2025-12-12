"""
Core infrastructure module for QuantCore.

Contains foundational components:
- calendar: Trading calendar system for exchange holidays and sessions
- errors: Exception hierarchy for consistent error handling
- config: Configuration classes with YAML support
"""

from quantcore.core.errors import (
    QuantCoreError,
    DataError,
    ValidationError,
    ConfigurationError,
    BacktestError,
    ExecutionError,
    CalendarError,
)
from quantcore.core.calendar import TradingCalendar
from quantcore.core.config import (
    BacktestConfig,
    FeatureConfig,
    RLConfig,
    RiskConfig,
    SpreadTradingConfig,
    load_config_from_yaml,
    save_config_to_yaml,
)

__all__ = [
    # Errors
    "QuantCoreError",
    "DataError",
    "ValidationError",
    "ConfigurationError",
    "BacktestError",
    "ExecutionError",
    "CalendarError",
    # Calendar
    "TradingCalendar",
    # Config
    "BacktestConfig",
    "FeatureConfig",
    "RLConfig",
    "RiskConfig",
    "SpreadTradingConfig",
    "load_config_from_yaml",
    "save_config_to_yaml",
]
