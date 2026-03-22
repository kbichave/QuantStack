"""
Core infrastructure module for QuantCore.

Contains foundational components:
- calendar: Trading calendar system for exchange holidays and sessions
- errors: Exception hierarchy for consistent error handling
- config: Configuration classes with YAML support
"""

from quantstack.core.core.calendar import TradingCalendar
from quantstack.core.core.config import (
    BacktestConfig,
    FeatureConfig,
    RiskConfig,
    RLConfig,
    SpreadTradingConfig,
    load_config_from_yaml,
    save_config_to_yaml,
)
from quantstack.core.core.errors import (
    BacktestError,
    CalendarError,
    ConfigurationError,
    DataError,
    ExecutionError,
    QuantCoreError,
    ValidationError,
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
