"""
Exception hierarchy for QuantCore.

Provides typed exceptions for consistent error handling across the library.
All QuantCore-specific exceptions inherit from QuantCoreError.

Usage:
    from quantcore.core.errors import DataError, ValidationError

    if data is None:
        raise DataError("No data available", symbol="AAPL", date="2024-01-01")
"""

from typing import Any, Dict, Optional


class QuantCoreError(Exception):
    """
    Base exception for all QuantCore errors.

    Provides structured error information including error codes and context.
    """

    error_code: str = "QC000"

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        **context: Any,
    ):
        """
        Initialize QuantCoreError.

        Args:
            message: Human-readable error message
            error_code: Optional specific error code (overrides class default)
            **context: Additional context key-value pairs for debugging
        """
        self.message = message
        self.error_code = error_code or self.__class__.error_code
        self.context: Dict[str, Any] = context

        # Build full message with context
        full_message = f"[{self.error_code}] {message}"
        if context:
            context_str = ", ".join(f"{k}={v!r}" for k, v in context.items())
            full_message += f" ({context_str})"

        super().__init__(full_message)

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/serialization."""
        return {
            "error_type": self.__class__.__name__,
            "error_code": self.error_code,
            "message": self.message,
            "context": self.context,
        }


class DataError(QuantCoreError):
    """
    Error related to data fetching, loading, or quality.

    Use for:
    - Missing data files
    - API fetch failures
    - Data quality issues (gaps, outliers)
    - Schema mismatches
    """

    error_code: str = "QC100"


class ValidationError(QuantCoreError):
    """
    Error related to input validation.

    Use for:
    - Invalid DataFrame schemas
    - Out-of-range parameters
    - Type mismatches
    - Missing required fields
    """

    error_code: str = "QC200"


class ConfigurationError(QuantCoreError):
    """
    Error related to configuration.

    Use for:
    - Invalid config values
    - Missing required config
    - Config file parsing errors
    - Environment variable issues
    """

    error_code: str = "QC300"


class BacktestError(QuantCoreError):
    """
    Error related to backtesting.

    Use for:
    - Invalid backtest parameters
    - Insufficient data for backtest
    - Strategy execution errors
    - Portfolio constraint violations
    """

    error_code: str = "QC400"


class ExecutionError(QuantCoreError):
    """
    Error related to trade execution.

    Use for:
    - Order placement failures
    - Position limit violations
    - Margin requirement issues
    - Broker connection errors
    """

    error_code: str = "QC500"


class CalendarError(QuantCoreError):
    """
    Error related to trading calendar.

    Use for:
    - Unknown exchange
    - Invalid date ranges
    - Holiday data unavailable
    """

    error_code: str = "QC600"


class FeatureError(QuantCoreError):
    """
    Error related to feature computation.

    Use for:
    - Feature calculation failures
    - Missing input data for features
    - Feature dependency issues
    """

    error_code: str = "QC700"


class ModelError(QuantCoreError):
    """
    Error related to ML models.

    Use for:
    - Model loading failures
    - Prediction errors
    - Training failures
    - Invalid model configuration
    """

    error_code: str = "QC800"


# Error code reference:
# QC000 - General/Unknown errors
# QC1xx - Data errors
# QC2xx - Validation errors
# QC3xx - Configuration errors
# QC4xx - Backtest errors
# QC5xx - Execution errors
# QC6xx - Calendar errors
# QC7xx - Feature errors
# QC8xx - Model errors
