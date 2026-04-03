"""Statistical validation and anti-leakage module.

Heavy imports (statsmodels, sklearn) are in submodules that callers import
directly. This __init__ only re-exports the lightweight utilities so that
`from quantstack.core.validation.input_validation import DataFrameValidator`
does not trigger loading the entire ML stack.
"""

from quantstack.core.validation.input_validation import (
    DataFrameValidator,
    ValidationResult,
    validate_in_range,
    validate_positive_number,
)

__all__ = [
    "ValidationResult",
    "DataFrameValidator",
    "validate_positive_number",
    "validate_in_range",
]
