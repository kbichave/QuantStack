"""Statistical validation and anti-leakage module."""

from quantcore.validation.purged_cv import (
    PurgedKFoldCV,
    CombinatorialPurgedCV,
    WalkForwardValidator,
)
from quantcore.validation.leakage import (
    LeakageDetector,
    FeatureShiftTest,
    PermutationTest,
)
from quantcore.validation.orthogonalization import (
    FeatureOrthogonalizer,
    PCAReducer,
    CorrelationFilter,
)
from quantcore.validation.integrity import (
    TemporalSplit,
    get_temporal_splits,
    validate_no_lookahead,
    validate_data_integrity,
    print_split_info,
    assert_no_future_data,
    LeakageDetector as RuntimeLeakageDetector,
)
from quantcore.validation.input_validation import (
    ValidationResult,
    DataFrameValidator,
    validate_positive_number,
    validate_in_range,
)

__all__ = [
    # Cross-validation
    "PurgedKFoldCV",
    "CombinatorialPurgedCV",
    "WalkForwardValidator",
    # Leakage detection
    "LeakageDetector",
    "FeatureShiftTest",
    "PermutationTest",
    # Orthogonalization
    "FeatureOrthogonalizer",
    "PCAReducer",
    "CorrelationFilter",
    # Integrity validation
    "TemporalSplit",
    "get_temporal_splits",
    "validate_no_lookahead",
    "validate_data_integrity",
    "print_split_info",
    "assert_no_future_data",
    "RuntimeLeakageDetector",
    # Input validation
    "ValidationResult",
    "DataFrameValidator",
    "validate_positive_number",
    "validate_in_range",
]
