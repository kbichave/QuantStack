"""Statistical validation and anti-leakage module."""

from quantstack.core.validation.causal_filter import (
    CausalFilter,
    CausalFilterResult,
    CausalTestResult,
)
from quantstack.core.validation.input_validation import (
    DataFrameValidator,
    ValidationResult,
    validate_in_range,
    validate_positive_number,
)
from quantstack.core.validation.integrity import (
    LeakageDetector as RuntimeLeakageDetector,
)
from quantstack.core.validation.integrity import (
    TemporalSplit,
    assert_no_future_data,
    get_temporal_splits,
    print_split_info,
    validate_data_integrity,
    validate_no_lookahead,
)
from quantstack.core.validation.leakage import (
    FeatureShiftTest,
    LeakageDetector,
    PermutationTest,
)
from quantstack.core.validation.orthogonalization import (
    CorrelationFilter,
    FeatureOrthogonalizer,
    PCAReducer,
)
from quantstack.core.validation.purged_cv import (
    CombinatorialPurgedCV,
    PurgedKFoldCV,
    WalkForwardValidator,
)

__all__ = [
    # Causal filtering
    "CausalFilter",
    "CausalFilterResult",
    "CausalTestResult",
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
