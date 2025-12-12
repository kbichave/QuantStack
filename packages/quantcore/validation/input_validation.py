"""
Input validation utilities for QuantCore.

Provides validators for common data structures to catch errors early
and provide clear error messages.

Usage:
    from quantcore.validation.input_validation import DataFrameValidator

    # Validate OHLCV data
    result = DataFrameValidator.validate_ohlcv(df)
    if not result.is_valid:
        raise ValueError(f"Invalid data: {result.errors}")

    # Validate with auto-raise
    DataFrameValidator.validate_ohlcv(df, raise_on_error=True)
"""

from dataclasses import dataclass, field
from typing import List, Optional, Set, Dict, Any
import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class ValidationResult:
    """
    Result of a validation check.

    Attributes:
        is_valid: Whether validation passed (no errors, warnings OK)
        errors: List of critical errors that should block processing
        warnings: List of issues that may affect quality but not correctness
        info: Dict of additional validation metadata
    """

    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    info: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        if self.is_valid:
            status = "VALID"
        else:
            status = "INVALID"

        parts = [f"ValidationResult: {status}"]

        if self.errors:
            parts.append(f"  Errors ({len(self.errors)}):")
            for e in self.errors[:5]:  # Show first 5
                parts.append(f"    - {e}")
            if len(self.errors) > 5:
                parts.append(f"    ... and {len(self.errors) - 5} more")

        if self.warnings:
            parts.append(f"  Warnings ({len(self.warnings)}):")
            for w in self.warnings[:3]:
                parts.append(f"    - {w}")

        return "\n".join(parts)

    def raise_if_invalid(self, context: str = "Data validation") -> None:
        """Raise ValueError if validation failed."""
        if not self.is_valid:
            error_summary = "; ".join(self.errors[:3])
            raise ValueError(f"{context} failed: {error_summary}")

    def log_warnings(self) -> None:
        """Log any warnings."""
        for warning in self.warnings:
            logger.warning(warning)


class DataFrameValidator:
    """
    Validates DataFrame inputs for quantcore modules.

    Provides static methods for validating common data types:
    - OHLCV price data
    - Spread trading data
    - Feature matrices
    """

    # Standard OHLCV column names
    OHLCV_COLUMNS = {"open", "high", "low", "close", "volume"}
    OHLCV_REQUIRED = {"close"}  # Minimum required

    # Spread data columns
    SPREAD_REQUIRED = {"spread"}
    SPREAD_OPTIONAL = {"wti", "brent", "usd", "curve"}

    @staticmethod
    def validate_ohlcv(
        df: pd.DataFrame,
        name: str = "data",
        require_volume: bool = False,
        raise_on_error: bool = False,
    ) -> ValidationResult:
        """
        Validate OHLCV DataFrame.

        Checks:
        - DataFrame is not empty
        - Required columns exist (at minimum 'close')
        - Index is DatetimeIndex and sorted
        - No NaN/inf in price columns
        - No negative prices
        - No zero/negative volume (if volume present)

        Args:
            df: DataFrame to validate
            name: Name for error messages
            require_volume: Whether volume column is required
            raise_on_error: Raise ValueError if validation fails

        Returns:
            ValidationResult with errors, warnings, and metadata
        """
        errors = []
        warnings = []
        info = {}

        # Check not None
        if df is None:
            errors.append(f"{name}: DataFrame is None")
            result = ValidationResult(
                is_valid=False, errors=errors, warnings=warnings, info=info
            )
            if raise_on_error:
                result.raise_if_invalid(f"{name} validation")
            return result

        # Check not empty
        if df.empty:
            errors.append(f"{name}: DataFrame is empty")
            result = ValidationResult(
                is_valid=False, errors=errors, warnings=warnings, info=info
            )
            if raise_on_error:
                result.raise_if_invalid(f"{name} validation")
            return result

        info["rows"] = len(df)
        info["columns"] = list(df.columns)

        # Check required columns
        columns_lower = {c.lower() for c in df.columns}
        required = DataFrameValidator.OHLCV_REQUIRED.copy()
        if require_volume:
            required.add("volume")

        missing_required = required - columns_lower
        if missing_required:
            errors.append(f"{name}: Missing required columns: {missing_required}")

        # Check for standard OHLCV columns
        missing_optional = DataFrameValidator.OHLCV_COLUMNS - columns_lower - required
        if missing_optional:
            warnings.append(
                f"{name}: Missing optional OHLCV columns: {missing_optional}"
            )

        # Check index is DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            warnings.append(
                f"{name}: Index is not DatetimeIndex (got {type(df.index).__name__})"
            )
        else:
            info["start_date"] = str(df.index[0])
            info["end_date"] = str(df.index[-1])

            # Check sorted
            if not df.index.is_monotonic_increasing:
                errors.append(f"{name}: Index is not sorted chronologically")

        # Check price columns for issues
        price_cols = ["open", "high", "low", "close"]
        for col in price_cols:
            if col in df.columns:
                col_data = df[col]

                # Check NaN
                nan_count = col_data.isna().sum()
                if nan_count > 0:
                    nan_pct = nan_count / len(df) * 100
                    if nan_pct > 5:
                        errors.append(
                            f"{name}: '{col}' has {nan_count} NaN values ({nan_pct:.1f}%)"
                        )
                    else:
                        warnings.append(
                            f"{name}: '{col}' has {nan_count} NaN values ({nan_pct:.1f}%)"
                        )

                # Check inf
                inf_count = np.isinf(col_data).sum()
                if inf_count > 0:
                    errors.append(f"{name}: '{col}' has {inf_count} infinite values")

                # Check negative
                neg_count = (col_data < 0).sum()
                if neg_count > 0:
                    errors.append(
                        f"{name}: '{col}' has {neg_count} negative values (prices cannot be negative)"
                    )

        # Check volume if present
        if "volume" in df.columns:
            volume = df["volume"]

            # NaN
            nan_count = volume.isna().sum()
            if nan_count > 0:
                warnings.append(f"{name}: 'volume' has {nan_count} NaN values")

            # Negative
            neg_count = (volume < 0).sum()
            if neg_count > 0:
                errors.append(f"{name}: 'volume' has {neg_count} negative values")

            # Zero (just a warning)
            zero_count = (volume == 0).sum()
            if zero_count > 0:
                zero_pct = zero_count / len(df) * 100
                if zero_pct > 10:
                    warnings.append(
                        f"{name}: 'volume' has {zero_count} zero values ({zero_pct:.1f}%)"
                    )

        # OHLC consistency checks
        if all(c in df.columns for c in ["open", "high", "low", "close"]):
            # High >= Low
            invalid_hl = (df["high"] < df["low"]).sum()
            if invalid_hl > 0:
                errors.append(
                    f"{name}: {invalid_hl} rows where high < low (invalid OHLC)"
                )

            # High >= Open, Close
            invalid_high = (
                (df["high"] < df["open"]) | (df["high"] < df["close"])
            ).sum()
            if invalid_high > 0:
                warnings.append(
                    f"{name}: {invalid_high} rows where high < open or close"
                )

            # Low <= Open, Close
            invalid_low = ((df["low"] > df["open"]) | (df["low"] > df["close"])).sum()
            if invalid_low > 0:
                warnings.append(f"{name}: {invalid_low} rows where low > open or close")

        is_valid = len(errors) == 0
        result = ValidationResult(
            is_valid=is_valid, errors=errors, warnings=warnings, info=info
        )

        if raise_on_error:
            result.raise_if_invalid(f"{name} validation")

        return result

    @staticmethod
    def validate_spread_data(
        df: pd.DataFrame,
        name: str = "spread_data",
        raise_on_error: bool = False,
    ) -> ValidationResult:
        """
        Validate spread trading DataFrame.

        Checks:
        - 'spread' column exists
        - No NaN/inf in spread
        - Optional columns (wti, brent, usd, curve) validated if present
        - Index is DatetimeIndex and sorted

        Args:
            df: DataFrame to validate
            name: Name for error messages
            raise_on_error: Raise ValueError if validation fails

        Returns:
            ValidationResult
        """
        errors = []
        warnings = []
        info = {}

        # Check not None/empty
        if df is None:
            errors.append(f"{name}: DataFrame is None")
            result = ValidationResult(
                is_valid=False, errors=errors, warnings=warnings, info=info
            )
            if raise_on_error:
                result.raise_if_invalid(f"{name} validation")
            return result

        if df.empty:
            errors.append(f"{name}: DataFrame is empty")
            result = ValidationResult(
                is_valid=False, errors=errors, warnings=warnings, info=info
            )
            if raise_on_error:
                result.raise_if_invalid(f"{name} validation")
            return result

        info["rows"] = len(df)
        info["columns"] = list(df.columns)

        # Check required column
        if "spread" not in df.columns:
            errors.append(f"{name}: Missing required 'spread' column")
        else:
            spread = df["spread"]

            # NaN
            nan_count = spread.isna().sum()
            if nan_count > 0:
                nan_pct = nan_count / len(df) * 100
                if nan_pct > 10:
                    errors.append(
                        f"{name}: 'spread' has {nan_count} NaN values ({nan_pct:.1f}%)"
                    )
                else:
                    warnings.append(
                        f"{name}: 'spread' has {nan_count} NaN values ({nan_pct:.1f}%)"
                    )

            # Inf
            inf_count = np.isinf(spread).sum()
            if inf_count > 0:
                errors.append(f"{name}: 'spread' has {inf_count} infinite values")

            info["spread_mean"] = float(spread.mean())
            info["spread_std"] = float(spread.std())

        # Check index
        if not isinstance(df.index, pd.DatetimeIndex):
            warnings.append(f"{name}: Index is not DatetimeIndex")
        else:
            if not df.index.is_monotonic_increasing:
                errors.append(f"{name}: Index is not sorted chronologically")

        # Check optional columns
        optional_present = []
        for col in DataFrameValidator.SPREAD_OPTIONAL:
            if col in df.columns:
                optional_present.append(col)
                col_data = df[col]

                # Basic validation
                nan_count = col_data.isna().sum()
                if nan_count > 0:
                    nan_pct = nan_count / len(df) * 100
                    if nan_pct > 20:
                        warnings.append(
                            f"{name}: '{col}' has {nan_count} NaN values ({nan_pct:.1f}%)"
                        )

        info["optional_columns_present"] = optional_present

        is_valid = len(errors) == 0
        result = ValidationResult(
            is_valid=is_valid, errors=errors, warnings=warnings, info=info
        )

        if raise_on_error:
            result.raise_if_invalid(f"{name} validation")

        return result

    @staticmethod
    def validate_feature_matrix(
        df: pd.DataFrame,
        name: str = "features",
        max_nan_pct: float = 5.0,
        raise_on_error: bool = False,
    ) -> ValidationResult:
        """
        Validate feature matrix DataFrame.

        Checks:
        - Not empty
        - No entirely NaN columns
        - NaN percentage below threshold
        - No infinite values
        - Reasonable value ranges (no extreme outliers)

        Args:
            df: Feature DataFrame to validate
            name: Name for error messages
            max_nan_pct: Maximum allowed NaN percentage per column
            raise_on_error: Raise ValueError if validation fails

        Returns:
            ValidationResult
        """
        errors = []
        warnings = []
        info = {}

        if df is None:
            errors.append(f"{name}: DataFrame is None")
            result = ValidationResult(
                is_valid=False, errors=errors, warnings=warnings, info=info
            )
            if raise_on_error:
                result.raise_if_invalid(f"{name} validation")
            return result

        if df.empty:
            errors.append(f"{name}: DataFrame is empty")
            result = ValidationResult(
                is_valid=False, errors=errors, warnings=warnings, info=info
            )
            if raise_on_error:
                result.raise_if_invalid(f"{name} validation")
            return result

        info["rows"] = len(df)
        info["columns"] = len(df.columns)

        # Check for entirely NaN columns
        all_nan_cols = df.columns[df.isna().all()].tolist()
        if all_nan_cols:
            errors.append(f"{name}: Columns with all NaN values: {all_nan_cols[:5]}")

        # Check NaN percentage per column
        high_nan_cols = []
        for col in df.columns:
            nan_pct = df[col].isna().sum() / len(df) * 100
            if nan_pct > max_nan_pct:
                high_nan_cols.append((col, nan_pct))

        if high_nan_cols:
            warnings.append(
                f"{name}: {len(high_nan_cols)} columns with >{max_nan_pct}% NaN values"
            )

        # Check for infinite values
        inf_cols = []
        for col in df.select_dtypes(include=[np.number]).columns:
            inf_count = np.isinf(df[col]).sum()
            if inf_count > 0:
                inf_cols.append((col, inf_count))

        if inf_cols:
            errors.append(
                f"{name}: Columns with infinite values: {[c[0] for c in inf_cols[:5]]}"
            )

        # Check for extreme outliers (values > 1e10)
        extreme_cols = []
        for col in df.select_dtypes(include=[np.number]).columns:
            max_abs = df[col].abs().max()
            if not np.isnan(max_abs) and max_abs > 1e10:
                extreme_cols.append((col, max_abs))

        if extreme_cols:
            warnings.append(
                f"{name}: {len(extreme_cols)} columns with extreme values (>1e10)"
            )

        info["nan_summary"] = f"{len(high_nan_cols)} high-NaN columns"

        is_valid = len(errors) == 0
        result = ValidationResult(
            is_valid=is_valid, errors=errors, warnings=warnings, info=info
        )

        if raise_on_error:
            result.raise_if_invalid(f"{name} validation")

        return result


def validate_positive_number(
    value: float,
    name: str,
    allow_zero: bool = False,
) -> None:
    """
    Validate that a value is a positive number.

    Args:
        value: Value to validate
        name: Parameter name for error message
        allow_zero: Whether zero is allowed

    Raises:
        ValueError: If validation fails
    """
    if value is None:
        raise ValueError(f"{name} cannot be None")

    if not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number, got {type(value).__name__}")

    if np.isnan(value):
        raise ValueError(f"{name} cannot be NaN")

    if np.isinf(value):
        raise ValueError(f"{name} cannot be infinite")

    if allow_zero:
        if value < 0:
            raise ValueError(f"{name} must be non-negative, got {value}")
    else:
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")


def validate_in_range(
    value: float,
    name: str,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
) -> None:
    """
    Validate that a value is within a specified range.

    Args:
        value: Value to validate
        name: Parameter name for error message
        min_val: Minimum allowed value (inclusive)
        max_val: Maximum allowed value (inclusive)

    Raises:
        ValueError: If validation fails
    """
    if value is None:
        raise ValueError(f"{name} cannot be None")

    if min_val is not None and value < min_val:
        raise ValueError(f"{name} must be >= {min_val}, got {value}")

    if max_val is not None and value > max_val:
        raise ValueError(f"{name} must be <= {max_val}, got {value}")
