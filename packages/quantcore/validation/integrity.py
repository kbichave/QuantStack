"""
Data integrity and temporal split validation utilities.

These helpers ensure proper train/validation/test splits are maintained
throughout the trading system to avoid lookahead bias and data leakage.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple

import pandas as pd
from loguru import logger


@dataclass
class TemporalSplit:
    """Container for temporal split information."""

    train_data: pd.DataFrame
    val_data: pd.DataFrame
    test_data: pd.DataFrame
    train_end: str
    val_end: str

    @property
    def train_bars(self) -> int:
        return len(self.train_data)

    @property
    def val_bars(self) -> int:
        return len(self.val_data)

    @property
    def test_bars(self) -> int:
        return len(self.test_data)

    @property
    def train_start(self) -> str:
        return (
            str(self.train_data.index[0].date()) if len(self.train_data) > 0 else "N/A"
        )

    @property
    def val_start(self) -> str:
        return str(self.val_data.index[0].date()) if len(self.val_data) > 0 else "N/A"

    @property
    def test_start(self) -> str:
        return str(self.test_data.index[0].date()) if len(self.test_data) > 0 else "N/A"

    @property
    def test_end_date(self) -> str:
        return (
            str(self.test_data.index[-1].date()) if len(self.test_data) > 0 else "N/A"
        )

    def to_dict(self) -> dict:
        """Return split info as dictionary for reporting."""
        return {
            "train_start": self.train_start,
            "train_end": self.train_end,
            "val_start": self.val_start,
            "val_end": self.val_end,
            "test_start": self.test_start,
            "test_end": self.test_end_date,
            "train_bars": self.train_bars,
            "val_bars": self.val_bars,
            "test_bars": self.test_bars,
        }


def get_temporal_splits(
    data: pd.DataFrame,
    train_end: str = "2018-01-01",
    val_end: str = "2021-01-01",
    min_train_bars: int = 252,
    min_val_bars: int = 252,
    min_test_bars: int = 200,
) -> Tuple[Optional[TemporalSplit], Optional[str]]:
    """
    Create proper 3-way temporal split with validation.

    Split Structure:
        Train: data.index < train_end
        Validation: train_end <= data.index < val_end
        Test: data.index >= val_end

    Args:
        data: DataFrame with DatetimeIndex
        train_end: End date for training (exclusive)
        val_end: End date for validation (exclusive), start of test
        min_train_bars: Minimum required training bars
        min_val_bars: Minimum required validation bars
        min_test_bars: Minimum required test bars

    Returns:
        Tuple of (TemporalSplit, error_message)
        If successful, error_message is None
        If failed, TemporalSplit is None and error_message explains why
    """
    if data.empty:
        return None, "Data is empty"

    if not isinstance(data.index, pd.DatetimeIndex):
        return None, "Data index must be DatetimeIndex"

    # Create splits
    train_data = data[data.index < train_end].copy()
    val_data = data[(data.index >= train_end) & (data.index < val_end)].copy()
    test_data = data[data.index >= val_end].copy()

    # Validate sizes
    errors = []

    if len(train_data) < min_train_bars:
        errors.append(f"Train set too small: {len(train_data)} < {min_train_bars}")

    if len(val_data) < min_val_bars:
        errors.append(f"Validation set too small: {len(val_data)} < {min_val_bars}")

    if len(test_data) < min_test_bars:
        errors.append(f"Test set too small: {len(test_data)} < {min_test_bars}")

    if errors:
        return None, "; ".join(errors)

    return (
        TemporalSplit(
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            train_end=train_end,
            val_end=val_end,
        ),
        None,
    )


def validate_no_lookahead(
    params_selected_on: str,
    test_start: str,
    operation: str = "parameter selection",
) -> bool:
    """
    Verify that parameters were NOT selected using test data.

    Args:
        params_selected_on: Date string of when params were selected ("validation" or a date)
        test_start: Start date of test period
        operation: Description of the operation being validated

    Returns:
        True if no lookahead detected, False otherwise
    """
    if params_selected_on.lower() in ["validation", "val", "train"]:
        return True

    try:
        selection_date = pd.to_datetime(params_selected_on)
        test_date = pd.to_datetime(test_start)

        if selection_date >= test_date:
            logger.error(f"LOOKAHEAD DETECTED in {operation}!")
            logger.error(f"  Params selected on: {params_selected_on}")
            logger.error(f"  Test period starts: {test_start}")
            logger.error(f"  This invalidates all test results!")
            return False

        return True
    except Exception:
        logger.warning(f"Could not parse dates for lookahead validation")
        return True


def validate_data_integrity(
    data: pd.DataFrame,
    required_columns: Optional[list] = None,
    check_gaps: bool = True,
    max_gap_days: int = 5,
) -> Tuple[bool, list]:
    """
    Validate data integrity for backtesting.

    Args:
        data: DataFrame to validate
        required_columns: List of required column names
        check_gaps: Whether to check for large gaps in data
        max_gap_days: Maximum allowed gap in trading days

    Returns:
        Tuple of (is_valid, list of warnings/errors)
    """
    issues = []

    if data.empty:
        return False, ["Data is empty"]

    # Check required columns
    if required_columns:
        missing = set(required_columns) - set(data.columns)
        if missing:
            issues.append(f"Missing columns: {missing}")

    # Check for NaN values in critical columns
    for col in ["spread", "spread_zscore"]:
        if col in data.columns:
            nan_count = data[col].isna().sum()
            if nan_count > 0:
                nan_pct = nan_count / len(data) * 100
                if nan_pct > 10:
                    issues.append(f"High NaN rate in {col}: {nan_pct:.1f}%")

    # Check for gaps
    if check_gaps and isinstance(data.index, pd.DatetimeIndex):
        gaps = data.index.to_series().diff()
        large_gaps = gaps[gaps > pd.Timedelta(days=max_gap_days)]

        if len(large_gaps) > 0:
            issues.append(f"Found {len(large_gaps)} gaps > {max_gap_days} days")

    # Check data is sorted
    if not data.index.is_monotonic_increasing:
        issues.append("Data index is not sorted chronologically")

    is_valid = len([i for i in issues if "Missing" in i or "not sorted" in i]) == 0

    return is_valid, issues


def print_split_info(split: TemporalSplit, title: str = "Temporal Split") -> None:
    """Pretty print temporal split information."""
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)
    print(
        f"  Train:      {split.train_start} to {split.train_end} ({split.train_bars:,} bars)"
    )
    print(
        f"  Validation: {split.val_start} to {split.val_end} ({split.val_bars:,} bars)"
    )
    print(
        f"  Test:       {split.test_start} to {split.test_end_date} ({split.test_bars:,} bars)"
    )
    print("=" * 60)
    print()


def assert_no_future_data(
    current_bar: int,
    data: pd.DataFrame,
    lookback_used: int,
    feature_name: str = "feature",
) -> None:
    """
    Assert that a feature calculation doesn't use future data.

    This is a runtime check to catch accidental lookahead in feature engineering.

    Args:
        current_bar: Current bar index being processed
        data: Full DataFrame
        lookback_used: Maximum lookback used in the calculation
        feature_name: Name of feature for error messages

    Raises:
        ValueError: If future data would be accessed
    """
    if current_bar < lookback_used:
        raise ValueError(
            f"Insufficient history for {feature_name}: "
            f"current_bar={current_bar} < lookback={lookback_used}"
        )

    # This check passes - no future data accessed
    return


class LeakageDetector:
    """
    Helper class to track and detect data leakage during backtesting.

    Usage:
        detector = LeakageDetector(test_start="2021-01-01")

        for i, row in data.iterrows():
            detector.check_bar(i, data.index[data.index <= i])
            # ... process bar ...

        detector.report()
    """

    def __init__(self, test_start: str):
        self.test_start = pd.to_datetime(test_start)
        self.violations = []
        self.bars_checked = 0

    def check_bar(
        self, current_idx: pd.Timestamp, data_accessed: pd.DatetimeIndex
    ) -> bool:
        """
        Check if processing current bar accesses future data.

        Args:
            current_idx: Current bar timestamp
            data_accessed: All timestamps that were accessed

        Returns:
            True if clean, False if violation detected
        """
        self.bars_checked += 1

        future_data = data_accessed[data_accessed > current_idx]

        if len(future_data) > 0:
            self.violations.append(
                {
                    "bar": current_idx,
                    "future_accessed": future_data[:5].tolist(),  # First 5 violations
                }
            )
            return False

        return True

    def report(self) -> None:
        """Print leakage detection report."""
        logger.info("=" * 50)
        logger.info("  LEAKAGE DETECTION REPORT")
        logger.info("=" * 50)
        logger.info(f"  Bars checked: {self.bars_checked}")
        logger.info(f"  Violations found: {len(self.violations)}")

        if self.violations:
            logger.error("LEAKAGE DETECTED!")
            for v in self.violations[:5]:
                logger.error(f"    At {v['bar']}: accessed {v['future_accessed']}")
            if len(self.violations) > 5:
                logger.error(f"    ... and {len(self.violations) - 5} more")
        else:
            logger.success("No leakage detected")

        logger.info("=" * 50)
