"""
Tests for input validation utilities.

Verifies that:
1. OHLCV validation catches common issues
2. Spread data validation works correctly
3. Feature matrix validation identifies problems
4. Helper validators work correctly
"""

import numpy as np
import pandas as pd
import pytest

from quantcore.validation.input_validation import (
    ValidationResult,
    DataFrameValidator,
    validate_positive_number,
    validate_in_range,
)


class TestValidationResult:
    """Test ValidationResult dataclass."""

    def test_valid_result(self):
        """Valid result should have is_valid=True and no errors."""
        result = ValidationResult(is_valid=True, errors=[], warnings=["minor warning"])
        assert result.is_valid
        assert len(result.errors) == 0
        assert len(result.warnings) == 1

    def test_invalid_result(self):
        """Invalid result should have is_valid=False and errors."""
        result = ValidationResult(is_valid=False, errors=["critical error"])
        assert not result.is_valid
        assert len(result.errors) == 1

    def test_raise_if_invalid(self):
        """raise_if_invalid should raise ValueError for invalid result."""
        result = ValidationResult(is_valid=False, errors=["test error"])
        with pytest.raises(ValueError, match="test error"):
            result.raise_if_invalid("Test")

    def test_raise_if_invalid_passes_for_valid(self):
        """raise_if_invalid should not raise for valid result."""
        result = ValidationResult(is_valid=True)
        result.raise_if_invalid("Test")  # Should not raise

    def test_str_representation(self):
        """String representation should include status and errors."""
        result = ValidationResult(
            is_valid=False, errors=["error 1", "error 2"], warnings=["warning 1"]
        )
        result_str = str(result)
        assert "INVALID" in result_str
        assert "error 1" in result_str


class TestOHLCVValidation:
    """Test OHLCV DataFrame validation."""

    @pytest.fixture
    def valid_ohlcv(self):
        """Create valid OHLCV DataFrame."""
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        return pd.DataFrame(
            {
                "open": np.random.uniform(90, 110, 100),
                "high": np.random.uniform(100, 120, 100),
                "low": np.random.uniform(80, 100, 100),
                "close": np.random.uniform(90, 110, 100),
                "volume": np.random.randint(1000, 10000, 100),
            },
            index=dates,
        )

    def test_valid_ohlcv_passes(self, valid_ohlcv):
        """Valid OHLCV data should pass validation."""
        # Make OHLC consistent
        valid_ohlcv["high"] = valid_ohlcv[["open", "close"]].max(axis=1) + 1
        valid_ohlcv["low"] = valid_ohlcv[["open", "close"]].min(axis=1) - 1

        result = DataFrameValidator.validate_ohlcv(valid_ohlcv)
        assert result.is_valid, f"Valid OHLCV should pass: {result.errors}"

    def test_none_dataframe_fails(self):
        """None DataFrame should fail validation."""
        result = DataFrameValidator.validate_ohlcv(None)
        assert not result.is_valid
        assert any("None" in e for e in result.errors)

    def test_empty_dataframe_fails(self):
        """Empty DataFrame should fail validation."""
        result = DataFrameValidator.validate_ohlcv(pd.DataFrame())
        assert not result.is_valid
        assert any("empty" in e.lower() for e in result.errors)

    def test_missing_close_column_fails(self):
        """Missing close column should fail validation."""
        df = pd.DataFrame(
            {
                "open": [100, 101, 102],
                "high": [105, 106, 107],
                "low": [95, 96, 97],
                "volume": [1000, 1100, 1200],
            }
        )
        result = DataFrameValidator.validate_ohlcv(df)
        assert not result.is_valid
        assert any("close" in e.lower() for e in result.errors)

    def test_nan_in_close_detected(self):
        """NaN values in close should be detected."""
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        df = pd.DataFrame(
            {
                "close": [100, 101, np.nan, 103, 104, np.nan, 106, 107, 108, 109],
            },
            index=dates,
        )

        result = DataFrameValidator.validate_ohlcv(df)
        # Should have warnings or errors about NaN
        has_nan_message = any(
            "nan" in (e.lower() + w.lower())
            for e in result.errors
            for w in result.warnings
        ) or any("nan" in w.lower() for w in result.warnings)
        assert has_nan_message, f"NaN should be detected: {result}"

    def test_negative_prices_fail(self):
        """Negative prices should fail validation."""
        dates = pd.date_range("2020-01-01", periods=5, freq="D")
        df = pd.DataFrame(
            {
                "close": [100, 101, -5, 103, 104],  # Negative price
            },
            index=dates,
        )

        result = DataFrameValidator.validate_ohlcv(df)
        assert not result.is_valid
        assert any("negative" in e.lower() for e in result.errors)

    def test_infinite_values_fail(self):
        """Infinite values should fail validation."""
        dates = pd.date_range("2020-01-01", periods=5, freq="D")
        df = pd.DataFrame(
            {
                "close": [100, 101, np.inf, 103, 104],
            },
            index=dates,
        )

        result = DataFrameValidator.validate_ohlcv(df)
        assert not result.is_valid
        assert any("inf" in e.lower() for e in result.errors)

    def test_unsorted_index_fails(self):
        """Unsorted DatetimeIndex should fail validation."""
        dates = pd.DatetimeIndex(["2020-01-03", "2020-01-01", "2020-01-02"])
        df = pd.DataFrame({"close": [100, 101, 102]}, index=dates)

        result = DataFrameValidator.validate_ohlcv(df)
        assert not result.is_valid
        assert any("sorted" in e.lower() for e in result.errors)

    def test_high_less_than_low_fails(self):
        """OHLC where high < low should fail."""
        dates = pd.date_range("2020-01-01", periods=3, freq="D")
        df = pd.DataFrame(
            {
                "open": [100, 101, 102],
                "high": [95, 96, 97],  # Less than low!
                "low": [100, 101, 102],
                "close": [98, 99, 100],
            },
            index=dates,
        )

        result = DataFrameValidator.validate_ohlcv(df)
        assert not result.is_valid
        assert any("high" in e.lower() and "low" in e.lower() for e in result.errors)

    def test_negative_volume_fails(self):
        """Negative volume should fail validation."""
        dates = pd.date_range("2020-01-01", periods=3, freq="D")
        df = pd.DataFrame(
            {
                "close": [100, 101, 102],
                "volume": [1000, -500, 1200],  # Negative volume
            },
            index=dates,
        )

        result = DataFrameValidator.validate_ohlcv(df)
        assert not result.is_valid
        assert any(
            "volume" in e.lower() and "negative" in e.lower() for e in result.errors
        )

    def test_raise_on_error_works(self):
        """raise_on_error=True should raise ValueError."""
        with pytest.raises(ValueError):
            DataFrameValidator.validate_ohlcv(None, raise_on_error=True)


class TestSpreadDataValidation:
    """Test spread data validation."""

    @pytest.fixture
    def valid_spread_data(self):
        """Create valid spread data."""
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        return pd.DataFrame(
            {
                "spread": np.random.normal(0, 1, 100),
                "wti": np.random.uniform(40, 80, 100),
                "brent": np.random.uniform(42, 82, 100),
            },
            index=dates,
        )

    def test_valid_spread_data_passes(self, valid_spread_data):
        """Valid spread data should pass validation."""
        result = DataFrameValidator.validate_spread_data(valid_spread_data)
        assert result.is_valid, f"Valid spread data should pass: {result.errors}"

    def test_missing_spread_column_fails(self):
        """Missing spread column should fail."""
        df = pd.DataFrame(
            {
                "wti": [50, 51, 52],
                "brent": [52, 53, 54],
            }
        )
        result = DataFrameValidator.validate_spread_data(df)
        assert not result.is_valid
        assert any("spread" in e.lower() for e in result.errors)

    def test_nan_in_spread_detected(self):
        """NaN in spread column should be detected."""
        dates = pd.date_range("2020-01-01", periods=5, freq="D")
        df = pd.DataFrame(
            {
                "spread": [0.1, np.nan, 0.3, 0.4, np.nan],
            },
            index=dates,
        )

        result = DataFrameValidator.validate_spread_data(df)
        # High NaN rate (40%) should be detected
        has_nan_message = any("nan" in e.lower() for e in result.errors) or any(
            "nan" in w.lower() for w in result.warnings
        )
        assert has_nan_message, f"NaN should be detected: {result}"

    def test_infinite_spread_fails(self):
        """Infinite spread values should fail."""
        dates = pd.date_range("2020-01-01", periods=3, freq="D")
        df = pd.DataFrame(
            {
                "spread": [0.1, np.inf, 0.3],
            },
            index=dates,
        )

        result = DataFrameValidator.validate_spread_data(df)
        assert not result.is_valid
        assert any("inf" in e.lower() for e in result.errors)

    def test_optional_columns_noted(self, valid_spread_data):
        """Optional columns should be noted in info."""
        result = DataFrameValidator.validate_spread_data(valid_spread_data)
        assert "wti" in result.info.get("optional_columns_present", [])
        assert "brent" in result.info.get("optional_columns_present", [])


class TestFeatureMatrixValidation:
    """Test feature matrix validation."""

    def test_valid_feature_matrix_passes(self):
        """Valid feature matrix should pass."""
        df = pd.DataFrame(
            {
                "feature1": np.random.normal(0, 1, 100),
                "feature2": np.random.normal(0, 1, 100),
                "feature3": np.random.normal(0, 1, 100),
            }
        )
        result = DataFrameValidator.validate_feature_matrix(df)
        assert result.is_valid

    def test_all_nan_column_fails(self):
        """Column with all NaN should fail."""
        df = pd.DataFrame(
            {
                "good": np.random.normal(0, 1, 10),
                "bad": [np.nan] * 10,
            }
        )
        result = DataFrameValidator.validate_feature_matrix(df)
        assert not result.is_valid
        assert any("nan" in e.lower() for e in result.errors)

    def test_infinite_values_fail(self):
        """Infinite values should fail."""
        df = pd.DataFrame(
            {
                "feature1": [1, 2, np.inf, 4, 5],
            }
        )
        result = DataFrameValidator.validate_feature_matrix(df)
        assert not result.is_valid
        assert any("inf" in e.lower() for e in result.errors)

    def test_high_nan_rate_warns(self):
        """High NaN rate should produce warnings."""
        df = pd.DataFrame(
            {
                "feature1": [1, np.nan, np.nan, np.nan, np.nan],  # 80% NaN
            }
        )
        result = DataFrameValidator.validate_feature_matrix(df, max_nan_pct=5.0)
        # Should have warning about high NaN
        assert any(
            "nan" in w.lower() for w in result.warnings
        ), f"High NaN rate should be warned: {result}"


class TestHelperValidators:
    """Test helper validation functions."""

    def test_positive_number_valid(self):
        """Valid positive numbers should pass."""
        validate_positive_number(1.5, "test")
        validate_positive_number(100, "test")
        validate_positive_number(0.001, "test")

    def test_positive_number_none_fails(self):
        """None should fail."""
        with pytest.raises(ValueError, match="None"):
            validate_positive_number(None, "test")

    def test_positive_number_negative_fails(self):
        """Negative numbers should fail."""
        with pytest.raises(ValueError, match="positive"):
            validate_positive_number(-1, "test")

    def test_positive_number_zero_fails_by_default(self):
        """Zero should fail by default."""
        with pytest.raises(ValueError, match="positive"):
            validate_positive_number(0, "test")

    def test_positive_number_zero_allowed(self):
        """Zero should pass when allow_zero=True."""
        validate_positive_number(0, "test", allow_zero=True)

    def test_positive_number_nan_fails(self):
        """NaN should fail."""
        with pytest.raises(ValueError, match="NaN"):
            validate_positive_number(np.nan, "test")

    def test_positive_number_inf_fails(self):
        """Infinite should fail."""
        with pytest.raises(ValueError, match="infinite"):
            validate_positive_number(np.inf, "test")

    def test_in_range_valid(self):
        """Values in range should pass."""
        validate_in_range(0.5, "test", min_val=0, max_val=1)
        validate_in_range(0, "test", min_val=0, max_val=1)
        validate_in_range(1, "test", min_val=0, max_val=1)

    def test_in_range_below_min_fails(self):
        """Values below min should fail."""
        with pytest.raises(ValueError, match=">="):
            validate_in_range(-1, "test", min_val=0)

    def test_in_range_above_max_fails(self):
        """Values above max should fail."""
        with pytest.raises(ValueError, match="<="):
            validate_in_range(2, "test", max_val=1)

    def test_in_range_none_fails(self):
        """None should fail."""
        with pytest.raises(ValueError, match="None"):
            validate_in_range(None, "test", min_val=0, max_val=1)
