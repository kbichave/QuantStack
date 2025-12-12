"""
Tests for economic features module.

Validates:
- No lookahead bias in forward-filling
- Correct merge with market data
- Proper handling of different frequencies
"""

from datetime import datetime, timedelta

import pandas as pd
import pytest

from quantcore.data.economic_storage import EconomicStorage
from quantcore.features.economic_features import (
    EconomicFeatureEngineer,
    create_economic_features_for_symbol,
)


@pytest.fixture
def mock_economic_data():
    """Create mock economic indicator data."""
    dates = pd.date_range("2020-01-01", "2023-12-31", freq="D")

    # Monthly indicator (e.g., unemployment)
    monthly_dates = pd.date_range("2020-01-01", "2023-12-31", freq="MS")
    monthly_data = pd.DataFrame(
        {
            "date": monthly_dates,
            "value": [5.0 + i * 0.1 for i in range(len(monthly_dates))],
            "indicator": "unemployment",
            "frequency": "monthly",
        }
    )

    # Daily indicator (e.g., treasury yield)
    daily_data = pd.DataFrame(
        {
            "date": dates,
            "value": [2.0 + i * 0.001 for i in range(len(dates))],
            "indicator": "treasury_10y",
            "frequency": "daily",
        }
    )

    return {"unemployment": monthly_data, "treasury_10y": daily_data}


@pytest.fixture
def mock_market_data():
    """Create mock market data."""
    dates = pd.bdate_range("2020-01-01", "2023-12-31")
    return pd.DataFrame(
        {
            "date": dates,
            "open": 100.0,
            "high": 105.0,
            "low": 95.0,
            "close": 102.0,
            "volume": 1000000,
        }
    ).set_index("date")


def test_forward_fill_no_lookahead(mock_economic_data, tmp_path):
    """Test that forward-fill doesn't introduce lookahead bias."""
    # Create temporary storage
    storage = EconomicStorage(db_path=tmp_path / "test_economic.duckdb")

    # Store monthly data
    storage.store_indicator("unemployment", mock_economic_data["unemployment"])

    # Create features
    engineer = EconomicFeatureEngineer(storage)
    date_range = pd.date_range("2020-01-01", "2020-03-31", freq="D")
    features = engineer.create_daily_features(date_range, indicators=["unemployment"])

    # Check that forward-fill is working correctly
    # January value should be used for all of January
    jan_value = mock_economic_data["unemployment"][
        mock_economic_data["unemployment"]["date"] == "2020-01-01"
    ]["value"].iloc[0]

    jan_features = features.loc["2020-01-01":"2020-01-31"]
    assert (jan_features["unemployment"] == jan_value).all()

    # February value should be used starting from Feb 1
    feb_value = mock_economic_data["unemployment"][
        mock_economic_data["unemployment"]["date"] == "2020-02-01"
    ]["value"].iloc[0]

    feb_features = features.loc["2020-02-01":"2020-02-29"]
    assert (feb_features["unemployment"] == feb_value).all()

    # Verify no future data is used
    for i in range(len(features) - 1):
        current_val = features["unemployment"].iloc[i]
        next_val = features["unemployment"].iloc[i + 1]
        # Value should stay same or increase, never decrease (since our test data is monotonic)
        # In real data, it could change, but should never jump ahead
        assert next_val >= current_val


def test_yield_curve_features(mock_economic_data, tmp_path):
    """Test yield curve spread calculation."""
    storage = EconomicStorage(db_path=tmp_path / "test_economic.duckdb")

    # Create 3-month treasury data
    dates = pd.date_range("2020-01-01", "2023-12-31", freq="D")
    treasury_3m = pd.DataFrame(
        {
            "date": dates,
            "value": [1.5 + i * 0.0005 for i in range(len(dates))],
            "indicator": "treasury_3m",
            "frequency": "daily",
        }
    )

    storage.store_indicator("treasury_10y", mock_economic_data["treasury_10y"])
    storage.store_indicator("treasury_3m", treasury_3m)

    engineer = EconomicFeatureEngineer(storage)
    features = engineer.create_daily_features(
        pd.date_range("2020-01-01", "2020-12-31", freq="D")
    )

    # Check yield curve spread is calculated
    assert "yield_curve_10y3m" in features.columns
    assert (features["yield_curve_10y3m"] > 0).all()  # Normal curve in our test data

    # Spread should equal 10y - 3m
    expected_spread = features["treasury_10y"] - features["treasury_3m"]
    pd.testing.assert_series_equal(
        features["yield_curve_10y3m"], expected_spread, check_names=False
    )


def test_change_features(mock_economic_data, tmp_path):
    """Test rate of change features."""
    storage = EconomicStorage(db_path=tmp_path / "test_economic.duckdb")
    storage.store_indicator("unemployment", mock_economic_data["unemployment"])

    engineer = EconomicFeatureEngineer(storage)
    features = engineer.create_daily_features(
        pd.date_range("2020-01-01", "2023-12-31", freq="D"),
        indicators=["unemployment"],
    )

    # Check change features exist
    assert "unemployment_mom" in features.columns
    assert "unemployment_qoq" in features.columns
    assert "unemployment_yoy" in features.columns

    # YoY should have ~252 NaN at start (1 year)
    assert features["unemployment_yoy"].isna().sum() >= 252


def test_regime_features(mock_economic_data, tmp_path):
    """Test regime classification features."""
    storage = EconomicStorage(db_path=tmp_path / "test_economic.duckdb")

    # Create inverted yield curve scenario
    dates = pd.date_range("2020-01-01", "2020-12-31", freq="D")
    treasury_10y = pd.DataFrame(
        {
            "date": dates,
            "value": [1.5] * len(dates),  # 10y at 1.5%
            "indicator": "treasury_10y",
            "frequency": "daily",
        }
    )
    treasury_3m = pd.DataFrame(
        {
            "date": dates,
            "value": [2.0] * len(dates),  # 3m at 2% (inverted!)
            "indicator": "treasury_3m",
            "frequency": "daily",
        }
    )

    storage.store_indicator("treasury_10y", treasury_10y)
    storage.store_indicator("treasury_3m", treasury_3m)

    engineer = EconomicFeatureEngineer(storage)
    features = engineer.create_daily_features(dates)
    features = engineer.create_regime_features(features)

    # Check recession risk is flagged
    assert "recession_risk" in features.columns
    assert (features["recession_risk"] == 1).all()  # Inverted curve


def test_merge_with_market_data(mock_economic_data, mock_market_data, tmp_path):
    """Test merging economic features with market data."""
    storage = EconomicStorage(db_path=tmp_path / "test_economic.duckdb")
    storage.store_indicator("treasury_10y", mock_economic_data["treasury_10y"])

    engineer = EconomicFeatureEngineer(storage)

    # Get business days from market data
    economic_features = engineer.create_daily_features(mock_market_data.index)

    # Merge
    merged = engineer.merge_with_market_data(mock_market_data, economic_features)

    # Check structure
    assert len(merged) == len(mock_market_data)
    assert "treasury_10y" in merged.columns
    assert "close" in merged.columns

    # Check no NaN in economic features (should be forward-filled)
    assert not merged["treasury_10y"].isna().any()


def test_convenience_function(mock_economic_data, mock_market_data, tmp_path):
    """Test convenience function for creating features."""
    storage = EconomicStorage(db_path=tmp_path / "test_economic.duckdb")
    storage.store_indicator("treasury_10y", mock_economic_data["treasury_10y"])

    result = create_economic_features_for_symbol("SPY", mock_market_data, storage)

    # Check that both market and economic features are present
    assert "close" in result.columns
    assert "treasury_10y" in result.columns
    assert len(result) == len(mock_market_data)


def test_feature_groups():
    """Test feature importance groupings."""
    engineer = EconomicFeatureEngineer()
    groups = engineer.get_feature_importance_groups()

    # Check expected groups exist
    assert "yield_curve" in groups
    assert "inflation" in groups
    assert "labor" in groups
    assert "growth" in groups
    assert "recession_signals" in groups

    # Check some specific features
    assert "treasury_10y" in groups["yield_curve"]
    assert "cpi" in groups["inflation"]
    assert "unemployment" in groups["labor"]
