# Copyright 2024 QuantCore Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for quantcore.config.settings module."""

import os
from unittest.mock import patch

import pytest

from quantcore.config.settings import Settings, get_settings


class TestSettings:
    """Test Settings class."""

    def test_default_values(self):
        """Test that Settings has correct default values."""
        settings = Settings()

        # API defaults - key may be overridden by env var, so just check it exists
        assert settings.alpha_vantage_api_key is not None
        assert settings.alpha_vantage_base_url == "https://www.alphavantage.co/query"
        assert settings.alpha_vantage_rate_limit == 5

        # Database defaults
        assert settings.database_path == "data/trader.duckdb"

        # Symbol defaults
        assert settings.symbols == ["SPY", "QQQ", "AAPL", "MSFT", "NVDA"]
        assert settings.benchmark_symbol == "SPY"

        # Data settings
        assert settings.market_timezone == "America/New_York"

        # Risk defaults
        assert settings.max_risk_per_trade_bps == 25.0
        assert settings.max_daily_risk_pct == 1.0
        assert settings.max_concurrent_trades == 5
        assert settings.soft_stop_drawdown_pct == 3.0
        assert settings.hard_stop_drawdown_pct == 7.0

        # Cost defaults
        assert settings.spread_cost_bps == 2.0
        assert settings.slippage_cost_bps == 1.0
        assert settings.fee_cost_bps == 2.0

        # Logging defaults
        assert settings.log_level == "INFO"
        assert settings.log_file == "logs/trader.log"

    def test_total_transaction_cost_property(self):
        """Test total transaction cost calculation."""
        settings = Settings()

        # Default: (2 + 1 + 2) * 2 = 10 bps round trip
        expected = (
            settings.spread_cost_bps
            + settings.slippage_cost_bps
            + settings.fee_cost_bps
        ) * 2
        assert settings.total_transaction_cost_bps == expected
        assert settings.total_transaction_cost_bps == 10.0

    def test_total_transaction_cost_with_custom_values(self):
        """Test transaction cost with custom values."""
        settings = Settings(
            spread_cost_bps=3.0,
            slippage_cost_bps=2.0,
            fee_cost_bps=1.0,
        )

        # (3 + 2 + 1) * 2 = 12 bps
        assert settings.total_transaction_cost_bps == 12.0

    def test_env_var_override(self):
        """Test that environment variables override defaults."""
        with patch.dict(
            os.environ,
            {
                "ALPHA_VANTAGE_API_KEY": "test_key_123",
                "DATABASE_PATH": "/custom/path/db.duckdb",
                "MAX_CONCURRENT_TRADES": "10",
            },
        ):
            # Clear cache to pick up new env vars
            get_settings.cache_clear()
            settings = Settings()

            assert settings.alpha_vantage_api_key == "test_key_123"
            assert settings.database_path == "/custom/path/db.duckdb"
            assert settings.max_concurrent_trades == 10

    def test_date_fields(self):
        """Test date fields have valid defaults."""
        settings = Settings()

        # Training dates
        assert settings.train_start_date == "2019-01-01"
        assert settings.train_end_date == "2022-12-31"

        # Validation dates
        assert settings.validation_start_date == "2023-01-01"
        assert settings.validation_end_date == "2023-06-30"

        # Test dates
        assert settings.test_start_date == "2023-07-01"
        assert settings.test_end_date == "2024-12-31"

    def test_model_settings(self):
        """Test model-related settings."""
        settings = Settings()
        assert settings.model_probability_threshold == 0.6
        assert 0 < settings.model_probability_threshold < 1


class TestGetSettings:
    """Test get_settings function."""

    def test_returns_settings_instance(self):
        """Test that get_settings returns a Settings instance."""
        get_settings.cache_clear()
        settings = get_settings()
        assert isinstance(settings, Settings)

    def test_caching(self):
        """Test that get_settings returns cached instance."""
        get_settings.cache_clear()
        settings1 = get_settings()
        settings2 = get_settings()
        assert settings1 is settings2

    def test_cache_clear(self):
        """Test that cache can be cleared."""
        get_settings.cache_clear()
        settings1 = get_settings()

        get_settings.cache_clear()
        settings2 = get_settings()

        # After cache clear, should be different instances
        # (though equal in value)
        assert settings1 is not settings2
        assert settings1.alpha_vantage_api_key == settings2.alpha_vantage_api_key
