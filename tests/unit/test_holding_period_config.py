"""Tests for HoldingPeriodSettings, TradingWindowSettings, and is_allowed() on HoldingPeriodManager."""

import pytest

from quantstack.holding_period import HoldingType, HoldingPeriodManager


def _clear_caches():
    """Clear all cached settings so env var changes take effect."""
    from quantstack.config.settings import (
        get_holding_period_settings,
        get_trading_window_settings,
    )

    get_holding_period_settings.cache_clear()
    get_trading_window_settings.cache_clear()


class TestHoldingPeriodSettingsParsing:
    """Verify legacy env var parsing into HoldingPeriodSettings (deprecated path)."""

    def test_parse_single_type(self, monkeypatch):
        """'short_swing' → {HoldingType.SHORT_SWING}."""
        monkeypatch.setenv("TRADING_HOLDING_PERIODS", "short_swing")
        monkeypatch.setenv("RESEARCH_HOLDING_PERIODS", "intraday,short_swing,swing,position")
        from quantstack.config.settings import HoldingPeriodSettings

        settings = HoldingPeriodSettings()
        assert settings.trading_holding_periods == {HoldingType.SHORT_SWING}

    def test_parse_all_four_types(self, monkeypatch):
        """'intraday,short_swing,swing,position' → all four types."""
        monkeypatch.setenv("TRADING_HOLDING_PERIODS", "intraday,short_swing,swing,position")
        monkeypatch.setenv("RESEARCH_HOLDING_PERIODS", "intraday,short_swing,swing,position")
        from quantstack.config.settings import HoldingPeriodSettings

        settings = HoldingPeriodSettings()
        assert settings.trading_holding_periods == {
            HoldingType.INTRADAY,
            HoldingType.SHORT_SWING,
            HoldingType.SWING,
            HoldingType.POSITION,
        }

    def test_parse_empty_string_defaults_to_all(self, monkeypatch):
        """'' → default all four types."""
        monkeypatch.setenv("TRADING_HOLDING_PERIODS", "")
        monkeypatch.setenv("RESEARCH_HOLDING_PERIODS", "")
        from quantstack.config.settings import HoldingPeriodSettings

        settings = HoldingPeriodSettings()
        assert settings.trading_holding_periods == {
            HoldingType.INTRADAY,
            HoldingType.SHORT_SWING,
            HoldingType.SWING,
            HoldingType.POSITION,
        }

    def test_parse_unset_defaults_to_all(self, monkeypatch):
        """When env var is not set at all, defaults to all four types."""
        monkeypatch.delenv("TRADING_HOLDING_PERIODS", raising=False)
        monkeypatch.delenv("RESEARCH_HOLDING_PERIODS", raising=False)
        from quantstack.config.settings import HoldingPeriodSettings

        settings = HoldingPeriodSettings()
        assert settings.trading_holding_periods == {
            HoldingType.INTRADAY,
            HoldingType.SHORT_SWING,
            HoldingType.SWING,
            HoldingType.POSITION,
        }
        assert settings.research_holding_periods == {
            HoldingType.INTRADAY,
            HoldingType.SHORT_SWING,
            HoldingType.SWING,
            HoldingType.POSITION,
        }

    def test_parse_invalid_type_raises(self, monkeypatch):
        """'invalid_type' → raise ValueError with clear message."""
        monkeypatch.setenv("TRADING_HOLDING_PERIODS", "invalid_type")
        from quantstack.config.settings import HoldingPeriodSettings

        settings = HoldingPeriodSettings()
        with pytest.raises(ValueError, match="invalid_type"):
            _ = settings.trading_holding_periods

    def test_parse_strips_whitespace(self, monkeypatch):
        """'short_swing, swing' (with spaces) → strips whitespace correctly."""
        monkeypatch.setenv("TRADING_HOLDING_PERIODS", "short_swing, swing")
        monkeypatch.setenv("RESEARCH_HOLDING_PERIODS", "intraday,short_swing,swing,position")
        from quantstack.config.settings import HoldingPeriodSettings

        settings = HoldingPeriodSettings()
        assert settings.trading_holding_periods == {
            HoldingType.SHORT_SWING,
            HoldingType.SWING,
        }

    def test_parse_case_insensitive(self, monkeypatch):
        """'SHORT_SWING' → works (case insensitive)."""
        monkeypatch.setenv("TRADING_HOLDING_PERIODS", "SHORT_SWING")
        monkeypatch.setenv("RESEARCH_HOLDING_PERIODS", "intraday,short_swing,swing,position")
        from quantstack.config.settings import HoldingPeriodSettings

        settings = HoldingPeriodSettings()
        assert settings.trading_holding_periods == {HoldingType.SHORT_SWING}


class TestIsAllowed:
    """Verify HoldingPeriodManager.is_allowed() with TradingWindow system."""

    def test_allowed_type_for_trading(self, monkeypatch):
        """is_allowed(SHORT_SWING, 'trading') returns True when options_weekly in TRADING_WINDOW."""
        monkeypatch.delenv("TRADING_HOLDING_PERIODS", raising=False)
        monkeypatch.delenv("RESEARCH_HOLDING_PERIODS", raising=False)
        monkeypatch.setenv("TRADING_WINDOW", "options_weekly")
        monkeypatch.setenv("RESEARCH_WINDOW", "all")
        _clear_caches()

        mgr = HoldingPeriodManager()
        # options_weekly compatible types: INTRADAY, SHORT_SWING
        assert mgr.is_allowed(HoldingType.SHORT_SWING, "trading") is True

    def test_disallowed_type_for_trading(self, monkeypatch):
        """is_allowed(POSITION, 'trading') returns False when only options_weekly."""
        monkeypatch.delenv("TRADING_HOLDING_PERIODS", raising=False)
        monkeypatch.delenv("RESEARCH_HOLDING_PERIODS", raising=False)
        monkeypatch.setenv("TRADING_WINDOW", "options_weekly")
        monkeypatch.setenv("RESEARCH_WINDOW", "all")
        _clear_caches()

        mgr = HoldingPeriodManager()
        # options_weekly compatible types: INTRADAY, SHORT_SWING — POSITION not included
        assert mgr.is_allowed(HoldingType.POSITION, "trading") is False

    def test_allowed_type_for_research(self, monkeypatch):
        """is_allowed(SWING, 'research') returns True when research=all."""
        monkeypatch.delenv("TRADING_HOLDING_PERIODS", raising=False)
        monkeypatch.delenv("RESEARCH_HOLDING_PERIODS", raising=False)
        monkeypatch.setenv("TRADING_WINDOW", "options_weekly")
        monkeypatch.setenv("RESEARCH_WINDOW", "all")
        _clear_caches()

        mgr = HoldingPeriodManager()
        assert mgr.is_allowed(HoldingType.SWING, "research") is True

    def test_default_config_all_allowed(self, monkeypatch):
        """With default config (unset env vars) → all types allowed for both contexts."""
        monkeypatch.delenv("TRADING_HOLDING_PERIODS", raising=False)
        monkeypatch.delenv("RESEARCH_HOLDING_PERIODS", raising=False)
        monkeypatch.delenv("TRADING_WINDOW", raising=False)
        monkeypatch.delenv("RESEARCH_WINDOW", raising=False)
        _clear_caches()

        mgr = HoldingPeriodManager()
        for ht in HoldingType:
            assert mgr.is_allowed(ht, "trading") is True
            assert mgr.is_allowed(ht, "research") is True

    def test_invalid_context_raises(self):
        """Invalid context string raises ValueError."""
        mgr = HoldingPeriodManager()
        with pytest.raises(ValueError, match="context"):
            mgr.is_allowed(HoldingType.SWING, "invalid")

    def test_equity_only_window_rejects_position_holding(self, monkeypatch):
        """equity_scalp only allows INTRADAY holding type."""
        monkeypatch.delenv("TRADING_HOLDING_PERIODS", raising=False)
        monkeypatch.setenv("TRADING_WINDOW", "equity_scalp")
        monkeypatch.setenv("RESEARCH_WINDOW", "all")
        _clear_caches()

        mgr = HoldingPeriodManager()
        assert mgr.is_allowed(HoldingType.INTRADAY, "trading") is True
        assert mgr.is_allowed(HoldingType.SWING, "trading") is False
        assert mgr.is_allowed(HoldingType.POSITION, "trading") is False

    def test_composite_window_expands_holding_types(self, monkeypatch):
        """options_short_term expands to 0dte+weekly+biweekly → INTRADAY,SHORT_SWING."""
        monkeypatch.delenv("TRADING_HOLDING_PERIODS", raising=False)
        monkeypatch.setenv("TRADING_WINDOW", "options_short_term")
        monkeypatch.setenv("RESEARCH_WINDOW", "all")
        _clear_caches()

        mgr = HoldingPeriodManager()
        assert mgr.is_allowed(HoldingType.INTRADAY, "trading") is True
        assert mgr.is_allowed(HoldingType.SHORT_SWING, "trading") is True
        assert mgr.is_allowed(HoldingType.SWING, "trading") is False
        assert mgr.is_allowed(HoldingType.POSITION, "trading") is False


class TestHoldingPeriodSettingsCaching:
    """Verify get_holding_period_settings() caching."""

    def test_returns_cached_instance(self, monkeypatch):
        """get_holding_period_settings() returns same object on repeated calls."""
        monkeypatch.delenv("TRADING_HOLDING_PERIODS", raising=False)
        monkeypatch.delenv("RESEARCH_HOLDING_PERIODS", raising=False)
        from quantstack.config.settings import get_holding_period_settings
        get_holding_period_settings.cache_clear()

        a = get_holding_period_settings()
        b = get_holding_period_settings()
        assert a is b
