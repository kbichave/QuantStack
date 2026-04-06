"""Tests for quantstack.trading_window module — enums, parsing, expansion, validation."""

import pytest

from quantstack.holding_period import HoldingType
from quantstack.trading_window import (
    COMPOSITE_EXPANSIONS,
    WINDOW_SPECS,
    InstrumentType,
    TradingWindow,
    allowed_holding_types,
    allowed_instrument_types,
    expand_windows,
    get_dte_bounds,
    get_hold_days_bounds,
    is_trade_allowed,
    parse_window_env,
)


# ---------------------------------------------------------------------------
# InstrumentType
# ---------------------------------------------------------------------------


class TestInstrumentType:
    def test_values(self):
        assert InstrumentType.EQUITY == "equity"
        assert InstrumentType.OPTIONS == "options"

    def test_construct_from_string(self):
        assert InstrumentType("equity") is InstrumentType.EQUITY
        assert InstrumentType("options") is InstrumentType.OPTIONS

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            InstrumentType("futures")


# ---------------------------------------------------------------------------
# TradingWindow enum
# ---------------------------------------------------------------------------


class TestTradingWindowEnum:
    def test_leaf_count(self):
        """14 leaf windows (those with WindowSpec entries)."""
        assert len(WINDOW_SPECS) == 14

    def test_composite_count(self):
        """6 composite shortcuts."""
        assert len(COMPOSITE_EXPANSIONS) == 6

    def test_all_composites_expand_to_leaves_only(self):
        for composite, leaves in COMPOSITE_EXPANSIONS.items():
            for leaf in leaves:
                assert leaf in WINDOW_SPECS, f"{composite} contains non-leaf {leaf}"


# ---------------------------------------------------------------------------
# expand_windows
# ---------------------------------------------------------------------------


class TestExpandWindows:
    def test_leaf_passes_through(self):
        result = expand_windows({TradingWindow.OPTIONS_WEEKLY})
        assert result == {TradingWindow.OPTIONS_WEEKLY}

    def test_composite_expands(self):
        result = expand_windows({TradingWindow.OPTIONS_SHORT_TERM})
        assert result == {
            TradingWindow.OPTIONS_0DTE,
            TradingWindow.OPTIONS_WEEKLY,
            TradingWindow.OPTIONS_BIWEEKLY,
        }

    def test_all_expands_to_all_leaves(self):
        result = expand_windows({TradingWindow.ALL})
        assert result == set(WINDOW_SPECS.keys())

    def test_mixed_leaf_and_composite(self):
        result = expand_windows({
            TradingWindow.EQUITY_INVESTMENT,
            TradingWindow.OPTIONS_SHORT_TERM,
        })
        assert TradingWindow.EQUITY_INVESTMENT in result
        assert TradingWindow.OPTIONS_0DTE in result
        assert TradingWindow.OPTIONS_WEEKLY in result
        assert TradingWindow.OPTIONS_BIWEEKLY in result
        assert len(result) == 4

    def test_equity_long_term_composite(self):
        result = expand_windows({TradingWindow.EQUITY_LONG_TERM})
        assert result == {
            TradingWindow.EQUITY_POSITION,
            TradingWindow.EQUITY_INVESTMENT,
        }


# ---------------------------------------------------------------------------
# parse_window_env
# ---------------------------------------------------------------------------


class TestParseWindowEnv:
    def test_empty_string_defaults_to_all(self):
        result = parse_window_env("")
        assert result == expand_windows({TradingWindow.ALL})

    def test_single_leaf(self):
        result = parse_window_env("options_weekly")
        assert result == {TradingWindow.OPTIONS_WEEKLY}

    def test_single_composite(self):
        result = parse_window_env("options_short_term")
        assert TradingWindow.OPTIONS_0DTE in result
        assert TradingWindow.OPTIONS_WEEKLY in result
        assert TradingWindow.OPTIONS_BIWEEKLY in result

    def test_comma_separated_mixed(self):
        result = parse_window_env("options_weekly,equity_swing")
        assert result == {TradingWindow.OPTIONS_WEEKLY, TradingWindow.EQUITY_SWING}

    def test_case_insensitive(self):
        result = parse_window_env("OPTIONS_WEEKLY")
        assert result == {TradingWindow.OPTIONS_WEEKLY}

    def test_strips_whitespace(self):
        result = parse_window_env("  options_weekly , equity_swing  ")
        assert result == {TradingWindow.OPTIONS_WEEKLY, TradingWindow.EQUITY_SWING}

    def test_invalid_token_raises(self):
        with pytest.raises(ValueError, match="invalid_window"):
            parse_window_env("invalid_window")

    def test_all_keyword(self):
        result = parse_window_env("all")
        assert len(result) == 14  # all 14 leaves


# ---------------------------------------------------------------------------
# is_trade_allowed
# ---------------------------------------------------------------------------


class TestIsTradeAllowed:
    def test_options_within_weekly_dte(self):
        windows = expand_windows({TradingWindow.OPTIONS_WEEKLY})
        assert is_trade_allowed("options", windows, dte=5) is True

    def test_options_outside_weekly_dte(self):
        windows = expand_windows({TradingWindow.OPTIONS_WEEKLY})
        assert is_trade_allowed("options", windows, dte=30) is False

    def test_options_0dte(self):
        windows = expand_windows({TradingWindow.OPTIONS_0DTE})
        assert is_trade_allowed("options", windows, dte=0) is True
        assert is_trade_allowed("options", windows, dte=1) is False

    def test_equity_allowed_by_swing_window(self):
        windows = expand_windows({TradingWindow.EQUITY_SWING})
        assert is_trade_allowed("equity", windows) is True

    def test_equity_rejected_when_only_options(self):
        windows = expand_windows({TradingWindow.OPTIONS_WEEKLY})
        assert is_trade_allowed("equity", windows) is False

    def test_options_rejected_when_only_equity(self):
        windows = expand_windows({TradingWindow.EQUITY_SWING})
        assert is_trade_allowed("options", windows, dte=5) is False

    def test_all_windows_allow_everything(self):
        windows = expand_windows({TradingWindow.ALL})
        assert is_trade_allowed("equity", windows) is True
        assert is_trade_allowed("options", windows, dte=0) is True
        assert is_trade_allowed("options", windows, dte=500) is True

    def test_options_without_dte_rejected(self):
        """Options trade without DTE info can't be validated."""
        windows = expand_windows({TradingWindow.OPTIONS_WEEKLY})
        assert is_trade_allowed("options", windows, dte=None) is False

    def test_instrument_type_as_enum(self):
        windows = expand_windows({TradingWindow.EQUITY_SWING})
        assert is_trade_allowed(InstrumentType.EQUITY, windows) is True

    def test_mixed_windows_allow_both(self):
        windows = expand_windows({TradingWindow.OPTIONS_WEEKLY, TradingWindow.EQUITY_SWING})
        assert is_trade_allowed("equity", windows) is True
        assert is_trade_allowed("options", windows, dte=5) is True
        assert is_trade_allowed("options", windows, dte=30) is False

    def test_equity_hold_days_gating(self):
        windows = expand_windows({TradingWindow.EQUITY_SCALP})
        # Scalp: hold_days 0-0, so hold_days=5 is outside
        assert is_trade_allowed("equity", windows, hold_days=5) is False
        assert is_trade_allowed("equity", windows, hold_days=0) is True


# ---------------------------------------------------------------------------
# get_dte_bounds / get_hold_days_bounds
# ---------------------------------------------------------------------------


class TestBounds:
    def test_dte_bounds_single_window(self):
        windows = expand_windows({TradingWindow.OPTIONS_WEEKLY})
        assert get_dte_bounds(windows) == (1, 7)

    def test_dte_bounds_short_term_composite(self):
        windows = expand_windows({TradingWindow.OPTIONS_SHORT_TERM})
        assert get_dte_bounds(windows) == (0, 14)

    def test_dte_bounds_no_options(self):
        windows = expand_windows({TradingWindow.EQUITY_SWING})
        assert get_dte_bounds(windows) is None

    def test_hold_days_bounds_single(self):
        windows = expand_windows({TradingWindow.EQUITY_SWING})
        assert get_hold_days_bounds(windows) == (1, 10)

    def test_hold_days_bounds_no_equity(self):
        windows = expand_windows({TradingWindow.OPTIONS_WEEKLY})
        assert get_hold_days_bounds(windows) is None

    def test_hold_days_bounds_composite(self):
        windows = expand_windows({TradingWindow.EQUITY_SHORT_TERM})
        assert get_hold_days_bounds(windows) == (0, 10)


# ---------------------------------------------------------------------------
# allowed_holding_types / allowed_instrument_types
# ---------------------------------------------------------------------------


class TestAllowedTypes:
    def test_options_weekly_holding_types(self):
        windows = expand_windows({TradingWindow.OPTIONS_WEEKLY})
        assert allowed_holding_types(windows) == {
            HoldingType.INTRADAY,
            HoldingType.SHORT_SWING,
        }

    def test_all_window_all_holding_types(self):
        windows = expand_windows({TradingWindow.ALL})
        assert allowed_holding_types(windows) == set(HoldingType)

    def test_options_only_instrument_types(self):
        windows = expand_windows({TradingWindow.OPTIONS_SHORT_TERM})
        assert allowed_instrument_types(windows) == {InstrumentType.OPTIONS}

    def test_equity_only_instrument_types(self):
        windows = expand_windows({TradingWindow.EQUITY_SWING})
        assert allowed_instrument_types(windows) == {InstrumentType.EQUITY}

    def test_mixed_instrument_types(self):
        windows = expand_windows({TradingWindow.OPTIONS_WEEKLY, TradingWindow.EQUITY_SWING})
        assert allowed_instrument_types(windows) == {
            InstrumentType.EQUITY,
            InstrumentType.OPTIONS,
        }

    def test_equity_scalp_only_intraday(self):
        windows = expand_windows({TradingWindow.EQUITY_SCALP})
        assert allowed_holding_types(windows) == {HoldingType.INTRADAY}


# ---------------------------------------------------------------------------
# TradingWindowSettings
# ---------------------------------------------------------------------------


class TestTradingWindowSettings:
    def test_new_env_vars(self, monkeypatch):
        monkeypatch.delenv("TRADING_HOLDING_PERIODS", raising=False)
        monkeypatch.delenv("RESEARCH_HOLDING_PERIODS", raising=False)
        monkeypatch.setenv("TRADING_WINDOW", "options_weekly")
        monkeypatch.setenv("RESEARCH_WINDOW", "all")

        from quantstack.config.settings import TradingWindowSettings

        settings = TradingWindowSettings()
        assert TradingWindow.OPTIONS_WEEKLY in settings.trading_windows
        assert len(settings.research_windows) == 14  # all leaves

    def test_default_is_all(self, monkeypatch):
        monkeypatch.delenv("TRADING_HOLDING_PERIODS", raising=False)
        monkeypatch.delenv("RESEARCH_HOLDING_PERIODS", raising=False)
        monkeypatch.delenv("TRADING_WINDOW", raising=False)
        monkeypatch.delenv("RESEARCH_WINDOW", raising=False)

        from quantstack.config.settings import TradingWindowSettings

        settings = TradingWindowSettings()
        assert len(settings.trading_windows) == 14
        assert len(settings.research_windows) == 14

    def test_legacy_bridge(self, monkeypatch):
        """Old TRADING_HOLDING_PERIODS bridges to windows when new var is unset."""
        monkeypatch.delenv("TRADING_WINDOW", raising=False)
        monkeypatch.delenv("RESEARCH_WINDOW", raising=False)
        monkeypatch.setenv("TRADING_HOLDING_PERIODS", "intraday")

        from quantstack.config.settings import TradingWindowSettings

        settings = TradingWindowSettings()
        windows = settings.trading_windows
        # Bridge maps intraday → EQUITY_SCALP, EQUITY_DAY_TRADE, OPTIONS_0DTE
        assert TradingWindow.EQUITY_SCALP in windows
        assert TradingWindow.EQUITY_DAY_TRADE in windows
        assert TradingWindow.OPTIONS_0DTE in windows

    def test_new_var_overrides_legacy(self, monkeypatch):
        """TRADING_WINDOW takes precedence over TRADING_HOLDING_PERIODS."""
        monkeypatch.setenv("TRADING_WINDOW", "equity_swing")
        monkeypatch.setenv("TRADING_HOLDING_PERIODS", "intraday")

        from quantstack.config.settings import TradingWindowSettings

        settings = TradingWindowSettings()
        # New var wins — only equity_swing
        assert settings.trading_windows == {TradingWindow.EQUITY_SWING}

    def test_caching(self, monkeypatch):
        monkeypatch.delenv("TRADING_WINDOW", raising=False)
        monkeypatch.delenv("RESEARCH_WINDOW", raising=False)
        monkeypatch.delenv("TRADING_HOLDING_PERIODS", raising=False)
        monkeypatch.delenv("RESEARCH_HOLDING_PERIODS", raising=False)

        from quantstack.config.settings import get_trading_window_settings

        get_trading_window_settings.cache_clear()
        a = get_trading_window_settings()
        b = get_trading_window_settings()
        assert a is b
