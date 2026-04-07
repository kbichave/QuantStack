"""Tests for options refresh expansion (section-13)."""

import os
import pytest
from unittest.mock import MagicMock, patch

import pandas as pd


class TestConfigurableTopN:
    """OPTIONS_REFRESH_TOP_N controls number of symbols refreshed."""

    def test_default_is_30(self):
        """Without env var, default is 30."""
        # Re-import to get fresh module state
        with patch.dict(os.environ, {}, clear=False):
            # Remove if set
            os.environ.pop("OPTIONS_REFRESH_TOP_N", None)
            import importlib
            import quantstack.data.scheduled_refresh as mod

            importlib.reload(mod)
            assert mod.OPTIONS_REFRESH_TOP_N == 30

    def test_custom_value_from_env(self):
        """Setting OPTIONS_REFRESH_TOP_N=50 is respected."""
        with patch.dict(os.environ, {"OPTIONS_REFRESH_TOP_N": "50"}):
            import importlib
            import quantstack.data.scheduled_refresh as mod

            importlib.reload(mod)
            assert mod.OPTIONS_REFRESH_TOP_N == 50

        # Cleanup: reload with default
        os.environ.pop("OPTIONS_REFRESH_TOP_N", None)
        import importlib
        import quantstack.data.scheduled_refresh as mod

        importlib.reload(mod)


class TestStrategyAwareSymbols:
    """Strategy-aware symbols are included in the options refresh list."""

    def test_get_options_strategy_symbols_returns_matching(self):
        """Symbols from active options strategies are returned."""
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchall.return_value = [
            ("AAPL",), ("TSLA",)
        ]
        cm = MagicMock()
        cm.__enter__ = MagicMock(return_value=mock_conn)
        cm.__exit__ = MagicMock(return_value=False)

        with patch("quantstack.data.scheduled_refresh.pg_conn", return_value=cm):
            from quantstack.data.scheduled_refresh import _get_options_strategy_symbols

            result = _get_options_strategy_symbols()

        assert "AAPL" in result
        assert "TSLA" in result

    def test_returns_empty_on_error(self):
        """On DB error, returns empty list (no crash)."""
        with patch(
            "quantstack.data.scheduled_refresh.pg_conn",
            side_effect=Exception("DB down"),
        ):
            from quantstack.data.scheduled_refresh import _get_options_strategy_symbols

            result = _get_options_strategy_symbols()

        assert result == []
