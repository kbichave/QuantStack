# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Tests for Phase 3 SignalEngine-backed MCP tools:
  - get_signal_brief — single-symbol analysis via mocked SignalEngine
  - run_multi_signal_brief — multi-symbol parallel analysis

SignalEngine and its DataStore dependencies are fully mocked. These tests
verify tool-level logic: context guards, cache population, error handling,
and response shaping — not the engine internals.
"""

from __future__ import annotations

from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from quantstack.mcp.tools.signal import get_signal_brief, run_multi_signal_brief
import quantstack.mcp._state as _mcp_state
from tests.quant_pod.mcp.conftest import _fn, assert_standard_response, assert_error_response


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


def _make_symbol_brief(symbol: str = "SPY") -> MagicMock:
    """Build a mock SymbolBrief with the fields accessed by _populate_signal_cache."""
    sb = MagicMock()
    sb.symbol = symbol
    sb.market_summary = f"{symbol} is trending up with strong momentum"
    sb.consensus_bias = "bullish"
    sb.consensus_conviction = 0.75
    sb.key_observations = ["RSI at 55", "Above 20-day MA", "Volume expanding"]
    return sb


def _make_signal_brief(
    symbol: str = "SPY",
    confidence: float = 0.7,
    failures: list[str] | None = None,
) -> MagicMock:
    """Build a mock SignalBrief that satisfies the tool's attribute access."""
    brief = MagicMock()
    brief.strategic_notes = "Initial strategic notes"
    brief.regime_detail = {"regime": "trending_up", "confidence": 0.8}
    brief.engine_version = "signal_engine_v1"
    brief.collector_failures = failures or []
    brief.overall_confidence = confidence

    sb = _make_symbol_brief(symbol)
    brief.symbol_briefs = [sb]

    # model_dump returns a dict that _serialize can process
    brief.model_dump.return_value = {
        "date": str(date.today()),
        "market_overview": f"{symbol} analysis complete",
        "market_bias": "bullish",
        "market_conviction": 0.7,
        "risk_environment": "normal",
        "symbol_briefs": [{"symbol": symbol, "market_summary": sb.market_summary}],
        "overall_confidence": confidence,
        "engine_version": "signal_engine_v1",
        "collector_failures": failures or [],
        "strategic_notes": "Initial strategic notes",
    }
    return brief


def _mock_engine_cls(briefs_by_symbol: dict[str, MagicMock] | None = None):
    """Create a mock SignalEngine class whose run/run_multi return pre-built briefs."""
    default_brief = _make_signal_brief()
    briefs = briefs_by_symbol or {}

    engine_instance = AsyncMock()
    engine_instance.run = AsyncMock(
        side_effect=lambda sym, regime=None: briefs.get(sym.upper(), default_brief)
    )
    engine_instance.run_multi = AsyncMock(
        side_effect=lambda syms, max_concurrent=5: [
            briefs.get(s.upper(), default_brief) for s in syms
        ]
    )

    engine_cls = MagicMock(return_value=engine_instance)
    return engine_cls


# ---------------------------------------------------------------------------
# get_signal_brief
# ---------------------------------------------------------------------------


class TestGetSignalBrief:
    @pytest.mark.asyncio
    async def test_happy_path(self, inject_ctx):
        """Mocked SignalEngine returns a valid brief; tool shapes it correctly."""
        mock_cls = _mock_engine_cls()
        with patch("quantstack.mcp.tools.signal.SignalEngine", mock_cls, create=True):
            # Patch the deferred import by patching the module-level import site
            with patch.dict(
                "sys.modules",
                {"quantstack.signal_engine": MagicMock(SignalEngine=mock_cls)},
            ):
                result = await _fn(get_signal_brief)(symbol="SPY")

        assert_standard_response(result)
        assert result["success"] is True
        assert "daily_brief" in result
        assert result["engine"] == "signal_engine_v1"
        assert isinstance(result["elapsed_seconds"], float)
        assert isinstance(result["collector_failures"], list)

    @pytest.mark.asyncio
    async def test_missing_context_returns_error(self):
        """Without initialized MCP state, tool should return error, not raise."""
        original = _mcp_state._ctx
        _mcp_state._ctx = None
        try:
            result = await _fn(get_signal_brief)(symbol="SPY")
            assert_error_response(result)
            assert "not initialized" in result["error"].lower()
            assert "elapsed_seconds" in result
        finally:
            _mcp_state._ctx = original

    @pytest.mark.asyncio
    async def test_engine_exception_returns_error(self, inject_ctx):
        """If SignalEngine.run raises, tool catches and returns error dict."""
        engine_instance = AsyncMock()
        engine_instance.run = AsyncMock(
            side_effect=RuntimeError("DataStore connection failed")
        )
        mock_cls = MagicMock(return_value=engine_instance)

        with patch.dict(
            "sys.modules",
            {"quantstack.signal_engine": MagicMock(SignalEngine=mock_cls)},
        ):
            result = await _fn(get_signal_brief)(symbol="SPY")

        assert_error_response(result)
        assert "DataStore connection failed" in result["error"]
        assert isinstance(result["elapsed_seconds"], float)

    @pytest.mark.asyncio
    async def test_symbol_uppercased_and_stripped(self, inject_ctx):
        """Tool should normalize ' spy ' → 'SPY' before passing to engine."""
        mock_cls = _mock_engine_cls()
        with patch.dict(
            "sys.modules",
            {"quantstack.signal_engine": MagicMock(SignalEngine=mock_cls)},
        ):
            result = await _fn(get_signal_brief)(symbol="  spy  ")

        assert result["success"] is True
        # The engine's run was called with uppercase stripped symbol
        engine_instance = mock_cls.return_value
        engine_instance.run.assert_called_once()
        call_args = engine_instance.run.call_args
        assert call_args[0][0] == "SPY" or call_args[1].get("symbol") == "SPY"

    @pytest.mark.asyncio
    async def test_strategy_context_injection(self, inject_ctx):
        """When include_strategy_context=True and session file exists, notes are appended."""
        brief = _make_signal_brief("AAPL")
        mock_cls = _mock_engine_cls({"AAPL": brief})

        with patch.dict(
            "sys.modules",
            {"quantstack.signal_engine": MagicMock(SignalEngine=mock_cls)},
        ):
            with patch(
                "quantstack.mcp.tools.signal._read_memory_file",
                return_value="Session context: bearish regime detected",
            ):
                result = await _fn(get_signal_brief)(
                    symbol="AAPL", include_strategy_context=True
                )

        assert result["success"] is True
        # Verify that strategic_notes was mutated on the brief
        assert "Session context" in brief.strategic_notes


# ---------------------------------------------------------------------------
# run_multi_signal_brief
# ---------------------------------------------------------------------------


class TestRunMultiSignalBrief:
    @pytest.mark.asyncio
    async def test_multi_happy_path(self, inject_ctx):
        """Two symbols should both appear in results with success=True."""
        spy_brief = _make_signal_brief("SPY", confidence=0.8)
        xom_brief = _make_signal_brief("XOM", confidence=0.6)
        mock_cls = _mock_engine_cls({"SPY": spy_brief, "XOM": xom_brief})

        with patch.dict(
            "sys.modules",
            {"quantstack.signal_engine": MagicMock(SignalEngine=mock_cls)},
        ):
            result = await _fn(run_multi_signal_brief)(symbols=["SPY", "XOM"])

        assert "results" in result
        assert "SPY" in result["results"]
        assert "XOM" in result["results"]
        assert result["results"]["SPY"]["success"] is True
        assert result["results"]["XOM"]["success"] is True
        assert set(result["symbols_succeeded"]) == {"SPY", "XOM"}
        assert result["symbols_failed"] == []
        assert isinstance(result["elapsed_seconds"], float)

    @pytest.mark.asyncio
    async def test_empty_symbols_list(self, inject_ctx):
        """Empty list should return immediately with no results."""
        result = await _fn(run_multi_signal_brief)(symbols=[])
        assert result["results"] == {}
        assert result["symbols_succeeded"] == []
        assert result["symbols_failed"] == []
        assert result["elapsed_seconds"] == 0.0

    @pytest.mark.asyncio
    async def test_missing_context_returns_error(self):
        original = _mcp_state._ctx
        _mcp_state._ctx = None
        try:
            result = await _fn(run_multi_signal_brief)(symbols=["SPY"])
            assert_error_response(result)
        finally:
            _mcp_state._ctx = original

    @pytest.mark.asyncio
    async def test_all_collectors_failed_marks_symbol_failed(self, inject_ctx):
        """A brief with confidence=0.0 and 'all' in failures should be marked failed."""
        bad_brief = _make_signal_brief("BAD", confidence=0.0, failures=["all"])
        mock_cls = _mock_engine_cls({"BAD": bad_brief})

        with patch.dict(
            "sys.modules",
            {"quantstack.signal_engine": MagicMock(SignalEngine=mock_cls)},
        ):
            result = await _fn(run_multi_signal_brief)(symbols=["BAD"])

        assert "BAD" in result["results"]
        assert result["results"]["BAD"]["success"] is False
        assert result["symbols_failed"] == ["BAD"]
        assert result["symbols_succeeded"] == []

    @pytest.mark.asyncio
    async def test_engine_exception_returns_error(self, inject_ctx):
        """If SignalEngine() constructor or run_multi raises, tool catches it."""
        mock_cls = MagicMock(
            side_effect=ImportError("signal_engine not installed")
        )

        with patch.dict(
            "sys.modules",
            {"quantstack.signal_engine": MagicMock(SignalEngine=mock_cls)},
        ):
            result = await _fn(run_multi_signal_brief)(symbols=["SPY"])

        assert_error_response(result)
        assert isinstance(result["elapsed_seconds"], float)
