# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Tests for the decode_strategy and decode_from_trades MCP tools.
End-to-end from raw signals to DecodedStrategy.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta

import pytest

from quant_pod.context import create_trading_context


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ctx():
    context = create_trading_context(db_path=":memory:", session_id=str(uuid.uuid4()))
    yield context
    context.db.close()


@pytest.fixture
def _inject_ctx(ctx):
    import quant_pod.mcp.server as srv
    original = srv._ctx
    srv._ctx = ctx
    yield ctx
    srv._ctx = original


def _fn(tool_obj):
    return tool_obj.fn if hasattr(tool_obj, "fn") else tool_obj


def _make_signals(n: int = 30) -> list:
    """Generate test trade signals."""
    signals = []
    base = datetime(2024, 3, 1, 10, 30)
    for i in range(n):
        entry = base + timedelta(days=i)
        exit_time = entry + timedelta(hours=3)
        entry_price = 450 + i * 0.2
        exit_price = entry_price * (1.015 if i % 3 != 0 else 0.99)
        signals.append({
            "symbol": "SPY",
            "direction": "long",
            "entry_time": entry.strftime("%Y-%m-%d %H:%M"),
            "entry_price": entry_price,
            "exit_time": exit_time.strftime("%Y-%m-%d %H:%M"),
            "exit_price": exit_price,
        })
    return signals


# ---------------------------------------------------------------------------
# decode_strategy MCP tool
# ---------------------------------------------------------------------------


class TestDecodeStrategyTool:
    @pytest.mark.asyncio
    async def test_decode_returns_decoded_strategy(self, _inject_ctx):
        from quant_pod.mcp.server import decode_strategy

        result = await _fn(decode_strategy)(
            signals=_make_signals(),
            source_name="test_discord",
        )
        assert result["success"] is True
        decoded = result["decoded_strategy"]
        assert decoded["source_trader"] == "test_discord"
        assert decoded["sample_size"] == 30
        assert decoded["win_rate"] > 0
        assert "edge_hypothesis" in decoded

    @pytest.mark.asyncio
    async def test_decode_with_auto_register(self, _inject_ctx):
        from quant_pod.mcp.server import decode_strategy, get_strategy

        result = await _fn(decode_strategy)(
            signals=_make_signals(),
            source_name="test_auto_reg",
            strategy_name="auto_decoded_test",
        )
        assert result["success"] is True
        assert "registered" in result
        assert result["registered"]["success"] is True

        # Verify strategy was registered
        strat = await _fn(get_strategy)(name="auto_decoded_test")
        assert strat["success"] is True
        assert strat["strategy"]["source"] == "decoded"

    @pytest.mark.asyncio
    async def test_decode_empty_signals_fails(self, _inject_ctx):
        from quant_pod.mcp.server import decode_strategy

        result = await _fn(decode_strategy)(signals=[])
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_decode_low_confidence_warning(self, _inject_ctx):
        from quant_pod.mcp.server import decode_strategy

        result = await _fn(decode_strategy)(
            signals=_make_signals(10),
            source_name="small_sample",
        )
        assert result["success"] is True
        assert result["low_confidence_warning"] is True


# ---------------------------------------------------------------------------
# decode_from_trades MCP tool
# ---------------------------------------------------------------------------


class TestDecodeFromTrades:
    @pytest.mark.asyncio
    async def test_no_trades_returns_error(self, _inject_ctx):
        from quant_pod.mcp.server import decode_from_trades

        result = await _fn(decode_from_trades)(source="closed_trades")
        assert result["success"] is False
        assert "No trades found" in result["error"]

    @pytest.mark.asyncio
    async def test_with_closed_trades(self, _inject_ctx, ctx):
        from quant_pod.mcp.server import decode_from_trades

        # Insert synthetic closed trades
        for i in range(25):
            entry_time = datetime(2024, 1, 2 + i, 10, 30)
            exit_time = entry_time + timedelta(hours=4)
            entry_price = 100 + i * 0.1
            exit_price = entry_price * (1.02 if i % 3 != 0 else 0.99)
            realized = (exit_price - entry_price) * 10

            ctx.db.execute(
                """
                INSERT INTO closed_trades
                    (id, symbol, side, quantity, entry_price, exit_price,
                     realized_pnl, opened_at, closed_at, holding_days)
                VALUES (?, ?, 'long', 10, ?, ?, ?, ?, ?, 0)
                """,
                [i + 1, "SPY", entry_price, exit_price, realized, entry_time, exit_time],
            )

        result = await _fn(decode_from_trades)(
            source="closed_trades",
            symbol="SPY",
            source_name="self_analysis",
        )
        assert result["success"] is True
        assert result["decoded_strategy"]["sample_size"] == 25

    @pytest.mark.asyncio
    async def test_invalid_source_fails(self, _inject_ctx):
        from quant_pod.mcp.server import decode_from_trades

        result = await _fn(decode_from_trades)(source="bad_source")
        assert result["success"] is False
        assert "Unknown source" in result["error"]
