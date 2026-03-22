"""Tests for shared.mcp_toolkit."""

import asyncio

import pytest

from quantstack.shared.mcp_toolkit import (
    mcp_tool_response,
    mcp_tool_safe,
    require_resource,
)


class TestMcpToolResponse:
    def test_success_minimal(self):
        r = mcp_tool_response(True)
        assert r == {"success": True}

    def test_success_with_data(self):
        r = mcp_tool_response(True, rows=42, symbol="SPY")
        assert r["success"] is True
        assert r["rows"] == 42
        assert r["symbol"] == "SPY"

    def test_failure_with_error(self):
        r = mcp_tool_response(False, error="boom")
        assert r == {"success": False, "error": "boom"}

    def test_failure_with_extra_data(self):
        r = mcp_tool_response(False, error="timeout", symbol="AAPL")
        assert r["success"] is False
        assert r["error"] == "timeout"
        assert r["symbol"] == "AAPL"


class TestMcpToolSafe:
    def test_sync_success(self):
        @mcp_tool_safe
        def good():
            return mcp_tool_response(True, value=1)

        assert good() == {"success": True, "value": 1}

    def test_sync_exception_caught(self):
        @mcp_tool_safe
        def bad():
            raise ValueError("oops")

        r = bad()
        assert r["success"] is False
        assert "oops" in r["error"]

    def test_async_success(self):
        @mcp_tool_safe
        async def good():
            return mcp_tool_response(True, value=2)

        r = asyncio.get_event_loop().run_until_complete(good())
        assert r == {"success": True, "value": 2}

    def test_async_exception_caught(self):
        @mcp_tool_safe
        async def bad():
            raise RuntimeError("async boom")

        r = asyncio.get_event_loop().run_until_complete(bad())
        assert r["success"] is False
        assert "async boom" in r["error"]

    def test_preserves_function_name(self):
        @mcp_tool_safe
        def my_tool():
            return mcp_tool_response(True)

        assert my_tool.__name__ == "my_tool"

    def test_preserves_async_function_name(self):
        @mcp_tool_safe
        async def my_async_tool():
            return mcp_tool_response(True)

        assert my_async_tool.__name__ == "my_async_tool"


class TestRequireResource:
    def test_returns_resource_when_present(self):
        obj = {"key": "value"}
        assert require_resource(obj, "test") is obj

    def test_raises_when_none(self):
        with pytest.raises(RuntimeError, match="Broker not initialized"):
            require_resource(None, "Broker")

    def test_allows_falsy_but_not_none(self):
        assert require_resource(0, "counter") == 0
        assert require_resource("", "name") == ""
        assert require_resource([], "list") == []
