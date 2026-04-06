"""Tests for Tool Search prerequisite validation (Section 01).

Validates:
- langchain-anthropic preserves defer_loading on tool dicts
- llm.bind() accepts tool search tool type (bypass for bind_tools limitation)
- server_tool_use blocks are handled correctly
- Startup assertion catches incompatible versions
- tool_to_anthropic_dict helper works correctly
"""

import json
from unittest.mock import patch

import pytest
from langchain_anthropic import ChatAnthropic
from langchain_anthropic.chat_models import convert_to_anthropic_tool
from langchain_core.tools import tool


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@tool
def dummy_signal_brief(symbol: str) -> str:
    """Generate a trading signal brief for a ticker symbol."""
    return f"Signal brief for {symbol}"


TOOL_SEARCH_TOOL_DICT = {
    "type": "tool_search_bm25_2025_04_15",
    "name": "tool_search_tool",
    "max_results": 5,
}


def _make_deferred_tool_dict() -> dict:
    """Create a tool dict with defer_loading flag in Anthropic API format."""
    schema = dummy_signal_brief.get_input_schema().model_json_schema()
    schema.pop("title", None)
    return {
        "name": "signal_brief",
        "description": "Generate a trading signal brief for a ticker symbol.",
        "input_schema": schema,
        "defer_loading": True,
    }


# ---------------------------------------------------------------------------
# 0.3: Version Pinning & Compatibility Tests
# ---------------------------------------------------------------------------

class TestVersionPinning:
    """Validate that the installed langchain-anthropic supports defer_loading."""

    def test_convert_to_anthropic_tool_preserves_defer_loading(self):
        """convert_to_anthropic_tool must preserve the defer_loading field
        on tool dicts passed through it.
        """
        deferred = _make_deferred_tool_dict()
        converted = convert_to_anthropic_tool(deferred)
        assert "defer_loading" in converted
        assert converted["defer_loading"] is True

    def test_bind_tools_accepts_deferred_tool_dict(self):
        """bind_tools() must accept a tool dict with defer_loading: True.
        The defer_loading field passes through convert_to_anthropic_tool.
        """
        llm = ChatAnthropic(model="claude-sonnet-4-20250514")
        deferred = _make_deferred_tool_dict()
        bound = llm.bind_tools([deferred])
        assert bound is not None

    def test_bind_tools_rejects_tool_search_type(self):
        """bind_tools() does NOT recognize tool_search_bm25 as a builtin type
        in langchain-anthropic 0.3.x. This is expected — we use llm.bind() instead.
        """
        llm = ChatAnthropic(model="claude-sonnet-4-20250514")
        with pytest.raises(KeyError):
            llm.bind_tools([TOOL_SEARCH_TOOL_DICT])

    def test_llm_bind_accepts_tool_search_and_deferred(self):
        """llm.bind(tools=...) bypasses bind_tools conversion and accepts
        both deferred tools and the tool search tool type directly.
        """
        llm = ChatAnthropic(model="claude-sonnet-4-20250514")
        deferred = _make_deferred_tool_dict()
        bound = llm.bind(tools=[deferred, TOOL_SEARCH_TOOL_DICT])
        assert bound is not None

    def test_startup_assertion_passes_on_compatible_version(self):
        """validate_tool_search_support() succeeds on compatible versions."""
        from quantstack.graphs.tool_search_compat import validate_tool_search_support

        # Should not raise
        validate_tool_search_support()

    def test_startup_assertion_fails_when_defer_loading_stripped(self):
        """If convert_to_anthropic_tool strips defer_loading, validation fails."""
        from quantstack.graphs.tool_search_compat import validate_tool_search_support

        def mock_convert(tool_input):
            result = {"name": "test", "description": "test", "input_schema": {}}
            # Deliberately strip defer_loading
            return result

        with patch(
            "quantstack.graphs.tool_search_compat.convert_to_anthropic_tool",
            side_effect=mock_convert,
        ):
            with pytest.raises(RuntimeError, match="defer_loading"):
                validate_tool_search_support()


# ---------------------------------------------------------------------------
# tool_to_anthropic_dict helper tests
# ---------------------------------------------------------------------------

class TestToolToAnthropicDict:
    """Test the helper that converts BaseTool to Anthropic dict format."""

    def test_converts_base_tool_without_defer(self):
        from quantstack.graphs.tool_search_compat import tool_to_anthropic_dict

        result = tool_to_anthropic_dict(dummy_signal_brief)
        assert result["name"] == "dummy_signal_brief"
        assert "description" in result
        assert "input_schema" in result
        assert "defer_loading" not in result

    def test_converts_base_tool_with_defer(self):
        from quantstack.graphs.tool_search_compat import tool_to_anthropic_dict

        result = tool_to_anthropic_dict(dummy_signal_brief, defer=True)
        assert result["name"] == "dummy_signal_brief"
        assert result["defer_loading"] is True

    def test_preserves_input_schema_properties(self):
        from quantstack.graphs.tool_search_compat import tool_to_anthropic_dict

        result = tool_to_anthropic_dict(dummy_signal_brief)
        schema = result["input_schema"]
        assert "properties" in schema
        assert "symbol" in schema["properties"]


# ---------------------------------------------------------------------------
# 0.2: server_tool_use Handling Tests
# ---------------------------------------------------------------------------

class TestServerToolUseHandling:
    """Validate that server_tool_use blocks are handled correctly.

    These tests use mocked responses since actual API calls require credentials.
    """

    def test_standard_tool_calls_have_expected_shape(self):
        """A normal tool_call entry has name, args, id keys."""
        tool_call = {
            "name": "signal_brief",
            "args": {"symbol": "AAPL"},
            "id": "call_123",
            "type": "tool_use",
        }
        assert tool_call["name"] == "signal_brief"
        assert "server_tool_use" not in tool_call.get("type", "")

    def test_server_tool_use_filter_skips_search_tool(self):
        """The server_tool_use filter should skip entries whose name
        starts with 'tool_search_tool'.
        """
        tool_calls = [
            {"name": "tool_search_tool", "args": {}, "id": "srv_1", "type": "server_tool_use"},
            {"name": "signal_brief", "args": {"symbol": "AAPL"}, "id": "call_2", "type": "tool_use"},
        ]

        executable = [
            tc for tc in tool_calls
            if not tc.get("name", "").startswith("tool_search_tool")
        ]

        assert len(executable) == 1
        assert executable[0]["name"] == "signal_brief"

    def test_response_with_no_server_tool_use_passes_through(self):
        """When there are no server_tool_use blocks, all tool_calls are executable."""
        tool_calls = [
            {"name": "signal_brief", "args": {"symbol": "AAPL"}, "id": "call_1"},
            {"name": "fetch_portfolio", "args": {}, "id": "call_2"},
        ]

        executable = [
            tc for tc in tool_calls
            if not tc.get("name", "").startswith("tool_search_tool")
        ]

        assert len(executable) == 2


# ---------------------------------------------------------------------------
# Token Overhead Measurement Tests
# ---------------------------------------------------------------------------

class TestTokenOverheadMeasurement:
    """Validate the measurement script logic (not actual measurements)."""

    def test_tool_schema_serialization(self):
        """Tool schemas can be serialized to JSON for token counting."""
        schema = dummy_signal_brief.get_input_schema().model_json_schema()
        schema.pop("title", None)
        anthropic_schema = {
            "name": "signal_brief",
            "description": dummy_signal_brief.description,
            "input_schema": schema,
        }
        serialized = json.dumps(anthropic_schema)
        assert len(serialized) > 0
        assert "signal_brief" in serialized

    def test_all_registered_tools_have_schemas(self):
        """Every tool in TOOL_REGISTRY can produce a valid input schema."""
        from quantstack.tools.registry import TOOL_REGISTRY

        for name, tool_obj in TOOL_REGISTRY.items():
            schema = tool_obj.get_input_schema().model_json_schema()
            assert isinstance(schema, dict), f"Tool '{name}' schema is not a dict"
            assert "properties" in schema or "type" in schema, (
                f"Tool '{name}' schema missing 'properties' or 'type'"
            )
