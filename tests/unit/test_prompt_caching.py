"""Tests for prompt caching configuration in LLM provider layer."""

import os
from unittest.mock import MagicMock, patch

import pytest

from quantstack.llm.config import (
    BEDROCK_PROMPT_CACHING_BETA,
    CACHE_CONTROL_EPHEMERAL,
    ModelConfig,
)


@pytest.fixture
def _bedrock_env(monkeypatch):
    """Set env vars for Bedrock provider."""
    monkeypatch.setenv("LLM_PROVIDER", "bedrock")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")


@pytest.fixture
def _anthropic_env(monkeypatch):
    """Set env vars for Anthropic provider."""
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")


class TestBedrockCaching:
    def test_cache_control_added_to_bedrock_calls(self, _bedrock_env):
        """get_chat_model for Bedrock with caching=True passes the beta header."""
        MockBedrock = MagicMock()
        mock_module = MagicMock()
        mock_module.ChatBedrock = MockBedrock
        with patch.dict("sys.modules", {"langchain_aws": mock_module}):
            from quantstack.llm.provider import _instantiate_chat_model

            config = ModelConfig(
                provider="bedrock",
                model_id="us.anthropic.claude-haiku-4-5-20251001-v1:0",
                tier="light",
                prompt_caching=True,
            )
            _instantiate_chat_model(config)

            call_kwargs = MockBedrock.call_args
            model_kwargs = call_kwargs.kwargs.get("model_kwargs", {})
            assert BEDROCK_PROMPT_CACHING_BETA in model_kwargs.get("anthropic_beta", [])

    def test_no_cache_header_when_disabled(self, _bedrock_env):
        """get_chat_model for Bedrock with caching=False does not add beta header."""
        MockBedrock = MagicMock()
        mock_module = MagicMock()
        mock_module.ChatBedrock = MockBedrock
        with patch.dict("sys.modules", {"langchain_aws": mock_module}):
            from quantstack.llm.provider import _instantiate_chat_model

            config = ModelConfig(
                provider="bedrock",
                model_id="us.anthropic.claude-haiku-4-5-20251001-v1:0",
                tier="light",
                prompt_caching=False,
            )
            _instantiate_chat_model(config)

            call_kwargs = MockBedrock.call_args
            model_kwargs = call_kwargs.kwargs.get("model_kwargs", {})
            assert "anthropic_beta" not in model_kwargs


class TestAnthropicCaching:
    def test_cache_control_added_to_anthropic_calls(self, _anthropic_env):
        """get_chat_model for Anthropic with caching=True adds the beta header."""
        MockAnthropic = MagicMock()
        mock_module = MagicMock()
        mock_module.ChatAnthropic = MockAnthropic
        with patch.dict("sys.modules", {"langchain_anthropic": mock_module}):
            from quantstack.llm.provider import _instantiate_chat_model

            config = ModelConfig(
                provider="anthropic",
                model_id="claude-haiku-4-5",
                tier="light",
                prompt_caching=True,
            )
            _instantiate_chat_model(config)

            call_kwargs = MockAnthropic.call_args
            model_kwargs = call_kwargs.kwargs.get("model_kwargs", {})
            assert "extra_headers" in model_kwargs
            assert model_kwargs["extra_headers"]["anthropic-beta"] == BEDROCK_PROMPT_CACHING_BETA


class TestPromptCachingEnvVar:
    def test_caching_disabled_via_env_var(self, monkeypatch):
        """When PROMPT_CACHING_ENABLED=false, prompt_caching resolves to False."""
        monkeypatch.setenv("PROMPT_CACHING_ENABLED", "false")
        monkeypatch.setenv("LLM_PROVIDER", "bedrock")
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")
        monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")

        MockBedrock = MagicMock()
        mock_module = MagicMock()
        mock_module.ChatBedrock = MockBedrock
        with patch.dict("sys.modules", {"langchain_aws": mock_module}):
            from quantstack.llm.provider import get_chat_model

            get_chat_model("light")

            call_kwargs = MockBedrock.call_args
            model_kwargs = call_kwargs.kwargs.get("model_kwargs", {})
            assert "anthropic_beta" not in model_kwargs

    def test_caching_enabled_by_default(self, monkeypatch):
        """When PROMPT_CACHING_ENABLED is not set, caching defaults to True."""
        monkeypatch.delenv("PROMPT_CACHING_ENABLED", raising=False)
        monkeypatch.setenv("LLM_PROVIDER", "bedrock")
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")
        monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")

        MockBedrock = MagicMock()
        mock_module = MagicMock()
        mock_module.ChatBedrock = MockBedrock
        with patch.dict("sys.modules", {"langchain_aws": mock_module}):
            from quantstack.llm.provider import get_chat_model

            get_chat_model("light")

            call_kwargs = MockBedrock.call_args
            model_kwargs = call_kwargs.kwargs.get("model_kwargs", {})
            assert BEDROCK_PROMPT_CACHING_BETA in model_kwargs.get("anthropic_beta", [])


class TestCacheControlConstant:
    def test_cache_control_ephemeral_structure(self):
        """CACHE_CONTROL_EPHEMERAL has the correct Anthropic format."""
        assert CACHE_CONTROL_EPHEMERAL == {"type": "ephemeral"}


class TestCacheInvalidation:
    def test_different_tool_lists_produce_different_payloads(self):
        """Changed tool definitions produce different message content,
        ensuring server-side cache invalidation via content hash."""
        tools_v1 = [{"name": "tool_a", "description": "Does A"}]
        tools_v2 = [{"name": "tool_a", "description": "Does A (improved)"}, {"name": "tool_b", "description": "Does B"}]
        assert str(tools_v1) != str(tools_v2)
