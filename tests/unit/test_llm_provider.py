"""Tests for Section 02: LLM Provider Management.

Validates model tier resolution, provider selection,
fallback chain, and env var validation.
"""

import os

import pytest


class TestProviderConfig:
    """Tests for config.py tier map."""

    def test_get_model_heavy_bedrock(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "bedrock")
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")
        monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
        from quantstack.llm.provider import get_model
        result = get_model("heavy")
        assert "bedrock" in result
        assert "claude" in result.lower() or "anthropic" in result.lower()

    def test_get_model_heavy_anthropic(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        from quantstack.llm.provider import get_model
        result = get_model("heavy")
        assert "anthropic" in result
        assert "sonnet" in result.lower() or "claude" in result.lower()

    def test_get_model_heavy_openai(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        from quantstack.llm.provider import get_model
        result = get_model("heavy")
        assert "openai" in result

    def test_get_model_heavy_gemini(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "gemini")
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        from quantstack.llm.provider import get_model
        result = get_model("heavy")
        assert "gemini" in result

    def test_get_model_heavy_ollama(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "ollama")
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")
        from quantstack.llm.provider import get_model
        result = get_model("heavy")
        assert "ollama" in result

    def test_get_model_medium(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "bedrock")
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")
        monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
        from quantstack.llm.provider import get_model
        result = get_model("medium")
        assert isinstance(result, str) and len(result) > 0

    def test_get_model_light(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "bedrock")
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")
        monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
        from quantstack.llm.provider import get_model
        result = get_model("light")
        assert isinstance(result, str) and len(result) > 0

    def test_get_model_embedding(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "bedrock")
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")
        monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
        from quantstack.llm.provider import get_model
        result = get_model("embedding")
        assert isinstance(result, str) and len(result) > 0

    def test_get_model_unknown_tier_raises(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "bedrock")
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")
        monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
        from quantstack.llm.provider import get_model
        with pytest.raises(ValueError, match="tier"):
            get_model("nonexistent")

    def test_default_provider_is_bedrock(self, monkeypatch):
        monkeypatch.delenv("LLM_PROVIDER", raising=False)
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")
        monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
        from quantstack.llm.provider import get_model
        result = get_model("heavy")
        assert "bedrock" in result


class TestFallbackChain:
    """Tests for provider fallback."""

    def test_fallback_activates_on_missing_credentials(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "bedrock")
        monkeypatch.setenv("LLM_FALLBACK_ENABLED", "true")
        monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
        monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        from quantstack.llm.provider import get_model_with_fallback
        result = get_model_with_fallback("heavy")
        assert "anthropic" in result

    def test_fallback_chain_order(self, monkeypatch):
        """When bedrock and anthropic fail, falls back to openai."""
        monkeypatch.setenv("LLM_PROVIDER", "bedrock")
        monkeypatch.setenv("LLM_FALLBACK_ENABLED", "true")
        monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
        monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        from quantstack.llm.provider import get_model_with_fallback
        result = get_model_with_fallback("heavy")
        assert "openai" in result

    def test_fallback_raises_when_all_fail(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "bedrock")
        monkeypatch.setenv("LLM_FALLBACK_ENABLED", "true")
        monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
        monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
        from quantstack.llm.provider import get_model_with_fallback, AllProvidersFailedError
        with pytest.raises(AllProvidersFailedError):
            get_model_with_fallback("heavy")

    def test_fallback_disabled_raises_config_error(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "bedrock")
        monkeypatch.setenv("LLM_FALLBACK_ENABLED", "false")
        monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
        monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
        from quantstack.llm.provider import get_model, ConfigurationError
        with pytest.raises(ConfigurationError):
            get_model("heavy")


class TestEnvVarValidation:
    """Tests for provider credential validation."""

    def test_bedrock_requires_aws_keys(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "bedrock")
        monkeypatch.setenv("LLM_FALLBACK_ENABLED", "false")
        monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
        from quantstack.llm.provider import get_model, ConfigurationError
        with pytest.raises(ConfigurationError, match="AWS_ACCESS_KEY_ID"):
            get_model("heavy")

    def test_anthropic_requires_api_key(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")
        monkeypatch.setenv("LLM_FALLBACK_ENABLED", "false")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        from quantstack.llm.provider import get_model, ConfigurationError
        with pytest.raises(ConfigurationError, match="ANTHROPIC_API_KEY"):
            get_model("heavy")

    def test_openai_requires_api_key(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.setenv("LLM_FALLBACK_ENABLED", "false")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        from quantstack.llm.provider import get_model, ConfigurationError
        with pytest.raises(ConfigurationError, match="OPENAI_API_KEY"):
            get_model("heavy")


class TestModelConfig:
    """Tests for ModelConfig dataclass."""

    def test_model_config_is_frozen(self):
        from quantstack.llm.config import ModelConfig
        cfg = ModelConfig(provider="anthropic", model_id="claude-sonnet-4", tier="heavy")
        with pytest.raises(AttributeError):
            cfg.provider = "openai"

    def test_model_config_defaults(self):
        from quantstack.llm.config import ModelConfig
        cfg = ModelConfig(provider="anthropic", model_id="claude-sonnet-4", tier="heavy")
        assert cfg.max_tokens == 4096
        assert cfg.temperature == 0.0


class TestGetChatModel:
    """Tests for get_chat_model() -> BaseChatModel."""

    def test_returns_base_chat_model_heavy(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        from langchain_core.language_models import BaseChatModel
        from quantstack.llm.provider import get_chat_model
        model = get_chat_model("heavy")
        assert isinstance(model, BaseChatModel)

    def test_returns_base_chat_model_medium(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        from langchain_core.language_models import BaseChatModel
        from quantstack.llm.provider import get_chat_model
        model = get_chat_model("medium")
        assert isinstance(model, BaseChatModel)

    def test_returns_base_chat_model_light(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        from langchain_core.language_models import BaseChatModel
        from quantstack.llm.provider import get_chat_model
        model = get_chat_model("light")
        assert isinstance(model, BaseChatModel)

    def test_invalid_tier_raises_value_error(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        from quantstack.llm.provider import get_chat_model
        with pytest.raises(ValueError, match="tier"):
            get_chat_model("invalid")

    def test_embedding_tier_raises_value_error(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        from quantstack.llm.provider import get_chat_model
        with pytest.raises(ValueError, match="embedding"):
            get_chat_model("embedding")

    def test_anthropic_provider_returns_chat_anthropic(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        from quantstack.llm.provider import get_chat_model
        model = get_chat_model("heavy")
        assert type(model).__name__ == "ChatAnthropic"

    def test_fallback_applies_to_chat_model(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "bedrock")
        monkeypatch.setenv("LLM_FALLBACK_ENABLED", "true")
        monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
        monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        from langchain_core.language_models import BaseChatModel
        from quantstack.llm.provider import get_chat_model
        model = get_chat_model("heavy")
        assert isinstance(model, BaseChatModel)
        assert type(model).__name__ == "ChatAnthropic"

    def test_get_model_still_returns_string(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        from quantstack.llm.provider import get_model
        result = get_model("heavy")
        assert isinstance(result, str)
