"""Provider fallback tests: verify LLM fallback chain works."""

import pytest
from unittest.mock import patch

from quantstack.llm.provider import get_model, get_model_with_fallback, ConfigurationError
from quantstack.llm.config import VALID_TIERS


class TestProviderFallback:
    @pytest.fixture(autouse=True)
    def _set_ollama_env(self, monkeypatch):
        """Use ollama provider with a fake base URL for all tests."""
        monkeypatch.setenv("LLM_PROVIDER", "ollama")
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")

    def test_get_model_returns_string_for_each_tier(self):
        for tier in VALID_TIERS:
            result = get_model(tier)
            assert isinstance(result, str)
            assert len(result) > 0

    def test_get_model_heavy_tier(self):
        result = get_model("heavy")
        assert isinstance(result, str)

    def test_get_model_light_tier(self):
        result = get_model("light")
        assert isinstance(result, str)

    def test_get_model_unknown_tier_raises(self):
        with pytest.raises(ValueError):
            get_model("nonexistent_tier")

    def test_missing_env_vars_raises_configuration_error(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(ConfigurationError):
            get_model("heavy")

    def test_fallback_tries_next_provider(self, monkeypatch):
        """When primary provider fails, fallback chain tries the next."""
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        # Ollama is available, so fallback should find it
        monkeypatch.setenv("LLM_FALLBACK_ENABLED", "true")
        result = get_model_with_fallback("heavy")
        assert isinstance(result, str)
