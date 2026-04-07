"""Tests for Section 05: Per-Agent Temperature Config.

Validates temperature field in AgentConfig and get_chat_model() signature.
"""

import pytest


class TestAgentConfigTemperature:
    """Temperature field in AgentConfig."""

    def test_accepts_temperature(self):
        from quantstack.graphs.config import AgentConfig
        cfg = AgentConfig(
            name="test", role="test", goal="test", backstory="test",
            llm_tier="heavy", temperature=0.7,
        )
        assert cfg.temperature == 0.7

    def test_temperature_defaults_to_none(self):
        from quantstack.graphs.config import AgentConfig
        cfg = AgentConfig(
            name="test", role="test", goal="test", backstory="test",
            llm_tier="heavy",
        )
        assert cfg.temperature is None

    def test_temperature_zero_is_valid(self):
        from quantstack.graphs.config import AgentConfig
        cfg = AgentConfig(
            name="test", role="test", goal="test", backstory="test",
            llm_tier="heavy", temperature=0.0,
        )
        assert cfg.temperature == 0.0


class TestGetChatModelTemperature:
    """get_chat_model() accepts temperature parameter."""

    def test_with_temperature(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        from quantstack.llm.provider import get_chat_model
        model = get_chat_model("heavy", temperature=0.5)
        # ChatAnthropic stores temperature
        assert hasattr(model, "temperature")
        assert model.temperature == 0.5

    def test_without_temperature_uses_default(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        from quantstack.llm.provider import get_chat_model
        model = get_chat_model("heavy")
        assert model.temperature == 0.0

    def test_temperature_none_uses_default(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        from quantstack.llm.provider import get_chat_model
        model = get_chat_model("heavy", temperature=None)
        assert model.temperature == 0.0

    def test_backward_compat_no_temperature_kwarg(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        from quantstack.llm.provider import get_chat_model
        # Should work exactly as before without temperature kwarg
        model = get_chat_model("heavy", thinking=None)
        assert model is not None
