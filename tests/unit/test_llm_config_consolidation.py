"""Tests for Section 01: Consolidate Dual LLM Config Systems.

Validates that:
1. Legacy tier names (IC/Pod/Workshop/etc.) resolve via aliases to standard tiers
2. get_model_for_role() handles bulk/research with special routing
3. Legacy env var overrides still work
4. No file in src/quantstack/ imports from the legacy llm_config module
5. Importing from quantstack.llm_config emits DeprecationWarning
"""

import os
import subprocess
from pathlib import Path

import pytest


class TestTierAliases:
    """Legacy tier names resolve to standard tiers via TIER_ALIASES."""

    def test_ic_maps_to_light(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        from quantstack.llm.provider import get_model, TIER_ALIASES
        assert TIER_ALIASES["ic"] == "light"
        # IC should resolve to the same model as light
        assert get_model("ic") == get_model("light")

    def test_pod_maps_to_heavy(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        from quantstack.llm.provider import get_model, TIER_ALIASES
        assert TIER_ALIASES["pod"] == "heavy"
        assert get_model("pod") == get_model("heavy")

    def test_workshop_maps_to_heavy(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        from quantstack.llm.provider import get_model, TIER_ALIASES
        assert TIER_ALIASES["workshop"] == "heavy"
        assert get_model("workshop") == get_model("heavy")

    def test_assistant_maps_to_heavy(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        from quantstack.llm.provider import get_model, TIER_ALIASES
        assert TIER_ALIASES["assistant"] == "heavy"
        assert get_model("assistant") == get_model("heavy")

    def test_decoder_maps_to_light(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        from quantstack.llm.provider import get_model, TIER_ALIASES
        assert TIER_ALIASES["decoder"] == "light"
        assert get_model("decoder") == get_model("light")

    def test_alias_case_insensitive(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        from quantstack.llm.provider import get_model
        assert get_model("IC") == get_model("light")
        assert get_model("Pod") == get_model("heavy")


class TestLegacyEnvVarOverrides:
    """Legacy env vars (LLM_MODEL_IC, etc.) still work after migration."""

    def test_llm_model_ic_overrides_light(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        monkeypatch.setenv("LLM_MODEL_IC", "groq/qwen/qwen3-32b")
        from quantstack.llm.provider import get_model_for_role
        result = get_model_for_role("ic")
        assert result == "groq/qwen/qwen3-32b"

    def test_llm_model_pod_overrides_heavy(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        monkeypatch.setenv("LLM_MODEL_POD", "bedrock/us.anthropic.claude-sonnet-4-6")
        from quantstack.llm.provider import get_model_for_role
        result = get_model_for_role("pod")
        assert result == "bedrock/us.anthropic.claude-sonnet-4-6"


class TestGetModelForRole:
    """get_model_for_role() returns model strings for litellm callers."""

    def test_bulk_returns_model_string(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        from quantstack.llm.provider import get_model_for_role
        result = get_model_for_role("bulk")
        assert isinstance(result, str)
        assert "/" in result  # provider/model format

    def test_bulk_prefers_groq(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        monkeypatch.setenv("GROQ_API_KEY", "test-key")
        from quantstack.llm.provider import get_model_for_role
        result = get_model_for_role("bulk")
        assert "groq" in result

    def test_bulk_falls_back_to_light_without_groq(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        from quantstack.llm.provider import get_model_for_role
        result = get_model_for_role("bulk")
        # Should fall back to light tier model
        light = get_model_for_role("light")
        assert result == light

    def test_research_returns_heavy_model(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        from quantstack.llm.provider import get_model_for_role
        result = get_model_for_role("research")
        heavy = get_model_for_role("heavy")
        assert result == heavy

    def test_heavy_returns_model_string(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        from quantstack.llm.provider import get_model_for_role
        result = get_model_for_role("heavy")
        assert isinstance(result, str)
        assert "/" in result

    def test_unknown_role_raises(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        from quantstack.llm.provider import get_model_for_role
        with pytest.raises(ValueError, match="Unknown role"):
            get_model_for_role("nonexistent")

    def test_env_override_for_bulk(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        monkeypatch.setenv("LLM_MODEL_BULK", "anthropic/claude-haiku-4-5")
        from quantstack.llm.provider import get_model_for_role
        result = get_model_for_role("bulk")
        assert result == "anthropic/claude-haiku-4-5"


class TestNoLegacyImports:
    """After migration, no production code imports from llm_config."""

    def test_no_imports_from_llm_config(self):
        """Grep src/quantstack/ for imports from the legacy module.
        Excludes llm_config.py itself (deprecated stub).
        """
        src_dir = Path(__file__).parent.parent.parent / "src" / "quantstack"
        result = subprocess.run(
            [
                "grep", "-rn",
                "from quantstack.llm_config import",
                str(src_dir),
                "--include=*.py",
            ],
            capture_output=True, text=True,
        )
        # Filter out llm_config.py itself
        hits = [
            line for line in result.stdout.strip().split("\n")
            if line and "llm_config.py" not in line
        ]
        assert hits == [], (
            f"Found legacy llm_config imports in production code:\n"
            + "\n".join(hits)
        )


class TestDeprecationWarning:
    """Importing from llm_config emits DeprecationWarning."""

    def test_import_emits_deprecation(self):
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            import importlib
            import quantstack.llm_config as mod
            importlib.reload(mod)
            deprecation_warnings = [
                x for x in w if issubclass(x.category, DeprecationWarning)
            ]
            assert len(deprecation_warnings) > 0, (
                "Expected DeprecationWarning when importing llm_config"
            )
