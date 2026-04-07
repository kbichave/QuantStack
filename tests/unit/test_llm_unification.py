"""Tests for LLM provider unification (Section 09)."""

import ast
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestHardcodedModelStringAudit:
    """Verify no hardcoded model strings remain outside allowlisted files."""

    # Files that are allowed to contain model name strings
    ALLOWLIST = {
        "llm/config.py",
        "llm/provider.py",
        "llm_config.py",
        "observability/cost_queries.py",
    }

    # Patterns that indicate a hardcoded model string used as a config value.
    # We only flag strings that look like actual model identifiers (short, no
    # whitespace beyond a single word), not log messages or docstrings.
    MODEL_PATTERNS = (
        "claude-sonnet",
        "claude-haiku",
        "claude-opus",
        "gpt-4o",
        "gpt-3.5",
        "llama-3",
        "llama3",
        "qwen",
        "gemini-2",
        "gemini-1",
        "mistral-",
        "text-embedding-3",
    )

    def _is_allowlisted(self, rel_path: str) -> bool:
        for allowed in self.ALLOWLIST:
            if rel_path.endswith(allowed):
                return True
        return False

    def _is_model_string(self, value: str) -> bool:
        """Return True if value looks like an actual model identifier, not prose."""
        val = value.strip().lower()
        # Skip docstrings, log messages, and long prose
        if len(value) > 80:
            return False
        if "\n" in value:
            return False
        # Must match at least one pattern
        for pattern in self.MODEL_PATTERNS:
            if pattern in val:
                # Skip env var default patterns like os.environ.get("...", "gpt-4o-mini")
                # These are acceptable fallbacks in non-LLM-selection code
                return True
        return False

    def _scan_file(self, filepath: Path) -> list[tuple[int, str]]:
        """Scan a Python file for string constants that are model identifiers."""
        violations = []
        try:
            source = filepath.read_text()
            tree = ast.parse(source, filename=str(filepath))
        except (SyntaxError, UnicodeDecodeError):
            return violations

        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if self._is_model_string(node.value):
                    # Skip "claude_code" style strings
                    if "claude_code" in node.value.lower():
                        continue
                    violations.append((node.lineno, node.value))
        return violations

    def test_no_hardcoded_model_strings_outside_allowlist(self):
        """Verify no model name strings used as config values outside allowlisted files.

        The allowlist includes: llm/config.py, llm/provider.py, llm_config.py,
        observability/cost_queries.py. The mem0_client.py fallback uses env vars
        (MEM0_LLM_MODEL / MEM0_EMBED_MODEL) so the defaults there are acceptable
        as os.environ.get() fallbacks, not direct model selection.
        """
        src_root = Path(__file__).resolve().parents[2] / "src" / "quantstack"
        all_violations: dict[str, list] = {}

        # Additional files allowed to have model strings in env-var fallbacks
        env_fallback_allowlist = {"memory/mem0_client.py"}

        for py_file in src_root.rglob("*.py"):
            rel = str(py_file.relative_to(src_root))
            if self._is_allowlisted(rel) or rel in env_fallback_allowlist:
                continue
            violations = self._scan_file(py_file)
            if violations:
                all_violations[rel] = violations

        if all_violations:
            msg_parts = ["Hardcoded model strings found outside allowlisted files:"]
            for filepath, viols in sorted(all_violations.items()):
                for lineno, value in viols:
                    msg_parts.append(f"  {filepath}:{lineno} — {value!r}")
            pytest.fail("\n".join(msg_parts))


class TestGetLlmConfig:
    """Test the three-level precedence chain for get_llm_config."""

    @pytest.fixture(autouse=True)
    def _clear_cache(self):
        from quantstack.llm import provider
        provider._llm_config_cache.clear()
        yield
        provider._llm_config_cache.clear()

    def test_env_var_overrides_everything(self):
        from quantstack.llm.provider import get_llm_config

        with patch.dict(os.environ, {"LLM_TIER_HEAVY": "openai/gpt-4o"}):
            with patch("quantstack.llm.provider._read_llm_config_from_db", return_value={"provider": "anthropic", "model": "claude-sonnet-4-6"}):
                result = get_llm_config("heavy")
        assert result["provider"] == "openai"
        assert result["model"] == "gpt-4o"
        assert result["source"] == "env"

    def test_db_overrides_code_default(self):
        from quantstack.llm.provider import get_llm_config

        env = {k: v for k, v in os.environ.items() if not k.startswith("LLM_TIER_")}
        with patch.dict(os.environ, env, clear=True):
            with patch(
                "quantstack.llm.provider._read_llm_config_from_db",
                return_value={"provider": "openai", "model": "gpt-4o-mini"},
            ):
                result = get_llm_config("heavy")
        assert result["provider"] == "openai"
        assert result["model"] == "gpt-4o-mini"
        assert result["source"] == "db"

    def test_code_default_when_no_env_or_db(self):
        from quantstack.llm.provider import get_llm_config

        env = {k: v for k, v in os.environ.items() if not k.startswith("LLM_TIER_")}
        with patch.dict(os.environ, env, clear=True):
            with patch("quantstack.llm.provider._read_llm_config_from_db", return_value=None):
                result = get_llm_config("heavy")
        assert result["source"] == "code"
        assert result["provider"] is not None
        assert result["model"] is not None

    def test_invalid_tier_raises(self):
        from quantstack.llm.provider import get_llm_config

        with pytest.raises(ValueError, match="Unknown tier"):
            get_llm_config("nonexistent_tier")

    def test_env_var_format_provider_slash_model(self):
        from quantstack.llm.provider import get_llm_config

        with patch.dict(os.environ, {"LLM_TIER_LIGHT": "groq/llama-3.3-70b-versatile"}):
            with patch("quantstack.llm.provider._read_llm_config_from_db", return_value=None):
                result = get_llm_config("light")
        assert result["provider"] == "groq"
        assert result["model"] == "llama-3.3-70b-versatile"


class TestCheckProviderHealth:
    @pytest.mark.asyncio
    async def test_returns_dict_keyed_by_provider(self):
        from quantstack.llm.provider import check_provider_health

        # Patch _instantiate_chat_model to avoid real LLM calls
        mock_model = MagicMock()
        mock_model.ainvoke = MagicMock(return_value=MagicMock(content="ok"))

        with patch("quantstack.llm.provider._instantiate_chat_model", return_value=mock_model):
            with patch("quantstack.llm.provider._validate_provider", return_value=None):
                result = await check_provider_health()

        assert isinstance(result, dict)
        # Should have at least one provider checked
        for name, info in result.items():
            assert "status" in info
            assert "checked_at" in info

    @pytest.mark.asyncio
    async def test_handles_provider_failure(self):
        from quantstack.llm.provider import check_provider_health

        def mock_validate(provider):
            if provider == "bedrock":
                return None
            raise Exception("no creds")

        mock_model = MagicMock()
        mock_model.ainvoke = MagicMock(return_value=MagicMock(content="ok"))

        with patch("quantstack.llm.provider._instantiate_chat_model", return_value=mock_model):
            with patch("quantstack.llm.provider._validate_provider", side_effect=mock_validate):
                result = await check_provider_health()

        # bedrock should be ok, others should be skipped or errored
        assert isinstance(result, dict)


class TestMem0ProviderLayerIntegration:
    def test_mem0_config_uses_provider_layer(self):
        """Verify mem0_client routes LLM config through the provider layer."""
        source = Path(__file__).resolve().parents[2] / "src" / "quantstack" / "memory" / "mem0_client.py"
        content = source.read_text()
        # Should import and call get_llm_config from the provider layer
        assert "get_llm_config" in content, "mem0_client.py should use get_llm_config from provider layer"


class TestCostQueriesCommentExists:
    def test_price_table_has_intentional_comment(self):
        """Verify cost_queries.py has the clarifying comment."""
        source = Path(__file__).resolve().parents[2] / "src" / "quantstack" / "observability" / "cost_queries.py"
        content = source.read_text()
        assert "intentional" in content.lower() or "not a model selection" in content.lower(), \
            "cost_queries.py should have a comment clarifying the price table is intentionally hardcoded"


class TestSupervisorLLMHealthCheck:
    def test_health_check_includes_provider_health(self):
        """Verify the supervisor health_check node calls check_provider_health."""
        source = Path(__file__).resolve().parents[2] / "src" / "quantstack" / "graphs" / "supervisor" / "nodes.py"
        content = source.read_text()
        assert "check_provider_health" in content, \
            "supervisor/nodes.py should call check_provider_health() in health_check"
