"""Tests for Section 07: No Hardcoded Model Strings.

Ensures no provider-specific model strings or direct LLM instantiation
exist outside the designated config files.
"""

import re
from pathlib import Path

import pytest

SRC_ROOT = Path("src/quantstack")

# Files that legitimately contain model strings (config/pricing tables)
ALLOWED_FILES = {
    "llm/config.py",       # Provider config definitions
    "llm_config.py",       # Legacy config (deprecated, still referenced)
    "llm/provider.py",     # Provider instantiation logic
    "observability/cost_queries.py",  # Pricing table (model names for cost calc)
    "memory/mem0_client.py",  # Mem0 library requires provider-specific model names in its config
}

# Patterns that indicate hardcoded model strings
_HARDCODED_PATTERNS = [
    r'"claude-sonnet',
    r'"claude-haiku',
    r'"gpt-4o',
    r'"llama-3',
    r'"text-embedding',
]

_HARDCODED_RE = re.compile("|".join(_HARDCODED_PATTERNS))

# Direct LLM class instantiation — only allowed in provider.py
_DIRECT_INSTANTIATION_RE = re.compile(
    r"(?:ChatAnthropic|ChatBedrock|ChatOpenAI|ChatOllama|ChatGoogleGenerativeAI)\("
)


def _collect_py_files():
    """Yield (relative_path, full_path) for all .py files under src/quantstack."""
    for p in SRC_ROOT.rglob("*.py"):
        rel = str(p.relative_to(SRC_ROOT))
        yield rel, p


class TestNoHardcodedModelStrings:
    """No hardcoded model strings outside config files."""

    def test_no_hardcoded_model_strings(self):
        violations = []
        for rel, path in _collect_py_files():
            if rel in ALLOWED_FILES:
                continue
            content = path.read_text()
            for i, line in enumerate(content.splitlines(), 1):
                # Skip comments and docstrings (rough heuristic)
                stripped = line.strip()
                if stripped.startswith("#") or stripped.startswith('"""') or stripped.startswith("'''"):
                    continue
                if _HARDCODED_RE.search(line):
                    violations.append(f"{rel}:{i}: {stripped}")
        assert violations == [], (
            "Hardcoded model strings found outside config files:\n"
            + "\n".join(violations)
        )


class TestNoDirectLLMInstantiation:
    """No direct ChatModel instantiation outside llm/provider.py."""

    def test_no_direct_instantiation(self):
        violations = []
        for rel, path in _collect_py_files():
            if rel == "llm/provider.py":
                continue
            content = path.read_text()
            for i, line in enumerate(content.splitlines(), 1):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                if _DIRECT_INSTANTIATION_RE.search(line):
                    violations.append(f"{rel}:{i}: {stripped}")
        assert violations == [], (
            "Direct LLM instantiation found outside llm/provider.py:\n"
            + "\n".join(violations)
        )
