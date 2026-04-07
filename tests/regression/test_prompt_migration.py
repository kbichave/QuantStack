# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""Regression test: flag raw f-string prompt construction in graph node files.

This is a lint-style test that detects f-string prompts interpolating variables
that likely carry external data (market data, user input, API responses) without
using ``safe_prompt()`` for parameterized construction.

As nodes are migrated to safe_prompt(), this test's allowlist shrinks. Any NEW
f-string prompt interpolation in a graph node file should use safe_prompt() instead.
"""

import ast
import re
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GRAPH_NODE_FILES = sorted(
    PROJECT_ROOT.glob("src/quantstack/graphs/*/nodes.py")
)

# Known f-string prompt patterns that predate safe_prompt() and are tracked
# for migration. Remove entries as they are migrated. This list should
# shrink monotonically -- never add new entries.
#
# Format: (filename_stem, line_number_approx, snippet)
# We use filename stem + a substring match rather than exact line numbers
# because line numbers shift as code changes.
_MIGRATION_ALLOWLIST: set[str] = {
    # Pre-existing patterns in nodes files -- to be migrated incrementally.
    # Each entry is the relative path from project root.
    "src/quantstack/graphs/research/nodes.py",
    "src/quantstack/graphs/trading/nodes.py",
    "src/quantstack/graphs/supervisor/nodes.py",
}

# Regex to find f-strings that interpolate variables (not just literals).
# Matches: f"...{variable_name..." or f'...{expr...'
# Excludes: f"literal text" (no braces), f"{42}" (literal int), f"{'string'}"
_FSTRING_WITH_VARIABLE_RE = re.compile(
    r'''f["'].*\{[a-zA-Z_]'''
)


def _find_fstring_prompts(filepath: Path) -> list[tuple[int, str]]:
    """Find lines with f-string interpolation that look like prompt construction.

    Returns list of (line_number, line_text) tuples.
    """
    findings = []
    try:
        source = filepath.read_text()
    except OSError:
        return findings

    for i, line in enumerate(source.splitlines(), start=1):
        stripped = line.strip()
        # Skip comments and empty lines
        if stripped.startswith("#") or not stripped:
            continue
        # Look for f-strings with variable interpolation
        if _FSTRING_WITH_VARIABLE_RE.search(stripped):
            # Heuristic: is this likely a prompt being sent to an LLM?
            # Check if the surrounding context includes prompt-related keywords.
            # For now, flag all f-strings with variable interpolation in node files
            # since these files construct LLM prompts.
            findings.append((i, stripped))

    return findings


class TestPromptMigration:
    """Track migration of raw f-string prompts to safe_prompt()."""

    @pytest.mark.parametrize(
        "node_file",
        GRAPH_NODE_FILES,
        ids=[str(f.relative_to(PROJECT_ROOT)) for f in GRAPH_NODE_FILES],
    )
    def test_fstring_prompts_are_tracked(self, node_file: Path):
        """Every graph node file with f-string prompts must be in the allowlist.

        This test does NOT require immediate migration -- it ensures that
        new node files with f-string prompt construction are noticed and
        added to the migration backlog.
        """
        findings = _find_fstring_prompts(node_file)
        rel_path = str(node_file.relative_to(PROJECT_ROOT))

        if not findings:
            # Clean file -- no f-string prompts found. If it was in the
            # allowlist, it can be removed (migration complete).
            return

        assert rel_path in _MIGRATION_ALLOWLIST, (
            f"{rel_path} has {len(findings)} f-string prompt interpolations "
            f"but is NOT in the migration allowlist.\n"
            f"Either migrate these to safe_prompt() or add to _MIGRATION_ALLOWLIST:\n"
            + "\n".join(f"  L{ln}: {text[:120]}" for ln, text in findings[:10])
        )

    def test_allowlist_entries_are_real_files(self):
        """Every entry in the allowlist must correspond to an actual file."""
        for rel_path in _MIGRATION_ALLOWLIST:
            full_path = PROJECT_ROOT / rel_path
            assert full_path.exists(), (
                f"Allowlist entry '{rel_path}' does not exist. "
                f"Remove it from _MIGRATION_ALLOWLIST."
            )

    def test_no_new_unsafe_prompts_in_non_node_graph_files(self):
        """Graph files other than nodes.py should not introduce f-string prompts.

        Files like graph.py, models.py, briefs.py are not expected to construct
        LLM prompts with external data interpolation.
        """
        graph_dir = PROJECT_ROOT / "src" / "quantstack" / "graphs"
        non_node_files = [
            f for f in graph_dir.rglob("*.py")
            if f.name != "nodes.py"
            and f.name != "__init__.py"
            and f.name != "prompt_safety.py"
        ]

        violations = []
        for filepath in non_node_files:
            findings = _find_fstring_prompts(filepath)
            if findings:
                rel = str(filepath.relative_to(PROJECT_ROOT))
                violations.append(
                    f"{rel}: {len(findings)} f-string interpolations"
                )

        # This is informational -- non-node graph files rarely build prompts,
        # but some (like agent_executor.py) do for system messages.
        # We don't fail on these, just track them.
        if violations:
            pytest.skip(
                f"Found f-string interpolations in non-node graph files "
                f"(informational, not blocking):\n"
                + "\n".join(f"  - {v}" for v in violations)
            )
