"""Tests for Section 10: Documentation and Cleanup.

These verify that all references to the deleted MCP infrastructure
have been removed from documentation and source files.
"""
import pathlib
import re

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
CLAUDE_MD = PROJECT_ROOT / "CLAUDE.md"


def test_claude_md_no_mcp_tools_reference():
    """CLAUDE.md must not reference 'mcp/tools/' or 'MCP bridge'."""
    text = CLAUDE_MD.read_text()
    assert "mcp/tools/" not in text, "CLAUDE.md still references mcp/tools/"
    assert "mcp bridge" not in text.lower(), "CLAUDE.md still references MCP bridge"


def test_claude_md_no_legacy_tool_tier():
    """The 'Legacy' tool tier line must be removed from CLAUDE.md."""
    text = CLAUDE_MD.read_text()
    # Check that there is no "Legacy:" line that also mentions mcp
    for line in text.splitlines():
        if "Legacy:" in line and "mcp" in line.lower():
            raise AssertionError(f"CLAUDE.md still lists Legacy MCP tool tier: {line}")


def test_no_dead_mcp_imports_in_source():
    """No Python file in src/ should import from quantstack.mcp (deleted)."""
    violations = []
    for py_file in SRC_DIR.rglob("*.py"):
        content = py_file.read_text()
        for line_no, line in enumerate(content.splitlines(), 1):
            if line.strip().startswith("#"):
                continue
            if re.search(r"from quantstack\.mcp\b", line) or re.search(r"import quantstack\.mcp\b", line):
                violations.append(f"{py_file}:{line_no}: {line.strip()}")
    assert not violations, f"Dead MCP imports found:\n" + "\n".join(violations)


def test_no_bridge_references_in_source():
    """No Python file in src/ should reference get_bridge, MCPBridge, or call_quantcore."""
    patterns = [r"\bget_bridge\b", r"\bMCPBridge\b", r"\bcall_quantcore\b"]
    violations = []
    for py_file in SRC_DIR.rglob("*.py"):
        content = py_file.read_text()
        for line_no, line in enumerate(content.splitlines(), 1):
            if line.strip().startswith("#"):
                continue
            for pat in patterns:
                if re.search(pat, line):
                    violations.append(f"{py_file}:{line_no}: {line.strip()}")
    assert not violations, f"Bridge references found:\n" + "\n".join(violations)
