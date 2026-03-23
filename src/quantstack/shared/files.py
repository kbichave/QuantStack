# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""Shared file utilities — zero intra-project dependencies."""

from pathlib import Path


def read_memory_file(filename: str, max_chars: int = 2000) -> str:
    """Read a .claude/memory/*.md file and return its content (truncated)."""
    candidates = [
        Path(__file__).parents[4] / ".claude" / "memory" / filename,
        Path.home() / ".claude" / "memory" / filename,
    ]
    for path in candidates:
        if path.exists():
            try:
                content = path.read_text(encoding="utf-8")
                return content[:max_chars] if len(content) > max_chars else content
            except OSError:
                pass
    return ""
