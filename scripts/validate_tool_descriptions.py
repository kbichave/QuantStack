#!/usr/bin/env python3
"""Validate tool descriptions for BM25 search quality.

Checks all registered tools for:
1. Minimum description length (80 chars)
2. Argument descriptions on all input fields
3. No generic/restated descriptions
4. Contains actionable guidance ("Use when", "Returns", etc.)

Usage:
    uv run python scripts/validate_tool_descriptions.py
"""

from __future__ import annotations

import re
import sys

from langchain_core.tools import BaseTool

MIN_DESC_LENGTH = 80
ACTIONABLE_PHRASES = ["use when", "returns", "use this", "call this", "provides", "retrieves", "computes", "calculates"]


def validate_tool_description(tool: BaseTool) -> list[str]:
    """Return list of violation messages. Empty list = pass."""
    violations = []
    desc = tool.description or ""

    # 1. Length check
    if len(desc) < MIN_DESC_LENGTH:
        violations.append(
            f"description too short ({len(desc)} chars, min {MIN_DESC_LENGTH})"
        )

    # 2. Generic/restated check — description is just the function name humanized
    name_words = set(tool.name.replace("_", " ").lower().split())
    desc_words = set(desc.lower().split())
    # If description words are a near-subset of the function name, it's too generic
    if len(desc.split()) <= 5 and name_words and name_words.issubset(desc_words | {"the", "a", "an", "for", "of"}):
        violations.append(
            "description appears to just restate the function name"
        )

    # 3. Actionable guidance check
    desc_lower = desc.lower()
    has_actionable = any(phrase in desc_lower for phrase in ACTIONABLE_PHRASES)
    if not has_actionable and len(desc) >= MIN_DESC_LENGTH:
        violations.append(
            "description lacks actionable guidance (missing 'Use when', 'Returns', etc.)"
        )

    # 4. Argument description check
    schema = tool.args_schema
    if schema is not None:
        try:
            json_schema = schema.model_json_schema()
            properties = json_schema.get("properties", {})
            for field_name, field_info in properties.items():
                field_desc = field_info.get("description", "")
                if not field_desc:
                    violations.append(
                        f"argument '{field_name}' missing description"
                    )
        except Exception:
            pass  # Some tools may not have introspectable schemas

    return violations


def main():
    from quantstack.tools.registry import TOOL_REGISTRY

    total = 0
    passed = 0
    failed = 0
    failures = []

    for name, tool in sorted(TOOL_REGISTRY.items()):
        total += 1
        violations = validate_tool_description(tool)
        if violations:
            failed += 1
            # Count described args
            schema = tool.args_schema
            arg_count = 0
            described = 0
            if schema:
                try:
                    props = schema.model_json_schema().get("properties", {})
                    arg_count = len(props)
                    described = sum(1 for p in props.values() if p.get("description"))
                except Exception:
                    pass
            desc_len = len(tool.description or "")
            print(f"FAIL  {name}: {'; '.join(violations)}")
            failures.append((name, violations))
        else:
            desc_len = len(tool.description or "")
            schema = tool.args_schema
            arg_count = 0
            described = 0
            if schema:
                try:
                    props = schema.model_json_schema().get("properties", {})
                    arg_count = len(props)
                    described = sum(1 for p in props.values() if p.get("description"))
                except Exception:
                    pass
            passed += 1
            print(f"PASS  {name} ({desc_len} chars, {described}/{arg_count} args described)")

    print(f"\nSummary: {passed}/{total} passed, {failed} failed")
    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
