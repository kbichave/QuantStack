#!/usr/bin/env python3
"""
PostToolUse hook for run_analysis — validates DailyBrief quality.

Reads the tool result from stdin (JSON). Warns (never blocks) if:
  - No symbol briefs in the DailyBrief
  - Regime confidence < 0.6
  - Overall confidence < 0.5

Always exits 0 — this is advisory feedback, not a gate.
"""

import json
import sys


def main():
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, EOFError):
        # No input or malformed — skip silently
        sys.exit(0)

    tool_response = payload.get("tool_response", {})

    # tool_response may be the raw return value
    # or wrapped in a content structure — handle both
    if isinstance(tool_response, str):
        try:
            tool_response = json.loads(tool_response)
        except json.JSONDecodeError:
            sys.exit(0)

    # Navigate to the daily_brief — may be nested differently
    brief = None
    if isinstance(tool_response, dict):
        brief = tool_response.get("daily_brief")
        if brief is None and "content" in tool_response:
            # Legacy content wrapping (array format)
            content = tool_response.get("content", [])
            if isinstance(content, list) and content:
                text_item = content[0] if isinstance(content[0], dict) else {}
                text = text_item.get("text", "")
                try:
                    parsed = json.loads(text)
                    brief = parsed.get("daily_brief")
                except json.JSONDecodeError:
                    pass

    if brief is None:
        sys.exit(0)

    warnings = []

    # Check for empty symbol briefs
    symbol_briefs = brief.get("symbol_briefs", [])
    if not symbol_briefs:
        warnings.append("DailyBrief has no symbol_briefs — analysis may be empty")

    # Check overall confidence
    overall = brief.get("overall_confidence", 1.0)
    if overall < 0.5:
        warnings.append(
            f"DailyBrief overall_confidence={overall:.2f} is below 0.5 — "
            "analysis quality may be low"
        )

    # Check regime confidence from regime_used
    regime = payload.get("tool_response", {})
    if isinstance(regime, dict):
        regime_used = regime.get("regime_used", {})
        regime_conf = regime_used.get("confidence", 1.0)
        if regime_conf < 0.6:
            warnings.append(
                f"Regime confidence={regime_conf:.2f} is below 0.6 — "
                "regime classification may be unreliable"
            )

    if warnings:
        # Output structured feedback for Claude
        result = {
            "decision": "block",
            "reason": "DailyBrief quality warnings:\n"
            + "\n".join(f"- {w}" for w in warnings),
        }
        json.dump(result, sys.stdout)
    # Always exit 0
    sys.exit(0)


if __name__ == "__main__":
    main()
