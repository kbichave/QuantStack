#!/usr/bin/env python3
"""
PostToolUse hook for execute_trade — logs trade decision to trade_journal.md.

Reads the tool result from stdin (JSON). Appends a formatted entry
to .claude/memory/trade_journal.md under "## Recent Trades".

Always exits 0.
"""

import json
import os
import sys
from datetime import datetime


JOURNAL_PATH = os.path.join(
    os.environ.get("CLAUDE_PROJECT_DIR", "."),
    ".claude",
    "memory",
    "trade_journal.md",
)


def main():
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, EOFError):
        sys.exit(0)

    tool_input = payload.get("tool_input", {})
    tool_response = payload.get("tool_response", {})

    # Parse response — may be string or dict
    if isinstance(tool_response, str):
        try:
            tool_response = json.loads(tool_response)
        except json.JSONDecodeError:
            tool_response = {}

    # Handle legacy content wrapping
    if isinstance(tool_response, dict) and "content" in tool_response:
        content = tool_response.get("content", [])
        if isinstance(content, list) and content:
            text_item = content[0] if isinstance(content[0], dict) else {}
            text = text_item.get("text", "")
            try:
                tool_response = json.loads(text)
            except json.JSONDecodeError:
                pass

    # Extract fields from input and response
    symbol = tool_input.get("symbol", "unknown")
    action = tool_input.get("action", "unknown")
    reasoning = tool_input.get("reasoning", "")
    confidence = tool_input.get("confidence", 0.0)
    strategy_id = tool_input.get("strategy_id", "none")
    paper_mode = tool_input.get("paper_mode", True)

    success = tool_response.get("success", False)
    fill_price = tool_response.get("fill_price", "N/A")
    filled_qty = tool_response.get("filled_quantity", "N/A")
    risk_approved = tool_response.get("risk_approved", False)
    violations = tool_response.get("risk_violations", [])
    error = tool_response.get("error", "")

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    mode = "PAPER" if paper_mode else "LIVE"

    entry = f"""
### {now} — {symbol} {action.upper()} ({mode})
- **Status:** {"FILLED" if success else "REJECTED"}
- **Confidence:** {confidence}
- **Strategy:** {strategy_id}
- **Fill Price:** {fill_price}
- **Quantity:** {filled_qty}
- **Risk Approved:** {risk_approved}
{"- **Violations:** " + ", ".join(violations) if violations else ""}
{"- **Error:** " + error if error else ""}
- **Reasoning:** {reasoning[:200]}
"""

    try:
        if os.path.exists(JOURNAL_PATH):
            with open(JOURNAL_PATH, "r") as f:
                content = f.read()

            marker = "## Recent Trades"
            if marker in content:
                # Insert after the marker line
                idx = content.index(marker) + len(marker)
                # Skip any blank lines or "(empty...)" placeholder
                rest = content[idx:]
                # Remove placeholder if present
                rest = rest.replace("\n(empty — populated starting Phase 3)\n", "\n")
                content = content[:idx] + "\n" + entry + rest

                with open(JOURNAL_PATH, "w") as f:
                    f.write(content)
    except Exception as e:
        print(f"Warning: could not update trade journal: {e}", file=sys.stderr)

    sys.exit(0)


if __name__ == "__main__":
    main()
