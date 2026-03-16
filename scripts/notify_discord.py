#!/usr/bin/env python3
"""
PostToolUse hook for execute_trade — sends Discord notification.

Checks DISCORD_WEBHOOK_URL env var. If not set, prints skip message and exits 0.
Posts trade fill as a Discord embed.

Always exits 0 regardless of HTTP errors.
"""

import json
import os
import sys
from datetime import datetime

try:
    import urllib.request
    import urllib.error
except ImportError:
    pass


WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL", "")


def main():
    if not WEBHOOK_URL:
        print("DISCORD_WEBHOOK_URL not set — skipping Discord notification", file=sys.stderr)
        sys.exit(0)

    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, EOFError):
        sys.exit(0)

    tool_input = payload.get("tool_input", {})
    tool_response = payload.get("tool_response", {})

    # Parse response
    if isinstance(tool_response, str):
        try:
            tool_response = json.loads(tool_response)
        except json.JSONDecodeError:
            tool_response = {}

    # Handle FastMCP content wrapping
    if isinstance(tool_response, dict) and "content" in tool_response:
        content = tool_response.get("content", [])
        if isinstance(content, list) and content:
            text_item = content[0] if isinstance(content[0], dict) else {}
            text = text_item.get("text", "")
            try:
                tool_response = json.loads(text)
            except json.JSONDecodeError:
                pass

    symbol = tool_input.get("symbol", "?")
    action = tool_input.get("action", "?")
    paper_mode = tool_input.get("paper_mode", True)
    strategy_id = tool_input.get("strategy_id", "manual")
    confidence = tool_input.get("confidence", 0.0)

    success = tool_response.get("success", False)
    fill_price = tool_response.get("fill_price", "N/A")
    filled_qty = tool_response.get("filled_quantity", "N/A")
    broker_mode = tool_response.get("broker_mode", "paper")

    mode_label = "PAPER" if paper_mode else "LIVE"
    color = 0x00FF00 if action.lower() == "buy" else 0xFF0000  # green/red
    status = "FILLED" if success else "REJECTED"

    embed = {
        "embeds": [
            {
                "title": f"{status}: {action.upper()} {symbol}",
                "color": color,
                "fields": [
                    {"name": "Mode", "value": mode_label, "inline": True},
                    {"name": "Broker", "value": broker_mode, "inline": True},
                    {"name": "Confidence", "value": f"{confidence:.0%}", "inline": True},
                    {"name": "Fill Price", "value": str(fill_price), "inline": True},
                    {"name": "Quantity", "value": str(filled_qty), "inline": True},
                    {"name": "Strategy", "value": strategy_id or "manual", "inline": True},
                ],
                "timestamp": datetime.utcnow().isoformat(),
                "footer": {"text": "QuantPod Trading Intelligence"},
            }
        ]
    }

    try:
        data = json.dumps(embed).encode("utf-8")
        req = urllib.request.Request(
            WEBHOOK_URL,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=10)
    except Exception as e:
        print(f"Discord notification failed: {e}", file=sys.stderr)

    sys.exit(0)


if __name__ == "__main__":
    main()
