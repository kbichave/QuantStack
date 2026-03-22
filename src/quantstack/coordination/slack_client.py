# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Slack client for QuantPod — posts to channels via Bot Token API.

Uses the Slack Web API (chat.postMessage) with a Bot Token for:
- Threaded agent conversations in #agent-activity
- Trade fills in #trades
- Portfolio summaries in #portfolio
- Signal briefs in #signals
- Alerts in #alerts
- System heartbeats in #system

Graceful degradation: if SLACK_BOT_TOKEN is not set, all posts silently
return False. Never crashes the trading loop.

The Slack MCP server (@modelcontextprotocol/server-slack) handles the
READ side — Claude can search and read Slack channels for self-optimization.
This module handles only the WRITE side.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Any

from loguru import logger

# Slack channel defaults (overridable via env)
_CHANNELS = {
    "agents": os.getenv("SLACK_CHANNEL_AGENTS", "#agent-activity"),
    "trades": os.getenv("SLACK_CHANNEL_TRADES", "#trades"),
    "portfolio": os.getenv("SLACK_CHANNEL_PORTFOLIO", "#portfolio"),
    "signals": os.getenv("SLACK_CHANNEL_SIGNALS", "#signals"),
    "alerts": os.getenv("SLACK_CHANNEL_ALERTS", "#alerts"),
    "system": os.getenv("SLACK_CHANNEL_SYSTEM", "#system"),
    "strategies": os.getenv("SLACK_CHANNEL_STRATEGIES", "#strategies"),
}

# Agent display config
_AGENT_EMOJI = {
    "market_intel": ":globe_with_meridians:",
    "alpha_research": ":mag:",
    "risk": ":shield:",
    "execution": ":zap:",
    "strategy_rd": ":test_tube:",
    "data_scientist": ":robot_face:",
    "watchlist": ":eyes:",
    "pm": ":brain:",
    "signal_engine": ":satellite:",
    "system": ":gear:",
}


class SlackClient:
    """
    Posts messages to Slack channels via the Web API (chat.postMessage).

    Requires SLACK_BOT_TOKEN env var. If not set, all methods silently
    return False without raising exceptions.
    """

    def __init__(self, token: str | None = None) -> None:
        self._token = token or os.getenv("SLACK_BOT_TOKEN", "")
        self._api_url = "https://slack.com/api/chat.postMessage"

    @property
    def is_configured(self) -> bool:
        return bool(self._token)

    def post(
        self,
        channel_key: str,
        text: str,
        blocks: list[dict] | None = None,
        thread_ts: str | None = None,
    ) -> str | None:
        """
        Post a message to a Slack channel.

        Args:
            channel_key: Key from _CHANNELS dict (e.g., "agents", "trades").
            text: Fallback text (shown in notifications).
            blocks: Slack Block Kit blocks for rich formatting.
            thread_ts: Thread timestamp to reply in a thread.

        Returns:
            Message timestamp (ts) if successful, None otherwise.
        """
        if not self._token:
            return None

        channel = _CHANNELS.get(channel_key, channel_key)
        payload: dict[str, Any] = {
            "channel": channel,
            "text": text,
        }
        if blocks:
            payload["blocks"] = blocks
        if thread_ts:
            payload["thread_ts"] = thread_ts

        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                self._api_url,
                data=data,
                headers={
                    "Content-Type": "application/json; charset=utf-8",
                    "Authorization": f"Bearer {self._token}",
                },
                method="POST",
            )
            resp = urllib.request.urlopen(req, timeout=10)
            result = json.loads(resp.read())
            if result.get("ok"):
                return result.get("ts")
            else:
                logger.debug(f"[Slack] API error: {result.get('error')}")
                return None
        except Exception as exc:
            logger.debug(f"[Slack] Post failed: {exc}")
            return None

    def post_agent_report(
        self,
        agent_name: str,
        symbol: str | None,
        summary: str,
        full_report: str,
        thread_ts: str | None = None,
    ) -> str | None:
        """Post a desk agent report to #agent-activity."""
        emoji = _AGENT_EMOJI.get(agent_name, ":speech_balloon:")
        header = f"{emoji} *{agent_name.replace('_', ' ').title()}*"
        if symbol:
            header += f" — `{symbol}`"

        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{agent_name.replace('_', ' ').title()} {'— ' + symbol if symbol else ''}",
                },
            },
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"*Summary:* {summary}"},
            },
            {"type": "divider"},
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": self._truncate(full_report, 2900)},
            },
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": f":clock1: {datetime.now(timezone.utc).strftime('%H:%M UTC')}",
                    }
                ],
            },
        ]

        return self.post("agents", f"{header}: {summary}", blocks, thread_ts)

    def post_trade(self, trade_result: dict[str, Any]) -> str | None:
        """Post a trade fill to #trades."""
        symbol = trade_result.get("underlying", trade_result.get("symbol", "?"))
        action = trade_result.get("action", "?")
        is_options = "option_type" in trade_result

        if is_options:
            opt_type = trade_result.get("option_type", "?")
            strike = trade_result.get("strike", "?")
            expiry = trade_result.get("expiry_date", "?")
            premium = trade_result.get(
                "fill_premium_per_contract", trade_result.get("total_premium", 0)
            )
            contracts = trade_result.get(
                "contracts", trade_result.get("contracts_filled", 1)
            )
            delta = trade_result.get("delta", "?")
            title = f"{action.upper()} {contracts} {symbol} {strike}{opt_type[0].upper()} exp {expiry}"
            detail = f"Premium: ${premium:.2f}/contract | Delta: {delta} | Mode: {trade_result.get('execution_mode', 'paper')}"
        else:
            qty = trade_result.get("filled_quantity", trade_result.get("quantity", 0))
            price = trade_result.get("fill_price", 0)
            title = f"{action.upper()} {qty} {symbol} @ ${price:.2f}"
            detail = f"Slippage: {trade_result.get('slippage_bps', 0):.1f} bps | Mode: {trade_result.get('broker_mode', 'paper')}"

        color = "#2eb886" if action in ("buy", "buy_call", "buy_put") else "#e01e5a"

        blocks = [
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"*{title}*\n{detail}"},
            },
        ]

        return self.post("trades", title, blocks)

    def post_portfolio_summary(self, snapshot: dict[str, Any]) -> str | None:
        """Post portfolio summary to #portfolio."""
        equity = snapshot.get("total_equity", 0)
        cash = snapshot.get("cash", 0)
        daily_pnl = snapshot.get("daily_pnl", 0)
        positions = snapshot.get("open_positions", 0)

        pnl_emoji = (
            ":chart_with_upwards_trend:"
            if daily_pnl >= 0
            else ":chart_with_downwards_trend:"
        )
        text = (
            f"{pnl_emoji} *Portfolio Update*\n"
            f"Equity: ${equity:,.0f} | Cash: ${cash:,.0f}\n"
            f"Daily P&L: ${daily_pnl:+,.0f} | Positions: {positions}"
        )

        return self.post("portfolio", text)

    def post_signal_brief(
        self,
        symbol: str,
        bias: str,
        conviction: float,
        raw_collectors: dict[str, Any] | None = None,
    ) -> str | None:
        """Post signal brief + raw collector data to #signals."""
        bias_emoji = {
            "bullish": ":green_circle:",
            "bearish": ":red_circle:",
            "neutral": ":white_circle:",
        }.get(bias, ":grey_question:")

        text = f"{bias_emoji} *{symbol}* — {bias} ({conviction:.0%} conviction)"

        blocks = [
            {"type": "section", "text": {"type": "mrkdwn", "text": text}},
        ]

        if raw_collectors:
            collector_lines = []
            for name, data in raw_collectors.items():
                if isinstance(data, dict):
                    # Show key metrics from each collector
                    metrics = ", ".join(f"{k}={v}" for k, v in list(data.items())[:4])
                    collector_lines.append(f"• *{name}*: {metrics}")
                else:
                    collector_lines.append(f"• *{name}*: {data}")
            if collector_lines:
                blocks.append(
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": "\n".join(collector_lines[:10]),
                        },
                    }
                )

        return self.post("signals", text, blocks)

    def post_alert(
        self,
        severity: str,
        title: str,
        detail: str,
    ) -> str | None:
        """Post an alert to #alerts."""
        severity_config = {
            "critical": (":rotating_light:", "<!channel> "),
            "warning": (":warning:", ""),
            "info": (":information_source:", ""),
        }
        emoji, mention = severity_config.get(severity.lower(), (":grey_question:", ""))

        text = f"{emoji} {mention}*[{severity.upper()}]* {title}\n{detail}"
        return self.post("alerts", text)

    def post_strategy_event(
        self,
        event_type: str,
        strategy_name: str,
        detail: str,
    ) -> str | None:
        """Post strategy lifecycle event to #strategies."""
        emoji_map = {
            "promoted": ":arrow_up:",
            "retired": ":coffin:",
            "demoted": ":arrow_down:",
            "registered": ":new:",
        }
        emoji = emoji_map.get(event_type, ":memo:")
        text = f"{emoji} *{strategy_name}* — {event_type}\n{detail}"
        return self.post("strategies", text)

    def post_system(self, text: str) -> str | None:
        """Post to #system channel."""
        return self.post("system", text)

    @staticmethod
    def _truncate(text: str, max_len: int) -> str:
        """Truncate text for Slack's 3000-char block limit."""
        if len(text) <= max_len:
            return text
        return text[: max_len - 20] + "\n... _(truncated)_"
