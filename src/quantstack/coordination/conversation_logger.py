# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Conversation logger — persists agent interactions to PostgreSQL and posts to Slack.

Every desk agent report, PM decision, signal scan, trade, and alert flows
through this module. It handles both sides:
  1. DuckDB INSERT for history, debugging, and optimization analysis
  2. Slack post for real-time monitoring

The /reflect skill and Slack MCP server can later read these conversations
back to identify prompt flaws, agent biases, and optimization opportunities.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any

from loguru import logger

from quantstack.db import PgConnection

from quantstack.coordination.slack_client import SlackClient


class ConversationLogger:
    """
    Unified logging for agent conversations, signals, trades, and alerts.

    Args:
        conn: PostgreSQL connection.
        slack: SlackClient instance. Pass None to disable Slack posting.
        session_id: Current session ID for grouping conversations.
    """

    def __init__(
        self,
        conn: PgConnection,
        slack: SlackClient | None = None,
        session_id: str = "",
    ) -> None:
        self._conn = conn
        self._slack = slack or SlackClient()
        self._session_id = session_id or uuid.uuid4().hex[:12]

    def log_agent_report(
        self,
        agent_name: str,
        symbol: str | None,
        content: str,
        summary: str = "",
        strategy_id: str | None = None,
        iteration: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Log a desk agent report to DuckDB and post to Slack #agent-activity.

        Args:
            agent_name: Agent identifier (market_intel, risk, etc.).
            symbol: Ticker symbol the report is about.
            content: Full report text from the agent.
            summary: 1-line summary for Slack compact view.
            strategy_id: Strategy this report relates to.
            iteration: Loop iteration number.
            metadata: Extra context (model, tokens, duration_ms).

        Returns:
            conversation_id for threading.
        """
        cid = uuid.uuid4().hex[:12]
        if not summary:
            summary = content[:150] + "..." if len(content) > 150 else content

        try:
            self._conn.execute(
                """
                INSERT INTO agent_conversations
                    (conversation_id, session_id, loop_name, iteration,
                     agent_name, role, symbol, strategy_id, content, summary,
                     created_at, metadata)
                VALUES (?, ?, 'trading_operator', ?, ?, 'desk_agent', ?, ?, ?, ?, ?, ?)
                """,
                [
                    cid,
                    self._session_id,
                    iteration,
                    agent_name,
                    symbol,
                    strategy_id,
                    content,
                    summary,
                    datetime.now(),
                    json.dumps(metadata or {}),
                ],
            )
        except Exception as exc:
            logger.debug(f"[ConvLogger] DB insert failed: {exc}")

        self._slack.post_agent_report(agent_name, symbol, summary, content)

        return cid

    def log_pm_decision(
        self,
        symbol: str | None,
        decision: str,
        reasoning: str,
        confidence: float = 0.0,
        iteration: int | None = None,
    ) -> str:
        """Log a PM (Trading Operator) decision."""
        cid = uuid.uuid4().hex[:12]
        content = f"Decision: {decision}\nConfidence: {confidence:.0%}\nReasoning: {reasoning}"
        summary = f"{decision} ({confidence:.0%})"

        try:
            self._conn.execute(
                """
                INSERT INTO agent_conversations
                    (conversation_id, session_id, loop_name, iteration,
                     agent_name, role, symbol, content, summary, created_at, metadata)
                VALUES (?, ?, 'trading_operator', ?, 'pm', 'pm_decision', ?, ?, ?, ?, '{}')
                """,
                [
                    cid,
                    self._session_id,
                    iteration,
                    symbol,
                    content,
                    summary,
                    datetime.now(),
                ],
            )
        except Exception as exc:
            logger.debug(f"[ConvLogger] DB insert failed: {exc}")

        self._slack.post_agent_report("pm", symbol, summary, content)
        return cid

    def log_signal_snapshot(
        self,
        symbol: str,
        collectors: dict[str, Any],
        bias: str = "neutral",
        conviction: float = 0.0,
        failures: list[str] | None = None,
    ) -> str:
        """
        Log raw SignalEngine collector outputs to DuckDB and post summary to Slack.

        Args:
            symbol: Ticker symbol.
            collectors: Dict of {collector_name: raw_output_dict}.
            bias: Consensus bias (bullish/bearish/neutral).
            conviction: Consensus conviction (0-1).
            failures: List of collector names that failed.

        Returns:
            snapshot_id.
        """
        sid = uuid.uuid4().hex[:12]

        try:
            self._conn.execute(
                """
                INSERT INTO signal_snapshots
                    (snapshot_id, symbol, created_at,
                     technical, regime, volume, risk, sentiment,
                     fundamentals, events,
                     consensus_bias, consensus_conviction, collector_failures)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    sid,
                    symbol,
                    datetime.now(),
                    json.dumps(collectors.get("technical", {})),
                    json.dumps(collectors.get("regime", {})),
                    json.dumps(collectors.get("volume", {})),
                    json.dumps(collectors.get("risk", {})),
                    json.dumps(collectors.get("sentiment", {})),
                    json.dumps(collectors.get("fundamentals", {})),
                    json.dumps(collectors.get("events", {})),
                    bias,
                    conviction,
                    json.dumps(failures or []),
                ],
            )
        except Exception as exc:
            logger.debug(f"[ConvLogger] Signal snapshot insert failed: {exc}")

        self._slack.post_signal_brief(symbol, bias, conviction, collectors)
        return sid

    def log_trade(self, trade_result: dict[str, Any]) -> None:
        """Post a trade fill to Slack #trades."""
        self._slack.post_trade(trade_result)

    def log_alert(self, severity: str, title: str, detail: str) -> None:
        """Post an alert to Slack #alerts and log as agent conversation."""
        cid = uuid.uuid4().hex[:12]
        try:
            self._conn.execute(
                """
                INSERT INTO agent_conversations
                    (conversation_id, session_id, agent_name, role,
                     content, summary, created_at, metadata)
                VALUES (?, ?, 'system', 'system', ?, ?, ?, ?)
                """,
                [
                    cid,
                    self._session_id,
                    f"[{severity.upper()}] {title}\n{detail}",
                    f"[{severity.upper()}] {title}",
                    datetime.now(),
                    json.dumps({"severity": severity}),
                ],
            )
        except Exception as exc:
            logger.debug(f"[ConvLogger] Alert insert failed: {exc}")

        self._slack.post_alert(severity, title, detail)

    def log_portfolio_summary(self, snapshot: dict[str, Any]) -> None:
        """Post portfolio summary to Slack #portfolio."""
        self._slack.post_portfolio_summary(snapshot)

    def log_strategy_event(
        self, event_type: str, strategy_name: str, detail: str
    ) -> None:
        """Post strategy lifecycle event to Slack #strategies."""
        self._slack.post_strategy_event(event_type, strategy_name, detail)

    def log_system(self, text: str) -> None:
        """Post to Slack #system channel."""
        self._slack.post_system(text)
