# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Markdown Blackboard - Shared memory for all agents.

A markdown-formatted file that all agents write to and read from.
Easy for LLMs to parse and understand with clear structure.

Usage:
    from quant_pod.memory.blackboard import Blackboard

    bb = Blackboard()

    # Write an entry
    bb.write("TrendPod", "SPY", "Detected strong uptrend with ADX=35")

    # Read recent entries
    entries = bb.read_recent(symbol="SPY", limit=10)
"""

import os
import re
from datetime import datetime, date
from pathlib import Path
from threading import Lock
from typing import List, Optional

from loguru import logger


class BlackboardEntry:
    """A single blackboard entry in markdown format."""

    def __init__(self, timestamp: str, agent: str, symbol: str, message: str):
        self.timestamp = timestamp
        self.agent = agent
        self.symbol = symbol
        self.message = message

    def to_markdown(self) -> str:
        """Convert entry to markdown format for file storage."""
        return f"""### [{self.timestamp}] {self.agent}
**Symbol:** {self.symbol}

{self.message}

---
"""

    def __str__(self) -> str:
        """String representation for display."""
        return self.to_markdown()

    @classmethod
    def from_markdown_block(cls, block: str) -> Optional["BlackboardEntry"]:
        """
        Parse a markdown block into a BlackboardEntry.

        Expected format:
        ### [timestamp] AgentName
        **Symbol:** SYMBOL

        message content here

        ---
        """
        try:
            block = block.strip()
            if not block or not block.startswith("###"):
                return None

            lines = block.split("\n")

            # Parse header: ### [timestamp] AgentName
            header_match = re.match(r"^###\s*\[([^\]]+)\]\s*(.+)$", lines[0])
            if not header_match:
                return None

            timestamp = header_match.group(1)
            agent = header_match.group(2).strip()

            # Parse symbol line: **Symbol:** SYMBOL
            symbol = "UNKNOWN"
            message_start = 1
            for i, line in enumerate(lines[1:], 1):
                symbol_match = re.match(r"^\*\*Symbol:\*\*\s*(.+)$", line.strip())
                if symbol_match:
                    symbol = symbol_match.group(1).strip()
                    message_start = i + 1
                    break

            # Rest is the message (excluding trailing ---)
            message_lines = []
            for line in lines[message_start:]:
                if line.strip() == "---":
                    break
                message_lines.append(line)

            message = "\n".join(message_lines).strip()

            return cls(timestamp, agent, symbol, message)

        except Exception as e:
            logger.debug(f"Failed to parse markdown block: {e}")
            return None

    @classmethod
    def from_line(cls, line: str) -> Optional["BlackboardEntry"]:
        """
        Legacy parser for old format: [timestamp] agent | symbol | message
        Kept for backward compatibility.
        """
        try:
            line = line.strip()
            if not line or not line.startswith("["):
                return None

            ts_end = line.index("]")
            timestamp = line[1:ts_end]
            rest = line[ts_end + 2 :]  # Skip "] "

            parts = rest.split(" | ", 2)
            if len(parts) < 3:
                return None

            return cls(timestamp, parts[0], parts[1], parts[2])
        except Exception:
            return None


class Blackboard:
    """
    Markdown-based shared memory for agents.

    All agents can write observations, decisions, and context.
    All agents can read recent entries to understand what happened.
    Format is markdown for easy LLM parsing.
    """

    _instance = None
    _lock = Lock()

    def __new__(cls, *args, **kwargs):
        """Singleton pattern."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, path: Optional[str] = None):
        """Initialize blackboard with file path."""
        if self._initialized:
            return

        if path is None:
            path = os.path.expanduser("~/.quant_pod/blackboard.md")

        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        # Create file with markdown header if it doesn't exist
        if not self.path.exists():
            with open(self.path, "w") as f:
                f.write("# QuantPod Blackboard\n\n")
                f.write("*Shared memory for all trading agents*\n\n")
                f.write("---\n\n")

        self._write_lock = Lock()
        self._initialized = True
        logger.info(f"Blackboard initialized at {self.path}")

    def write(
        self,
        agent: str,
        symbol: str,
        message: str,
        sim_date: Optional[date] = None,
    ) -> None:
        """
        Write an entry to the blackboard in markdown format.

        Args:
            agent: Name of the agent writing
            symbol: Symbol being discussed
            message: The message/observation/decision
            sim_date: Simulation date (uses real time if not provided)
        """
        if sim_date:
            timestamp = sim_date.isoformat()
        else:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        entry = BlackboardEntry(timestamp, agent, symbol, message)

        with self._write_lock:
            with open(self.path, "a") as f:
                f.write(entry.to_markdown())

    def _parse_entries(self, content: str) -> List[BlackboardEntry]:
        """Parse all entries from file content."""
        entries = []

        # Split by markdown blocks (### headers)
        # Each entry starts with ### [timestamp]
        blocks = re.split(r"(?=^### \[)", content, flags=re.MULTILINE)

        for block in blocks:
            entry = BlackboardEntry.from_markdown_block(block)
            if entry:
                entries.append(entry)

        return entries

    def read_recent(
        self,
        symbol: Optional[str] = None,
        agent: Optional[str] = None,
        limit: int = 20,
    ) -> List[BlackboardEntry]:
        """
        Read recent entries from the blackboard.

        Args:
            symbol: Filter by symbol (optional)
            agent: Filter by agent (optional)
            limit: Maximum entries to return

        Returns:
            List of entries, most recent first
        """
        try:
            with open(self.path, "r") as f:
                content = f.read()
        except FileNotFoundError:
            return []

        # Parse all entries
        all_entries = self._parse_entries(content)

        # Filter and limit (reverse for most recent first)
        entries = []
        for entry in reversed(all_entries):
            # Apply filters
            if symbol and entry.symbol != symbol:
                continue
            if agent and entry.agent != agent:
                continue

            entries.append(entry)
            if len(entries) >= limit:
                break

        return entries

    def read_as_context(
        self,
        symbol: str,
        limit: int = 10,
    ) -> str:
        """
        Read recent entries formatted as markdown context for LLM.

        Args:
            symbol: Symbol to get context for
            limit: Maximum entries

        Returns:
            Markdown-formatted context string
        """
        entries = self.read_recent(symbol=symbol, limit=limit)

        if not entries:
            return "## Recent History\n\n*No recent history available for this symbol.*"

        lines = [
            f"## Recent History for {symbol}",
            "",
            f"*Last {len(entries)} entries:*",
            "",
        ]

        for entry in entries:
            lines.append(f"### [{entry.timestamp}] {entry.agent}")
            lines.append(f"")
            lines.append(f"{entry.message}")
            lines.append("")
            lines.append("---")
            lines.append("")

        return "\n".join(lines)

    def read_all_as_markdown(self, limit: int = 50) -> str:
        """
        Read all recent entries as a single markdown document.

        Args:
            limit: Maximum entries to include

        Returns:
            Complete markdown document
        """
        entries = self.read_recent(limit=limit)

        if not entries:
            return "## Trading Blackboard\n\n*No entries yet.*"

        lines = [
            "## Trading Blackboard",
            "",
            f"*Showing {len(entries)} most recent entries*",
            "",
        ]

        for entry in entries:
            lines.append(entry.to_markdown())

        return "\n".join(lines)

    def clear(self) -> None:
        """Clear the blackboard (for testing or new simulation runs)."""
        with self._write_lock:
            with open(self.path, "w") as f:
                f.write("# QuantPod Blackboard\n\n")
                f.write("*Shared memory for all trading agents*\n\n")
                f.write("---\n\n")
        logger.info("Blackboard cleared")

    def clear_before_date(self, cutoff_date: date) -> int:
        """
        Remove entries before a certain date.

        Args:
            cutoff_date: Remove entries before this date

        Returns:
            Number of entries removed
        """
        try:
            with open(self.path, "r") as f:
                content = f.read()
        except FileNotFoundError:
            return 0

        all_entries = self._parse_entries(content)
        cutoff_str = cutoff_date.isoformat()

        kept = []
        removed = 0

        for entry in all_entries:
            if entry.timestamp < cutoff_str:
                removed += 1
            else:
                kept.append(entry)

        # Rewrite file with kept entries
        with self._write_lock:
            with open(self.path, "w") as f:
                f.write("# QuantPod Blackboard\n\n")
                f.write("*Shared memory for all trading agents*\n\n")
                f.write("---\n\n")
                for entry in kept:
                    f.write(entry.to_markdown())

        return removed


# Convenience functions
_blackboard = None


def get_blackboard() -> Blackboard:
    """Get the singleton blackboard instance."""
    global _blackboard
    if _blackboard is None:
        _blackboard = Blackboard()
    return _blackboard


def write_to_blackboard(
    agent: str, symbol: str, message: str, sim_date: Optional[date] = None
) -> None:
    """Write to the blackboard."""
    get_blackboard().write(agent, symbol, message, sim_date)


def read_blackboard_context(symbol: str, limit: int = 10) -> str:
    """Read blackboard context for a symbol as markdown."""
    return get_blackboard().read_as_context(symbol, limit)
