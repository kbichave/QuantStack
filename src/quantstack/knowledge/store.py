# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
DuckDB-based knowledge store for trade journal and agent state.

Provides persistent storage for:
- Trade records and journal
- Market observations
- Wave scenarios
- Regime states
- Agent messages
- Performance metrics

The KnowledgeStore class composes domain-specific mixins:
- SchemaMixin      — DDL (CREATE TABLE / INDEX / SEQUENCE)
- TradesMixin      — Trade journal, observations, signals
- WavesRegimeMixin — Wave scenarios and regime states
- MessagesMixin    — Agent message bus
- PerformanceMixin — Performance metrics
- LearningMixin    — Historical arena, lessons, A/B tests, portfolios
"""

import os
from pathlib import Path

import duckdb
from loguru import logger

from quantstack.knowledge._learning import LearningMixin
from quantstack.knowledge._messages import MessagesMixin
from quantstack.knowledge._performance import PerformanceMixin
from quantstack.knowledge._schema import SchemaMixin
from quantstack.knowledge._trades import TradesMixin
from quantstack.knowledge._waves_regime import WavesRegimeMixin


# =============================================================================
# KNOWLEDGE STORE CLASS
# =============================================================================


class KnowledgeStore(
    SchemaMixin,
    TradesMixin,
    WavesRegimeMixin,
    MessagesMixin,
    PerformanceMixin,
    LearningMixin,
):
    """
    DuckDB-based storage for QuantPod knowledge.

    Provides CRUD operations for all knowledge types with
    automatic schema management and JSON serialization.

    Usage:
        store = KnowledgeStore()

        # Store a trade
        trade = TradeRecord(symbol="SPY", ...)
        trade_id = store.save_trade(trade)

        # Query trades
        trades = store.get_trades(symbol="SPY", status=TradeStatus.OPEN)

        # Get performance metrics
        metrics = store.get_agent_performance("executor", days=30)
    """

    def __init__(self, db_path: str | None = None, read_only: bool = False):
        """
        Initialize the knowledge store.

        Args:
            db_path: Path to DuckDB file. Defaults to ~/.quant_pod/knowledge.duckdb
            read_only: Open the database in read-only mode (no schema init)
        """
        if db_path is None:
            db_path = os.getenv("DUCKDB_PATH", "~/.quant_pod/knowledge.duckdb")

        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # DuckDB cannot open a zero-byte placeholder file. When tests create a
        # temporary file ahead of time, remove the empty stub so DuckDB can
        # initialize a fresh database.
        if not read_only and self.db_path.exists() and self.db_path.stat().st_size == 0:
            self.db_path.unlink()
        self.read_only = read_only

        self._conn: duckdb.DuckDBPyConnection | None = None

        # Only initialize schema when writable; read-only consumers (frontend)
        # should not attempt to mutate or create the DB.
        if not self.read_only:
            self._init_schema()

        logger.info(f"KnowledgeStore initialized at {self.db_path}")

    @property
    def conn(self) -> duckdb.DuckDBPyConnection:
        """Get database connection, creating if needed."""
        if self._conn is None:
            # When read_only is True, avoid creating WAL/locks to allow
            # simultaneous writer (simulation) and reader (UI).
            if self.read_only and not self.db_path.exists():
                raise FileNotFoundError(f"Knowledge DB not found: {self.db_path}")
            self._conn = duckdb.connect(str(self.db_path), read_only=self.read_only)
        return self._conn

    def close(self) -> None:
        """Close database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
