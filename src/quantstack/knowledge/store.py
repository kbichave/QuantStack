# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
PostgreSQL-backed knowledge store for trade journal and agent state.

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

from loguru import logger

from quantstack.db import PgConnection, open_db, run_migrations
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
    PostgreSQL-backed storage for QuantPod knowledge.

    Provides CRUD operations for all knowledge types with
    automatic schema management and JSON serialization.

    Usage:
        store = KnowledgeStore(conn)

        # Store a trade
        trade = TradeRecord(symbol="SPY", ...)
        trade_id = store.save_trade(trade)

        # Query trades
        trades = store.get_trades(symbol="SPY", status=TradeStatus.OPEN)

        # Get performance metrics
        metrics = store.get_agent_performance("executor", days=30)
    """

    def __init__(self, conn: PgConnection | None = None):
        """
        Initialize the knowledge store.

        Args:
            conn: PostgreSQL connection. If None, a connection is opened from
                  the shared pool and migrations run automatically.
        """
        if conn is not None:
            self.conn: PgConnection = conn
        else:
            self.conn = open_db()
            run_migrations(self.conn)

        # Schema is managed by run_migrations; _init_schema is a no-op
        # safety net kept for standalone construction outside the MCP server.
        self._init_schema()

        logger.info("KnowledgeStore initialized (PostgreSQL)")

    def close(self) -> None:
        """Release the connection back to the pool."""
        self.conn.release()
