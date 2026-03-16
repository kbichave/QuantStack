# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""Trades mixin — Trade, observation, and signal CRUD for KnowledgeStore."""

import json
from datetime import datetime, timedelta

import duckdb
from loguru import logger

from quant_pod.knowledge.models import (
    MarketObservation,
    TradeRecord,
    TradeStatus,
    TradingSignal,
)


class TradesMixin:
    """Trade journal, market observations, and trading signal operations."""

    conn: duckdb.DuckDBPyConnection

    # =========================================================================
    # TRADE OPERATIONS
    # =========================================================================

    def save_trade(self, trade: TradeRecord) -> int:
        """Save a trade record, returning the ID."""
        data = trade.model_dump()
        data["legs"] = json.dumps([leg.model_dump() for leg in trade.legs])
        data["tags"] = json.dumps(trade.tags)

        if trade.id is None:
            # Insert
            cols = [k for k in data.keys() if k != "id"]
            placeholders = ", ".join(["?" for _ in cols])
            col_names = ", ".join(cols)

            result = self.conn.execute(
                f"INSERT INTO trade_journal ({col_names}) VALUES ({placeholders}) RETURNING id",
                [data[k] for k in cols],
            ).fetchone()

            trade_id = result[0]
        else:
            # Update
            trade_id = trade.id
            data["updated_at"] = datetime.now()
            cols = [k for k in data.keys() if k != "id"]
            set_clause = ", ".join([f"{k} = ?" for k in cols])

            self.conn.execute(
                f"UPDATE trade_journal SET {set_clause} WHERE id = ?",
                [data[k] for k in cols] + [trade_id],
            )

        self.conn.commit()
        return trade_id

    def get_trade(self, trade_id: int) -> TradeRecord | None:
        """Get a trade by ID."""
        result = self.conn.execute(
            "SELECT * FROM trade_journal WHERE id = ?", [trade_id]
        ).fetchone()

        if result is None:
            return None

        return self._row_to_trade(result)

    def get_trades(
        self,
        symbol: str | None = None,
        status: TradeStatus | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        limit: int = 100,
    ) -> list[TradeRecord]:
        """Query trades with filters."""
        query = "SELECT * FROM trade_journal WHERE 1=1"
        params = []

        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        if status:
            query += " AND status = ?"
            params.append(status.value)
        if start_date:
            query += " AND created_at >= ?"
            params.append(start_date)
        if end_date:
            query += " AND created_at <= ?"
            params.append(end_date)

        query += f" ORDER BY created_at DESC LIMIT {limit}"

        results = self.conn.execute(query, params).fetchall()
        return [self._row_to_trade(row) for row in results]

    def get_open_trades(self) -> list[TradeRecord]:
        """Get all open trades."""
        return self.get_trades(status=TradeStatus.OPEN)

    def _row_to_trade(self, row: tuple) -> TradeRecord:
        """Convert database row to TradeRecord."""
        cols = [desc[0] for desc in self.conn.description]
        data = dict(zip(cols, row, strict=False))

        # Parse JSON fields
        if data.get("legs"):
            data["legs"] = json.loads(data["legs"])
        if data.get("tags"):
            data["tags"] = json.loads(data["tags"])

        return TradeRecord(**data)

    def get_recent_trades(
        self,
        symbol: str | None = None,
        limit: int = 5,
    ) -> list[dict]:
        """
        Get recent trades for historical context.

        Args:
            symbol: Filter by symbol (optional)
            limit: Maximum number of trades to return

        Returns:
            List of trade dicts sorted by entry_date descending
        """
        query = """
            SELECT symbol, side, entry_date, exit_date, entry_price, exit_price,
                   quantity, pnl, status, strategy_tag
            FROM trade_journal
            WHERE 1=1
        """
        params = []

        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)

        query += " ORDER BY entry_date DESC, id DESC LIMIT ?"
        params.append(limit)

        try:
            results = self.conn.execute(query, params).fetchall()
            cols = [
                "symbol",
                "side",
                "entry_date",
                "exit_date",
                "entry_price",
                "exit_price",
                "quantity",
                "pnl",
                "status",
                "strategy_tag",
            ]
            return [dict(zip(cols, row, strict=False)) for row in results]
        except Exception as e:
            logger.debug(f"Failed to get recent trades: {e}")
            return []

    # =========================================================================
    # OBSERVATION OPERATIONS
    # =========================================================================

    def save_observation(self, obs: MarketObservation) -> int:
        """Save a market observation."""
        data = obs.model_dump()

        cols = [k for k in data.keys() if k != "id"]
        placeholders = ", ".join(["?" for _ in cols])
        col_names = ", ".join(cols)

        result = self.conn.execute(
            f"INSERT INTO market_observations ({col_names}) VALUES ({placeholders}) RETURNING id",
            [data[k] for k in cols],
        ).fetchone()

        self.conn.commit()
        return result[0]

    def get_recent_observations(
        self,
        symbol: str | None = None,
        hours: int = 24,
        unprocessed_only: bool = False,
    ) -> list[MarketObservation]:
        """Get recent market observations."""
        query = "SELECT * FROM market_observations WHERE timestamp > ?"
        params = [datetime.now() - timedelta(hours=hours)]

        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        if unprocessed_only:
            query += " AND processed = FALSE"

        query += " ORDER BY timestamp DESC"

        results = self.conn.execute(query, params).fetchall()
        cols = [desc[0] for desc in self.conn.description]

        return [MarketObservation(**dict(zip(cols, row, strict=False))) for row in results]

    def mark_observations_processed(self, obs_ids: list[int]) -> None:
        """Mark observations as processed."""
        if not obs_ids:
            return

        placeholders = ", ".join(["?" for _ in obs_ids])
        self.conn.execute(
            f"UPDATE market_observations SET processed = TRUE WHERE id IN ({placeholders})",
            obs_ids,
        )
        self.conn.commit()

    # =========================================================================
    # TRADING SIGNAL OPERATIONS
    # =========================================================================

    def save_signal(self, signal: TradingSignal) -> str:
        """Save a trading signal."""
        import uuid

        if signal.id is None:
            signal.id = str(uuid.uuid4())[:8]

        data = signal.model_dump()
        data["observation_ids"] = json.dumps(data["observation_ids"])

        cols = list(data.keys())
        placeholders = ", ".join(["?" for _ in cols])
        col_names = ", ".join(cols)

        self.conn.execute(
            f"INSERT INTO trading_signals ({col_names}) VALUES ({placeholders})",
            [data[k] for k in cols],
        )

        self.conn.commit()
        return signal.id

    def get_active_signals(
        self,
        symbol: str | None = None,
        unprocessed_only: bool = False,
    ) -> list[TradingSignal]:
        """Get active trading signals."""
        query = "SELECT * FROM trading_signals WHERE is_active = TRUE"
        params = []

        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        if unprocessed_only:
            query += " AND processed = FALSE"

        query += " ORDER BY confidence DESC, timestamp DESC"

        results = self.conn.execute(query, params).fetchall()
        cols = [desc[0] for desc in self.conn.description]

        signals = []
        for row in results:
            data = dict(zip(cols, row, strict=False))
            if data.get("observation_ids"):
                data["observation_ids"] = json.loads(data["observation_ids"])
            signals.append(TradingSignal(**data))

        return signals
