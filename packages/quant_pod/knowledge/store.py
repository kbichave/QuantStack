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
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar

import duckdb
from loguru import logger
from pydantic import BaseModel

from quant_pod.knowledge.models import (
    AgentMessage,
    MarketObservation,
    PerformanceMetrics,
    RegimeState,
    TradeRecord,
    TradeStatus,
    TradingSignal,
    WaveScenario,
)


T = TypeVar("T", bound=BaseModel)


# =============================================================================
# KNOWLEDGE STORE CLASS
# =============================================================================


class KnowledgeStore:
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

    def __init__(self, db_path: Optional[str] = None, read_only: bool = False):
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

        self._conn: Optional[duckdb.DuckDBPyConnection] = None

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

    def _init_schema(self) -> None:
        """Initialize database schema."""
        # Trade journal
        self.conn.execute("CREATE SEQUENCE IF NOT EXISTS seq_trade_journal START 1")
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS trade_journal (
                id BIGINT PRIMARY KEY DEFAULT nextval('seq_trade_journal'),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                symbol VARCHAR NOT NULL,
                direction VARCHAR NOT NULL,
                structure_type VARCHAR NOT NULL,
                status VARCHAR DEFAULT 'PENDING',
                legs JSON,
                entry_price DOUBLE,
                exit_price DOUBLE,
                quantity INTEGER DEFAULT 1,
                pnl DOUBLE,
                pnl_pct DOUBLE,
                max_profit_potential DOUBLE,
                max_loss_potential DOUBLE,
                wave_scenario_id VARCHAR,
                regime_at_entry VARCHAR,
                volatility_at_entry DOUBLE,
                confidence_score DOUBLE DEFAULT 0.5,
                agent_rationale TEXT,
                research_score DOUBLE,
                arena_rank INTEGER,
                outcome_correct BOOLEAN,
                lessons_learned TEXT,
                tags JSON,
                entry_order_id VARCHAR,
                exit_order_id VARCHAR
            )
        """
        )

        # Market observations
        self.conn.execute(
            "CREATE SEQUENCE IF NOT EXISTS seq_market_observations START 1"
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS market_observations (
                id BIGINT PRIMARY KEY DEFAULT nextval('seq_market_observations'),
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                symbol VARCHAR NOT NULL,
                observation_type VARCHAR NOT NULL,
                current_price DOUBLE,
                price_change_pct DOUBLE,
                volume BIGINT,
                volume_ratio DOUBLE,
                support_level DOUBLE,
                resistance_level DOUBLE,
                alert_message TEXT,
                severity VARCHAR DEFAULT 'INFO',
                source_agent VARCHAR,
                processed BOOLEAN DEFAULT FALSE
            )
        """
        )

        # Wave scenarios
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS wave_scenarios (
                id VARCHAR PRIMARY KEY,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                symbol VARCHAR NOT NULL,
                timeframe VARCHAR NOT NULL,
                wave_position VARCHAR,
                wave_degree VARCHAR,
                confidence DOUBLE DEFAULT 0.5,
                primary_target DOUBLE,
                secondary_target DOUBLE,
                invalidation_level DOUBLE NOT NULL,
                scenario_type VARCHAR,
                description TEXT,
                is_active BOOLEAN DEFAULT TRUE,
                invalidated_at TIMESTAMP,
                target_hit_at TIMESTAMP,
                source_agent VARCHAR
            )
        """
        )

        # Regime states
        self.conn.execute("CREATE SEQUENCE IF NOT EXISTS seq_regime_states START 1")
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS regime_states (
                id BIGINT PRIMARY KEY DEFAULT nextval('seq_regime_states'),
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                symbol VARCHAR NOT NULL,
                timeframe VARCHAR NOT NULL,
                trend_regime VARCHAR,
                volatility_regime VARCHAR,
                atr DOUBLE,
                atr_percentile DOUBLE,
                adx DOUBLE,
                correlation_to_spy DOUBLE,
                regime_changed BOOLEAN DEFAULT FALSE,
                previous_trend_regime VARCHAR,
                previous_volatility_regime VARCHAR,
                confidence DOUBLE DEFAULT 0.5,
                source_agent VARCHAR
            )
        """
        )

        # Agent messages
        self.conn.execute("CREATE SEQUENCE IF NOT EXISTS seq_agent_messages START 1")
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_messages (
                id BIGINT PRIMARY KEY DEFAULT nextval('seq_agent_messages'),
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                from_agent VARCHAR NOT NULL,
                to_agent VARCHAR,
                message_type VARCHAR NOT NULL,
                subject VARCHAR,
                content TEXT,
                data JSON,
                priority INTEGER DEFAULT 5,
                requires_response BOOLEAN DEFAULT FALSE,
                response_deadline TIMESTAMP,
                acknowledged BOOLEAN DEFAULT FALSE,
                acknowledged_at TIMESTAMP
            )
        """
        )

        # Trading signals
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS trading_signals (
                id VARCHAR PRIMARY KEY,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                symbol VARCHAR NOT NULL,
                direction VARCHAR NOT NULL,
                signal_type VARCHAR NOT NULL,
                strength DOUBLE DEFAULT 0.5,
                confidence DOUBLE DEFAULT 0.5,
                entry_price DOUBLE,
                target_price DOUBLE,
                stop_loss DOUBLE,
                wave_scenario_id VARCHAR,
                regime_state_id INTEGER,
                observation_ids JSON,
                rationale TEXT,
                is_active BOOLEAN DEFAULT TRUE,
                processed BOOLEAN DEFAULT FALSE,
                trade_id INTEGER,
                source_agent VARCHAR
            )
        """
        )

        # Performance metrics
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                entity_type VARCHAR NOT NULL,
                entity_name VARCHAR NOT NULL,
                total_trades INTEGER DEFAULT 0,
                winning_trades INTEGER DEFAULT 0,
                losing_trades INTEGER DEFAULT 0,
                win_rate DOUBLE DEFAULT 0.0,
                total_pnl DOUBLE DEFAULT 0.0,
                avg_pnl_per_trade DOUBLE DEFAULT 0.0,
                largest_win DOUBLE DEFAULT 0.0,
                largest_loss DOUBLE DEFAULT 0.0,
                sharpe_ratio DOUBLE,
                sortino_ratio DOUBLE,
                max_drawdown DOUBLE,
                profit_factor DOUBLE,
                period_start TIMESTAMP,
                period_end TIMESTAMP
            )
        """
        )

        # Create indexes
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trade_journal(symbol)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_trades_status ON trade_journal(status)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_obs_symbol ON market_observations(symbol)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_waves_symbol ON wave_scenarios(symbol)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_regime_symbol ON regime_states(symbol)"
        )

        # =====================================================================
        # HISTORICAL ALPHA ARENA TABLES
        # =====================================================================

        # Daily portfolio state (for equity curve)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS daily_state (
                date DATE PRIMARY KEY,
                equity DOUBLE NOT NULL,
                cash DOUBLE NOT NULL,
                max_drawdown DOUBLE DEFAULT 0,
                exposures JSON,
                regime_summary VARCHAR
            )
        """
        )

        # Historical signals from strategy pods
        # Use SEQUENCE for auto-increment in DuckDB
        self.conn.execute(
            "CREATE SEQUENCE IF NOT EXISTS seq_historical_signals START 1"
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS historical_signals (
                id INTEGER PRIMARY KEY DEFAULT nextval('seq_historical_signals'),
                date DATE NOT NULL,
                symbol VARCHAR NOT NULL,
                agent VARCHAR NOT NULL,
                signal_type VARCHAR NOT NULL,
                confidence DOUBLE DEFAULT 0.5,
                regime VARCHAR,
                structural_label VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Policy evolution snapshots
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS policy_snapshots (
                effective_date DATE PRIMARY KEY,
                pod_weights JSON NOT NULL,
                thresholds JSON,
                comment VARCHAR
            )
        """
        )

        # Agent logs for chat timeline (KEY TABLE)
        # Message field is TEXT to allow full reasoning chains
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_logs (
                log_id INTEGER,
                date DATE NOT NULL,
                agent_name VARCHAR NOT NULL,
                symbol VARCHAR,
                message TEXT NOT NULL,
                role VARCHAR DEFAULT 'analysis',
                context_id VARCHAR,
                created_at_sim_time VARCHAR
            )
        """
        )

        # Indexes for historical tables
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_daily_state_date ON daily_state(date)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_hist_signals_date ON historical_signals(date)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_hist_signals_symbol ON historical_signals(symbol)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_agent_logs_date ON agent_logs(date)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_agent_logs_symbol ON agent_logs(symbol)"
        )

        # =====================================================================
        # AGENTIC LEARNING SYSTEM TABLES
        # =====================================================================

        # Lessons learned from trade outcomes (ReflectionAgent)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_lessons (
                lesson_id TEXT PRIMARY KEY,
                lesson_text TEXT NOT NULL,
                applies_to JSON,
                confidence REAL DEFAULT 0.5,
                source_trade_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_by TEXT DEFAULT 'ReflectionAgent',
                usage_count INT DEFAULT 0,
                effectiveness_score REAL
            )
        """
        )

        # Prompt modification proposals (PromptTunerAgent)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS prompt_proposals (
                proposal_id TEXT PRIMARY KEY,
                agent_id TEXT NOT NULL,
                section TEXT NOT NULL,
                old_text TEXT,
                new_text TEXT NOT NULL,
                reason TEXT,
                status TEXT DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_by TEXT DEFAULT 'PromptTunerAgent'
            )
        """
        )

        # A/B tests for prompt changes
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ab_tests (
                test_id TEXT PRIMARY KEY,
                agent_id TEXT NOT NULL,
                control_version TEXT,
                treatment_version TEXT,
                traffic_pct REAL DEFAULT 0.2,
                status TEXT DEFAULT 'running',
                started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                ended_at TIMESTAMP,
                control_trades INT DEFAULT 0,
                control_wins INT DEFAULT 0,
                treatment_trades INT DEFAULT 0,
                treatment_wins INT DEFAULT 0
            )
        """
        )

        # Active lesson injections
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS lesson_injections (
                agent_id TEXT,
                lesson_id TEXT,
                injected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_active BOOLEAN DEFAULT TRUE,
                PRIMARY KEY (agent_id, lesson_id)
            )
        """
        )

        # Learning metrics for tracking improvement over time
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS learning_metrics (
                checkpoint_id INTEGER PRIMARY KEY,
                date DATE NOT NULL,
                trade_count INTEGER NOT NULL,
                rolling_win_rate REAL,
                cumulative_win_rate REAL,
                rolling_pnl REAL,
                cumulative_pnl REAL,
                lessons_active INTEGER DEFAULT 0,
                prompt_changes INTEGER DEFAULT 0,
                strategy_weights JSON,
                avg_confidence REAL,
                regime_at_checkpoint VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Indexes for learning tables
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_lessons_created ON agent_lessons(created_at)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_proposals_agent ON prompt_proposals(agent_id)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_proposals_status ON prompt_proposals(status)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ab_tests_agent ON ab_tests(agent_id)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ab_tests_status ON ab_tests(status)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_injections_active ON lesson_injections(is_active)"
        )

        self.conn.commit()

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

    def get_trade(self, trade_id: int) -> Optional[TradeRecord]:
        """Get a trade by ID."""
        result = self.conn.execute(
            "SELECT * FROM trade_journal WHERE id = ?", [trade_id]
        ).fetchone()

        if result is None:
            return None

        return self._row_to_trade(result)

    def get_trades(
        self,
        symbol: Optional[str] = None,
        status: Optional[TradeStatus] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[TradeRecord]:
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

    def get_open_trades(self) -> List[TradeRecord]:
        """Get all open trades."""
        return self.get_trades(status=TradeStatus.OPEN)

    def _row_to_trade(self, row: tuple) -> TradeRecord:
        """Convert database row to TradeRecord."""
        cols = [desc[0] for desc in self.conn.description]
        data = dict(zip(cols, row))

        # Parse JSON fields
        if data.get("legs"):
            data["legs"] = json.loads(data["legs"])
        if data.get("tags"):
            data["tags"] = json.loads(data["tags"])

        return TradeRecord(**data)

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
        symbol: Optional[str] = None,
        hours: int = 24,
        unprocessed_only: bool = False,
    ) -> List[MarketObservation]:
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

        return [MarketObservation(**dict(zip(cols, row))) for row in results]

    def mark_observations_processed(self, obs_ids: List[int]) -> None:
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
    # WAVE SCENARIO OPERATIONS
    # =========================================================================

    def save_wave_scenario(self, scenario: WaveScenario) -> str:
        """Save a wave scenario."""
        import uuid

        if scenario.id is None:
            scenario.id = str(uuid.uuid4())[:8]

        data = scenario.model_dump()

        # Check if exists
        existing = self.conn.execute(
            "SELECT id FROM wave_scenarios WHERE id = ?", [scenario.id]
        ).fetchone()

        if existing:
            # Update
            cols = [k for k in data.keys() if k != "id"]
            set_clause = ", ".join([f"{k} = ?" for k in cols])
            self.conn.execute(
                f"UPDATE wave_scenarios SET {set_clause} WHERE id = ?",
                [data[k] for k in cols] + [scenario.id],
            )
        else:
            # Insert
            cols = list(data.keys())
            placeholders = ", ".join(["?" for _ in cols])
            col_names = ", ".join(cols)
            self.conn.execute(
                f"INSERT INTO wave_scenarios ({col_names}) VALUES ({placeholders})",
                [data[k] for k in cols],
            )

        self.conn.commit()
        return scenario.id

    def get_active_wave_scenarios(
        self,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
    ) -> List[WaveScenario]:
        """Get active wave scenarios."""
        query = "SELECT * FROM wave_scenarios WHERE is_active = TRUE"
        params = []

        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        if timeframe:
            query += " AND timeframe = ?"
            params.append(timeframe)

        query += " ORDER BY confidence DESC"

        results = self.conn.execute(query, params).fetchall()
        cols = [desc[0] for desc in self.conn.description]

        return [WaveScenario(**dict(zip(cols, row))) for row in results]

    def invalidate_wave_scenario(self, scenario_id: str) -> None:
        """Mark a wave scenario as invalidated."""
        self.conn.execute(
            "UPDATE wave_scenarios SET is_active = FALSE, invalidated_at = ? WHERE id = ?",
            [datetime.now(), scenario_id],
        )
        self.conn.commit()

    # =========================================================================
    # REGIME STATE OPERATIONS
    # =========================================================================

    def save_regime_state(self, state: RegimeState) -> int:
        """Save a regime state."""
        data = state.model_dump()

        cols = [k for k in data.keys() if k != "id"]
        placeholders = ", ".join(["?" for _ in cols])
        col_names = ", ".join(cols)

        result = self.conn.execute(
            f"INSERT INTO regime_states ({col_names}) VALUES ({placeholders}) RETURNING id",
            [data[k] for k in cols],
        ).fetchone()

        self.conn.commit()
        return result[0]

    def get_current_regime(
        self,
        symbol: str,
        timeframe: str = "daily",
    ) -> Optional[RegimeState]:
        """Get most recent regime state for symbol."""
        result = self.conn.execute(
            """SELECT * FROM regime_states 
               WHERE symbol = ? AND timeframe = ?
               ORDER BY timestamp DESC LIMIT 1""",
            [symbol, timeframe],
        ).fetchone()

        if result is None:
            return None

        cols = [desc[0] for desc in self.conn.description]
        return RegimeState(**dict(zip(cols, result)))

    # =========================================================================
    # AGENT MESSAGE OPERATIONS
    # =========================================================================

    def send_message(self, message: AgentMessage) -> int:
        """Send an agent message."""
        data = message.model_dump()
        data["data"] = json.dumps(data["data"])

        cols = [k for k in data.keys() if k != "id"]
        placeholders = ", ".join(["?" for _ in cols])
        col_names = ", ".join(cols)

        result = self.conn.execute(
            f"INSERT INTO agent_messages ({col_names}) VALUES ({placeholders}) RETURNING id",
            [data[k] for k in cols],
        ).fetchone()

        self.conn.commit()
        return result[0]

    def get_messages(
        self,
        to_agent: Optional[str] = None,
        unacknowledged_only: bool = False,
        hours: int = 24,
    ) -> List[AgentMessage]:
        """Get messages for an agent."""
        query = "SELECT * FROM agent_messages WHERE timestamp > ?"
        params = [datetime.now() - timedelta(hours=hours)]

        if to_agent:
            query += " AND (to_agent = ? OR to_agent IS NULL)"
            params.append(to_agent)
        if unacknowledged_only:
            query += " AND acknowledged = FALSE"

        query += " ORDER BY priority ASC, timestamp DESC"

        results = self.conn.execute(query, params).fetchall()
        cols = [desc[0] for desc in self.conn.description]

        messages = []
        for row in results:
            data = dict(zip(cols, row))
            if data.get("data"):
                data["data"] = json.loads(data["data"])
            messages.append(AgentMessage(**data))

        return messages

    def acknowledge_message(self, message_id: int) -> None:
        """Acknowledge a message."""
        self.conn.execute(
            "UPDATE agent_messages SET acknowledged = TRUE, acknowledged_at = ? WHERE id = ?",
            [datetime.now(), message_id],
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
        symbol: Optional[str] = None,
        unprocessed_only: bool = False,
    ) -> List[TradingSignal]:
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
            data = dict(zip(cols, row))
            if data.get("observation_ids"):
                data["observation_ids"] = json.loads(data["observation_ids"])
            signals.append(TradingSignal(**data))

        return signals

    # =========================================================================
    # PERFORMANCE METRICS OPERATIONS
    # =========================================================================

    def save_performance_metrics(self, metrics: PerformanceMetrics) -> int:
        """Save performance metrics."""
        data = metrics.model_dump()

        cols = [k for k in data.keys()]
        placeholders = ", ".join(["?" for _ in cols])
        col_names = ", ".join(cols)

        result = self.conn.execute(
            f"INSERT INTO performance_metrics ({col_names}) VALUES ({placeholders}) RETURNING id",
            [data[k] for k in cols],
        ).fetchone()

        self.conn.commit()
        return result[0]

    def get_agent_performance(
        self,
        agent_name: str,
        days: int = 30,
    ) -> Optional[PerformanceMetrics]:
        """Get recent performance metrics for an agent."""
        result = self.conn.execute(
            """SELECT * FROM performance_metrics 
               WHERE entity_type = 'AGENT' AND entity_name = ?
               AND timestamp > ?
               ORDER BY timestamp DESC LIMIT 1""",
            [agent_name, datetime.now() - timedelta(days=days)],
        ).fetchone()

        if result is None:
            return None

        cols = [desc[0] for desc in self.conn.description]
        return PerformanceMetrics(**dict(zip(cols, result)))

    def get_structure_performance(
        self, days: int = 90
    ) -> Dict[str, PerformanceMetrics]:
        """Get performance by structure type."""
        results = self.conn.execute(
            """SELECT * FROM performance_metrics 
               WHERE entity_type = 'STRUCTURE'
               AND timestamp > ?
               ORDER BY entity_name, timestamp DESC""",
            [datetime.now() - timedelta(days=days)],
        ).fetchall()

        cols = [desc[0] for desc in self.conn.description]

        # Get most recent for each structure
        metrics = {}
        for row in results:
            data = dict(zip(cols, row))
            name = data["entity_name"]
            if name not in metrics:
                metrics[name] = PerformanceMetrics(**data)

        return metrics

    # =========================================================================
    # HISTORICAL ALPHA ARENA OPERATIONS
    # =========================================================================

    def save_daily_state(self, state: Dict) -> None:
        """
        Save daily portfolio state.

        Args:
            state: Dict with date, equity, cash, max_drawdown, exposures, regime_summary
        """
        date_val = state.get("date")
        if hasattr(date_val, "isoformat"):
            date_val = date_val.isoformat()

        exposures = state.get("exposures", {})
        if isinstance(exposures, dict):
            exposures = json.dumps(exposures)

        # Upsert
        self.conn.execute(
            """
            INSERT INTO daily_state (date, equity, cash, max_drawdown, exposures, regime_summary)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT (date) DO UPDATE SET
                equity = excluded.equity,
                cash = excluded.cash,
                max_drawdown = excluded.max_drawdown,
                exposures = excluded.exposures,
                regime_summary = excluded.regime_summary
        """,
            [
                date_val,
                state.get("equity", 0),
                state.get("cash", 0),
                state.get("max_drawdown", 0),
                exposures,
                state.get("regime_summary", ""),
            ],
        )
        self.conn.commit()

    def save_historical_signal(self, signal: Dict) -> int:
        """
        Save a historical signal (dict-based, used by historical engine).

        Args:
            signal: Dict with symbol, agent, signal_type, confidence, regime, etc.

        Returns:
            Signal ID
        """
        date_val = signal.get("date")
        if hasattr(date_val, "isoformat"):
            date_val = date_val.isoformat()

        result = self.conn.execute(
            """
            INSERT INTO historical_signals 
            (date, symbol, agent, signal_type, confidence, regime, structural_label)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            RETURNING id
        """,
            [
                date_val,
                signal.get("symbol", ""),
                signal.get("agent", "unknown"),
                signal.get("signal_type", "flat"),
                signal.get("confidence", 0.5),
                signal.get("regime"),
                signal.get("structural_label"),
            ],
        ).fetchone()

        self.conn.commit()
        return result[0]

    def save_policy_snapshot(self, snapshot: Dict) -> None:
        """
        Save a policy snapshot.

        Args:
            snapshot: Dict with effective_date, pod_weights, thresholds, comment
        """
        date_val = snapshot.get("effective_date")
        if hasattr(date_val, "isoformat"):
            date_val = date_val.isoformat()

        pod_weights = snapshot.get("pod_weights", {})
        if isinstance(pod_weights, dict):
            pod_weights = json.dumps(pod_weights)

        thresholds = snapshot.get("thresholds", {})
        if isinstance(thresholds, dict):
            thresholds = json.dumps(thresholds)

        # Upsert
        self.conn.execute(
            """
            INSERT INTO policy_snapshots (effective_date, pod_weights, thresholds, comment)
            VALUES (?, ?, ?, ?)
            ON CONFLICT (effective_date) DO UPDATE SET
                pod_weights = excluded.pod_weights,
                thresholds = excluded.thresholds,
                comment = excluded.comment
        """,
            [
                date_val,
                pod_weights,
                thresholds,
                snapshot.get("comment", ""),
            ],
        )
        self.conn.commit()

    def save_agent_log(self, log: Dict) -> int:
        """
        Save an agent log message.

        Args:
            log: Dict with date, agent_name, symbol, message, role, context_id, created_at_sim_time

        Returns:
            Log ID
        """
        date_val = log.get("date")
        if hasattr(date_val, "isoformat"):
            date_val = date_val.isoformat()

        # Allow full reasoning messages (no truncation)
        message = str(log.get("message", ""))

        # Get next log_id manually
        try:
            max_id = self.conn.execute(
                "SELECT COALESCE(MAX(log_id), 0) FROM agent_logs"
            ).fetchone()[0]
        except Exception:
            max_id = 0
        next_id = max_id + 1

        self.conn.execute(
            """
            INSERT INTO agent_logs 
            (log_id, date, agent_name, symbol, message, role, context_id, created_at_sim_time)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            [
                next_id,
                date_val,
                log.get("agent_name", "unknown"),
                log.get("symbol"),
                message,
                log.get("role", "analysis"),
                log.get("context_id"),
                log.get("created_at_sim_time", ""),
            ],
        )

        self.conn.commit()
        return next_id

    def load_equity_curve(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[Dict]:
        """
        Load equity curve data.

        Args:
            start_date: Start of date range
            end_date: End of date range

        Returns:
            List of daily state dicts
        """
        query = "SELECT * FROM daily_state WHERE 1=1"
        params = []

        if start_date:
            query += " AND date >= ?"
            params.append(
                start_date.isoformat()
                if hasattr(start_date, "isoformat")
                else start_date
            )
        if end_date:
            query += " AND date <= ?"
            params.append(
                end_date.isoformat() if hasattr(end_date, "isoformat") else end_date
            )

        query += " ORDER BY date ASC"

        results = self.conn.execute(query, params).fetchall()
        cols = [desc[0] for desc in self.conn.description]

        states = []
        for row in results:
            data = dict(zip(cols, row))
            if data.get("exposures"):
                try:
                    data["exposures"] = json.loads(data["exposures"])
                except:
                    data["exposures"] = {}
            states.append(data)

        return states

    def load_agent_logs(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        symbol: Optional[str] = None,
        agent_name: Optional[str] = None,
        limit: int = 1000,
    ) -> List[Dict]:
        """
        Load agent logs for chat timeline.

        Args:
            start_date: Start of date range
            end_date: End of date range
            symbol: Optional symbol filter (None = all including portfolio-level)
            agent_name: Optional agent filter
            limit: Maximum number of logs to return

        Returns:
            List of agent log dicts, ordered by date and sim_time
        """
        query = "SELECT * FROM agent_logs WHERE 1=1"
        params = []

        if start_date:
            query += " AND date >= ?"
            params.append(
                start_date.isoformat()
                if hasattr(start_date, "isoformat")
                else start_date
            )
        if end_date:
            query += " AND date <= ?"
            params.append(
                end_date.isoformat() if hasattr(end_date, "isoformat") else end_date
            )
        if symbol:
            query += " AND (symbol = ? OR symbol IS NULL)"
            params.append(symbol)
        if agent_name:
            query += " AND agent_name = ?"
            params.append(agent_name)

        query += f" ORDER BY date ASC, created_at_sim_time ASC LIMIT {limit}"

        results = self.conn.execute(query, params).fetchall()
        cols = [desc[0] for desc in self.conn.description]

        return [dict(zip(cols, row)) for row in results]

    def load_historical_signals(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        symbol: Optional[str] = None,
        limit: int = 1000,
    ) -> List[Dict]:
        """
        Load historical signals.

        Args:
            start_date: Start of date range
            end_date: End of date range
            symbol: Optional symbol filter
            limit: Maximum number of signals to return

        Returns:
            List of signal dicts
        """
        query = "SELECT * FROM historical_signals WHERE 1=1"
        params = []

        if start_date:
            query += " AND date >= ?"
            params.append(
                start_date.isoformat()
                if hasattr(start_date, "isoformat")
                else start_date
            )
        if end_date:
            query += " AND date <= ?"
            params.append(
                end_date.isoformat() if hasattr(end_date, "isoformat") else end_date
            )
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)

        query += f" ORDER BY date DESC LIMIT {limit}"

        results = self.conn.execute(query, params).fetchall()
        cols = [desc[0] for desc in self.conn.description]

        return [dict(zip(cols, row)) for row in results]

    def load_policy_snapshots(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[Dict]:
        """
        Load policy snapshots.

        Args:
            start_date: Start of date range
            end_date: End of date range

        Returns:
            List of policy snapshot dicts
        """
        query = "SELECT * FROM policy_snapshots WHERE 1=1"
        params = []

        if start_date:
            query += " AND effective_date >= ?"
            params.append(
                start_date.isoformat()
                if hasattr(start_date, "isoformat")
                else start_date
            )
        if end_date:
            query += " AND effective_date <= ?"
            params.append(
                end_date.isoformat() if hasattr(end_date, "isoformat") else end_date
            )

        query += " ORDER BY effective_date ASC"

        results = self.conn.execute(query, params).fetchall()
        cols = [desc[0] for desc in self.conn.description]

        snapshots = []
        for row in results:
            data = dict(zip(cols, row))
            if data.get("pod_weights"):
                try:
                    data["pod_weights"] = json.loads(data["pod_weights"])
                except:
                    data["pod_weights"] = {}
            if data.get("thresholds"):
                try:
                    data["thresholds"] = json.loads(data["thresholds"])
                except:
                    data["thresholds"] = {}
            snapshots.append(data)

        return snapshots

    def get_latest_policy(
        self, as_of_date: Optional[datetime] = None
    ) -> Optional[Dict]:
        """
        Get the most recent policy snapshot as of a date.

        Args:
            as_of_date: Date to get policy for (default: now)

        Returns:
            Policy snapshot dict or None
        """
        if as_of_date is None:
            as_of_date = datetime.now()

        date_str = (
            as_of_date.isoformat()
            if hasattr(as_of_date, "isoformat")
            else str(as_of_date)
        )

        result = self.conn.execute(
            """
            SELECT * FROM policy_snapshots 
            WHERE effective_date <= ?
            ORDER BY effective_date DESC
            LIMIT 1
        """,
            [date_str],
        ).fetchone()

        if result is None:
            return None

        cols = [desc[0] for desc in self.conn.description]
        data = dict(zip(cols, result))

        if data.get("pod_weights"):
            try:
                data["pod_weights"] = json.loads(data["pod_weights"])
            except:
                data["pod_weights"] = {}
        if data.get("thresholds"):
            try:
                data["thresholds"] = json.loads(data["thresholds"])
            except:
                data["thresholds"] = {}

        return data

    # =========================================================================
    # AGENTIC LEARNING SYSTEM OPERATIONS
    # =========================================================================

    def save_lesson(self, lesson: Dict) -> str:
        """
        Save a lesson learned from trade outcomes.

        Args:
            lesson: Dict with lesson_id, lesson_text, applies_to, confidence, etc.

        Returns:
            lesson_id
        """
        import uuid

        lesson_id = lesson.get("lesson_id") or str(uuid.uuid4())[:12]
        applies_to = lesson.get("applies_to", [])
        if isinstance(applies_to, list):
            applies_to = json.dumps(applies_to)

        self.conn.execute(
            """
            INSERT INTO agent_lessons 
            (lesson_id, lesson_text, applies_to, confidence, source_trade_id, created_by)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT (lesson_id) DO UPDATE SET
                lesson_text = excluded.lesson_text,
                applies_to = excluded.applies_to,
                confidence = excluded.confidence
        """,
            [
                lesson_id,
                lesson.get("lesson_text", ""),
                applies_to,
                lesson.get("confidence", 0.5),
                lesson.get("source_trade_id"),
                lesson.get("created_by", "ReflectionAgent"),
            ],
        )
        self.conn.commit()
        return lesson_id

    def get_lessons(
        self,
        limit: int = 50,
        min_confidence: float = 0.0,
    ) -> List[Dict]:
        """Get all lessons, optionally filtered by confidence."""
        results = self.conn.execute(
            """
            SELECT * FROM agent_lessons 
            WHERE confidence >= ?
            ORDER BY created_at DESC 
            LIMIT ?
        """,
            [min_confidence, limit],
        ).fetchall()

        cols = [desc[0] for desc in self.conn.description]
        lessons = []
        for row in results:
            data = dict(zip(cols, row))
            if data.get("applies_to"):
                try:
                    data["applies_to"] = json.loads(data["applies_to"])
                except:
                    data["applies_to"] = []
            lessons.append(data)

        return lessons

    def get_lessons_for_agent(self, agent_id: str, limit: int = 5) -> List[Dict]:
        """Get lessons applicable to a specific agent."""
        # Query lessons where agent_id is in applies_to JSON array
        results = self.conn.execute(
            """
            SELECT * FROM agent_lessons 
            WHERE applies_to LIKE ?
            ORDER BY confidence DESC, created_at DESC 
            LIMIT ?
        """,
            [f'%"{agent_id}"%', limit],
        ).fetchall()

        cols = [desc[0] for desc in self.conn.description]
        lessons = []
        for row in results:
            data = dict(zip(cols, row))
            if data.get("applies_to"):
                try:
                    data["applies_to"] = json.loads(data["applies_to"])
                except:
                    data["applies_to"] = []
            lessons.append(data)

        return lessons

    def update_lesson_usage(
        self, lesson_id: str, was_effective: Optional[bool] = None
    ) -> None:
        """Update lesson usage count and optionally effectiveness."""
        self.conn.execute(
            """
            UPDATE agent_lessons 
            SET usage_count = usage_count + 1
            WHERE lesson_id = ?
        """,
            [lesson_id],
        )

        if was_effective is not None:
            # Update effectiveness with exponential moving average
            self.conn.execute(
                """
                UPDATE agent_lessons 
                SET effectiveness_score = COALESCE(effectiveness_score * 0.8 + ? * 0.2, ?)
                WHERE lesson_id = ?
            """,
                [
                    1.0 if was_effective else 0.0,
                    1.0 if was_effective else 0.0,
                    lesson_id,
                ],
            )

        self.conn.commit()

    def save_prompt_proposal(self, proposal: Dict) -> str:
        """
        Save a prompt modification proposal.

        Args:
            proposal: Dict with agent_id, section, old_text, new_text, reason

        Returns:
            proposal_id
        """
        import uuid

        proposal_id = proposal.get("proposal_id") or str(uuid.uuid4())[:12]

        self.conn.execute(
            """
            INSERT INTO prompt_proposals 
            (proposal_id, agent_id, section, old_text, new_text, reason, status, created_by)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            [
                proposal_id,
                proposal.get("agent_id", ""),
                proposal.get("section", "backstory"),
                proposal.get("old_text"),
                proposal.get("new_text", ""),
                proposal.get("reason"),
                proposal.get("status", "pending"),
                proposal.get("created_by", "PromptTunerAgent"),
            ],
        )
        self.conn.commit()
        return proposal_id

    def get_prompt_proposals(
        self,
        agent_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict]:
        """Get prompt proposals, optionally filtered."""
        query = "SELECT * FROM prompt_proposals WHERE 1=1"
        params = []

        if agent_id:
            query += " AND agent_id = ?"
            params.append(agent_id)
        if status:
            query += " AND status = ?"
            params.append(status)

        query += f" ORDER BY created_at DESC LIMIT {limit}"

        results = self.conn.execute(query, params).fetchall()
        cols = [desc[0] for desc in self.conn.description]

        return [dict(zip(cols, row)) for row in results]

    def update_proposal_status(self, proposal_id: str, status: str) -> None:
        """Update the status of a proposal."""
        self.conn.execute(
            """
            UPDATE prompt_proposals SET status = ? WHERE proposal_id = ?
        """,
            [status, proposal_id],
        )
        self.conn.commit()

    def create_ab_test(self, test: Dict) -> str:
        """
        Create an A/B test for a prompt change.

        Args:
            test: Dict with agent_id, control_version, treatment_version, traffic_pct

        Returns:
            test_id
        """
        import uuid

        test_id = test.get("test_id") or str(uuid.uuid4())[:12]

        self.conn.execute(
            """
            INSERT INTO ab_tests 
            (test_id, agent_id, control_version, treatment_version, traffic_pct, status)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            [
                test_id,
                test.get("agent_id", ""),
                test.get("control_version"),
                test.get("treatment_version"),
                test.get("traffic_pct", 0.2),
                test.get("status", "running"),
            ],
        )
        self.conn.commit()
        return test_id

    def get_ab_test(self, test_id: str) -> Optional[Dict]:
        """Get an A/B test by ID."""
        result = self.conn.execute(
            """
            SELECT * FROM ab_tests WHERE test_id = ?
        """,
            [test_id],
        ).fetchone()

        if result is None:
            return None

        cols = [desc[0] for desc in self.conn.description]
        return dict(zip(cols, result))

    def get_active_ab_tests(self, agent_id: Optional[str] = None) -> List[Dict]:
        """Get all running A/B tests."""
        query = "SELECT * FROM ab_tests WHERE status = 'running'"
        params = []

        if agent_id:
            query += " AND agent_id = ?"
            params.append(agent_id)

        results = self.conn.execute(query, params).fetchall()
        cols = [desc[0] for desc in self.conn.description]

        return [dict(zip(cols, row)) for row in results]

    def update_ab_test_results(
        self,
        test_id: str,
        is_treatment: bool,
        won: bool,
    ) -> None:
        """Update A/B test results after a trade."""
        if is_treatment:
            self.conn.execute(
                """
                UPDATE ab_tests 
                SET treatment_trades = treatment_trades + 1,
                    treatment_wins = treatment_wins + ?
                WHERE test_id = ?
            """,
                [1 if won else 0, test_id],
            )
        else:
            self.conn.execute(
                """
                UPDATE ab_tests 
                SET control_trades = control_trades + 1,
                    control_wins = control_wins + ?
                WHERE test_id = ?
            """,
                [1 if won else 0, test_id],
            )

        self.conn.commit()

    def end_ab_test(self, test_id: str, status: str = "completed") -> None:
        """End an A/B test."""
        self.conn.execute(
            """
            UPDATE ab_tests 
            SET status = ?, ended_at = CURRENT_TIMESTAMP
            WHERE test_id = ?
        """,
            [status, test_id],
        )
        self.conn.commit()

    def inject_lesson(self, agent_id: str, lesson_id: str) -> None:
        """Mark a lesson to be injected into an agent's context."""
        self.conn.execute(
            """
            INSERT INTO lesson_injections (agent_id, lesson_id, is_active)
            VALUES (?, ?, TRUE)
            ON CONFLICT (agent_id, lesson_id) DO UPDATE SET
                is_active = TRUE,
                injected_at = CURRENT_TIMESTAMP
        """,
            [agent_id, lesson_id],
        )
        self.conn.commit()

    def get_injected_lessons(self, agent_id: str) -> List[Dict]:
        """Get all active lessons injected for an agent."""
        results = self.conn.execute(
            """
            SELECT l.* FROM agent_lessons l
            JOIN lesson_injections i ON l.lesson_id = i.lesson_id
            WHERE i.agent_id = ? AND i.is_active = TRUE
            ORDER BY l.confidence DESC, l.created_at DESC
        """,
            [agent_id],
        ).fetchall()

        cols = [desc[0] for desc in self.conn.description]
        lessons = []
        for row in results:
            data = dict(zip(cols, row))
            if data.get("applies_to"):
                try:
                    data["applies_to"] = json.loads(data["applies_to"])
                except:
                    data["applies_to"] = []
            lessons.append(data)

        return lessons

    def deactivate_injection(self, agent_id: str, lesson_id: str) -> None:
        """Deactivate a lesson injection."""
        self.conn.execute(
            """
            UPDATE lesson_injections
            SET is_active = FALSE
            WHERE agent_id = ? AND lesson_id = ?
        """,
            [agent_id, lesson_id],
        )
        self.conn.commit()

    # =========================================================================
    # LEARNING METRICS OPERATIONS
    # =========================================================================

    def save_learning_checkpoint(self, metrics: Dict) -> int:
        """
        Save a learning metrics checkpoint.

        Args:
            metrics: Dict with trade_count, rolling_win_rate, cumulative_win_rate, etc.

        Returns:
            checkpoint_id
        """
        strategy_weights = metrics.get("strategy_weights")
        if isinstance(strategy_weights, dict):
            strategy_weights = json.dumps(strategy_weights)

        result = self.conn.execute(
            """
            INSERT INTO learning_metrics 
            (date, trade_count, rolling_win_rate, cumulative_win_rate, 
             rolling_pnl, cumulative_pnl, lessons_active, prompt_changes,
             strategy_weights, avg_confidence, regime_at_checkpoint)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            RETURNING checkpoint_id
        """,
            [
                metrics.get("date", datetime.now().date()),
                metrics.get("trade_count", 0),
                metrics.get("rolling_win_rate"),
                metrics.get("cumulative_win_rate"),
                metrics.get("rolling_pnl"),
                metrics.get("cumulative_pnl"),
                metrics.get("lessons_active", 0),
                metrics.get("prompt_changes", 0),
                strategy_weights,
                metrics.get("avg_confidence"),
                metrics.get("regime_at_checkpoint"),
            ],
        ).fetchone()

        self.conn.commit()
        return result[0] if result else 0

    def get_learning_metrics(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 500,
    ) -> List[Dict]:
        """
        Get learning metrics checkpoints.

        Args:
            start_date: Start of date range
            end_date: End of date range
            limit: Maximum records to return

        Returns:
            List of metrics dicts
        """
        query = "SELECT * FROM learning_metrics WHERE 1=1"
        params = []

        if start_date:
            query += " AND date >= ?"
            params.append(
                start_date.isoformat()
                if hasattr(start_date, "isoformat")
                else start_date
            )
        if end_date:
            query += " AND date <= ?"
            params.append(
                end_date.isoformat() if hasattr(end_date, "isoformat") else end_date
            )

        query += f" ORDER BY checkpoint_id ASC LIMIT {limit}"

        results = self.conn.execute(query, params).fetchall()
        cols = [desc[0] for desc in self.conn.description]

        metrics = []
        for row in results:
            data = dict(zip(cols, row))
            if data.get("strategy_weights"):
                try:
                    data["strategy_weights"] = json.loads(data["strategy_weights"])
                except:
                    data["strategy_weights"] = {}
            metrics.append(data)

        return metrics

    def get_latest_learning_checkpoint(self) -> Optional[Dict]:
        """Get the most recent learning checkpoint."""
        result = self.conn.execute(
            """
            SELECT * FROM learning_metrics 
            ORDER BY checkpoint_id DESC 
            LIMIT 1
        """
        ).fetchone()

        if result is None:
            return None

        cols = [desc[0] for desc in self.conn.description]
        data = dict(zip(cols, result))

        if data.get("strategy_weights"):
            try:
                data["strategy_weights"] = json.loads(data["strategy_weights"])
            except:
                data["strategy_weights"] = {}

        return data

    def compute_rolling_win_rate(self, window: int = 20) -> Optional[float]:
        """
        Compute rolling win rate from the last N trades.

        Args:
            window: Number of trades to include (default: 20)

        Returns:
            Win rate as float (0.0-1.0) or None if insufficient data
        """
        from quant_pod.knowledge.models import TradeStatus

        # Get recent closed trades
        trades = self.get_trades(limit=window * 2)  # Get extra to ensure enough closed

        closed_trades = [
            t for t in trades if t.status == TradeStatus.CLOSED and t.pnl is not None
        ][:window]

        if len(closed_trades) < 5:  # Minimum trades for meaningful rate
            return None

        wins = sum(1 for t in closed_trades if t.pnl > 0)
        return wins / len(closed_trades)

    def get_recent_agent_logs(
        self,
        symbol: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict]:
        """
        Get recent agent logs for historical context.

        Args:
            symbol: Filter by symbol (optional)
            limit: Maximum number of logs to return

        Returns:
            List of agent log dicts sorted by date descending
        """
        query = """
            SELECT date, agent_name, symbol, message, reasoning 
            FROM agent_logs 
            WHERE 1=1
        """
        params = []

        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)

        query += " ORDER BY date DESC, log_id DESC LIMIT ?"
        params.append(limit)

        try:
            results = self.conn.execute(query, params).fetchall()
            cols = ["date", "agent_name", "symbol", "message", "reasoning"]
            return [dict(zip(cols, row)) for row in results]
        except Exception as e:
            logger.debug(f"Failed to get agent logs: {e}")
            return []

    def get_recent_trades(
        self,
        symbol: Optional[str] = None,
        limit: int = 5,
    ) -> List[Dict]:
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
            return [dict(zip(cols, row)) for row in results]
        except Exception as e:
            logger.debug(f"Failed to get recent trades: {e}")
            return []
