# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""Learning mixin — Historical arena, lessons, prompts, A/B tests, portfolio snapshots."""

import json
from datetime import datetime
from typing import Any

import duckdb
from loguru import logger

from quantstack.knowledge.models import TradeStatus


class LearningMixin:
    """Historical alpha arena, agentic learning, and portfolio snapshot operations."""

    conn: duckdb.DuckDBPyConnection

    # =========================================================================
    # HISTORICAL ALPHA ARENA OPERATIONS
    # =========================================================================

    def save_daily_state(self, state: dict) -> None:
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

    def save_historical_signal(self, signal: dict) -> int:
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

    def save_policy_snapshot(self, snapshot: dict) -> None:
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

    def save_agent_log(self, log: dict) -> int:
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
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> list[dict]:
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
            data = dict(zip(cols, row, strict=False))
            if data.get("exposures"):
                try:
                    data["exposures"] = json.loads(data["exposures"])
                except Exception:
                    data["exposures"] = {}
            states.append(data)

        return states

    def load_agent_logs(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        symbol: str | None = None,
        agent_name: str | None = None,
        limit: int = 1000,
    ) -> list[dict]:
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

        return [dict(zip(cols, row, strict=False)) for row in results]

    def load_historical_signals(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        symbol: str | None = None,
        limit: int = 1000,
    ) -> list[dict]:
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

        return [dict(zip(cols, row, strict=False)) for row in results]

    def load_policy_snapshots(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> list[dict]:
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
            data = dict(zip(cols, row, strict=False))
            if data.get("pod_weights"):
                try:
                    data["pod_weights"] = json.loads(data["pod_weights"])
                except Exception:
                    data["pod_weights"] = {}
            if data.get("thresholds"):
                try:
                    data["thresholds"] = json.loads(data["thresholds"])
                except Exception:
                    data["thresholds"] = {}
            snapshots.append(data)

        return snapshots

    def get_latest_policy(self, as_of_date: datetime | None = None) -> dict | None:
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
        data = dict(zip(cols, result, strict=False))

        if data.get("pod_weights"):
            try:
                data["pod_weights"] = json.loads(data["pod_weights"])
            except Exception:
                data["pod_weights"] = {}
        if data.get("thresholds"):
            try:
                data["thresholds"] = json.loads(data["thresholds"])
            except Exception:
                data["thresholds"] = {}

        return data

    # =========================================================================
    # AGENTIC LEARNING SYSTEM OPERATIONS
    # =========================================================================

    def save_lesson(self, lesson: dict) -> str:
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
    ) -> list[dict]:
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
            data = dict(zip(cols, row, strict=False))
            if data.get("applies_to"):
                try:
                    data["applies_to"] = json.loads(data["applies_to"])
                except Exception:
                    data["applies_to"] = []
            lessons.append(data)

        return lessons

    def get_lessons_for_agent(self, agent_id: str, limit: int = 5) -> list[dict]:
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
            data = dict(zip(cols, row, strict=False))
            if data.get("applies_to"):
                try:
                    data["applies_to"] = json.loads(data["applies_to"])
                except Exception:
                    data["applies_to"] = []
            lessons.append(data)

        return lessons

    def update_lesson_usage(
        self, lesson_id: str, was_effective: bool | None = None
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

    def save_prompt_proposal(self, proposal: dict) -> str:
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
        agent_id: str | None = None,
        status: str | None = None,
        limit: int = 20,
    ) -> list[dict]:
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

        return [dict(zip(cols, row, strict=False)) for row in results]

    def update_proposal_status(self, proposal_id: str, status: str) -> None:
        """Update the status of a proposal."""
        self.conn.execute(
            """
            UPDATE prompt_proposals SET status = ? WHERE proposal_id = ?
        """,
            [status, proposal_id],
        )
        self.conn.commit()

    def create_ab_test(self, test: dict) -> str:
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

    def get_ab_test(self, test_id: str) -> dict | None:
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
        return dict(zip(cols, result, strict=False))

    def get_active_ab_tests(self, agent_id: str | None = None) -> list[dict]:
        """Get all running A/B tests."""
        query = "SELECT * FROM ab_tests WHERE status = 'running'"
        params = []

        if agent_id:
            query += " AND agent_id = ?"
            params.append(agent_id)

        results = self.conn.execute(query, params).fetchall()
        cols = [desc[0] for desc in self.conn.description]

        return [dict(zip(cols, row, strict=False)) for row in results]

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

    def get_injected_lessons(self, agent_id: str) -> list[dict]:
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
            data = dict(zip(cols, row, strict=False))
            if data.get("applies_to"):
                try:
                    data["applies_to"] = json.loads(data["applies_to"])
                except Exception:
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

    def save_learning_checkpoint(self, metrics: dict) -> int:
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
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        limit: int = 500,
    ) -> list[dict]:
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
            data = dict(zip(cols, row, strict=False))
            if data.get("strategy_weights"):
                try:
                    data["strategy_weights"] = json.loads(data["strategy_weights"])
                except Exception:
                    data["strategy_weights"] = {}
            metrics.append(data)

        return metrics

    def get_latest_learning_checkpoint(self) -> dict | None:
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
        data = dict(zip(cols, result, strict=False))

        if data.get("strategy_weights"):
            try:
                data["strategy_weights"] = json.loads(data["strategy_weights"])
            except Exception:
                data["strategy_weights"] = {}

        return data

    def compute_rolling_win_rate(self, window: int = 20) -> float | None:
        """
        Compute rolling win rate from the last N trades.

        Args:
            window: Number of trades to include (default: 20)

        Returns:
            Win rate as float (0.0-1.0) or None if insufficient data
        """
        from quantstack.knowledge.models import TradeStatus

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
        symbol: str | None = None,
        limit: int = 10,
    ) -> list[dict]:
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
            return [dict(zip(cols, row, strict=False)) for row in results]
        except Exception as e:
            logger.debug(f"Failed to get agent logs: {e}")
            return []

    # =========================================================================
    # PORTFOLIO SNAPSHOT — queryable history of portfolio state over time
    # =========================================================================

    def save_portfolio_snapshot(self, snapshot: dict[str, Any]) -> None:
        """
        Save a point-in-time portfolio snapshot for historical tracking.

        Called at session start so the knowledge store has a time-series
        of portfolio equity, cash, and position count — queryable for
        performance attribution and drawdown analysis.

        Args:
            snapshot: Dict with keys: cash, positions_value, total_equity,
                      daily_pnl, total_realized_pnl, position_count,
                      largest_position_pct
        """
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                id          BIGINT,
                captured_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                cash        DOUBLE,
                positions_value DOUBLE,
                total_equity    DOUBLE,
                daily_pnl       DOUBLE,
                total_realized_pnl DOUBLE,
                position_count  INTEGER,
                largest_position_pct DOUBLE
            )
            """
        )
        self.conn.execute(
            "CREATE SEQUENCE IF NOT EXISTS seq_portfolio_snapshots START 1"
        )
        self.conn.execute(
            """
            INSERT INTO portfolio_snapshots
                (id, cash, positions_value, total_equity, daily_pnl,
                 total_realized_pnl, position_count, largest_position_pct)
            VALUES (nextval('seq_portfolio_snapshots'), ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                snapshot.get("cash", 0.0),
                snapshot.get("positions_value", 0.0),
                snapshot.get("total_equity", 0.0),
                snapshot.get("daily_pnl", 0.0),
                snapshot.get("total_realized_pnl", 0.0),
                snapshot.get("position_count", 0),
                snapshot.get("largest_position_pct", 0.0),
            ],
        )
        self.conn.commit()

    def get_latest_portfolio_snapshot(self) -> dict[str, Any] | None:
        """
        Return the most recently saved portfolio snapshot.

        Used at startup to surface the last known portfolio state without
        requiring the full PortfolioState DuckDB to be open.
        """
        try:
            row = self.conn.execute(
                """
                SELECT cash, positions_value, total_equity, daily_pnl,
                       total_realized_pnl, position_count, largest_position_pct,
                       captured_at
                FROM portfolio_snapshots
                ORDER BY captured_at DESC LIMIT 1
                """
            ).fetchone()
        except Exception:
            return None

        if row is None:
            return None

        return {
            "cash": row[0],
            "positions_value": row[1],
            "total_equity": row[2],
            "daily_pnl": row[3],
            "total_realized_pnl": row[4],
            "position_count": row[5],
            "largest_position_pct": row[6],
            "captured_at": row[7].isoformat() if row[7] else None,
        }
