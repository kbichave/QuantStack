# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""Performance mixin — Performance metrics CRUD for KnowledgeStore."""

from datetime import datetime, timedelta

import duckdb

from quant_pod.knowledge.models import PerformanceMetrics


class PerformanceMixin:
    """Performance metrics operations."""

    conn: duckdb.DuckDBPyConnection

    # =========================================================================
    # PERFORMANCE METRICS OPERATIONS
    # =========================================================================

    def save_performance_metrics(self, metrics: PerformanceMetrics) -> int:
        """Save performance metrics."""
        data = metrics.model_dump()

        cols = list(data.keys())
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
    ) -> PerformanceMetrics | None:
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
        return PerformanceMetrics(**dict(zip(cols, result, strict=False)))

    def get_structure_performance(self, days: int = 90) -> dict[str, PerformanceMetrics]:
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
            data = dict(zip(cols, row, strict=False))
            name = data["entity_name"]
            if name not in metrics:
                metrics[name] = PerformanceMetrics(**data)

        return metrics
