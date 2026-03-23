"""
Model registry — DuckDB-backed storage for trained FinRL models.

Tracks model metadata, training metrics, evaluation results, and lifecycle
status (shadow → live → retired).

Usage:
    from quantstack.finrl.model_registry import ModelRegistry

    registry = ModelRegistry(db_conn)
    registry.register(model_id, env_type="execution", algorithm="dqn", ...)
    registry.update_status(model_id, "live")
    models = registry.list_models(status="shadow")
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from loguru import logger


class ModelRegistry:
    """
    DuckDB-backed registry for FinRL model metadata and lifecycle.

    Table: finrl_models
    """

    def __init__(self, db_conn: Any):
        self.db = db_conn
        self._ensure_table()

    def _ensure_table(self) -> None:
        """Create the finrl_models table if it doesn't exist."""
        self.db.execute(
            """
            CREATE TABLE IF NOT EXISTS finrl_models (
                model_id        VARCHAR PRIMARY KEY,
                name            VARCHAR,
                env_type        VARCHAR,
                algorithm       VARCHAR,
                symbols         VARCHAR,
                train_start     VARCHAR,
                train_end       VARCHAR,
                hyperparams     VARCHAR,
                training_metrics VARCHAR,
                eval_metrics    VARCHAR,
                checkpoint_path VARCHAR,
                status          VARCHAR DEFAULT 'shadow',
                shadow_start    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                promoted_at     TIMESTAMP,
                created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

    def register(
        self,
        model_id: str,
        env_type: str,
        algorithm: str,
        checkpoint_path: str,
        name: str | None = None,
        symbols: list[str] | None = None,
        train_start: str | None = None,
        train_end: str | None = None,
        hyperparams: dict | None = None,
        training_metrics: dict | None = None,
    ) -> dict[str, Any]:
        """Register a newly trained model."""
        self.db.execute(
            """
            INSERT INTO finrl_models
                (model_id, name, env_type, algorithm, symbols, train_start,
                 train_end, hyperparams, training_metrics, checkpoint_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                model_id,
                name or model_id,
                env_type,
                algorithm,
                json.dumps(symbols or []),
                train_start,
                train_end,
                json.dumps(hyperparams or {}),
                json.dumps(training_metrics or {}),
                checkpoint_path,
            ],
        )
        logger.info(f"[ModelRegistry] Registered model {model_id} ({env_type}/{algorithm})")
        return {"model_id": model_id, "status": "shadow"}

    def get(self, model_id: str) -> dict[str, Any] | None:
        """Get model metadata by ID."""
        row = self.db.execute(
            "SELECT * FROM finrl_models WHERE model_id = ?", [model_id]
        ).fetchone()
        if not row:
            return None
        return self._row_to_dict(row)

    def list_models(
        self,
        env_type: str | None = None,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        """List models with optional filters."""
        query = "SELECT * FROM finrl_models WHERE 1=1"
        params: list[Any] = []

        if env_type:
            query += " AND env_type = ?"
            params.append(env_type)
        if status:
            query += " AND status = ?"
            params.append(status)

        query += " ORDER BY created_at DESC"
        rows = self.db.execute(query, params).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def update_eval_metrics(
        self, model_id: str, metrics: dict[str, Any]
    ) -> None:
        """Update evaluation metrics for a model."""
        self.db.execute(
            """
            UPDATE finrl_models
            SET eval_metrics = ?, updated_at = CURRENT_TIMESTAMP
            WHERE model_id = ?
            """,
            [json.dumps(metrics), model_id],
        )

    def update_status(
        self, model_id: str, new_status: str, reason: str | None = None
    ) -> None:
        """Update model status (shadow → live → retired)."""
        extra = ""
        params: list[Any] = [new_status]

        if new_status == "live":
            extra = ", promoted_at = CURRENT_TIMESTAMP"

        self.db.execute(
            f"""
            UPDATE finrl_models
            SET status = ?, updated_at = CURRENT_TIMESTAMP{extra}
            WHERE model_id = ?
            """,
            params + [model_id],
        )
        logger.info(
            f"[ModelRegistry] Model {model_id} → {new_status}"
            + (f" ({reason})" if reason else "")
        )

    def delete(self, model_id: str) -> None:
        """Delete a model record."""
        self.db.execute("DELETE FROM finrl_models WHERE model_id = ?", [model_id])

    def _row_to_dict(self, row: tuple) -> dict[str, Any]:
        """Convert a DuckDB row to dict."""
        cols = [
            "model_id", "name", "env_type", "algorithm", "symbols",
            "train_start", "train_end", "hyperparams", "training_metrics",
            "eval_metrics", "checkpoint_path", "status", "shadow_start",
            "promoted_at", "created_at", "updated_at",
        ]
        d = dict(zip(cols, row))

        # Parse JSON fields
        for key in ("symbols", "hyperparams", "training_metrics", "eval_metrics"):
            if d.get(key) and isinstance(d[key], str):
                try:
                    d[key] = json.loads(d[key])
                except (json.JSONDecodeError, TypeError):
                    pass

        # Stringify timestamps
        for key in ("shadow_start", "promoted_at", "created_at", "updated_at"):
            if d.get(key) and not isinstance(d[key], str):
                d[key] = str(d[key])

        return d
