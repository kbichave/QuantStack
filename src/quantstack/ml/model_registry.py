"""
Model versioning registry — champion/challenger lifecycle.

Tracks model versions per strategy, manages shadow evaluation periods,
and enforces promotion criteria before a new model can replace the
current champion.

Tables: model_registry, model_shadow_predictions (created in db.py migrations).

Promotion criteria (all must be met):
  - Challenger IC > champion IC + 0.005
  - Challenger Sharpe > champion Sharpe + 0.15
  - Challenger max DD <= 1.1x champion max DD
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any

from loguru import logger

from quantstack.db import db_conn


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ModelVersion:
    """A registered model version."""

    model_id: str
    strategy_id: str
    version: int
    train_date: date
    train_data_range: str
    features_hash: str
    hyperparams: dict
    backtest_sharpe: float
    backtest_ic: float
    backtest_max_dd: float
    model_path: str
    status: str  # champion | challenger | retired
    promoted_at: datetime | None
    retired_at: datetime | None
    shadow_start: date | None
    shadow_ic: float | None
    shadow_sharpe: float | None
    created_at: datetime


def _row_to_model(row: dict) -> ModelVersion:
    """Convert a DB row dict to ModelVersion."""
    return ModelVersion(
        model_id=row["model_id"],
        strategy_id=row["strategy_id"],
        version=row["version"],
        train_date=row["train_date"],
        train_data_range=row.get("train_data_range", ""),
        features_hash=row.get("features_hash", ""),
        hyperparams=row.get("hyperparams") or {},
        backtest_sharpe=row.get("backtest_sharpe") or 0.0,
        backtest_ic=row.get("backtest_ic") or 0.0,
        backtest_max_dd=row.get("backtest_max_dd") or 0.0,
        model_path=row["model_path"],
        status=row["status"],
        promoted_at=row.get("promoted_at"),
        retired_at=row.get("retired_at"),
        shadow_start=row.get("shadow_start"),
        shadow_ic=row.get("shadow_ic"),
        shadow_sharpe=row.get("shadow_sharpe"),
        created_at=row.get("created_at") or datetime.now(timezone.utc),
    )


# ---------------------------------------------------------------------------
# CRUD operations
# ---------------------------------------------------------------------------


def register_model(
    strategy_id: str,
    train_date: date,
    train_data_range: str,
    features_hash: str,
    hyperparams: dict,
    backtest_sharpe: float,
    backtest_ic: float,
    backtest_max_dd: float,
    model_path: str,
) -> ModelVersion:
    """Register a new model as challenger. Version auto-increments per strategy."""
    with db_conn() as conn:
        row = conn.fetchone(
            "SELECT COALESCE(MAX(version), 0) AS max_version "
            "FROM model_registry WHERE strategy_id = %s",
            (strategy_id,),
        )
        next_version = (row["max_version"] if row else 0) + 1
        model_id = f"{strategy_id}_v{next_version}"
        now = datetime.now(timezone.utc)

        conn.execute(
            "INSERT INTO model_registry "
            "(model_id, strategy_id, version, train_date, train_data_range, "
            " features_hash, hyperparams, backtest_sharpe, backtest_ic, "
            " backtest_max_dd, model_path, status, shadow_start, created_at) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'challenger', %s, %s)",
            (
                model_id, strategy_id, next_version, train_date,
                train_data_range, features_hash,
                json.dumps(hyperparams), backtest_sharpe, backtest_ic,
                backtest_max_dd, model_path, train_date, now,
            ),
        )

        logger.info(
            f"[model_registry] Registered {model_id} as challenger "
            f"(Sharpe={backtest_sharpe:.2f}, IC={backtest_ic:.4f})"
        )

        return ModelVersion(
            model_id=model_id,
            strategy_id=strategy_id,
            version=next_version,
            train_date=train_date,
            train_data_range=train_data_range,
            features_hash=features_hash,
            hyperparams=hyperparams,
            backtest_sharpe=backtest_sharpe,
            backtest_ic=backtest_ic,
            backtest_max_dd=backtest_max_dd,
            model_path=model_path,
            status="challenger",
            promoted_at=None,
            retired_at=None,
            shadow_start=train_date,
            shadow_ic=None,
            shadow_sharpe=None,
            created_at=now,
        )


def query_champion(strategy_id: str) -> ModelVersion | None:
    """Return the current champion for a strategy, or None."""
    with db_conn() as conn:
        row = conn.fetchone(
            "SELECT * FROM model_registry "
            "WHERE strategy_id = %s AND status = 'champion' "
            "LIMIT 1",
            (strategy_id,),
        )
        if row is None:
            return None
        return _row_to_model(row)


def promote_challenger(model_id: str) -> None:
    """Promote challenger to champion. Retire the current champion."""
    with db_conn() as conn:
        row = conn.fetchone(
            "SELECT * FROM model_registry WHERE model_id = %s",
            (model_id,),
        )
        if row is None:
            logger.warning(f"[model_registry] Cannot promote {model_id}: not found")
            return

        strategy_id = row["strategy_id"]
        now = datetime.now(timezone.utc)

        # Retire current champion
        conn.execute(
            "UPDATE model_registry SET status = 'retired', retired_at = %s "
            "WHERE strategy_id = %s AND status = 'champion'",
            (now, strategy_id),
        )

        # Promote challenger
        conn.execute(
            "UPDATE model_registry SET status = 'champion', promoted_at = %s "
            "WHERE model_id = %s",
            (now, model_id),
        )

        logger.info(
            f"[model_registry] Promoted {model_id} to champion for {strategy_id}"
        )


def retire_model(model_id: str) -> None:
    """Set status to retired, set retired_at timestamp."""
    with db_conn() as conn:
        conn.execute(
            "UPDATE model_registry SET status = 'retired', retired_at = %s "
            "WHERE model_id = %s",
            (datetime.now(timezone.utc), model_id),
        )
        logger.info(f"[model_registry] Retired {model_id}")


def get_challengers_for_review() -> list[ModelVersion]:
    """Return all challengers with >= 30 trading days of shadow data."""
    with db_conn() as conn:
        rows = conn.fetchall(
            "SELECT * FROM model_registry "
            "WHERE status = 'challenger' "
            "AND shadow_start <= CURRENT_DATE - INTERVAL '30 days'",
        )
        return [_row_to_model(r) for r in rows]


def get_stale_challengers(max_shadow_days: int = 60) -> list[ModelVersion]:
    """Return challengers in shadow > max_shadow_days without promotion."""
    with db_conn() as conn:
        rows = conn.fetchall(
            "SELECT * FROM model_registry "
            "WHERE status = 'challenger' "
            f"AND shadow_start <= CURRENT_DATE - INTERVAL '{max_shadow_days} days'",
        )
        return [_row_to_model(r) for r in rows]


# ---------------------------------------------------------------------------
# Promotion criteria
# ---------------------------------------------------------------------------

_IC_IMPROVEMENT_THRESHOLD = 0.005
_SHARPE_IMPROVEMENT_THRESHOLD = 0.15
_MAX_DD_MULTIPLIER = 1.1


def evaluate_promotion(
    challenger_ic: float,
    champion_ic: float,
    challenger_sharpe: float,
    champion_sharpe: float,
    challenger_max_dd: float,
    champion_max_dd: float,
) -> tuple[bool, str]:
    """Evaluate whether a challenger should be promoted.

    All three criteria must be met:
      1. challenger IC > champion IC + 0.005
      2. challenger Sharpe > champion Sharpe + 0.15
      3. challenger max DD <= 1.1x champion max DD

    Returns:
        (should_promote, reason)
    """
    if challenger_ic < champion_ic + _IC_IMPROVEMENT_THRESHOLD:
        return False, f"IC improvement insufficient ({challenger_ic:.4f} vs {champion_ic:.4f} + {_IC_IMPROVEMENT_THRESHOLD})"

    if challenger_sharpe < champion_sharpe + _SHARPE_IMPROVEMENT_THRESHOLD:
        return False, f"Sharpe improvement insufficient ({challenger_sharpe:.2f} vs {champion_sharpe:.2f} + {_SHARPE_IMPROVEMENT_THRESHOLD})"

    if challenger_max_dd > champion_max_dd * _MAX_DD_MULTIPLIER:
        return False, f"Drawdown regression ({challenger_max_dd:.4f} > {champion_max_dd:.4f} * {_MAX_DD_MULTIPLIER})"

    return True, "all_criteria_met"


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def compute_features_hash(feature_list: list[str]) -> str:
    """Deterministic hash of feature list for versioning."""
    return hashlib.sha256(
        json.dumps(sorted(feature_list)).encode()
    ).hexdigest()[:16]
