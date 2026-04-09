# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Health dashboard (P15) — aggregated system health for the autonomous fund.

Provides a single ``HealthDashboard.check_all()`` entry point that probes
every major subsystem and returns structured health data. Designed to be
called by the supervisor graph, the HTTP /health endpoint, or diagnostic
scripts.

Subsystems checked:
  1. Feedback loops — are all five loops running and healthy?
  2. Data freshness — is market data recent enough for trading?
  3. Model health — are ML models loaded and not stale?
  4. Execution health — is the broker connection alive, any stuck orders?
  5. Reconciliation — do broker and system positions agree?

The dashboard does NOT take corrective action. It reports status. The
supervisor graph and watchdog consume this data to decide on actions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from loguru import logger

from quantstack.db import pg_conn


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class HealthStatus(str, Enum):
    """Health status levels, ordered from best to worst."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    BROKEN = "broken"
    UNKNOWN = "unknown"


# Severity ordering for worst-of aggregation
_STATUS_SEVERITY = {
    HealthStatus.HEALTHY: 0,
    HealthStatus.DEGRADED: 1,
    HealthStatus.BROKEN: 2,
    HealthStatus.UNKNOWN: 3,
}


@dataclass
class SubsystemHealth:
    """Health report for a single subsystem."""

    name: str
    status: HealthStatus
    last_check: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    details: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------


class HealthDashboard:
    """Aggregates health checks across all autonomous fund subsystems.

    Stateless — each call to ``check_all()`` performs fresh probes. No
    caching, no background threads. The caller controls check frequency.
    """

    def check_all(self) -> list[SubsystemHealth]:
        """Run all subsystem health checks and return results."""
        checks = [
            self.check_feedback_loops(),
            self.check_data_freshness(),
            self.check_model_health(),
            self.check_execution_health(),
            self.check_reconciliation(),
        ]
        return checks

    def check_feedback_loops(self) -> SubsystemHealth:
        """Check whether the feedback loop manager reports all loops healthy."""
        try:
            from quantstack.autonomous.feedback_loops import FeedbackLoopManager

            mgr = FeedbackLoopManager()
            report = mgr.health_report()
            all_healthy = report.get("overall_healthy", False)

            unhealthy = [
                name
                for name, info in report.get("loops", {}).items()
                if not info.get("is_healthy", True)
            ]

            if all_healthy:
                status = HealthStatus.HEALTHY
            elif len(unhealthy) <= 2:
                status = HealthStatus.DEGRADED
            else:
                status = HealthStatus.BROKEN

            return SubsystemHealth(
                name="feedback_loops",
                status=status,
                details={
                    "overall_healthy": all_healthy,
                    "unhealthy_loops": unhealthy,
                },
            )
        except Exception as exc:
            logger.error(f"[HEALTH] Feedback loop check failed: {exc}")
            return SubsystemHealth(
                name="feedback_loops",
                status=HealthStatus.UNKNOWN,
                details={"error": str(exc)},
            )

    def check_data_freshness(self) -> SubsystemHealth:
        """Check whether market data is fresh enough for trading decisions.

        Looks at the most recent daily_prices row. If it's older than 2 calendar
        days (accounting for weekends), data is stale.
        """
        try:
            with pg_conn() as conn:
                row = conn.execute(
                    """
                    SELECT MAX(price_date) AS latest_date
                    FROM daily_prices
                    """,
                ).fetchone()

            if row and row["latest_date"]:
                latest = row["latest_date"]
                if isinstance(latest, str):
                    from datetime import date as _date

                    latest = _date.fromisoformat(latest)

                today = datetime.now(timezone.utc).date()
                age_days = (today - latest).days

                if age_days <= 2:
                    status = HealthStatus.HEALTHY
                elif age_days <= 5:
                    status = HealthStatus.DEGRADED
                else:
                    status = HealthStatus.BROKEN

                return SubsystemHealth(
                    name="data_freshness",
                    status=status,
                    details={
                        "latest_price_date": str(latest),
                        "age_days": age_days,
                    },
                )
            else:
                return SubsystemHealth(
                    name="data_freshness",
                    status=HealthStatus.BROKEN,
                    details={"error": "no daily_prices data found"},
                )
        except Exception as exc:
            logger.error(f"[HEALTH] Data freshness check failed: {exc}")
            return SubsystemHealth(
                name="data_freshness",
                status=HealthStatus.UNKNOWN,
                details={"error": str(exc)},
            )

    def check_model_health(self) -> SubsystemHealth:
        """Check ML model freshness and availability.

        Looks at the ml_experiments table for the most recent successful
        training run. Models older than 30 days are degraded; 90 days is broken.
        """
        try:
            with pg_conn() as conn:
                row = conn.execute(
                    """
                    SELECT MAX(created_at) AS latest_train
                    FROM ml_experiments
                    WHERE status = 'completed'
                    """,
                ).fetchone()

            if row and row["latest_train"]:
                latest = row["latest_train"]
                if isinstance(latest, str):
                    latest = datetime.fromisoformat(latest)
                if latest.tzinfo is None:
                    latest = latest.replace(tzinfo=timezone.utc)

                age_days = (datetime.now(timezone.utc) - latest).days

                if age_days <= 30:
                    status = HealthStatus.HEALTHY
                elif age_days <= 90:
                    status = HealthStatus.DEGRADED
                else:
                    status = HealthStatus.BROKEN

                return SubsystemHealth(
                    name="model_health",
                    status=status,
                    details={
                        "latest_training": latest.isoformat(),
                        "age_days": age_days,
                    },
                )
            else:
                return SubsystemHealth(
                    name="model_health",
                    status=HealthStatus.DEGRADED,
                    details={"note": "no completed ml_experiments found"},
                )
        except Exception as exc:
            logger.error(f"[HEALTH] Model health check failed: {exc}")
            return SubsystemHealth(
                name="model_health",
                status=HealthStatus.UNKNOWN,
                details={"error": str(exc)},
            )

    def check_execution_health(self) -> SubsystemHealth:
        """Check execution pipeline: stuck orders, recent fill rate.

        Stuck orders: any order submitted > 5 minutes ago still in 'pending'.
        Fill rate: fraction of orders that filled in the last 24h.
        """
        try:
            with pg_conn() as conn:
                # Stuck orders
                stuck_row = conn.execute(
                    """
                    SELECT COUNT(*) AS cnt
                    FROM orders
                    WHERE status = 'pending'
                      AND submitted_at < NOW() - INTERVAL '5 minutes'
                    """,
                ).fetchone()
                stuck_count = int(stuck_row["cnt"]) if stuck_row else 0

                # Fill rate (last 24h)
                rate_row = conn.execute(
                    """
                    SELECT
                        COUNT(*) FILTER (WHERE status = 'filled') AS filled,
                        COUNT(*) AS total
                    FROM orders
                    WHERE submitted_at >= NOW() - INTERVAL '24 hours'
                    """,
                ).fetchone()

                total = int(rate_row["total"]) if rate_row else 0
                filled = int(rate_row["filled"]) if rate_row else 0
                fill_rate = filled / max(total, 1)

            if stuck_count == 0 and fill_rate >= 0.8:
                status = HealthStatus.HEALTHY
            elif stuck_count <= 2 and fill_rate >= 0.5:
                status = HealthStatus.DEGRADED
            else:
                status = HealthStatus.BROKEN

            return SubsystemHealth(
                name="execution_health",
                status=status,
                details={
                    "stuck_orders": stuck_count,
                    "fill_rate_24h": round(fill_rate, 2),
                    "orders_24h": total,
                },
            )
        except Exception as exc:
            logger.error(f"[HEALTH] Execution health check failed: {exc}")
            return SubsystemHealth(
                name="execution_health",
                status=HealthStatus.UNKNOWN,
                details={"error": str(exc)},
            )

    def check_reconciliation(self) -> SubsystemHealth:
        """Check most recent reconciliation result from system_events.

        Reads the latest reconciliation_mismatch event. If none exists or
        the last run was clean, healthy. If mismatches exist, degraded or broken
        depending on total discrepancy.
        """
        try:
            with pg_conn() as conn:
                row = conn.execute(
                    """
                    SELECT payload, created_at
                    FROM system_events
                    WHERE event_type = 'reconciliation_mismatch'
                    ORDER BY created_at DESC
                    LIMIT 1
                    """,
                ).fetchone()

            if not row:
                # No mismatch events ever recorded — assume clean
                return SubsystemHealth(
                    name="reconciliation",
                    status=HealthStatus.HEALTHY,
                    details={"note": "no reconciliation events found"},
                )

            import json

            payload = row["payload"]
            if isinstance(payload, str):
                payload = json.loads(payload)

            discrepancy = float(payload.get("total_discrepancy_usd", 0))

            if discrepancy == 0:
                status = HealthStatus.HEALTHY
            elif discrepancy < 500:
                status = HealthStatus.DEGRADED
            else:
                status = HealthStatus.BROKEN

            return SubsystemHealth(
                name="reconciliation",
                status=status,
                details={
                    "last_check": str(row["created_at"]),
                    "total_discrepancy_usd": discrepancy,
                    "mismatches": payload.get("mismatches", 0),
                },
            )
        except Exception as exc:
            logger.error(f"[HEALTH] Reconciliation check failed: {exc}")
            return SubsystemHealth(
                name="reconciliation",
                status=HealthStatus.UNKNOWN,
                details={"error": str(exc)},
            )

    def overall_status(self) -> HealthStatus:
        """Return the worst status across all subsystems."""
        checks = self.check_all()
        if not checks:
            return HealthStatus.UNKNOWN
        return max(
            (c.status for c in checks),
            key=lambda s: _STATUS_SEVERITY.get(s, 99),
        )

    def to_dict(self) -> dict[str, Any]:
        """Full dashboard as a JSON-serializable dict."""
        checks = self.check_all()
        overall = max(
            (c.status for c in checks),
            key=lambda s: _STATUS_SEVERITY.get(s, 99),
        )
        return {
            "overall_status": overall.value,
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "subsystems": [
                {
                    "name": c.name,
                    "status": c.status.value,
                    "last_check": c.last_check.isoformat(),
                    "details": c.details,
                }
                for c in checks
            ],
        }
