# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Degradation enforcement bridge.

Bridges the gap between *advisory* degradation detection (DegradationDetector,
AlphaMonitor) and *enforced* risk reduction (StrategyBreaker).

Without this bridge:
  - DegradationDetector returns ``recommended_size_multiplier = 0.25``
  - Nobody applies it — the Trader loop might not read the report
  - Strategy keeps trading at full size while degrading

With this bridge:
  - CRITICAL → force_trip() on StrategyBreaker (0x size) + event
  - WARNING → force_scale() on StrategyBreaker (recommended multiplier) + event
  - Both publish events so Factory loop can investigate and Trader adjusts sizing

Integration point: called from IntradayMonitorFlow after the existing
DegradationDetector.check() step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from loguru import logger

from quantstack.coordination.event_bus import Event, EventType
from quantstack.coordination.slack_client import SlackClient


@dataclass
class EnforcementResult:
    """Outcome of a degradation enforcement check."""

    strategy_id: str
    severity: str  # "clean", "warning", "critical"
    action_taken: str  # "none", "scaled", "tripped"
    size_multiplier: float = 1.0
    findings: list[str] | None = None


class DegradationEnforcer:
    """
    Bridges DegradationDetector output to StrategyBreaker state changes.

    Args:
        detector: DegradationDetector instance (or compatible object with check()).
        breaker: StrategyBreaker instance.
        event_bus: EventBus instance for publishing degradation events.
    """

    def __init__(
        self,
        detector: Any,
        breaker: Any,
        event_bus: Any | None = None,
    ) -> None:
        self._detector = detector
        self._breaker = breaker
        self._bus = event_bus

    def enforce(self, strategy_id: str) -> EnforcementResult:
        """
        Check degradation for a strategy and enforce via StrategyBreaker.

        Returns:
            EnforcementResult with the action taken.
        """
        try:
            report = self._detector.check(strategy_id)
        except Exception as exc:
            logger.debug(
                f"[DegradationEnforcer] Detector failed for {strategy_id}: {exc}"
            )
            return EnforcementResult(
                strategy_id=strategy_id,
                severity="unknown",
                action_taken="none",
                findings=[f"Detector error: {exc}"],
            )

        # Extract severity from detector report
        severity = getattr(report, "status", getattr(report, "severity", "clean"))
        if isinstance(severity, str):
            severity = severity.lower()
        else:
            severity = str(severity).lower()

        findings = getattr(report, "findings", [])
        multiplier = getattr(report, "recommended_size_multiplier", 1.0)

        if "critical" in severity:
            self._apply_critical(strategy_id, findings, multiplier)
            return EnforcementResult(
                strategy_id=strategy_id,
                severity="critical",
                action_taken="tripped",
                size_multiplier=0.0,
                findings=findings,
            )

        if "warning" in severity:
            self._apply_warning(strategy_id, findings, multiplier)
            return EnforcementResult(
                strategy_id=strategy_id,
                severity="warning",
                action_taken="scaled",
                size_multiplier=multiplier,
                findings=findings,
            )

        return EnforcementResult(
            strategy_id=strategy_id,
            severity="clean",
            action_taken="none",
            size_multiplier=1.0,
        )

    def enforce_all(self, strategy_ids: list[str]) -> list[EnforcementResult]:
        """Enforce degradation checks for multiple strategies."""
        return [self.enforce(sid) for sid in strategy_ids]

    def _apply_critical(
        self, strategy_id: str, findings: list[str], multiplier: float
    ) -> None:
        """Force-trip the strategy breaker and publish event."""
        reason = f"Degradation CRITICAL: {'; '.join(findings) if findings else 'metrics below threshold'}"

        if hasattr(self._breaker, "force_trip"):
            self._breaker.force_trip(strategy_id, reason=reason)
        else:
            logger.warning(
                f"[DegradationEnforcer] StrategyBreaker lacks force_trip — "
                f"cannot enforce CRITICAL for {strategy_id}"
            )

        self._publish_event(strategy_id, "critical", multiplier, findings)
        self._post_slack_alert("critical", f"Strategy {strategy_id} TRIPPED", reason)
        logger.warning(f"[DegradationEnforcer] TRIPPED {strategy_id}: {reason}")

    def _apply_warning(
        self, strategy_id: str, findings: list[str], multiplier: float
    ) -> None:
        """Force-scale the strategy breaker and publish event."""
        reason = f"Degradation WARNING: {'; '.join(findings) if findings else 'metrics degrading'}"

        if hasattr(self._breaker, "force_scale"):
            self._breaker.force_scale(
                strategy_id, scale_factor=multiplier, reason=reason
            )
        else:
            logger.warning(
                f"[DegradationEnforcer] StrategyBreaker lacks force_scale — "
                f"cannot enforce WARNING for {strategy_id}"
            )

        self._publish_event(strategy_id, "warning", multiplier, findings)
        self._post_slack_alert(
            "warning", f"Strategy {strategy_id} SCALED to {multiplier:.0%}", reason
        )
        logger.info(
            f"[DegradationEnforcer] SCALED {strategy_id} to {multiplier:.0%}: {reason}"
        )

    def _post_slack_alert(self, severity: str, title: str, detail: str) -> None:
        """Post degradation alert to Slack #alerts."""
        try:
            SlackClient().post_alert(severity, title, detail)
        except Exception as exc:
            logger.debug(f"[DegradationEnforcer] Slack alert failed: {exc}")

    def _publish_event(
        self,
        strategy_id: str,
        severity: str,
        multiplier: float,
        findings: list[str],
    ) -> None:
        """Publish degradation event to bus if available."""
        if not self._bus:
            return
        try:
            self._bus.publish(
                Event(
                    event_type=EventType.DEGRADATION_DETECTED,
                    source_loop="degradation_enforcer",
                    payload={
                        "strategy_id": strategy_id,
                        "severity": severity,
                        "size_multiplier": multiplier,
                        "findings": findings or [],
                    },
                )
            )
        except Exception as exc:
            logger.debug(f"[DegradationEnforcer] Failed to publish event: {exc}")
