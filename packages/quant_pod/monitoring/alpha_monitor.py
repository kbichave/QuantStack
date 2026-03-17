"""
Production alpha decay monitor — rolling IC alerting with Discord webhook notifications.

Wires the IC/ICIR metrics from SkillTracker (added in Sprint 2) into a
production monitoring layer that runs after each trading session and emits
structured alerts when signals show signs of decay.

Why this matters (from the gap analysis):
  "Maven Securities research shows US alpha decays at 36 bps/year, accelerating.
   A strategy that was solid 12 months ago may be underwater today without you
   knowing — until P&L shows it."

Alert thresholds (calibrated to avoid noise while catching real decay):
  - CRITICAL: rolling_ic_30 < 0        (signal is actively losing money)
  - WARNING:  rolling_ic_30 < 0.01 AND ic_trend == DECAYING
  - INFO:     needs_retraining == True  (win-rate criterion, existing)

Discord alert format: Discord Incoming Webhook (no bot required).
Create one at: Server Settings → Integrations → Webhooks.
Set DISCORD_WEBHOOK_URL in .env.

Failure modes:
  - Webhook URL missing → logs warning, skips alert (silent degradation preferred
    to broken production loop).
  - Webhook POST fails → logs error, continues; does not re-raise.
  - SkillTracker DB connection fails → propagates (caller decides to halt or continue).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import requests
from loguru import logger


class AlertSeverity(str, Enum):
    CRITICAL = "critical"  # rolling IC negative — signal is destroying value
    WARNING = "warning"  # IC decaying toward zero — monitor closely
    INFO = "info"  # win-rate or retraining flag


@dataclass
class DegradationAlert:
    """A single agent-level degradation alert."""

    agent_id: str
    severity: AlertSeverity
    message: str
    rolling_ic_30: float
    icir: float
    ic_trend: str
    needs_retraining: bool
    detected_at: datetime = field(default_factory=datetime.now)

    @property
    def emoji(self) -> str:
        return {"critical": "🔴", "warning": "🟡", "info": "🔵"}[self.severity.value]


@dataclass
class DegradationReport:
    """Full monitoring run result."""

    checked_at: datetime
    n_agents_checked: int
    alerts: list[DegradationAlert]
    all_agents_ic_summary: list[dict]

    @property
    def has_critical(self) -> bool:
        return any(a.severity == AlertSeverity.CRITICAL for a in self.alerts)

    @property
    def has_warning(self) -> bool:
        return any(a.severity == AlertSeverity.WARNING for a in self.alerts)

    @property
    def overall_status(self) -> str:
        if self.has_critical:
            return "critical"
        if self.has_warning:
            return "warning"
        if self.alerts:
            return "info"
        return "clean"


class AlphaMonitor:
    """
    Production alpha decay monitor.

    Typical usage (called by TradingDayFlow at end of session):
        monitor = AlphaMonitor()
        report = monitor.check_all_agents()
        if report.has_critical:
            # Log to audit, surface in /skills/degradation endpoint
            ...

    Discord alerts fire automatically when DISCORD_WEBHOOK_URL is set.
    """

    # IC thresholds — set conservatively to avoid alert fatigue.
    # A signal with rolling IC between 0 and 0.01 is marginal but not dead;
    # only flag it when also showing DECAYING trend.
    CRITICAL_IC_THRESHOLD = 0.0  # rolling IC < 0 → actively harmful
    WARNING_IC_THRESHOLD = 0.01  # rolling IC < 0.01 AND DECAYING

    def __init__(
        self,
        webhook_url: str | None = None,
        min_observations: int = 10,
    ) -> None:
        """
        Args:
            webhook_url: Discord incoming webhook URL. If None, reads
                         DISCORD_WEBHOOK_URL from environment.
            min_observations: Minimum IC observations before alerting.
                              Prevents false alarms on new agents.
        """
        self._webhook_url = webhook_url or os.getenv("DISCORD_WEBHOOK_URL")
        self._min_observations = min_observations

    def check_all_agents(self) -> DegradationReport:
        """
        Run degradation check across all tracked agents.

        Fetches IC summary from SkillTracker, classifies each agent,
        and fires Discord alerts for any degradation detected.

        Returns DegradationReport — also exposed via /skills/degradation API.
        """
        from quant_pod.knowledge.store import KnowledgeStore
        from quant_pod.learning.skill_tracker import SkillTracker

        store = KnowledgeStore()
        tracker = SkillTracker(store)
        ic_summary = tracker.ic_summary()  # sorted by ICIR desc

        alerts: list[DegradationAlert] = []
        now = datetime.now()

        for entry in ic_summary:
            alert = self._classify_agent(entry, tracker)
            if alert is not None:
                alerts.append(alert)

        report = DegradationReport(
            checked_at=now,
            n_agents_checked=len(ic_summary),
            alerts=alerts,
            all_agents_ic_summary=ic_summary,
        )

        if alerts:
            logger.warning(
                f"[MONITOR] {len(alerts)} degradation alerts: "
                f"{sum(1 for a in alerts if a.severity == AlertSeverity.CRITICAL)} critical, "
                f"{sum(1 for a in alerts if a.severity == AlertSeverity.WARNING)} warning"
            )
            self._send_discord_alert(report)
        else:
            logger.info(f"[MONITOR] Alpha check clean — {len(ic_summary)} agents, no degradation")

        return report

    def check_agent(self, agent_id: str) -> DegradationAlert | None:
        """
        Check a single agent. Returns None if the agent is healthy.

        Useful for post-trade checks in TradingDayFlow without loading all agents.
        """
        from quant_pod.knowledge.store import KnowledgeStore
        from quant_pod.learning.skill_tracker import SkillTracker

        store = KnowledgeStore()
        tracker = SkillTracker(store)
        ic_summary = tracker.ic_summary()

        for entry in ic_summary:
            if entry["agent_id"] == agent_id:
                return self._classify_agent(entry, tracker)

        return None  # Agent not found — no history, no alert

    # -------------------------------------------------------------------------
    # Classification logic
    # -------------------------------------------------------------------------

    def _classify_agent(
        self,
        entry: dict,
        tracker,
    ) -> DegradationAlert | None:
        """
        Classify an agent's IC summary entry into an alert.

        Returns None if the agent is healthy or has insufficient data.
        """
        agent_id = entry["agent_id"]
        n_obs = entry["n_ic_observations"]
        rolling_ic = entry["rolling_ic_30"]
        ic_trend = entry["ic_trend"]
        icir = entry["icir"]
        needs_retraining = entry["needs_retraining"]

        # Not enough data to make a call — avoid false positives on new agents
        if n_obs < self._min_observations:
            return None

        # CRITICAL: negative rolling IC — signal is actively losing money
        if rolling_ic < self.CRITICAL_IC_THRESHOLD:
            return DegradationAlert(
                agent_id=agent_id,
                severity=AlertSeverity.CRITICAL,
                message=(
                    f"Rolling IC(30)={rolling_ic:.4f} is NEGATIVE. "
                    f"Signal has inverted or lost all predictive power. "
                    f"Recommend halting new entries and reviewing signal inputs. "
                    f"(ICIR={icir:.2f}, trend={ic_trend})"
                ),
                rolling_ic_30=rolling_ic,
                icir=icir,
                ic_trend=ic_trend,
                needs_retraining=needs_retraining,
            )

        # WARNING: decaying toward zero
        if rolling_ic < self.WARNING_IC_THRESHOLD and ic_trend == "DECAYING":
            return DegradationAlert(
                agent_id=agent_id,
                severity=AlertSeverity.WARNING,
                message=(
                    f"Rolling IC(30)={rolling_ic:.4f} is near zero AND trend=DECAYING. "
                    f"Alpha appears to be fading. "
                    f"Reduce position sizing and monitor for further decay. "
                    f"(ICIR={icir:.2f}, n_obs={n_obs})"
                ),
                rolling_ic_30=rolling_ic,
                icir=icir,
                ic_trend=ic_trend,
                needs_retraining=needs_retraining,
            )

        # INFO: win-rate-based retraining flag (existing criterion)
        if needs_retraining and n_obs >= self._min_observations:
            return DegradationAlert(
                agent_id=agent_id,
                severity=AlertSeverity.INFO,
                message=(
                    f"Agent flagged for retraining via win-rate criterion. "
                    f"Rolling IC(30)={rolling_ic:.4f} (not yet negative). "
                    f"Review signal logic before next session."
                ),
                rolling_ic_30=rolling_ic,
                icir=icir,
                ic_trend=ic_trend,
                needs_retraining=needs_retraining,
            )

        return None  # Healthy

    # -------------------------------------------------------------------------
    # Discord webhook
    # -------------------------------------------------------------------------

    def _send_discord_alert(self, report: DegradationReport) -> None:
        """
        Post degradation alerts to Discord via Incoming Webhook.

        Uses embeds for formatted, coloured output.
        Color codes: red=critical, yellow=warning, blue=info, green=clean.
        """
        if not self._webhook_url:
            logger.debug("[MONITOR] DISCORD_WEBHOOK_URL not set — skipping alert")
            return

        color_map = {
            "critical": 0xFF0000,  # Red
            "warning": 0xFFAA00,  # Amber
            "info": 0x3498DB,  # Blue
            "clean": 0x2ECC71,  # Green
        }

        status = report.overall_status
        embeds = []

        # Summary embed
        embeds.append(
            {
                "title": f"QuantPod Alpha Monitor — {status.upper()}",
                "color": color_map.get(status, 0x95A5A6),
                "timestamp": report.checked_at.isoformat(),
                "fields": [
                    {
                        "name": "Agents Checked",
                        "value": str(report.n_agents_checked),
                        "inline": True,
                    },
                    {
                        "name": "Alerts",
                        "value": str(len(report.alerts)),
                        "inline": True,
                    },
                ],
            }
        )

        # One embed per alert (cap at 5 to avoid Discord 10-embed limit)
        for alert in report.alerts[:5]:
            embed_color = color_map.get(alert.severity.value, 0x95A5A6)
            embeds.append(
                {
                    "title": f"{alert.emoji} {alert.agent_id} — {alert.severity.value.upper()}",
                    "description": alert.message,
                    "color": embed_color,
                    "fields": [
                        {
                            "name": "Rolling IC(30)",
                            "value": f"{alert.rolling_ic_30:.4f}",
                            "inline": True,
                        },
                        {"name": "ICIR", "value": f"{alert.icir:.3f}", "inline": True},
                        {"name": "IC Trend", "value": alert.ic_trend, "inline": True},
                    ],
                }
            )

        payload = {
            "username": "QuantPod Monitor",
            "embeds": embeds,
        }

        try:
            resp = requests.post(
                self._webhook_url,
                json=payload,
                timeout=10,
            )
            resp.raise_for_status()
            logger.info(f"[MONITOR] Discord alert sent ({status})")
        except Exception as e:
            logger.error(f"[MONITOR] Discord webhook failed: {e}")
            # Do not re-raise — a broken alert channel must not halt trading


    def check_strategy_drift(
        self,
        strategy_ids: list[str],
        signal_features: dict[str, dict] | None = None,
    ) -> list[DegradationAlert]:
        """
        Check drift for a list of strategies and fire alerts.

        Called after check_all_agents() in monitoring flows. Reuses the same
        Discord alert channel.

        Args:
            strategy_ids: Strategies to check.
            signal_features: Optional pre-computed features per strategy_id.
                If None, drift check is skipped (features must come from
                SignalEngine run — can't be manufactured here).

        Returns:
            List of DegradationAlert for strategies with drift.
        """
        if signal_features is None:
            return []

        try:
            from quant_pod.learning.drift_detector import DriftDetector
        except ImportError:
            return []

        detector = DriftDetector()
        alerts: list[DegradationAlert] = []

        for sid in strategy_ids:
            if not detector.has_baseline(sid):
                continue

            features = signal_features.get(sid)
            if not features:
                continue

            report = detector.check_drift(sid, features)
            if report.severity == "NONE":
                continue

            severity = (
                AlertSeverity.CRITICAL
                if report.severity == "CRITICAL"
                else AlertSeverity.WARNING
            )
            alerts.append(DegradationAlert(
                agent_id=f"drift:{sid}",
                severity=severity,
                message=(
                    f"[DRIFT] Strategy {sid}: PSI={report.overall_psi:.3f} "
                    f"({report.severity}). Drifted features: {report.drifted_features}"
                ),
                detail={
                    "type": "signal_drift",
                    "strategy_id": sid,
                    "overall_psi": report.overall_psi,
                    "feature_psis": report.feature_psis,
                    "drifted_features": report.drifted_features,
                },
            ))

        if alerts:
            logger.warning(
                f"[MONITOR] {len(alerts)} drift alerts: "
                f"{sum(1 for a in alerts if a.severity == AlertSeverity.CRITICAL)} critical"
            )

        return alerts


def get_alpha_monitor() -> AlphaMonitor:
    """Convenience factory — reads DISCORD_WEBHOOK_URL from environment."""
    return AlphaMonitor()
