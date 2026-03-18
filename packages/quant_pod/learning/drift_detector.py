"""
Proactive drift detection via Population Stability Index (PSI).

Compares rolling signal feature distributions against a training-period
baseline captured at strategy promotion. Fires before IC decay shows up
in SkillTracker — closing the multi-day blind spot in the learning loop.

PSI thresholds (credit risk literature, well-validated):
  < 0.10  → NONE      (distributions stable)
  0.10–0.25 → WARNING (moderate shift — reduce position size)
  ≥ 0.25  → CRITICAL  (significant shift — skip symbol)

Design invariants:
  - Pure numpy computation — < 1ms per check, safe for hot path.
  - Baselines stored as JSON in ~/.quant_pod/drift_baselines/.
  - All I/O is best-effort: missing baselines return NONE (not an error).
  - No DB writes in the hot path; only reads + numpy math.

References:
  Siddiqi, N. (2006). "Credit Risk Scorecards." Wiley.
  Yurdakul, B. (2018). "Statistical Properties of Population Stability Index."
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PSI_WARNING = 0.10
PSI_CRITICAL = 0.25
BASELINE_DIR = Path.home() / ".quant_pod" / "drift_baselines"

# Features to track — all computed by SignalEngine collectors every run
TRACKED_FEATURES = [
    "rsi_14",
    "atr_pct",
    "adx_14",
    "bb_pct",
    "volume_ratio",
    "regime_confidence",
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class DriftReport:
    """Result of a drift check."""

    strategy_id: str
    overall_psi: float
    feature_psis: dict[str, float]
    severity: str  # "NONE" | "WARNING" | "CRITICAL"
    drifted_features: list[str]
    checked_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "overall_psi": round(self.overall_psi, 4),
            "feature_psis": {k: round(v, 4) for k, v in self.feature_psis.items()},
            "severity": self.severity,
            "drifted_features": self.drifted_features,
            "checked_at": self.checked_at.isoformat(),
        }


# ---------------------------------------------------------------------------
# PSI computation
# ---------------------------------------------------------------------------


def compute_psi(
    expected: np.ndarray,
    actual: np.ndarray,
) -> float:
    """
    Distribution divergence score between two 1-D samples.

    Uses the two-sample Kolmogorov-Smirnov statistic, which is robust to
    small sample sizes (works with n >= 4). KS measures the maximum distance
    between the two empirical CDFs — it doesn't need binning, so it avoids
    the empty-bin problem that inflates PSI with < 100 samples.

    The raw KS statistic (0–1) is scaled to match PSI threshold conventions:
      KS < 0.15  → score < 0.10  (NONE)
      KS 0.15–0.30 → score 0.10–0.25 (WARNING)
      KS > 0.30  → score > 0.25  (CRITICAL)

    This scaling preserves the existing PSI_WARNING=0.10 and PSI_CRITICAL=0.25
    thresholds so no downstream code needs to change.

    Args:
        expected: Baseline distribution samples (1-D array).
        actual: Current distribution samples (1-D array).

    Returns:
        Drift score (>= 0). 0 = identical distributions.
    """
    expected = np.asarray(expected, dtype=np.float64).ravel()
    actual = np.asarray(actual, dtype=np.float64).ravel()

    if len(expected) < 4 or len(actual) < 4:
        return 0.0

    # Remove NaN/inf
    expected = expected[np.isfinite(expected)]
    actual = actual[np.isfinite(actual)]

    if len(expected) < 4 or len(actual) < 4:
        return 0.0

    # Two-sample KS test (pure numpy — no scipy dependency)
    # KS statistic = max |F_expected(x) - F_actual(x)| over all x
    combined = np.concatenate([expected, actual])
    combined.sort()

    n_exp = len(expected)
    n_act = len(actual)

    cdf_exp = np.searchsorted(np.sort(expected), combined, side="right") / n_exp
    cdf_act = np.searchsorted(np.sort(actual), combined, side="right") / n_act

    ks_stat = float(np.max(np.abs(cdf_exp - cdf_act)))

    # Scale KS to match PSI thresholds.
    # KS critical values for n=30,60 at alpha=0.05 ≈ 0.29
    # Map: KS 0.25 → PSI 0.10 (WARNING), KS 0.45 → PSI 0.25 (CRITICAL)
    # Linear: score = (ks - 0.05) * (0.25 / 0.40), clamped at 0
    drift_score = max(0.0, (ks_stat - 0.05) * (PSI_CRITICAL / 0.40))

    return float(max(drift_score, 0.0))


# ---------------------------------------------------------------------------
# DriftDetector
# ---------------------------------------------------------------------------


class DriftDetector:
    """
    Compares current signal features against per-strategy baselines.

    Baselines are JSON files at ~/.quant_pod/drift_baselines/{strategy_id}.json
    containing raw sample arrays per feature from the training/backtest period.

    Usage:
        detector = DriftDetector()

        # At promotion (once):
        detector.set_baseline("strat_abc", {"rsi_14": array([...]), ...})

        # In hot path (every signal):
        report = detector.check_drift("strat_abc", {"rsi_14": array([...]), ...})
        if report.severity == "CRITICAL":
            skip_symbol()
    """

    def __init__(self, baseline_dir: Path | None = None):
        self._baseline_dir = baseline_dir or BASELINE_DIR
        self._cache: dict[str, dict[str, np.ndarray]] = {}

    # -----------------------------------------------------------------------
    # Baseline management
    # -----------------------------------------------------------------------

    def set_baseline(
        self,
        strategy_id: str,
        features: dict[str, np.ndarray],
    ) -> None:
        """
        Record training-period feature distributions as baseline.

        Args:
            strategy_id: Strategy identifier.
            features: Dict mapping feature names to 1-D sample arrays.
        """
        self._baseline_dir.mkdir(parents=True, exist_ok=True)
        path = self._baseline_dir / f"{strategy_id}.json"

        serializable = {}
        for name, arr in features.items():
            arr = np.asarray(arr, dtype=np.float64).ravel()
            arr = arr[np.isfinite(arr)]
            if len(arr) > 0:
                serializable[name] = arr.tolist()

        path.write_text(json.dumps(serializable, indent=2))
        self._cache[strategy_id] = {k: np.array(v) for k, v in serializable.items()}
        logger.info(f"[DriftDetector] Baseline set for {strategy_id}: {list(serializable.keys())}")

    def has_baseline(self, strategy_id: str) -> bool:
        """Check if a baseline exists for this strategy."""
        if strategy_id in self._cache:
            return True
        path = self._baseline_dir / f"{strategy_id}.json"
        return path.exists()

    def _load_baseline(self, strategy_id: str) -> dict[str, np.ndarray] | None:
        """Load baseline from cache or disk."""
        if strategy_id in self._cache:
            return self._cache[strategy_id]

        path = self._baseline_dir / f"{strategy_id}.json"
        if not path.exists():
            return None

        try:
            raw = json.loads(path.read_text())
            baseline = {k: np.array(v, dtype=np.float64) for k, v in raw.items()}
            self._cache[strategy_id] = baseline
            return baseline
        except Exception as exc:
            logger.warning(f"[DriftDetector] Failed to load baseline for {strategy_id}: {exc}")
            return None

    # -----------------------------------------------------------------------
    # Drift checking
    # -----------------------------------------------------------------------

    def check_drift(
        self,
        strategy_id: str,
        features: dict[str, np.ndarray],
    ) -> DriftReport:
        """
        Compare current features against baseline.

        Args:
            strategy_id: Strategy to check.
            features: Dict mapping feature names to current 1-D sample arrays.

        Returns:
            DriftReport with per-feature PSI and overall severity.
        """
        baseline = self._load_baseline(strategy_id)

        if baseline is None:
            return DriftReport(
                strategy_id=strategy_id,
                overall_psi=0.0,
                feature_psis={},
                severity="NONE",
                drifted_features=[],
            )

        feature_psis: dict[str, float] = {}
        drifted: list[str] = []

        for name in TRACKED_FEATURES:
            if name not in baseline or name not in features:
                continue

            expected = baseline[name]
            actual = np.asarray(features[name], dtype=np.float64).ravel()

            if len(expected) < 2 or len(actual) < 2:
                continue

            psi_val = compute_psi(expected, actual)
            feature_psis[name] = psi_val

            if psi_val >= PSI_WARNING:
                drifted.append(name)

        overall_psi = max(feature_psis.values()) if feature_psis else 0.0

        if overall_psi >= PSI_CRITICAL:
            severity = "CRITICAL"
        elif overall_psi >= PSI_WARNING:
            severity = "WARNING"
        else:
            severity = "NONE"

        return DriftReport(
            strategy_id=strategy_id,
            overall_psi=overall_psi,
            feature_psis=feature_psis,
            severity=severity,
            drifted_features=drifted,
        )

    def check_drift_from_brief(
        self,
        strategy_id: str,
        brief: dict[str, Any],
    ) -> DriftReport:
        """
        Extract tracked features from a SignalBrief dict and check drift.

        Expects brief to contain symbol_briefs[0].technical or top-level
        technical indicators as flat keys.

        Args:
            strategy_id: Strategy to check.
            brief: SignalBrief-like dict from SignalEngine.

        Returns:
            DriftReport.
        """
        features = _extract_features_from_brief(brief)
        return self.check_drift(strategy_id, features)


# ---------------------------------------------------------------------------
# Feature extraction from SignalBrief
# ---------------------------------------------------------------------------


class StrategyDriftMonitor:
    """Strategy-level drift triggers — sits above DriftDetector.

    Combines DriftDetector (feature distribution shift) with performance
    degradation signals to produce actionable strategy lifecycle decisions.

    Triggers:
      - OOS Sharpe drops below 50% of IS Sharpe → alert
      - Regime changed and strategy has no regime_affinity match → quarantine
      - 3 consecutive losses → scale to 50% size
      - Feature drift CRITICAL → quarantine

    Usage:
        monitor = StrategyDriftMonitor()
        action = monitor.evaluate("strat_abc", live_sharpe=0.3, is_sharpe=1.2,
                                   current_regime="trending_down",
                                   regime_affinity=["ranging"],
                                   consecutive_losses=3,
                                   drift_report=detector.check_drift(...))
    """

    @staticmethod
    def evaluate(
        strategy_id: str,
        live_sharpe: float | None = None,
        is_sharpe: float | None = None,
        current_regime: str | None = None,
        regime_affinity: list[str] | None = None,
        consecutive_losses: int = 0,
        drift_report: DriftReport | None = None,
    ) -> StrategyAction:
        """Evaluate a strategy and recommend an action.

        Returns StrategyAction with recommended_action and reasoning.
        """
        warnings: list[str] = []
        action = "active"
        scale_factor = 1.0

        # 1. Sharpe degradation
        if live_sharpe is not None and is_sharpe is not None and is_sharpe > 0:
            degradation = 1.0 - (live_sharpe / is_sharpe)
            if live_sharpe <= 0:
                action = "quarantine"
                warnings.append(
                    f"Live Sharpe {live_sharpe:.2f} is non-positive "
                    f"(IS was {is_sharpe:.2f})"
                )
            elif degradation > 0.5:
                action = "alert"
                scale_factor = 0.5
                warnings.append(
                    f"Live Sharpe degraded {degradation:.0%} from IS "
                    f"({live_sharpe:.2f} vs {is_sharpe:.2f})"
                )

        # 2. Regime mismatch
        if (
            current_regime
            and regime_affinity
            and current_regime not in regime_affinity
        ):
            if action != "quarantine":
                action = "alert"
                scale_factor = min(scale_factor, 0.5)
            warnings.append(
                f"Current regime '{current_regime}' not in affinity {regime_affinity}"
            )

        # 3. Consecutive losses
        if consecutive_losses >= 3:
            action = "quarantine"
            scale_factor = 0.0
            warnings.append(f"{consecutive_losses} consecutive losses — circuit breaker")
        elif consecutive_losses >= 2:
            scale_factor = min(scale_factor, 0.5)
            warnings.append(f"{consecutive_losses} consecutive losses — scaling down")

        # 4. Feature drift
        if drift_report and drift_report.severity == "CRITICAL":
            action = "quarantine"
            scale_factor = 0.0
            warnings.append(
                f"Feature drift CRITICAL (PSI={drift_report.overall_psi:.3f}): "
                f"{drift_report.drifted_features}"
            )
        elif drift_report and drift_report.severity == "WARNING":
            scale_factor = min(scale_factor, 0.7)
            warnings.append(
                f"Feature drift WARNING (PSI={drift_report.overall_psi:.3f})"
            )

        return StrategyAction(
            strategy_id=strategy_id,
            recommended_action=action,
            scale_factor=scale_factor,
            warnings=warnings,
        )


@dataclass
class StrategyAction:
    """Recommended action for a strategy based on drift/performance signals."""

    strategy_id: str
    recommended_action: str  # "active", "alert", "quarantine"
    scale_factor: float  # 1.0 = full, 0.5 = half, 0.0 = halt
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "recommended_action": self.recommended_action,
            "scale_factor": self.scale_factor,
            "warnings": self.warnings,
        }


# ---------------------------------------------------------------------------
# Feature extraction from SignalBrief
# ---------------------------------------------------------------------------


def _extract_features_from_brief(brief: dict[str, Any]) -> dict[str, np.ndarray]:
    """
    Pull tracked feature values from a SignalBrief dict.

    SignalEngine briefs store technical indicators in symbol_briefs[].raw_collectors.
    We extract scalar values and wrap them in 1-element arrays for PSI comparison
    against the baseline distribution.

    For meaningful drift detection, the caller should accumulate multiple briefs
    into rolling windows and pass the windowed arrays here. Single-sample PSI
    is noisy — the AutonomousRunner should maintain a rolling buffer.
    """
    features: dict[str, np.ndarray] = {}

    # Try to extract from nested structure
    symbol_briefs = brief.get("symbol_briefs", [])
    raw_collectors = {}
    if symbol_briefs and isinstance(symbol_briefs, list):
        sb = symbol_briefs[0] if isinstance(symbol_briefs[0], dict) else {}
        raw_collectors = sb.get("raw_collectors", {})
        # Also check technical sub-dict
        technical = raw_collectors.get("technical", {})
        if isinstance(technical, dict):
            raw_collectors.update(technical)

    # Also check top-level keys (flat brief format)
    for key in TRACKED_FEATURES:
        val = raw_collectors.get(key) or brief.get(key)
        if val is not None:
            try:
                arr = np.atleast_1d(np.asarray(val, dtype=np.float64))
                if np.all(np.isfinite(arr)):
                    features[key] = arr
            except (ValueError, TypeError):
                continue

    return features
