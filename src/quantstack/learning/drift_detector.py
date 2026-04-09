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
  - Baselines stored as JSON in ~/.quantstack/drift_baselines/.
  - All I/O is best-effort: missing baselines return NONE (not an error).
  - No DB writes in the hot path; only reads + numpy math.

References:
  Siddiqi, N. (2006). "Credit Risk Scorecards." Wiley.
  Yurdakul, B. (2018). "Statistical Properties of Population Stability Index."
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PSI_WARNING = 0.10
PSI_CRITICAL = 0.25
BASELINE_DIR = Path.home() / ".quantstack" / "drift_baselines"

# Default features to track — fallback when baseline has no feature list
DEFAULT_TRACKED_FEATURES = [
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


@dataclass
class ICDriftReport:
    """Result of IC-based concept drift check."""

    z_scores: dict[str, float]
    drifted_features: list[str] = field(default_factory=list)

    @staticmethod
    def no_op() -> ICDriftReport:
        return ICDriftReport(z_scores={}, drifted_features=[])


@dataclass
class LabelDriftReport:
    """Result of label drift check (KS test on return distributions)."""

    ks_statistic: float = 0.0
    p_value: float = 1.0
    is_drifted: bool = False
    training_mean: float = 0.0
    training_std: float = 0.0
    recent_mean: float = 0.0
    recent_std: float = 0.0

    @staticmethod
    def no_op() -> LabelDriftReport:
        return LabelDriftReport()


@dataclass
class InteractionDriftReport:
    """Result of adversarial validation for interaction drift."""

    auc: float = 0.5
    is_drifted: bool = False

    @staticmethod
    def no_op() -> InteractionDriftReport:
        return InteractionDriftReport()


@dataclass
class RetrainDecision:
    """Output of the auto-retrain decision tree."""

    should_retrain: bool
    reason: str
    data_window: int | None = None
    publish_event: bool = False
    event_payload: dict | None = None


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

    Baselines are JSON files at ~/.quantstack/drift_baselines/{strategy_id}.json
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
        self._threshold_cache: dict[str, dict[str, list[float]]] = {}

    # -----------------------------------------------------------------------
    # Baseline management
    # -----------------------------------------------------------------------

    def set_baseline(
        self,
        strategy_id: str,
        features: dict[str, np.ndarray],
        calibrate_thresholds: bool = True,
        n_bootstrap: int = 50,
    ) -> None:
        """
        Record training-period feature distributions as baseline.

        When calibrate_thresholds=True, computes per-feature PSI thresholds
        by bootstrapping self-PSI (splitting baseline in half repeatedly).
        Thresholds are floored at the global PSI_WARNING/PSI_CRITICAL defaults
        to never weaken detection.

        Args:
            strategy_id: Strategy identifier.
            features: Dict mapping feature names to 1-D sample arrays.
            calibrate_thresholds: Whether to compute per-feature thresholds.
            n_bootstrap: Number of bootstrap splits for calibration.
        """
        self._baseline_dir.mkdir(parents=True, exist_ok=True)
        path = self._baseline_dir / f"{strategy_id}.json"

        serializable: dict[str, Any] = {}
        for name, arr in features.items():
            arr = np.asarray(arr, dtype=np.float64).ravel()
            arr = arr[np.isfinite(arr)]
            if len(arr) > 0:
                serializable[name] = arr.tolist()

        # Per-feature threshold calibration via bootstrap self-PSI
        thresholds: dict[str, list[float]] = {}
        if calibrate_thresholds:
            rng = np.random.default_rng(42)
            for name, values in serializable.items():
                arr = np.array(values, dtype=np.float64)
                if len(arr) < 20:
                    continue
                self_psis = []
                for _ in range(n_bootstrap):
                    idx = rng.permutation(len(arr))
                    half = len(arr) // 2
                    psi_val = compute_psi(arr[idx[:half]], arr[idx[half:]])
                    self_psis.append(psi_val)

                self_psis_sorted = sorted(self_psis)
                p95 = self_psis_sorted[int(0.95 * len(self_psis_sorted))]
                p99 = self_psis_sorted[int(0.99 * len(self_psis_sorted))]

                # Floor at global defaults — never weaken
                warn_thresh = max(PSI_WARNING, p95 * 1.5)
                crit_thresh = max(PSI_CRITICAL, p99 * 1.5)
                thresholds[name] = [warn_thresh, crit_thresh]

        baseline_data: dict[str, Any] = {
            "_features": serializable,
            "_thresholds": thresholds,
        }
        # Top-level keys are the feature arrays (backward compat)
        baseline_data.update(serializable)

        path.write_text(json.dumps(baseline_data, indent=2))
        self._cache[strategy_id] = {k: np.array(v) for k, v in serializable.items()}
        self._threshold_cache[strategy_id] = thresholds
        logger.info(
            f"[DriftDetector] Baseline set for {strategy_id}: "
            f"{list(serializable.keys())} ({len(thresholds)} calibrated thresholds)"
        )

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
            features_raw = raw["_features"]
            thresholds = raw.get("_thresholds", {})

            baseline = {
                k: np.array(v, dtype=np.float64)
                for k, v in features_raw.items()
                if isinstance(v, list)
            }
            self._cache[strategy_id] = baseline
            self._threshold_cache[strategy_id] = thresholds
            return baseline
        except Exception as exc:
            logger.warning(
                f"[DriftDetector] Failed to load baseline for {strategy_id}: {exc}"
            )
            return None

    def _get_tracked_features(self, strategy_id: str) -> list[str]:
        """Return feature names to track for this strategy.

        Uses all features present in the baseline file when available,
        falling back to DEFAULT_TRACKED_FEATURES otherwise.
        """
        baseline = self._load_baseline(strategy_id)
        if baseline:
            return list(baseline.keys())
        return list(DEFAULT_TRACKED_FEATURES)

    def _get_feature_thresholds(
        self, strategy_id: str, feature_name: str
    ) -> tuple[float, float]:
        """Return (warning, critical) PSI thresholds for a feature.

        Uses per-feature calibrated thresholds when available,
        falling back to global defaults.
        """
        thresholds = self._threshold_cache.get(strategy_id, {})
        if feature_name in thresholds:
            return tuple(thresholds[feature_name])  # type: ignore[return-value]
        return (PSI_WARNING, PSI_CRITICAL)

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

        Uses all features present in the baseline (dynamic), not just the
        default set. Per-feature calibrated thresholds are used when available.

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

        tracked = self._get_tracked_features(strategy_id)

        feature_psis: dict[str, float] = {}
        drifted: list[str] = []

        for name in tracked:
            if name not in baseline or name not in features:
                continue

            expected = baseline[name]
            actual = np.asarray(features[name], dtype=np.float64).ravel()

            if len(expected) < 2 or len(actual) < 2:
                continue

            psi_val = compute_psi(expected, actual)
            feature_psis[name] = psi_val

            warn_thresh, _ = self._get_feature_thresholds(strategy_id, name)
            if psi_val >= warn_thresh:
                drifted.append(name)

        overall_psi = max(feature_psis.values()) if feature_psis else 0.0

        # Determine severity using the most sensitive threshold across features
        severity = "NONE"
        for name, psi_val in feature_psis.items():
            _, crit_thresh = self._get_feature_thresholds(strategy_id, name)
            warn_thresh, _ = self._get_feature_thresholds(strategy_id, name)
            if psi_val >= crit_thresh:
                severity = "CRITICAL"
                break
            elif psi_val >= warn_thresh and severity != "CRITICAL":
                severity = "WARNING"

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

    # -------------------------------------------------------------------
    # Layer 1: IC-based concept drift (daily)
    # -------------------------------------------------------------------

    def check_ic_drift(
        self,
        current_ic: dict[str, float],
        baseline_ic: dict[str, tuple[float, float]],
        z_threshold: float = 2.0,
    ) -> ICDriftReport:
        """Compare rolling IC against baseline IC statistics.

        Args:
            current_ic: {feature_name: current_rolling_ic}.
            baseline_ic: {feature_name: (baseline_mean, baseline_std)}.
            z_threshold: Number of std deviations to flag drift.

        Returns:
            ICDriftReport with per-feature z-scores and drifted list.
        """
        z_scores: dict[str, float] = {}
        drifted: list[str] = []

        for feature, (mean, std) in baseline_ic.items():
            if feature not in current_ic or std <= 0:
                continue
            z = (mean - current_ic[feature]) / std
            z_scores[feature] = round(z, 4)
            if z > z_threshold:
                drifted.append(feature)

        return ICDriftReport(z_scores=z_scores, drifted_features=drifted)

    def check_ic_drift_gated(
        self,
        current_ic: dict[str, float],
        baseline_ic: dict[str, tuple[float, float]],
    ) -> ICDriftReport:
        """Config-flag-gated wrapper for check_ic_drift."""
        if os.getenv("FEEDBACK_DRIFT_DETECTION", "false").lower() != "true":
            return ICDriftReport.no_op()
        return self.check_ic_drift(current_ic, baseline_ic)

    def record_rolling_ic(
        self,
        strategy_id: str,
        feature_ics: dict[str, float],
    ) -> None:
        """Append daily rolling IC values for drift monitoring.

        Stored at ~/.quantstack/drift_baselines/{strategy_id}_ic_history.json
        as a list of {date, ics} entries. The first 60 entries serve as the
        baseline for check_ic_drift when no explicit baseline is provided.
        """
        self._baseline_dir.mkdir(parents=True, exist_ok=True)
        path = self._baseline_dir / f"{strategy_id}_ic_history.json"

        history: list[dict] = []
        if path.exists():
            try:
                history = json.loads(path.read_text())
            except Exception:
                history = []

        history.append({
            "date": date.today().isoformat(),
            "ics": {k: round(v, 6) for k, v in feature_ics.items()},
        })

        path.write_text(json.dumps(history, indent=2))

    def load_ic_baseline(
        self,
        strategy_id: str,
        warmup_entries: int = 60,
    ) -> dict[str, tuple[float, float]]:
        """Compute IC baseline statistics from the first N history entries.

        Returns {feature_name: (mean_ic, std_ic)} from the warmup period.
        Returns {} if insufficient history.
        """
        path = self._baseline_dir / f"{strategy_id}_ic_history.json"
        if not path.exists():
            return {}

        try:
            history = json.loads(path.read_text())
        except Exception:
            return {}

        if len(history) < warmup_entries:
            return {}

        warmup = history[:warmup_entries]
        # Collect per-feature IC arrays
        feature_arrays: dict[str, list[float]] = {}
        for entry in warmup:
            for feat, ic in entry.get("ics", {}).items():
                feature_arrays.setdefault(feat, []).append(ic)

        baseline: dict[str, tuple[float, float]] = {}
        for feat, values in feature_arrays.items():
            arr = np.array(values, dtype=np.float64)
            arr = arr[np.isfinite(arr)]
            if len(arr) >= 10:
                baseline[feat] = (float(arr.mean()), float(max(arr.std(), 0.005)))

        return baseline

    # -------------------------------------------------------------------
    # Layer 2: Label drift (weekly, KS test)
    # -------------------------------------------------------------------

    def check_label_drift(
        self,
        training_returns: np.ndarray,
        recent_returns: np.ndarray,
        p_threshold: float = 0.01,
    ) -> LabelDriftReport:
        """Two-sample KS test on return distributions (pure numpy).

        Uses the asymptotic approximation: p ~ 2*exp(-2*n_eff*D^2).

        Args:
            training_returns: Return distribution from training period.
            recent_returns: Rolling recent return distribution.
            p_threshold: p-value below which label drift is flagged.

        Returns:
            LabelDriftReport.
        """
        training = np.asarray(training_returns, dtype=np.float64).ravel()
        recent = np.asarray(recent_returns, dtype=np.float64).ravel()
        training = training[np.isfinite(training)]
        recent = recent[np.isfinite(recent)]

        if len(training) < 10 or len(recent) < 10:
            return LabelDriftReport()

        # Two-sample KS statistic (same approach as compute_psi)
        combined = np.concatenate([training, recent])
        combined.sort()
        n1, n2 = len(training), len(recent)
        cdf1 = np.searchsorted(np.sort(training), combined, side="right") / n1
        cdf2 = np.searchsorted(np.sort(recent), combined, side="right") / n2
        ks_stat = float(np.max(np.abs(cdf1 - cdf2)))

        # Asymptotic p-value: p ~ 2 * exp(-2 * n_eff * D^2)
        n_eff = (n1 * n2) / (n1 + n2)
        p_value = 2.0 * math.exp(-2.0 * n_eff * ks_stat ** 2)
        p_value = min(1.0, max(0.0, p_value))

        return LabelDriftReport(
            ks_statistic=round(ks_stat, 6),
            p_value=round(p_value, 8),
            is_drifted=p_value < p_threshold,
            training_mean=float(np.mean(training)),
            training_std=float(np.std(training)),
            recent_mean=float(np.mean(recent)),
            recent_std=float(np.std(recent)),
        )

    def check_label_drift_gated(
        self,
        training_returns: np.ndarray,
        recent_returns: np.ndarray,
    ) -> LabelDriftReport:
        """Config-flag-gated wrapper for check_label_drift."""
        if os.getenv("FEEDBACK_DRIFT_DETECTION", "false").lower() != "true":
            return LabelDriftReport.no_op()
        return self.check_label_drift(training_returns, recent_returns)

    # -------------------------------------------------------------------
    # Layer 3: Interaction drift (monthly, adversarial validation)
    # -------------------------------------------------------------------

    def check_interaction_drift(
        self,
        training_data: np.ndarray,
        recent_data: np.ndarray,
        auc_threshold: float = 0.60,
    ) -> InteractionDriftReport:
        """Adversarial validation: can a classifier tell training from recent?

        Uses logistic regression via numpy (no sklearn required).
        AUC > threshold means the joint feature-target distribution has shifted.

        Args:
            training_data: Feature-target pairs from training (n_samples, n_cols).
            recent_data: Feature-target pairs from recent period.
            auc_threshold: AUC above which drift is flagged.

        Returns:
            InteractionDriftReport.
        """
        training = np.asarray(training_data, dtype=np.float64)
        recent = np.asarray(recent_data, dtype=np.float64)

        if len(training) < 20 or len(recent) < 20:
            return InteractionDriftReport()

        # Label: 0 = training, 1 = recent
        X = np.vstack([training, recent])
        y = np.concatenate([np.zeros(len(training)), np.ones(len(recent))])

        # Standardize features
        mu = X.mean(axis=0)
        sigma = X.std(axis=0)
        sigma[sigma == 0] = 1.0
        X = (X - mu) / sigma

        # Add intercept
        X = np.column_stack([np.ones(len(X)), X])

        # Train/test split (first 70% train, last 30% test)
        n = len(X)
        split = int(0.7 * n)
        # Shuffle with deterministic seed
        rng = np.random.default_rng(42)
        idx = rng.permutation(n)
        X_train, X_test = X[idx[:split]], X[idx[split:]]
        y_train, y_test = y[idx[:split]], y[idx[split:]]

        # Logistic regression via gradient descent
        auc = _logistic_auc(X_train, y_train, X_test, y_test)

        return InteractionDriftReport(
            auc=round(auc, 4),
            is_drifted=auc > auc_threshold,
        )


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
        if current_regime and regime_affinity and current_regime not in regime_affinity:
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
            warnings.append(
                f"{consecutive_losses} consecutive losses — circuit breaker"
            )
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


# ---------------------------------------------------------------------------
# Auto-retrain decision tree
# ---------------------------------------------------------------------------

_RETRAIN_COOLDOWN_DAYS = 20
_IC_HEALTHY_THRESHOLD = 0.01
_ABRUPT_WINDOW = 5  # days


def evaluate_retrain_decision(
    current_ic: float,
    ic_history: list[float],
    last_retrain_date: date | None,
    ic_drift_report: ICDriftReport | None,
) -> RetrainDecision:
    """Decide whether to auto-retrain based on IC trajectory.

    Decision tree:
      1. Cooldown: if last retrain < 20 trading days ago -> no.
      2. Benign covariate shift: feature drift but IC healthy (> 0.01) -> no.
      3. Abrupt IC drop (step change in 5 days) -> no retrain, publish event.
      4. Gradual IC decline (60+ day negative slope) -> retrain with 252-day window.
    """
    # 1. Cooldown check
    if last_retrain_date is not None:
        days_since = (date.today() - last_retrain_date).days
        if days_since < _RETRAIN_COOLDOWN_DAYS:
            return RetrainDecision(should_retrain=False, reason="cooldown")

    # 2. Benign covariate shift: features drifted but IC still healthy
    if (
        ic_drift_report is not None
        and len(ic_drift_report.drifted_features) > 0
        and current_ic > _IC_HEALTHY_THRESHOLD
    ):
        return RetrainDecision(
            should_retrain=False, reason="benign_covariate_shift"
        )

    # Need enough history for slope analysis
    if len(ic_history) < 60:
        return RetrainDecision(should_retrain=False, reason="insufficient_history")

    # 3. Abrupt shift detection: check if IC dropped sharply in a 5-day window
    recent_window = ic_history[-_ABRUPT_WINDOW:]
    earlier = ic_history[-60:-_ABRUPT_WINDOW]
    if len(earlier) > 0:
        earlier_std = float(np.std(earlier)) if len(earlier) > 1 else 0.005
        # Use a minimum std floor to handle constant-IC histories
        effective_std = max(earlier_std, 0.005)
        earlier_mean = float(np.mean(earlier))
        recent_mean = float(np.mean(recent_window))
        drop = earlier_mean - recent_mean
        if drop > 0 and drop / effective_std > 2.0:
            return RetrainDecision(
                should_retrain=False,
                reason="abrupt_shift",
                publish_event=True,
                event_payload={
                    "type": "MODEL_DEGRADATION",
                    "current_ic": current_ic,
                    "drop_magnitude": round(earlier_mean - recent_mean, 6),
                },
            )

    # 4. Gradual decline: linear regression slope over 60+ days
    x = np.arange(len(ic_history), dtype=np.float64)
    y = np.array(ic_history, dtype=np.float64)
    # Linear regression: slope and R-squared
    x_mean = x.mean()
    y_mean = y.mean()
    ss_xy = float(np.sum((x - x_mean) * (y - y_mean)))
    ss_xx = float(np.sum((x - x_mean) ** 2))
    ss_yy = float(np.sum((y - y_mean) ** 2))
    slope = ss_xy / ss_xx if ss_xx > 0 else 0.0
    r_squared = (ss_xy ** 2) / (ss_xx * ss_yy) if ss_xx > 0 and ss_yy > 0 else 0.0

    if slope < 0 and r_squared > 0.5 and current_ic < _IC_HEALTHY_THRESHOLD:
        return RetrainDecision(
            should_retrain=True,
            reason="gradual_ic_decline",
            data_window=252,
        )

    return RetrainDecision(should_retrain=False, reason="no_action_needed")


# ---------------------------------------------------------------------------
# Logistic regression helper (pure numpy, no sklearn dependency)
# ---------------------------------------------------------------------------


def _logistic_auc(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    lr: float = 0.1,
    n_iter: int = 200,
) -> float:
    """Train logistic regression and compute AUC on test set.

    Uses gradient descent. Returns AUC (0.5 = random, 1.0 = perfect).
    """
    n_features = X_train.shape[1]
    w = np.zeros(n_features)

    for _ in range(n_iter):
        z = X_train @ w
        # Clip to avoid overflow
        z = np.clip(z, -20, 20)
        pred = 1.0 / (1.0 + np.exp(-z))
        grad = X_train.T @ (pred - y_train) / len(y_train)
        w -= lr * grad

    # Predict probabilities on test set
    z_test = np.clip(X_test @ w, -20, 20)
    probs = 1.0 / (1.0 + np.exp(-z_test))

    # Compute AUC via rank-sum (Wilcoxon-Mann-Whitney)
    pos_probs = probs[y_test == 1]
    neg_probs = probs[y_test == 0]
    if len(pos_probs) == 0 or len(neg_probs) == 0:
        return 0.5

    auc = 0.0
    for p in pos_probs:
        auc += np.sum(p > neg_probs) + 0.5 * np.sum(p == neg_probs)
    auc /= len(pos_probs) * len(neg_probs)
    return float(auc)


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
    for key in DEFAULT_TRACKED_FEATURES:
        val = raw_collectors.get(key) or brief.get(key)
        if val is not None:
            try:
                arr = np.atleast_1d(np.asarray(val, dtype=np.float64))
                if np.all(np.isfinite(arr)):
                    features[key] = arr
            except (ValueError, TypeError):
                continue

    return features
