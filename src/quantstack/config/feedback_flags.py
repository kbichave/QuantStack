# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Centralized feedback loop kill-switch flags (Section 16, Phase 7).

Each flag independently enables/disables one feedback adjustment.
Default: false (safe). Enable one at a time after verifying data accumulation.
Data collection runs regardless of these flags.

Why functions instead of module-level constants: env vars may be changed
at runtime (e.g., supervisor command). Functions re-read on each call.
"""

from __future__ import annotations

import os


def _flag(name: str) -> bool:
    """Read a boolean env var. Defaults to False (safe-off)."""
    return os.environ.get(name, "false").lower() in ("true", "1", "yes")


def correlation_penalty_enabled() -> bool:
    """Section 08: Signal correlation penalties."""
    return _flag("FEEDBACK_CORRELATION_PENALTY")


def sharpe_demotion_enabled() -> bool:
    """Section 12: Live vs. backtest Sharpe demotion."""
    return _flag("FEEDBACK_SHARPE_DEMOTION")


def drift_detection_enabled() -> bool:
    """Section 13: Concept drift detection and auto-retrain."""
    return _flag("FEEDBACK_DRIFT_DETECTION")


def transition_sizing_enabled() -> bool:
    """Section 15: Regime transition probability sizing."""
    return _flag("FEEDBACK_TRANSITION_SIZING")


def regime_affinity_sizing_enabled() -> bool:
    """P00 Wire 2: Scale position size by OutcomeTracker regime affinity."""
    return _flag("FEEDBACK_REGIME_AFFINITY_SIZING")


def skill_confidence_enabled() -> bool:
    """P00 Wire 4: Adjust conviction by SkillTracker confidence score."""
    return _flag("FEEDBACK_SKILL_CONFIDENCE")


# ---------------------------------------------------------------------------
# P01: Signal Statistical Rigor
# ---------------------------------------------------------------------------


def ic_gate_enabled() -> bool:
    """P01 §1.1: Auto-disable collectors with rolling 63d IC < 0.02."""
    return _flag("FEEDBACK_IC_GATE")


def signal_ci_enabled() -> bool:
    """P01 §1.2: Bootstrap confidence intervals on conviction."""
    return _flag("FEEDBACK_SIGNAL_CI")


def signal_decay_enabled() -> bool:
    """P01 §1.3: Exponential decay on cached signal conviction."""
    return _flag("FEEDBACK_SIGNAL_DECAY")


# ---------------------------------------------------------------------------
# P05: Adaptive Signal Synthesis
# ---------------------------------------------------------------------------


def ic_driven_weights_enabled() -> bool:
    """P05 §5.1: Full IC-driven regime-conditioned weights (replaces 80/20 EWMA blend)."""
    return _flag("FEEDBACK_IC_DRIVEN_WEIGHTS")


def transition_signal_dampening_enabled() -> bool:
    """P05 §5.2: Halve composite signal score during regime transitions (P(transition) > 0.3)."""
    return _flag("FEEDBACK_TRANSITION_SIGNAL_DAMPENING")


def ensemble_ab_test_enabled() -> bool:
    """P05 §5.4: A/B test ensemble aggregation methods (median, trimmed mean vs weighted avg)."""
    return _flag("FEEDBACK_ENSEMBLE_AB_TEST")


def ensemble_active_method() -> str:
    """P05 §6.4: Currently promoted ensemble method (weighted_avg | weighted_median | trimmed_mean)."""
    return os.environ.get("ENSEMBLE_ACTIVE_METHOD", "weighted_avg")


def transition_position_sizing_enabled() -> bool:
    """P05 §4: Halve position sizing during regime transitions."""
    return _flag("FEEDBACK_TRANSITION_POSITION_SIZING")


# ---------------------------------------------------------------------------
# P10: Meta-Learning & Self-Improvement
# ---------------------------------------------------------------------------


def prompt_ab_testing_enabled() -> bool:
    """P10 §2: A/B test prompt variants per agent."""
    return _flag("FEEDBACK_PROMPT_AB_TESTING")


def meta_strategy_allocation_enabled() -> bool:
    """P10 §3: Strategy-of-strategies meta-model for capital allocation."""
    return _flag("FEEDBACK_META_STRATEGY_ALLOCATION")


def research_priority_scoring_enabled() -> bool:
    """P10 §4: Score-based research prioritization (replaces FIFO queue)."""
    return _flag("FEEDBACK_RESEARCH_PRIORITY_SCORING")


# ---------------------------------------------------------------------------
# P12: Multi-Asset Expansion
# ---------------------------------------------------------------------------


def multi_asset_enabled() -> bool:
    """P12 §1: Enable multi-asset class trading (futures, crypto, forex)."""
    return _flag("MULTI_ASSET_ENABLED")


# ---------------------------------------------------------------------------
# P14: Advanced ML
# ---------------------------------------------------------------------------


def conformal_prediction_enabled() -> bool:
    """P14 §1: Conformal prediction intervals for position sizing."""
    return _flag("FEEDBACK_CONFORMAL_PREDICTION")


def transformer_forecast_enabled() -> bool:
    """P14 §2: Transformer time-series signal collector."""
    return _flag("FEEDBACK_TRANSFORMER_FORECAST")


# ---------------------------------------------------------------------------
# P15: Autonomous Fund
# ---------------------------------------------------------------------------


def feedback_loops_enabled() -> bool:
    """P15 §2: Closed feedback loops (trade loss->research, IC->weight, etc.)."""
    return _flag("FEEDBACK_LOOPS_ENABLED")


def authority_gate_enabled() -> bool:
    """P15 §3: Authority matrix decision ceilings."""
    return _flag("AUTHORITY_GATE_ENABLED")
