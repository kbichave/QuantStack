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


def ic_weight_adjustment_enabled() -> bool:
    """Section 07: IC-based signal weight adjustment."""
    return _flag("FEEDBACK_IC_WEIGHT_ADJUSTMENT")


def correlation_penalty_enabled() -> bool:
    """Section 08: Signal correlation penalties."""
    return _flag("FEEDBACK_CORRELATION_PENALTY")


def conviction_multiplicative_enabled() -> bool:
    """Section 10: Multiplicative conviction calibration."""
    return _flag("FEEDBACK_CONVICTION_MULTIPLICATIVE")


def sharpe_demotion_enabled() -> bool:
    """Section 12: Live vs. backtest Sharpe demotion."""
    return _flag("FEEDBACK_SHARPE_DEMOTION")


def drift_detection_enabled() -> bool:
    """Section 13: Concept drift detection and auto-retrain."""
    return _flag("FEEDBACK_DRIFT_DETECTION")


def transition_sizing_enabled() -> bool:
    """Section 15: Regime transition probability sizing."""
    return _flag("FEEDBACK_TRANSITION_SIZING")
