# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
MicrostructureSignalAgent — converts tick-level microstructure features into orders.

Why this exists:
    The MicrostructureFeatureEngine computes OFI, VPIN, Kyle's Lambda, and spread
    metrics from live tick data.  Without this agent those signals have no consumer.
    This agent is the bridge between the tick feature pipeline and the execution layer.

Design decisions:
  - Stateless evaluation: each call to evaluate() makes a fresh decision based on
    the current feature snapshot.  No position tracking here — that lives in FillTracker.
  - VPIN gate: suppress all orders when VPIN > MICRO_VPIN_THRESHOLD (elevated adverse
    selection).  This is a hard suppression, not a soft skip, because executing against
    an informed counterparty in a high-VPIN environment is reliably unprofitable.
  - OFI thresholds: trade when normalised OFI exceeds ±MICRO_OFI_THRESHOLD, indicating
    a sustained directional imbalance in order flow.
  - Size cap: order size = OFI magnitude × equity_fraction, hard-capped at
    MICRO_MAX_SHARES to prevent size creep during high-flow periods.
  - Warmup guard: no signals during the warmup period (is_warm=False).  Indicators
    computed on fewer-than-window trades have high estimation error.

Failure modes:
  - features.vpin is None (bucket not yet filled): treated as vpin=0.0 (not suppressed).
  - features.ofi_normalised is 0.0 (no trades in window): no order generated.
  - All OFI but VPIN is suppressed: logs a warning so the operator knows signals were
    skipped, preventing silent non-trading in high-volume sessions.
"""

from __future__ import annotations

import os
import uuid

from loguru import logger
from quantstack.core.execution.unified_models import UnifiedOrder
from quantstack.core.microstructure.microstructure_features import MicrostructureFeatures

# ------- Tuning parameters (overridable via env vars) -------

# Suppression threshold: orders blocked when VPIN ≥ this value
_VPIN_THRESHOLD = float(os.getenv("MICRO_VPIN_THRESHOLD", "0.55"))

# Order trigger: generate a directional order when |ofi_normalised| ≥ this value
_OFI_THRESHOLD = float(os.getenv("MICRO_OFI_THRESHOLD", "0.30"))

# Hard cap on order size (shares); prevents outsized HF orders
_MAX_SHARES = int(os.getenv("MICRO_MAX_SHARES", "100"))


class MicrostructureSignalAgent:
    """
    Converts a MicrostructureFeatures snapshot into a UnifiedOrder or None.

    Intended to be registered as a callback on MicrostructureFeatureEngine via:
        engine.add_callback(agent.on_features)

    Then wired through MicrostructurePipeline → SmartOrderRouter.
    """

    # -------------------------------------------------------------------------
    # Public interface
    # -------------------------------------------------------------------------

    async def on_features(
        self, features: MicrostructureFeatures
    ) -> UnifiedOrder | None:
        """
        Evaluate a microstructure feature snapshot.

        Args:
            features: Tick-level feature vector for one symbol.

        Returns:
            UnifiedOrder to submit, or None to hold.
        """
        # Guard 1: warmup — indicators not valid until window is filled
        if not features.is_warm:
            return None

        # Guard 2: VPIN suppression — elevated adverse selection, do not trade
        vpin = features.vpin if features.vpin is not None else 0.0
        if vpin >= _VPIN_THRESHOLD:
            logger.debug(
                f"[MicroAgent] {features.symbol} suppressed (VPIN={vpin:.3f} ≥ {_VPIN_THRESHOLD})"
            )
            return None

        ofi = features.ofi_normalised
        if abs(ofi) < _OFI_THRESHOLD:
            return None  # Imbalance below signal threshold — hold

        side = "buy" if ofi > 0 else "sell"
        shares = self._compute_size(ofi)

        if shares <= 0:
            return None

        order = UnifiedOrder(
            symbol=features.symbol,
            side=side,
            quantity=float(shares),
            order_type="market",
            limit_price=None,
            client_order_id=f"micro_{features.symbol}_{uuid.uuid4().hex[:8]}",
        )

        logger.debug(
            f"[MicroAgent] Signal: {side.upper()} {shares} {features.symbol} "
            f"| OFI={ofi:+.3f} VPIN={vpin:.3f}"
        )
        return order

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _compute_size(self, ofi_normalised: float) -> int:
        """
        Size = floor(|ofi_normalised| × _MAX_SHARES), capped at _MAX_SHARES.

        Proportional to OFI magnitude so stronger imbalances get larger size,
        but the hard cap prevents unbounded position growth.
        """
        raw = abs(ofi_normalised) * _MAX_SHARES
        return min(int(raw), _MAX_SHARES)
