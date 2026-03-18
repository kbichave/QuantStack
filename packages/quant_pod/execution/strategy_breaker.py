# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Strategy-level circuit breakers — automatic risk reduction per strategy.

Unlike the portfolio-level RiskGate (which halts ALL trading on daily loss limit),
strategy breakers detect when a SINGLE strategy is underperforming and reduce
its capital allocation without affecting other strategies.

Breaker states:
  ACTIVE   — strategy operating normally
  SCALED   — position sizes reduced to 50% after warning threshold
  TRIPPED  — strategy halted (paper mode only) after trip threshold
  RESET    — manually reset via /review session

Persistence: breaker state saved to ~/.quant_pod/strategy_breakers.json

Usage:
    from quant_pod.execution.strategy_breaker import StrategyBreaker

    breaker = StrategyBreaker()

    # After each trade settles
    state = breaker.record_trade("mean_rev_v2", pnl=-150.0, equity=9850.0)
    # state.status == "SCALED" if warning threshold hit

    # Before sizing a new trade
    factor = breaker.get_scale_factor("mean_rev_v2")
    # factor == 0.5 (scaled), 1.0 (active), or 0.0 (tripped)

    # Manual reset after investigation
    breaker.reset("mean_rev_v2", reason="Reviewed — regime mismatch resolved")
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from threading import RLock

from loguru import logger

# =============================================================================
# CONFIG + STATE
# =============================================================================

# Breaker status constants — using strings for JSON serializability and clarity
# in logs. A full enum adds import overhead without meaningful safety here since
# the only writer is this module.
STATUS_ACTIVE = "ACTIVE"
STATUS_SCALED = "SCALED"
STATUS_TRIPPED = "TRIPPED"

_VALID_STATUSES = {STATUS_ACTIVE, STATUS_SCALED, STATUS_TRIPPED}


@dataclass
class BreakerConfig:
    """Thresholds for strategy circuit breakers."""

    # Trip thresholds — breaker fully disengages the strategy
    max_drawdown_pct: float = 5.0       # Trip at 5% strategy drawdown
    consecutive_loss_limit: int = 3     # Trip after 3 consecutive losses

    # Scale-down thresholds — reduce position size before tripping
    scale_drawdown_pct: float = 3.0     # Scale down at 3% drawdown
    scale_consecutive_losses: int = 2   # Scale down after 2 consecutive losses

    # Cooldown — minimum time in TRIPPED before manual reset is allowed
    cooldown_hours: int = 24


@dataclass
class BreakerState:
    """Current circuit breaker state for a single strategy."""

    strategy_id: str
    status: str = STATUS_ACTIVE  # "ACTIVE", "SCALED", "TRIPPED"
    consecutive_losses: int = 0
    peak_equity: float = 0.0
    current_equity: float = 0.0
    drawdown_pct: float = 0.0
    tripped_at: datetime | None = None
    scale_factor: float = 1.0  # 1.0 = full size, 0.5 = scaled, 0.0 = tripped
    reason: str = ""


# =============================================================================
# STRATEGY BREAKER
# =============================================================================


class StrategyBreaker:
    """
    Per-strategy circuit breaker.

    Tracks consecutive losses and drawdown per strategy. Automatically scales
    down position sizes or halts a strategy when thresholds are breached.

    State is persisted to JSON so breaker status survives process restarts.
    A threading Lock guards all state mutations and file I/O.
    """

    _DEFAULT_STATE_PATH = "~/.quant_pod/strategy_breakers.json"

    def __init__(
        self,
        config: BreakerConfig | None = None,
        state_path: str | None = None,
    ):
        self._config = config or BreakerConfig()
        self._state_path = Path(
            state_path or os.getenv("STRATEGY_BREAKER_PATH", self._DEFAULT_STATE_PATH)
        ).expanduser()
        self._lock = RLock()
        self._states: dict[str, BreakerState] = {}
        self._load()
        logger.info(
            f"StrategyBreaker initialized | path={self._state_path} "
            f"| strategies_tracked={len(self._states)} "
            f"| max_dd={self._config.max_drawdown_pct}% "
            f"| consec_loss_limit={self._config.consecutive_loss_limit}"
        )

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def record_trade(
        self, strategy_id: str, pnl: float, equity: float
    ) -> BreakerState:
        """
        Record a trade outcome and evaluate breaker thresholds.

        Args:
            strategy_id: Identifier of the strategy that produced the trade.
            pnl: Realized P&L of the trade (positive = win, negative = loss).
            equity: Strategy-level equity AFTER this trade settles.

        Returns:
            Updated BreakerState for the strategy.
        """
        with self._lock:
            state = self._get_or_create(strategy_id)

            # If already tripped, log and return — no state changes until manual reset
            if state.status == STATUS_TRIPPED:
                logger.warning(
                    f"[BREAKER] {strategy_id} is TRIPPED — ignoring trade record. "
                    f"Call reset() after cooldown to re-enable."
                )
                return self._copy_state(state)

            # Update consecutive losses
            if pnl < 0:
                state.consecutive_losses += 1
            else:
                state.consecutive_losses = 0

            # Update equity tracking
            state.current_equity = equity
            if equity > state.peak_equity:
                state.peak_equity = equity

            # Compute drawdown
            if state.peak_equity > 0:
                state.drawdown_pct = (
                    (state.peak_equity - state.current_equity)
                    / state.peak_equity
                    * 100.0
                )
            else:
                state.drawdown_pct = 0.0

            # Evaluate thresholds (trip checks first — they are stricter)
            prev_status = state.status
            self._evaluate_thresholds(state)

            if state.status != prev_status:
                logger.warning(
                    f"[BREAKER] {strategy_id} transitioned {prev_status} -> {state.status} "
                    f"| dd={state.drawdown_pct:.2f}% "
                    f"| consec_losses={state.consecutive_losses} "
                    f"| reason={state.reason}"
                )

            self._persist()
            return self._copy_state(state)

    def check(self, strategy_id: str) -> BreakerState:
        """
        Check current breaker state for a strategy.

        Returns a copy of the state. Does not modify anything.
        """
        with self._lock:
            state = self._states.get(strategy_id)
            if state is None:
                return BreakerState(strategy_id=strategy_id)
            return self._copy_state(state)

    def get_scale_factor(self, strategy_id: str) -> float:
        """
        Get the position size multiplier for a strategy.

        Returns:
            1.0 — ACTIVE (full size)
            0.5 — SCALED (reduced)
            0.0 — TRIPPED (halted)
        """
        state = self.check(strategy_id)
        return state.scale_factor

    def reset(self, strategy_id: str, reason: str) -> BreakerState:
        """
        Manually reset a tripped breaker after investigation.

        Resets the strategy to ACTIVE with full scale factor. Consecutive losses
        and drawdown tracking are zeroed — the strategy gets a clean slate.

        Args:
            strategy_id: Strategy to reset.
            reason: Required explanation of why the reset is safe (logged).

        Returns:
            Updated BreakerState.

        Raises:
            ValueError: If cooldown period has not elapsed.
            KeyError: If strategy_id has no tracked state.
        """
        with self._lock:
            state = self._states.get(strategy_id)
            if state is None:
                raise KeyError(
                    f"No breaker state for strategy '{strategy_id}'. "
                    f"Tracked strategies: {list(self._states.keys())}"
                )

            if state.status != STATUS_TRIPPED:
                logger.info(
                    f"[BREAKER] {strategy_id} is {state.status}, not TRIPPED — "
                    f"reset is a no-op but recording reason."
                )
                state.reason = f"Reset (was {state.status}): {reason}"
                state.status = STATUS_ACTIVE
                state.scale_factor = 1.0
                state.consecutive_losses = 0
                state.drawdown_pct = 0.0
                self._persist()
                return self._copy_state(state)

            # Enforce cooldown
            if state.tripped_at is not None:
                cooldown_end = state.tripped_at + timedelta(
                    hours=self._config.cooldown_hours
                )
                if datetime.now() < cooldown_end:
                    remaining = cooldown_end - datetime.now()
                    hours_left = remaining.total_seconds() / 3600
                    raise ValueError(
                        f"Cooldown not elapsed for {strategy_id}. "
                        f"Tripped at {state.tripped_at.isoformat()}, "
                        f"cooldown ends at {cooldown_end.isoformat()} "
                        f"({hours_left:.1f}h remaining). "
                        f"Investigate root cause before waiting out the cooldown."
                    )

            # Reset to clean state
            state.status = STATUS_ACTIVE
            state.scale_factor = 1.0
            state.consecutive_losses = 0
            state.drawdown_pct = 0.0
            state.tripped_at = None
            state.reason = f"Reset: {reason}"

            # Preserve peak_equity and current_equity — they represent real money.
            # Zeroing peak_equity here would suppress future drawdown detection
            # if the strategy is already below its historical peak.
            # Instead, set peak = current so drawdown measurement restarts from now.
            state.peak_equity = state.current_equity

            self._persist()
            logger.info(
                f"[BREAKER] {strategy_id} manually RESET | reason={reason} "
                f"| equity={state.current_equity:.2f}"
            )
            return self._copy_state(state)

    def get_all_states(self) -> dict[str, BreakerState]:
        """Return a copy of all tracked strategy states."""
        with self._lock:
            return {sid: self._copy_state(s) for sid, s in self._states.items()}

    # -------------------------------------------------------------------------
    # Threshold evaluation
    # -------------------------------------------------------------------------

    def _evaluate_thresholds(self, state: BreakerState) -> None:
        """
        Check if a strategy should be SCALED or TRIPPED.

        Trip conditions (any one triggers):
          - drawdown_pct >= max_drawdown_pct
          - consecutive_losses >= consecutive_loss_limit

        Scale conditions (only from ACTIVE):
          - drawdown_pct >= scale_drawdown_pct
          - consecutive_losses >= scale_consecutive_losses

        Trip takes priority over scale. Once TRIPPED, only manual reset recovers.
        """
        cfg = self._config

        # Trip checks — these override everything
        if state.drawdown_pct >= cfg.max_drawdown_pct:
            state.status = STATUS_TRIPPED
            state.scale_factor = 0.0
            state.tripped_at = datetime.now()
            state.reason = (
                f"Drawdown {state.drawdown_pct:.2f}% >= limit {cfg.max_drawdown_pct}%"
            )
            return

        if state.consecutive_losses >= cfg.consecutive_loss_limit:
            state.status = STATUS_TRIPPED
            state.scale_factor = 0.0
            state.tripped_at = datetime.now()
            state.reason = (
                f"Consecutive losses {state.consecutive_losses} >= "
                f"limit {cfg.consecutive_loss_limit}"
            )
            return

        # Scale checks — only escalate from ACTIVE to SCALED (never de-escalate
        # automatically; recovery from SCALED to ACTIVE requires a winning trade
        # that resets consecutive losses below the threshold AND drawdown below
        # scale threshold)
        should_scale = (
            state.drawdown_pct >= cfg.scale_drawdown_pct
            or state.consecutive_losses >= cfg.scale_consecutive_losses
        )

        if should_scale and state.status == STATUS_ACTIVE:
            state.status = STATUS_SCALED
            state.scale_factor = 0.5
            state.reason = (
                f"Scale-down triggered: dd={state.drawdown_pct:.2f}% "
                f"(limit={cfg.scale_drawdown_pct}%), "
                f"consec_losses={state.consecutive_losses} "
                f"(limit={cfg.scale_consecutive_losses})"
            )
            return

        # Recovery from SCALED back to ACTIVE: both conditions must clear
        if state.status == STATUS_SCALED and not should_scale:
            state.status = STATUS_ACTIVE
            state.scale_factor = 1.0
            state.reason = "Recovered — drawdown and losses below scale thresholds"

    # -------------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------------

    def _persist(self) -> None:
        """Save all breaker states to JSON. Called under the lock."""
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {}
            for strategy_id, state in self._states.items():
                entry = asdict(state)
                # datetime -> ISO string for JSON
                if state.tripped_at is not None:
                    entry["tripped_at"] = state.tripped_at.isoformat()
                else:
                    entry["tripped_at"] = None
                payload[strategy_id] = entry

            # Atomic write: write to temp file then rename to avoid partial reads
            tmp_path = self._state_path.with_suffix(".tmp")
            with open(tmp_path, "w") as f:
                json.dump(payload, f, indent=2)
            tmp_path.rename(self._state_path)
        except Exception as exc:
            # Persistence failure is serious but must not crash the trading process.
            # Log at error level so monitoring picks it up.
            logger.error(f"[BREAKER] Failed to persist state to {self._state_path}: {exc}")

    def _load(self) -> None:
        """Load breaker states from JSON on startup."""
        if not self._state_path.exists():
            logger.debug(f"[BREAKER] No state file at {self._state_path} — starting fresh")
            return

        try:
            with open(self._state_path) as f:
                payload = json.load(f)

            for strategy_id, entry in payload.items():
                tripped_at = None
                if entry.get("tripped_at"):
                    tripped_at = datetime.fromisoformat(entry["tripped_at"])

                status = entry.get("status", STATUS_ACTIVE)
                if status not in _VALID_STATUSES:
                    logger.warning(
                        f"[BREAKER] Invalid status '{status}' for {strategy_id} "
                        f"in state file — resetting to ACTIVE"
                    )
                    status = STATUS_ACTIVE

                self._states[strategy_id] = BreakerState(
                    strategy_id=strategy_id,
                    status=status,
                    consecutive_losses=entry.get("consecutive_losses", 0),
                    peak_equity=entry.get("peak_equity", 0.0),
                    current_equity=entry.get("current_equity", 0.0),
                    drawdown_pct=entry.get("drawdown_pct", 0.0),
                    tripped_at=tripped_at,
                    scale_factor=entry.get("scale_factor", 1.0),
                    reason=entry.get("reason", ""),
                )

            logger.info(
                f"[BREAKER] Loaded {len(self._states)} strategy states from {self._state_path}"
            )

            # Log any strategies that are tripped or scaled from a previous session
            for sid, state in self._states.items():
                if state.status == STATUS_TRIPPED:
                    logger.warning(
                        f"[BREAKER] {sid} is TRIPPED from previous session | "
                        f"reason={state.reason} | tripped_at={state.tripped_at}"
                    )
                elif state.status == STATUS_SCALED:
                    logger.warning(
                        f"[BREAKER] {sid} is SCALED from previous session | "
                        f"reason={state.reason}"
                    )

        except json.JSONDecodeError as exc:
            logger.error(
                f"[BREAKER] Corrupt state file at {self._state_path}: {exc} — "
                f"starting with empty state. The corrupt file is preserved for debugging."
            )
        except Exception as exc:
            logger.error(f"[BREAKER] Failed to load state from {self._state_path}: {exc}")

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _get_or_create(self, strategy_id: str) -> BreakerState:
        """Get existing state or create a new ACTIVE entry. Called under the lock."""
        if strategy_id not in self._states:
            self._states[strategy_id] = BreakerState(strategy_id=strategy_id)
        return self._states[strategy_id]

    @staticmethod
    def _copy_state(state: BreakerState) -> BreakerState:
        """Return a defensive copy so callers cannot mutate internal state."""
        return BreakerState(
            strategy_id=state.strategy_id,
            status=state.status,
            consecutive_losses=state.consecutive_losses,
            peak_equity=state.peak_equity,
            current_equity=state.current_equity,
            drawdown_pct=state.drawdown_pct,
            tripped_at=state.tripped_at,
            scale_factor=state.scale_factor,
            reason=state.reason,
        )
