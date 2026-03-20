# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
DecisionRouter — fully deterministic execution routing.

v1.1 upgrade: Removed GROQ_SYNTHESIS path. All decisions are now deterministic.

Two paths:
    RULE_BASED      Default + exception conditions. Signal + context → execute.
    SKIP            Do not trade (kill switch, bad regime, conflicts, etc.)

Design rationale for removing LLM from execution:
- LLMs are stochastic: same inputs can produce different decisions.
- LLMs hallucinate: they can invent plausible-sounding reasoning for bad trades.
- LLMs have no memory of what worked: they can't learn from last week's fills.
- Deterministic decisions are reproducible, auditable, and backtestable.
- The LLM (Claude) remains valuable for research (/workshop), review (/reflect),
  and strategy design — high-latency, high-judgment tasks where stochasticity
  is acceptable because a human reviews the output.

The 6 former "Groq exception" conditions are now handled with explicit rules:
1. Regime flip → SKIP (don't trade during regime transitions)
2. Strategy conflict → SKIP (when strategies disagree, hold)
3. Collector failures → SKIP (unreliable data = no trade)
4. High conviction + existing position → size-up with conservative cap
5. Extreme volatility → RULE_BASED with forced quarter-size
6. Position under stress → SKIP (let /review handle stop decisions)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

from loguru import logger


class DecisionPath(Enum):
    RULE_BASED      = auto()
    GROQ_SYNTHESIS  = auto()  # Kept for backward-compat; never routed to in v1.1+
    SKIP            = auto()


@dataclass
class RouteContext:
    """Additional context passed to the runner for exception-condition handling."""
    force_size: str | None = None  # Override position size (e.g., "quarter" for extreme vol)
    allow_size_up: bool = False    # Allow adding to existing position
    exception_type: str | None = None  # Which exception condition triggered, if any


# Conviction threshold below which we skip rather than trade
_MIN_CONVICTION = 0.45
# Conviction threshold above which a size-up may be appropriate
_HIGH_CONVICTION = 0.85


class DecisionRouter:
    """
    Fully deterministic routing logic. No LLM calls. All decisions are
    reproducible for a given input.

    SKIP conditions (checked first):
    - Kill switch active
    - Risk halted
    - Regime confidence < 0.60
    - Neutral bias with low conviction
    - No strategy matches current regime
    - Regime just flipped (new: let the transition settle)
    - Strategy conflict (new: >1 strategy disagrees on direction)
    - Critical collector failures (new: unreliable data)
    - Position under >5% loss (new: needs human review, not auto-trading)

    RULE_BASED (with context modifications):
    - Extreme volatility → force quarter-size
    - High conviction + existing position → allow size-up (capped at half)
    - Everything else → standard rule-based execution
    """

    def route(
        self,
        symbol: str,
        brief: Any,            # SignalBrief
        strategies: list[dict],
        portfolio: dict,
        last_regime: dict | None = None,
        system_status: dict | None = None,
    ) -> tuple[DecisionPath, str, RouteContext]:
        """
        Return (path, reason, context) for the given signal.

        v1.1: Now returns a RouteContext with additional execution instructions.
        """
        ctx = RouteContext()

        # --- SKIP checks (unconditional overrides) ---
        if system_status:
            if system_status.get("kill_switch_active"):
                return DecisionPath.SKIP, "kill switch active", ctx
            if system_status.get("risk_halted"):
                return DecisionPath.SKIP, "daily risk limit breached", ctx

        regime = brief.regime_detail or {}
        confidence = regime.get("confidence", 0.5)
        if confidence < 0.60:
            return DecisionPath.SKIP, f"regime confidence {confidence:.0%} < 60%", ctx

        bias = brief.market_bias
        conviction = brief.market_conviction
        if bias == "neutral" and conviction < _MIN_CONVICTION:
            return DecisionPath.SKIP, f"neutral bias with low conviction ({conviction:.0%})", ctx

        # If no strategy matches the current regime, skip.
        trend_regime = regime.get("trend_regime", "unknown")
        matching_strategies = _strategies_for_regime(strategies, trend_regime)
        if not matching_strategies:
            return DecisionPath.SKIP, f"no active strategy matches regime '{trend_regime}'", ctx

        # --- Exception conditions (formerly routed to Groq) ---

        # 1. Regime flip: don't trade during transitions — let the new regime settle
        if last_regime:
            prev_trend = last_regime.get("trend_regime", "unknown")
            curr_trend = regime.get("trend_regime", "unknown")
            if prev_trend != curr_trend and prev_trend != "unknown" and curr_trend != "unknown":
                ctx.exception_type = "regime_flip"
                return (
                    DecisionPath.SKIP,
                    f"regime flipped: {prev_trend} -> {curr_trend} — skipping until settled",
                    ctx,
                )

        # 2. Strategy conflict: when strategies disagree, the conservative move is to hold
        if len(matching_strategies) >= 2:
            disagreements = _count_strategy_disagreements(matching_strategies, bias)
            if disagreements >= 2:
                ctx.exception_type = "strategy_conflict"
                return (
                    DecisionPath.SKIP,
                    f"{disagreements} strategies disagree on direction — holding",
                    ctx,
                )

        # 3. Critical collector failures: unreliable data = no trade
        failures = getattr(brief, "collector_failures", [])
        if "technical" in failures or "regime" in failures:
            ctx.exception_type = "collector_failure"
            return (
                DecisionPath.SKIP,
                f"critical collector failures: {failures} — data unreliable",
                ctx,
            )

        # 4. Position under stress: needs human review via /review, not auto-action
        open_positions = portfolio.get("positions", {})
        if symbol in open_positions:
            pnl_pct = _get_position_pnl_pct(open_positions[symbol])
            if pnl_pct is not None and pnl_pct < -5.0:
                ctx.exception_type = "position_stress"
                return (
                    DecisionPath.SKIP,
                    f"open {symbol} position at {pnl_pct:.1f}% loss — requires human review",
                    ctx,
                )

        # 5. Extreme volatility: trade with forced quarter-size
        vol_regime = regime.get("volatility_regime", "normal")
        if vol_regime == "extreme":
            ctx.force_size = "quarter"
            ctx.exception_type = "extreme_vol"
            return (
                DecisionPath.RULE_BASED,
                "extreme volatility — rule-based with forced quarter-size",
                ctx,
            )

        # 6. High conviction + existing position: allow size-up (capped at half)
        if symbol in open_positions and conviction > _HIGH_CONVICTION:
            ctx.allow_size_up = True
            ctx.force_size = "half"  # Cap size-up at half to limit concentration
            ctx.exception_type = "size_up"
            return (
                DecisionPath.RULE_BASED,
                f"high conviction ({conviction:.0%}) + existing position — size-up capped at half",
                ctx,
            )

        return DecisionPath.RULE_BASED, "routine signal, rule-based execution", ctx


def _strategies_for_regime(strategies: list[dict], trend_regime: str) -> list[dict]:
    """Filter strategies whose regime_affinity includes the current trend regime."""
    if not trend_regime or trend_regime == "unknown":
        return []
    result = []
    for strat in strategies:
        affinity = strat.get("regime_affinity", [])
        if isinstance(affinity, str):
            import json
            try:
                affinity = json.loads(affinity)
            except (ValueError, TypeError):
                affinity = [affinity]
        # Match if any affinity entry is a substring match (e.g. "ranging" matches "ranging")
        if any(trend_regime in str(a) or str(a) in trend_regime for a in affinity):
            result.append(strat)
    return result


def _count_strategy_disagreements(strategies: list[dict], brief_bias: str) -> int:
    """
    Count strategies whose expected direction contradicts the brief bias.
    A mean-reversion strategy expects the opposite of the trend direction.
    """
    brief_is_bullish = brief_bias in ("bullish", "strong_bullish")
    brief_is_bearish = brief_bias in ("bearish", "strong_bearish")
    if not brief_is_bullish and not brief_is_bearish:
        return 0  # neutral brief — not a conflict

    disagreements = 0
    for strat in strategies:
        style = str(strat.get("description", "") + strat.get("name", "")).lower()
        is_mean_reversion = any(kw in style for kw in ("mean_rev", "rsimr", "reversion", "mr"))
        if is_mean_reversion and brief_is_bullish:
            # Mean reversion enters on oversold — needs BEARISH price action to enter
            pass  # actually aligned: brief oversold → mr buys → both bullish
        if is_mean_reversion and brief_is_bearish:
            disagreements += 1  # overbought brief but mr strategy doesn't short
    return disagreements


def _get_position_pnl_pct(position: dict) -> float | None:
    """Extract unrealized P&L percentage from a position dict."""
    try:
        return float(position.get("unrealized_pnl_pct", position.get("pnl_pct", 0)))
    except (TypeError, ValueError):
        return None
