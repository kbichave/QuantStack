# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
DecisionRouter — determines execution path for each signal.

Three paths:
    RULE_BASED      Default. Signal + strategy spec → execute directly. No LLM.
    GROQ_SYNTHESIS  Non-routine condition detected → ask Groq PM for guidance.
    SKIP            Do not trade this bar (kill switch, bad regime, conflicts, etc.)

Design: Groq is called ONLY for enumerated exception conditions. The default
is RULE_BASED. This keeps P99 latency low and Groq cost near zero on routine days.
"""

from __future__ import annotations

from enum import Enum, auto
from typing import Any

from loguru import logger


class DecisionPath(Enum):
    RULE_BASED      = auto()
    GROQ_SYNTHESIS  = auto()
    SKIP            = auto()


# Conviction threshold below which we skip rather than trade
_MIN_CONVICTION = 0.45
# Conviction threshold above which a size-up decision may need Groq review
_HIGH_CONVICTION = 0.85


class DecisionRouter:
    """
    Stateless routing logic. All decisions are deterministic for a given input.

    Groq conditions (any single condition triggers GROQ_SYNTHESIS):
    1. Regime flipped since last bar
    2. Two+ active strategies disagree on direction for the same symbol
    3. technical or regime collector failed (unreliable brief)
    4. market_conviction > 0.85 AND position already open (size-up review)
    5. volatility_regime == "extreme"
    6. Any open position unrealized_pnl_pct < -5%

    SKIP conditions (checked first — override Groq routing):
    - Kill switch active
    - Risk halted
    - regime confidence < 0.60
    - market_bias == "neutral" and conviction < MIN_CONVICTION
    - No strategy in registry matches current regime

    Everything else → RULE_BASED.
    """

    def route(
        self,
        symbol: str,
        brief: Any,            # SignalBrief
        strategies: list[dict],
        portfolio: dict,
        last_regime: dict | None = None,
        system_status: dict | None = None,
    ) -> tuple[DecisionPath, str]:
        """
        Return (path, reason) for the given signal and context.

        Args:
            symbol: Ticker being analyzed.
            brief: SignalBrief from SignalEngine.
            strategies: Active strategies that match this symbol.
            portfolio: Current portfolio snapshot dict.
            last_regime: Regime dict from the previous bar (for flip detection).
            system_status: dict from get_system_status (kill_switch, risk_halted).

        Returns:
            (DecisionPath, human-readable reason string)
        """
        # --- SKIP checks (unconditional overrides) ---
        if system_status:
            if system_status.get("kill_switch_active"):
                return DecisionPath.SKIP, "kill switch active"
            if system_status.get("risk_halted"):
                return DecisionPath.SKIP, "daily risk limit breached"

        regime = brief.regime_detail or {}
        confidence = regime.get("confidence", 0.5)
        if confidence < 0.60:
            return DecisionPath.SKIP, f"regime confidence {confidence:.0%} < 60%"

        bias = brief.market_bias
        conviction = brief.market_conviction
        if bias == "neutral" and conviction < _MIN_CONVICTION:
            return DecisionPath.SKIP, f"neutral bias with low conviction ({conviction:.0%})"

        # If no strategy matches the current regime, skip.
        trend_regime = regime.get("trend_regime", "unknown")
        matching_strategies = _strategies_for_regime(strategies, trend_regime)
        if not matching_strategies:
            return DecisionPath.SKIP, f"no active strategy matches regime '{trend_regime}'"

        # --- Groq exception checks ---
        reason = _check_groq_conditions(
            symbol=symbol,
            brief=brief,
            regime=regime,
            matching_strategies=matching_strategies,
            portfolio=portfolio,
            last_regime=last_regime,
        )
        if reason:
            logger.info(f"[DecisionRouter] {symbol}: GROQ_SYNTHESIS — {reason}")
            return DecisionPath.GROQ_SYNTHESIS, reason

        return DecisionPath.RULE_BASED, "routine signal, rule-based execution"


def _check_groq_conditions(
    symbol: str,
    brief: Any,
    regime: dict,
    matching_strategies: list[dict],
    portfolio: dict,
    last_regime: dict | None,
) -> str | None:
    """
    Check all conditions that warrant Groq PM synthesis.
    Returns the reason string if Groq should be called, None otherwise.
    """
    # 1. Regime flipped since last bar
    if last_regime:
        prev_trend = last_regime.get("trend_regime", "unknown")
        curr_trend = regime.get("trend_regime", "unknown")
        if prev_trend != curr_trend and prev_trend != "unknown" and curr_trend != "unknown":
            return f"regime flipped: {prev_trend} → {curr_trend}"

    # 2. Strategy conflict — two+ strategies disagree on direction
    if len(matching_strategies) >= 2:
        bias = brief.market_bias
        # All strategies should produce aligned signals for the same symbol/regime.
        # If brief shows strong_bullish but one strategy is mean-reversion (may be bearish-biased),
        # we flag it for PM review rather than silently choosing.
        disagreements = _count_strategy_disagreements(matching_strategies, bias)
        if disagreements >= 2:
            return f"{disagreements} strategies disagree on direction"

    # 3. Collector failures make the brief unreliable
    failures = getattr(brief, "collector_failures", [])
    if "technical" in failures or "regime" in failures:
        return f"critical collector failures: {failures}"

    # 4. High conviction while already holding a position (size-up review)
    conviction = brief.market_conviction
    open_positions = portfolio.get("positions", {})
    if symbol in open_positions and conviction > _HIGH_CONVICTION:
        return f"high conviction ({conviction:.0%}) with existing {symbol} position — review size-up"

    # 5. Extreme volatility
    vol_regime = regime.get("volatility_regime", "normal")
    if vol_regime == "extreme":
        return "extreme volatility regime"

    # 6. Open position under significant stress
    if symbol in open_positions:
        pnl_pct = _get_position_pnl_pct(open_positions[symbol])
        if pnl_pct is not None and pnl_pct < -5.0:
            return f"open {symbol} position at {pnl_pct:.1f}% loss — stop review required"

    return None


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
