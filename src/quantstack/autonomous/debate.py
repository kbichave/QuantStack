# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
DebateFilter — adversarial bull/bear challenge on borderline signals.

Inspired by TradingAgents' multi-agent debate pattern, but applied
selectively (not on every signal) and deterministically scored.

Trigger: conviction in [0.50, 0.75] AND position size >= half.
Below 0.50 we already skip. Above 0.75 conviction is strong enough.
On quarter-size trades the debate cost isn't worth it.

Architecture:
  1. Build a structured case summary from the SignalBrief
  2. Generate bull case (reasons the trade works)
  3. Generate bear case (reasons it fails)
  4. Score both cases against available evidence
  5. Adjust conviction up/down or veto the trade

The debate is DETERMINISTIC (rule-based scoring against evidence)
with an optional LLM pass for insight generation (non-blocking, async).
The deterministic path ensures no LLM failure can block execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from loguru import logger


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEBATE_CONVICTION_MIN = 0.50  # Below this we already SKIP
DEBATE_CONVICTION_MAX = 0.75  # Above this conviction is strong enough
DEBATE_SIZE_THRESHOLD = "half"  # Only debate on half-size or larger


@dataclass
class DebateVerdict:
    """Result of the adversarial challenge."""

    challenged: bool = False  # Was the signal debated?
    original_conviction: float = 0.0
    adjusted_conviction: float = 0.0
    bull_score: float = 0.0  # Evidence strength for the trade
    bear_score: float = 0.0  # Evidence strength against the trade
    bull_points: list[str] = field(default_factory=list)
    bear_points: list[str] = field(default_factory=list)
    verdict: str = "pass"  # "pass" | "downgrade" | "veto"
    reason: str = ""


class DebateFilter:
    """Adversarial challenge filter for borderline trading signals.

    Sits between DecisionRouter.route() and trade execution.
    Only fires on borderline conviction (0.50-0.75) + material size.
    Uses structured evidence scoring, not LLM calls.
    """

    def challenge(
        self,
        symbol: str,
        brief: Any,
        conviction: float,
        bias: str,
        position_size: str,
        regime: dict[str, Any],
        portfolio: dict[str, Any],
        strategies: list[dict],
        past_lessons: list[Any] | None = None,
    ) -> DebateVerdict:
        """Challenge a borderline signal with adversarial evidence.

        Args:
            past_lessons: Optional list of ReflexionEpisode objects from
                ReflexionMemory. Each contributes +0.1 to the bear case
                (capped at +0.3 total). Implements Reflexion-style
                episodic conditioning (Shinn et al., NeurIPS 2023).

        Returns a DebateVerdict with adjusted conviction and verdict.
        """
        verdict = DebateVerdict(
            original_conviction=conviction,
            adjusted_conviction=conviction,
        )

        # Gate: only debate borderline signals on material positions
        if conviction < DEBATE_CONVICTION_MIN or conviction > DEBATE_CONVICTION_MAX:
            verdict.reason = f"conviction {conviction:.0%} outside debate range"
            return verdict

        size_rank = {"quarter": 1, "half": 2, "full": 3}
        if size_rank.get(position_size, 0) < size_rank.get(DEBATE_SIZE_THRESHOLD, 2):
            verdict.reason = f"position size '{position_size}' below debate threshold"
            return verdict

        verdict.challenged = True
        logger.info(
            f"[Debate] Challenging {symbol} {bias} signal (conviction={conviction:.0%})"
        )

        # --- Build bull case ---
        bull_points, bull_score = self._build_bull_case(brief, bias, regime, portfolio)
        verdict.bull_points = bull_points
        verdict.bull_score = bull_score

        # --- Build bear case ---
        bear_points, bear_score = self._build_bear_case(brief, bias, regime, portfolio)

        # --- Inject reflexion episodes as bear evidence ---
        if past_lessons:
            lesson_penalty = 0.0
            for lesson in past_lessons[:3]:  # Max 3 lessons
                reinforcement = getattr(lesson, "verbal_reinforcement", "")
                root_cause = getattr(lesson, "root_cause", "unknown")
                pnl = getattr(lesson, "pnl_pct", 0.0)
                if reinforcement:
                    bear_points.append(
                        f"Past lesson [{root_cause}] ({pnl:+.1f}%): {reinforcement[:100]}"
                    )
                    lesson_penalty += 0.10
            bear_score += min(lesson_penalty, 0.30)  # Cap at +0.30

        verdict.bear_points = bear_points
        verdict.bear_score = bear_score

        # --- Score and adjust ---
        net_evidence = bull_score - bear_score

        if net_evidence > 0.15:
            # Bull case dominates — upgrade conviction
            bump = min(net_evidence * 0.5, 0.10)
            verdict.adjusted_conviction = min(conviction + bump, 0.95)
            verdict.verdict = "pass"
            verdict.reason = f"bull case wins ({bull_score:.2f} vs {bear_score:.2f}), conviction +{bump:.0%}"
        elif net_evidence < -0.15:
            # Bear case dominates
            if net_evidence < -0.30:
                # Strong bear case — veto the trade
                verdict.adjusted_conviction = 0.0
                verdict.verdict = "veto"
                verdict.reason = f"bear case dominates ({bear_score:.2f} vs {bull_score:.2f}), trade vetoed"
            else:
                # Moderate bear case — downgrade
                penalty = min(abs(net_evidence) * 0.5, 0.15)
                verdict.adjusted_conviction = max(conviction - penalty, 0.05)
                verdict.verdict = "downgrade"
                verdict.reason = f"bear case stronger ({bear_score:.2f} vs {bull_score:.2f}), conviction -{penalty:.0%}"
        else:
            # Close call — no change
            verdict.verdict = "pass"
            verdict.reason = (
                f"debate inconclusive ({bull_score:.2f} vs {bear_score:.2f})"
            )

        logger.info(
            f"[Debate] {symbol}: {verdict.verdict} | "
            f"bull={bull_score:.2f} bear={bear_score:.2f} | "
            f"conviction {conviction:.0%} → {verdict.adjusted_conviction:.0%}"
        )
        return verdict

    # ------------------------------------------------------------------
    # Bull case: reasons the trade works
    # ------------------------------------------------------------------

    def _build_bull_case(
        self,
        brief: Any,
        bias: str,
        regime: dict,
        portfolio: dict,
    ) -> tuple[list[str], float]:
        """Score evidence supporting the trade direction."""
        points: list[str] = []
        score = 0.0
        is_bullish = bias in ("bullish", "strong_bullish")

        # 1. Trend alignment
        trend = regime.get("trend_regime", "unknown")
        if (is_bullish and trend == "trending_up") or (
            not is_bullish and trend == "trending_down"
        ):
            points.append(f"Trend alignment: {bias} signal in {trend} regime")
            score += 0.20

        # 2. HMM stability
        stability = regime.get("hmm_stability", 0.0)
        if stability > 0.8:
            points.append(f"Regime stable (HMM={stability:.0%})")
            score += 0.10

        # 3. RSI confirmation
        rsi = getattr(brief, "rsi_14", None)
        if rsi is not None:
            if is_bullish and rsi < 35:
                points.append(f"RSI oversold ({rsi:.0f}) — bounce potential")
                score += 0.15
            elif not is_bullish and rsi > 65:
                points.append(f"RSI overbought ({rsi:.0f}) — pullback potential")
                score += 0.15

        # 4. Options flow alignment
        gex = getattr(brief, "opt_gex", None)
        if gex is not None:
            if is_bullish and gex > 0:
                points.append(
                    f"Positive GEX ({gex:,.0f}) — dealer dampening supports longs"
                )
                score += 0.10
            elif not is_bullish and gex < 0:
                points.append(
                    f"Negative GEX ({gex:,.0f}) — dealer amplifying supports shorts"
                )
                score += 0.10

        # 5. ML agreement
        ml_bias = getattr(brief, "ml_bias", None)
        if ml_bias and ml_bias == bias:
            ml_conf = getattr(brief, "ml_confidence", 0)
            points.append(f"ML agrees ({ml_bias}, {ml_conf:.0%})")
            score += 0.15

        # 6. Volume confirmation
        observations = getattr(brief, "observations", []) or []
        for obs in observations:
            if (
                isinstance(obs, str)
                and "volume" in obs.lower()
                and "above" in obs.lower()
            ):
                points.append("Volume above average — confirming move")
                score += 0.10
                break

        # 7. No open position conflict
        positions = portfolio.get("positions", {})
        symbol = getattr(brief, "symbol", "")
        if symbol not in positions:
            points.append("No existing position — clean entry")
            score += 0.05

        return points, min(score, 1.0)

    # ------------------------------------------------------------------
    # Bear case: reasons the trade fails
    # ------------------------------------------------------------------

    def _build_bear_case(
        self,
        brief: Any,
        bias: str,
        regime: dict,
        portfolio: dict,
    ) -> tuple[list[str], float]:
        """Score evidence against the trade direction."""
        points: list[str] = []
        score = 0.0
        is_bullish = bias in ("bullish", "strong_bullish")

        # 1. Counter-trend trade
        trend = regime.get("trend_regime", "unknown")
        if (is_bullish and trend == "trending_down") or (
            not is_bullish and trend == "trending_up"
        ):
            points.append(f"Counter-trend: {bias} signal in {trend} regime")
            score += 0.25

        # 2. Regime instability
        stability = regime.get("hmm_stability", 0.0)
        if stability < 0.6:
            points.append(
                f"Regime unstable (HMM={stability:.0%}) — signal may be noise"
            )
            score += 0.15

        # 3. Options flow opposition
        gex = getattr(brief, "opt_gex", None)
        if gex is not None:
            if is_bullish and gex < -1e9:
                points.append(
                    f"Strongly negative GEX ({gex:,.0f}) — amplified downside risk"
                )
                score += 0.15
            elif not is_bullish and gex > 1e9:
                points.append(
                    f"Strongly positive GEX ({gex:,.0f}) — dealers support upside"
                )
                score += 0.15

        # 4. Near max pain (pin risk for options expiry)
        max_pain = getattr(brief, "opt_max_pain", None)
        close = getattr(brief, "close", None)
        if max_pain and close and abs(close - max_pain) / close < 0.01:
            points.append(f"Price near max pain ({max_pain:.2f}) — pin risk")
            score += 0.10

        # 5. Upcoming event risk
        risk_factors = getattr(brief, "risk_factors", []) or []
        for rf in risk_factors:
            if isinstance(rf, str):
                if "earnings" in rf.lower():
                    points.append("Earnings event imminent — binary risk")
                    score += 0.20
                    break
                if "fomc" in rf.lower() or "cpi" in rf.lower():
                    points.append(f"Macro event risk: {rf}")
                    score += 0.15
                    break

        # 6. ML disagreement
        ml_bias = getattr(brief, "ml_bias", None)
        if ml_bias and ml_bias != bias and ml_bias != "neutral":
            points.append(f"ML disagrees ({ml_bias} vs {bias})")
            score += 0.20

        # 7. Collector failures (partial data)
        failures = getattr(brief, "collector_failures", []) or []
        if failures:
            points.append(f"Data gaps: {', '.join(failures)}")
            score += 0.10 * min(len(failures), 3)

        # 8. Concentration risk
        positions = portfolio.get("positions", {})
        symbol = getattr(brief, "symbol", "")
        if symbol in positions:
            points.append("Already have open position — adding concentration risk")
            score += 0.10

        # 9. Drawdown context
        cash = portfolio.get("cash", 100000)
        equity = portfolio.get("total_equity", 100000)
        if equity < cash * 0.95:
            points.append(
                f"Portfolio in drawdown ({(equity/cash - 1)*100:.1f}%) — defensive stance warranted"
            )
            score += 0.15

        return points, min(score, 1.0)
