# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
RuleBasedSynthesizer — deterministic signal synthesis with regime-conditional weights.

v1.1 upgrade: Two fundamental changes from the original fixed-weight synthesizer:

1. **Regime-conditional weights**: Synthesis weights adapt to the detected regime.
   In trending markets, momentum indicators (trend, MACD) get higher weight.
   In ranging markets, mean-reversion indicators (RSI, BB) dominate.
   Rationale: RSI oversold in a trending-down market is a falling knife, not a
   buy signal. MACD momentum in a ranging market is noise, not signal.

2. **ML signal integration**: When an ML model has been trained and returns a
   prediction, it gets a vote in the synthesis (15% weight, stolen from the
   weakest regime-specific indicators). ML adds non-obvious cross-feature
   signals that rule-based indicators miss.

Design intent (unchanged):
- No LLM calls.
- Rules are numerically grounded.
- When in doubt, bias toward neutral / lower conviction.
- The synthesizer does not make trade decisions; it produces a structured brief.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Literal

from loguru import logger

from quantstack.shared.schemas import KeyLevel, SymbolBrief

# Type aliases for readability
_Bias = Literal["strong_bullish", "bullish", "neutral", "bearish", "strong_bearish"]
_Agreement = Literal["unanimous", "strong", "moderate", "mixed", "conflicting"]
_Quality = Literal["high", "medium", "low"]
_MktBias = Literal["bullish", "bearish", "neutral"]
_RiskEnv = Literal["low", "normal", "elevated", "high"]


# ------------------------------------------------------------------ #
# Regime-conditional weight profiles                                   #
# ------------------------------------------------------------------ #
# Each profile defines weights for: trend, rsi, macd, bb, sentiment, ml
# Weights must sum to 1.0.  The "ml" weight is only active when the
# ML collector returns a valid prediction; otherwise it's redistributed
# proportionally to the other voters.

_WEIGHT_PROFILES: dict[str, dict[str, float]] = {
    # Trending up: trust momentum (trend + MACD), downweight mean-reversion
    "trending_up": {
        "trend": 0.35,
        "rsi": 0.10,
        "macd": 0.20,
        "bb": 0.05,
        "sentiment": 0.10,
        "ml": 0.15,
        "flow": 0.05,
    },
    # Trending down: trust momentum, but RSI gets more weight (oversold bounces)
    "trending_down": {
        "trend": 0.30,
        "rsi": 0.15,
        "macd": 0.20,
        "bb": 0.05,
        "sentiment": 0.10,
        "ml": 0.15,
        "flow": 0.05,
    },
    # Ranging: mean-reversion dominates — RSI and BB are the primary signals
    "ranging": {
        "trend": 0.05,
        "rsi": 0.25,
        "macd": 0.10,
        "bb": 0.25,
        "sentiment": 0.10,
        "ml": 0.15,
        "flow": 0.10,
    },
    # Unknown regime: conservative — spread weight evenly, lean on ML
    "unknown": {
        "trend": 0.15,
        "rsi": 0.15,
        "macd": 0.15,
        "bb": 0.15,
        "sentiment": 0.10,
        "ml": 0.20,
        "flow": 0.10,
    },
}

# Default (legacy) weights — used as absolute fallback
_DEFAULT_WEIGHTS = {
    "trend": 0.30,
    "rsi": 0.18,
    "macd": 0.15,
    "bb": 0.12,
    "sentiment": 0.10,
    "ml": 0.15,
    "flow": 0.00,
}


def _get_weights(
    trend_regime: str,
    has_ml: bool,
    has_flow: bool,
) -> dict[str, float]:
    """
    Get synthesis weights for the given regime.

    If ML signal is unavailable, redistribute its weight proportionally
    to the remaining voters. Same for flow.
    """
    profile = _WEIGHT_PROFILES.get(trend_regime, _DEFAULT_WEIGHTS).copy()

    # Redistribute inactive voter weights
    inactive_weight = 0.0
    if not has_ml:
        inactive_weight += profile.pop("ml", 0.0)
    if not has_flow:
        inactive_weight += profile.pop("flow", 0.0)

    if inactive_weight > 0 and profile:
        active_total = sum(profile.values())
        if active_total > 0:
            scale = (active_total + inactive_weight) / active_total
            for key in profile:
                profile[key] = round(profile[key] * scale, 4)

    return profile


class RuleBasedSynthesizer:
    """
    Builds a SymbolBrief from collector output dicts.

    v1.1: Regime-conditional weights + ML signal integration.

    Weights adapt to regime:
        trending_up:  trend 35%, MACD 20%, ML 15%, RSI 10%, sentiment 10%, flow 5%, BB 5%
        trending_down: trend 30%, MACD 20%, RSI 15%, ML 15%, sentiment 10%, flow 5%, BB 5%
        ranging:      RSI 25%, BB 25%, ML 15%, MACD 10%, sentiment 10%, flow 10%, trend 5%
        unknown:      ML 20%, even split on others

    Conviction scaling:
        Base: abs(weighted vote score)
        +0.10 if ADX > 25 (strong trend confirms momentum)
        +0.05 if HMM stability > 0.8 (regime is well-established)
        -0.15 if weekly trend contradicts daily regime
        -0.10 if HMM and rule-based disagree on regime
        -0.20 if collector_failures contains "technical" or "regime"
        Clamped to [0.05, 0.95]
    """

    # Sentiment thresholds (score in [0, 1]; 0.5 = neutral)
    SENTIMENT_BULLISH_THRESHOLD = 0.65
    SENTIMENT_BEARISH_THRESHOLD = 0.35

    # RSI thresholds
    RSI_OVERSOLD = 35
    RSI_OVERBOUGHT = 65

    # MACD threshold
    MACD_THRESHOLD = 0.0

    # Bollinger threshold
    BB_LOWER_ZONE = 0.20
    BB_UPPER_ZONE = 0.80

    # ADX threshold
    ADX_STRONG_TREND = 25

    # Bias score to label mapping
    SCORE_STRONG = 0.60
    SCORE_WEAK = 0.25

    # ML signal thresholds
    ML_BULLISH_THRESHOLD = 0.55
    ML_BEARISH_THRESHOLD = 0.45

    def synthesize(
        self,
        symbol: str,
        technical: dict[str, Any],
        regime: dict[str, Any],
        volume: dict[str, Any],
        risk: dict[str, Any],
        events: dict[str, Any],
        fundamentals: dict[str, Any],
        collector_failures: list[str],
        strategy_context: str = "",
        sentiment: dict[str, Any] | None = None,
        ml_signal: dict[str, Any] | None = None,
        flow: dict[str, Any] | None = None,
    ) -> SymbolBrief:
        """Build a SymbolBrief from collector outputs."""

        bias, conviction = self._compute_bias_and_conviction(
            technical,
            regime,
            collector_failures,
            sentiment=sentiment or {},
            ml_signal=ml_signal or {},
            flow=flow or {},
        )
        pod_agreement = self._compute_pod_agreement(
            technical,
            regime,
            volume,
            ml_signal=ml_signal or {},
        )
        critical_levels = self._extract_critical_levels(technical, volume, risk)
        observations = self._build_observations(
            technical,
            regime,
            volume,
            events,
            ml_signal=ml_signal or {},
        )
        risk_factors = self._build_risk_factors(
            risk, events, regime, collector_failures
        )
        insights = self._build_insights(bias, conviction, technical, regime, volume)
        quality = self._assess_quality(regime, technical, collector_failures)
        summary = self._build_summary(symbol, bias, conviction, regime, events)

        return SymbolBrief(
            symbol=symbol,
            market_summary=summary,
            consensus_bias=bias,
            consensus_conviction=round(conviction, 3),
            pod_agreement=pod_agreement,
            critical_levels=critical_levels,
            key_observations=observations,
            risk_factors=risk_factors,
            actionable_insights=insights,
            contributing_pods=_active_pods(collector_failures),
            analysis_quality=quality,
        )

    # ------------------------------------------------------------------ #
    # Bias and conviction                                                  #
    # ------------------------------------------------------------------ #

    def _compute_bias_and_conviction(
        self,
        technical: dict,
        regime: dict,
        failures: list[str],
        sentiment: dict | None = None,
        ml_signal: dict | None = None,
        flow: dict | None = None,
    ) -> tuple[_Bias, float]:

        scores: dict[str, float] = {}

        # 1. Trend direction signal from regime
        trend = regime.get("trend_regime", "unknown")
        scores["trend"] = {
            "trending_up": 1.0,
            "trending_down": -1.0,
            "ranging": 0.0,
            "unknown": 0.0,
        }.get(trend, 0.0)

        # 2. RSI zone — mean-reversion signal
        rsi = technical.get("rsi_14")
        if rsi is not None:
            if rsi < self.RSI_OVERSOLD:
                scores["rsi"] = 1.0
            elif rsi > self.RSI_OVERBOUGHT:
                scores["rsi"] = -1.0
            else:
                scores["rsi"] = (50 - rsi) / 15 * 0.5
        else:
            scores["rsi"] = 0.0

        # 3. MACD histogram — momentum signal
        macd_hist = technical.get("macd_hist")
        if macd_hist is not None:
            scores["macd"] = 1.0 if macd_hist > self.MACD_THRESHOLD else -1.0
        else:
            scores["macd"] = 0.0

        # 4. Bollinger Band position — volatility-relative signal
        bb_pct = technical.get("bb_pct")
        if bb_pct is not None:
            if bb_pct < self.BB_LOWER_ZONE:
                scores["bb"] = 1.0
            elif bb_pct > self.BB_UPPER_ZONE:
                scores["bb"] = -1.0
            else:
                scores["bb"] = 0.0
        else:
            scores["bb"] = 0.0

        # 5. Sentiment signal — only with actual headlines
        sent = sentiment or {}
        n_headlines = sent.get("n_headlines", 0)
        if n_headlines > 0:
            sent_score = sent.get("sentiment_score", 0.5)
            if sent_score > self.SENTIMENT_BULLISH_THRESHOLD:
                scores["sentiment"] = 1.0
            elif sent_score < self.SENTIMENT_BEARISH_THRESHOLD:
                scores["sentiment"] = -1.0
            else:
                scores["sentiment"] = 0.0
        else:
            scores["sentiment"] = 0.0

        # 6. ML signal — calibrated probability from trained model
        ml = ml_signal or {}
        ml_pred = ml.get("ml_prediction")
        has_ml = ml_pred is not None and ml.get("ml_confidence", 0) > 0
        if has_ml:
            if ml_pred > self.ML_BULLISH_THRESHOLD:
                # Scale ML vote by confidence: strong prediction = full vote
                ml_conf = ml.get("ml_confidence", 0.5)
                scores["ml"] = min(1.0, ml_conf * 2)  # 0.5 conf → 1.0 vote
            elif ml_pred < self.ML_BEARISH_THRESHOLD:
                ml_conf = ml.get("ml_confidence", 0.5)
                scores["ml"] = -min(1.0, ml_conf * 2)
            else:
                scores["ml"] = 0.0
        else:
            scores["ml"] = 0.0

        # 7. Flow signal (institutional/insider flow direction)
        fl = flow or {}
        flow_signal = fl.get("flow_signal")
        has_flow = flow_signal is not None
        if has_flow:
            scores["flow"] = {
                "bullish": 1.0,
                "bearish": -1.0,
                "neutral": 0.0,
                "unknown": 0.0,
            }.get(str(flow_signal), 0.0)
        else:
            scores["flow"] = 0.0

        # --- Regime-conditional weighted sum ---
        weights = _get_weights(trend, has_ml=has_ml, has_flow=has_flow)

        score = sum(scores.get(key, 0.0) * w for key, w in weights.items())

        # Map score to bias label
        if score >= self.SCORE_STRONG:
            bias: _Bias = "strong_bullish"
        elif score >= self.SCORE_WEAK:
            bias = "bullish"
        elif score <= -self.SCORE_STRONG:
            bias = "strong_bearish"
        elif score <= -self.SCORE_WEAK:
            bias = "bearish"
        else:
            bias = "neutral"

        # --- Conviction scaling ---
        conviction = abs(score)

        # ADX > 25: trend is real, trust the direction more
        adx = technical.get("adx_14")
        if adx is not None and adx > self.ADX_STRONG_TREND:
            conviction += 0.10

        # HMM stability bonus: well-established regimes deserve more trust
        hmm_stability = regime.get("hmm_stability")
        if hmm_stability is not None and hmm_stability > 0.8:
            conviction += 0.05

        # Weekly trend contradicts daily → reduce conviction
        weekly_trend = technical.get("weekly_trend", "unknown")
        if weekly_trend != "unknown" and trend != "unknown":
            if (weekly_trend == "bullish" and trend == "trending_down") or (
                weekly_trend == "bearish" and trend == "trending_up"
            ):
                conviction -= 0.15

        # HMM/rule-based regime disagreement → reduce conviction
        if regime.get("regime_disagreement"):
            conviction -= 0.10

        # ML model agreement bonus: if ML agrees with rule-based direction, boost
        if has_ml and scores.get("ml", 0) != 0:
            ml_direction = 1 if scores["ml"] > 0 else -1
            rule_direction = 1 if score > 0 else (-1 if score < 0 else 0)
            if ml_direction == rule_direction and rule_direction != 0:
                conviction += 0.05  # ML confirms rule-based → small boost

        # Collector failures reduce reliability
        if "technical" in failures:
            conviction -= 0.20
        if "regime" in failures:
            conviction -= 0.20

        conviction = round(max(0.05, min(0.95, conviction)), 3)
        return bias, conviction

    # ------------------------------------------------------------------ #
    # Pod agreement                                                        #
    # ------------------------------------------------------------------ #

    def _compute_pod_agreement(
        self,
        technical: dict,
        regime: dict,
        volume: dict,
        ml_signal: dict | None = None,
    ) -> _Agreement:
        """
        Count how many signal sources agree on direction.
        Now includes ML as a voter when available.
        """
        signals: list[int] = []  # +1 bullish, -1 bearish, 0 neutral

        # Technical signal: RSI + MACD both same direction
        rsi = technical.get("rsi_14")
        macd_h = technical.get("macd_hist")
        if rsi is not None and macd_h is not None:
            if rsi < 40 and macd_h > 0:
                signals.append(1)
            elif rsi > 60 and macd_h < 0:
                signals.append(-1)
            else:
                signals.append(0)

        # Regime signal
        trend = regime.get("trend_regime", "unknown")
        if trend == "trending_up":
            signals.append(1)
        elif trend == "trending_down":
            signals.append(-1)
        else:
            signals.append(0)

        # Volume signal
        at_hvn = volume.get("at_hvn", False)
        vol_trend = volume.get("volume_trend", "flat")
        vol_confirms = volume.get("vol_confirms_move", False)
        if at_hvn and vol_trend == "increasing":
            signals.append(1)
        elif vol_confirms and trend == "trending_down":
            signals.append(-1)
        else:
            signals.append(0)

        # Weekly trend signal
        weekly = technical.get("weekly_trend", "unknown")
        if weekly == "bullish":
            signals.append(1)
        elif weekly == "bearish":
            signals.append(-1)
        else:
            signals.append(0)

        # ML signal (new voter)
        ml = ml_signal or {}
        ml_dir = ml.get("ml_direction")
        if ml_dir == "bullish":
            signals.append(1)
        elif ml_dir == "bearish":
            signals.append(-1)
        elif ml_dir is not None:
            signals.append(0)

        if not signals:
            return "mixed"

        bullish = signals.count(1)
        bearish = signals.count(-1)
        neutral = signals.count(0)
        n = len(signals)

        if bullish == n or bearish == n:
            return "unanimous"
        if bullish >= n - 1 or bearish >= n - 1:
            return "strong"
        if neutral >= n - 1:
            return "mixed"
        if bullish >= 2 and bearish == 0:
            return "moderate"
        if bearish >= 2 and bullish == 0:
            return "moderate"
        if bullish > 0 and bearish > 0:
            return "conflicting"
        return "mixed"

    # ------------------------------------------------------------------ #
    # Critical levels                                                      #
    # ------------------------------------------------------------------ #

    def _extract_critical_levels(
        self, technical: dict, volume: dict, risk: dict
    ) -> list[KeyLevel]:
        levels: list[KeyLevel] = []

        close = technical.get("close")
        if close is None:
            return levels

        # Bollinger Bands as dynamic support/resistance
        bb_upper = technical.get("bb_upper")
        bb_lower = technical.get("bb_lower")
        if bb_upper:
            levels.append(
                KeyLevel(
                    price=round(bb_upper, 4),
                    level_type="resistance",
                    strength=0.6,
                    source="BB_upper",
                )
            )
        if bb_lower:
            levels.append(
                KeyLevel(
                    price=round(bb_lower, 4),
                    level_type="support",
                    strength=0.6,
                    source="BB_lower",
                )
            )

        # SMA 20 / 50 / 200 as support/resistance depending on trend
        for period, sma_key in ((20, "sma_20"), (50, "sma_50"), (200, "sma_200")):
            sma = technical.get(sma_key)
            if sma is None:
                continue
            level_type: Literal["support", "resistance"] = (
                "support" if close > sma else "resistance"
            )
            strength = {20: 0.5, 50: 0.65, 200: 0.80}[period]
            levels.append(
                KeyLevel(
                    price=round(sma, 4),
                    level_type=level_type,
                    strength=strength,
                    source=f"SMA_{period}",
                )
            )

        # ATR-based stop levels
        atr_stop_long = risk.get("atr_stop_long")
        if atr_stop_long:
            levels.append(
                KeyLevel(
                    price=round(atr_stop_long, 4),
                    level_type="stop",
                    strength=0.7,
                    source="ATR_2x_stop",
                )
            )

        # High-volume nodes from volume profile
        for hvn in volume.get("hvn_levels", [])[:2]:
            level_type = "support" if close > hvn else "resistance"
            levels.append(
                KeyLevel(
                    price=round(hvn, 4),
                    level_type=level_type,
                    strength=0.75,
                    source="HVN",
                )
            )

        # Sort by proximity to current price, keep top 6
        levels.sort(key=lambda l: abs(l.price - close))
        return levels[:6]

    # ------------------------------------------------------------------ #
    # Observations, risk factors, insights                                 #
    # ------------------------------------------------------------------ #

    def _build_observations(
        self,
        technical: dict,
        regime: dict,
        volume: dict,
        events: dict,
        ml_signal: dict | None = None,
    ) -> list[str]:
        obs: list[str] = []

        rsi = technical.get("rsi_14")
        if rsi is not None:
            zone = "oversold" if rsi < 35 else "overbought" if rsi > 65 else "neutral"
            obs.append(f"RSI(14) at {rsi:.1f} — {zone} territory")

        adx = technical.get("adx_14")
        if adx is not None:
            strength = "strong" if adx > 25 else "weak"
            obs.append(f"ADX(14) at {adx:.1f} — {strength} trend")

        macd_h = technical.get("macd_hist")
        if macd_h is not None:
            direction = "bullish" if macd_h > 0 else "bearish"
            obs.append(f"MACD histogram {macd_h:+.3f} — {direction} momentum")

        trend = regime.get("trend_regime", "unknown")
        confidence = regime.get("confidence", 0)
        source = regime.get("regime_source", "rule_based")
        obs.append(f"Regime: {trend} (confidence {confidence:.0%}, source: {source})")

        # HMM-specific observations
        hmm_stability = regime.get("hmm_stability")
        if hmm_stability is not None:
            duration = regime.get("hmm_expected_duration", 0)
            obs.append(
                f"HMM stability: {hmm_stability:.0%}, expected duration: {duration:.0f} bars"
            )

        weekly = technical.get("weekly_trend", "unknown")
        if weekly != "unknown":
            obs.append(f"Weekly trend: {weekly}")

        if volume.get("at_hvn"):
            obs.append("Price at High Volume Node — strong support/resistance zone")
        if volume.get("at_lvn"):
            obs.append("Price at Low Volume Node — expect fast moves through this area")
        if volume.get("vol_confirms_move"):
            obs.append("Last bar volume > 1.5x ADV — move is volume-confirmed")

        # ML signal observation
        ml = ml_signal or {}
        ml_pred = ml.get("ml_prediction")
        if ml_pred is not None:
            ml_dir = ml.get("ml_direction", "neutral")
            ml_conf = ml.get("ml_confidence", 0)
            top_feats = ml.get("ml_top_features", [])
            feat_str = f" (top: {', '.join(top_feats[:2])})" if top_feats else ""
            obs.append(
                f"ML signal: {ml_dir} (prob={ml_pred:.2f}, conf={ml_conf:.0%}){feat_str}"
            )

        if events.get("has_earnings_24h"):
            obs.append(f"Earnings within 24h — {events.get('next_event_desc', '')}")

        return obs[:10]  # increased cap for richer output

    def _build_risk_factors(
        self,
        risk: dict,
        events: dict,
        regime: dict,
        failures: list[str],
    ) -> list[str]:
        factors: list[str] = []

        var_95 = risk.get("var_95")
        if var_95 and var_95 > 2.5:
            factors.append(f"1-day VaR(95%) = {var_95:.2f}% — elevated single-day risk")

        liq = risk.get("liquidity_score")
        if liq is not None and liq < 0.3:
            factors.append("Low liquidity — wide spreads, limited position size")

        if events.get("has_earnings_24h"):
            factors.append(
                "Earnings event in next 24h — binary risk, undefined magnitude"
            )
        elif events.get("has_earnings_7d"):
            factors.append(
                "Earnings within 7 days — hold through event or exit beforehand"
            )

        if events.get("has_fomc_24h"):
            factors.append("FOMC meeting within 24h — rate decision risk")

        if events.get("has_macro_event"):
            factors.append(
                "Macro event (CPI/NFP/GDP) in next 24h — position sizing caution"
            )

        vol_regime = regime.get("volatility_regime", "normal")
        if vol_regime in ("high", "extreme"):
            factors.append(f"Volatility regime: {vol_regime} — reduce position sizes")

        if regime.get("trend_regime") == "unknown":
            factors.append("Regime confidence too low — paper mode only")

        # HMM-specific: regime transition warning
        hmm_stability = regime.get("hmm_stability")
        if hmm_stability is not None and hmm_stability < 0.5:
            factors.append(
                f"HMM regime unstable ({hmm_stability:.0%}) — possible regime transition"
            )

        if regime.get("regime_disagreement"):
            factors.append(
                "HMM and rule-based regime detectors disagree — ambiguous regime"
            )

        if failures:
            factors.append(
                f"Partial analysis — {', '.join(failures)} collector(s) unavailable"
            )

        dd = risk.get("max_drawdown_90d")
        if dd is not None and dd < -10:
            factors.append(
                f"Max drawdown last 90 days: {dd:.1f}% — symbol under stress"
            )

        return factors[:8]  # increased cap

    def _build_insights(
        self,
        bias: _Bias,
        conviction: float,
        technical: dict,
        regime: dict,
        volume: dict,
    ) -> list[str]:
        insights: list[str] = []

        rsi = technical.get("rsi_14")
        close = technical.get("close")
        sma_200 = technical.get("sma_200")
        trend = regime.get("trend_regime", "unknown")

        # Classic RSI mean-reversion setups
        if rsi is not None and rsi < 35 and sma_200 and close and close > sma_200:
            insights.append(
                f"RSI oversold ({rsi:.1f}) + price above SMA200 — "
                "quality mean-reversion setup (matches quality_rsimr strategy)"
            )
        elif rsi is not None and rsi < 35 and trend == "ranging":
            insights.append(
                f"RSI oversold ({rsi:.1f}) in ranging regime — "
                "mean-reversion entry candidate"
            )

        if volume.get("at_hvn") and bias in ("bullish", "strong_bullish"):
            insights.append("Entry at HVN — strong support, better risk/reward")
        elif volume.get("at_lvn"):
            insights.append("Price at LVN — expect fast move; tight stop required")

        if conviction < 0.3:
            insights.append(
                f"Low conviction ({conviction:.0%}) — wait for stronger signal or reduce size"
            )

        if bias == "neutral" and regime.get("confidence", 0) > 0.7:
            insights.append(
                "Regime clear but indicators mixed — HOLD, no new positions"
            )

        return insights[:4]

    # ------------------------------------------------------------------ #
    # Summary and quality                                                  #
    # ------------------------------------------------------------------ #

    def _build_summary(
        self,
        symbol: str,
        bias: _Bias,
        conviction: float,
        regime: dict,
        events: dict,
    ) -> str:
        trend = regime.get("trend_regime", "unknown")
        vol = regime.get("volatility_regime", "normal")
        source = regime.get("regime_source", "rule_based")
        next_event = events.get("next_event_desc", "none")
        return (
            f"{symbol}: {bias.replace('_', ' ')} bias "
            f"(conviction {conviction:.0%}). "
            f"Regime: {trend}, vol: {vol} [{source}]. "
            f"Next event: {next_event}."
        )

    def _assess_quality(
        self, regime: dict, technical: dict, failures: list[str]
    ) -> _Quality:
        if "technical" in failures and "regime" in failures:
            return "low"
        if regime.get("trend_regime") == "unknown":
            return "low"
        # HMM with high stability is more trustworthy
        hmm_stability = regime.get("hmm_stability")
        if hmm_stability is not None and hmm_stability > 0.7 and not failures:
            return "high"
        if regime.get("confidence", 0) > 0.70 and not failures:
            return "high"
        if len(failures) > 1:
            return "low"
        return "medium"


# ------------------------------------------------------------------ #
# Portfolio-level helpers (used by engine.py to build SignalBrief)   #
# ------------------------------------------------------------------ #


def _active_pods(failures: list[str]) -> list[str]:
    all_pods = ["technical", "regime", "volume", "risk", "events", "fundamentals"]
    return [p for p in all_pods if p not in failures]


def map_to_market_bias(symbol_briefs: list[SymbolBrief]) -> tuple[_MktBias, float]:
    """Aggregate per-symbol biases into a portfolio-level market bias + conviction."""
    if not symbol_briefs:
        return "neutral", 0.5

    scores = []
    for sb in symbol_briefs:
        score_map = {
            "strong_bullish": 1.0,
            "bullish": 0.5,
            "neutral": 0.0,
            "bearish": -0.5,
            "strong_bearish": -1.0,
        }
        scores.append(score_map.get(sb.consensus_bias, 0.0) * sb.consensus_conviction)

    avg = sum(scores) / len(scores)
    conviction = min(0.95, abs(avg) + 0.1 * len(scores) / 5)

    if avg > 0.2:
        return "bullish", round(conviction, 3)
    if avg < -0.2:
        return "bearish", round(conviction, 3)
    return "neutral", round(conviction, 3)


def map_to_risk_environment(
    symbol_briefs: list[SymbolBrief], regimes: list[dict]
) -> _RiskEnv:
    """Derive portfolio-level risk environment from per-symbol regime outputs."""
    vol_levels = [r.get("volatility_regime", "normal") for r in regimes]
    if "extreme" in vol_levels:
        return "high"
    high_count = vol_levels.count("high")
    if high_count >= len(vol_levels) * 0.5:
        return "elevated"
    low_count = vol_levels.count("low")
    if low_count == len(vol_levels):
        return "low"
    return "normal"
