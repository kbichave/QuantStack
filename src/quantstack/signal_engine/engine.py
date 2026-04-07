# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
SignalEngine — orchestrates collectors and builds a SignalBrief.

Replaces TradingCrew (13 IC agents + 6 pod managers + trading assistant).
Total wall-clock time: 2–6 seconds (I/O-bound on first OHLCV load).
No LLM calls. No CrewAI. No Ollama.

Design invariants:
- Each collector runs concurrently via asyncio.gather with a per-collector timeout.
- A failure (timeout or exception) in one collector yields an empty dict and
  records the collector name in collector_failures; other collectors continue.
- The result is always a valid SignalBrief.  An all-failures run produces a brief
  with market_bias="neutral", overall_confidence=0.0, analysis_quality="low".
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import date
from typing import Any

from loguru import logger

from quantstack.db import db_conn

from quantstack.data.storage import DataStore
from quantstack.learning.drift_detector import DriftDetector
from quantstack.shared.files import read_memory_file
from quantstack.shared.schemas import SymbolBrief as SB
from quantstack.signal_engine.brief import SignalBrief
from quantstack.signal_engine.collectors.cross_asset import collect_cross_asset
from quantstack.signal_engine.collectors.events import collect_events
from quantstack.signal_engine.collectors.flow import collect_flow
from quantstack.signal_engine.collectors.fundamentals import collect_fundamentals
from quantstack.signal_engine.collectors.macro import collect_macro
from quantstack.signal_engine.collectors.ml_signal import collect_ml_signal
from quantstack.signal_engine.collectors.options_flow_collector import (
    collect_options_flow_async,
)
from quantstack.signal_engine.collectors.quality import collect_quality
from quantstack.signal_engine.collectors.regime import collect_regime
from quantstack.signal_engine.collectors.risk import collect_risk
from quantstack.signal_engine.collectors.sector import collect_sector
from quantstack.signal_engine.collectors.sentiment import collect_sentiment
from quantstack.signal_engine.collectors.sentiment_alphavantage import (
    collect_sentiment_alphavantage,
)
from quantstack.signal_engine.collectors.insider_signals import collect_insider_signals
from quantstack.signal_engine.collectors.short_interest import collect_short_interest
from quantstack.signal_engine.collectors.put_call_ratio import collect_put_call_ratio
from quantstack.signal_engine.collectors.earnings_momentum import (
    collect_earnings_momentum,
)
from quantstack.signal_engine.collectors.commodity import collect_commodity_signals
from quantstack.signal_engine.collectors.social_sentiment import collect_social_sentiment
from quantstack.signal_engine.collectors.statarb import collect_statarb
from quantstack.signal_engine.collectors.technical import collect_technical
from quantstack.signal_engine.collectors.volume import collect_volume
from quantstack.signal_engine.collectors.ewf_collector import collect_ewf
from quantstack.signal_engine.synthesis import (
    RuleBasedSynthesizer,
    map_to_market_bias,
    map_to_risk_environment,
)

# Per-collector wall-clock timeout.  Events collector has its own internal
# 6-second timeout; this is the outer safety net.
_COLLECTOR_TIMEOUT = 10.0


class SignalEngine:
    """
    Pure-Python analysis pipeline.

    Usage:
        brief = await SignalEngine().run("XOM")
        briefs = await SignalEngine().run_multi(["XOM", "MSFT", "SPY"])
    """

    def __init__(self, db_path: str | None = None) -> None:
        # DataStore is opened read-only — multiple concurrent reads are safe.
        self._store = DataStore(db_path=db_path, read_only=True)
        self._synthesizer = RuleBasedSynthesizer()

    async def run(
        self,
        symbol: str,
        regime: dict[str, Any] | None = None,
    ) -> SignalBrief:
        """
        Run the full analysis pipeline for a single symbol.

        Args:
            symbol: Ticker (e.g., "XOM").
            regime: Pre-computed regime dict.  If None, detected by regime collector.

        Returns:
            SignalBrief — DailyBrief-compatible structured analysis.
        """
        t0 = time.monotonic()
        symbol = symbol.upper().strip()

        # Check TTL cache before running collectors.
        from quantstack.signal_engine.cache import get as _cache_get, put as _cache_put

        cached = _cache_get(symbol)
        if cached is not None:
            logger.debug(
                f"[SignalEngine] {symbol} served from cache "
                f"(original duration={cached.collection_duration_ms}ms)"
            )
            return cached

        logger.info(f"[SignalEngine] Starting analysis for {symbol}")

        outputs, failures = await self._run_collectors(symbol)

        # If a pre-computed regime was supplied, override the regime collector output.
        if regime:
            outputs["regime"] = {**outputs.get("regime", {}), **regime}

        brief = self._build_brief(symbol, outputs, failures)
        duration_ms = (time.monotonic() - t0) * 1000
        brief.collection_duration_ms = round(duration_ms, 1)

        # Drift detection — best-effort, never blocks brief delivery.
        # Uses symbol as strategy_id proxy for feature distribution tracking.
        drift_report = None
        try:
            drift_report = DriftDetector().check_drift_from_brief(
                strategy_id=symbol,
                brief=brief.model_dump(),
            )
            if drift_report.severity == "CRITICAL":
                brief.drift_warning = True
                logger.warning(
                    f"[SignalEngine] {symbol} CRITICAL drift: "
                    f"PSI={drift_report.overall_psi:.3f} "
                    f"features={drift_report.drifted_features}"
                )
                # Queue an ML architecture search — AutoResearchClaw will investigate
                # whether the current model needs retraining or feature engineering.
                try:
                    with db_conn() as _conn:
                        _conn.execute(
                            """
                            INSERT INTO research_queue
                                (task_type, priority, context_json, source)
                            VALUES ('ml_arch_search', %s, %s, 'drift_detector')
                            ON CONFLICT DO NOTHING
                            """,
                            [
                                8,
                                json.dumps({
                                    "symbol": symbol,
                                    "psi": drift_report.overall_psi,
                                    "drifted_features": drift_report.drifted_features,
                                    "severity": drift_report.severity,
                                }),
                            ],
                        )
                except Exception as _rq_exc:
                    logger.debug(
                        f"[SignalEngine] research_queue insert failed (non-critical): {_rq_exc}"
                    )
        except Exception as _drift_exc:
            logger.debug(f"[SignalEngine] drift check failed (non-critical): {_drift_exc}")

        # Apply confidence penalty and determine cache TTL based on drift severity.
        cache_ttl: int | None = None
        if drift_report is not None:
            if drift_report.severity == "WARNING":
                brief.overall_confidence = max(0.0, brief.overall_confidence - 0.10)
                cache_ttl = 1800
            elif drift_report.severity == "CRITICAL":
                brief.overall_confidence = max(0.0, brief.overall_confidence - 0.30)
                cache_ttl = 300
                try:
                    with db_conn() as _conn:
                        _conn.execute(
                            """
                            INSERT INTO system_events
                                (event_type, symbol, severity, details, created_at)
                            VALUES ('DRIFT_CRITICAL', %s, 'critical', %s, NOW())
                            """,
                            [
                                symbol,
                                json.dumps({
                                    "psi": drift_report.overall_psi,
                                    "drifted_features": drift_report.drifted_features,
                                }),
                            ],
                        )
                except Exception:
                    logger.debug("[SignalEngine] system_events insert failed (non-critical)")

        logger.info(
            f"[SignalEngine] {symbol} done in {duration_ms:.0f}ms "
            f"| bias={brief.market_bias} confidence={brief.overall_confidence:.2f}"
            f"{' | failures: ' + str(failures) if failures else ''}"
        )

        _cache_put(symbol, brief, ttl=cache_ttl)
        return brief

    async def run_multi(
        self,
        symbols: list[str],
        max_concurrent: int = 5,
    ) -> list[SignalBrief]:
        """
        Run analysis for multiple symbols with bounded concurrency.

        Args:
            symbols: List of tickers.
            max_concurrent: Max parallel symbol analyses (default 5).

        Returns:
            List of SignalBriefs in the same order as *symbols*.
        """
        sem = asyncio.Semaphore(max_concurrent)

        async def _bounded(sym: str) -> SignalBrief:
            async with sem:
                try:
                    return await self.run(sym)
                except Exception as exc:
                    logger.error(f"[SignalEngine] {sym} failed: {exc}")
                    return _empty_brief(sym)

        return list(await asyncio.gather(*[_bounded(s) for s in symbols]))

    # ------------------------------------------------------------------ #
    # Internal                                                             #
    # ------------------------------------------------------------------ #

    async def _run_collectors(self, symbol: str) -> tuple[dict[str, Any], list[str]]:
        """Run all collectors concurrently; isolate failures."""
        collector_map = {
            # Core collectors (v0.6.0)
            "technical": collect_technical(symbol, self._store),
            "regime": collect_regime(symbol, self._store),
            "volume": collect_volume(symbol, self._store),
            "risk": collect_risk(symbol, self._store),
            "events": collect_events(symbol, self._store),
            "fundamentals": collect_fundamentals(symbol, self._store),
            # v1.1: Sentiment now uses Alpha Vantage + Groq reasoning with context
            "sentiment": collect_sentiment_alphavantage(symbol, self._store),
            # Phase 3 collectors (v1.0) — all optional, graceful {} on failure
            "macro": collect_macro(symbol, self._store),
            "sector": collect_sector(symbol, self._store),
            "flow": collect_flow(symbol, self._store),
            "cross_asset": collect_cross_asset(symbol, self._store),
            "quality": collect_quality(symbol, self._store),
            "ml_signal": collect_ml_signal(symbol, self._store),
            "statarb": collect_statarb(symbol, self._store),
            # Phase 4 collectors (v1.1) — options flow (GEX, gamma flip, DEX, etc.)
            "options_flow": collect_options_flow_async(symbol, self._store),
            # Phase 5 collectors (v1.2) — community social sentiment (Reddit + Stocktwits)
            "social": collect_social_sentiment(symbol, self._store),
            # Phase 6 collectors (v1.3) — alternative data signals
            "insider": collect_insider_signals(symbol, self._store),
            "short_interest": collect_short_interest(symbol, self._store),
            # Phase 7 collectors (v2.0) — AV data expansion
            "put_call_ratio": collect_put_call_ratio(symbol, self._store),
            "earnings_momentum": collect_earnings_momentum(symbol, self._store),
            "commodity": collect_commodity_signals(symbol, self._store),
            # EWF Elliott Wave Forecast — non-critical, returns {} when no fresh analysis
            "ewf": collect_ewf(symbol, self._store),
        }

        names = list(collector_map.keys())
        coros = [
            asyncio.wait_for(coro, timeout=_COLLECTOR_TIMEOUT)
            for coro in collector_map.values()
        ]

        raw_results = await asyncio.gather(*coros, return_exceptions=True)

        outputs: dict[str, Any] = {}
        failures: list[str] = []

        for name, result in zip(names, raw_results):
            if isinstance(result, (Exception, BaseException)):
                logger.warning(
                    f"[SignalEngine] collector '{name}' failed: "
                    f"{type(result).__name__}: {result}"
                )
                failures.append(name)
                outputs[name] = {}
            else:
                outputs[name] = result if result else {}

        return outputs, failures

    def _build_brief(
        self,
        symbol: str,
        outputs: dict[str, Any],
        failures: list[str],
    ) -> SignalBrief:
        """Synthesize collector outputs into a SignalBrief."""
        technical = outputs.get("technical", {})
        regime = outputs.get("regime", {})
        volume = outputs.get("volume", {})
        risk = outputs.get("risk", {})
        events = outputs.get("events", {})
        fundamentals = outputs.get("fundamentals", {})
        sentiment = outputs.get("sentiment", {})
        macro = outputs.get("macro", {})
        sector = outputs.get("sector", {})
        flow = outputs.get("flow", {})
        cross_asset = outputs.get("cross_asset", {})
        quality = outputs.get("quality", {})
        ml_signal = outputs.get("ml_signal", {})
        statarb = outputs.get("statarb", {})
        options_flow = outputs.get("options_flow", {})
        social = outputs.get("social", {})
        put_call_ratio = outputs.get("put_call_ratio", {})
        earnings_momentum = outputs.get("earnings_momentum", {})
        commodity = outputs.get("commodity", {})
        ewf = outputs.get("ewf", {})

        # Inject strategy context from memory (same as run_analysis).
        strategy_context = _read_strategy_context()

        symbol_brief = self._synthesizer.synthesize(
            symbol=symbol,
            technical=technical,
            regime=regime,
            volume=volume,
            risk=risk,
            events=events,
            fundamentals=fundamentals,
            sentiment=sentiment,
            collector_failures=failures,
            strategy_context=strategy_context,
            ml_signal=ml_signal,
            flow=flow,
            put_call_ratio=put_call_ratio,
            earnings_momentum=earnings_momentum,
        )

        market_bias, market_conviction = map_to_market_bias([symbol_brief])
        risk_env = map_to_risk_environment([symbol_brief], [regime])

        # Build top-level key risks and opportunities lists.
        key_risks = symbol_brief.risk_factors[:3]
        top_opportunities = (
            [symbol]
            if symbol_brief.consensus_bias in ("bullish", "strong_bullish")
            and symbol_brief.consensus_conviction >= 0.5
            else []
        )

        # overall_confidence is the symbol brief conviction, adjusted for failures.
        # EWF is excluded: returning {} is expected when no fresh analysis exists.
        base_confidence = symbol_brief.consensus_conviction
        _penalized_failures = [f for f in failures if f != "ewf"]
        if _penalized_failures:
            base_confidence = max(0.1, base_confidence - 0.05 * len(_penalized_failures))

        return SignalBrief(
            date=date.today(),
            market_overview=symbol_brief.market_summary,
            market_bias=market_bias,
            market_conviction=round(market_conviction, 3),
            risk_environment=risk_env,
            symbol_briefs=[symbol_brief],
            top_opportunities=top_opportunities,
            key_risks=key_risks,
            strategic_notes=strategy_context,
            pods_reporting=len(_active_pods(failures)),
            total_analyses=len(outputs) - len(failures),
            overall_confidence=round(base_confidence, 3),
            collector_failures=failures,
            regime_detail=regime or None,
            sentiment_score=sentiment.get("sentiment_score", 0.5),
            dominant_sentiment=sentiment.get("dominant_sentiment", "neutral"),
            # Phase 3 fields — all default to "unknown"/None if collector returned {}
            macro_rate_regime=macro.get("rate_regime", "unknown"),
            yield_curve_slope=macro.get("yield_curve_slope"),
            sector_signal=sector.get("sector_trend", "unknown"),
            rotation_signal=sector.get("rotation_signal", "unknown"),
            breadth_positive_sectors=sector.get("breadth_positive_sectors"),
            flow_signal=flow.get("flow_signal"),
            insider_direction=flow.get("insider_direction", "unknown"),
            cross_asset_regime=cross_asset.get("cross_asset_regime", "unknown"),
            risk_on_score=cross_asset.get("risk_on_score"),
            quality_score=quality.get("quality_score"),
            ml_prediction=ml_signal.get("ml_prediction"),
            ml_direction=ml_signal.get("ml_direction", "unknown"),
            statarb_signal=statarb.get("statarb_signal", "unknown"),
            spread_zscore=statarb.get("spread_zscore"),
            # Options flow (dealer positioning)
            opt_gex=options_flow.get("opt_gex"),
            opt_gamma_flip=options_flow.get("opt_gamma_flip"),
            opt_above_gamma_flip=options_flow.get("opt_above_gamma_flip"),
            opt_dex=options_flow.get("opt_dex"),
            opt_max_pain=options_flow.get("opt_max_pain"),
            opt_iv_skew=options_flow.get("opt_iv_skew"),
            opt_iv_skew_zscore=options_flow.get("opt_iv_skew_zscore"),
            opt_vrp=options_flow.get("opt_vrp"),
            opt_charm=options_flow.get("opt_charm"),
            opt_vanna=options_flow.get("opt_vanna"),
            opt_ehd=options_flow.get("opt_ehd"),
            # Community sentiment (16th collector)
            social=social,
            # Phase 7 — AV data expansion
            pcr_signal=put_call_ratio.get("pcr_signal"),
            pcr_10d_sma=put_call_ratio.get("pcr_10d_sma"),
            earnings_momentum_score=earnings_momentum.get("earnings_momentum_score"),
            consecutive_beats=earnings_momentum.get("consecutive_beats"),
            drift_active=earnings_momentum.get("drift_active", False),
            commodity_regime=commodity.get("commodity_regime", "unknown"),
            sector_rotation_signal_commodity=commodity.get(
                "sector_rotation_signal", "unknown"
            ),
            risk_off_score=commodity.get("risk_off_score"),
            gold_silver_ratio=commodity.get("gold_silver_ratio"),
            copper_gold_ratio=commodity.get("copper_gold_ratio"),
            usd_strength_proxy=commodity.get("usd_strength_proxy"),
            # EWF Elliott Wave fields
            ewf_bias=ewf.get("ewf_bias"),
            ewf_turning_signal=ewf.get("ewf_turning_signal"),
            ewf_wave_position=ewf.get("ewf_wave_position"),
            ewf_wave_degree=ewf.get("ewf_wave_degree"),
            ewf_current_wave_label=ewf.get("ewf_current_wave_label"),
            ewf_confidence=ewf.get("ewf_confidence"),
            ewf_key_support=ewf.get("ewf_key_support", []),
            ewf_key_resistance=ewf.get("ewf_key_resistance", []),
            ewf_invalidation_level=ewf.get("ewf_invalidation_level"),
            ewf_target=ewf.get("ewf_target"),
            ewf_blue_box_active=ewf.get("ewf_blue_box_active", False),
            ewf_blue_box_low=ewf.get("ewf_blue_box_low"),
            ewf_blue_box_high=ewf.get("ewf_blue_box_high"),
            ewf_summary=ewf.get("ewf_summary"),
            ewf_projected_path=ewf.get("ewf_projected_path"),
            ewf_timeframe_used=ewf.get("ewf_timeframe_used"),
            ewf_age_hours=ewf.get("ewf_age_hours"),
        )


# ------------------------------------------------------------------ #
# Helpers                                                             #
# ------------------------------------------------------------------ #


def _read_strategy_context() -> str:
    """Read strategy_registry.md from memory (same injection as run_analysis)."""
    try:
        return read_memory_file("strategy_registry.md", max_chars=2000)
    except Exception:
        return ""


_ALL_COLLECTORS = (
    "technical",
    "regime",
    "volume",
    "risk",
    "events",
    "fundamentals",
    "sentiment",
    "macro",
    "sector",
    "flow",
    "cross_asset",
    "quality",
    "ml_signal",
    "statarb",
    "options_flow",
    "social",
    "insider",
    "short_interest",
    "put_call_ratio",
    "earnings_momentum",
    "commodity",
    "ewf",
)


def _active_pods(failures: list[str]) -> list[str]:
    return [p for p in _ALL_COLLECTORS if p not in failures]


def _empty_brief(symbol: str) -> SignalBrief:
    """Return a minimal valid brief for a symbol that completely failed analysis."""
    return SignalBrief(
        date=date.today(),
        market_overview=f"{symbol}: analysis failed — no data available",
        market_bias="neutral",
        market_conviction=0.0,
        risk_environment="normal",
        symbol_briefs=[
            SB(
                symbol=symbol,
                market_summary="Analysis failed",
                consensus_bias="neutral",
                consensus_conviction=0.0,
                pod_agreement="mixed",
                analysis_quality="low",
            )
        ],
        top_opportunities=[],
        key_risks=["Analysis pipeline failure — no data for this symbol"],
        strategic_notes="",
        pods_reporting=0,
        total_analyses=0,
        overall_confidence=0.0,
        collector_failures=["all"],
    )
