# Copyright 2024 QuantPod Contributors
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
import time
from datetime import date
from typing import Any

from loguru import logger

from quant_pod.signal_engine.brief import SignalBrief
from quant_pod.signal_engine.collectors.events import collect_events
from quant_pod.signal_engine.collectors.fundamentals import collect_fundamentals
from quant_pod.signal_engine.collectors.regime import collect_regime
from quant_pod.signal_engine.collectors.risk import collect_risk
from quant_pod.signal_engine.collectors.sentiment import collect_sentiment
from quant_pod.signal_engine.collectors.technical import collect_technical
from quant_pod.signal_engine.collectors.volume import collect_volume
from quant_pod.signal_engine.synthesis import (
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
        from quantcore.data.storage import DataStore
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
        logger.info(f"[SignalEngine] Starting analysis for {symbol}")

        outputs, failures = await self._run_collectors(symbol)

        # If a pre-computed regime was supplied, override the regime collector output.
        if regime:
            outputs["regime"] = {**outputs.get("regime", {}), **regime}

        brief = self._build_brief(symbol, outputs, failures)
        duration_ms = (time.monotonic() - t0) * 1000
        brief.collection_duration_ms = round(duration_ms, 1)

        logger.info(
            f"[SignalEngine] {symbol} done in {duration_ms:.0f}ms "
            f"| bias={brief.market_bias} confidence={brief.overall_confidence:.2f}"
            f"{' | failures: ' + str(failures) if failures else ''}"
        )
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

    async def _run_collectors(
        self, symbol: str
    ) -> tuple[dict[str, Any], list[str]]:
        """Run all collectors concurrently; isolate failures."""
        collector_map = {
            "technical":    collect_technical(symbol, self._store),
            "regime":       collect_regime(symbol, self._store),
            "volume":       collect_volume(symbol, self._store),
            "risk":         collect_risk(symbol, self._store),
            "events":       collect_events(symbol, self._store),
            "fundamentals": collect_fundamentals(symbol, self._store),
            "sentiment":    collect_sentiment(symbol, self._store),
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
                logger.warning(f"[SignalEngine] collector '{name}' failed: {result}")
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
        technical    = outputs.get("technical", {})
        regime       = outputs.get("regime", {})
        volume       = outputs.get("volume", {})
        risk         = outputs.get("risk", {})
        events       = outputs.get("events", {})
        fundamentals = outputs.get("fundamentals", {})
        sentiment    = outputs.get("sentiment", {})

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
        )

        market_bias, market_conviction = map_to_market_bias([symbol_brief])
        risk_env = map_to_risk_environment([symbol_brief], [regime])

        # Build top-level key risks and opportunities lists.
        key_risks = symbol_brief.risk_factors[:3]
        top_opportunities = (
            [symbol] if symbol_brief.consensus_bias in ("bullish", "strong_bullish")
                     and symbol_brief.consensus_conviction >= 0.5
            else []
        )

        # overall_confidence is the symbol brief conviction, adjusted for failures.
        base_confidence = symbol_brief.consensus_conviction
        if failures:
            base_confidence = max(0.1, base_confidence - 0.05 * len(failures))

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
        )


# ------------------------------------------------------------------ #
# Helpers                                                             #
# ------------------------------------------------------------------ #

def _read_strategy_context() -> str:
    """Read strategy_registry.md from memory (same injection as run_analysis)."""
    try:
        from quant_pod.mcp._state import _read_memory_file
        return _read_memory_file("strategy_registry.md", max_chars=2000)
    except Exception:
        return ""


def _active_pods(failures: list[str]) -> list[str]:
    return [p for p in ("technical", "regime", "volume", "risk", "events", "fundamentals", "sentiment")
            if p not in failures]


def _empty_brief(symbol: str) -> SignalBrief:
    """Return a minimal valid brief for a symbol that completely failed analysis."""
    from quant_pod.crews.schemas import SymbolBrief as SB
    return SignalBrief(
        date=date.today(),
        market_overview=f"{symbol}: analysis failed — no data available",
        market_bias="neutral",
        market_conviction=0.0,
        risk_environment="normal",
        symbol_briefs=[SB(
            symbol=symbol,
            market_summary="Analysis failed",
            consensus_bias="neutral",
            consensus_conviction=0.0,
            pod_agreement="mixed",
            analysis_quality="low",
        )],
        top_opportunities=[],
        key_risks=["Analysis pipeline failure — no data for this symbol"],
        strategic_notes="",
        pods_reporting=0,
        total_analyses=0,
        overall_confidence=0.0,
        collector_failures=["all"],
    )
