# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Autonomous screener — scores universe symbols and produces a tiered watchlist.

Runs daily at 08:00 ET (after CacheWarmer at 06:00 ET).  Uses ONLY cached
cached database data — no API calls.  This keeps the screener fast (~2-5 seconds for
700 symbols) and free from rate-limit concerns.

Scoring weights match the interactive watchlist agent (.claude/agents/watchlist.md):
  momentum: 25%, volatility_rank: 20%, catalyst_proximity: 20%,
  regime_fit: 20%, volume_surge: 15%

Output tiers:
  Tier 1 (top 15): Full treatment — SignalEngine + ML + Groq fallback
  Tier 2 (next 20): SignalEngine + rule-based routing only
  Tier 3 (next 15): Monitored, not actively traded

Hard filters (applied before scoring):
  ADV < 500k shares → excluded
  In RISK_RESTRICTED_SYMBOLS → excluded
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any

from loguru import logger

from quantstack.db import PgConnection

import numpy as np


@dataclass
class ScreenerScore:
    """Scored symbol with tier assignment."""

    symbol: str
    momentum_score: float = 0.0
    volatility_rank: float = 0.0
    catalyst_proximity: float = 0.0
    regime_fit: float = 0.0
    volume_surge: float = 0.0
    composite: float = 0.0
    tier: int = 3


@dataclass
class ScreenerResult:
    """Output of a screening pass."""

    screened_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    regime_used: str = "unknown"
    universe_size: int = 0
    filtered_out: int = 0
    tier_1: list[ScreenerScore] = field(default_factory=list)
    tier_2: list[ScreenerScore] = field(default_factory=list)
    tier_3: list[ScreenerScore] = field(default_factory=list)

    @property
    def total_watchlist(self) -> int:
        return len(self.tier_1) + len(self.tier_2) + len(self.tier_3)


# Scoring weights — match watchlist.md
_W_MOMENTUM = 0.25
_W_VOLATILITY = 0.20
_W_CATALYST = 0.20
_W_REGIME_FIT = 0.20
_W_VOLUME = 0.15

# Tier sizes
_TIER_1_SIZE = 15
_TIER_2_SIZE = 20
_TIER_3_SIZE = 15

# Hard filter: minimum average daily volume
_MIN_ADV = 500_000


class AutonomousScreener:
    """
    Scores universe symbols using cached OHLCV data.

    Does NOT call SignalEngine (too expensive for 700 symbols).
    Uses only cached data from DataStore + universe table.
    """

    def __init__(
        self,
        conn: PgConnection,
        active_regimes: list[str] | None = None,
    ) -> None:
        self._conn = conn
        self._active_regimes = active_regimes or []
        self._restricted = set(
            s.strip().upper()
            for s in os.getenv("RISK_RESTRICTED_SYMBOLS", "").split(",")
            if s.strip()
        )

    async def screen(self, regime: str = "unknown") -> ScreenerResult:
        """
        Score all active universe symbols and return a tiered watchlist.

        This is an async method but all work is CPU/database-bound (no I/O).
        It's async for compatibility with the runner's event loop.
        """
        return await asyncio.to_thread(self._screen_sync, regime)

    def _screen_sync(self, regime: str) -> ScreenerResult:
        """Synchronous screening implementation."""
        result = ScreenerResult(regime_used=regime)

        # Load universe symbols with their metadata
        symbols = self._load_universe()
        result.universe_size = len(symbols)

        # Apply hard filters
        eligible = self._apply_hard_filters(symbols)
        result.filtered_out = result.universe_size - len(eligible)

        if not eligible:
            logger.warning("[Screener] No eligible symbols after filtering")
            return result

        # Score each symbol
        scores: list[ScreenerScore] = []
        for sym_data in eligible:
            symbol = sym_data["symbol"]
            score = self._score_symbol(symbol, sym_data, regime)
            scores.append(score)

        # Sort by composite score descending
        scores.sort(key=lambda s: s.composite, reverse=True)

        # Assign tiers
        for i, score in enumerate(scores):
            if i < _TIER_1_SIZE:
                score.tier = 1
                result.tier_1.append(score)
            elif i < _TIER_1_SIZE + _TIER_2_SIZE:
                score.tier = 2
                result.tier_2.append(score)
            elif i < _TIER_1_SIZE + _TIER_2_SIZE + _TIER_3_SIZE:
                score.tier = 3
                result.tier_3.append(score)
            else:
                break  # Only keep top 50

        # Persist to screener_results table
        self._persist_results(result)

        logger.info(
            f"[Screener] Scored {len(scores)} symbols: "
            f"T1={len(result.tier_1)} T2={len(result.tier_2)} T3={len(result.tier_3)} "
            f"(filtered out {result.filtered_out})"
        )

        return result

    def _load_universe(self) -> list[dict[str, Any]]:
        """Load active universe symbols with metadata."""
        rows = self._conn.execute(
            """
            SELECT symbol, name, sector, source, market_cap, avg_daily_volume
            FROM universe
            WHERE is_active = TRUE
            ORDER BY symbol
            """
        ).fetchall()
        return [
            {
                "symbol": r[0],
                "name": r[1],
                "sector": r[2],
                "source": r[3],
                "market_cap": r[4],
                "avg_daily_volume": r[5],
            }
            for r in rows
        ]

    def _apply_hard_filters(
        self, symbols: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Apply hard exclusion filters."""
        eligible: list[dict[str, Any]] = []
        for sym in symbols:
            symbol = sym["symbol"]

            # Restricted symbols
            if symbol in self._restricted:
                continue

            # ADV filter — use stored value if available, otherwise skip filter
            adv = sym.get("avg_daily_volume")
            if adv is not None and adv < _MIN_ADV:
                continue

            eligible.append(sym)
        return eligible

    def _score_symbol(
        self, symbol: str, sym_data: dict[str, Any], regime: str
    ) -> ScreenerScore:
        """Compute composite score for a single symbol."""
        momentum = self._compute_momentum(symbol)
        volatility = self._compute_volatility_rank(symbol)
        catalyst = self._compute_catalyst_proximity(symbol)
        regime_fit = self._compute_regime_fit(symbol, sym_data, regime)
        volume = self._compute_volume_surge(symbol)

        composite = (
            _W_MOMENTUM * momentum
            + _W_VOLATILITY * volatility
            + _W_CATALYST * catalyst
            + _W_REGIME_FIT * regime_fit
            + _W_VOLUME * volume
        )

        return ScreenerScore(
            symbol=symbol,
            momentum_score=momentum,
            volatility_rank=volatility,
            catalyst_proximity=catalyst,
            regime_fit=regime_fit,
            volume_surge=volume,
            composite=composite,
        )

    def _compute_momentum(self, symbol: str) -> float:
        """
        Momentum score from SMA slopes (0-1).

        Uses 20d, 50d, 200d SMAs from cached OHLCV.
        Higher score = stronger upward momentum.
        """
        try:
            rows = self._conn.execute(
                """
                SELECT close FROM ohlcv
                WHERE symbol = ? AND timeframe = 'D1'
                ORDER BY timestamp DESC
                LIMIT 200
                """,
                [symbol],
            ).fetchall()

            if len(rows) < 20:
                return 0.5  # Neutral if insufficient data

            closes = [r[0] for r in reversed(rows)]

            sma_20 = sum(closes[-20:]) / 20
            current = closes[-1]

            # Price vs SMA-20: positive = above, negative = below
            pct_above_20 = (current - sma_20) / sma_20 if sma_20 else 0

            # 20-day return
            ret_20 = (closes[-1] - closes[-20]) / closes[-20] if closes[-20] else 0

            # Combine: normalize to 0-1 range
            # +10% above SMA = 1.0, -10% below = 0.0
            score = 0.5 + (pct_above_20 * 5.0)  # Scale so ±10% maps to 0-1
            score = max(0.0, min(1.0, score))

            # Boost for strong recent returns
            if ret_20 > 0.05:
                score = min(1.0, score + 0.1)
            elif ret_20 < -0.05:
                score = max(0.0, score - 0.1)

            return round(score, 3)
        except Exception:
            return 0.5

    def _compute_volatility_rank(self, symbol: str) -> float:
        """
        Volatility rank score (0-1).

        Targets 30th-70th percentile ATR. Extremes (very low or very high) score lower.
        """
        try:
            rows = self._conn.execute(
                """
                SELECT high, low, close FROM ohlcv
                WHERE symbol = ? AND timeframe = 'D1'
                ORDER BY timestamp DESC
                LIMIT 20
                """,
                [symbol],
            ).fetchall()

            if len(rows) < 14:
                return 0.5

            # Simple ATR proxy: average of (high - low) / close
            ratios = [(r[0] - r[1]) / r[2] if r[2] else 0 for r in rows]
            avg_ratio = sum(ratios) / len(ratios)

            # Ideal range: 1-3% daily range. Score peaks at 2%.
            # Below 0.5% or above 5% = low score.
            if avg_ratio < 0.005:
                score = avg_ratio / 0.005 * 0.3  # Low vol = low score
            elif avg_ratio < 0.01:
                score = 0.3 + (avg_ratio - 0.005) / 0.005 * 0.4
            elif avg_ratio < 0.03:
                score = 0.7 + (0.03 - avg_ratio) / 0.02 * 0.3  # Sweet spot
            elif avg_ratio < 0.05:
                score = 0.5 - (avg_ratio - 0.03) / 0.02 * 0.3  # Getting too volatile
            else:
                score = max(0.0, 0.2 - (avg_ratio - 0.05) * 5)  # Extreme vol

            return round(max(0.0, min(1.0, score)), 3)
        except Exception:
            return 0.5

    def _compute_catalyst_proximity(self, symbol: str) -> float:
        """
        Catalyst proximity score (0-1).

        Higher score if earnings are 5-14 days away (pre-event positioning).
        Today = 0 (skip via /earnings). 1-4 days = lower. >14 days = neutral.
        """
        try:
            today = date.today()
            row = self._conn.execute(
                """
                SELECT MIN(report_date) FROM earnings_calendar
                WHERE symbol = ? AND report_date >= ?
                """,
                [symbol, today.isoformat()],
            ).fetchone()

            if not row or not row[0]:
                return 0.3  # No catalyst = lower score

            report_date = row[0]
            if isinstance(report_date, str):
                report_date = datetime.strptime(report_date, "%Y-%m-%d").date()
            elif isinstance(report_date, datetime):
                report_date = report_date.date()

            days_until = (report_date - today).days

            if days_until == 0:
                return 0.0  # Earnings today — use /earnings skill instead
            elif days_until <= 4:
                return 0.4  # Too close for setup
            elif days_until <= 14:
                return 0.8 + (14 - days_until) / 10 * 0.2  # Sweet spot
            elif days_until <= 30:
                return 0.5
            else:
                return 0.3  # Too far away
        except Exception:
            return 0.3

    def _compute_regime_fit(
        self, symbol: str, sym_data: dict[str, Any], regime: str
    ) -> float:
        """
        Regime fit score (0-1).

        Higher if the symbol's characteristics match the current regime and
        there are active strategies for this regime.
        """
        try:
            # Check how many active strategies match the current regime
            row = self._conn.execute(
                """
                SELECT COUNT(*) FROM regime_strategy_matrix
                WHERE regime = ?
                """,
                [regime],
            ).fetchone()
            strategy_count = row[0] if row else 0

            if strategy_count == 0:
                return 0.3  # No strategies for this regime

            # Sector bonus: some sectors do better in certain regimes
            sector = sym_data.get("sector", "")
            sector_bonus = 0.0
            if regime in ("trending_up", "bull"):
                if sector in ("Technology", "Consumer Discretionary", "Communication"):
                    sector_bonus = 0.15
            elif regime in ("trending_down", "bear"):
                if sector in ("Utilities", "Consumer Staples", "Healthcare"):
                    sector_bonus = 0.15
                elif sector in ("Commodities", "Energy"):
                    sector_bonus = 0.1

            base_score = min(1.0, 0.4 + strategy_count * 0.1)
            return round(min(1.0, base_score + sector_bonus), 3)
        except Exception:
            return 0.5

    def _compute_volume_surge(self, symbol: str) -> float:
        """
        Volume surge score (0-1).

        Compares recent volume (5d) to 20d average.
        Higher surge = higher score (indicates institutional interest).
        """
        try:
            rows = self._conn.execute(
                """
                SELECT volume FROM ohlcv
                WHERE symbol = ? AND timeframe = 'D1'
                ORDER BY timestamp DESC
                LIMIT 20
                """,
                [symbol],
            ).fetchall()

            if len(rows) < 10:
                return 0.5

            volumes = [r[0] for r in rows]
            recent_avg = sum(volumes[:5]) / 5
            full_avg = sum(volumes) / len(volumes)

            if full_avg == 0:
                return 0.5

            ratio = recent_avg / full_avg

            # ratio = 1.0 (normal) → 0.5
            # ratio = 1.5 (50% surge) → 0.8
            # ratio = 2.0+ (100% surge) → 1.0
            # ratio = 0.5 (50% decline) → 0.2
            score = 0.5 + (ratio - 1.0) * 0.5
            return round(max(0.0, min(1.0, score)), 3)
        except Exception:
            return 0.5

    def _persist_results(self, result: ScreenerResult) -> None:
        """Write screener results to the screener_results table."""
        try:
            all_scores = result.tier_1 + result.tier_2 + result.tier_3
            for score in all_scores:
                self._conn.execute(
                    """
                    INSERT INTO screener_results
                        (symbol, screened_at, regime_used, tier, composite_score,
                         momentum_score, volatility_rank, volume_surge,
                         regime_fit, catalyst_proximity)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        score.symbol,
                        result.screened_at,
                        result.regime_used,
                        score.tier,
                        score.composite,
                        score.momentum_score,
                        score.volatility_rank,
                        score.volume_surge,
                        score.regime_fit,
                        score.catalyst_proximity,
                    ],
                )
        except Exception as exc:
            logger.error(f"[Screener] Failed to persist results: {exc}")
