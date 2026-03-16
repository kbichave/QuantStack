# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Decoder Crew — reverse-engineers trading strategies from signal history.

Given a list of historical trade signals (entry/exit times and prices),
the decoder identifies:
  - Entry patterns (time-of-day, technical conditions)
  - Exit patterns (time-based, target-based, indicator-based)
  - Position sizing patterns (conviction model)
  - Regime affinity (which market regimes this strategy works in)

Architecture:
  4 IC analyzers (Python-based, no LLM needed for pattern extraction)
  + 1 synthesizer that combines IC outputs into a DecodedStrategy

The analysis is vectorized for speed and testability.  Future versions
can add LLM-based ICs for nuanced pattern interpretation.

Usage:
    from quant_pod.crews.decoder_crew import decode_signals

    signals = [
        {"symbol": "SPY", "direction": "long", "entry_time": "2024-01-15 10:30",
         "entry_price": 470.5, "exit_time": "2024-01-15 14:00", "exit_price": 473.2},
        ...
    ]
    result = decode_signals(signals, source_name="discord_trader_x")
"""

from __future__ import annotations

import statistics
from collections import Counter
from datetime import datetime
from typing import Any

from loguru import logger

# =============================================================================
# Signal data model
# =============================================================================


def _parse_signal(raw: dict[str, Any]) -> dict[str, Any] | None:
    """Normalize a raw signal dict, parsing timestamps and computing P&L."""
    try:
        entry_time = raw.get("entry_time")
        exit_time = raw.get("exit_time")
        entry_price = float(raw.get("entry_price", 0))
        exit_price = float(raw.get("exit_price", 0))
        direction = str(raw.get("direction", "long")).lower()
        symbol = str(raw.get("symbol", "UNKNOWN"))

        if isinstance(entry_time, str):
            # Try common formats
            for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
                try:
                    entry_time = datetime.strptime(entry_time, fmt)
                    break
                except ValueError:
                    continue
        if isinstance(exit_time, str):
            for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
                try:
                    exit_time = datetime.strptime(exit_time, fmt)
                    break
                except ValueError:
                    continue

        if not isinstance(entry_time, datetime) or not isinstance(exit_time, datetime):
            return None
        if entry_price <= 0 or exit_price <= 0:
            return None

        # P&L
        if direction == "long":
            pnl_pct = (exit_price - entry_price) / entry_price * 100
        else:
            pnl_pct = (entry_price - exit_price) / entry_price * 100

        holding_minutes = (exit_time - entry_time).total_seconds() / 60

        return {
            "symbol": symbol,
            "direction": direction,
            "entry_time": entry_time,
            "exit_time": exit_time,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "size": raw.get("size"),
            "notes": raw.get("notes", ""),
            "pnl_pct": pnl_pct,
            "is_winner": pnl_pct > 0,
            "holding_minutes": holding_minutes,
            "entry_hour": entry_time.hour,
            "entry_minute": entry_time.minute,
            "entry_dow": entry_time.weekday(),
            "exit_hour": exit_time.hour,
        }
    except Exception as e:
        logger.warning(f"Failed to parse signal: {e}")
        return None


# =============================================================================
# IC 1: Entry Pattern Analyzer
# =============================================================================


def _analyze_entry_patterns(signals: list[dict]) -> dict[str, Any]:
    """Identify entry timing and condition patterns."""
    if not signals:
        return {"error": "No signals to analyze"}

    entry_hours = [s["entry_hour"] for s in signals]
    entry_dows = [s["entry_dow"] for s in signals]
    directions = [s["direction"] for s in signals]

    # Most common entry hour
    hour_counts = Counter(entry_hours)
    peak_hour = hour_counts.most_common(1)[0]
    hour_concentration = peak_hour[1] / len(signals)

    # Day-of-week pattern
    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    dow_counts = Counter(entry_dows)
    peak_dow = dow_counts.most_common(1)[0]

    # Direction bias
    dir_counts = Counter(directions)
    long_pct = dir_counts.get("long", 0) / len(signals) * 100

    # Time clustering: are entries concentrated in a narrow window?
    morning = sum(1 for h in entry_hours if 9 <= h < 11)
    midday = sum(1 for h in entry_hours if 11 <= h < 14)
    afternoon = sum(1 for h in entry_hours if 14 <= h <= 16)

    if morning / len(signals) > 0.6:
        timing_pattern = "morning_trader"
    elif midday / len(signals) > 0.5:
        timing_pattern = "midday_trader"
    elif afternoon / len(signals) > 0.5:
        timing_pattern = "afternoon_trader"
    else:
        timing_pattern = "all_day_trader"

    return {
        "peak_entry_hour": peak_hour[0],
        "hour_concentration": round(hour_concentration, 2),
        "peak_entry_dow": dow_names[peak_dow[0]],
        "timing_pattern": timing_pattern,
        "long_pct": round(long_pct, 1),
        "direction_bias": "long" if long_pct > 65 else ("short" if long_pct < 35 else "mixed"),
        "total_signals": len(signals),
    }


# =============================================================================
# IC 2: Exit Pattern Analyzer
# =============================================================================


def _analyze_exit_patterns(signals: list[dict]) -> dict[str, Any]:
    """Classify exit patterns and compute holding period distribution."""
    if not signals:
        return {"error": "No signals"}

    holdings = [s["holding_minutes"] for s in signals]
    [s["pnl_pct"] for s in signals]
    winners = [s for s in signals if s["is_winner"]]
    losers = [s for s in signals if not s["is_winner"]]

    avg_holding = statistics.mean(holdings)
    median_holding = statistics.median(holdings)

    # Classify holding period
    if median_holding < 60:
        style = "scalper"
    elif median_holding < 480:
        style = "intraday"
    elif median_holding < 1440 * 5:
        style = "swing"
    else:
        style = "position"

    # Check if there's a fixed holding period (low variance)
    if len(holdings) > 3:
        cv = statistics.stdev(holdings) / (avg_holding + 1e-10)
        fixed_holding = cv < 0.3
    else:
        cv = 0.0
        fixed_holding = False

    # Profit target analysis: do winners cluster around a specific P&L?
    winner_pnls = [s["pnl_pct"] for s in winners]
    avg_win = statistics.mean(winner_pnls) if winner_pnls else 0
    avg_loss = statistics.mean([s["pnl_pct"] for s in losers]) if losers else 0

    return {
        "avg_holding_minutes": round(avg_holding, 1),
        "median_holding_minutes": round(median_holding, 1),
        "holding_cv": round(cv, 3),
        "style": style,
        "fixed_holding_period": fixed_holding,
        "avg_win_pct": round(avg_win, 3),
        "avg_loss_pct": round(avg_loss, 3),
        "exit_trigger": "time_based" if fixed_holding else "target_based",
    }


# =============================================================================
# IC 3: Sizing Pattern Analyzer
# =============================================================================


def _analyze_sizing_patterns(signals: list[dict]) -> dict[str, Any]:
    """Analyze position sizing patterns if size data is available."""
    sized_signals = [s for s in signals if s.get("size") is not None]

    if not sized_signals:
        # No size data — infer from holding period vs outcome
        winners = [s for s in signals if s["is_winner"]]
        losers = [s for s in signals if not s["is_winner"]]

        if winners and losers:
            avg_winner_hold = statistics.mean([s["holding_minutes"] for s in winners])
            avg_loser_hold = statistics.mean([s["holding_minutes"] for s in losers])
            patience_ratio = avg_winner_hold / (avg_loser_hold + 1e-10)
        else:
            patience_ratio = 1.0

        return {
            "has_size_data": False,
            "conviction_model": "unknown",
            "patience_ratio": round(patience_ratio, 2),
            "note": "No position size data available. Patience ratio measures if winners are held longer than losers.",
        }

    # Size data available — correlate size with outcome
    sizes = [float(s["size"]) for s in sized_signals]
    [s["pnl_pct"] for s in sized_signals]

    avg_size = statistics.mean(sizes)
    big_trades = [s for s in sized_signals if float(s["size"]) > avg_size]
    small_trades = [s for s in sized_signals if float(s["size"]) <= avg_size]

    big_win_rate = sum(1 for s in big_trades if s["is_winner"]) / max(1, len(big_trades))
    small_win_rate = sum(1 for s in small_trades if s["is_winner"]) / max(1, len(small_trades))

    if big_win_rate > small_win_rate + 0.15:
        conviction_model = "conviction_sizes_up"
    elif small_win_rate > big_win_rate + 0.15:
        conviction_model = "inverse_conviction"
    else:
        conviction_model = "flat_sizing"

    return {
        "has_size_data": True,
        "conviction_model": conviction_model,
        "avg_size": round(avg_size, 2),
        "big_trade_win_rate": round(big_win_rate * 100, 1),
        "small_trade_win_rate": round(small_win_rate * 100, 1),
    }


# =============================================================================
# IC 4: Regime Affinity Analyzer
# =============================================================================


def _analyze_regime_affinity(signals: list[dict]) -> dict[str, Any]:
    """
    Cross-reference trades with market regime at entry time.

    Since we don't have real-time regime data for historical signals,
    we use the entry time to infer regime from volatility of returns
    around the trade date.  This is approximate — a real implementation
    would look up the actual ADX/ATR at entry time.
    """
    if not signals:
        return {"error": "No signals"}

    # Simple heuristic: classify based on P&L magnitude as a proxy
    # (high-P&L trades in volatile markets, low-P&L in calm)
    pnl_magnitudes = [abs(s["pnl_pct"]) for s in signals]
    avg_mag = statistics.mean(pnl_magnitudes)

    # Group by direction and compute stats
    longs = [s for s in signals if s["direction"] == "long"]
    shorts = [s for s in signals if s["direction"] == "short"]

    long_wr = sum(1 for s in longs if s["is_winner"]) / max(1, len(longs)) * 100
    short_wr = sum(1 for s in shorts if s["is_winner"]) / max(1, len(shorts)) * 100

    # Infer best regime based on trading style
    exit_info = _analyze_exit_patterns(signals)
    style = exit_info.get("style", "swing")

    regime_affinity = {}
    if style in ("scalper", "intraday"):
        regime_affinity = {
            "ranging": 0.7,
            "trending_up": 0.5,
            "trending_down": 0.5,
        }
    elif style == "swing":
        if long_wr > short_wr + 10:
            regime_affinity = {"trending_up": 0.8, "ranging": 0.4, "trending_down": 0.2}
        elif short_wr > long_wr + 10:
            regime_affinity = {"trending_down": 0.8, "ranging": 0.4, "trending_up": 0.2}
        else:
            regime_affinity = {"trending_up": 0.6, "trending_down": 0.6, "ranging": 0.5}
    else:
        regime_affinity = {"trending_up": 0.7, "trending_down": 0.3, "ranging": 0.4}

    return {
        "regime_affinity": regime_affinity,
        "long_win_rate": round(long_wr, 1),
        "short_win_rate": round(short_wr, 1),
        "avg_pnl_magnitude": round(avg_mag, 3),
        "style_inferred": style,
    }


# =============================================================================
# Synthesizer: Combine IC outputs → DecodedStrategy
# =============================================================================


def _synthesize(
    signals: list[dict],
    entry_analysis: dict,
    exit_analysis: dict,
    sizing_analysis: dict,
    regime_analysis: dict,
    source_name: str,
) -> dict[str, Any]:
    """Combine IC outputs into a DecodedStrategy."""
    total = len(signals)
    winners = sum(1 for s in signals if s["is_winner"])
    win_rate = winners / total * 100 if total > 0 else 0
    avg_win = exit_analysis.get("avg_win_pct", 0)
    avg_loss = exit_analysis.get("avg_loss_pct", 0)

    # Build edge hypothesis
    timing = entry_analysis.get("timing_pattern", "unknown")
    direction = entry_analysis.get("direction_bias", "mixed")
    style = exit_analysis.get("style", "unknown")
    exit_trigger = exit_analysis.get("exit_trigger", "unknown")

    edge_hypothesis = (
        f"{timing} {direction}-biased {style} strategy. "
        f"Entries cluster at hour {entry_analysis.get('peak_entry_hour', '?')} "
        f"({entry_analysis.get('peak_entry_dow', '?')} preferred). "
        f"Exits are {exit_trigger} with avg holding {exit_analysis.get('avg_holding_minutes', 0):.0f} min. "
        f"Win rate: {win_rate:.1f}%."
    )

    # Confidence: based on sample size and consistency
    confidence = min(1.0, total / 100)  # scales 0-1 with 100 signals = 1.0
    if entry_analysis.get("hour_concentration", 0) > 0.5:
        confidence = min(1.0, confidence + 0.1)  # consistent timing boosts confidence

    # Build date range
    entry_times = [s["entry_time"] for s in signals]
    date_range = f"{min(entry_times).date()} to {max(entry_times).date()}"

    return {
        "source_trader": source_name,
        "sample_size": total,
        "date_range": date_range,
        "style": style,
        "entry_trigger": f"{timing} at hour {entry_analysis.get('peak_entry_hour', '?')}, {direction} bias",
        "exit_trigger": exit_trigger,
        "timing_pattern": timing,
        "avg_holding_minutes": exit_analysis.get("avg_holding_minutes", 0),
        "win_rate": round(win_rate, 2),
        "avg_win": round(avg_win, 3),
        "avg_loss": round(avg_loss, 3),
        "best_regime": max(
            regime_analysis.get("regime_affinity", {}),
            key=regime_analysis.get("regime_affinity", {}).get,
            default="unknown",
        ),
        "regime_affinity": regime_analysis.get("regime_affinity", {}),
        "edge_hypothesis": edge_hypothesis,
        "confidence": round(confidence, 2),
        "entry_analysis": entry_analysis,
        "exit_analysis": exit_analysis,
        "sizing_analysis": sizing_analysis,
        "regime_analysis": regime_analysis,
    }


# =============================================================================
# Public API
# =============================================================================


def decode_signals(
    raw_signals: list[dict[str, Any]],
    source_name: str = "unknown",
) -> dict[str, Any]:
    """
    Decode a list of trade signals into a strategy specification.

    Args:
        raw_signals: List of signal dicts with keys:
            symbol, direction, entry_time, entry_price, exit_time, exit_price,
            size? (optional), notes? (optional)
        source_name: Name of the signal source (e.g., "discord_trader_x")

    Returns:
        Dict with decoded_strategy and per-IC analysis details.
    """
    # Parse and validate signals
    parsed = []
    errors = 0
    for raw in raw_signals:
        signal = _parse_signal(raw)
        if signal is not None:
            parsed.append(signal)
        else:
            errors += 1

    if not parsed:
        return {
            "success": False,
            "error": f"No valid signals parsed. {errors} signals failed validation.",
        }

    low_confidence = len(parsed) < 20
    if low_confidence:
        logger.warning(
            f"[DECODER] Only {len(parsed)} signals — decode confidence will be low. "
            "Recommend 20+ signals for reliable pattern extraction."
        )

    # Run all 4 ICs
    entry_analysis = _analyze_entry_patterns(parsed)
    exit_analysis = _analyze_exit_patterns(parsed)
    sizing_analysis = _analyze_sizing_patterns(parsed)
    regime_analysis = _analyze_regime_affinity(parsed)

    # Synthesize
    decoded = _synthesize(
        parsed, entry_analysis, exit_analysis, sizing_analysis, regime_analysis, source_name
    )

    return {
        "success": True,
        "decoded_strategy": decoded,
        "signals_parsed": len(parsed),
        "signals_failed": errors,
        "low_confidence_warning": low_confidence,
    }
