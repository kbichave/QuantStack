# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Data validation layer — runs before any data reaches agents.

Every bar must pass validation. Invalid data is logged and discarded,
never silently passed to an agent as if it were good.

Rules:
  1. OHLC consistency:  high >= close >= low, open in [low, high]
  2. Volume plausibility: volume > 0
  3. Price plausibility: no single-bar move > MAX_DAILY_MOVE_PCT
  4. Data freshness: stale if most recent bar is > STALE_THRESHOLD old
  5. Completeness: flag if expected symbols missing from a batch

Usage:
    validator = DataValidator()

    valid_bars = validator.validate_bars(raw_bars)
    freshness = validator.check_freshness(bars, symbol="SPY")
    issues = validator.validate_batch(symbol_bars_dict)
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from loguru import logger

from quantstack.data.provider import Bar

# =============================================================================
# VALIDATOR
# =============================================================================


class DataValidator:
    """
    Validates market data before it is used by any agent or model.

    If data fails validation, it is logged and dropped — never silently
    passed downstream as if it were good data.
    """

    # A single bar move this large is almost certainly bad data
    MAX_DAILY_MOVE_PCT = 0.50  # 50% — anything larger is flagged

    # Data is "stale" if the most recent bar is this old
    STALE_THRESHOLD_HOURS = 8  # > 8h old during trading hours = stale

    def validate_bar(self, bar: Bar) -> tuple[bool, str | None]:
        """
        Validate a single bar.

        Returns:
            (is_valid, reason_if_invalid)
        """
        # OHLC consistency
        if bar.high < bar.low:
            return False, f"high ({bar.high}) < low ({bar.low})"

        if bar.close < bar.low or bar.close > bar.high:
            return False, (
                f"close ({bar.close}) outside [low={bar.low}, high={bar.high}]"
            )

        if bar.open < bar.low or bar.open > bar.high:
            return False, (
                f"open ({bar.open}) outside [low={bar.low}, high={bar.high}]"
            )

        # Non-negative prices
        if bar.open <= 0 or bar.close <= 0 or bar.high <= 0 or bar.low <= 0:
            return False, "non-positive price value"

        # Volume
        if bar.volume < 0:
            return False, f"negative volume ({bar.volume})"

        return True, None

    def validate_bars(self, bars: list[Bar]) -> list[Bar]:
        """
        Validate a list of bars. Returns only valid bars.

        Logs each invalid bar with the reason.
        """
        if not bars:
            return []

        valid = []
        for bar in bars:
            ok, reason = self.validate_bar(bar)
            if ok:
                valid.append(bar)
            else:
                logger.warning(
                    f"[VALIDATOR] Dropped {bar.symbol} bar @ {bar.timestamp}: {reason}"
                )

        dropped = len(bars) - len(valid)
        if dropped > 0:
            pct = dropped / len(bars) * 100
            logger.warning(
                f"[VALIDATOR] {bar.symbol}: dropped {dropped}/{len(bars)} "
                f"bars ({pct:.1f}%) due to validation failures"
            )

        return valid

    def validate_bars_with_plausibility(
        self, bars: list[Bar], max_move_pct: float | None = None
    ) -> list[Bar]:
        """
        Validate bars including single-bar move plausibility check.

        Useful for catching feed errors where a single bar has a 500% spike.
        """
        max_move = max_move_pct or self.MAX_DAILY_MOVE_PCT
        valid = self.validate_bars(bars)
        filtered = []

        for i, bar in enumerate(valid):
            if i == 0:
                filtered.append(bar)
                continue

            prev_close = valid[i - 1].close
            if prev_close > 0:
                move = abs(bar.close - prev_close) / prev_close
                if move > max_move:
                    logger.warning(
                        f"[VALIDATOR] Plausibility: {bar.symbol} @ {bar.timestamp} "
                        f"moved {move:.1%} from prev close {prev_close:.2f} "
                        f"to {bar.close:.2f} (threshold {max_move:.0%}) — dropped"
                    )
                    continue
            filtered.append(bar)

        return filtered

    def check_freshness(
        self,
        bars: list[Bar],
        symbol: str,
        stale_hours: float | None = None,
    ) -> dict[str, Any]:
        """
        Check if the most recent bar is stale.

        Returns a dict with keys:
            fresh (bool), latest_bar_time, hours_old, message
        """
        threshold = stale_hours or self.STALE_THRESHOLD_HOURS

        if not bars:
            return {
                "fresh": False,
                "latest_bar_time": None,
                "hours_old": None,
                "message": f"{symbol}: no bars returned",
            }

        latest = max(bars, key=lambda b: b.timestamp)
        now = datetime.utcnow()

        # Make both naive for comparison
        latest_ts = latest.timestamp
        if latest_ts.tzinfo is not None:
            latest_ts = latest_ts.replace(tzinfo=None)

        hours_old = (now - latest_ts).total_seconds() / 3600
        is_fresh = hours_old <= threshold

        result = {
            "fresh": is_fresh,
            "latest_bar_time": latest.timestamp,
            "hours_old": round(hours_old, 1),
            "message": (
                f"{symbol}: latest bar {hours_old:.1f}h old ({'FRESH' if is_fresh else 'STALE'})"
            ),
        }

        if not is_fresh:
            logger.warning(f"[VALIDATOR] STALE DATA: {result['message']}")

        return result

    def validate_batch(
        self,
        symbol_bars: dict[str, list[Bar]],
        expected_symbols: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Validate a batch of symbol→bars mappings.

        Returns a summary with:
          - valid_symbols: symbols with at least 1 valid bar
          - invalid_symbols: symbols with 0 valid bars
          - missing_symbols: expected but not present
          - stale_symbols: symbols with stale data
          - bar_counts: {symbol: (total, valid)}
        """
        valid_symbols = []
        invalid_symbols = []
        stale_symbols = []
        bar_counts: dict[str, tuple[int, int]] = {}

        for symbol, bars in symbol_bars.items():
            valid = self.validate_bars(bars)
            bar_counts[symbol] = (len(bars), len(valid))

            if len(valid) == 0:
                invalid_symbols.append(symbol)
            else:
                valid_symbols.append(symbol)
                freshness = self.check_freshness(valid, symbol)
                if not freshness["fresh"]:
                    stale_symbols.append(symbol)

        missing_symbols: list[str] = []
        if expected_symbols:
            present = set(symbol_bars.keys())
            missing_symbols = [s for s in expected_symbols if s not in present]
            if missing_symbols:
                logger.warning(
                    f"[VALIDATOR] Missing symbols in batch: {missing_symbols}"
                )

        return {
            "valid_symbols": valid_symbols,
            "invalid_symbols": invalid_symbols,
            "missing_symbols": missing_symbols,
            "stale_symbols": stale_symbols,
            "bar_counts": bar_counts,
            "total": len(symbol_bars),
            "valid": len(valid_symbols),
        }
