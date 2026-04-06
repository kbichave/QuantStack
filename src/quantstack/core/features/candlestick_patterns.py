"""
Candlestick pattern recognition features.

Implements classic candlestick patterns with a pure-Python fallback when TA-Lib
is not available. TA-Lib (C library) is faster and more precise, but the numpy
fallback covers all 18 patterns adequately for ML feature generation.

TA-Lib returns:
- 0: no pattern
- +100: bullish pattern
- -100: bearish pattern

References:
    QuantAgent: https://github.com/Y-Research-SBU/QuantAgent
    TA-Lib: https://mrjbq7.github.io/ta-lib/func_groups/pattern_recognition.html
"""

import numpy as np
import pandas as pd
from loguru import logger

from quantstack.config.timeframes import Timeframe
from quantstack.core.features.base import FeatureBase

try:
    import talib

    _HAS_TALIB = True
except ImportError:
    _HAS_TALIB = False
    logger.info(
        "TA-Lib not installed — using pure-Python candlestick pattern fallback"
    )


# ---------------------------------------------------------------------------
# Pure-Python candlestick pattern implementations
# ---------------------------------------------------------------------------
# Each function takes (open, high, low, close) as numpy arrays and returns
# an integer array with values in {-100, 0, +100}.
# ---------------------------------------------------------------------------


def _body(o: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Signed body size (positive = bullish)."""
    return c - o


def _body_abs(o: np.ndarray, c: np.ndarray) -> np.ndarray:
    return np.abs(c - o)


def _upper_shadow(o: np.ndarray, h: np.ndarray, c: np.ndarray) -> np.ndarray:
    return h - np.maximum(o, c)


def _lower_shadow(o: np.ndarray, l: np.ndarray, c: np.ndarray) -> np.ndarray:
    return np.minimum(o, c) - l


def _range(h: np.ndarray, l: np.ndarray) -> np.ndarray:
    r = h - l
    r[r == 0] = 1e-10  # avoid division by zero
    return r


def _py_cdlhammer(o, h, l, c):
    ba = _body_abs(o, c)
    rng = _range(h, l)
    ls = _lower_shadow(o, l, c)
    us = _upper_shadow(o, h, c)
    signal = np.where(
        (ls >= 2 * ba) & (us <= ba * 0.3) & (ba > rng * 0.1),
        np.where(c >= o, 100, -100),
        0,
    )
    return signal.astype(np.int64)


def _py_cdlinvertedhammer(o, h, l, c):
    ba = _body_abs(o, c)
    rng = _range(h, l)
    us = _upper_shadow(o, h, c)
    ls = _lower_shadow(o, l, c)
    signal = np.where(
        (us >= 2 * ba) & (ls <= ba * 0.3) & (ba > rng * 0.1),
        np.where(c >= o, 100, -100),
        0,
    )
    return signal.astype(np.int64)


def _py_cdlhangingman(o, h, l, c):
    # Same shape as hammer but in context of uptrend; without trend context,
    # we identify the shape and mark as bearish.
    ba = _body_abs(o, c)
    rng = _range(h, l)
    ls = _lower_shadow(o, l, c)
    us = _upper_shadow(o, h, c)
    signal = np.where(
        (ls >= 2 * ba) & (us <= ba * 0.3) & (ba > rng * 0.1),
        -100,
        0,
    )
    return signal.astype(np.int64)


def _py_cdlshootingstar(o, h, l, c):
    ba = _body_abs(o, c)
    rng = _range(h, l)
    us = _upper_shadow(o, h, c)
    ls = _lower_shadow(o, l, c)
    signal = np.where(
        (us >= 2 * ba) & (ls <= ba * 0.3) & (ba > rng * 0.1),
        -100,
        0,
    )
    return signal.astype(np.int64)


def _py_cdldoji(o, h, l, c):
    rng = _range(h, l)
    ba = _body_abs(o, c)
    signal = np.where((ba <= rng * 0.05) & (rng > 0), 100, 0)
    return signal.astype(np.int64)


def _py_cdldragonflydoji(o, h, l, c):
    rng = _range(h, l)
    ba = _body_abs(o, c)
    ls = _lower_shadow(o, l, c)
    us = _upper_shadow(o, h, c)
    signal = np.where(
        (ba <= rng * 0.05) & (ls >= rng * 0.6) & (us <= rng * 0.1),
        100,
        0,
    )
    return signal.astype(np.int64)


def _py_cdlgravestonedoji(o, h, l, c):
    rng = _range(h, l)
    ba = _body_abs(o, c)
    us = _upper_shadow(o, h, c)
    ls = _lower_shadow(o, l, c)
    signal = np.where(
        (ba <= rng * 0.05) & (us >= rng * 0.6) & (ls <= rng * 0.1),
        -100,
        0,
    )
    return signal.astype(np.int64)


def _py_cdlengulfing(o, h, l, c):
    n = len(o)
    result = np.zeros(n, dtype=np.int64)
    for i in range(1, n):
        prev_body = c[i - 1] - o[i - 1]
        curr_body = c[i] - o[i]
        if prev_body < 0 and curr_body > 0:
            if o[i] <= c[i - 1] and c[i] >= o[i - 1]:
                result[i] = 100
        elif prev_body > 0 and curr_body < 0:
            if o[i] >= c[i - 1] and c[i] <= o[i - 1]:
                result[i] = -100
    return result


def _py_cdlharami(o, h, l, c):
    n = len(o)
    result = np.zeros(n, dtype=np.int64)
    for i in range(1, n):
        prev_body = c[i - 1] - o[i - 1]
        if abs(prev_body) < 1e-10:
            continue
        # Current body inside previous body
        curr_max = max(o[i], c[i])
        curr_min = min(o[i], c[i])
        prev_max = max(o[i - 1], c[i - 1])
        prev_min = min(o[i - 1], c[i - 1])
        if curr_max <= prev_max and curr_min >= prev_min:
            curr_body = c[i] - o[i]
            if abs(curr_body) < abs(prev_body) * 0.5:
                result[i] = 100 if prev_body < 0 else -100
    return result


def _py_cdlpiercing(o, h, l, c):
    n = len(o)
    result = np.zeros(n, dtype=np.int64)
    for i in range(1, n):
        if c[i - 1] < o[i - 1] and c[i] > o[i]:  # bearish then bullish
            midpoint = (o[i - 1] + c[i - 1]) / 2
            if o[i] < c[i - 1] and c[i] > midpoint and c[i] < o[i - 1]:
                result[i] = 100
    return result


def _py_cdldarkcloudcover(o, h, l, c):
    n = len(o)
    result = np.zeros(n, dtype=np.int64)
    for i in range(1, n):
        if c[i - 1] > o[i - 1] and c[i] < o[i]:  # bullish then bearish
            midpoint = (o[i - 1] + c[i - 1]) / 2
            if o[i] > c[i - 1] and c[i] < midpoint and c[i] > o[i - 1]:
                result[i] = -100
    return result


def _py_cdlmorningstar(o, h, l, c):
    n = len(o)
    result = np.zeros(n, dtype=np.int64)
    for i in range(2, n):
        body0 = c[i - 2] - o[i - 2]
        body1_abs = abs(c[i - 1] - o[i - 1])
        body2 = c[i] - o[i]
        range0 = h[i - 2] - l[i - 2] if h[i - 2] != l[i - 2] else 1e-10
        if (
            body0 < 0
            and body1_abs < abs(body0) * 0.3
            and body2 > 0
            and c[i] > (o[i - 2] + c[i - 2]) / 2
        ):
            result[i] = 100
    return result


def _py_cdleveningstar(o, h, l, c):
    n = len(o)
    result = np.zeros(n, dtype=np.int64)
    for i in range(2, n):
        body0 = c[i - 2] - o[i - 2]
        body1_abs = abs(c[i - 1] - o[i - 1])
        body2 = c[i] - o[i]
        if (
            body0 > 0
            and body1_abs < abs(body0) * 0.3
            and body2 < 0
            and c[i] < (o[i - 2] + c[i - 2]) / 2
        ):
            result[i] = -100
    return result


def _py_cdl3whitesoldiers(o, h, l, c):
    n = len(o)
    result = np.zeros(n, dtype=np.int64)
    for i in range(2, n):
        if (
            c[i - 2] > o[i - 2]
            and c[i - 1] > o[i - 1]
            and c[i] > o[i]
            and c[i - 1] > c[i - 2]
            and c[i] > c[i - 1]
            and o[i - 1] > o[i - 2]
            and o[i] > o[i - 1]
        ):
            result[i] = 100
    return result


def _py_cdl3blackcrows(o, h, l, c):
    n = len(o)
    result = np.zeros(n, dtype=np.int64)
    for i in range(2, n):
        if (
            c[i - 2] < o[i - 2]
            and c[i - 1] < o[i - 1]
            and c[i] < o[i]
            and c[i - 1] < c[i - 2]
            and c[i] < c[i - 1]
            and o[i - 1] < o[i - 2]
            and o[i] < o[i - 1]
        ):
            result[i] = -100
    return result


def _py_cdlabandonedbaby(o, h, l, c):
    n = len(o)
    result = np.zeros(n, dtype=np.int64)
    for i in range(2, n):
        body1_abs = abs(c[i - 1] - o[i - 1])
        range1 = h[i - 1] - l[i - 1] if h[i - 1] != l[i - 1] else 1e-10
        is_doji = body1_abs <= range1 * 0.1
        # Bullish abandoned baby
        if (
            c[i - 2] < o[i - 2]
            and is_doji
            and h[i - 1] < l[i - 2]
            and l[i - 1] < l[i]
            and c[i] > o[i]
        ):
            result[i] = 100
        # Bearish abandoned baby
        elif (
            c[i - 2] > o[i - 2]
            and is_doji
            and l[i - 1] > h[i - 2]
            and h[i - 1] > h[i]
            and c[i] < o[i]
        ):
            result[i] = -100
    return result


def _py_cdlkicking(o, h, l, c):
    n = len(o)
    result = np.zeros(n, dtype=np.int64)
    for i in range(1, n):
        # Bullish kicking: bearish marubozu followed by bullish marubozu with gap up
        prev_bearish_marubozu = (
            o[i - 1] > c[i - 1]
            and abs(h[i - 1] - o[i - 1]) < (o[i - 1] - c[i - 1]) * 0.05
            and abs(c[i - 1] - l[i - 1]) < (o[i - 1] - c[i - 1]) * 0.05
        )
        curr_bullish_marubozu = (
            c[i] > o[i]
            and abs(o[i] - l[i]) < (c[i] - o[i]) * 0.05
            and abs(h[i] - c[i]) < (c[i] - o[i]) * 0.05
        )
        if prev_bearish_marubozu and curr_bullish_marubozu and o[i] > o[i - 1]:
            result[i] = 100
        # Bearish kicking
        prev_bullish_marubozu = (
            c[i - 1] > o[i - 1]
            and abs(o[i - 1] - l[i - 1]) < (c[i - 1] - o[i - 1]) * 0.05
            and abs(h[i - 1] - c[i - 1]) < (c[i - 1] - o[i - 1]) * 0.05
        )
        curr_bearish_marubozu = (
            o[i] > c[i]
            and abs(h[i] - o[i]) < (o[i] - c[i]) * 0.05
            and abs(c[i] - l[i]) < (o[i] - c[i]) * 0.05
        )
        if prev_bullish_marubozu and curr_bearish_marubozu and o[i] < o[i - 1]:
            result[i] = -100
    return result


def _py_cdl3linestrike(o, h, l, c):
    n = len(o)
    result = np.zeros(n, dtype=np.int64)
    for i in range(3, n):
        # Bullish: 3 bearish candles then one bullish that engulfs all 3
        if (
            c[i - 3] < o[i - 3]
            and c[i - 2] < o[i - 2]
            and c[i - 1] < o[i - 1]
            and c[i - 2] < c[i - 3]
            and c[i - 1] < c[i - 2]
            and c[i] > o[i]
            and o[i] <= c[i - 1]
            and c[i] >= o[i - 3]
        ):
            result[i] = 100
        # Bearish: 3 bullish candles then one bearish that engulfs all 3
        elif (
            c[i - 3] > o[i - 3]
            and c[i - 2] > o[i - 2]
            and c[i - 1] > o[i - 1]
            and c[i - 2] > c[i - 3]
            and c[i - 1] > c[i - 2]
            and c[i] < o[i]
            and o[i] >= c[i - 1]
            and c[i] <= o[i - 3]
        ):
            result[i] = -100
    return result


def _py_cdlrisefall3methods(o, h, l, c):
    n = len(o)
    result = np.zeros(n, dtype=np.int64)
    for i in range(4, n):
        first_body = c[i - 4] - o[i - 4]
        last_body = c[i] - o[i]
        # Rising three methods: long bullish, 3 small bearish inside, long bullish
        if first_body > 0 and last_body > 0 and c[i] > c[i - 4]:
            inner_ok = all(
                c[i - j] < o[i - j]  # bearish
                and max(o[i - j], c[i - j]) < h[i - 4]
                and min(o[i - j], c[i - j]) > l[i - 4]
                for j in [3, 2, 1]
            )
            if inner_ok:
                result[i] = 100
        # Falling three methods
        elif first_body < 0 and last_body < 0 and c[i] < c[i - 4]:
            inner_ok = all(
                c[i - j] > o[i - j]  # bullish
                and max(o[i - j], c[i - j]) < h[i - 4]
                and min(o[i - j], c[i - j]) > l[i - 4]
                for j in [3, 2, 1]
            )
            if inner_ok:
                result[i] = -100
    return result


# Map of pattern name → (talib_func, python_fallback)
_PATTERN_FUNCTIONS = {
    "CDLHAMMER": ("CDLHAMMER", _py_cdlhammer),
    "CDLINVERTEDHAMMER": ("CDLINVERTEDHAMMER", _py_cdlinvertedhammer),
    "CDLHANGINGMAN": ("CDLHANGINGMAN", _py_cdlhangingman),
    "CDLSHOOTINGSTAR": ("CDLSHOOTINGSTAR", _py_cdlshootingstar),
    "CDLDOJI": ("CDLDOJI", _py_cdldoji),
    "CDLDRAGONFLYDOJI": ("CDLDRAGONFLYDOJI", _py_cdldragonflydoji),
    "CDLGRAVESTONEDOJI": ("CDLGRAVESTONEDOJI", _py_cdlgravestonedoji),
    "CDLENGULFING": ("CDLENGULFING", _py_cdlengulfing),
    "CDLHARAMI": ("CDLHARAMI", _py_cdlharami),
    "CDLPIERCING": ("CDLPIERCING", _py_cdlpiercing),
    "CDLDARKCLOUDCOVER": ("CDLDARKCLOUDCOVER", _py_cdldarkcloudcover),
    "CDLMORNINGSTAR": ("CDLMORNINGSTAR", _py_cdlmorningstar),
    "CDLEVENINGSTAR": ("CDLEVENINGSTAR", _py_cdleveningstar),
    "CDL3WHITESOLDIERS": ("CDL3WHITESOLDIERS", _py_cdl3whitesoldiers),
    "CDL3BLACKCROWS": ("CDL3BLACKCROWS", _py_cdl3blackcrows),
    "CDLABANDONEDBABY": ("CDLABANDONEDBABY", _py_cdlabandonedbaby),
    "CDLKICKING": ("CDLKICKING", _py_cdlkicking),
    "CDL3LINESTRIKE": ("CDL3LINESTRIKE", _py_cdl3linestrike),
    "CDLRISEFALL3METHODS": ("CDLRISEFALL3METHODS", _py_cdlrisefall3methods),
}


def _call_pattern(name: str, o, h, l, c) -> np.ndarray:
    """Call a pattern function, preferring TA-Lib when available."""
    talib_name, py_fallback = _PATTERN_FUNCTIONS[name]
    if _HAS_TALIB:
        return getattr(talib, talib_name)(o, h, l, c)
    return py_fallback(o, h, l, c)


class CandlestickPatternFeatures(FeatureBase):
    """
    Candlestick pattern recognition features.

    Uses TA-Lib when available, falls back to pure-Python implementations.

    Focuses on reversal and continuation patterns that align with mean reversion:
    - Reversal patterns: Hammer, Inverted Hammer, Doji, Engulfing, etc.
    - Continuation patterns: Three Line Strike, Rising/Falling Three Methods
    - Key patterns from QuantAgent: Double Bottom, H&S, Triangles, Wedges

    Features are normalized to [-1, 1] range for ML compatibility.
    """

    def __init__(self, timeframe: Timeframe):
        super().__init__(timeframe)
        if timeframe in [Timeframe.W1, Timeframe.D1]:
            self.include_continuation = False
        else:
            self.include_continuation = True

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()

        if len(result) < 10:
            logger.warning(
                f"Insufficient data for pattern recognition: {len(result)} bars"
            )
            for col in self.get_feature_names():
                result[col] = 0
            return result

        o = result["open"].values.astype(np.float64)
        h = result["high"].values.astype(np.float64)
        l = result["low"].values.astype(np.float64)
        c = result["close"].values.astype(np.float64)

        # === REVERSAL PATTERNS ===
        result["cdl_hammer"] = self._normalize_pattern(
            _call_pattern("CDLHAMMER", o, h, l, c)
        )
        result["cdl_inverted_hammer"] = self._normalize_pattern(
            _call_pattern("CDLINVERTEDHAMMER", o, h, l, c)
        )
        result["cdl_hanging_man"] = self._normalize_pattern(
            _call_pattern("CDLHANGINGMAN", o, h, l, c)
        )
        result["cdl_shooting_star"] = self._normalize_pattern(
            _call_pattern("CDLSHOOTINGSTAR", o, h, l, c)
        )
        result["cdl_doji"] = self._normalize_pattern(
            _call_pattern("CDLDOJI", o, h, l, c)
        )
        result["cdl_dragonfly_doji"] = self._normalize_pattern(
            _call_pattern("CDLDRAGONFLYDOJI", o, h, l, c)
        )
        result["cdl_gravestone_doji"] = self._normalize_pattern(
            _call_pattern("CDLGRAVESTONEDOJI", o, h, l, c)
        )
        result["cdl_engulfing"] = self._normalize_pattern(
            _call_pattern("CDLENGULFING", o, h, l, c)
        )
        result["cdl_harami"] = self._normalize_pattern(
            _call_pattern("CDLHARAMI", o, h, l, c)
        )
        result["cdl_piercing"] = self._normalize_pattern(
            _call_pattern("CDLPIERCING", o, h, l, c)
        )
        result["cdl_dark_cloud"] = self._normalize_pattern(
            _call_pattern("CDLDARKCLOUDCOVER", o, h, l, c)
        )
        result["cdl_morning_star"] = self._normalize_pattern(
            _call_pattern("CDLMORNINGSTAR", o, h, l, c)
        )
        result["cdl_evening_star"] = self._normalize_pattern(
            _call_pattern("CDLEVENINGSTAR", o, h, l, c)
        )
        result["cdl_three_white_soldiers"] = self._normalize_pattern(
            _call_pattern("CDL3WHITESOLDIERS", o, h, l, c)
        )
        result["cdl_three_black_crows"] = self._normalize_pattern(
            _call_pattern("CDL3BLACKCROWS", o, h, l, c)
        )
        result["cdl_abandoned_baby"] = self._normalize_pattern(
            _call_pattern("CDLABANDONEDBABY", o, h, l, c)
        )
        result["cdl_kicking"] = self._normalize_pattern(
            _call_pattern("CDLKICKING", o, h, l, c)
        )

        # === CONTINUATION PATTERNS (if enabled) ===
        if self.include_continuation:
            result["cdl_three_line_strike"] = self._normalize_pattern(
                _call_pattern("CDL3LINESTRIKE", o, h, l, c)
            )
            result["cdl_rising_three_methods"] = self._normalize_pattern(
                _call_pattern("CDLRISEFALL3METHODS", o, h, l, c)
            )

        # === AGGREGATE FEATURES ===
        bullish_cols = [col for col in result.columns if col.startswith("cdl_")]
        result["cdl_bullish_count"] = (result[bullish_cols] > 0).sum(axis=1)
        result["cdl_bearish_count"] = (result[bullish_cols] < 0).sum(axis=1)
        result["cdl_net_signal"] = result[bullish_cols].sum(axis=1) / len(bullish_cols)
        result["cdl_max_bullish"] = result[bullish_cols].max(axis=1)
        result["cdl_max_bearish"] = result[bullish_cols].min(axis=1)

        # === CUSTOM PATTERN LOGIC (QuantAgent-inspired) ===
        result["cdl_double_bottom"] = self._detect_double_bottom(result)
        result["cdl_v_reversal"] = self._detect_v_reversal(result)

        return result

    def _normalize_pattern(self, pattern: np.ndarray) -> np.ndarray:
        """Normalize TA-Lib pattern output from [-100, 100] to [-1, 1]."""
        return pattern / 100.0

    def _detect_double_bottom(self, df: pd.DataFrame, lookback: int = 20) -> pd.Series:
        """
        Detect double bottom pattern (simplified version).

        A double bottom has:
        - Two local lows within lookback period
        - Lows are within 2% of each other
        - Middle high is at least 3% above lows
        - Current price breaking above middle high
        """
        result = pd.Series(0, index=df.index)
        if len(df) < lookback:
            return result

        close = df["close"].values
        low = df["low"].values

        for i in range(lookback, len(df)):
            window_low = low[i - lookback : i]
            window_close = close[i - lookback : i]
            sorted_indices = np.argsort(window_low)
            lowest_idx = sorted_indices[0]
            second_lowest_idx = sorted_indices[1]

            if (
                abs(window_low[lowest_idx] - window_low[second_lowest_idx])
                / window_low[lowest_idx]
                > 0.02
            ):
                continue

            if lowest_idx < second_lowest_idx:
                middle_high = window_close[lowest_idx:second_lowest_idx].max()
            else:
                middle_high = window_close[second_lowest_idx:lowest_idx].max()

            avg_low = (window_low[lowest_idx] + window_low[second_lowest_idx]) / 2
            if middle_high < avg_low * 1.03:
                continue

            if close[i] > middle_high:
                result.iloc[i] = 1

        return result

    def _detect_v_reversal(self, df: pd.DataFrame, lookback: int = 10) -> pd.Series:
        """
        Detect V-shaped reversal (sharp decline followed by sharp recovery).

        A V-reversal has:
        - Sharp decline (>5% over lookback/2 bars)
        - Sharp recovery (>5% over next lookback/2 bars)
        """
        result = pd.Series(0, index=df.index)
        if len(df) < lookback:
            return result

        close = df["close"].values

        for i in range(lookback, len(df)):
            window = close[i - lookback : i + 1]
            bottom_idx = window.argmin()

            if bottom_idx >= lookback // 2:
                decline_start = close[i - lookback]
                decline_end = window[bottom_idx]
                decline_pct = (decline_end - decline_start) / decline_start

                if decline_pct < -0.05:
                    if bottom_idx < len(window) - lookback // 2:
                        recovery_start = window[bottom_idx]
                        recovery_end = window[-1]
                        recovery_pct = (recovery_end - recovery_start) / recovery_start
                        if recovery_pct > 0.05:
                            result.iloc[i] = 1

            top_idx = window.argmax()
            if top_idx >= lookback // 2:
                rise_start = close[i - lookback]
                rise_end = window[top_idx]
                rise_pct = (rise_end - rise_start) / rise_start

                if rise_pct > 0.05:
                    if top_idx < len(window) - lookback // 2:
                        fall_start = window[top_idx]
                        fall_end = window[-1]
                        fall_pct = (fall_end - fall_start) / fall_start
                        if fall_pct < -0.05:
                            result.iloc[i] = -1

        return result

    def get_feature_names(self) -> list[str]:
        """Return list of candlestick pattern feature names."""
        features = [
            "cdl_hammer",
            "cdl_inverted_hammer",
            "cdl_hanging_man",
            "cdl_shooting_star",
            "cdl_doji",
            "cdl_dragonfly_doji",
            "cdl_gravestone_doji",
            "cdl_engulfing",
            "cdl_harami",
            "cdl_piercing",
            "cdl_dark_cloud",
            "cdl_morning_star",
            "cdl_evening_star",
            "cdl_three_white_soldiers",
            "cdl_three_black_crows",
            "cdl_abandoned_baby",
            "cdl_kicking",
            "cdl_bullish_count",
            "cdl_bearish_count",
            "cdl_net_signal",
            "cdl_max_bullish",
            "cdl_max_bearish",
            "cdl_double_bottom",
            "cdl_v_reversal",
        ]

        if self.include_continuation:
            features.extend(
                [
                    "cdl_three_line_strike",
                    "cdl_rising_three_methods",
                ]
            )

        return features
