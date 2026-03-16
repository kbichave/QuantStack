"""
IncrementalFeatureEngine — O(1) per-bar feature updates for live streaming.

Why not just call FeatureFactory on every bar?
----------------------------------------------
The full ``MultiTimeframeFeatureFactory`` batch-computes 50+ indicators over the
entire history window on every call.  At 1-minute granularity that's acceptable
for end-of-day backtests, but at tick/S5/M1 frequencies with 100+ symbols it
becomes a bottleneck.  Wilder-smoothed indicators (EMA, RSI, ATR) admit an exact
recursive update that requires only the *previous state* and the *new bar* —
O(1) per indicator, per bar.

This engine maintains per-symbol state and applies those recursive updates.
Batch initialization (warmup) is done once per symbol using the window from
``LiveBarStore``.  After warmup every on_bar() call is sub-millisecond.

Computed features
-----------------
All features are intraday-relevant; indicator periods come from ``TimeframeParams``:

  ``ema_fast``          Exponential Moving Average (fast period)
  ``ema_slow``          Exponential Moving Average (slow period)
  ``ema_cross``         ema_fast − ema_slow  (positive = bullish)
  ``rsi``               RSI [0–100] via Wilder's smoothed method
  ``atr``               Average True Range via Wilder's SMMA
  ``atr_pct``           atr / close  (normalized volatility)
  ``price_to_ema``      (close − ema_fast) / atr  (signed distance in ATR units)
  ``bb_upper``          Bollinger upper  (SMA ± bb_std σ over bb_period bars)
  ``bb_lower``          Bollinger lower
  ``bb_pct_b``          (close − bb_lower) / (bb_upper − bb_lower); 0–1 within bands
  ``volume_ratio``      bar_volume / EMA(volume, vol_ma_period)
  ``roc``               Rate-of-Change over roc_period bars
  ``vwap_deviation``    (close − bar_vwap) / atr; None if vwap not in bar

Usage
-----
    store   = LiveBarStore(data_store)
    engine  = IncrementalFeatureEngine(store, timeframe=Timeframe.M1)

    # Register a consumer
    async def on_features(features: IncrementalFeatures) -> None:
        signal = features.ema_cross > 0 and features.rsi < 70

    engine.add_callback(on_features)

    # Wire to BarPublisher
    feature_queue = publisher.subscribe("feature_engine")
    asyncio.create_task(engine.run_from_queue(feature_queue))

    # Or wire directly as a streaming adapter callback
    adapter.add_callback(engine.on_bar)
"""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
from loguru import logger

from quantcore.config.timeframes import TIMEFRAME_PARAMS, Timeframe, TimeframeParams
from quantcore.data.streaming.base import BarEvent
from quantcore.data.streaming.live_store import LiveBarStore

# ---------------------------------------------------------------------------
# Output data model
# ---------------------------------------------------------------------------


@dataclass
class IncrementalFeatures:
    """Feature vector produced after each bar update."""

    symbol:            str
    timestamp:         datetime
    timeframe:         Timeframe
    close:             float

    # Trend
    ema_fast:          float
    ema_slow:          float
    ema_cross:         float   # ema_fast − ema_slow; >0 = bullish

    # Momentum
    rsi:               float   # [0, 100]
    roc:               float   # rate-of-change over roc_period bars

    # Volatility
    atr:               float
    atr_pct:           float   # atr / close
    bb_upper:          float
    bb_lower:          float
    bb_pct_b:          float   # (close − lower) / (upper − lower)

    # Volume
    volume_ratio:      float   # bar_vol / ema(vol)

    # Distance metrics
    price_to_ema:      float   # (close − ema_fast) / atr; signed distance in ATR units

    # VWAP (optional — only when bar carries vwap)
    vwap_deviation:    float | None

    # Warmup flag: False until the engine has seen enough bars to initialize
    is_warm:           bool


FeaturesCallback = Callable[[IncrementalFeatures], Awaitable[None]]


# ---------------------------------------------------------------------------
# Per-symbol state (private)
# ---------------------------------------------------------------------------


@dataclass
class _SymbolState:
    """Recursive state for one symbol.  All fields are scalars after warmup."""

    # EMA state (k = 2 / (period + 1))
    ema_fast:      float
    ema_slow:      float
    ema_vol:       float   # EMA of volume for volume_ratio

    # RSI state — Wilder's SMMA (alpha = 1 / rsi_period)
    avg_gain:      float
    avg_loss:      float
    prev_close:    float

    # ATR state — Wilder's SMMA (alpha = 1 / atr_period)
    atr:           float
    prev_high:     float
    prev_low:      float

    # Rolling windows for Bollinger Bands and ROC (tiny fixed-size deques)
    close_window:  deque   # last bb_period closes
    vol_window:    deque   # last vol_ma_period volumes

    is_warm:       bool = False


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class IncrementalFeatureEngine:
    """Maintains per-symbol incremental state and emits IncrementalFeatures.

    Args:
        live_store: ``LiveBarStore`` used to seed state on first bar per symbol.
        timeframe:  Timeframe whose ``TimeframeParams`` govern indicator periods.
        warmup_multiplier: Factor applied to the longest indicator period to
                           determine the minimum warmup window.  Default 3 gives
                           a statistically stable seed (e.g. 3 × 21 = 63 bars).
    """

    def __init__(
        self,
        live_store: LiveBarStore,
        timeframe: Timeframe = Timeframe.M1,
        warmup_multiplier: int = 3,
    ) -> None:
        self._store     = live_store
        self._tf        = timeframe
        self._params: TimeframeParams = TIMEFRAME_PARAMS[timeframe]
        self._warmup_bars = (
            max(
                self._params.ema_slow,
                self._params.rsi_period,
                self._params.atr_period,
                self._params.bb_period,
                self._params.volume_ma_period,
                self._params.roc_period,
            )
            * warmup_multiplier
        )

        # Per-symbol recursive state
        self._states: dict[str, _SymbolState] = {}

        # Feature callbacks (same pattern as BarPublisher)
        self._callbacks: list[FeaturesCallback] = []

    # ── Callback registration ─────────────────────────────────────────────────

    def add_callback(self, callback: FeaturesCallback) -> None:
        """Register an async callback to receive IncrementalFeatures."""
        self._callbacks.append(callback)

    def remove_callback(self, callback: FeaturesCallback) -> None:
        self._callbacks = [c for c in self._callbacks if c is not callback]

    # ── BarCallback interface ─────────────────────────────────────────────────

    async def on_bar(self, bar: BarEvent) -> None:
        """Receive one bar, update state, compute features, call callbacks.

        Compatible with ``StreamingAdapter.add_callback()`` and
        ``BarPublisher.subscribe()`` (via ``run_from_queue``).
        """
        features = self._update(bar)
        if features and self._callbacks:
            await asyncio.gather(
                *(cb(features) for cb in self._callbacks),
                return_exceptions=True,
            )

    async def run_from_queue(
        self, queue: asyncio.Queue[BarEvent | None]
    ) -> None:
        """Consume bars from a ``BarPublisher`` queue until None (shutdown).

        Typical wiring::

            feature_queue = publisher.subscribe("incremental_features")
            asyncio.create_task(engine.run_from_queue(feature_queue))
        """
        while True:
            bar = await queue.get()
            if bar is None:
                return
            await self.on_bar(bar)

    # ── Read API ──────────────────────────────────────────────────────────────

    def get_features(self, symbol: str) -> IncrementalFeatures | None:
        """Return the last computed feature vector for ``symbol``, or None."""
        return self._last.get(symbol)

    def is_warm(self, symbol: str) -> bool:
        """True once the engine has a fully initialized state for ``symbol``."""
        state = self._states.get(symbol)
        return bool(state and state.is_warm)

    # ── Internal update ───────────────────────────────────────────────────────

    def __init_last(self) -> None:
        """Lazy init for _last dict (called only when needed)."""
        if not hasattr(self, "_last"):
            self._last: dict[str, IncrementalFeatures] = {}

    def _update(self, bar: BarEvent) -> IncrementalFeatures | None:
        """Core update: initialize state on first bar, then recurse."""
        self.__init_last()
        sym = bar.symbol

        if sym not in self._states or not self._states[sym].is_warm:
            state = self._initialize_state(sym, bar)
            if state is None:
                return None  # Still waiting for warmup bars
        else:
            state = self._states[sym]
            self._apply_bar(state, bar)

        features = self._build_features(bar, state)
        self._last[sym] = features
        return features

    def _initialize_state(
        self, symbol: str, bar: BarEvent
    ) -> _SymbolState | None:
        """Batch-initialize recursive state from the LiveBarStore window.

        Returns the new state if the window is deep enough, else None.
        """
        p = self._params
        window = self._store.get_window(symbol, self._warmup_bars)

        if len(window) < max(p.ema_slow, p.rsi_period, p.atr_period) + 1:
            logger.debug(
                f"[IncrementalFeatureEngine] Warming up '{symbol}': "
                f"{len(window)}/{self._warmup_bars} bars"
            )
            return None

        closes  = np.array([b.close  for b in window], dtype=float)
        highs   = np.array([b.high   for b in window], dtype=float)
        lows    = np.array([b.low    for b in window], dtype=float)
        volumes = np.array([b.volume for b in window], dtype=float)

        # ── EMA batch seed (pandas ewm is exact)
        ema_fast_s = _batch_ema(closes, p.ema_fast)
        ema_slow_s = _batch_ema(closes, p.ema_slow)
        ema_vol_s  = _batch_ema(volumes, p.volume_ma_period)

        # ── RSI seed (Wilder's SMMA from batch)
        avg_gain, avg_loss = _batch_rsi_seed(closes, p.rsi_period)

        # ── ATR seed (Wilder's SMMA from batch)
        atr_seed = _batch_atr_seed(highs, lows, closes, p.atr_period)

        # ── BB window: last bb_period closes
        close_window: deque = deque(
            closes[-p.bb_period:].tolist(), maxlen=p.bb_period
        )

        # ── Volume window: last vol_ma_period volumes
        vol_window: deque = deque(
            volumes[-p.volume_ma_period:].tolist(), maxlen=p.volume_ma_period
        )

        state = _SymbolState(
            ema_fast     = ema_fast_s,
            ema_slow     = ema_slow_s,
            ema_vol      = ema_vol_s,
            avg_gain     = avg_gain,
            avg_loss     = avg_loss,
            prev_close   = closes[-1],
            atr          = atr_seed,
            prev_high    = highs[-1],
            prev_low     = lows[-1],
            close_window = close_window,
            vol_window   = vol_window,
            is_warm      = True,
        )
        self._states[symbol] = state
        logger.info(
            f"[IncrementalFeatureEngine] '{symbol}' warmed up "
            f"({len(window)} bars, tf={self._tf.value})"
        )
        return state

    def _apply_bar(self, state: _SymbolState, bar: BarEvent) -> None:
        """Apply one new bar to the recursive state in O(1)."""
        p = self._params
        close  = bar.close
        high   = bar.high
        low    = bar.low
        volume = bar.volume

        # ── EMA update: k = 2/(period+1)
        k_fast = 2.0 / (p.ema_fast + 1)
        k_slow = 2.0 / (p.ema_slow + 1)
        k_vol  = 2.0 / (p.volume_ma_period + 1)
        state.ema_fast = close  * k_fast + state.ema_fast * (1.0 - k_fast)
        state.ema_slow = close  * k_slow + state.ema_slow * (1.0 - k_slow)
        state.ema_vol  = volume * k_vol  + state.ema_vol  * (1.0 - k_vol)

        # ── RSI update (Wilder's SMMA: alpha = 1/period)
        alpha_rsi = 1.0 / p.rsi_period
        change    = close - state.prev_close
        gain      = max(change, 0.0)
        loss      = max(-change, 0.0)
        state.avg_gain = alpha_rsi * gain + (1.0 - alpha_rsi) * state.avg_gain
        state.avg_loss = alpha_rsi * loss + (1.0 - alpha_rsi) * state.avg_loss
        state.prev_close = close

        # ── ATR update (Wilder's SMMA)
        alpha_atr = 1.0 / p.atr_period
        tr = max(high - low, abs(high - state.prev_close), abs(low - state.prev_close))
        state.atr      = alpha_atr * tr    + (1.0 - alpha_atr) * state.atr
        state.prev_high = high
        state.prev_low  = low

        # ── Rolling windows (O(1) append — deque handles eviction)
        state.close_window.append(close)
        state.vol_window.append(volume)

    def _build_features(
        self, bar: BarEvent, state: _SymbolState
    ) -> IncrementalFeatures:
        """Compute derived features from current state."""
        p = self._params
        close  = bar.close
        volume = bar.volume

        # EMA cross
        ema_cross = state.ema_fast - state.ema_slow

        # RSI
        if state.avg_loss == 0.0:
            rsi = 100.0
        else:
            rs  = state.avg_gain / state.avg_loss
            rsi = 100.0 - 100.0 / (1.0 + rs)

        # ATR-derived
        atr     = state.atr
        atr_pct = atr / close if close != 0.0 else 0.0
        price_to_ema = (close - state.ema_fast) / atr if atr != 0.0 else 0.0

        # Bollinger Bands (rolling std over close_window)
        closes_arr = list(state.close_window)
        bb_sma = float(np.mean(closes_arr))
        bb_std = float(np.std(closes_arr, ddof=1)) if len(closes_arr) > 1 else 0.0
        bb_upper = bb_sma + p.bb_std * bb_std
        bb_lower = bb_sma - p.bb_std * bb_std
        band_width = bb_upper - bb_lower
        bb_pct_b = (close - bb_lower) / band_width if band_width != 0.0 else 0.5

        # Volume ratio
        volume_ratio = volume / state.ema_vol if state.ema_vol != 0.0 else 1.0

        # ROC: (close - close[roc_period bars ago]) / close[roc_period bars ago]
        closes_list = list(state.close_window)
        if len(closes_list) >= p.roc_period:
            past_close = closes_list[-p.roc_period]
            roc = (close - past_close) / past_close if past_close != 0.0 else 0.0
        else:
            roc = 0.0

        # VWAP deviation (only if the bar carries a vwap value)
        vwap_dev: float | None = None
        if bar.vwap is not None and atr != 0.0:
            vwap_dev = (close - bar.vwap) / atr

        return IncrementalFeatures(
            symbol          = bar.symbol,
            timestamp       = bar.timestamp,
            timeframe       = self._tf,
            close           = close,
            ema_fast        = state.ema_fast,
            ema_slow        = state.ema_slow,
            ema_cross       = ema_cross,
            rsi             = rsi,
            roc             = roc,
            atr             = atr,
            atr_pct         = atr_pct,
            bb_upper        = bb_upper,
            bb_lower        = bb_lower,
            bb_pct_b        = bb_pct_b,
            volume_ratio    = volume_ratio,
            price_to_ema    = price_to_ema,
            vwap_deviation  = vwap_dev,
            is_warm         = state.is_warm,
        )


# ---------------------------------------------------------------------------
# Batch math helpers (used only during warmup)
# ---------------------------------------------------------------------------


def _batch_ema(values: np.ndarray, period: int) -> float:
    """Return the final EMA value from a batch of prices.

    Uses pandas ewm (adjust=False) which exactly matches the recursive formula
    ``ema_t = k * price_t + (1-k) * ema_{t-1}`` with k = 2/(period+1).
    """
    s = pd.Series(values, dtype=float)
    return float(s.ewm(span=period, adjust=False).mean().iloc[-1])


def _batch_rsi_seed(closes: np.ndarray, period: int) -> tuple[float, float]:
    """Compute Wilder's smoothed avg_gain / avg_loss seed from a price series.

    Steps:
    1. Compute first ``period`` changes to get the SMA-based seed (standard RSI step 1).
    2. Apply Wilder's SMMA over the remaining prices to arrive at the final values.

    Returns (avg_gain, avg_loss) ready for recursive single-bar updates.
    """
    if len(closes) < period + 1:
        return 0.0, 0.0

    changes = np.diff(closes)
    gains   = np.maximum(changes, 0.0)
    losses  = np.maximum(-changes, 0.0)

    # Seed with simple average of first `period` changes
    avg_gain = float(np.mean(gains[:period]))
    avg_loss = float(np.mean(losses[:period]))

    # Wilder's SMMA over remaining changes
    alpha = 1.0 / period
    for g, loss in zip(gains[period:], losses[period:], strict=False):
        avg_gain = alpha * g + (1.0 - alpha) * avg_gain
        avg_loss = alpha * loss + (1.0 - alpha) * avg_loss

    return avg_gain, avg_loss


def _batch_atr_seed(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    period: int,
) -> float:
    """Compute Wilder's SMMA ATR seed from a price series.

    Matches the recursive update formula:
        ATR_t = (1/period) * TR_t + (1 - 1/period) * ATR_{t-1}
    """
    if len(closes) < period + 1:
        # Fallback: simple average of TR over available bars
        tr_arr = np.maximum(
            highs[1:] - lows[1:],
            np.maximum(
                np.abs(highs[1:] - closes[:-1]),
                np.abs(lows[1:]  - closes[:-1]),
            ),
        )
        return float(np.mean(tr_arr)) if len(tr_arr) else 0.0

    tr_arr = np.maximum(
        highs[1:] - lows[1:],
        np.maximum(
            np.abs(highs[1:] - closes[:-1]),
            np.abs(lows[1:]  - closes[:-1]),
        ),
    )

    # SMA seed
    atr = float(np.mean(tr_arr[:period]))

    # Wilder's SMMA over remaining TRs
    alpha = 1.0 / period
    for tr in tr_arr[period:]:
        atr = alpha * float(tr) + (1.0 - alpha) * atr

    return atr
