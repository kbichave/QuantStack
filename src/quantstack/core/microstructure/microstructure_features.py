"""
MicrostructureFeatureEngine — real-time microstructure signals from tick data.

Computed features
-----------------

Order Flow Imbalance (OFI)
    OFI_t = Σ_{k: side="buy"} size_k  −  Σ_{k: side="sell"} size_k
    over the last ``ofi_window`` trades.
    Positive = buy-side pressure; negative = sell-side pressure.
    Normalised OFI divides by total volume over the window.

Kyle's Lambda (price impact)
    Estimated via OLS regression of mid-price changes on signed order flow
    over a rolling window of ``kyle_window`` trades:
        Δmid_t = α + λ × OF_t + ε_t
    where OF_t is the signed trade size (+ for buys, − for sells).
    λ > 0 indicates a market with price-impact cost.

Bid-Ask Spread (from QuoteTick)
    Rolling mean and std of spread over ``spread_window`` quote updates.
    Includes spread in basis points and spread in ticks.

Volume-Synchronized Probability of Informed Trading (VPIN)
    Approximation of Easley–de Prado–O'Hara VPIN using bulk-volume classification:
        VPIN = |V_buy − V_sell| / V_total  over ``vpin_buckets`` volume buckets.
    Each bucket has fixed size ``bucket_volume``.
    VPIN ∈ [0, 1]; high VPIN (~0.5+) signals elevated adverse selection.

Trade Intensity
    Trades per second over the last ``intensity_window`` seconds.

Roll Spread Estimate
    Roll (1984) effective spread: 2 × √(−Cov(ΔP_t, ΔP_{t−1}))
    Requires at least ``roll_window`` consecutive trade prices.

Usage
-----
    engine = MicrostructureFeatureEngine()

    tick_adapter.add_trade_callback(engine.on_trade)
    tick_adapter.add_quote_callback(engine.on_quote)

    async def on_features(features: MicrostructureFeatures) -> None:
        if features.vpin > 0.5:
            logger.warning("High adverse selection detected")

    engine.add_callback(on_features)
"""

from __future__ import annotations

import asyncio
import math
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import numpy as np

from quantstack.data.streaming.tick_models import QuoteTick, TradeTick

# ---------------------------------------------------------------------------
# Output model
# ---------------------------------------------------------------------------


@dataclass
class MicrostructureFeatures:
    """Microstructure feature vector for one symbol at one point in time."""

    symbol: str
    timestamp_ns: int

    # Order flow
    ofi: float  # signed order flow (buy volume − sell volume)
    ofi_normalised: float  # ofi / total_volume  ∈ [−1, 1]

    # Price impact
    kyle_lambda: float | None  # None until kyle_window trades seen

    # Spread (from NBBO quotes)
    spread_mean: float | None  # mean spread over window
    spread_std: float | None  # std of spread
    spread_bps_mean: float | None  # mean spread in bps

    # Volume-Synchronized Probability of Informed Trading
    vpin: float | None  # [0, 1]; None until first full bucket

    # Trade intensity
    trades_per_second: float

    # Roll (1984) spread estimator
    roll_spread: float | None  # None until roll_window trades seen

    # Meta
    trade_count: int  # number of trades used for this snapshot
    is_warm: bool  # False until minimum windows are filled


MicroFeaturesCallback = Callable[[MicrostructureFeatures], Awaitable[None]]


# ---------------------------------------------------------------------------
# Per-symbol state (private)
# ---------------------------------------------------------------------------


@dataclass
class _SymState:
    # OFI window: list of (signed_size,) with sign: +1=buy, −1=sell, 0=unknown
    ofi_trades: deque  # deque[(signed_size: float)]

    # Kyle regression window: list of (signed_flow, mid_change)
    kyle_data: deque  # deque[(signed_flow: float, delta_mid: float)]
    prev_mid: float | None

    # Quote/spread window
    spreads: deque  # deque[float]
    spread_bps: deque  # deque[float]

    # VPIN: fixed-size volume buckets
    buy_vol: float
    sell_vol: float
    bucket_vol: float
    vpin_buckets: deque  # deque[float]  — |V_buy − V_sell| per bucket
    total_bucket_vol: float

    # Trade intensity
    trade_times_s: deque  # deque[float]  — epoch seconds of recent trades

    # Roll spread
    roll_prices: deque  # deque[float]

    # Metadata
    trade_count: int
    is_warm: bool


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class MicrostructureFeatureEngine:
    """Computes real-time microstructure signals from TradeTick / QuoteTick.

    Args:
        ofi_window:        Number of trades for OFI calculation.
        kyle_window:       Number of trades for Kyle lambda estimation.
        spread_window:     Number of quote updates for spread statistics.
        intensity_window_s: Seconds window for trade intensity (trades/sec).
        roll_window:       Number of consecutive trades for Roll spread.
        bucket_volume:     Volume per VPIN bucket.
        vpin_buckets:      Number of buckets for VPIN estimate.
        emit_every_n:      Emit features every N trades (default 1 = every trade).
    """

    def __init__(
        self,
        ofi_window: int = 100,
        kyle_window: int = 200,
        spread_window: int = 50,
        intensity_window_s: float = 60.0,
        roll_window: int = 50,
        bucket_volume: float = 1000.0,
        vpin_buckets: int = 50,
        emit_every_n: int = 1,
    ) -> None:
        self._ofi_window = ofi_window
        self._kyle_window = kyle_window
        self._spread_window = spread_window
        self._intensity_window_s = intensity_window_s
        self._roll_window = roll_window
        self._bucket_volume = bucket_volume
        self._vpin_buckets = vpin_buckets
        self._emit_every_n = emit_every_n

        self._states: dict[str, _SymState] = {}
        self._callbacks: list[MicroFeaturesCallback] = []

    # ── Callback registration ─────────────────────────────────────────────────

    def add_callback(self, callback: MicroFeaturesCallback) -> None:
        self._callbacks.append(callback)

    def remove_callback(self, callback: MicroFeaturesCallback) -> None:
        self._callbacks = [c for c in self._callbacks if c is not callback]

    # ── Tick callbacks ────────────────────────────────────────────────────────

    async def on_trade(self, tick: TradeTick) -> None:
        state = self._get_or_create(tick.symbol)
        state.trade_count += 1

        signed_size = self._signed_size(tick)
        ts_s = tick.timestamp_ns / 1e9

        # OFI accumulation
        state.ofi_trades.append(signed_size)

        # Trade intensity
        state.trade_times_s.append(ts_s)
        # prune old entries
        cutoff = ts_s - self._intensity_window_s
        while state.trade_times_s and state.trade_times_s[0] < cutoff:
            state.trade_times_s.popleft()

        # VPIN bucket accumulation
        if tick.side == "buy":
            state.buy_vol += tick.size
        elif tick.side == "sell":
            state.sell_vol += tick.size
        state.bucket_vol += tick.size
        if state.bucket_vol >= self._bucket_volume:
            imbalance = abs(state.buy_vol - state.sell_vol)
            state.vpin_buckets.append(imbalance)
            state.total_bucket_vol += state.bucket_vol
            state.buy_vol = 0.0
            state.sell_vol = 0.0
            state.bucket_vol = 0.0

        # Kyle window: need previous mid price
        if state.prev_mid is not None:
            # mid will be updated by on_quote; use last known mid
            pass

        # Roll prices
        state.roll_prices.append(tick.price)

        # Emit based on schedule
        if state.trade_count % self._emit_every_n == 0:
            features = self._compute(tick.symbol, tick.timestamp_ns)
            if features:
                await self._emit(features)

    async def on_quote(self, tick: QuoteTick) -> None:
        state = self._get_or_create(tick.symbol)

        mid = tick.mid
        state.spreads.append(tick.spread)
        state.spread_bps.append(tick.spread_bps)

        # Kyle: track mid-price changes for Kyle regression
        if state.prev_mid is not None:
            delta_mid = mid - state.prev_mid
            # signed flow = last OFI trade signed size (best proxy without queuing)
            signed_flow = state.ofi_trades[-1] if state.ofi_trades else 0.0
            state.kyle_data.append((signed_flow, delta_mid))
        state.prev_mid = mid

    # ── Feature computation ───────────────────────────────────────────────────

    def _compute(self, symbol: str, timestamp_ns: int) -> MicrostructureFeatures | None:
        state = self._states.get(symbol)
        if state is None:
            return None

        # OFI
        ofi_list = list(state.ofi_trades)
        buy_vol = sum(x for x in ofi_list if x > 0)
        sell_vol = sum(abs(x) for x in ofi_list if x < 0)
        total_vol = buy_vol + sell_vol
        ofi = buy_vol - sell_vol
        ofi_norm = ofi / total_vol if total_vol > 0 else 0.0

        # Kyle's lambda
        kyle_lambda = self._estimate_kyle(state)

        # Spread statistics
        if state.spreads:
            spread_arr = np.array(list(state.spreads), dtype=float)
            spread_mean = float(np.mean(spread_arr))
            spread_std = (
                float(np.std(spread_arr, ddof=1)) if len(spread_arr) > 1 else 0.0
            )
            bps_arr = np.array(list(state.spread_bps), dtype=float)
            spread_bps_m = float(np.mean(bps_arr))
        else:
            spread_mean = spread_std = spread_bps_m = None

        # VPIN
        vpin = self._compute_vpin(state)

        # Trade intensity
        trades_per_s = len(state.trade_times_s) / self._intensity_window_s

        # Roll spread
        roll_spread = self._compute_roll(state)

        is_warm = (
            len(state.ofi_trades) >= self._ofi_window // 2
            and len(state.spreads) >= self._spread_window // 2
        )

        return MicrostructureFeatures(
            symbol=symbol,
            timestamp_ns=timestamp_ns,
            ofi=ofi,
            ofi_normalised=ofi_norm,
            kyle_lambda=kyle_lambda,
            spread_mean=spread_mean,
            spread_std=spread_std,
            spread_bps_mean=spread_bps_m,
            vpin=vpin,
            trades_per_second=trades_per_s,
            roll_spread=roll_spread,
            trade_count=state.trade_count,
            is_warm=is_warm,
        )

    def _estimate_kyle(self, state: _SymState) -> float | None:
        """OLS estimate of Kyle's lambda from rolling window."""
        if len(state.kyle_data) < self._kyle_window // 2:
            return None
        data = list(state.kyle_data)
        flows = np.array([d[0] for d in data], dtype=float)
        mid_changes = np.array([d[1] for d in data], dtype=float)
        if np.std(flows) < 1e-10:
            return None
        # Simple OLS: lambda = Cov(ΔMid, OF) / Var(OF)
        cov = np.cov(flows, mid_changes)[0, 1]
        var = np.var(flows)
        return float(cov / var) if var > 1e-10 else None

    def _compute_vpin(self, state: _SymState) -> float | None:
        """VPIN from completed buckets."""
        buckets = list(state.vpin_buckets)
        if len(buckets) < self._vpin_buckets // 2:
            return None
        total = len(buckets) * self._bucket_volume
        return float(sum(buckets) / total) if total > 0 else None

    def _compute_roll(self, state: _SymState) -> float | None:
        """Roll (1984) effective spread estimate."""
        prices = list(state.roll_prices)
        if len(prices) < 3:
            return None
        changes = np.diff(prices)
        if len(changes) < 2:
            return None
        cov = np.cov(changes[:-1], changes[1:])[0, 1]
        if cov >= 0:
            return None  # Roll is undefined when covariance is non-negative
        return float(2 * math.sqrt(-cov))

    async def _emit(self, features: MicrostructureFeatures) -> None:
        if not self._callbacks:
            return
        await asyncio.gather(
            *(cb(features) for cb in self._callbacks),
            return_exceptions=True,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_or_create(self, symbol: str) -> _SymState:
        if symbol not in self._states:
            self._states[symbol] = _SymState(
                ofi_trades=deque(maxlen=self._ofi_window),
                kyle_data=deque(maxlen=self._kyle_window),
                prev_mid=None,
                spreads=deque(maxlen=self._spread_window),
                spread_bps=deque(maxlen=self._spread_window),
                buy_vol=0.0,
                sell_vol=0.0,
                bucket_vol=0.0,
                vpin_buckets=deque(maxlen=self._vpin_buckets),
                total_bucket_vol=0.0,
                trade_times_s=deque(),
                roll_prices=deque(maxlen=self._roll_window + 1),
                trade_count=0,
                is_warm=False,
            )
        return self._states[symbol]

    @staticmethod
    def _signed_size(tick: TradeTick) -> float:
        """Return +size for buys, −size for sells, ±0 for unknown."""
        if tick.side == "buy":
            return tick.size
        if tick.side == "sell":
            return -tick.size
        return 0.0
