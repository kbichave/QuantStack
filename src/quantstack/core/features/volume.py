"""
Volume-related features.

Includes volume ratio, OBV, and volume-price analysis.
"""

import numpy as np
import pandas as pd

from quantstack.core.features.base import FeatureBase


class VolumeFeatures(FeatureBase):
    """
    Volume technical indicators.

    Features:
    - Volume ratio (current vs average)
    - OBV (On-Balance Volume)
    - Volume z-score
    - Volume-price relationship indicators
    """

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute volume features.

        Args:
            df: OHLCV DataFrame

        Returns:
            DataFrame with volume features added
        """
        result = df.copy()
        close = result["close"]
        volume = result["volume"]
        high = result["high"]
        low = result["low"]

        # Volume moving average
        result["volume_ma"] = self.sma(volume, self.params.volume_ma_period)

        # Volume ratio (current vs average)
        vol_ma_safe = result["volume_ma"].replace(0, np.nan)
        result["volume_ratio"] = volume / vol_ma_safe

        # Volume z-score
        result["volume_zscore"] = self.zscore(volume, self.params.volume_ma_period)

        # High volume flag
        result["high_volume"] = (result["volume_ratio"] > 1.5).astype(int)
        result["low_volume"] = (result["volume_ratio"] < 0.5).astype(int)

        # On-Balance Volume (OBV)
        result["obv"] = self._compute_obv(close, volume)

        # OBV change (momentum of OBV)
        result["obv_change"] = result["obv"].diff(self.params.obv_period)

        # OBV z-score
        result["obv_zscore"] = self.zscore(result["obv"], self.params.obv_period * 2)

        # Volume-weighted average price (VWAP) - rolling
        result["vwap"] = self._compute_rolling_vwap(
            high,
            low,
            close,
            volume,
            period=self.params.volume_ma_period,
        )

        # Price distance from VWAP
        result["price_dist_vwap"] = (close - result["vwap"]) / result["vwap"] * 100

        # Money Flow Index (MFI)
        result["mfi"] = self._compute_mfi(
            high,
            low,
            close,
            volume,
            period=self.params.rsi_period,
        )

        # Accumulation/Distribution Line
        result["ad_line"] = self._compute_ad_line(high, low, close, volume)
        result["ad_change"] = result["ad_line"].diff(self.params.volume_ma_period)

        # Chaikin Money Flow
        result["cmf"] = self._compute_cmf(
            high,
            low,
            close,
            volume,
            period=self.params.volume_ma_period,
        )

        # Force Index
        result["force_index"] = self._compute_force_index(close, volume)
        result["force_index_ma"] = self.ema(result["force_index"], 13)

        # Volume trend (increasing or decreasing)
        vol_short_ma = self.sma(volume, 5)
        vol_long_ma = self.sma(volume, self.params.volume_ma_period)
        result["volume_trend"] = np.where(
            vol_short_ma > vol_long_ma, 1, np.where(vol_short_ma < vol_long_ma, -1, 0)
        )

        return result

    def _compute_obv(
        self,
        close: pd.Series,
        volume: pd.Series,
    ) -> pd.Series:
        """
        Compute On-Balance Volume (OBV).

        OBV adds volume on up days, subtracts on down days.
        """
        direction = np.sign(close.diff())
        obv = (direction * volume).cumsum()
        return obv

    def _compute_rolling_vwap(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
        period: int,
    ) -> pd.Series:
        """
        Compute rolling Volume-Weighted Average Price.

        VWAP = Cumulative(Typical Price * Volume) / Cumulative(Volume)
        """
        typical_price = (high + low + close) / 3

        tp_volume = typical_price * volume

        rolling_tp_vol = tp_volume.rolling(window=period).sum()
        rolling_vol = volume.rolling(window=period).sum()

        rolling_vol_safe = rolling_vol.replace(0, np.nan)

        return rolling_tp_vol / rolling_vol_safe

    def _compute_mfi(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
        period: int,
    ) -> pd.Series:
        """
        Compute Money Flow Index (MFI).

        MFI is a volume-weighted RSI.
        """
        typical_price = (high + low + close) / 3
        raw_money_flow = typical_price * volume

        # Positive and negative money flow
        tp_change = typical_price.diff()
        positive_flow = raw_money_flow.where(tp_change > 0, 0)
        negative_flow = raw_money_flow.where(tp_change < 0, 0)

        # Rolling sums
        positive_sum = positive_flow.rolling(window=period).sum()
        negative_sum = negative_flow.rolling(window=period).sum()

        # Money flow ratio
        negative_sum_safe = negative_sum.replace(0, np.nan)
        mf_ratio = positive_sum / negative_sum_safe

        # MFI
        mfi = 100 - (100 / (1 + mf_ratio))

        return mfi

    def _compute_ad_line(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
    ) -> pd.Series:
        """
        Compute Accumulation/Distribution Line.

        AD = Previous AD + Money Flow Volume
        Money Flow Volume = Money Flow Multiplier * Volume
        Money Flow Multiplier = ((Close - Low) - (High - Close)) / (High - Low)
        """
        hl_range = high - low
        hl_range_safe = hl_range.replace(0, np.nan)

        mf_multiplier = ((close - low) - (high - close)) / hl_range_safe
        mf_volume = mf_multiplier * volume

        return mf_volume.cumsum()

    def _compute_cmf(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
        period: int,
    ) -> pd.Series:
        """
        Compute Chaikin Money Flow.

        CMF = Sum(Money Flow Volume) / Sum(Volume) over period
        """
        hl_range = high - low
        hl_range_safe = hl_range.replace(0, np.nan)

        mf_multiplier = ((close - low) - (high - close)) / hl_range_safe
        mf_volume = mf_multiplier * volume

        rolling_mf_vol = mf_volume.rolling(window=period).sum()
        rolling_vol = volume.rolling(window=period).sum()
        rolling_vol_safe = rolling_vol.replace(0, np.nan)

        return rolling_mf_vol / rolling_vol_safe

    def _compute_force_index(
        self,
        close: pd.Series,
        volume: pd.Series,
    ) -> pd.Series:
        """
        Compute Force Index.

        Force Index = Price Change * Volume
        """
        return close.diff() * volume

    def get_feature_names(self) -> list[str]:
        """Return list of volume feature names."""
        return [
            "volume_ma",
            "volume_ratio",
            "volume_zscore",
            "high_volume",
            "low_volume",
            "obv",
            "obv_change",
            "obv_zscore",
            "vwap",
            "price_dist_vwap",
            "mfi",
            "ad_line",
            "ad_change",
            "cmf",
            "force_index",
            "force_index_ma",
            "volume_trend",
        ]


# ---------------------------------------------------------------------------
# VPOC / VAH / VAL — Volume Point of Control and Value Area
# ---------------------------------------------------------------------------


class VolumePointOfControl:
    """
    Volume Point of Control (VPOC), Value Area High (VAH), and Value Area Low (VAL).

    The VPOC is the single price bin with the highest volume over the lookback
    window — the "fairest" price where the most trading occurred.

    VAH/VAL mark the upper and lower edges of the 68.2% value area (by default):
    the range containing 68.2% of total volume centred on the VPOC.

    These are often magnets when price deviates far from VPOC (mean-reversion
    signal) and act as invisible support/resistance on re-tests.

    Parameters
    ----------
    lookback : int
        Bars to include in the volume profile. Default 20.
    n_bins : int
        Price bins for the profile. Default 20.
    value_area_pct : float
        Fraction of total volume defining the value area. Default 0.682 (1σ).
    """

    def __init__(
        self, lookback: int = 20, n_bins: int = 20, value_area_pct: float = 0.682
    ) -> None:
        self.lookback = lookback
        self.n_bins = n_bins
        self.value_area_pct = value_area_pct

    def compute(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
    ) -> pd.DataFrame:
        """
        Returns
        -------
        pd.DataFrame with columns:
            vpoc        – price level of maximum volume
            vah         – value area high
            val         – value area low
            above_vah   – 1 if close > vah
            below_val   – 1 if close < val
            in_value    – 1 if close within [val, vah]
        """
        n = len(close)
        vpoc_arr = np.full(n, np.nan)
        vah_arr = np.full(n, np.nan)
        val_arr = np.full(n, np.nan)

        for i in range(self.lookback - 1, n):
            start = max(0, i - self.lookback + 1)
            h_win = high.iloc[start : i + 1].values
            l_win = low.iloc[start : i + 1].values
            v_win = volume.iloc[start : i + 1].values

            price_low = l_win.min()
            price_high = h_win.max()
            if price_high <= price_low:
                continue

            edges = np.linspace(price_low, price_high, self.n_bins + 1)
            bin_vol = np.zeros(self.n_bins)
            bin_mid = (edges[:-1] + edges[1:]) / 2

            for j in range(len(h_win)):
                # Distribute bar's volume across bins it spans
                bar_lo, bar_hi = l_win[j], h_win[j]
                if bar_hi <= bar_lo:
                    bar_hi = bar_lo + 1e-9
                bar_span = bar_hi - bar_lo
                for b in range(self.n_bins):
                    overlap = min(bar_hi, edges[b + 1]) - max(bar_lo, edges[b])
                    if overlap > 0:
                        bin_vol[b] += v_win[j] * overlap / bar_span

            vpoc_bin = int(np.argmax(bin_vol))
            vpoc_arr[i] = bin_mid[vpoc_bin]

            # Value area: expand from VPOC until value_area_pct of total volume captured
            total_vol = bin_vol.sum()
            target_vol = total_vol * self.value_area_pct
            lo_idx = hi_idx = vpoc_bin
            captured = bin_vol[vpoc_bin]
            while captured < target_vol:
                can_up = hi_idx + 1 < self.n_bins
                can_dn = lo_idx - 1 >= 0
                if not can_up and not can_dn:
                    break
                up_vol = bin_vol[hi_idx + 1] if can_up else -1
                dn_vol = bin_vol[lo_idx - 1] if can_dn else -1
                if up_vol >= dn_vol:
                    hi_idx += 1
                    captured += bin_vol[hi_idx]
                else:
                    lo_idx -= 1
                    captured += bin_vol[lo_idx]

            vah_arr[i] = edges[hi_idx + 1]
            val_arr[i] = edges[lo_idx]

        vpoc = pd.Series(vpoc_arr, index=close.index)
        vah = pd.Series(vah_arr, index=close.index)
        val = pd.Series(val_arr, index=close.index)

        above_vah = ((close > vah) & vah.notna()).astype(int)
        below_val = ((close < val) & val.notna()).astype(int)
        in_value = ((close >= val) & (close <= vah) & vah.notna()).astype(int)

        return pd.DataFrame(
            {
                "vpoc": vpoc,
                "vah": vah,
                "val": val,
                "above_vah": above_vah,
                "below_val": below_val,
                "in_value": in_value,
            },
            index=close.index,
        )


# ---------------------------------------------------------------------------
# Anchored VWAP
# ---------------------------------------------------------------------------


class AnchoredVWAP:
    """
    Anchored VWAP — VWAP computed from a specific anchor bar.

    Unlike rolling VWAP, the anchor accumulates from a chosen starting bar
    (e.g. a swing low, earnings gap, or structural event). Price returning
    to anchored VWAP after deviation is a high-probability mean-reversion setup.

    Anchors can be:
    - A specific integer index into the series
    - A pandas Timestamp / datetime-like matched against the series index

    Parameters
    ----------
    anchor : int or datetime-like
        Starting bar. int = positional index; datetime = matched to series index.
    """

    def __init__(self, anchor: int | object = 0) -> None:
        self.anchor = anchor

    def compute(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
    ) -> pd.DataFrame:
        """
        Returns
        -------
        pd.DataFrame with columns:
            avwap           – Anchored VWAP value
            avwap_deviation – (close - avwap) / avwap × 100
            above_avwap     – 1 if close > avwap
        """
        # Resolve anchor to positional index
        if isinstance(self.anchor, int):
            anchor_idx = self.anchor
        else:
            try:
                loc = close.index.get_loc(self.anchor)
                anchor_idx = int(loc) if not hasattr(loc, "__len__") else int(loc[0])
            except KeyError:
                anchor_idx = 0

        anchor_idx = max(0, min(anchor_idx, len(close) - 1))
        typical = (high + low + close) / 3.0

        avwap_vals = np.full(len(close), np.nan)
        cum_tpv = 0.0
        cum_vol = 0.0
        for i in range(len(close)):
            if i >= anchor_idx:
                cum_tpv += typical.iloc[i] * volume.iloc[i]
                cum_vol += volume.iloc[i]
                avwap_vals[i] = cum_tpv / cum_vol if cum_vol > 0 else np.nan

        avwap = pd.Series(avwap_vals, index=close.index)
        deviation = ((close - avwap) / avwap * 100).where(avwap.notna())
        above = (close > avwap).astype(int).where(avwap.notna()).fillna(0).astype(int)

        return pd.DataFrame(
            {"avwap": avwap, "avwap_deviation": deviation, "above_avwap": above},
            index=close.index,
        )
