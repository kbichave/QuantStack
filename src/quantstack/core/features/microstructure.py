"""
Market microstructure signals derivable from OHLCV data only.

All estimators here require no Level-2 or tick data — they approximate
bid-ask spread and liquidity from high/low/close/volume alone.

Indicators
----------
AmihudIlliquidity   – |return| / dollar_volume; measures price impact per $ traded
RollImpliedSpread   – 2 * sqrt(-Cov(ΔP_t, ΔP_{t-1})); spread from serial covariance
CorwinSchultzSpread – spread estimate from high/low ratio (Corwin & Schultz 2012)
RealizedVarianceDecomposition – overnight gap² vs intraday range²; regime context
VWAPSessionDeviation – close vs session VWAP at end-of-day; mean-reversion predictor
OvernightGapPersistence – whether opening gap persists vs. fills by close
"""

import numpy as np
import pandas as pd


class AmihudIlliquidity:
    """
    Amihud (2002) illiquidity ratio.

    ILLIQ_t = |R_t| / DollarVolume_t

    A rolling average smooths out single-day spikes. Higher values mean
    each dollar of trading moves price more — low-liquidity regime signal.

    Parameters
    ----------
    period : int
        Rolling window for the smoothed average. Default 22 (~1 month).
    """

    def __init__(self, period: int = 22) -> None:
        self.period = period

    def compute(self, close: pd.Series, volume: pd.Series) -> pd.DataFrame:
        """
        Parameters
        ----------
        close, volume : pd.Series
            Daily close prices and share volume with shared DatetimeIndex.

        Returns
        -------
        pd.DataFrame with columns:
            amihud_raw  – single-day illiquidity ratio
            amihud      – rolling mean (smoothed, primary signal)
            amihud_zscore – z-score vs rolling window (regime indicator)
        """
        returns = close.pct_change().abs()
        dollar_vol = close * volume
        dollar_vol_safe = dollar_vol.replace(0, np.nan)

        amihud_raw = (returns / dollar_vol_safe) * 1e6  # scale to readable units

        amihud = amihud_raw.rolling(window=self.period).mean()
        amihud_std = amihud_raw.rolling(window=self.period).std().replace(0, np.nan)
        amihud_zscore = (amihud_raw - amihud) / amihud_std

        return pd.DataFrame(
            {
                "amihud_raw": amihud_raw,
                "amihud": amihud,
                "amihud_zscore": amihud_zscore,
            },
            index=close.index,
        )


class RollImpliedSpread:
    """
    Roll (1984) implied bid-ask spread from serial covariance of price changes.

    Spread = 2 * sqrt(-Cov(ΔP_t, ΔP_{t-1}))

    When the covariance is positive (momentum), the formula returns NaN —
    these bars are recorded as NaN rather than forcing a zero.

    Parameters
    ----------
    period : int
        Rolling window for covariance computation. Default 22.
    """

    def __init__(self, period: int = 22) -> None:
        self.period = period

    def compute(self, close: pd.Series) -> pd.DataFrame:
        """
        Returns
        -------
        pd.DataFrame with columns:
            roll_spread – implied spread in price units (NaN when cov >= 0)
            roll_spread_pct – spread as % of close price
        """
        delta = close.diff()
        delta_lag = delta.shift(1)

        # Rolling covariance between ΔP_t and ΔP_{t-1}
        cov = delta.rolling(window=self.period).cov(delta_lag)

        # Spread is only defined when cov < 0 (bid-ask bounce dominates)
        cov_neg = cov.where(cov < 0, other=np.nan)
        roll_spread = 2 * np.sqrt(-cov_neg)

        roll_spread_pct = roll_spread / close * 100

        return pd.DataFrame(
            {
                "roll_spread": roll_spread,
                "roll_spread_pct": roll_spread_pct,
            },
            index=close.index,
        )


class CorwinSchultzSpread:
    """
    Corwin & Schultz (2012) high-low spread estimator.

    Uses the ratio of two-day high/low ranges to one-day ranges to
    back out an implied bid-ask spread. Works on standard OHLCV data.

    The key insight: a two-day H/L range spans more of the bid-ask
    spread than a one-day range, and the difference isolates spread costs.

    Parameters
    ----------
    period : int
        Rolling window for the smoothed spread. Default 22.
    """

    def __init__(self, period: int = 22) -> None:
        self.period = period

    @staticmethod
    def _beta(high: pd.Series, low: pd.Series) -> pd.Series:
        """β = [ln(H_t/L_t)]² + [ln(H_{t+1}/L_{t+1})]²"""
        log_hl = np.log(high / low)
        return log_hl**2 + log_hl.shift(-1) ** 2

    @staticmethod
    def _gamma(high: pd.Series, low: pd.Series) -> pd.Series:
        """γ = [ln(max(H_t, H_{t+1}) / min(L_t, L_{t+1}))]²"""
        two_day_high = high.rolling(2).max()
        two_day_low = low.rolling(2).min()
        return np.log(two_day_high / two_day_low) ** 2

    def compute(
        self, high: pd.Series, low: pd.Series, close: pd.Series
    ) -> pd.DataFrame:
        """
        Returns
        -------
        pd.DataFrame with columns:
            cs_spread      – Corwin-Schultz spread (price units; NaN when formula undefined)
            cs_spread_pct  – spread as % of close
            cs_spread_ma   – rolling mean of cs_spread (smoothed)
        """
        beta = self._beta(high, low)
        gamma = self._gamma(high, low)

        # α = (sqrt(2β) - sqrt(β)) / (3 - 2√2) - sqrt(γ / (3 - 2√2))
        k = 3 - 2 * np.sqrt(2)  # ≈ 0.1716
        alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)

        # Spread = 2(e^α - 1) / (1 + e^α); clip negatives to NaN
        alpha_valid = alpha.where(alpha >= 0, other=np.nan)
        exp_alpha = np.exp(alpha_valid)
        cs_spread = 2 * (exp_alpha - 1) / (1 + exp_alpha) * close

        cs_spread_pct = cs_spread / close * 100
        cs_spread_ma = cs_spread.rolling(window=self.period).mean()

        return pd.DataFrame(
            {
                "cs_spread": cs_spread,
                "cs_spread_pct": cs_spread_pct,
                "cs_spread_ma": cs_spread_ma,
            },
            index=close.index,
        )


class RealizedVarianceDecomposition:
    """
    Decomposes total realized variance into overnight and intraday components.

    overnight_var = (open_t - close_{t-1})²  — news/gap risk
    intraday_var  = (high_t - low_t)²        — execution risk proxy
    ratio = overnight_var / total_var         — regime indicator

    A high ratio means most vol is coming from gaps (information events,
    after-hours news). A low ratio means intraday vol dominates (order flow,
    momentum, mean reversion).

    Parameters
    ----------
    period : int
        Rolling window for smoothed components. Default 22.
    """

    def __init__(self, period: int = 22) -> None:
        self.period = period

    def compute(
        self,
        open_: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
    ) -> pd.DataFrame:
        """
        Parameters
        ----------
        open_, high, low, close : pd.Series
            OHLCV price series with shared DatetimeIndex.

        Returns
        -------
        pd.DataFrame with columns:
            overnight_var       – squared overnight gap (close[t-1] to open[t])
            intraday_var        – squared intraday range (high - low)
            total_var           – sum of both components
            overnight_var_ratio – fraction of total variance from overnight gaps
            rv_overnight_ma     – rolling mean of overnight_var
            rv_intraday_ma      – rolling mean of intraday_var
        """
        overnight_var = (open_ - close.shift(1)) ** 2
        intraday_var = (high - low) ** 2
        total_var = overnight_var + intraday_var
        total_safe = total_var.replace(0, np.nan)

        overnight_ratio = overnight_var / total_safe

        return pd.DataFrame(
            {
                "overnight_var": overnight_var,
                "intraday_var": intraday_var,
                "total_var": total_var,
                "overnight_var_ratio": overnight_ratio,
                "rv_overnight_ma": overnight_var.rolling(self.period).mean(),
                "rv_intraday_ma": intraday_var.rolling(self.period).mean(),
            },
            index=close.index,
        )


class VWAPSessionDeviation:
    """
    End-of-session VWAP deviation signal.

    Computes a rolling VWAP (typical price × volume) and measures how far
    close is from it. A persistent VWAP deviation at session end tends to
    partially revert at the next open — useful as a mean-reversion predictor.

    Parameters
    ----------
    period : int
        Rolling window for VWAP computation. Default 20 (≈ 1 month).
    """

    def __init__(self, period: int = 20) -> None:
        self.period = period

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
            vwap_rolling    – rolling VWAP (typical price × volume weighted)
            vwap_deviation  – (close - vwap) / vwap * 100 (percentage)
            vwap_deviation_zscore – z-score of deviation over same window
        """
        typical = (high + low + close) / 3.0
        dollar_vol = typical * volume
        cumulative_dollar = dollar_vol.rolling(self.period).sum()
        cumulative_vol = volume.rolling(self.period).sum().replace(0, np.nan)

        vwap = cumulative_dollar / cumulative_vol
        deviation = (close - vwap) / vwap.replace(0, np.nan) * 100

        dev_mean = deviation.rolling(self.period).mean()
        dev_std = deviation.rolling(self.period).std().replace(0, np.nan)
        dev_zscore = (deviation - dev_mean) / dev_std

        return pd.DataFrame(
            {
                "vwap_rolling": vwap,
                "vwap_deviation": deviation,
                "vwap_deviation_zscore": dev_zscore,
            },
            index=close.index,
        )


class OvernightGapPersistence:
    """
    Classifies whether an opening gap persists or fills by close.

    A gap that persists (close confirms gap direction) indicates informed
    trading or structural news. A gap that fills (price reverts) suggests
    an overreaction or liquidity grab.

    When volume is provided, identifies gaps accompanied by abnormal auction
    volume — the primary institutional intent signal. A gap-up on 2× average
    volume that persists is a strong institutional accumulation signal; a
    gap-down on high volume that persists is institutional distribution.

    Parameters
    ----------
    min_gap_pct : float
        Minimum gap size (as % of prior close) to count as a gap. Default 0.2%.
    volume_spike_mult : float
        Volume multiple above rolling average to classify as a spike. Default 2.0.
    volume_avg_window : int
        Rolling window for average volume computation. Default 20.
    """

    def __init__(
        self,
        min_gap_pct: float = 0.2,
        volume_spike_mult: float = 2.0,
        volume_avg_window: int = 20,
    ) -> None:
        self.min_gap_pct = min_gap_pct
        self.volume_spike_mult = volume_spike_mult
        self.volume_avg_window = volume_avg_window

    def compute(
        self,
        open_: pd.Series,
        close: pd.Series,
        volume: pd.Series | None = None,
    ) -> pd.DataFrame:
        """
        Parameters
        ----------
        open_ : pd.Series
            Daily open prices.
        close : pd.Series
            Daily close prices.
        volume : pd.Series | None
            Daily volume. When provided, adds volume_spike and
            institutional_gap columns.

        Returns
        -------
        pd.DataFrame with columns:
            gap_pct             – overnight gap as % of prior close (+up, -down)
            gap_up              – 1 if gap > min_gap_pct
            gap_down            – 1 if gap < -min_gap_pct
            gap_filled          – 1 if close moved against gap direction
            gap_persisted       – 1 if close confirmed gap direction
            gap_filled_pct      – rolling fraction of gaps that filled (22-bar)
            volume_spike        – 1 if volume > volume_spike_mult × rolling_avg
                                  (only present when volume provided)
            institutional_gap   – 1 when gap persisted AND volume_spike occurred;
                                  primary institutional-intent signal
                                  (only present when volume provided)
        """
        prev_close = close.shift(1)
        gap_pct = (open_ - prev_close) / prev_close.replace(0, np.nan) * 100

        gap_up = (gap_pct > self.min_gap_pct).astype(int)
        gap_down = (gap_pct < -self.min_gap_pct).astype(int)

        # Filled: gap up but close < open; gap down but close > open
        intraday_move = close - open_
        gap_filled = (
            ((gap_up == 1) & (intraday_move < 0))
            | ((gap_down == 1) & (intraday_move > 0))
        ).astype(int)

        any_gap = (gap_up | gap_down).astype(bool)
        gap_persisted = (any_gap & (gap_filled == 0)).astype(int)

        # Rolling fill rate (meaningful only when gaps occur)
        gap_filled_pct = gap_filled.rolling(22).mean() * 100

        out = pd.DataFrame(
            {
                "gap_pct": gap_pct,
                "gap_up": gap_up,
                "gap_down": gap_down,
                "gap_filled": gap_filled,
                "gap_persisted": gap_persisted,
                "gap_filled_pct": gap_filled_pct,
            },
            index=close.index,
        )

        # Volume spike and institutional gap (require volume)
        if volume is not None:
            vol_avg = volume.rolling(self.volume_avg_window, min_periods=5).mean()
            volume_spike = (volume > self.volume_spike_mult * vol_avg).astype(int)
            institutional_gap = ((gap_persisted == 1) & (volume_spike == 1)).astype(int)
            out["volume_spike"] = volume_spike
            out["institutional_gap"] = institutional_gap

        return out
