"""
Interest rate and yield curve signals.

All computations operate on time series of Treasury yields (or equivalent
risk-free rates) fetched from FD.ai's interest rates endpoint.

Signals
-------
YieldCurveFeatures  – 2s10s spread, 3m10y spread, curve slope, inversion flag
DualMomentum        – Antonacci absolute + relative momentum (OHLCV-only)
SpreadSignals       – TED spread (credit risk proxy) and FRA-OIS (liquidity risk)
                      derived from Treasury 3M and a Fed Funds / SOFR proxy
"""

import numpy as np
import pandas as pd


class YieldCurveFeatures:
    """
    Yield curve shape indicators from Treasury rate time series.

    Key spreads used by the Fed and practitioners as recession/regime indicators:
    - 2s10s: 10Y minus 2Y (traditional yield curve steepness)
    - 3m10y: 10Y minus 3M (Estrella-Mishkin recession indicator — Fed preferred)
    - curve_slope: signed direction, smoothed
    - inverted: binary flag when 2s10s < 0

    All spreads are in basis points (multiply by 100 if inputs are in percent).

    Parameters
    ----------
    smooth_period : int
        Rolling mean window applied to spreads for noise reduction. Default 5.
    """

    def __init__(self, smooth_period: int = 5) -> None:
        self.smooth_period = smooth_period

    def compute(
        self,
        rate_3m: pd.Series,
        rate_2y: pd.Series,
        rate_10y: pd.Series,
    ) -> pd.DataFrame:
        """
        Parameters
        ----------
        rate_3m, rate_2y, rate_10y : pd.Series
            Treasury yield time series (annualized, in % or decimal — consistent units).
            All must share the same DatetimeIndex.

        Returns
        -------
        pd.DataFrame with columns:
            spread_2s10s        – 10Y - 2Y (basis points × 100 if inputs in %)
            spread_3m10y        – 10Y - 3M
            spread_2s10s_smooth – smoothed 2s10s spread
            spread_3m10y_smooth – smoothed 3m10y spread
            curve_inverted      – 1 when 2s10s < 0
            curve_deeply_inv    – 1 when 2s10s < -0.5 (deep inversion)
            spread_2s10s_zscore – z-score vs 252-bar rolling window (regime)
            spread_3m10y_zscore – z-score vs 252-bar rolling window
        """
        s_2s10s = rate_10y - rate_2y
        s_3m10y = rate_10y - rate_3m

        smooth = self.smooth_period
        s_2s10s_sm = s_2s10s.rolling(smooth).mean()
        s_3m10y_sm = s_3m10y.rolling(smooth).mean()

        inverted = (s_2s10s < 0).astype(int)
        deeply_inv = (s_2s10s < -0.5).astype(int)

        def _zscore(series: pd.Series, period: int = 252) -> pd.Series:
            mu = series.rolling(period).mean()
            sigma = series.rolling(period).std().replace(0, np.nan)
            return (series - mu) / sigma

        return pd.DataFrame(
            {
                "spread_2s10s": s_2s10s,
                "spread_3m10y": s_3m10y,
                "spread_2s10s_smooth": s_2s10s_sm,
                "spread_3m10y_smooth": s_3m10y_sm,
                "curve_inverted": inverted,
                "curve_deeply_inv": deeply_inv,
                "spread_2s10s_zscore": _zscore(s_2s10s),
                "spread_3m10y_zscore": _zscore(s_3m10y),
            },
            index=rate_10y.index,
        )


class DualMomentum:
    """
    Antonacci (2012) dual momentum — absolute + relative momentum.

    Filters for both conditions before entering:
    1. Absolute momentum: 12-month-minus-1-month return > 0 (asset beats cash)
    2. Relative momentum: asset ranks in top N% of universe by momentum score

    For single-asset use (no universe), only absolute momentum applies.
    The signal is conservative: both must be positive for a long signal.

    Parameters
    ----------
    abs_lookback : int
        Lookback for absolute momentum in bars. Default 252 (1Y daily).
    skip_period : int
        Most-recent bars to skip (standard 1-month reversal skip). Default 21.
    """

    def __init__(self, abs_lookback: int = 252, skip_period: int = 21) -> None:
        self.abs_lookback = abs_lookback
        self.skip_period = skip_period

    def compute(self, close: pd.Series, risk_free_rate: float = 0.0) -> pd.DataFrame:
        """
        Parameters
        ----------
        close : pd.Series
            Asset close price series.
        risk_free_rate : float
            Annualized risk-free rate as decimal (e.g., 0.05 = 5%).
            Used as hurdle for absolute momentum. Default 0.

        Returns
        -------
        pd.DataFrame with columns:
            momentum_12m1m      – 12m-1m return (standard academic factor)
            abs_momentum_signal – 1 if momentum_12m1m > risk_free threshold
            momentum_6m         – 6-month return (alternative lookback)
            momentum_3m         – 3-month return (shorter lookback)
        """
        # 12-month-minus-1-month: return from bar -(abs_lookback) to bar -(skip_period)
        past_price = close.shift(self.abs_lookback)
        recent_price = close.shift(self.skip_period)

        mom_12m1m = (recent_price - past_price) / past_price.replace(0, np.nan)

        # Daily hurdle: (1 + rfr)^(1/252) - 1
        daily_hurdle = (1 + risk_free_rate) ** (1 / 252) - 1
        period_hurdle = (1 + daily_hurdle) ** (self.abs_lookback - self.skip_period) - 1

        abs_signal = (mom_12m1m > period_hurdle).astype(int)

        mom_6m = close.pct_change(126)  # ~6 months
        mom_3m = close.pct_change(63)  # ~3 months

        return pd.DataFrame(
            {
                "momentum_12m1m": mom_12m1m * 100,  # expressed as %
                "abs_momentum_signal": abs_signal,
                "momentum_6m": mom_6m * 100,
                "momentum_3m": mom_3m * 100,
            },
            index=close.index,
        )


class SpreadSignals:
    """
    Credit and liquidity risk spreads derived from Treasury rates.

    TED Spread
    ----------
    Originally T-Bill minus LIBOR (Treasury–Euro Dollar). Since LIBOR's
    retirement, the closest equivalent is 3M Treasury yield minus the
    overnight risk-free rate (SOFR, Fed Funds, or an equivalent proxy).
    A widening TED spread signals increasing credit/counterparty stress;
    historically has led equity drawdowns by 2–6 weeks.

    FRA-OIS Approximation
    ---------------------
    Forward Rate Agreement (FRA) vs. Overnight Index Swap (OIS) spread
    measures interbank lending risk above the overnight risk-free rate.
    Approximated as: 3M_yield - overnight_rate.
    This is structurally the same calculation as TED in a post-LIBOR world;
    both are exposed here for clarity on which rate pair is being used.

    Parameters
    ----------
    stress_threshold : float
        TED spread level (in same units as inputs) above which the market
        is considered in credit stress. Default 0.5 (50 bps if rates in %).
    smooth_period : int
        Rolling mean window for noise reduction. Default 5.
    """

    def __init__(self, stress_threshold: float = 0.5, smooth_period: int = 5) -> None:
        self.stress_threshold = stress_threshold
        self.smooth_period = smooth_period

    def compute(
        self,
        rate_3m: pd.Series,
        overnight_rate: pd.Series,
    ) -> pd.DataFrame:
        """
        Compute TED spread and FRA-OIS approximation.

        Parameters
        ----------
        rate_3m : pd.Series
            3-month Treasury yield time series (annualized, consistent units).
        overnight_rate : pd.Series
            Overnight risk-free rate: SOFR, Fed Funds Effective Rate, or
            equivalent. Must share the same DatetimeIndex as rate_3m.

        Returns
        -------
        pd.DataFrame with columns:
            ted_spread          – rate_3m - overnight_rate (the spread)
            ted_spread_smooth   – smoothed ted_spread (rolling mean)
            ted_spread_zscore   – z-score vs. 252-bar rolling window
            fra_ois_approx      – same as ted_spread (structural equivalent
                                  post-LIBOR; included for naming clarity)
            credit_stress       – 1 when ted_spread > stress_threshold
            spread_widening     – 1 when ted_spread increased vs. prior bar
                                  (momentum of credit deterioration)
        """
        ted = rate_3m - overnight_rate

        smooth = ted.rolling(self.smooth_period).mean()

        mu = ted.rolling(252).mean()
        sigma = ted.rolling(252).std().replace(0, np.nan)
        zscore = (ted - mu) / sigma

        credit_stress = (ted > self.stress_threshold).astype(int)
        spread_widening = (ted.diff() > 0).astype(int)

        return pd.DataFrame(
            {
                "ted_spread": ted,
                "ted_spread_smooth": smooth,
                "ted_spread_zscore": zscore,
                "fra_ois_approx": ted,  # same calculation, different naming convention
                "credit_stress": credit_stress,
                "spread_widening": spread_widening,
            },
            index=rate_3m.index,
        )
