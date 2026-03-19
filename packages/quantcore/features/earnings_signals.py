"""
Earnings-based alpha signals.

Sources: FD.ai earnings endpoint (get_earnings), analyst estimates endpoint
(get_analyst_estimates). All functions operate on DataFrames with the schemas
returned by those endpoints — no network calls are made here.

Signals
-------
EarningsSurpriseSignals – PEAD (Post-Earnings Announcement Drift), SUE
                           (Standardized Unexpected Earnings), earnings streak
AnalystRevisionSignals  – estimate revision momentum, dispersion, whisper gap
EarningsImpliedMove     – options-implied expected move vs. realized post-earnings
                           move; identifies IV crush and move surprises
"""

import numpy as np
import pandas as pd


class EarningsSurpriseSignals:
    """
    Post-Earnings Announcement Drift (PEAD) and Standardized Unexpected
    Earnings (SUE) signals.

    PEAD is one of the most persistent anomalies in empirical finance: high-SUE
    stocks drift upward for 60+ days post-earnings; low-SUE stocks drift down.

    SUE = (actual_EPS - consensus_EPS) / rolling_std(historical_surprises)

    Parameters
    ----------
    sue_lookback : int
        Number of prior quarters to compute the surprise std-dev normalization.
        Default 8 (2 years of quarterly data).
    """

    def __init__(self, sue_lookback: int = 8) -> None:
        self.sue_lookback = sue_lookback

    def compute(self, earnings_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute PEAD-related signals from a quarterly earnings history DataFrame.

        Parameters
        ----------
        earnings_df : pd.DataFrame
            Must contain columns:
            - report_date (datetime or str)     — earnings announcement date
            - actual_eps (float)                — reported EPS
            - estimated_eps (float)             — consensus estimate at time of report
            - symbol (str, optional)            — ticker; used if multiple tickers present

        Returns
        -------
        pd.DataFrame (same index as input, sorted by report_date) with columns:
            eps_surprise        – actual_eps - estimated_eps (raw beat/miss)
            eps_surprise_pct    – surprise as % of |estimated_eps|
            sue                 – standardized unexpected earnings
            sue_positive        – 1 if sue > 1 (strong beat, top decile)
            sue_negative        – 1 if sue < -1 (strong miss, bottom decile)
            beat_streak         – consecutive quarters of positive surprise
            miss_streak         – consecutive quarters of negative surprise
        """
        df = earnings_df.copy()
        df["report_date"] = pd.to_datetime(df["report_date"])
        df = df.sort_values("report_date").reset_index(drop=True)

        df["eps_surprise"] = df["actual_eps"] - df["estimated_eps"]

        est_safe = df["estimated_eps"].replace(0, np.nan).abs()
        df["eps_surprise_pct"] = df["eps_surprise"] / est_safe * 100

        # SUE: surprise normalized by historical std-dev of surprises
        surprise_std = df["eps_surprise"].rolling(window=self.sue_lookback, min_periods=2).std()
        surprise_std_safe = surprise_std.replace(0, np.nan)
        df["sue"] = df["eps_surprise"] / surprise_std_safe

        df["sue_positive"] = (df["sue"] > 1).astype(int)
        df["sue_negative"] = (df["sue"] < -1).astype(int)

        # Consecutive beat/miss streaks
        beat = (df["eps_surprise"] > 0).astype(int)
        miss = (df["eps_surprise"] < 0).astype(int)

        df["beat_streak"] = beat.groupby((beat != beat.shift()).cumsum()).cumcount() + 1
        df["beat_streak"] = df["beat_streak"].where(beat == 1, 0)
        df["miss_streak"] = miss.groupby((miss != miss.shift()).cumsum()).cumcount() + 1
        df["miss_streak"] = df["miss_streak"].where(miss == 1, 0)

        return df


class AnalystRevisionSignals:
    """
    Analyst estimate revision momentum and dispersion signals.

    Revision momentum (Gleason & Lee 2003) is a persistent alpha factor:
    stocks with upward estimate revisions continue to outperform those with
    downward revisions over the next 1–3 months.

    Parameters
    ----------
    revision_window : int
        Number of prior estimates to use for momentum computation. Default 4.
    """

    def __init__(self, revision_window: int = 4) -> None:
        self.revision_window = revision_window

    def compute(self, estimates_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute analyst revision signals from an estimate history DataFrame.

        Parameters
        ----------
        estimates_df : pd.DataFrame
            Must contain columns:
            - estimate_date (datetime or str) — when estimate was published
            - eps_estimate (float)            — EPS estimate value
            - analyst_count (int, optional)   — number of analysts contributing

        Returns
        -------
        pd.DataFrame (same index as input, sorted by estimate_date) with columns:
            consensus_eps        – current consensus (mean of window)
            prior_consensus_eps  – consensus from revision_window periods ago
            revision_momentum    – (current - prior) / |prior| * 100 (%)
            revision_up          – 1 if revision_momentum > 2%
            revision_down        – 1 if revision_momentum < -2%
            estimate_dispersion  – std_dev / |mean| of estimates in window
            dispersion_high      – 1 if dispersion > 0.2 (high uncertainty)
        """
        df = estimates_df.copy()
        df["estimate_date"] = pd.to_datetime(df["estimate_date"])
        df = df.sort_values("estimate_date").reset_index(drop=True)

        w = self.revision_window
        df["consensus_eps"] = df["eps_estimate"].rolling(window=w, min_periods=1).mean()
        df["prior_consensus_eps"] = df["consensus_eps"].shift(w)

        prior_safe = df["prior_consensus_eps"].replace(0, np.nan).abs()
        df["revision_momentum"] = (df["consensus_eps"] - df["prior_consensus_eps"]) / prior_safe * 100

        df["revision_up"] = (df["revision_momentum"] > 2).astype(int)
        df["revision_down"] = (df["revision_momentum"] < -2).astype(int)

        rolling_std = df["eps_estimate"].rolling(window=w, min_periods=2).std()
        rolling_mean = df["eps_estimate"].rolling(window=w, min_periods=1).mean().replace(0, np.nan).abs()
        df["estimate_dispersion"] = rolling_std / rolling_mean
        df["dispersion_high"] = (df["estimate_dispersion"] > 0.2).astype(int)

        return df


class EarningsImpliedMove:
    """
    Options-implied expected move vs. realized post-earnings move.

    The implied move is derived from at-the-money IV at a given DTE:
        implied_move_pct = atm_iv × sqrt(dte_days / 365)

    This is the market's 1-standard-deviation expectation of the earnings
    move. Comparing it to the realized move reveals whether IV systematically
    overstates or understates earnings volatility for this name.

    Persistent IV overstatement (iv_overstated > 0 on average) is a
    short-volatility alpha opportunity (sell straddles into earnings).
    Persistent understatement flags gap-risk names.

    Parameters
    ----------
    min_iv : float
        Minimum IV to consider valid (filters data errors). Default 0.01.
    """

    def __init__(self, min_iv: float = 0.01) -> None:
        self.min_iv = min_iv

    def compute(self, earnings_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute implied vs. realized move signals per earnings event.

        Parameters
        ----------
        earnings_df : pd.DataFrame
            Must contain columns:
            - period_end (datetime or str)      — fiscal quarter end date
            - atm_iv (float)                    — ATM implied volatility at the
                                                  time of earnings (annualized)
            - dte_days (int/float)              — days to nearest option expiry
                                                  used for pricing the IV
            - pre_earnings_price (float)        — closing price before announcement
            - post_earnings_price (float)       — closing price after announcement
                                                  (typically next-day open or close)

        Returns
        -------
        pd.DataFrame (same index as input, sorted by period_end) with columns:
            implied_move_pct    – IV-implied 1-std expected move as % (>0)
            realized_move_pct   – abs(post/pre - 1) × 100 — actual % move
            iv_overstated       – 1 when implied_move > realized_move (IV crush)
            move_surprise       – realized_move - implied_move (positive = bigger
                                  than priced; negative = smaller than priced)
            move_surprise_norm  – move_surprise / implied_move (relative overshoot)
            historical_iv_bias  – rolling mean of move_surprise (last 4 events);
                                  positive = this stock typically outperforms IV
        """
        df = earnings_df.copy()
        df["period_end"] = pd.to_datetime(df["period_end"])
        df = df.sort_values("period_end").reset_index(drop=True)

        # Mask rows with invalid IV or prices
        valid_iv = (df["atm_iv"] >= self.min_iv) & df["atm_iv"].notna()
        valid_prices = (
            df["pre_earnings_price"].gt(0)
            & df["post_earnings_price"].gt(0)
            & df["pre_earnings_price"].notna()
            & df["post_earnings_price"].notna()
        )
        valid_dte = df["dte_days"].gt(0) & df["dte_days"].notna()

        df["implied_move_pct"] = np.where(
            valid_iv & valid_dte,
            df["atm_iv"] * np.sqrt(df["dte_days"] / 365.0) * 100,
            np.nan,
        )

        df["realized_move_pct"] = np.where(
            valid_prices,
            (df["post_earnings_price"] / df["pre_earnings_price"] - 1).abs() * 100,
            np.nan,
        )

        df["iv_overstated"] = (
            (df["implied_move_pct"] > df["realized_move_pct"])
            & df["implied_move_pct"].notna()
            & df["realized_move_pct"].notna()
        ).astype(int)

        df["move_surprise"] = df["realized_move_pct"] - df["implied_move_pct"]

        implied_safe = df["implied_move_pct"].replace(0, np.nan)
        df["move_surprise_norm"] = df["move_surprise"] / implied_safe

        # Rolling bias: mean move_surprise over last 4 events
        df["historical_iv_bias"] = df["move_surprise"].rolling(window=4, min_periods=2).mean()

        return df
