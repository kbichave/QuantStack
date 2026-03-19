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
