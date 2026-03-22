"""
Institutional ownership signals from 13F filings.

Source: FD.ai institutional ownership endpoint. All computations operate on
quarterly 13F ownership snapshots — no network calls here.

Signals
-------
LSVHerding     – Lakonishok, Shleifer & Vishny (1992) herding measure
InstitutionalConcentration – change in number of institutional holders
"""

import numpy as np
import pandas as pd


class LSVHerding:
    """
    Lakonishok, Shleifer & Vishny (1992) institutional herding measure.

    H = |fraction_buying - E[fraction_buying]| - AF

    Where:
      fraction_buying = (# institutions increasing / total changes)
      E[fraction_buying] = rolling mean of fraction_buying across all stocks
        (approximated here as the time-series mean when cross-section unavailable)
      AF = adjustment factor for small sample sizes

    Interpretation:
      H > 0.05 = significant herding (all buying or all selling together)
      H > 0 after recent large price move = momentum-chasing (continuation)
      H > 0 after price reversal = value-contrarian (mean reversion)

    Parameters
    ----------
    rolling_window : int
        Quarters to use for estimating E[fraction_buying]. Default 8.
    """

    def __init__(self, rolling_window: int = 8) -> None:
        self.rolling_window = rolling_window

    def compute(self, ownership_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute LSV herding measure from 13F ownership history.

        Parameters
        ----------
        ownership_df : pd.DataFrame
            Must contain columns:
            - period_end (datetime or str)  — 13F filing period
            - total_holders (int)           — total institutional holders
            - holders_increased (int)       — institutions that increased position
            - holders_decreased (int)       — institutions that decreased position

        Returns
        -------
        pd.DataFrame with columns:
            fraction_buying      – proportion of active managers that increased
            expected_buying      – rolling mean of fraction_buying
            herding_measure      – absolute herding (H, unsigned)
            herding_buy_bias     – 1 if net buying herding, -1 if net selling
            herding_high         – 1 if herding_measure > 0.05 (significant)
        """
        df = ownership_df.copy()
        df["period_end"] = pd.to_datetime(df["period_end"])
        df = df.sort_values("period_end").reset_index(drop=True)

        total_changes = df["holders_increased"] + df["holders_decreased"]
        total_safe = total_changes.replace(0, np.nan)

        df["fraction_buying"] = df["holders_increased"] / total_safe

        expected = (
            df["fraction_buying"]
            .rolling(window=self.rolling_window, min_periods=2)
            .mean()
        )
        df["expected_buying"] = expected

        # Adjustment factor: p*(1-p) / (total_changes - 1) — small sample correction
        p = expected.fillna(0.5)
        af = (p * (1 - p)) / (total_safe - 1).replace(0, np.nan)
        af = af.fillna(0)

        raw_h = (df["fraction_buying"] - expected).abs() - af
        df["herding_measure"] = raw_h.clip(lower=0)

        buy_bias = (df["fraction_buying"] > expected).map({True: 1, False: -1})
        df["herding_buy_bias"] = buy_bias.where(df["herding_measure"] > 0, 0)

        df["herding_high"] = (df["herding_measure"] > 0.05).astype(int)

        return df[
            [
                "period_end",
                "fraction_buying",
                "expected_buying",
                "herding_measure",
                "herding_buy_bias",
                "herding_high",
            ]
        ]


class InstitutionalConcentration:
    """
    Tracks change in number of institutional holders — a liquidity/legitimacy signal.

    When the number of institutional holders grows QoQ, liquidity improves
    and the stock enters the "institutionalization" phase (often bullish for
    small/mid caps). When holders shrink, it precedes liquidity deterioration.

    Parameters
    ----------
    lookback_quarters : int
        QoQ lookback for concentration change. Default 1 (sequential quarter).
    """

    def __init__(self, lookback_quarters: int = 1) -> None:
        self.lookback_quarters = lookback_quarters

    def compute(self, ownership_df: pd.DataFrame) -> pd.DataFrame:
        """
        Parameters
        ----------
        ownership_df : pd.DataFrame
            Must contain columns:
            - period_end (datetime or str)
            - total_holders (int)
            - total_shares_held (float, optional)

        Returns
        -------
        pd.DataFrame with columns:
            holder_change       – absolute change in holder count
            holder_change_pct   – % change in holder count
            institutionalizing  – 1 if holder_change_pct > 5%
            de_institutionalizing – 1 if holder_change_pct < -5%
        """
        df = ownership_df.copy()
        df["period_end"] = pd.to_datetime(df["period_end"])
        df = df.sort_values("period_end").reset_index(drop=True)

        prior = df["total_holders"].shift(self.lookback_quarters)
        prior_safe = prior.replace(0, np.nan)

        df["holder_change"] = df["total_holders"] - prior
        df["holder_change_pct"] = df["holder_change"] / prior_safe * 100

        df["institutionalizing"] = (df["holder_change_pct"] > 5).astype(int)
        df["de_institutionalizing"] = (df["holder_change_pct"] < -5).astype(int)

        return df[
            [
                "period_end",
                "total_holders",
                "holder_change",
                "holder_change_pct",
                "institutionalizing",
                "de_institutionalizing",
            ]
        ]
