"""
Insider trading signals from Form 4 filings.

Source: FD.ai insider trades endpoint. All computations operate on a
DataFrame of Form 4 transactions — no network calls here.

Signals
-------
InsiderSignals – cluster buy detection, adjusted buy/sell ratio,
                 ownership stake change
"""

import numpy as np
import pandas as pd


class InsiderSignals:
    """
    Insider transaction signals from Form 4 filings.

    Cluster Buy (Lakonishok & Lee 2001): when 3+ distinct insiders (by role)
    make open-market purchases within a rolling window, it signals high
    conviction — insiders rarely co-ordinate unless they independently see
    value. This is one of the most durable insider signals in empirical research.

    Parameters
    ----------
    cluster_window_days : int
        Rolling lookback for cluster buy detection. Default 30.
    cluster_min_insiders : int
        Minimum distinct insiders required to trigger cluster signal. Default 3.
    """

    def __init__(
        self,
        cluster_window_days: int = 30,
        cluster_min_insiders: int = 3,
    ) -> None:
        self.cluster_window_days = cluster_window_days
        self.cluster_min_insiders = cluster_min_insiders

    def compute(self, transactions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute insider signals from a Form 4 transaction history.

        Parameters
        ----------
        transactions_df : pd.DataFrame
            Must contain columns:
            - transaction_date (datetime or str)
            - transaction_type (str): 'P' = open-market purchase, 'S' = sale
            - shares (float): shares transacted (positive)
            - price (float): price per share
            - insider_name (str): reporting person
            - insider_role (str): CEO, CFO, Director, etc.
            - is_plan_trade (bool): True if under 10b5-1 plan (exclude from ratio)

        Returns
        -------
        pd.DataFrame indexed by transaction_date with columns:
            buy_value           – total $ value of open-market buys in window
            sell_value          – total $ value of non-plan, non-option sells in window
            distinct_buyers     – count of distinct insiders buying in window
            cluster_buy         – 1 when distinct_buyers >= cluster_min_insiders
            adj_buy_sell_ratio  – buy_value / (buy_value + sell_value)
            stake_change_pct    – rolling % change in total reported holdings
        """
        df = transactions_df.copy()
        df["transaction_date"] = pd.to_datetime(df["transaction_date"])
        df = df.sort_values("transaction_date").reset_index(drop=True)

        df["dollar_value"] = df["shares"] * df["price"]

        # Open-market purchases only
        buys = df[df["transaction_type"] == "P"].copy()
        # Non-plan sells (exclude 10b5-1 automatic plans and option exercises)
        sells = df[
            (df["transaction_type"] == "S") &
            (~df.get("is_plan_trade", pd.Series(False, index=df.index)).fillna(False))
        ].copy()

        # Build per-day summary
        buy_daily = (
            buys.groupby("transaction_date")
            .agg(
                buy_value=("dollar_value", "sum"),
                distinct_buyers=("insider_name", "nunique"),
            )
            .reindex(df["transaction_date"].unique())
            .fillna(0)
            .sort_index()
        )

        sell_daily = (
            sells.groupby("transaction_date")["dollar_value"]
            .sum()
            .reindex(buy_daily.index)
            .fillna(0)
        )

        buy_daily["sell_value"] = sell_daily

        window = f"{self.cluster_window_days}D"
        buy_daily["buy_value_roll"] = (
            buy_daily["buy_value"].rolling(window, min_periods=1).sum()
        )
        buy_daily["sell_value_roll"] = (
            buy_daily["sell_value"].rolling(window, min_periods=1).sum()
        )
        buy_daily["distinct_buyers_roll"] = (
            buy_daily["distinct_buyers"].rolling(window, min_periods=1).sum()
        )

        buy_daily["cluster_buy"] = (
            buy_daily["distinct_buyers_roll"] >= self.cluster_min_insiders
        ).astype(int)

        total_side = buy_daily["buy_value_roll"] + buy_daily["sell_value_roll"]
        total_safe = total_side.replace(0, np.nan)
        buy_daily["adj_buy_sell_ratio"] = buy_daily["buy_value_roll"] / total_safe

        # Ownership stake change (requires total_shares_held column if available)
        if "total_shares_held" in df.columns:
            held_daily = (
                df.groupby("transaction_date")["total_shares_held"]
                .last()
                .reindex(buy_daily.index)
                .ffill()
            )
            prior_held = held_daily.shift(1).replace(0, np.nan)
            buy_daily["stake_change_pct"] = (held_daily - held_daily.shift(1)) / prior_held * 100
        else:
            buy_daily["stake_change_pct"] = np.nan

        return buy_daily[[
            "buy_value_roll",
            "sell_value_roll",
            "distinct_buyers_roll",
            "cluster_buy",
            "adj_buy_sell_ratio",
            "stake_change_pct",
        ]].rename(columns={
            "buy_value_roll": "buy_value",
            "sell_value_roll": "sell_value",
            "distinct_buyers_roll": "distinct_buyers",
        })
