"""Execution algo feedback loop.

Reads TCA results, groups by (symbol_adv_bucket, algo_used, time_bucket),
runs t-tests between algo pairs, and identifies statistically significant
winners for data-driven algo selection.

Why t-test: Appropriate for comparing two means when data is approximately
normal (shortfall distributions tend to be). At 30-100 fills per group,
it provides sufficient power without unnecessary complexity.
"""

from __future__ import annotations

import pandas as pd
from loguru import logger
from scipy import stats


def aggregate_tca_results(df: pd.DataFrame) -> pd.DataFrame:
    """Group TCA results by (symbol_adv_bucket, algo_used, time_bucket).

    Returns DataFrame with mean_shortfall, std_shortfall, count per group.
    """
    required = {"symbol_adv_bucket", "algo_used", "time_bucket", "shortfall_vs_arrival_bps"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"Missing columns: {missing}")

    return (
        df.groupby(["symbol_adv_bucket", "algo_used", "time_bucket"])
        .agg(
            mean_shortfall=("shortfall_vs_arrival_bps", "mean"),
            std_shortfall=("shortfall_vs_arrival_bps", "std"),
            count=("shortfall_vs_arrival_bps", "count"),
        )
        .reset_index()
    )


def find_best_algo(
    df: pd.DataFrame,
    symbol_adv_bucket: str,
    time_bucket: str,
    min_fills: int = 30,
) -> dict | None:
    """Compare algos for a given (symbol_adv_bucket, time_bucket) via t-test.

    Returns dict with preferred_algo, p_value, mean_shortfall if significant,
    or None if insufficient data or no significant difference.
    """
    subset = df[
        (df["symbol_adv_bucket"] == symbol_adv_bucket)
        & (df["time_bucket"] == time_bucket)
    ]

    if subset.empty:
        return None

    algos = subset["algo_used"].unique()
    if len(algos) < 2:
        return None

    best_result = None

    for i, algo_a in enumerate(algos):
        for algo_b in algos[i + 1:]:
            data_a = subset[subset["algo_used"] == algo_a]["shortfall_vs_arrival_bps"]
            data_b = subset[subset["algo_used"] == algo_b]["shortfall_vs_arrival_bps"]

            if len(data_a) < min_fills or len(data_b) < min_fills:
                continue

            t_stat, p_value = stats.ttest_ind(data_a, data_b)

            if p_value < 0.05:
                # Lower mean shortfall is better
                if data_a.mean() < data_b.mean():
                    preferred, comparison = algo_a, algo_b
                    mean_shortfall = data_a.mean()
                    sample_size = len(data_a)
                else:
                    preferred, comparison = algo_b, algo_a
                    mean_shortfall = data_b.mean()
                    sample_size = len(data_b)

                candidate = {
                    "preferred_algo": preferred,
                    "comparison_algo": comparison,
                    "mean_shortfall_bps": float(mean_shortfall),
                    "p_value": float(p_value),
                    "sample_size": int(sample_size),
                    "symbol_adv_bucket": symbol_adv_bucket,
                    "time_bucket": time_bucket,
                }

                if best_result is None or p_value < best_result["p_value"]:
                    best_result = candidate

    return best_result
