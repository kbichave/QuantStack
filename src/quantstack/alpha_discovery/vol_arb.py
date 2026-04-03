"""Volatility arbitrage: trade the IV-RV spread.

Computes the spread between implied and realized volatility and generates
signals when the spread deviates significantly from its historical mean.

Limitations:
- Alpha Vantage options data is EOD only (no intraday IV surface).
- Delta hedging incurs transaction costs; weekly adjustments recommended.
- Calibration is monthly; intra-month regime shifts can cause stale params.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger


def compute_vol_spread(iv_30d: float, rv_30d: float) -> float:
    """Return IV - RV spread."""
    return iv_30d - rv_30d


def calibrate_vol_params(
    spread_history: pd.Series,
    window: int = 252,
) -> dict:
    """Compute mean and std from trailing spread history.

    Returns dict with mean, std, n_observations.
    """
    tail = spread_history.dropna().iloc[-window:]
    if len(tail) < 60:
        return {"mean": 0.0, "std": 0.0, "n_observations": len(tail)}

    return {
        "mean": float(tail.mean()),
        "std": float(tail.std(ddof=1)),
        "n_observations": len(tail),
    }


def generate_vol_signal(
    current_spread: float,
    params: dict,
    z_threshold: float = 1.0,
) -> str | None:
    """Return 'sell_vol', 'buy_vol', or None based on z-score of spread.

    sell_vol: spread > mean + z_threshold * std (IV is rich)
    buy_vol: spread < mean - z_threshold * std (IV is cheap)
    """
    mean = params.get("mean", 0)
    std = params.get("std", 0)

    if std <= 0:
        return None

    z_score = (current_spread - mean) / std

    if z_score > z_threshold:
        return "sell_vol"
    elif z_score < -z_threshold:
        return "buy_vol"

    return None


def select_structure(
    signal: str,
    equity: float,
    max_loss_pct: float = 0.02,
) -> dict:
    """Return options structure spec with defined max loss.

    sell_vol -> iron_condor (premium collection, defined risk)
    buy_vol -> straddle (long gamma, benefits from vol expansion)
    """
    max_loss = equity * max_loss_pct

    if signal == "sell_vol":
        return {
            "type": "iron_condor",
            "max_loss": max_loss,
            "description": "Sell IV premium via iron condor",
        }
    elif signal == "buy_vol":
        return {
            "type": "straddle",
            "max_loss": max_loss,
            "description": "Buy gamma via straddle",
        }
    else:
        return {
            "type": "none",
            "max_loss": 0,
            "description": "No signal",
        }
