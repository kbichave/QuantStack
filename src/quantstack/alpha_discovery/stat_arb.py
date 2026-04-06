"""Statistical arbitrage research tool.

Cointegration scanner and spread monitor for identifying stat arb pair
candidates. Results are presented to the quant_researcher agent for manual
review — no auto-execution.

Limitation: Stat arb at retail scale (5-10 pairs) has too much idiosyncratic
risk, and Alpaca cannot guarantee simultaneous leg execution. Auto-execution
becomes viable only with 50+ pairs and a broker supporting linked orders.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger


def scan_cointegrated_pairs(
    price_data: dict[str, pd.Series],
    pairs: list[tuple[str, str]],
    significance: float = 0.05,
) -> list[dict]:
    """Scan for cointegrated pairs using the Engle-Granger test.

    Applies Bonferroni correction (significance / n_pairs). Returns list of
    dicts with pair, p_value, half_life, current_z_score for passing pairs.

    Args:
        price_data: Dict of symbol -> price series.
        pairs: List of (symbol_a, symbol_b) tuples to test.
        significance: Base significance level (Bonferroni-corrected internally).

    Returns:
        List of dicts for pairs passing the corrected threshold.
    """
    from statsmodels.tsa.stattools import coint

    n_pairs = max(len(pairs), 1)
    corrected_sig = significance / n_pairs  # Bonferroni

    results = []
    for sym_a, sym_b in pairs:
        if sym_a not in price_data or sym_b not in price_data:
            continue

        series_a = price_data[sym_a].dropna()
        series_b = price_data[sym_b].dropna()

        # Align dates
        common = series_a.index.intersection(series_b.index)
        if len(common) < 100:
            continue

        a = series_a.loc[common].values
        b = series_b.loc[common].values

        try:
            _, p_value, _ = coint(a, b)
        except Exception as exc:
            logger.debug(f"[stat_arb] coint failed for {sym_a}/{sym_b}: {exc}")
            continue

        if p_value > corrected_sig:
            continue

        # Compute spread and metrics
        # OLS: b = alpha + beta * a
        beta = np.cov(b, a)[0, 1] / np.var(a) if np.var(a) > 0 else 1.0
        spread = pd.Series(b - beta * a, index=common)

        hl = compute_half_life(spread)
        z = compute_spread_z_score(spread)

        results.append({
            "pair": (sym_a, sym_b),
            "p_value": float(p_value),
            "half_life": hl,
            "current_z_score": z,
            "beta": float(beta),
        })

    return results


def compute_half_life(spread: pd.Series) -> float | None:
    """Fit AR(1) to spread and return half-life in days.

    half_life = -log(2) / log(phi) where phi is the AR(1) coefficient.
    Returns None if spread is not mean-reverting (phi >= 1 or phi <= 0).
    """
    spread = spread.dropna()
    if len(spread) < 20:
        return None

    y = spread.values[1:]
    x = spread.values[:-1]

    # OLS: y = phi * x + epsilon
    if np.var(x) < 1e-12:
        return None

    phi = np.sum(x * y) / np.sum(x * x)

    if phi <= 0 or phi >= 1:
        return None

    half_life = -np.log(2) / np.log(phi)
    return float(half_life)


def compute_spread_z_score(
    spread: pd.Series,
    lookback: int = 60,
) -> float | None:
    """Return z-score of current spread vs trailing window."""
    spread = spread.dropna()
    if len(spread) < 2:
        return None

    window = spread.iloc[-lookback:] if len(spread) >= lookback else spread
    mean = window.mean()
    std = window.std(ddof=1)

    if std < 1e-10:
        return None

    current = spread.iloc[-1]
    return float((current - mean) / std)
