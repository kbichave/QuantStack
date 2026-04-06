"""Factor scoring model for portfolio tilts.

Computes value, momentum, and quality factor scores per symbol, normalized
cross-sectionally to [-1, +1]. Composite score is an equal-weighted average.

Factor tilts only adjust long-side allocation — the optimizer enforces
weight >= 0. Negative scores result in underweight, not short.
"""

from __future__ import annotations

import numpy as np


def compute_factor_scores(
    symbol_data: dict[str, dict],
) -> dict[str, dict[str, float]]:
    """Compute factor scores for a universe of symbols.

    Args:
        symbol_data: Dict of symbol -> fundamental data dict. Each dict has:
            pe_ratio, return_12m, return_1m, roe, debt_equity.

    Returns:
        Dict of symbol -> {value_score, momentum_score, quality_score, composite_score}.
        All scores normalized to [-1, +1].
    """
    if not symbol_data:
        return {}

    symbols = list(symbol_data.keys())
    n = len(symbols)

    # Extract raw values
    pe_ratios = np.array([symbol_data[s].get("pe_ratio", 0) or 0 for s in symbols], dtype=float)
    return_12m = np.array([symbol_data[s].get("return_12m", 0) or 0 for s in symbols], dtype=float)
    return_1m = np.array([symbol_data[s].get("return_1m", 0) or 0 for s in symbols], dtype=float)
    roe = np.array([symbol_data[s].get("roe", 0) or 0 for s in symbols], dtype=float)
    debt_equity = np.array([symbol_data[s].get("debt_equity", 0) or 0 for s in symbols], dtype=float)

    # Value: earnings yield (inverse PE) — higher is cheaper
    earnings_yield = np.where(pe_ratios > 0, 1.0 / pe_ratios, 0)
    value_raw = earnings_yield

    # Momentum: 12-month return minus 1-month return (classic 12-1)
    momentum_raw = return_12m - return_1m

    # Quality: ROE minus debt/equity (higher ROE + lower leverage = higher quality)
    quality_raw = roe - 0.1 * debt_equity

    # Cross-sectional z-score normalization, clipped to [-1, +1]
    def _normalize(arr: np.ndarray) -> np.ndarray:
        if n < 2:
            return np.zeros(n)
        std = arr.std(ddof=1)
        if std < 1e-10:
            return np.zeros(n)
        z = (arr - arr.mean()) / std
        return np.clip(z, -1.0, 1.0)

    value_scores = _normalize(value_raw)
    momentum_scores = _normalize(momentum_raw)
    quality_scores = _normalize(quality_raw)

    result = {}
    for i, sym in enumerate(symbols):
        composite = (value_scores[i] + momentum_scores[i] + quality_scores[i]) / 3.0
        result[sym] = {
            "value_score": float(value_scores[i]),
            "momentum_score": float(momentum_scores[i]),
            "quality_score": float(quality_scores[i]),
            "composite_score": float(composite),
        }

    return result
