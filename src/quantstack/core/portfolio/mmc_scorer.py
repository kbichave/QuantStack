"""Meta-Model Contribution (MMC) scorer.

Measures how much *new* information a candidate strategy adds beyond what the
existing portfolio already captures.  Used by the promotion gate to block or
penalise strategies that are redundant with the current book.

Reference: Numerai MMC — orthogonalise candidate predictions against the
portfolio signal, then measure covariance with realised returns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MIN_STRATEGIES_FOR_MMC: int = 20
MMC_BLOCK_THRESHOLD: float = 0.70
MMC_PENALTY_THRESHOLD: float = 0.50


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gaussianize(x: np.ndarray) -> np.ndarray:
    """Rank-based inverse-normal (Gaussianisation) transform."""
    ranks = rankdata(x)
    uniform = (ranks - 0.5) / len(x)
    return norm.ppf(uniform)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_portfolio_signal(
    strategy_signals: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Equal-weighted mean signal across all strategies.

    Parameters
    ----------
    strategy_signals:
        Mapping of *strategy_id* -> DataFrame with columns
        ``(signal_date, symbol, signal_value)``.

    Returns
    -------
    pd.DataFrame
        Columns ``(signal_date, symbol, signal_value)`` with the
        equal-weighted mean ``signal_value`` per ``(signal_date, symbol)``.

    Raises
    ------
    ValueError
        If *strategy_signals* is empty.
    """
    if not strategy_signals:
        raise ValueError("strategy_signals must not be empty")

    frames = list(strategy_signals.values())
    combined = pd.concat(frames, ignore_index=True)
    portfolio = (
        combined.groupby(["signal_date", "symbol"], as_index=False)["signal_value"]
        .mean()
    )
    return portfolio


def compute_mmc(
    new_signal: np.ndarray,
    portfolio_signal: np.ndarray,
    realized_returns: np.ndarray,
) -> float:
    """Compute Meta-Model Contribution.

    Steps:
    1. Gaussianize both signal vectors via rank-inverse-normal.
    2. Orthogonalise *new_signal* w.r.t. *portfolio_signal* (OLS residual).
    3. MMC = covariance of the orthogonal component with centred realised
       returns.

    Parameters
    ----------
    new_signal:
        Candidate strategy predictions (1-D, length *n*).
    portfolio_signal:
        Current portfolio predictions (1-D, length *n*).
    realized_returns:
        Actual returns corresponding to the same rows (1-D, length *n*).

    Returns
    -------
    float
        The MMC value.  Positive means the candidate adds information;
        zero means it is redundant.
    """
    p = _gaussianize(new_signal)
    mm = _gaussianize(portfolio_signal)

    # Orthogonalise: regress p on mm, keep residual.
    var_mm = np.var(mm, ddof=0)
    if var_mm == 0:
        # Portfolio signal is constant after Gaussianisation (degenerate).
        p_ortho = p
    else:
        beta = np.cov(p, mm, ddof=0)[0, 1] / var_mm
        p_ortho = p - beta * mm

    # MMC = cov(orthogonal component, centred returns)
    centred_returns = realized_returns - np.mean(realized_returns)
    mmc_value: float = float(np.cov(p_ortho, centred_returns, ddof=0)[0, 1])
    return mmc_value


def get_capital_weight_scalar(correlation: float) -> float:
    """Map portfolio-signal correlation to a capital weight scalar.

    Returns
    -------
    float
        * ``0.0`` when *correlation* > :data:`MMC_BLOCK_THRESHOLD` (blocked).
        * ``0.5`` when :data:`MMC_PENALTY_THRESHOLD` <= *correlation*
          <= :data:`MMC_BLOCK_THRESHOLD`.
        * ``1.0`` when *correlation* < :data:`MMC_PENALTY_THRESHOLD`.
    """
    if correlation > MMC_BLOCK_THRESHOLD:
        return 0.0
    if correlation >= MMC_PENALTY_THRESHOLD:
        return 0.5
    return 1.0
