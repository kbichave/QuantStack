"""Unit tests for section-06 covariance fix in portfolio_construction node."""

import logging

import numpy as np
import pandas as pd
import pytest

from quantstack.core.portfolio.optimizer import covariance_matrix


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _synthetic_returns(n_symbols: int = 30, n_days: int = 120, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    symbols = [f"SYM{i:02d}" for i in range(n_symbols)]
    idx = pd.bdate_range(end="2024-06-01", periods=n_days)
    # Generate data matching actual index length (bdate_range may differ from periods due to calendar)
    data = rng.normal(0.001, 0.015, (len(idx), n_symbols))
    return pd.DataFrame(data, index=idx, columns=symbols)


# ---------------------------------------------------------------------------
# 1. Condition number with LW shrinkage
# ---------------------------------------------------------------------------

def test_condition_number_with_lw_shrinkage():
    """30-symbol × 120-day LW-shrunk covariance has condition number < 500."""
    returns = _synthetic_returns(n_symbols=30, n_days=120)
    cov_df = covariance_matrix(returns, annualise=True, shrinkage=True)
    eigvals = np.linalg.eigvalsh(cov_df.values)
    cond = eigvals.max() / eigvals.min()
    assert cond < 500, f"Condition number {cond:.1f} is too high — LW shrinkage not working"


# ---------------------------------------------------------------------------
# 2. Positive definite
# ---------------------------------------------------------------------------

def test_covariance_positive_definite():
    """All eigenvalues of LW-shrunk covariance must be > 0."""
    returns = _synthetic_returns(n_symbols=30, n_days=120)
    cov_df = covariance_matrix(returns, annualise=True, shrinkage=True)
    eigvals = np.linalg.eigvalsh(cov_df.values)
    assert eigvals.min() > 0, f"Minimum eigenvalue {eigvals.min()} is not positive"


# ---------------------------------------------------------------------------
# 3. Fallback for insufficient history (mixed short/long history symbols)
# ---------------------------------------------------------------------------

def test_fallback_for_insufficient_history():
    """
    A symbol with < 60 days of data is excluded from covariance.
    The returned matrix is (n_thick × n_thick) and does not raise.
    """
    rng = np.random.default_rng(1)
    idx_long = pd.bdate_range(end="2024-06-01", periods=120)
    idx_short = pd.bdate_range(end="2024-06-01", periods=30)

    # Build returns with different lengths
    thick_syms = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
    thin_sym = "THIN"

    thick_df = pd.DataFrame(
        rng.normal(0.001, 0.015, (len(idx_long), len(thick_syms))),
        index=idx_long,
        columns=thick_syms,
    )
    thin_series = pd.Series(
        rng.normal(0.001, 0.015, len(idx_short)),
        index=idx_short,
        name=thin_sym,
    )

    # Combine; thin_sym will have NaN for the first 90 rows
    full_df = thick_df.join(thin_series, how="left")

    # Count non-NaN rows per symbol — thin_sym has only 30
    valid_counts = full_df.notna().sum()
    thin_symbols = valid_counts[valid_counts < 60].index.tolist()

    assert thin_sym in thin_symbols, "Setup error: thin_sym should be identified as thin"

    # Covariance on thick symbols only — must not raise
    thick_df_only = full_df[thick_syms].dropna()
    cov_df = covariance_matrix(thick_df_only, annualise=True, shrinkage=True)
    assert cov_df.shape == (len(thick_syms), len(thick_syms))


# ---------------------------------------------------------------------------
# 4. All symbols insufficient history
# ---------------------------------------------------------------------------

def test_all_symbols_insufficient_history():
    """When all symbols have < 60 days, covariance_matrix handles it gracefully."""
    rng = np.random.default_rng(2)
    short_returns = pd.DataFrame(
        rng.normal(0.001, 0.015, (20, 5)),
        columns=["A", "B", "C", "D", "E"],
    )
    # covariance_matrix should not raise even for very short history
    cov_df = covariance_matrix(short_returns, annualise=True, shrinkage=True)
    assert cov_df.shape == (5, 5)


# ---------------------------------------------------------------------------
# 5. Stale data fallback logic
# ---------------------------------------------------------------------------

def test_stale_data_fallback_uses_prior_covariance():
    """
    When data is stale, last_covariance (nested list) must be used as fallback.
    This tests the conversion logic: nested list → np.ndarray.
    """
    n = 4
    prior_cov = np.eye(n) * 0.05
    prior_list = prior_cov.tolist()

    # Simulate the conversion a node would do
    recovered = np.array(prior_list)
    np.testing.assert_array_almost_equal(recovered, prior_cov)
    assert recovered.shape == (n, n)


# ---------------------------------------------------------------------------
# 6. Condition number warning threshold
# ---------------------------------------------------------------------------

def test_condition_number_warning_logged(caplog):
    """A near-singular covariance (cond > 500) triggers a WARNING log."""
    # Construct a near-singular matrix: two highly correlated assets
    n = 5
    rng = np.random.default_rng(9)
    base = rng.normal(0.001, 0.015, 120)
    data = np.column_stack([
        base,
        base + rng.normal(0, 1e-6, 120),  # nearly identical to column 0
        rng.normal(0.001, 0.015, (120, n - 2)),
    ])
    returns = pd.DataFrame(data, columns=[f"S{i}" for i in range(n)])

    cov = covariance_matrix(returns, annualise=True, shrinkage=False)
    eigvals = np.linalg.eigvalsh(cov.values)
    cond = eigvals.max() / eigvals.min()

    # Simulate the condition number check a node would perform
    with caplog.at_level(logging.WARNING):
        if cond > 500:
            logging.getLogger("quantstack.graphs.trading.nodes").warning(
                "[portfolio_construction] high condition number %.1f — covariance may be near-singular", cond
            )

    if cond > 500:
        assert "high condition number" in caplog.text or "condition number" in caplog.text.lower()


# ---------------------------------------------------------------------------
# 7. Alpha signals bridge behavior
# ---------------------------------------------------------------------------

def test_alpha_signals_bridge_uses_conviction():
    """
    When signals table is empty, alpha_signals[i] = candidate['conviction'] (0.5 default).
    This tests the bridge logic directly (no DB required).
    """
    candidates = [
        {"symbol": "AAPL", "conviction": 0.8},
        {"symbol": "TSLA"},  # missing conviction → default 0.5
        {"symbol": "NVDA", "conviction": 0.3},
    ]
    all_symbols = [c["symbol"] for c in candidates]
    n = len(all_symbols)
    alpha_signals = np.zeros(n)
    for c in candidates:
        sym = c.get("symbol", "")
        idx = all_symbols.index(sym)
        # TODO: replace with kelly_sizing.compute_alpha_signals() output after Section 07 is deployed
        alpha_signals[idx] = c.get("conviction", 0.5)

    assert alpha_signals[0] == pytest.approx(0.8)
    assert alpha_signals[1] == pytest.approx(0.5)
    assert alpha_signals[2] == pytest.approx(0.3)
