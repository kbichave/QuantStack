"""Tests for factor tilts (Section 10.2)."""

from __future__ import annotations

import numpy as np
import pytest

from quantstack.portfolio.factor_model import compute_factor_scores
from quantstack.portfolio.optimizer import optimize_portfolio, PortfolioConstraints


def test_factor_scores_computed_for_universe():
    """Factor scores returned per symbol, normalized to [-1, +1]."""
    data = {
        "AAPL": {"pe_ratio": 15, "return_12m": 0.20, "return_1m": 0.02, "roe": 0.35, "debt_equity": 0.5},
        "MSFT": {"pe_ratio": 30, "return_12m": 0.10, "return_1m": 0.01, "roe": 0.40, "debt_equity": 0.3},
        "GE": {"pe_ratio": 50, "return_12m": -0.05, "return_1m": -0.02, "roe": 0.10, "debt_equity": 2.0},
    }
    scores = compute_factor_scores(data)

    assert len(scores) == 3
    for sym, row in scores.items():
        assert -1.0 <= row["value_score"] <= 1.0
        assert -1.0 <= row["momentum_score"] <= 1.0
        assert -1.0 <= row["quality_score"] <= 1.0
        assert "composite_score" in row


def test_high_quality_momentum_gets_higher_score():
    """Symbol with high quality + momentum gets higher composite than low."""
    data = {
        "GOOD": {"pe_ratio": 12, "return_12m": 0.30, "return_1m": 0.01, "roe": 0.40, "debt_equity": 0.2},
        "BAD": {"pe_ratio": 60, "return_12m": -0.10, "return_1m": -0.05, "roe": 0.05, "debt_equity": 3.0},
    }
    scores = compute_factor_scores(data)

    assert scores["GOOD"]["composite_score"] > scores["BAD"]["composite_score"]


def test_scores_consumed_as_alpha_in_optimizer():
    """Factor scores shift optimizer weights toward high-score symbols."""
    n = 3
    cov = np.eye(n) * 0.01
    current = np.ones(n) / n
    sector_map = {f"SYM{i}": f"S{i}" for i in range(n)}
    strategy_map = {f"SYM{i}": f"St{i}" for i in range(n)}

    # SYM0 has high alpha, SYM2 has low alpha
    alpha = np.array([0.9, 0.5, 0.1])

    constraints = PortfolioConstraints(position_min=0.01, position_max=0.80, turnover_max=0.50)
    weights, _ = optimize_portfolio(
        cov, alpha, current, sector_map, strategy_map, constraints=constraints,
    )

    # SYM0 (highest alpha) should get more weight than SYM2 (lowest)
    assert weights[0] > weights[2]


def test_no_short_positions_created():
    """Even with very negative factor scores, no weight < 0."""
    n = 4
    cov = np.eye(n) * 0.01
    alpha = np.array([0.9, -0.5, -0.8, 0.3])
    current = np.ones(n) / n
    sector_map = {f"SYM{i}": f"S{i}" for i in range(n)}
    strategy_map = {f"SYM{i}": f"St{i}" for i in range(n)}

    constraints = PortfolioConstraints(position_min=0.01, turnover_max=0.50)
    weights, _ = optimize_portfolio(
        cov, alpha, current, sector_map, strategy_map, constraints=constraints,
    )

    assert all(w >= 0 for w in weights)
