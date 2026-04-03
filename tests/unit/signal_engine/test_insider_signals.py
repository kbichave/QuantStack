"""Tests for insider trading signal collector (Section 09)."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from quantstack.signal_engine.collectors.insider_signals import (
    compute_insider_score,
    detect_cluster_buy,
    detect_csuite_buy,
    detect_unusual_size,
)


def _make_transactions(
    n: int = 5,
    names: list[str] | None = None,
    titles: list[str] | None = None,
    types: list[str] | None = None,
    shares: list[int] | None = None,
    prices: list[float] | None = None,
    days_ago: list[int] | None = None,
) -> list[dict]:
    """Build a list of insider transactions."""
    now = datetime.utcnow()
    if names is None:
        names = [f"Insider{i}" for i in range(n)]
    if titles is None:
        titles = ["Director"] * n
    if types is None:
        types = ["P"] * n  # P=purchase
    if shares is None:
        shares = [1000] * n
    if prices is None:
        prices = [50.0] * n
    if days_ago is None:
        days_ago = list(range(1, n + 1))

    return [
        {
            "insider_name": names[i],
            "title": titles[i],
            "transaction_type": types[i],
            "shares": shares[i],
            "price": prices[i],
            "date": (now - timedelta(days=days_ago[i])).isoformat(),
            "value": shares[i] * prices[i],
        }
        for i in range(n)
    ]


def test_cluster_buy_detected():
    """3+ distinct insiders buying within 30 days -> cluster_buy."""
    txns = _make_transactions(
        n=4,
        names=["Alice", "Bob", "Carol", "Dave"],
        types=["P", "P", "P", "P"],
        days_ago=[5, 10, 15, 20],
    )
    result = detect_cluster_buy(txns, window_days=30, min_insiders=3)
    assert result is True


def test_no_cluster_buy_insufficient_insiders():
    """Only 2 distinct buyers -> no cluster."""
    txns = _make_transactions(
        n=3,
        names=["Alice", "Alice", "Bob"],
        types=["P", "P", "P"],
        days_ago=[5, 10, 15],
    )
    result = detect_cluster_buy(txns, window_days=30, min_insiders=3)
    assert result is False


def test_csuite_buy_detected():
    """CEO purchase > $100K -> csuite_buy."""
    txns = _make_transactions(
        n=1,
        names=["John CEO"],
        titles=["CEO"],
        types=["P"],
        shares=[3000],
        prices=[50.0],
        days_ago=[5],
    )
    result = detect_csuite_buy(txns, min_value=100_000)
    assert result is True


def test_csuite_buy_below_threshold():
    """CEO purchase < $100K -> no signal."""
    txns = _make_transactions(
        n=1,
        names=["John CEO"],
        titles=["CEO"],
        types=["P"],
        shares=[100],
        prices=[50.0],
        days_ago=[5],
    )
    result = detect_csuite_buy(txns, min_value=100_000)
    assert result is False


def test_unusual_size_detected():
    """Transaction > 10x average -> unusual_size."""
    txns = _make_transactions(
        n=5,
        names=["Alice"] * 5,
        shares=[100, 100, 100, 100, 5000],
        days_ago=[90, 70, 50, 30, 5],
    )
    result = detect_unusual_size(txns, multiplier=10)
    assert result is True


def test_insider_score_all_buys():
    """All buys -> score near +1."""
    txns = _make_transactions(n=5, types=["P"] * 5)
    score = compute_insider_score(txns)
    assert score > 0.5


def test_insider_score_all_sells():
    """All sells -> score near -1."""
    txns = _make_transactions(n=5, types=["S"] * 5)
    score = compute_insider_score(txns)
    assert score < -0.3


def test_insider_score_mixed():
    """Mixed buys/sells -> score near 0."""
    txns = _make_transactions(
        n=4,
        types=["P", "S", "P", "S"],
        shares=[1000, 1000, 1000, 1000],
    )
    score = compute_insider_score(txns)
    assert -0.5 <= score <= 0.5


def test_insider_score_empty():
    """No transactions -> score 0."""
    score = compute_insider_score([])
    assert score == 0.0


def test_option_exercises_excluded_from_sells():
    """Transaction type 'A' (award/exercise) excluded from sell scoring."""
    txns = _make_transactions(
        n=3,
        types=["A", "A", "P"],  # 2 exercises + 1 buy
        shares=[5000, 5000, 1000],
    )
    score = compute_insider_score(txns)
    # Should be positive (exercises excluded, only buy counted)
    assert score > 0
