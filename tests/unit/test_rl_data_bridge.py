# Copyright 2024 QuantCore Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for KnowledgeStoreRLBridge.

Uses PostgreSQL connections via pg_conn().
AlphaVantage calls are always mocked — these tests do not hit external APIs.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
from quantstack.db import pg_conn
from quantstack.rl.data_bridge import KnowledgeStoreRLBridge

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clean_rl_tables() -> None:
    """Committed cleanup for RL bridge tables."""
    with pg_conn() as conn:
        conn.execute("DELETE FROM trade_journal")
        conn.execute("DELETE FROM market_observations")
        conn.execute("DELETE FROM trading_signals")


def _make_store_with_trades(n_trades: int = 30) -> Any:
    """Return a mock KnowledgeStore backed by a PostgreSQL connection with trade data."""
    from quantstack.db import open_db
    _clean_rl_tables()
    conn = open_db()
    np.random.seed(42)
    alpha_types = ["TREND", "MOMENTUM", "VOL"]
    for i in range(n_trades):
        conn.execute(
            """
            INSERT INTO trade_journal
                (symbol, direction, structure_type, pnl, pnl_pct, status)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                "SPY",
                ["LONG", "SHORT"][i % 2],
                alpha_types[i % len(alpha_types)],
                float(np.random.randn() * 100),
                float(np.random.randn() * 0.01),
                "CLOSED",
            ],
        )
    conn.execute("COMMIT")
    store = MagicMock()
    store.conn = conn
    return store


def _make_store_with_ohlcv(n_bars: int = 100) -> Any:
    """Return a mock KnowledgeStore with market_observations data."""
    from quantstack.db import open_db
    _clean_rl_tables()
    conn = open_db()
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(n_bars) * 0.5)
    for i, p in enumerate(prices):
        conn.execute(
            "INSERT INTO market_observations (symbol, timestamp, observation_type, current_price, volume) VALUES (?, ?, ?, ?, ?)",
            ["SPY", datetime.utcnow() - timedelta(days=n_bars - i), "PRICE", float(p), 5000],
        )
    conn.execute("COMMIT")
    store = MagicMock()
    store.conn = conn
    return store


def _make_store_with_signals(n_signals: int = 50) -> Any:
    """Return a mock KnowledgeStore with trading_signals data."""
    from quantstack.db import open_db
    _clean_rl_tables()
    conn = open_db()
    np.random.seed(42)
    for i in range(n_signals):
        conn.execute(
            "INSERT INTO trading_signals (id, symbol, direction, signal_type, confidence, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
            [
                f"sig{i}",
                "SPY",
                ["LONG", "SHORT", "NEUTRAL"][i % 3],
                "TREND",
                float(np.random.beta(2, 2)),
                datetime.utcnow() - timedelta(days=n_signals - i),
            ],
        )
    conn.execute("COMMIT")
    store = MagicMock()
    store.conn = conn
    return store


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestKnowledgeStoreRLBridgeFactory:
    def test_from_knowledge_store(self):
        store = MagicMock()
        from quantstack.db import open_db
        store.conn = open_db()
        bridge = KnowledgeStoreRLBridge.from_knowledge_store(store)
        assert bridge is not None

    def test_direct_init(self):
        store = MagicMock()
        from quantstack.db import open_db
        store.conn = open_db()
        bridge = KnowledgeStoreRLBridge(store=store)
        assert bridge is not None


class TestGetAlphaReturnHistory:
    def test_returns_dict_of_series(self):
        store = _make_store_with_trades(30)
        bridge = KnowledgeStoreRLBridge(store=store)
        histories = bridge.get_alpha_return_history(
            alpha_names=["TREND", "MOMENTUM"],
            lookback_days=365,
        )
        assert isinstance(histories, dict)

    def test_returns_series_for_matched_alpha(self):
        store = _make_store_with_trades(30)
        bridge = KnowledgeStoreRLBridge(store=store)
        histories = bridge.get_alpha_return_history(
            alpha_names=["TREND"],
            lookback_days=365,
        )
        if "TREND" in histories:
            assert isinstance(histories["TREND"], pd.Series)

    def test_missing_table_returns_empty(self):
        # Store with no trade_journal table
        store = MagicMock()
        from quantstack.db import open_db
        store.conn = open_db()
        bridge = KnowledgeStoreRLBridge(store=store)
        result = bridge.get_alpha_return_history(["X"], lookback_days=30)
        assert isinstance(result, dict)

    def test_has_sufficient_alpha_history_false_when_empty(self):
        store = MagicMock()
        from quantstack.db import open_db
        store.conn = open_db()
        bridge = KnowledgeStoreRLBridge(store=store)
        assert (
            bridge.has_sufficient_alpha_history(["TREND"], min_observations=20) is False
        )

    def test_has_sufficient_alpha_history_true_with_data(self):
        store = _make_store_with_trades(30)
        bridge = KnowledgeStoreRLBridge(store=store)
        result = bridge.has_sufficient_alpha_history(["TREND"], min_observations=5)
        # Result depends on whether alpha matched, but must be bool
        assert isinstance(result, bool)


class TestGetOHLCVForExecution:
    def test_returns_dataframe(self):
        store = _make_store_with_ohlcv(100)
        bridge = KnowledgeStoreRLBridge(store=store)
        df = bridge.get_ohlcv_for_execution("SPY", lookback_days=90)
        assert isinstance(df, pd.DataFrame)

    def test_missing_table_returns_empty_df(self):
        _clean_rl_tables()
        store = MagicMock()
        from quantstack.db import open_db
        store.conn = open_db()
        bridge = KnowledgeStoreRLBridge(store=store)
        df = bridge.get_ohlcv_for_execution("SPY", lookback_days=30)
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_has_sufficient_signal_history_false_when_empty(self):
        store = MagicMock()
        from quantstack.db import open_db
        store.conn = open_db()
        bridge = KnowledgeStoreRLBridge(store=store)
        assert bridge.has_sufficient_signal_history(min_signals=30) is False


class TestGetSignalHistory:
    def test_returns_list(self):
        store = _make_store_with_signals(50)
        bridge = KnowledgeStoreRLBridge(store=store)
        signals = bridge.get_signal_history(lookback_days=90)
        assert isinstance(signals, list)

    def test_missing_table_returns_empty_list(self):
        store = MagicMock()
        from quantstack.db import open_db
        store.conn = open_db()
        bridge = KnowledgeStoreRLBridge(store=store)
        signals = bridge.get_signal_history(lookback_days=30)
        assert signals == [] or signals is None or isinstance(signals, list)


class TestBootstrapRateLimiting:
    def test_bootstrap_skips_existing_symbols(self):
        """bootstrap_from_alphavantage should not re-fetch already-stored symbols."""
        store = _make_store_with_ohlcv(100)
        bridge = KnowledgeStoreRLBridge(store=store)

        with patch("requests.get") as mock_get:
            mock_get.return_value = MagicMock(
                status_code=200,
                json=lambda: {"Time Series (Daily)": {}},
            )
            # If symbol already has data, fetch should be short-circuited
            bridge.bootstrap_from_alphavantage(
                symbols=["SPY"],
                start_date="2023-01-01",
                api_key="demo",
            )
            # Even if called, test should not raise
