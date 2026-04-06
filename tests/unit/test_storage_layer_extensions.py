"""Tests for PgDataStore extensions (AV data expansion Section 05).

Validates the 5 new methods:
- load_options_volume_summary
- save_put_call_ratio
- load_put_call_ratio
- update_delisting_status
- get_delisted_symbols
"""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from quantstack.data.pg_storage import PgDataStore


@pytest.fixture()
def store() -> PgDataStore:
    return PgDataStore()


def _mock_conn_ctx(mock_conn: MagicMock):
    """Return a context-manager that yields *mock_conn*."""
    from contextlib import contextmanager

    @contextmanager
    def _ctx():
        yield mock_conn

    return _ctx


# ======================================================================
# load_options_volume_summary
# ======================================================================


class TestLoadOptionsVolumeSummary:
    @patch("quantstack.data.pg_storage.pg_conn")
    def test_aggregates_put_call_volume(self, mock_pg_conn, store):
        """Verify the SQL uses CASE WHEN for put/call split and returns correct columns."""
        mock_conn = MagicMock()
        mock_pg_conn.return_value = _mock_conn_ctx(mock_conn)()

        result_df = pd.DataFrame(
            {
                "date": [date(2025, 1, 2), date(2025, 1, 3)],
                "put_volume": [1500, 2000],
                "call_volume": [3000, 2500],
            }
        )
        with patch.object(PgDataStore, "_df_from_query", return_value=result_df):
            df = store.load_options_volume_summary(
                "AAPL", date(2025, 1, 1), date(2025, 1, 5)
            )

        assert list(df.columns) == ["date", "put_volume", "call_volume"]
        assert len(df) == 2

    @patch("quantstack.data.pg_storage.pg_conn")
    def test_returns_empty_df_with_correct_columns(self, mock_pg_conn, store):
        """When no data exists, return empty DataFrame with expected schema."""
        mock_conn = MagicMock()
        mock_pg_conn.return_value = _mock_conn_ctx(mock_conn)()

        with patch.object(
            PgDataStore,
            "_df_from_query",
            return_value=pd.DataFrame(),
        ):
            df = store.load_options_volume_summary(
                "ZZZZ", date(2025, 1, 1), date(2025, 1, 5)
            )

        assert df.empty
        assert list(df.columns) == ["date", "put_volume", "call_volume"]

    @patch("quantstack.data.pg_storage.pg_conn")
    def test_passes_correct_params(self, mock_pg_conn, store):
        """Symbol and date range must appear in the query params."""
        mock_conn = MagicMock()
        mock_pg_conn.return_value = _mock_conn_ctx(mock_conn)()

        with patch.object(
            PgDataStore, "_df_from_query", return_value=pd.DataFrame()
        ) as mock_query:
            store.load_options_volume_summary(
                "TSLA", date(2025, 3, 1), date(2025, 3, 31)
            )

        args = mock_query.call_args
        query_str = args[0][0]
        params = args[0][1]
        assert "TSLA" in params
        assert date(2025, 3, 1) in params
        assert date(2025, 3, 31) in params
        # Verify CASE WHEN aggregation is in the SQL
        assert "CASE WHEN" in query_str.upper() or "case when" in query_str.lower()


# ======================================================================
# save_put_call_ratio
# ======================================================================


class TestSavePutCallRatio:
    @patch("quantstack.data.pg_storage.pg_conn")
    def test_upserts_on_conflict(self, mock_pg_conn, store):
        """Verify ON CONFLICT (symbol, date, source) DO UPDATE is used."""
        mock_conn = MagicMock()
        mock_pg_conn.return_value = _mock_conn_ctx(mock_conn)()

        df = pd.DataFrame(
            {
                "symbol": ["AAPL", "AAPL"],
                "date": [date(2025, 1, 2), date(2025, 1, 3)],
                "put_volume": [1500, 2000],
                "call_volume": [3000, 2500],
                "pcr": [0.50, 0.80],
                "source": ["alpha_vantage", "alpha_vantage"],
            }
        )

        with patch("quantstack.data.pg_storage.psycopg2.extras.execute_values") as mock_ev:
            count = store.save_put_call_ratio(df)

        assert count == 2
        sql_arg = mock_ev.call_args[0][1]
        assert "ON CONFLICT" in sql_arg
        assert "symbol" in sql_arg
        assert "source" in sql_arg

    @patch("quantstack.data.pg_storage.pg_conn")
    def test_returns_row_count(self, mock_pg_conn, store):
        mock_conn = MagicMock()
        mock_pg_conn.return_value = _mock_conn_ctx(mock_conn)()

        df = pd.DataFrame(
            {
                "symbol": ["SPY"],
                "date": [date(2025, 1, 2)],
                "put_volume": [5000],
                "call_volume": [4000],
                "pcr": [1.25],
                "source": ["computed"],
            }
        )

        with patch("quantstack.data.pg_storage.psycopg2.extras.execute_values"):
            count = store.save_put_call_ratio(df)

        assert count == 1

    def test_empty_dataframe_returns_zero(self, store):
        """Empty input must return 0 without touching the DB."""
        count = store.save_put_call_ratio(pd.DataFrame())
        assert count == 0


# ======================================================================
# load_put_call_ratio
# ======================================================================


class TestLoadPutCallRatio:
    @patch("quantstack.data.pg_storage.pg_conn")
    def test_filters_by_symbol_and_date_range(self, mock_pg_conn, store):
        mock_conn = MagicMock()
        mock_pg_conn.return_value = _mock_conn_ctx(mock_conn)()

        result_df = pd.DataFrame(
            {
                "symbol": ["AAPL"],
                "date": [date(2025, 1, 2)],
                "put_volume": [1500],
                "call_volume": [3000],
                "pcr": [0.5],
                "source": ["alpha_vantage"],
            }
        )

        with patch.object(
            PgDataStore, "_df_from_query", return_value=result_df
        ) as mock_query:
            df = store.load_put_call_ratio(
                "AAPL", date(2025, 1, 1), date(2025, 1, 5)
            )

        assert len(df) == 1
        params = mock_query.call_args[0][1]
        assert "AAPL" in params
        assert date(2025, 1, 1) in params
        assert date(2025, 1, 5) in params

    @patch("quantstack.data.pg_storage.pg_conn")
    def test_returns_empty_for_unknown_symbol(self, mock_pg_conn, store):
        mock_conn = MagicMock()
        mock_pg_conn.return_value = _mock_conn_ctx(mock_conn)()

        with patch.object(
            PgDataStore, "_df_from_query", return_value=pd.DataFrame()
        ):
            df = store.load_put_call_ratio(
                "ZZZZ", date(2025, 1, 1), date(2025, 1, 5)
            )

        assert df.empty


# ======================================================================
# update_delisting_status
# ======================================================================


class TestUpdateDelistingStatus:
    @patch("quantstack.data.pg_storage.pg_conn")
    def test_sets_delisted_at(self, mock_pg_conn, store):
        """Verify UPDATE uses WHERE symbol = ANY(%s)."""
        mock_conn = MagicMock()
        mock_pg_conn.return_value = _mock_conn_ctx(mock_conn)()

        # Simulate cursor.rowcount via the PgConnection wrapper
        mock_conn.execute.return_value = mock_conn
        mock_conn._cur = MagicMock()
        mock_conn._cur.rowcount = 3

        count = store.update_delisting_status(
            ["ACME", "FAIL", "GONE"], date(2025, 6, 1)
        )

        assert count == 3
        sql_arg = mock_conn.execute.call_args[0][0]
        assert "ANY" in sql_arg.upper()
        params = mock_conn.execute.call_args[0][1]
        assert date(2025, 6, 1) in params

    @patch("quantstack.data.pg_storage.pg_conn")
    def test_returns_updated_count(self, mock_pg_conn, store):
        mock_conn = MagicMock()
        mock_pg_conn.return_value = _mock_conn_ctx(mock_conn)()
        mock_conn.execute.return_value = mock_conn
        mock_conn._cur = MagicMock()
        mock_conn._cur.rowcount = 0

        count = store.update_delisting_status(["UNKNOWN"], date(2025, 1, 1))
        assert count == 0

    def test_empty_symbols_returns_zero(self, store):
        """Empty symbol list must return 0 without touching the DB."""
        count = store.update_delisting_status([], date(2025, 1, 1))
        assert count == 0


# ======================================================================
# get_delisted_symbols
# ======================================================================


class TestGetDelistedSymbols:
    @patch("quantstack.data.pg_storage.pg_conn")
    def test_returns_delisted_symbols(self, mock_pg_conn, store):
        mock_conn = MagicMock()
        mock_pg_conn.return_value = _mock_conn_ctx(mock_conn)()

        mock_conn.execute.return_value = mock_conn
        mock_conn.fetchall.return_value = [("ACME",), ("FAIL",)]

        result = store.get_delisted_symbols()
        assert result == ["ACME", "FAIL"]

        sql_arg = mock_conn.execute.call_args[0][0]
        assert "delisted_at IS NOT NULL" in sql_arg

    @patch("quantstack.data.pg_storage.pg_conn")
    def test_returns_empty_list_when_none_delisted(self, mock_pg_conn, store):
        mock_conn = MagicMock()
        mock_pg_conn.return_value = _mock_conn_ctx(mock_conn)()

        mock_conn.execute.return_value = mock_conn
        mock_conn.fetchall.return_value = []

        result = store.get_delisted_symbols()
        assert result == []
