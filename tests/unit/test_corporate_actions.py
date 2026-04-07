"""Unit tests for corporate actions monitor."""

from __future__ import annotations

import asyncio
import json
import math
from contextlib import contextmanager
from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from quantstack.data.corporate_actions import (
    CIKMapper,
    CorporateAction,
    SplitAdjustment,
    _8K_ITEM_MAP,
    _parse_date_or_none,
    _store_actions,
    _flag_ma_events,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(coro):
    """Run an async coroutine synchronously."""
    return asyncio.get_event_loop().run_until_complete(coro)


def _make_mock_conn(fetchone_val=None, rows=None, rowcount=1):
    conn = MagicMock()
    cursor = MagicMock()
    cursor.fetchone.return_value = fetchone_val
    cursor.fetchall.return_value = rows or []
    cursor.rowcount = rowcount
    conn.execute.return_value = cursor

    @contextmanager
    def _mock_db():
        yield conn

    return conn, _mock_db


def _mock_av_client():
    """Create a mock AlphaVantageClient."""
    client = MagicMock()
    client.api_key = "test_key"
    client.base_url = "https://www.alphavantage.co/query"
    client._wait_for_rate_limit = MagicMock()
    return client


def _mock_httpx_get(response_data):
    """Create a mock httpx.AsyncClient that returns response_data from .get()."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = response_data
    mock_resp.raise_for_status = MagicMock()

    mock_http = AsyncMock()
    mock_http.__aenter__ = AsyncMock(return_value=mock_http)
    mock_http.__aexit__ = AsyncMock(return_value=False)
    mock_http.get = AsyncMock(return_value=mock_resp)
    return mock_http


# ---------------------------------------------------------------------------
# _parse_date_or_none
# ---------------------------------------------------------------------------


class TestParseDateOrNone:
    def test_valid_date(self):
        assert _parse_date_or_none("2024-01-15") == date(2024, 1, 15)

    def test_none_string(self):
        assert _parse_date_or_none("None") is None

    def test_null_string(self):
        assert _parse_date_or_none("null") is None

    def test_empty_string(self):
        assert _parse_date_or_none("") is None

    def test_actual_none(self):
        assert _parse_date_or_none(None) is None

    def test_invalid_format(self):
        assert _parse_date_or_none("Jan 15 2024") is None


# ---------------------------------------------------------------------------
# fetch_av_dividends
# ---------------------------------------------------------------------------


class TestFetchAVDividends:
    def test_parses_av_response(self):
        from quantstack.data.corporate_actions import fetch_av_dividends

        av_data = {
            "data": [
                {"ex_dividend_date": "2024-01-15", "declaration_date": "2024-01-02", "amount": "0.24"},
                {"ex_dividend_date": "2024-04-12", "declaration_date": "None", "amount": "0.25"},
            ]
        }
        mock_http = _mock_httpx_get(av_data)

        with patch("quantstack.data.fetcher.AlphaVantageClient", return_value=_mock_av_client()), \
             patch("quantstack.data.corporate_actions.httpx.AsyncClient", return_value=mock_http):
            actions = _run(fetch_av_dividends("AAPL"))

        assert len(actions) == 2
        assert actions[0].event_type == "dividend"
        assert actions[0].source == "alpha_vantage"
        assert actions[0].effective_date == date(2024, 1, 15)
        assert actions[0].announcement_date == date(2024, 1, 2)
        assert actions[1].announcement_date is None

    def test_handles_empty_response(self):
        from quantstack.data.corporate_actions import fetch_av_dividends

        mock_http = _mock_httpx_get({"data": []})

        with patch("quantstack.data.fetcher.AlphaVantageClient", return_value=_mock_av_client()), \
             patch("quantstack.data.corporate_actions.httpx.AsyncClient", return_value=mock_http):
            actions = _run(fetch_av_dividends("AAPL"))

        assert actions == []


# ---------------------------------------------------------------------------
# fetch_av_splits
# ---------------------------------------------------------------------------


class TestFetchAVSplits:
    def test_parses_split_factor(self):
        from quantstack.data.corporate_actions import fetch_av_splits

        av_data = {"LastSplitFactor": "4:1", "LastSplitDate": "2020-08-31"}
        mock_http = _mock_httpx_get(av_data)

        with patch("quantstack.data.fetcher.AlphaVantageClient", return_value=_mock_av_client()), \
             patch("quantstack.data.corporate_actions.httpx.AsyncClient", return_value=mock_http):
            actions = _run(fetch_av_splits("AAPL"))

        assert len(actions) == 1
        assert actions[0].event_type == "split"
        assert actions[0].raw_payload["split_ratio"] == 4.0
        assert actions[0].effective_date == date(2020, 8, 31)


# ---------------------------------------------------------------------------
# fetch_edgar_8k_events
# ---------------------------------------------------------------------------


class TestFetchEdgar8KEvents:
    def test_parses_target_items(self):
        from quantstack.data.corporate_actions import fetch_edgar_8k_events

        mock_data = {
            "filings": {
                "recent": {
                    "form": ["8-K", "10-Q", "8-K"],
                    "filingDate": ["2024-03-15", "2024-03-10", "2024-03-05"],
                    "items": ["1.01", "", "5.07"],
                    "accessionNumber": ["acc1", "acc2", "acc3"],
                }
            }
        }
        mock_http = _mock_httpx_get(mock_data)

        with patch("quantstack.data.corporate_actions.httpx.AsyncClient", return_value=mock_http):
            actions = _run(fetch_edgar_8k_events("AAPL", "0000320193"))

        assert len(actions) == 1
        assert actions[0].event_type == "merger_signing"
        assert actions[0].effective_date == date(2024, 3, 15)

    def test_skips_non_target_items(self):
        from quantstack.data.corporate_actions import fetch_edgar_8k_events

        mock_data = {
            "filings": {
                "recent": {
                    "form": ["8-K"],
                    "filingDate": ["2024-01-10"],
                    "items": ["5.07"],
                    "accessionNumber": ["acc1"],
                }
            }
        }
        mock_http = _mock_httpx_get(mock_data)

        with patch("quantstack.data.corporate_actions.httpx.AsyncClient", return_value=mock_http):
            actions = _run(fetch_edgar_8k_events("AAPL", "0000320193"))

        assert actions == []

    def test_missing_cik_returns_empty(self):
        from quantstack.data.corporate_actions import fetch_edgar_8k_events

        actions = _run(fetch_edgar_8k_events("AAPL", None))
        assert actions == []

        actions = _run(fetch_edgar_8k_events("AAPL", ""))
        assert actions == []


# ---------------------------------------------------------------------------
# CIK Mapping
# ---------------------------------------------------------------------------


class TestCIKMapper:
    def test_load_and_lookup(self):
        mapper = CIKMapper()
        mock_data = {
            "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
            "1": {"cik_str": 789019, "ticker": "MSFT", "title": "Microsoft Corp."},
        }
        mock_http = _mock_httpx_get(mock_data)

        with patch("quantstack.data.corporate_actions.httpx.AsyncClient", return_value=mock_http):
            _run(mapper.load())

        assert mapper.lookup("AAPL") == "0000320193"
        assert mapper.lookup("MSFT") == "0000789019"

    def test_unknown_ticker_returns_none(self):
        mapper = CIKMapper()
        assert mapper.lookup("ZZZZZ") is None


# ---------------------------------------------------------------------------
# apply_split_adjustment
# ---------------------------------------------------------------------------


class TestApplySplitAdjustment:
    def test_4_to_1_split(self):
        from quantstack.data.corporate_actions import apply_split_adjustment

        # Multiple db_conn calls: check existing, update position, insert audit
        call_count = [0]
        conns = []
        for _ in range(4):
            conn, _ = _make_mock_conn(fetchone_val=None)
            conns.append(conn)

        @contextmanager
        def multi_db():
            idx = min(call_count[0], len(conns) - 1)
            call_count[0] += 1
            yield conns[idx]

        mock_pos = MagicMock()
        mock_pos.quantity = 10
        mock_pos.avg_cost = 200.0
        mock_ps = MagicMock()
        mock_ps.get_position.return_value = mock_pos

        with patch("quantstack.data.corporate_actions.db_conn", multi_db), \
             patch("quantstack.execution.portfolio_state.get_portfolio_state_readonly", return_value=mock_ps), \
             patch("quantstack.execution.alpaca_broker.AlpacaBroker", side_effect=Exception("no broker")), \
             patch("quantstack.tools.functions.system_alerts.emit_system_alert", new_callable=AsyncMock, return_value=1):
            result = _run(apply_split_adjustment("AAPL", 4.0, date(2024, 8, 31)))

        assert result is not None
        assert result.new_quantity == 40
        assert result.new_cost_basis == 50.0
        assert result.old_quantity == 10
        assert result.old_cost_basis == 200.0

    def test_reverse_split_floors_quantity(self):
        from quantstack.data.corporate_actions import apply_split_adjustment

        call_count = [0]
        conns = []
        for _ in range(4):
            conn, _ = _make_mock_conn(fetchone_val=None)
            conns.append(conn)

        @contextmanager
        def multi_db():
            idx = min(call_count[0], len(conns) - 1)
            call_count[0] += 1
            yield conns[idx]

        mock_pos = MagicMock()
        mock_pos.quantity = 15
        mock_pos.avg_cost = 5.0
        mock_ps = MagicMock()
        mock_ps.get_position.return_value = mock_pos

        with patch("quantstack.data.corporate_actions.db_conn", multi_db), \
             patch("quantstack.execution.portfolio_state.get_portfolio_state_readonly", return_value=mock_ps), \
             patch("quantstack.execution.alpaca_broker.AlpacaBroker", side_effect=Exception("no broker")), \
             patch("quantstack.tools.functions.system_alerts.emit_system_alert", new_callable=AsyncMock, return_value=1):
            result = _run(apply_split_adjustment("XYZ", 0.1, date(2024, 6, 1)))

        assert result is not None
        assert result.new_quantity == 1  # floor(15 * 0.1) = 1

    def test_idempotent_when_already_applied(self):
        from quantstack.data.corporate_actions import apply_split_adjustment

        conn, db = _make_mock_conn(fetchone_val={"exists": True})

        with patch("quantstack.data.corporate_actions.db_conn", db):
            result = _run(apply_split_adjustment("AAPL", 4.0, date(2024, 8, 31)))

        assert result is None

    def test_invariant_preserved_for_standard_split(self):
        old_qty, old_cost, ratio = 10, 200.0, 4.0
        new_qty = old_qty * ratio
        new_cost = old_cost / ratio
        assert abs(old_qty * old_cost - new_qty * new_cost) < 0.01


# ---------------------------------------------------------------------------
# _store_actions
# ---------------------------------------------------------------------------


class TestStoreActions:
    def test_deduplicates_on_insert(self):
        conn, db = _make_mock_conn(rowcount=0)

        with patch("quantstack.data.corporate_actions.db_conn", db):
            stored = _store_actions([
                CorporateAction("AAPL", "dividend", "alpha_vantage", date(2024, 1, 15)),
            ])

        assert stored == 0

    def test_counts_new_inserts(self):
        conn, db = _make_mock_conn(rowcount=1)

        with patch("quantstack.data.corporate_actions.db_conn", db):
            stored = _store_actions([
                CorporateAction("AAPL", "dividend", "alpha_vantage", date(2024, 1, 15)),
            ])

        assert stored == 1


# ---------------------------------------------------------------------------
# _flag_ma_events
# ---------------------------------------------------------------------------


class TestFlagMAEvents:
    def test_flags_merger_for_held_symbol(self):
        actions = [
            CorporateAction("AAPL", "merger_signing", "edgar_8k", date(2024, 3, 15)),
        ]
        with patch("quantstack.tools.functions.system_alerts.emit_system_alert", new_callable=AsyncMock, return_value=1) as mock_alert:
            flagged = _run(_flag_ma_events(actions, {"AAPL"}))

        assert flagged == 1
        mock_alert.assert_called_once()
        kwargs = mock_alert.call_args[1]
        assert kwargs["category"] == "thesis_review"
        assert kwargs["severity"] == "critical"

    def test_ignores_non_held_symbol(self):
        actions = [
            CorporateAction("TSLA", "merger_signing", "edgar_8k", date(2024, 3, 15)),
        ]
        with patch("quantstack.tools.functions.system_alerts.emit_system_alert", new_callable=AsyncMock) as mock_alert:
            flagged = _run(_flag_ma_events(actions, {"AAPL"}))

        assert flagged == 0
        mock_alert.assert_not_called()
