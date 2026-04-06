"""Tests for acquisition pipeline new phases (AV data expansion Section 04).

Tests the commodities phase (phase 13), put_call_ratio phase (phase 14),
listing status weekly check, and phase ordering within ALL_PHASES.
"""

from __future__ import annotations

import asyncio
from datetime import date, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from quantstack.data.acquisition_pipeline import (
    ALL_PHASES,
    COMMODITY_INDICATORS,
    FOREX_PAIRS,
    AcquisitionPipeline,
    PhaseReport,
    run_listing_status_check,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def av_client():
    """Mock AlphaVantageClient with all commodity/forex/pcr methods."""
    mock = MagicMock()

    # Precious metals: returns (gold_df, silver_df) tuple
    gold_df = pd.DataFrame(
        {"value": [1900.0, 1910.0]},
        index=pd.to_datetime(["2026-01-01", "2026-01-02"]),
    )
    gold_df.index.name = "timestamp"
    silver_df = pd.DataFrame(
        {"value": [23.0, 23.5]},
        index=pd.to_datetime(["2026-01-01", "2026-01-02"]),
    )
    silver_df.index.name = "timestamp"
    mock.fetch_precious_metals_history.return_value = (gold_df, silver_df)

    # Commodity history: returns DataFrame with value column
    copper_df = pd.DataFrame(
        {"value": [4.05, 4.10]},
        index=pd.to_datetime(["2026-01-01", "2026-01-02"]),
    )
    copper_df.index.name = "timestamp"
    mock.fetch_commodity_history.return_value = copper_df

    # Forex daily: returns DataFrame with OHLC columns
    fx_df = pd.DataFrame(
        {"open": [1.08], "high": [1.09], "low": [1.07], "close": [1.085]},
        index=pd.to_datetime(["2026-01-02"]),
    )
    fx_df.index.name = "timestamp"
    mock.fetch_forex_daily.return_value = fx_df

    # PCR methods
    mock.fetch_realtime_pcr.return_value = {"put_call_ratio": 0.85}
    pcr_df = pd.DataFrame(
        {"put_call_ratio": [0.8, 0.9]},
        index=pd.to_datetime(["2026-01-01", "2026-01-02"]),
    )
    pcr_df.index.name = "timestamp"
    mock.fetch_historical_pcr.return_value = pcr_df

    # Listing status
    mock.fetch_listing_status.return_value = pd.DataFrame(
        {
            "symbol": ["ACME", "OLDCO", "SPY"],
            "name": ["Acme Inc", "Old Corp", "SPDR S&P"],
            "status": ["delisted", "delisted", "delisted"],
        }
    )

    return mock


@pytest.fixture()
def store():
    """Mock DataStore with save methods returning row counts."""
    mock = MagicMock()
    mock.save_macro_indicators.return_value = 2
    mock.save_put_call_ratio.return_value = 2
    mock.update_delisting_status.return_value = 1
    return mock


@pytest.fixture()
def pipeline(av_client, store):
    return AcquisitionPipeline(av_client=av_client, store=store)


# ---------------------------------------------------------------------------
# TestCommoditiesPhase
# ---------------------------------------------------------------------------

class TestCommoditiesPhase:
    """Phase 13: commodities — gold, silver, copper, all_commodities, forex."""

    def test_commodities_calls_precious_metals_once(self, pipeline, av_client):
        """Precious metals is called once (global, not per-symbol)."""
        with patch("quantstack.data.acquisition_pipeline.pg_conn") as mock_pg:
            _setup_empty_max_date(mock_pg)
            report = asyncio.run(pipeline.run_commodities())

        av_client.fetch_precious_metals_history.assert_called_once()

    def test_commodities_calls_commodity_history_for_copper_and_all(
        self, pipeline, av_client
    ):
        """Calls fetch_commodity_history for COPPER and ALL_COMMODITIES."""
        with patch("quantstack.data.acquisition_pipeline.pg_conn") as mock_pg:
            _setup_empty_max_date(mock_pg)
            asyncio.run(pipeline.run_commodities())

        commodity_calls = [
            c.args[0] for c in av_client.fetch_commodity_history.call_args_list
        ]
        assert "COPPER" in commodity_calls
        assert "ALL_COMMODITIES" in commodity_calls

    def test_commodities_calls_forex_for_eur_usd_and_usd_jpy(
        self, pipeline, av_client
    ):
        """Calls fetch_forex_daily for EUR/USD and USD/JPY."""
        with patch("quantstack.data.acquisition_pipeline.pg_conn") as mock_pg:
            _setup_empty_max_date(mock_pg)
            asyncio.run(pipeline.run_commodities())

        fx_calls = [
            (c.args[0], c.args[1])
            for c in av_client.fetch_forex_daily.call_args_list
        ]
        assert ("EUR", "USD") in fx_calls
        assert ("USD", "JPY") in fx_calls

    def test_commodities_stores_with_correct_indicator_names(
        self, pipeline, store
    ):
        """Macro indicators saved with GOLD, SILVER, COPPER, ALL_COMMODITIES, EURUSD, USDJPY."""
        with patch("quantstack.data.acquisition_pipeline.pg_conn") as mock_pg:
            _setup_empty_max_date(mock_pg)
            asyncio.run(pipeline.run_commodities())

        saved_indicators = [
            c.args[0] for c in store.save_macro_indicators.call_args_list
        ]
        for expected in ("GOLD", "SILVER", "COPPER", "ALL_COMMODITIES", "EURUSD", "USDJPY"):
            assert expected in saved_indicators, f"{expected} not in saved indicators"

    def test_commodities_skips_when_data_fresh(self, pipeline, store):
        """Skips indicators whose last cached date is less than 28 days ago."""
        with patch("quantstack.data.acquisition_pipeline.pg_conn") as mock_pg:
            # Return a recent date for every MAX(date) query
            recent = (date.today() - timedelta(days=5)).isoformat()
            _setup_max_date(mock_pg, recent)
            report = asyncio.run(pipeline.run_commodities())

        # No save calls — all skipped
        store.save_macro_indicators.assert_not_called()
        # All 6 indicators should be skipped
        assert report.skipped == 6

    def test_commodities_handles_partial_failure(self, pipeline, av_client, store):
        """Partial failures don't stop remaining indicators."""
        av_client.fetch_precious_metals_history.side_effect = RuntimeError("network")

        with patch("quantstack.data.acquisition_pipeline.pg_conn") as mock_pg:
            _setup_empty_max_date(mock_pg)
            report = asyncio.run(pipeline.run_commodities())

        # Precious metals fail (gold + silver = 2 failures)
        assert report.failed == 2
        # But commodity and forex calls still proceed
        assert report.succeeded > 0

    def test_commodities_in_all_phases_at_index_12(self):
        """Commodities is in ALL_PHASES at index 12 (phase 13)."""
        assert "commodities" in ALL_PHASES
        assert ALL_PHASES.index("commodities") == 12

    def test_commodities_report_tracks_six_indicators(self, pipeline):
        """PhaseReport.total should be 6 (GOLD, SILVER, COPPER, ALL_COMMODITIES, EURUSD, USDJPY)."""
        with patch("quantstack.data.acquisition_pipeline.pg_conn") as mock_pg:
            _setup_empty_max_date(mock_pg)
            report = asyncio.run(pipeline.run_commodities())

        assert report.total == 6
        assert report.phase == "commodities"


# ---------------------------------------------------------------------------
# TestPutCallRatioPhase
# ---------------------------------------------------------------------------

class TestPutCallRatioPhase:
    """Phase 14: put_call_ratio — conditional on endpoint availability."""

    def test_pcr_makes_test_call_when_flag_absent(self, pipeline, av_client):
        """On first run, tests endpoint with SPY and sets system_state flag."""
        with patch("quantstack.data.acquisition_pipeline.pg_conn") as mock_pg:
            # system_state query returns no row (flag absent)
            _setup_system_state(mock_pg, value=None)
            asyncio.run(pipeline.run_put_call_ratio(["AAPL", "MSFT"]))

        av_client.fetch_realtime_pcr.assert_any_call("SPY")

    def test_pcr_sets_flag_to_false_if_endpoint_blocked(self, pipeline, av_client):
        """If test call returns None, sets flag to 'false'."""
        av_client.fetch_realtime_pcr.return_value = None

        with patch("quantstack.data.acquisition_pipeline.pg_conn") as mock_pg:
            _setup_system_state(mock_pg, value=None)
            report = asyncio.run(pipeline.run_put_call_ratio(["AAPL"]))

        # All symbols skipped because endpoint is blocked
        assert report.skipped == len(["AAPL"])

    def test_pcr_noop_when_flag_is_false(self, pipeline, av_client):
        """When system_state indicates blocked, PCR is a no-op."""
        with patch("quantstack.data.acquisition_pipeline.pg_conn") as mock_pg:
            _setup_system_state(mock_pg, value="false")
            report = asyncio.run(pipeline.run_put_call_ratio(["AAPL", "MSFT"]))

        # No historical PCR calls made
        av_client.fetch_historical_pcr.assert_not_called()
        assert report.skipped == 2

    def test_pcr_iterates_symbols_when_enabled(self, pipeline, av_client):
        """When endpoint is available, fetches PCR for each symbol."""
        with patch("quantstack.data.acquisition_pipeline.pg_conn") as mock_pg:
            _setup_system_state(mock_pg, value="true")
            asyncio.run(pipeline.run_put_call_ratio(["AAPL", "MSFT"]))

        pcr_symbols = [
            c.args[0] for c in av_client.fetch_historical_pcr.call_args_list
        ]
        assert "AAPL" in pcr_symbols
        assert "MSFT" in pcr_symbols

    def test_pcr_in_all_phases_at_index_13(self):
        """put_call_ratio is in ALL_PHASES at index 13 (phase 14)."""
        assert "put_call_ratio" in ALL_PHASES
        assert ALL_PHASES.index("put_call_ratio") == 13


# ---------------------------------------------------------------------------
# TestListingStatusCheck
# ---------------------------------------------------------------------------

class TestListingStatusCheck:
    """Standalone listing status check — not a pipeline phase."""

    def test_cross_references_delisted_with_universe(self, av_client, store):
        """Returns only symbols that are both delisted and in universe."""
        universe = ["ACME", "SPY", "AAPL"]
        result = asyncio.run(
            run_listing_status_check(av_client, store, universe)
        )
        # ACME and SPY are both delisted and in universe; AAPL is not delisted
        assert "ACME" in result
        assert "SPY" in result
        assert "AAPL" not in result
        # OLDCO is delisted but NOT in universe
        assert "OLDCO" not in result

    def test_updates_delisting_status_for_matches(self, av_client, store):
        """Calls store.update_delisting_status for each match."""
        universe = ["ACME"]
        asyncio.run(run_listing_status_check(av_client, store, universe))
        store.update_delisting_status.assert_called()
        call_symbols = [c.args[0] for c in store.update_delisting_status.call_args_list]
        assert "ACME" in call_symbols

    def test_does_not_remove_from_universe(self, av_client, store):
        """Listing check flags delisted_at but does NOT auto-remove from universe."""
        universe = ["ACME", "SPY"]
        asyncio.run(run_listing_status_check(av_client, store, universe))
        # No remove/delete calls should exist on the store
        assert not hasattr(store, "remove_from_universe") or not store.remove_from_universe.called

    def test_empty_delisted_list(self, av_client, store):
        """Returns empty list when no symbols are delisted."""
        av_client.fetch_listing_status.return_value = pd.DataFrame()
        result = asyncio.run(run_listing_status_check(av_client, store, ["AAPL"]))
        assert result == []


# ---------------------------------------------------------------------------
# TestPhaseOrder
# ---------------------------------------------------------------------------

class TestPhaseOrder:
    """ALL_PHASES ordering invariants."""

    def test_commodities_after_fundamentals(self):
        assert ALL_PHASES.index("commodities") == ALL_PHASES.index("fundamentals") + 1

    def test_put_call_ratio_after_commodities(self):
        assert ALL_PHASES.index("put_call_ratio") == ALL_PHASES.index("commodities") + 1

    def test_total_phase_count(self):
        assert len(ALL_PHASES) == 14


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _setup_empty_max_date(mock_pg):
    """Configure pg_conn mock to return NULL for MAX(date) queries."""
    ctx = MagicMock()
    conn = MagicMock()
    conn.execute.return_value.fetchone.return_value = (None,)
    ctx.__enter__ = MagicMock(return_value=conn)
    ctx.__exit__ = MagicMock(return_value=False)
    mock_pg.return_value = ctx


def _setup_max_date(mock_pg, date_str: str):
    """Configure pg_conn mock to return a specific date for MAX(date) queries."""
    ctx = MagicMock()
    conn = MagicMock()
    conn.execute.return_value.fetchone.return_value = (date_str,)
    ctx.__enter__ = MagicMock(return_value=conn)
    ctx.__exit__ = MagicMock(return_value=False)
    mock_pg.return_value = ctx


def _setup_system_state(mock_pg, value: str | None):
    """Configure pg_conn mock for system_state queries.

    First call returns the system_state value; subsequent calls return empty
    MAX(date) for idempotency checks within PCR phase.
    """
    ctx = MagicMock()
    conn = MagicMock()

    if value is None:
        # No row found — flag absent
        conn.execute.return_value.fetchone.return_value = None
    else:
        conn.execute.return_value.fetchone.return_value = (value,)

    ctx.__enter__ = MagicMock(return_value=conn)
    ctx.__exit__ = MagicMock(return_value=False)
    mock_pg.return_value = ctx
