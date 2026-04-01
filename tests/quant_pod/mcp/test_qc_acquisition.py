# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Tests for qc_acquisition.py — acquire_historical_data, register_ticker.

Mocks AlphaVantageClient, PgDataStore, AcquisitionPipeline, and AlpacaAdapter
to avoid real API/DB calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.quantstack.mcp.conftest import _fn


# ---------------------------------------------------------------------------
# acquire_historical_data
# ---------------------------------------------------------------------------


class TestAcquireHistoricalData:

    @pytest.mark.asyncio
    async def test_dry_run(self):
        """dry_run=True returns estimated API calls without making any."""
        from quantstack.mcp.tools.qc_acquisition import acquire_historical_data

        result = await _fn(acquire_historical_data)(
            phases=["ohlcv_daily", "financials"],
            symbols=["SPY", "QQQ"],
            dry_run=True,
        )

        assert result["success"] is True
        assert result["dry_run"] is True
        assert "estimated_api_calls" in result
        assert result["symbols_count"] == 2
        assert "ohlcv_daily" in result["estimated_api_calls"]
        assert "financials" in result["estimated_api_calls"]

    @pytest.mark.asyncio
    async def test_invalid_phase(self):
        """Invalid phase name returns error."""
        from quantstack.mcp.tools.qc_acquisition import acquire_historical_data

        result = await _fn(acquire_historical_data)(
            phases=["ohlcv_daily", "not_a_phase"],
            symbols=["SPY"],
        )

        assert result["success"] is False
        assert "not_a_phase" in str(result["error"])

    @pytest.mark.asyncio
    async def test_happy_path(self):
        """Full pipeline run returns phase reports."""
        from quantstack.mcp.tools.qc_acquisition import acquire_historical_data

        @dataclass
        class MockReport:
            phase: str
            succeeded: int
            skipped: int
            failed: int
            elapsed_seconds: float
            errors: list

        mock_reports = [
            MockReport("ohlcv_daily", 2, 0, 0, 1.5, []),
        ]

        mock_pipeline = MagicMock()
        mock_pipeline.run = AsyncMock(return_value=mock_reports)

        with (
            patch("quantstack.mcp.tools.qc_acquisition.AlphaVantageClient"),
            patch("quantstack.mcp.tools.qc_acquisition.PgDataStore") as MockStore,
            patch("quantstack.mcp.tools.qc_acquisition.AlpacaAdapter"),
            patch("quantstack.mcp.tools.qc_acquisition.AcquisitionPipeline", return_value=mock_pipeline),
        ):
            MockStore.return_value.close = MagicMock()

            result = await _fn(acquire_historical_data)(
                phases=["ohlcv_daily"],
                symbols=["SPY", "QQQ"],
            )

        assert result["success"] is True
        assert result["total_ok"] == 2
        assert result["total_fail"] == 0
        assert len(result["reports"]) == 1

    @pytest.mark.asyncio
    async def test_pipeline_failure(self):
        """When pipeline raises, returns error."""
        from quantstack.mcp.tools.qc_acquisition import acquire_historical_data

        mock_pipeline = MagicMock()
        mock_pipeline.run = AsyncMock(side_effect=RuntimeError("AV rate limit"))

        with (
            patch("quantstack.mcp.tools.qc_acquisition.AlphaVantageClient"),
            patch("quantstack.mcp.tools.qc_acquisition.PgDataStore") as MockStore,
            patch("quantstack.mcp.tools.qc_acquisition.AlpacaAdapter"),
            patch("quantstack.mcp.tools.qc_acquisition.AcquisitionPipeline", return_value=mock_pipeline),
        ):
            MockStore.return_value.close = MagicMock()

            result = await _fn(acquire_historical_data)(
                phases=["ohlcv_daily"],
                symbols=["SPY"],
            )

        assert result["success"] is False
        assert "error" in result

    @pytest.mark.asyncio
    async def test_default_phases_all(self):
        """Without specifying phases, all phases are selected."""
        from quantstack.mcp.tools.qc_acquisition import acquire_historical_data, ALL_PHASES

        result = await _fn(acquire_historical_data)(
            symbols=["SPY"],
            dry_run=True,
        )

        assert result["success"] is True
        assert set(result["phases"]) == set(ALL_PHASES)


# ---------------------------------------------------------------------------
# register_ticker
# ---------------------------------------------------------------------------


class TestRegisterTicker:

    @pytest.mark.asyncio
    async def test_dry_run(self):
        """dry_run returns metadata without DB writes."""
        from quantstack.mcp.tools.qc_acquisition import register_ticker

        mock_av = MagicMock()
        mock_av.fetch_company_overview.return_value = {
            "Symbol": "HIMS",
            "Name": "Hims & Hers Health Inc",
            "Sector": "Healthcare",
            "Industry": "Drug Manufacturers",
            "Description": "Hims & Hers is a telehealth company offering wellness products.",
        }

        with patch("quantstack.mcp.tools.qc_acquisition.AlphaVantageClient", return_value=mock_av):
            result = await _fn(register_ticker)(
                symbol="HIMS",
                dry_run=True,
            )

        assert result["success"] is True
        assert result["symbol"] == "HIMS"
        assert result["name"] == "Hims & Hers Health Inc"
        assert result["dry_run"] is True
        assert "acquisition_estimate" in result

    @pytest.mark.asyncio
    async def test_invalid_ticker(self):
        """When AV returns no data, returns error."""
        from quantstack.mcp.tools.qc_acquisition import register_ticker

        mock_av = MagicMock()
        mock_av.fetch_company_overview.return_value = {}

        with patch("quantstack.mcp.tools.qc_acquisition.AlphaVantageClient", return_value=mock_av):
            result = await _fn(register_ticker)(symbol="ZZZZ")

        assert result["success"] is False
        assert "ZZZZ" in result["symbol"]

    @pytest.mark.asyncio
    async def test_symbol_uppercased(self):
        """Symbol is uppercased and stripped."""
        from quantstack.mcp.tools.qc_acquisition import register_ticker

        mock_av = MagicMock()
        mock_av.fetch_company_overview.return_value = {
            "Symbol": "HIMS",
            "Name": "Test",
        }

        with patch("quantstack.mcp.tools.qc_acquisition.AlphaVantageClient", return_value=mock_av):
            result = await _fn(register_ticker)(
                symbol="  hims  ",
                dry_run=True,
            )

        assert result["symbol"] == "HIMS"

    @pytest.mark.asyncio
    async def test_av_exception(self):
        """When AV client raises, returns error."""
        from quantstack.mcp.tools.qc_acquisition import register_ticker

        with patch(
            "quantstack.mcp.tools.qc_acquisition.AlphaVantageClient",
            side_effect=Exception("API key invalid"),
        ):
            result = await _fn(register_ticker)(symbol="AAPL")

        assert result["success"] is False
        assert "error" in result


# ---------------------------------------------------------------------------
# _estimate_calls helper
# ---------------------------------------------------------------------------


class TestEstimateCalls:

    def test_ohlcv_5min_scales_by_months(self):
        from quantstack.mcp.tools.qc_acquisition import _estimate_calls

        est = _estimate_calls(["ohlcv_5min"], ["SPY", "QQQ"], months=12)
        assert est["ohlcv_5min"] == 24  # 2 symbols * 12 months

    def test_macro_constant(self):
        from quantstack.mcp.tools.qc_acquisition import _estimate_calls

        est = _estimate_calls(["macro"], ["SPY"], months=1)
        assert est["macro"] == 9  # constant

    def test_news_ceil_division(self):
        from quantstack.mcp.tools.qc_acquisition import _estimate_calls

        est = _estimate_calls(["news"], ["A", "B", "C"], months=1)
        # ceil(3 / 5) = 1
        assert est["news"] == 1
