"""Tests for earnings extraction verification (AV data expansion prerequisite).

Validates that:
1. _safe_float handles sentinel values correctly
2. _fetch_and_store_earnings maps AV response keys to the correct DataFrame columns
3. save_earnings_calendar upserts all required columns
4. earnings_calendar DDL includes all required columns
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from quantstack.data.acquisition_pipeline import _safe_float


# ---------------------------------------------------------------------------
# _safe_float edge cases
# ---------------------------------------------------------------------------


class TestSafeFloat:
    """Verify _safe_float handles sentinel and valid values."""

    @pytest.mark.parametrize(
        "value",
        [None, "", "-", "None"],
        ids=["none", "empty_string", "dash", "none_string"],
    )
    def test_sentinel_values_return_none(self, value: object) -> None:
        assert _safe_float(value) is None

    @pytest.mark.parametrize(
        "value, expected",
        [
            ("1.23", 1.23),
            ("-0.05", -0.05),
            ("0", 0.0),
            (42, 42.0),
            (0.0, 0.0),
        ],
        ids=["positive_str", "negative_str", "zero_str", "int", "float_zero"],
    )
    def test_valid_values_return_float(self, value: object, expected: float) -> None:
        assert _safe_float(value) == expected

    def test_non_numeric_string_returns_none(self) -> None:
        assert _safe_float("not_a_number") is None

    def test_dict_returns_none(self) -> None:
        assert _safe_float({"key": "val"}) is None


# ---------------------------------------------------------------------------
# _fetch_and_store_earnings column mapping
# ---------------------------------------------------------------------------


class TestEarningsExtraction:
    """Verify _fetch_and_store_earnings maps AV keys to correct columns."""

    @staticmethod
    def _make_pipeline(av_response: dict) -> "AcquisitionPipeline":
        """Build a minimal AcquisitionPipeline with mocked AV + store."""
        from quantstack.data.acquisition_pipeline import AcquisitionPipeline

        mock_av = MagicMock()
        mock_av.fetch_earnings_history.return_value = av_response

        mock_store = MagicMock()
        # Return the row count from whatever DataFrame is passed in
        mock_store.save_earnings_calendar.side_effect = lambda df: len(df)

        pipeline = AcquisitionPipeline.__new__(AcquisitionPipeline)
        pipeline._av = mock_av
        pipeline._store = mock_store
        return pipeline

    def test_quarterly_maps_estimatedEPS_to_estimate(self) -> None:
        av_resp = {
            "quarterlyEarnings": [
                {
                    "fiscalDateEnding": "2025-03-31",
                    "reportedDate": "2025-04-25",
                    "reportedEPS": "1.50",
                    "estimatedEPS": "1.40",
                    "surprise": "0.10",
                    "surprisePercentage": "7.14",
                },
            ],
        }
        pipeline = self._make_pipeline(av_resp)
        result = pipeline._fetch_and_store_earnings("AAPL")

        assert result == 1
        saved_df: pd.DataFrame = pipeline._store.save_earnings_calendar.call_args[0][0]
        row = saved_df.iloc[0]

        assert row["estimate"] == 1.40
        assert row["reported_eps"] == 1.50
        assert row["surprise"] == 0.10
        assert row["surprise_pct"] == pytest.approx(7.14)

    def test_quarterly_maps_surprise_column(self) -> None:
        av_resp = {
            "quarterlyEarnings": [
                {
                    "fiscalDateEnding": "2025-06-30",
                    "reportedDate": "2025-07-20",
                    "reportedEPS": "2.00",
                    "estimatedEPS": "1.80",
                    "surprise": "0.20",
                    "surprisePercentage": "11.11",
                },
            ],
        }
        pipeline = self._make_pipeline(av_resp)
        pipeline._fetch_and_store_earnings("MSFT")

        saved_df: pd.DataFrame = pipeline._store.save_earnings_calendar.call_args[0][0]
        row = saved_df.iloc[0]

        assert row["surprise"] == 0.20
        assert row["surprise_pct"] == pytest.approx(11.11)

    def test_annual_missing_surprise_keys_no_crash(self) -> None:
        """Annual earnings from AV lack surprise/surprisePercentage keys."""
        av_resp = {
            "annualEarnings": [
                {
                    "fiscalDateEnding": "2024-12-31",
                    "reportedEPS": "6.50",
                    # No estimatedEPS, surprise, surprisePercentage
                },
            ],
        }
        pipeline = self._make_pipeline(av_resp)
        result = pipeline._fetch_and_store_earnings("GOOG")

        assert result == 1
        saved_df: pd.DataFrame = pipeline._store.save_earnings_calendar.call_args[0][0]
        row = saved_df.iloc[0]

        assert row["reported_eps"] == 6.50
        assert row["estimate"] is None
        assert row["surprise"] is None
        assert row["surprise_pct"] is None

    def test_empty_earnings_response_returns_zero(self) -> None:
        pipeline = self._make_pipeline({})
        result = pipeline._fetch_and_store_earnings("XYZ")
        assert result == 0
        pipeline._store.save_earnings_calendar.assert_not_called()

    def test_none_earnings_response_returns_zero(self) -> None:
        pipeline = self._make_pipeline(None)
        result = pipeline._fetch_and_store_earnings("XYZ")
        assert result == 0

    def test_report_date_falls_back_to_fiscal_date_ending(self) -> None:
        """Annual earnings have no reportedDate; should use fiscalDateEnding."""
        av_resp = {
            "annualEarnings": [
                {
                    "fiscalDateEnding": "2024-12-31",
                    "reportedEPS": "5.00",
                },
            ],
        }
        pipeline = self._make_pipeline(av_resp)
        pipeline._fetch_and_store_earnings("TSLA")

        saved_df: pd.DataFrame = pipeline._store.save_earnings_calendar.call_args[0][0]
        assert saved_df.iloc[0]["report_date"] == "2024-12-31"


# ---------------------------------------------------------------------------
# DDL verification
# ---------------------------------------------------------------------------


class TestEarningsCalendarDDL:
    """Verify the earnings_calendar CREATE TABLE includes all required columns."""

    REQUIRED_COLUMNS = {"estimate", "reported_eps", "surprise", "surprise_pct"}

    def test_ddl_includes_all_required_columns(self) -> None:
        """Parse the DDL string from db.py and verify required columns exist."""
        import inspect

        from quantstack.db import _migrate_market_data_pg

        source = inspect.getsource(_migrate_market_data_pg)

        # Find the earnings_calendar CREATE TABLE block
        start = source.find("CREATE TABLE IF NOT EXISTS earnings_calendar")
        assert start != -1, "earnings_calendar DDL not found in ensure_schema"

        # Extract up to closing paren
        end = source.find(")", start)
        ddl_block = source[start : end + 1].lower()

        for col in self.REQUIRED_COLUMNS:
            assert col in ddl_block, (
                f"Column '{col}' missing from earnings_calendar DDL"
            )


# ---------------------------------------------------------------------------
# save_earnings_calendar upsert coverage
# ---------------------------------------------------------------------------


class TestSaveEarningsCalendarUpsert:
    """Verify save_earnings_calendar includes all fields in the ON CONFLICT UPDATE."""

    def test_upsert_updates_all_value_columns(self) -> None:
        """The valid_columns list must include estimate, surprise, surprise_pct."""
        import inspect

        from quantstack.data.pg_storage import PgDataStore

        source = inspect.getsource(PgDataStore.save_earnings_calendar)

        for col in ("estimate", "reported_eps", "surprise", "surprise_pct"):
            assert col in source, (
                f"Column '{col}' missing from save_earnings_calendar valid_columns"
            )
