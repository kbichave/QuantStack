# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for alternative data framework stubs."""

import numpy as np
import pandas as pd
import pytest

from quantcore.features.alternative_data import (
    BorrowRateSignals,
    DataNotAvailableError,
    DarkPoolSignals,
    EarningsTranscriptNLP,
    ShortInterestSignals,
)


# ---------------------------------------------------------------------------
# DarkPoolSignals
# ---------------------------------------------------------------------------


@pytest.fixture
def dark_pool_df():
    dates = pd.date_range("2023-01-01", periods=50, freq="D")
    np.random.seed(0)
    total = np.random.randint(1_000_000, 5_000_000, 50).astype(float)
    dark  = total * np.random.uniform(0.2, 0.6, 50)
    dp_price = 100 + np.cumsum(np.random.randn(50) * 0.3)
    close    = dp_price + np.random.randn(50) * 0.1
    return pd.DataFrame({
        "dark_pool_volume": dark,
        "total_volume":     total,
        "dark_pool_price":  dp_price,
        "close":            close,
    }, index=dates)


class TestDarkPoolSignals:
    def test_returns_dataframe(self, dark_pool_df):
        result = DarkPoolSignals().compute(dark_pool_df)
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self, dark_pool_df):
        result = DarkPoolSignals().compute(dark_pool_df)
        assert {"dark_pct", "dark_zscore", "dark_elevated", "price_premium"}.issubset(
            set(result.columns)
        )

    def test_dark_pct_in_unit_interval(self, dark_pool_df):
        result = DarkPoolSignals().compute(dark_pool_df)
        assert (result["dark_pct"] >= 0).all() and (result["dark_pct"] <= 1).all()

    def test_dark_elevated_binary(self, dark_pool_df):
        result = DarkPoolSignals().compute(dark_pool_df)
        assert set(result["dark_elevated"].unique()).issubset({0, 1})

    def test_missing_columns_raises(self):
        df = pd.DataFrame({"other_col": [1, 2, 3]})
        with pytest.raises(DataNotAvailableError):
            DarkPoolSignals().compute(df)

    def test_same_length(self, dark_pool_df):
        result = DarkPoolSignals().compute(dark_pool_df)
        assert len(result) == len(dark_pool_df)


# ---------------------------------------------------------------------------
# BorrowRateSignals
# ---------------------------------------------------------------------------


@pytest.fixture
def borrow_df():
    dates = pd.date_range("2023-01-01", periods=60, freq="D")
    np.random.seed(1)
    return pd.DataFrame({
        "borrow_rate_pct": np.random.uniform(1, 25, 60),
        "short_interest":  np.random.randint(1_000_000, 5_000_000, 60).astype(float),
        "shares_float":    np.full(60, 20_000_000.0),
        "days_to_cover":   np.random.uniform(1, 10, 60),
    }, index=dates)


class TestBorrowRateSignals:
    def test_returns_dataframe(self, borrow_df):
        result = BorrowRateSignals().compute(borrow_df)
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self, borrow_df):
        result = BorrowRateSignals().compute(borrow_df)
        assert {
            "short_float_pct", "borrow_rate_pct", "borrow_rising",
            "hard_to_borrow", "squeeze_risk", "borrow_zscore",
        }.issubset(set(result.columns))

    def test_hard_to_borrow_binary(self, borrow_df):
        result = BorrowRateSignals().compute(borrow_df)
        assert set(result["hard_to_borrow"].unique()).issubset({0, 1})

    def test_squeeze_risk_binary(self, borrow_df):
        result = BorrowRateSignals().compute(borrow_df)
        assert set(result["squeeze_risk"].unique()).issubset({0, 1})

    def test_short_float_pct_in_range(self, borrow_df):
        result = BorrowRateSignals().compute(borrow_df)
        valid = result["short_float_pct"].dropna()
        assert (valid >= 0).all() and (valid <= 1).all()

    def test_missing_columns_raises(self):
        df = pd.DataFrame({"other_col": [1, 2]})
        with pytest.raises(DataNotAvailableError):
            BorrowRateSignals().compute(df)

    def test_htb_fires_at_threshold(self):
        dates = pd.date_range("2023-01-01", periods=10, freq="D")
        df = pd.DataFrame({
            "borrow_rate_pct": [5.0] * 5 + [15.0] * 5,
            "short_interest":  np.full(10, 1_000_000.0),
            "shares_float":    np.full(10, 10_000_000.0),
            "days_to_cover":   np.full(10, 3.0),
        }, index=dates)
        result = BorrowRateSignals(htb_threshold=10.0).compute(df)
        assert result["hard_to_borrow"].iloc[:5].sum() == 0
        assert result["hard_to_borrow"].iloc[5:].sum() == 5


# ---------------------------------------------------------------------------
# ShortInterestSignals
# ---------------------------------------------------------------------------


@pytest.fixture
def si_df():
    dates = pd.date_range("2023-01-01", periods=40, freq="2W")
    return pd.DataFrame({
        "short_interest":  np.linspace(1_000_000, 3_000_000, 40),
        "shares_float":    np.full(40, 15_000_000.0),
        "avg_daily_volume": np.full(40, 500_000.0),
    }, index=dates)


class TestShortInterestSignals:
    def test_returns_dataframe(self, si_df):
        result = ShortInterestSignals().compute(si_df)
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self, si_df):
        result = ShortInterestSignals().compute(si_df)
        assert {
            "si_pct", "days_to_cover", "si_crowded", "squeeze_setup", "si_momentum",
        }.issubset(set(result.columns))

    def test_si_pct_in_range(self, si_df):
        result = ShortInterestSignals().compute(si_df)
        valid = result["si_pct"].dropna()
        assert (valid >= 0).all()

    def test_si_crowded_binary(self, si_df):
        result = ShortInterestSignals().compute(si_df)
        assert set(result["si_crowded"].unique()).issubset({0, 1})

    def test_missing_columns_raises(self):
        with pytest.raises(DataNotAvailableError):
            ShortInterestSignals().compute(pd.DataFrame({"a": [1]}))

    def test_si_momentum_rising_when_interest_grows(self, si_df):
        result = ShortInterestSignals().compute(si_df)
        # Monotone increasing SI → momentum should be positive after bar 0
        assert (result["si_momentum"].iloc[1:] > 0).all()


# ---------------------------------------------------------------------------
# EarningsTranscriptNLP
# ---------------------------------------------------------------------------


class TestEarningsTranscriptNLP:
    @pytest.fixture
    def positive_transcript(self):
        return (
            "Good morning everyone. We are proud to announce strong and record results. "
            "Revenue increased significantly and profits expanded beyond our expectations. "
            "We are confident in our momentum and outlook. "
            "We expect continued growth and solid performance in the next quarter. "
            "Operator: Thank you. Our first question comes from the analyst. "
            "Q: Can you elaborate on the growth trajectory? "
            "A: Absolutely. We are accelerating across all segments and outperforming peers."
        )

    @pytest.fixture
    def negative_transcript(self):
        return (
            "We face significant challenges this quarter. Results declined sharply "
            "due to adverse market conditions and supply disruptions. "
            "We are cautious about the outlook given continued headwinds. "
            "Pressure from costs may weaken our margins. We anticipate difficult conditions. "
            "Operator: Questions now. "
            "Q: Are you concerned about the loss of market share? "
            "A: Yes, we are uncertain about the timing of recovery and see risk ahead."
        )

    def test_returns_dict(self, positive_transcript):
        result = EarningsTranscriptNLP().analyze(positive_transcript)
        assert isinstance(result, dict)

    def test_expected_keys(self, positive_transcript):
        result = EarningsTranscriptNLP().analyze(positive_transcript)
        assert {
            "tone_score", "uncertainty_score", "guidance_polarity",
            "hedging_ratio", "qa_tone_delta", "word_count",
            "positive_count", "negative_count", "has_qa_section",
        }.issubset(result.keys())

    def test_positive_transcript_higher_tone(self, positive_transcript, negative_transcript):
        pos_result = EarningsTranscriptNLP().analyze(positive_transcript)
        neg_result = EarningsTranscriptNLP().analyze(negative_transcript)
        assert pos_result["tone_score"] > neg_result["tone_score"]

    def test_positive_count_exceeds_negative_for_bullish_text(self, positive_transcript):
        result = EarningsTranscriptNLP().analyze(positive_transcript)
        assert result["positive_count"] > result["negative_count"]

    def test_negative_count_exceeds_positive_for_bearish_text(self, negative_transcript):
        result = EarningsTranscriptNLP().analyze(negative_transcript)
        assert result["negative_count"] > result["positive_count"]

    def test_qa_section_detected(self, positive_transcript):
        result = EarningsTranscriptNLP().analyze(positive_transcript)
        assert result["has_qa_section"] == 1

    def test_no_qa_section_when_absent(self):
        text = "We had a great quarter with strong growth and record profits."
        result = EarningsTranscriptNLP().analyze(text)
        assert result["has_qa_section"] == 0

    def test_guidance_raise_detected(self):
        text = "We raised guidance above prior expectations for the year."
        result = EarningsTranscriptNLP().analyze(text)
        assert result["guidance_polarity"] == 1

    def test_guidance_lower_detected(self):
        text = "We lowered guidance below prior expectations for the coming year."
        result = EarningsTranscriptNLP().analyze(text)
        assert result["guidance_polarity"] == -1

    def test_hedging_ratio_nonzero_with_hedges(self):
        text = (
            "We may see improvement. We could benefit from market recovery. "
            "We expect possibly better results. We might achieve targets."
        )
        result = EarningsTranscriptNLP().analyze(text)
        assert result["hedging_ratio"] > 0

    def test_empty_transcript(self):
        result = EarningsTranscriptNLP().analyze("")
        assert result["word_count"] == 1  # max(0, 1)
        assert result["tone_score"] == pytest.approx(0.0)

    def test_batch_analyze_returns_dataframe(self, positive_transcript, negative_transcript):
        records = [
            {"text": positive_transcript, "date": "2024-01-01", "ticker": "AAPL"},
            {"text": negative_transcript, "date": "2024-04-01", "ticker": "AAPL"},
        ]
        result = EarningsTranscriptNLP().batch_analyze(records)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert "tone_score" in result.columns

    def test_batch_analyze_empty_list(self):
        result = EarningsTranscriptNLP().batch_analyze([])
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
