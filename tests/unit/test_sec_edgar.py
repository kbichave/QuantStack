# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for quantcore.data.sec_edgar — Sprint 2.

Tests the FilingSentimentScorer (pure LM sentiment, no network calls)
and FilingSentiment derived properties.
No external I/O — all tests use canned text.
"""

from __future__ import annotations

from datetime import date

import pytest
from quantstack.data.sec_edgar import (
    EdgarSignal,
    FilingMetadata,
    FilingSentiment,
    FilingSentimentScorer,
)


def _make_metadata(form: str = "10-K") -> FilingMetadata:
    return FilingMetadata(
        cik="0000320193",
        accession_number="0000320193-24-000001",
        form_type=form,
        filed_date=date(2024, 1, 15),
        period_of_report=date(2023, 12, 31),
        company_name="Apple Inc.",
        document_url="https://example.com/filing.htm",
    )


@pytest.fixture
def scorer() -> FilingSentimentScorer:
    return FilingSentimentScorer()


# ---------------------------------------------------------------------------
# FilingSentimentScorer
# ---------------------------------------------------------------------------


class TestFilingSentimentScorer:
    def test_empty_text_returns_zero_counts(self, scorer):
        meta = _make_metadata()
        result = scorer.score(meta, "")
        assert result.positive_count == 0
        assert result.negative_count == 0
        assert result.total_words == 0

    def test_positive_text_scores_positive(self, scorer):
        text = (
            "Our strong growth and excellent revenue performance demonstrate "
            "our leading position. We expand successfully and remain profitable."
        )
        meta = _make_metadata()
        result = scorer.score(meta, text)
        assert result.positive_count > 0
        assert result.net_sentiment > 0

    def test_negative_text_scores_negative(self, scorer):
        text = (
            "We face significant risks including litigation, losses, and adverse "
            "regulatory actions. Our liabilities may result in bankruptcy. "
            "There are material uncertainty factors that could cause failure."
        )
        meta = _make_metadata()
        result = scorer.score(meta, text)
        assert result.negative_count > 0
        assert result.net_sentiment < 0

    def test_neutral_boilerplate_near_zero(self, scorer):
        text = "The company was incorporated in California in 1977."
        meta = _make_metadata()
        result = scorer.score(meta, text)
        # Mostly neutral — net sentiment close to zero
        assert abs(result.net_sentiment) < 0.1

    def test_total_words_counted(self, scorer):
        text = "strong growth revenue"
        meta = _make_metadata()
        result = scorer.score(meta, text)
        assert result.total_words == 3

    def test_uncertainty_words_counted(self, scorer):
        text = "the company may possibly depend on estimates and approximations"
        meta = _make_metadata()
        result = scorer.score(meta, text)
        assert result.uncertainty_count > 0


# ---------------------------------------------------------------------------
# FilingSentiment derived properties
# ---------------------------------------------------------------------------


class TestFilingSentimentProperties:
    def _make_sentiment(
        self, pos: int = 10, neg: int = 5, unc: int = 3, total: int = 100
    ) -> FilingSentiment:
        return FilingSentiment(
            filing=_make_metadata(),
            positive_count=pos,
            negative_count=neg,
            uncertainty_count=unc,
            total_words=total,
        )

    def test_net_sentiment_formula(self):
        s = self._make_sentiment(pos=10, neg=5, total=100)
        expected = (10 - 5) / 100
        assert abs(s.net_sentiment - expected) < 1e-9

    def test_net_sentiment_zero_words(self):
        s = self._make_sentiment(total=0)
        assert s.net_sentiment == 0.0

    def test_positivity_ratio_in_range(self):
        s = self._make_sentiment(pos=8, neg=4)
        assert 0.0 <= s.positivity_ratio <= 1.0

    def test_signal_scales_up(self):
        s = self._make_sentiment(pos=10, neg=5, total=100)
        # signal = net_sentiment * 100
        assert abs(s.signal - s.net_sentiment * 100) < 1e-9

    def test_uncertainty_ratio_in_range(self):
        s = self._make_sentiment(unc=5, total=100)
        assert 0.0 <= s.uncertainty_ratio <= 1.0


# ---------------------------------------------------------------------------
# EdgarSignal tone property
# ---------------------------------------------------------------------------


class TestEdgarSignalTone:
    def _make_signal(self, signal_value: float) -> EdgarSignal:
        return EdgarSignal(
            ticker="AAPL",
            signal=signal_value,
            latest_form_type="10-K",
            latest_filed_date=date(2024, 1, 15),
            n_filings_scored=1,
        )

    def test_positive_tone(self):
        assert self._make_signal(10.0).tone == "POSITIVE"

    def test_negative_tone(self):
        assert self._make_signal(-10.0).tone == "NEGATIVE"

    def test_neutral_tone(self):
        assert self._make_signal(1.0).tone == "NEUTRAL"

    def test_zero_signal_neutral(self):
        assert self._make_signal(0.0).tone == "NEUTRAL"
