"""Tests for sentiment collector fallback behavior.

Verifies that both sentiment collectors return {} when data is unavailable,
and that the synthesis pipeline handles {} sentiment without error.
"""

from unittest.mock import patch, MagicMock

from quantstack.signal_engine.collectors.sentiment_alphavantage import (
    _safe_defaults as av_safe_defaults,
)
from quantstack.signal_engine.collectors.sentiment import (
    _safe_defaults as generic_safe_defaults,
)


class TestSafeDefaults:

    def test_alphavantage_safe_defaults_returns_empty_dict(self):
        assert av_safe_defaults() == {}

    def test_generic_safe_defaults_returns_empty_dict(self):
        assert generic_safe_defaults() == {}


class TestAlphavantageCollectorFallback:

    @patch(
        "quantstack.signal_engine.collectors.sentiment_alphavantage._fetch_alphavantage_headlines",
        return_value=None,
    )
    def test_returns_empty_dict_when_no_headlines(self, _mock_fetch):
        from quantstack.signal_engine.collectors.sentiment_alphavantage import (
            _collect_sentiment_alphavantage_sync,
        )

        result = _collect_sentiment_alphavantage_sync("TEST", MagicMock())
        assert result == {}


class TestGenericCollectorFallback:

    @patch(
        "quantstack.signal_engine.collectors.sentiment._fetch_headlines",
        return_value=None,
    )
    def test_returns_empty_dict_when_no_headlines(self, _mock_fetch):
        from quantstack.signal_engine.collectors.sentiment import (
            _collect_sentiment_sync,
        )

        result = _collect_sentiment_sync("TEST")
        assert result == {}


class TestSynthesisHandlesEmptySentiment:

    def test_synthesis_handles_empty_dict_sentiment(self):
        from quantstack.signal_engine.synthesis import RuleBasedSynthesizer

        synth = RuleBasedSynthesizer()
        result = synth.synthesize(
            symbol="TEST",
            technical={"trend": "neutral", "momentum": 0.0},
            regime={"trend_regime": "unknown"},
            volume={},
            risk={},
            events={},
            fundamentals={},
            collector_failures=[],
            sentiment={},
        )
        # Should return a valid brief without raising
        assert result is not None

    def test_synthesis_handles_none_sentiment(self):
        from quantstack.signal_engine.synthesis import RuleBasedSynthesizer

        synth = RuleBasedSynthesizer()
        result = synth.synthesize(
            symbol="TEST",
            technical={"trend": "neutral", "momentum": 0.0},
            regime={"trend_regime": "unknown"},
            volume={},
            risk={},
            events={},
            fundamentals={},
            collector_failures=[],
            sentiment=None,
        )
        assert result is not None
