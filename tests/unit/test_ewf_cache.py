"""Tests for Section 06: EWF Deduplication via Module-Level Cache.

Validates cache population, lookup, and lifecycle.
"""

import pytest


class TestCachePopulation:
    """populate_ewf_cache stores results for symbol:timeframe combos."""

    def test_populate_stores_results(self):
        from quantstack.tools.langchain.ewf_tools import (
            _ewf_cycle_cache, populate_ewf_cache, clear_ewf_cache,
        )
        clear_ewf_cache()
        # Manually populate with test data
        populate_ewf_cache(
            preloaded_data={"SPY:4h": {"bias": "bullish"}, "SPY:daily": {"bias": "bearish"}},
        )
        assert "SPY:4h" in _ewf_cycle_cache
        assert "SPY:daily" in _ewf_cycle_cache
        clear_ewf_cache()

    def test_populate_handles_empty(self):
        from quantstack.tools.langchain.ewf_tools import (
            _ewf_cycle_cache, populate_ewf_cache, clear_ewf_cache,
        )
        clear_ewf_cache()
        populate_ewf_cache(preloaded_data={})
        assert len(_ewf_cycle_cache) == 0

    def test_clear_empties_cache(self):
        from quantstack.tools.langchain.ewf_tools import (
            _ewf_cycle_cache, populate_ewf_cache, clear_ewf_cache,
        )
        populate_ewf_cache(preloaded_data={"SPY:4h": {"bias": "bullish"}})
        assert len(_ewf_cycle_cache) > 0
        clear_ewf_cache()
        assert len(_ewf_cycle_cache) == 0


class TestCacheLookup:
    """Cache hit/miss behavior."""

    def test_cache_hit_returns_data(self):
        from quantstack.tools.langchain.ewf_tools import (
            _ewf_cycle_cache, populate_ewf_cache, clear_ewf_cache,
            get_ewf_cache_entry,
        )
        clear_ewf_cache()
        populate_ewf_cache(preloaded_data={"AAPL:daily": {"bias": "neutral"}})
        result = get_ewf_cache_entry("AAPL", "daily")
        assert result is not None
        assert result["bias"] == "neutral"
        clear_ewf_cache()

    def test_cache_miss_returns_none(self):
        from quantstack.tools.langchain.ewf_tools import (
            clear_ewf_cache, get_ewf_cache_entry,
        )
        clear_ewf_cache()
        result = get_ewf_cache_entry("AAPL", "daily")
        assert result is None

    def test_different_timeframes_different_entries(self):
        from quantstack.tools.langchain.ewf_tools import (
            populate_ewf_cache, clear_ewf_cache, get_ewf_cache_entry,
        )
        clear_ewf_cache()
        populate_ewf_cache(preloaded_data={
            "SPY:4h": {"bias": "bullish"},
            "SPY:daily": {"bias": "bearish"},
        })
        assert get_ewf_cache_entry("SPY", "4h")["bias"] == "bullish"
        assert get_ewf_cache_entry("SPY", "daily")["bias"] == "bearish"
        clear_ewf_cache()


class TestCacheLifecycle:
    """Cache is empty at expected lifecycle points."""

    def test_empty_before_populate(self):
        from quantstack.tools.langchain.ewf_tools import (
            _ewf_cycle_cache, clear_ewf_cache,
        )
        clear_ewf_cache()
        assert len(_ewf_cycle_cache) == 0

    def test_empty_after_clear(self):
        from quantstack.tools.langchain.ewf_tools import (
            _ewf_cycle_cache, populate_ewf_cache, clear_ewf_cache,
        )
        populate_ewf_cache(preloaded_data={"SPY:4h": {"bias": "bullish"}})
        clear_ewf_cache()
        assert len(_ewf_cycle_cache) == 0
