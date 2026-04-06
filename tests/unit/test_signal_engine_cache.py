# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for SignalEngine brief caching."""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from quantstack.shared.cache import TTLCache
from quantstack.signal_engine import cache as signal_cache


class _FakeBrief:
    """Minimal stand-in for SignalBrief."""

    def __init__(self, symbol: str):
        self.symbol = symbol
        self.collection_duration_ms = 5000.0


@pytest.fixture(autouse=True)
def _reset_cache():
    """Clear cache and counters before each test."""
    signal_cache.clear()
    signal_cache.hits = 0
    signal_cache.misses = 0
    yield
    signal_cache.clear()


class TestSignalEngineCache:
    def test_miss_returns_none(self):
        assert signal_cache.get("AAPL") is None

    def test_put_then_get(self):
        brief = _FakeBrief("AAPL")
        signal_cache.put("AAPL", brief)
        cached = signal_cache.get("AAPL")
        assert cached is brief

    def test_case_insensitive(self):
        brief = _FakeBrief("MSFT")
        signal_cache.put("msft", brief)
        assert signal_cache.get("MSFT") is brief

    def test_hit_miss_counters(self):
        signal_cache.get("X")  # miss
        signal_cache.get("Y")  # miss
        signal_cache.put("X", _FakeBrief("X"))
        signal_cache.get("X")  # hit

        s = signal_cache.stats()
        assert s["hits"] == 1
        assert s["misses"] == 2

    def test_invalidate(self):
        signal_cache.put("TSLA", _FakeBrief("TSLA"))
        signal_cache.invalidate("TSLA")
        assert signal_cache.get("TSLA") is None

    def test_clear(self):
        signal_cache.put("A", _FakeBrief("A"))
        signal_cache.put("B", _FakeBrief("B"))
        signal_cache.clear()
        assert signal_cache.get("A") is None
        assert signal_cache.get("B") is None

    def test_ttl_expiry(self):
        """Entries expire after TTL."""
        short_cache = TTLCache(ttl_seconds=1)
        short_cache.set("SPY", _FakeBrief("SPY"))
        assert short_cache.get("SPY") is not None
        time.sleep(1.1)
        assert short_cache.get("SPY") is None

    def test_disabled_flag(self):
        """When SIGNAL_ENGINE_CACHE_ENABLED=false, always returns None."""
        with patch.object(signal_cache, "_enabled", False):
            signal_cache.put("NVDA", _FakeBrief("NVDA"))
            assert signal_cache.get("NVDA") is None
