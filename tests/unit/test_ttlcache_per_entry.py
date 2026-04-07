"""Tests for per-entry TTL support in TTLCache."""

import time
from unittest.mock import patch

from quantstack.shared.cache import TTLCache


class TestPerEntryTTL:
    """Verify TTLCache supports optional per-entry TTL overrides."""

    def test_default_ttl_backward_compatibility(self):
        """set() with no ttl parameter uses instance default — existing behavior."""
        cache = TTLCache(ttl_seconds=10)
        cache.set("key", "value")

        # Immediately: value is present
        assert cache.get("key") == "value"

        # After default TTL: expired
        with patch("quantstack.shared.cache.time") as mock_time:
            mock_time.monotonic.return_value = time.monotonic() + 11
            assert cache.get("key") is None

    def test_per_entry_ttl_shorter_than_default(self):
        """set(key, value, ttl=2) expires at 2s, not the instance default 10s."""
        base = time.monotonic()
        cache = TTLCache(ttl_seconds=10)

        with patch("quantstack.shared.cache.time") as mock_time:
            mock_time.monotonic.return_value = base
            cache.set("key", "value", ttl=2)

            # After 1s: still alive
            mock_time.monotonic.return_value = base + 1
            assert cache.get("key") == "value"

            # After 3s: expired (per-entry TTL of 2s)
            mock_time.monotonic.return_value = base + 3
            assert cache.get("key") is None

    def test_per_entry_ttl_longer_than_default(self):
        """set(key, value, ttl=20) lives beyond the instance default 10s."""
        base = time.monotonic()
        cache = TTLCache(ttl_seconds=10)

        with patch("quantstack.shared.cache.time") as mock_time:
            mock_time.monotonic.return_value = base
            cache.set("key", "value", ttl=20)

            # After 15s: still alive (entry TTL is 20s)
            mock_time.monotonic.return_value = base + 15
            assert cache.get("key") == "value"

            # After 21s: expired
            mock_time.monotonic.return_value = base + 21
            assert cache.get("key") is None

    def test_mixed_entries_with_different_ttls(self):
        """Entries with different TTLs expire independently."""
        base = time.monotonic()
        cache = TTLCache(ttl_seconds=10)

        with patch("quantstack.shared.cache.time") as mock_time:
            mock_time.monotonic.return_value = base
            cache.set("a", "default")        # uses instance TTL (10s)
            cache.set("b", "short", ttl=2)   # 2s TTL
            cache.set("c", "long", ttl=30)   # 30s TTL

            # After 3s: b expired, a and c alive
            mock_time.monotonic.return_value = base + 3
            assert cache.get("a") == "default"
            assert cache.get("b") is None
            assert cache.get("c") == "long"

            # After 11s: a expired (default 10s), c alive
            mock_time.monotonic.return_value = base + 11
            assert cache.get("a") is None
            assert cache.get("c") == "long"

            # After 31s: all expired
            mock_time.monotonic.return_value = base + 31
            assert cache.get("c") is None

    def test_contains_respects_per_entry_ttl(self):
        """__contains__ uses per-entry TTL, not just instance default."""
        base = time.monotonic()
        cache = TTLCache(ttl_seconds=10)

        with patch("quantstack.shared.cache.time") as mock_time:
            mock_time.monotonic.return_value = base
            cache.set("key", "value", ttl=2)

            mock_time.monotonic.return_value = base + 1
            assert "key" in cache

            mock_time.monotonic.return_value = base + 3
            assert "key" not in cache

    def test_clear_expired_respects_per_entry_ttl(self):
        """clear_expired() removes entries based on their individual TTLs."""
        base = time.monotonic()
        cache = TTLCache(ttl_seconds=10)

        with patch("quantstack.shared.cache.time") as mock_time:
            mock_time.monotonic.return_value = base
            cache.set("a", "default")
            cache.set("b", "short", ttl=2)
            cache.set("c", "long", ttl=30)

            mock_time.monotonic.return_value = base + 5
            removed = cache.clear_expired()
            assert removed == 1  # only "b" expired (ttl=2)

            # "a" still alive (default 10s), "c" still alive (30s)
            assert cache.get("a") == "default"
            assert cache.get("c") == "long"

    def test_signal_engine_cache_regression(self):
        """Signal engine cache usage pattern still works (no ttl param)."""
        cache = TTLCache(ttl_seconds=3600)
        cache.set("AAPL", {"brief": "data"})

        assert cache.get("AAPL") == {"brief": "data"}
        assert "AAPL" in cache

        cache.delete("AAPL")
        assert cache.get("AAPL") is None

        cache.set("MSFT", {"brief": "data2"})
        cache.clear()
        assert len(cache) == 0
