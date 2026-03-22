"""Tests for shared.cache."""

import time
from unittest.mock import patch

from quantstack.shared.cache import TTLCache


class TestTTLCache:
    def test_set_and_get(self):
        cache = TTLCache(ttl_seconds=60)
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

    def test_get_missing_key(self):
        cache = TTLCache(ttl_seconds=60)
        assert cache.get("nonexistent") is None

    def test_expiry(self):
        cache = TTLCache(ttl_seconds=1)
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

        # Simulate time passing beyond TTL
        with patch("quantstack.shared.cache.time") as mock_time:
            # First call for set was real; now mock monotonic to be past TTL
            mock_time.monotonic.return_value = time.monotonic() + 2
            assert cache.get("key1") is None

    def test_overwrite(self):
        cache = TTLCache(ttl_seconds=60)
        cache.set("key1", "v1")
        cache.set("key1", "v2")
        assert cache.get("key1") == "v2"

    def test_delete_existing(self):
        cache = TTLCache(ttl_seconds=60)
        cache.set("key1", "value1")
        assert cache.delete("key1") is True
        assert cache.get("key1") is None

    def test_delete_nonexistent(self):
        cache = TTLCache(ttl_seconds=60)
        assert cache.delete("nope") is False

    def test_clear(self):
        cache = TTLCache(ttl_seconds=60)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.clear()
        assert cache.get("a") is None
        assert cache.get("b") is None
        assert len(cache) == 0

    def test_len(self):
        cache = TTLCache(ttl_seconds=60)
        assert len(cache) == 0
        cache.set("a", 1)
        assert len(cache) == 1
        cache.set("b", 2)
        assert len(cache) == 2

    def test_contains(self):
        cache = TTLCache(ttl_seconds=60)
        cache.set("key1", "value1")
        assert "key1" in cache
        assert "key2" not in cache

    def test_clear_expired(self):
        cache = TTLCache(ttl_seconds=1)
        cache.set("a", 1)
        cache.set("b", 2)

        # Nothing expired yet
        assert cache.clear_expired() == 0

        # Force expiry by manipulating internal timestamps
        now = time.monotonic()
        cache._store["a"] = (1, now - 2)  # expired
        cache._store["b"] = (2, now + 100)  # still valid

        removed = cache.clear_expired()
        assert removed == 1
        assert cache.get("a") is None
        assert cache.get("b") == 2

    def test_stores_various_types(self):
        cache = TTLCache(ttl_seconds=60)
        cache.set("int", 42)
        cache.set("list", [1, 2, 3])
        cache.set("dict", {"a": 1})
        cache.set("none", None)

        assert cache.get("int") == 42
        assert cache.get("list") == [1, 2, 3]
        assert cache.get("dict") == {"a": 1}
        # None is a valid value — but get() returns None for missing keys too,
        # so use `in` to distinguish
        assert "none" in cache
