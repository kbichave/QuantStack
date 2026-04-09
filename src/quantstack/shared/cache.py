# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Generic in-memory TTL cache.

Extracted from quantstack's IC output cache so it can be reused across
any subsystem that needs short-lived memoization.
"""

from __future__ import annotations

import time
from typing import Any


class TTLCache:
    """In-memory key→value store with per-entry time-to-live expiry.

    Args:
        ttl_seconds: Entries older than this are treated as absent.
    """

    def __init__(self, ttl_seconds: int = 1800) -> None:
        self._ttl = ttl_seconds
        self._store: dict[str, tuple[Any, float, int | None]] = {}

    # -- public API ----------------------------------------------------------

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Store *value* under *key* with the current monotonic timestamp.

        Args:
            ttl: Optional per-entry TTL in seconds. If None, the instance
                 default (self._ttl) is used at read time.
        """
        self._store[key] = (value, time.monotonic(), ttl)

    def get(self, key: str) -> Any | None:
        """Return the value for *key*, or ``None`` if absent/expired."""
        entry = self._store.get(key)
        if entry is None:
            return None
        value, ts, entry_ttl = entry
        effective_ttl = entry_ttl if entry_ttl is not None else self._ttl
        if time.monotonic() - ts > effective_ttl:
            del self._store[key]
            return None
        return value

    def get_with_age(self, key: str) -> tuple[Any, float] | None:
        """Return ``(value, age_seconds)`` or ``None`` if absent/expired."""
        entry = self._store.get(key)
        if entry is None:
            return None
        value, ts, entry_ttl = entry
        effective_ttl = entry_ttl if entry_ttl is not None else self._ttl
        age = time.monotonic() - ts
        if age > effective_ttl:
            del self._store[key]
            return None
        return value, age

    def delete(self, key: str) -> bool:
        """Remove *key* if present.  Returns True if it existed."""
        return self._store.pop(key, None) is not None

    def clear(self) -> None:
        """Remove all entries."""
        self._store.clear()

    def clear_expired(self) -> int:
        """Remove all expired entries.  Returns the number removed."""
        now = time.monotonic()
        expired = [
            k
            for k, (_, ts, entry_ttl) in self._store.items()
            if now - ts > (entry_ttl if entry_ttl is not None else self._ttl)
        ]
        for k in expired:
            del self._store[k]
        return len(expired)

    def __len__(self) -> int:
        """Return the number of entries (including possibly-expired ones)."""
        return len(self._store)

    def __contains__(self, key: str) -> bool:
        """Check membership (respects TTL).  Distinguishes stored-None from missing."""
        entry = self._store.get(key)
        if entry is None:
            return False
        _, ts, entry_ttl = entry
        effective_ttl = entry_ttl if entry_ttl is not None else self._ttl
        if time.monotonic() - ts > effective_ttl:
            del self._store[key]
            return False
        return True
