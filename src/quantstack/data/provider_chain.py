# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Provider chain — multi-provider fallback for data acquisition.

Tries providers in order until one succeeds. Tracks per-provider
success rate and latency for monitoring. Circuit breaker support
prevents hammering a provider that's down.

Usage:
    from quantstack.data.provider_chain import ProviderChain
    from quantstack.data.polygon import PolygonProvider

    chain = ProviderChain([AlphaVantageProvider(), PolygonProvider(), YahooProvider()])
    bars = chain.fetch_bars("AAPL", interval="1d", limit=252)
"""

from __future__ import annotations

import time
from collections import defaultdict
from datetime import date
from threading import Lock

from loguru import logger

from quantstack.data.provider import Bar, DataProvider, Quote, SymbolInfo


class _ProviderStats:
    """Per-provider success/failure tracking."""

    __slots__ = ("successes", "failures", "total_latency_ms", "consecutive_failures")

    def __init__(self) -> None:
        self.successes = 0
        self.failures = 0
        self.total_latency_ms = 0.0
        self.consecutive_failures = 0

    def record_success(self, latency_ms: float) -> None:
        self.successes += 1
        self.total_latency_ms += latency_ms
        self.consecutive_failures = 0

    def record_failure(self) -> None:
        self.failures += 1
        self.consecutive_failures += 1

    @property
    def is_circuit_open(self) -> bool:
        """Circuit breaker: open after 5 consecutive failures."""
        return self.consecutive_failures >= 5

    @property
    def success_rate(self) -> float:
        total = self.successes + self.failures
        return self.successes / total if total > 0 else 1.0


class ProviderChain(DataProvider):
    """Try providers in order until one succeeds.

    Wraps multiple DataProvider instances into a single DataProvider
    interface with automatic fallback. Tracks per-provider stats for
    monitoring.
    """

    def __init__(self, providers: list[DataProvider]) -> None:
        if not providers:
            raise ValueError("ProviderChain requires at least one provider")
        self._providers = providers
        self._stats: dict[str, _ProviderStats] = defaultdict(_ProviderStats)
        self._lock = Lock()

    @property
    def name(self) -> str:
        names = [p.name for p in self._providers]
        return f"chain({' → '.join(names)})"

    def get_bars(
        self,
        symbol: str,
        interval: str = "1d",
        limit: int = 252,
        start: date | None = None,
        end: date | None = None,
    ) -> list[Bar]:
        return self._try_providers(
            "get_bars", symbol=symbol, interval=interval, limit=limit, start=start, end=end,
        )

    def get_quote(self, symbol: str) -> Quote:
        return self._try_providers("get_quote", symbol=symbol)

    def get_symbol_info(self, symbol: str) -> SymbolInfo:
        return self._try_providers("get_symbol_info", symbol=symbol)

    def _try_providers(self, method: str, **kwargs):
        """Try each provider's method in order, return first success."""
        last_error: Exception | None = None

        for provider in self._providers:
            stats = self._stats[provider.name]

            if stats.is_circuit_open:
                logger.debug(
                    "provider_chain | skipping %s (circuit open, %d consecutive failures)",
                    provider.name, stats.consecutive_failures,
                )
                continue

            start_time = time.monotonic()
            try:
                result = getattr(provider, method)(**kwargs)
                latency_ms = (time.monotonic() - start_time) * 1000
                with self._lock:
                    stats.record_success(latency_ms)
                logger.debug(
                    "provider_chain | %s.%s succeeded in %.0fms",
                    provider.name, method, latency_ms,
                )
                return result
            except Exception as exc:
                with self._lock:
                    stats.record_failure()
                last_error = exc
                logger.warning(
                    "provider_chain | %s.%s failed: %s (consecutive=%d)",
                    provider.name, method, exc, stats.consecutive_failures,
                )

        # All providers failed
        logger.error(
            "provider_chain | all providers failed for %s(%s)",
            method, kwargs,
        )
        if last_error:
            raise last_error
        raise RuntimeError(f"No providers available for {method}")

    def get_stats(self) -> dict[str, dict]:
        """Return per-provider stats for monitoring."""
        return {
            name: {
                "successes": s.successes,
                "failures": s.failures,
                "success_rate": round(s.success_rate, 3),
                "consecutive_failures": s.consecutive_failures,
                "circuit_open": s.is_circuit_open,
            }
            for name, s in self._stats.items()
        }

    def reset_circuit(self, provider_name: str) -> None:
        """Manually reset a provider's circuit breaker."""
        if provider_name in self._stats:
            self._stats[provider_name].consecutive_failures = 0
            logger.info("provider_chain | circuit reset for %s", provider_name)
