"""Provider registry — routes data requests with failover and circuit breaking.

The registry is the single entry point for all data fetching. It maps each
data type to an ordered provider chain, tries them in sequence, tracks failures
in the database, and enforces a circuit breaker to skip known-broken providers.
"""

from __future__ import annotations

import json
import time

import pandas as pd
from loguru import logger

from quantstack.data.providers.base import ConfigurationError, DataProvider
from quantstack.db import pg_conn

# Data type -> DataProvider method name
_METHOD_MAP: dict[str, str] = {
    "ohlcv_daily": "fetch_ohlcv_daily",
    "ohlcv_intraday": "fetch_ohlcv_intraday",
    "macro_indicator": "fetch_macro_indicator",
    "fundamentals": "fetch_fundamentals",
    "earnings_history": "fetch_earnings_history",
    "insider_transactions": "fetch_insider_transactions",
    "institutional_holdings": "fetch_institutional_holdings",
    "options_chain": "fetch_options_chain",
    "sec_filings": "fetch_sec_filings",
    "news_sentiment": "fetch_news_sentiment",
}

# Best-source routing: data_type -> ordered list of provider names
# Primary is first, fallbacks follow. Providers not in this list are skipped.
_ROUTING_TABLE: dict[str, list[str]] = {
    "ohlcv_daily": ["alpha_vantage"],
    "ohlcv_intraday": ["alpha_vantage"],
    "macro_indicator": ["alpha_vantage", "fred"],
    "fundamentals": ["alpha_vantage", "edgar"],
    "earnings_history": ["alpha_vantage", "edgar"],
    "insider_transactions": ["edgar", "alpha_vantage"],
    "institutional_holdings": ["edgar", "alpha_vantage"],
    "options_chain": ["alpha_vantage"],
    "news_sentiment": ["alpha_vantage"],
    "sec_filings": ["edgar"],
    "commodities": ["alpha_vantage", "fred"],
}

# Circuit breaker thresholds
_CB_FAILURE_THRESHOLD = 3
_CB_COOLDOWN_MINUTES = 10


class ProviderRegistry:
    """Routes data requests to the best available provider with failover.

    Holds an ordered provider chain per data type. Tries providers in order,
    skipping any whose circuit breaker is open. Tracks failures in the
    data_provider_failures table and fires alerts after 3 consecutive failures.
    """

    def __init__(self, providers: list[DataProvider]) -> None:
        self._providers: dict[str, DataProvider] = {}
        for p in providers:
            self._providers[p.name()] = p
            logger.info("[Registry] Registered provider: %s", p.name())

    def fetch(
        self, data_type: str, symbol: str, **kwargs
    ) -> pd.DataFrame | dict | None:
        """Route a data request to the best available provider.

        Tries providers in routing-table order, skipping circuit-broken ones.
        NotImplementedError is a silent skip (not a failure).
        """
        method_name = _METHOD_MAP.get(data_type)
        if not method_name:
            logger.error("[Registry] Unknown data_type: %s", data_type)
            return None

        chain = _ROUTING_TABLE.get(data_type, [])
        fallback_used = False

        for i, provider_name in enumerate(chain):
            provider = self._providers.get(provider_name)
            if provider is None:
                continue

            # Circuit breaker check
            cb_tripped = False
            try:
                cb_tripped = self._check_circuit_breaker(provider_name, data_type)
            except Exception:
                pass  # DB unavailable — try the provider anyway

            if cb_tripped:
                logger.debug(
                    "[Registry] Circuit breaker open: %s/%s, skipping",
                    provider_name, data_type,
                )
                fallback_used = True
                continue

            t0 = time.monotonic()
            try:
                method = getattr(provider, method_name)
                # Build call args: positional is symbol (or indicator for macro)
                if data_type == "macro_indicator":
                    result = method(kwargs.get("indicator", symbol))
                elif data_type == "options_chain":
                    result = method(symbol, kwargs.get("date", ""))
                elif data_type == "sec_filings":
                    result = method(symbol, kwargs.get("form_types"))
                else:
                    result = method(symbol)

                latency_ms = round((time.monotonic() - t0) * 1000, 1)

                logger.info(
                    "[Registry] provider_fetch provider=%s data_type=%s symbol=%s "
                    "latency_ms=%.1f success=True fallback_used=%s",
                    provider_name, data_type, symbol, latency_ms, fallback_used,
                )

                try:
                    self._record_success(provider_name, data_type)
                except Exception:
                    pass  # Non-critical

                return result

            except NotImplementedError:
                # Provider doesn't support this data type — skip silently
                fallback_used = True
                continue

            except Exception as exc:
                latency_ms = round((time.monotonic() - t0) * 1000, 1)
                logger.warning(
                    "[Registry] provider_fetch provider=%s data_type=%s symbol=%s "
                    "latency_ms=%.1f success=False error=%s fallback_used=%s",
                    provider_name, data_type, symbol, latency_ms, exc, fallback_used,
                )
                try:
                    self._record_failure(provider_name, data_type, str(exc))
                except Exception:
                    pass  # Non-critical
                fallback_used = True
                continue

        logger.error(
            "[Registry] All providers exhausted for %s/%s", data_type, symbol,
        )
        return None

    def _check_circuit_breaker(self, provider_name: str, data_type: str) -> bool:
        """Return True if circuit breaker is open (should skip this provider).

        Open when consecutive_failures >= 3 AND last_failure_at > now() - 10 minutes.
        """
        with pg_conn() as conn:
            row = conn.execute(
                """SELECT consecutive_failures
                   FROM data_provider_failures
                   WHERE provider = %s AND data_type = %s
                     AND consecutive_failures >= %s
                     AND last_failure_at > NOW() - INTERVAL '%s minutes'""",
                [provider_name, data_type, _CB_FAILURE_THRESHOLD, _CB_COOLDOWN_MINUTES],
            ).fetchone()
        return row is not None

    def _record_failure(self, provider_name: str, data_type: str, error: str) -> None:
        """Increment consecutive_failures. Alert after 3."""
        with pg_conn() as conn:
            row = conn.execute(
                """INSERT INTO data_provider_failures
                       (provider, data_type, consecutive_failures, last_failure_at, last_error)
                   VALUES (%s, %s, 1, NOW(), %s)
                   ON CONFLICT (provider, data_type)
                   DO UPDATE SET
                       consecutive_failures = data_provider_failures.consecutive_failures + 1,
                       last_failure_at = NOW(),
                       last_error = %s
                   RETURNING consecutive_failures""",
                [provider_name, data_type, error, error],
            ).fetchone()

            if row and row[0] >= _CB_FAILURE_THRESHOLD:
                details = json.dumps({
                    "provider_name": provider_name,
                    "data_type": data_type,
                    "consecutive_failures": row[0],
                    "last_error": error,
                })
                conn.execute(
                    """INSERT INTO system_events
                           (event_type, symbol, severity, details, created_at)
                       VALUES ('PROVIDER_FAILURE', %s, 'warning', %s, NOW())""",
                    [provider_name, details],
                )
                logger.warning(
                    "[Registry] ALERT: %s has %d consecutive failures on %s",
                    provider_name, row[0], data_type,
                )

    def _record_success(self, provider_name: str, data_type: str) -> None:
        """Reset consecutive_failures to 0."""
        with pg_conn() as conn:
            conn.execute(
                """UPDATE data_provider_failures
                   SET consecutive_failures = 0
                   WHERE provider = %s AND data_type = %s""",
                [provider_name, data_type],
            )


def build_registry() -> ProviderRegistry:
    """Factory that instantiates all known providers and builds the registry.

    Providers that raise ConfigurationError are excluded with a warning log.
    """
    from quantstack.data.providers.alpha_vantage import AVProvider
    from quantstack.data.providers.edgar import EDGARProvider
    from quantstack.data.providers.fred import FREDProvider

    providers: list[DataProvider] = []
    for cls in [AVProvider, FREDProvider, EDGARProvider]:
        try:
            providers.append(cls())
        except ConfigurationError as exc:
            logger.warning("[Registry] Skipping %s: %s", cls.__name__, exc)
        except Exception as exc:
            logger.warning("[Registry] Unexpected error initializing %s: %s", cls.__name__, exc)

    return ProviderRegistry(providers)
