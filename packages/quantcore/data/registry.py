"""
DataProviderRegistry — priority-ordered, fallback-aware routing layer.

Why a registry instead of a single adapter per asset class:
  - Providers go down, rate-limit, or have data gaps.  A registry
    silently retries the next provider, keeping the calling code clean.
  - Priority ordering lets you pick Alpaca for US equities (best latency)
    but fall back to Polygon or AlphaVantage if the Alpaca key is absent.
  - ``from_settings()`` reads env vars once at startup; callers get a
    fully-wired registry without knowing which providers are available.

Thread safety
-------------
Read paths (``fetch_ohlcv``, ``get_primary``, ``get_all``) are lock-free
after startup.  Write paths (``register``, ``set_primary``) use an RLock.
In production the registry is initialised once and then only read.
"""

from __future__ import annotations

import threading

import pandas as pd
from loguru import logger

from quantcore.config.timeframes import Timeframe
from quantcore.data.base import AssetClass, AssetClassAdapter
from quantcore.data.provider_enum import DataProvider


class DataProviderRegistry:
    """Routes OHLCV fetch requests to the highest-priority available provider.

    Internal state:
        ``_adapters: Dict[AssetClass, List[AssetClassAdapter]]``

        Index 0 in each list = highest priority (tried first).
        Lower-priority providers act as automatic fallbacks.

    Usage::

        registry = DataProviderRegistry()
        registry.register(AlpacaAdapter(...))    # becomes primary
        registry.register(PolygonAdapter(...))   # fallback if Alpaca fails

        df = registry.fetch_ohlcv("SPY", AssetClass.EQUITY, Timeframe.D1)
    """

    def __init__(self) -> None:
        self._adapters: dict[AssetClass, list[AssetClassAdapter]] = {}
        self._lock = threading.RLock()

    # ── Registration ──────────────────────────────────────────────────────────

    def register(self, adapter: AssetClassAdapter, priority: int = 0) -> None:
        """Register an adapter for all asset classes it declares support for.

        Args:
            adapter:  Adapter instance (must implement ``asset_class`` property).
            priority: 0 = insert at front (highest priority), any other value
                      appends at the end.  Use 0 when you want this provider
                      to be tried first.
        """
        asset_class = adapter.asset_class
        with self._lock:
            bucket = self._adapters.setdefault(asset_class, [])
            # Remove any existing registration for the same provider to avoid
            # duplicates on hot-reload / test scenarios.
            bucket = [a for a in bucket if a.provider != adapter.provider]
            if priority == 0:
                bucket.insert(0, adapter)
            else:
                bucket.append(adapter)
            self._adapters[asset_class] = bucket

        logger.info(
            f"Registered {adapter.provider.value} for {asset_class.value} "
            f"(priority={'front' if priority == 0 else 'back'})"
        )

    def set_primary(self, asset_class: AssetClass, provider: DataProvider) -> None:
        """Promote ``provider`` to index 0 for the given asset class.

        Raises:
            KeyError: If ``provider`` is not registered for ``asset_class``.
        """
        with self._lock:
            bucket = self._adapters.get(asset_class, [])
            idx = next((i for i, a in enumerate(bucket) if a.provider == provider), None)
            if idx is None:
                raise KeyError(f"{provider.value} is not registered for {asset_class.value}")
            if idx != 0:
                bucket.insert(0, bucket.pop(idx))
            self._adapters[asset_class] = bucket

    # ── Inspection ────────────────────────────────────────────────────────────

    def get_primary(self, asset_class: AssetClass) -> AssetClassAdapter:
        """Return the highest-priority adapter for the asset class.

        Raises:
            ValueError: If no adapter is registered for the asset class.
        """
        bucket = self._adapters.get(asset_class, [])
        if not bucket:
            raise ValueError(
                f"No adapter registered for {asset_class.value}. "
                f"Call registry.register() with a compatible adapter."
            )
        return bucket[0]

    def get_all(self, asset_class: AssetClass) -> list[AssetClassAdapter]:
        """Return all adapters for the asset class in priority order."""
        return list(self._adapters.get(asset_class, []))

    def is_registered(self, asset_class: AssetClass) -> bool:
        return bool(self._adapters.get(asset_class))

    # ── Data fetch with fallback ───────────────────────────────────────────────

    def fetch_ohlcv(
        self,
        symbol: str,
        asset_class: AssetClass,
        timeframe: Timeframe,
        start_date=None,
        end_date=None,
    ) -> pd.DataFrame:
        """Fetch OHLCV bars, trying providers in priority order.

        Returns the first non-empty result.  If all providers fail or return
        empty data, returns an empty DataFrame rather than raising.

        This behaviour makes the registry transparent to callers: they always
        get a DataFrame back and can check ``df.empty`` to detect failures.
        """
        bucket = self._adapters.get(asset_class, [])
        if not bucket:
            logger.error(
                f"No adapter registered for {asset_class.value} — "
                f"cannot fetch {symbol} {timeframe.value}"
            )
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        last_exc: Exception | None = None
        for adapter in bucket:
            try:
                df = adapter.fetch_ohlcv(symbol, timeframe, start_date, end_date)
                if not df.empty:
                    logger.debug(
                        f"[Registry] {adapter.provider.value} → {symbol} "
                        f"{timeframe.value}: {len(df)} bars"
                    )
                    return df
                logger.debug(
                    f"[Registry] {adapter.provider.value} returned empty for "
                    f"{symbol} {timeframe.value}, trying next provider"
                )
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    f"[Registry] {adapter.provider.value} failed for "
                    f"{symbol} {timeframe.value}: {exc}. Trying next provider."
                )

        logger.error(
            f"[Registry] All providers failed for {symbol} {timeframe.value}. "
            f"Last error: {last_exc}"
        )
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_settings(cls, settings) -> DataProviderRegistry:
        """Build a registry from application settings.

        Reads provider credentials from ``settings`` and registers only
        those providers whose API keys are present.  Providers are registered
        in the order specified by ``settings.data_provider_priority``
        (highest priority first).

        Example::

            registry = DataProviderRegistry.from_settings(get_settings())
            df = registry.fetch_ohlcv("SPY", AssetClass.EQUITY, Timeframe.D1)
        """
        registry = cls()

        # Parse priority order from settings
        priority_order: list[str] = [
            p.strip().lower() for p in settings.data_provider_priority.split(",") if p.strip()
        ]

        # Register from lowest priority to highest so that the final
        # ``register(..., priority=0)`` call leaves the first item in
        # priority_order at index 0.
        for provider_name in reversed(priority_order):
            try:
                _register_provider(registry, provider_name, settings)
            except Exception as exc:
                logger.warning(f"[Registry] Skipping provider '{provider_name}': {exc}")

        registered = {
            ac.value: [a.provider.value for a in adapters]
            for ac, adapters in registry._adapters.items()
        }
        if not registered:
            logger.error(
                "DataProviderRegistry: no providers could be registered. "
                "Check that at least one of ALPACA_API_KEY, POLYGON_API_KEY, "
                "or ALPHA_VANTAGE_API_KEY is set in your .env file."
            )
        else:
            logger.info(f"DataProviderRegistry ready: {registered}")

        return registry


def _register_provider(
    registry: DataProviderRegistry,
    provider_name: str,
    settings,
) -> None:
    """Instantiate and register one provider by name.

    Raises if the provider is not recognised or credentials are missing.
    This lets ``from_settings`` skip unavailable providers gracefully.
    """
    if provider_name == "alpaca":
        if not settings.alpaca.api_key:
            raise ValueError("ALPACA_API_KEY is not set")
        from quantcore.data.adapters.alpaca import AlpacaAdapter

        registry.register(
            AlpacaAdapter(
                api_key=settings.alpaca.api_key,
                secret_key=settings.alpaca.secret_key,
                paper=settings.alpaca.paper,
            ),
            priority=0,
        )

    elif provider_name == "polygon":
        if not settings.polygon.api_key:
            raise ValueError("POLYGON_API_KEY is not set")
        from quantcore.data.adapters.polygon_adapter import PolygonAdapter

        registry.register(
            PolygonAdapter(api_key=settings.polygon.api_key),
            priority=0,
        )

    elif provider_name in ("alpha_vantage", "alphavantage"):
        if not settings.alpha_vantage_api_key or settings.alpha_vantage_api_key == "demo":
            raise ValueError("ALPHA_VANTAGE_API_KEY is not set or is the demo key")
        from quantcore.data.adapters.alphavantage import AlphaVantageAdapter

        registry.register(
            AlphaVantageAdapter(api_key=settings.alpha_vantage_api_key),
            priority=0,
        )

    elif provider_name == "ibkr":
        # IBKRDataAdapter requires a running IB Gateway — defer import to
        # avoid hard dependency on ib_insync for users without IBKR.
        try:
            from quantcore.data.adapters.ibkr import IBKRDataAdapter
        except ImportError as exc:
            raise ImportError(
                "ib_insync is required for IBKR. Run: uv pip install -e '.[ibkr]'"
            ) from exc
        registry.register(IBKRDataAdapter(settings=settings.ibkr), priority=0)

    else:
        raise ValueError(f"Unknown provider: '{provider_name}'")
