# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""Thread-safe singleton registry for asset-class implementations.

Usage::

    from quantstack.asset_classes.registry import register, get, enabled
    from quantstack.asset_classes.equity import EquityAssetClass

    register(EquityAssetClass())
    eq = get(AssetClassType.EQUITY)
"""

from __future__ import annotations

import threading

from loguru import logger

from quantstack.asset_classes.base import AssetClass
from quantstack.asset_classes.types import AssetClassType

_lock = threading.Lock()
_registry: dict[AssetClassType, AssetClass] = {}


def register(asset_class: AssetClass) -> None:
    """Register an :class:`AssetClass` implementation.

    Re-registering the same ``asset_type`` replaces the previous entry
    (useful during hot-reload).
    """
    with _lock:
        prev = _registry.get(asset_class.asset_type)
        _registry[asset_class.asset_type] = asset_class
        if prev is not None:
            logger.warning(
                "Replaced existing {} registration ({} → {})",
                asset_class.asset_type.value,
                type(prev).__name__,
                type(asset_class).__name__,
            )
        else:
            logger.info("Registered asset class: {}", asset_class.asset_type.value)


def get(asset_type: AssetClassType) -> AssetClass:
    """Return the registered implementation for *asset_type*.

    Raises :class:`KeyError` if no implementation has been registered.
    """
    with _lock:
        try:
            return _registry[asset_type]
        except KeyError:
            raise KeyError(
                f"No asset class registered for {asset_type.value!r}. "
                f"Registered: {[t.value for t in _registry]}"
            ) from None


def enabled() -> list[AssetClassType]:
    """Return all asset types that have a registered implementation."""
    with _lock:
        return list(_registry.keys())


def is_registered(asset_type: AssetClassType) -> bool:
    """Return *True* if *asset_type* has a registered implementation."""
    with _lock:
        return asset_type in _registry
