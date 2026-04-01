# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Hook registry — lightweight callback dispatch for the execution layer.

Lives at L7 (execution) so portfolio_state and trade_service can fire hooks
without importing learning/optimization modules (L8). Higher layers register
their callbacks at app startup via `register()`.

This inverts the dependency: hooks (L8) → execution.hook_registry (L7)
instead of execution (L7) → hooks (L8).
"""

from __future__ import annotations

from typing import Any, Callable

from loguru import logger


# Callback type: (hook_name, **kwargs) -> None
_HookFn = Callable[..., None]

# Registry: hook_name -> list of callbacks
_hooks: dict[str, list[_HookFn]] = {}


def register(hook_name: str, callback: _HookFn) -> None:
    """Register a callback for a named hook. Called at app startup by higher layers."""
    _hooks.setdefault(hook_name, []).append(callback)


def fire(hook_name: str, **kwargs: Any) -> None:
    """Fire all registered callbacks for a hook. Non-blocking, best-effort."""
    for fn in _hooks.get(hook_name, []):
        try:
            fn(**kwargs)
        except Exception as exc:
            logger.debug(f"[hook_registry] {hook_name} callback {fn.__name__} failed: {exc}")


def clear(hook_name: str | None = None) -> None:
    """Clear registered hooks. If hook_name is None, clear all."""
    if hook_name is None:
        _hooks.clear()
    else:
        _hooks.pop(hook_name, None)
