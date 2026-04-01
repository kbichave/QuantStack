# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""Pure-Python stubs for symbols previously imported from CrewAI.

CrewAI was removed as a dependency in v0.6.0.  These stubs keep downstream
code (tools, flows) importable without any external package.  Only symbols
that are still referenced elsewhere are retained.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


# ---------------------------------------------------------------------------
# BaseTool — used by ~50 tool classes in tools/ and mcp_bridge/
# ---------------------------------------------------------------------------


class BaseTool:
    """Minimal tool base class with a ``_run`` contract."""

    name: str
    description: str
    args_schema: Any
    return_direct: bool

    def __init__(
        self,
        name: str | None = None,
        description: str | None = None,
        args_schema: Any = None,
        return_direct: bool = False,
        **kwargs: Any,
    ) -> None:
        # Respect class-level defaults (e.g. name: str = "rl_position_size")
        # Only override if explicitly passed or no class default exists.
        if name is not None:
            self.name = name
        elif "name" not in type(self).__dict__:
            self.name = self.__class__.__name__

        if description is not None:
            self.description = description
        elif "description" not in type(self).__dict__:
            self.description = ""

        if args_schema is not None:
            self.args_schema = args_schema

        self.return_direct = return_direct

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("_run must be implemented by subclasses")

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._run(*args, **kwargs)


# ---------------------------------------------------------------------------
# Flow — used by trading_day_flow.py and strategy_validation_flow.py
# ---------------------------------------------------------------------------


class Flow:
    """Minimal Flow container that holds state."""

    def __init__(self, state: Any = None, *args: Any, **kwargs: Any) -> None:
        self.state = (
            state
            or getattr(self, "state", None)
            or getattr(self, "state_class", lambda: None)()
        )


def start(event: str | None = None) -> Callable:
    def decorator(fn: Callable) -> Callable:
        return fn

    return decorator


def listen(event: str | None = None) -> Callable:
    def decorator(fn: Callable) -> Callable:
        return fn

    return decorator


def router(fn: Callable) -> Callable:
    return fn


__all__ = [
    "BaseTool",
    "Flow",
    "listen",
    "router",
    "start",
]
