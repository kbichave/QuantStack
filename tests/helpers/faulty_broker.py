# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""FaultyBroker — wraps any broker adapter and injects configurable failures.

Used in chaos tests and failure injection scenarios to verify that the system
degrades safely (SL retries, kill switch activation) when the broker misbehaves.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from quantstack.execution.paper_broker import Fill, OrderRequest


class BrokerAPIError(Exception):
    """Simulated broker API error (HTTP 500, timeout, etc.)."""

    def __init__(self, message: str = "broker API error", status_code: int = 500):
        self.status_code = status_code
        super().__init__(message)


class FaultyBroker:
    """Wraps any broker adapter and injects configurable failures.

    Parameters
    ----------
    inner : object
        The real broker to delegate to when not failing.
    fail_next_n : int
        Number of consecutive calls that should fail before succeeding.
    error : Exception | None
        The exception to raise on failure. Defaults to BrokerAPIError("500").
    fail_on : str | None
        If set, only fail on this method name ("execute", "execute_bracket").
        Other methods pass through to the inner broker.
    """

    def __init__(
        self,
        inner: Any,
        fail_next_n: int = 0,
        error: Exception | None = None,
        fail_on: str | None = None,
    ):
        self._inner = inner
        self._fail_remaining = fail_next_n
        self._error = error or BrokerAPIError("500")
        self._fail_on = fail_on
        self._call_log: list[dict] = []

    @property
    def call_log(self) -> list[dict]:
        """History of all calls for test assertions."""
        return self._call_log

    def _maybe_fail(self, method_name: str) -> None:
        self._call_log.append({"method": method_name, "failed": False})
        if self._fail_remaining > 0:
            if self._fail_on is None or self._fail_on == method_name:
                self._fail_remaining -= 1
                self._call_log[-1]["failed"] = True
                raise self._error

    def execute(self, req: OrderRequest) -> Fill:
        self._maybe_fail("execute")
        return self._inner.execute(req)

    def execute_bracket(
        self,
        req: OrderRequest,
        *,
        stop_price: float,
        take_profit_price: float | None = None,
    ) -> Fill:
        self._maybe_fail("execute_bracket")
        if hasattr(self._inner, "execute_bracket"):
            return self._inner.execute_bracket(
                req, stop_price=stop_price, take_profit_price=take_profit_price
            )
        return self._inner.execute(req)

    def supports_bracket_orders(self) -> bool:
        return getattr(self._inner, "supports_bracket_orders", lambda: False)()

    def reset(self, fail_next_n: int = 0, error: Exception | None = None) -> None:
        """Reset failure state for a new test scenario."""
        self._fail_remaining = fail_next_n
        if error is not None:
            self._error = error
        self._call_log.clear()
