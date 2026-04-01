"""Shared helpers for MCP tool implementations.

Provides:
- ``ok()`` / ``err()`` — standard response envelope so all tools return a
  consistent shape that callers can check with ``result["success"]``.
- ``with_retry()`` — decorator for external calls that should retry on
  transient failures (network errors, DB connection drops, rate limits).
- ``coerce_json()`` — safely parse a value that may be a raw JSON string
  or already a parsed Python object (needed because db.py keeps psycopg2
  returning JSON columns as raw strings for backward compat).
"""

from __future__ import annotations

import json
import time
from functools import wraps
from typing import Any, Callable, TypeVar

from loguru import logger

F = TypeVar("F", bound=Callable[..., Any])


def ok(data: dict[str, Any] | None = None, **kwargs: Any) -> dict[str, Any]:
    """Return a successful tool response.

    Args:
        data: Primary payload dict. Merged with ``kwargs`` if both provided.
        **kwargs: Convenience keyword args merged into the response.

    Returns:
        ``{"success": True, ...data_fields}``
    """
    result: dict[str, Any] = {"success": True}
    if data:
        result.update(data)
    if kwargs:
        result.update(kwargs)
    return result


def err(msg: str, **ctx: Any) -> dict[str, Any]:
    """Return a failed tool response.

    Args:
        msg: Human-readable error description.
        **ctx: Additional context fields (e.g. ``symbol="AAPL"``, ``code=404``).

    Returns:
        ``{"success": False, "error": msg, ...ctx_fields}``
    """
    result: dict[str, Any] = {"success": False, "error": msg}
    if ctx:
        result.update(ctx)
    return result


def with_retry(
    max_attempts: int = 3,
    backoff_base: float = 0.5,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable[[F], F]:
    """Decorator that retries a function on transient failures.

    Uses exponential backoff: attempt 1 → no wait, attempt 2 → backoff_base,
    attempt 3 → backoff_base * 2, etc.

    Args:
        max_attempts: Maximum number of attempts (default 3).
        backoff_base: Base wait time in seconds between retries (default 0.5s).
        exceptions: Exception types to catch and retry on.

    Example::

        @with_retry(max_attempts=3, exceptions=(requests.HTTPError,))
        def fetch_data(symbol: str) -> dict: ...
    """

    def decorator(fn: F) -> F:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exc: Exception | None = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    if attempt < max_attempts:
                        wait = backoff_base * (2 ** (attempt - 1))
                        logger.warning(
                            f"[retry] {fn.__name__} attempt {attempt}/{max_attempts} "
                            f"failed: {exc}. Retrying in {wait:.1f}s."
                        )
                        time.sleep(wait)
                    else:
                        logger.error(
                            f"[retry] {fn.__name__} failed after {max_attempts} attempts: {exc}"
                        )
            raise last_exc  # type: ignore[misc]

        return wrapper  # type: ignore[return-value]

    return decorator


def coerce_json(value: Any) -> Any:
    """Safely coerce a value that may be a raw JSON string or already parsed.

    db.py configures psycopg2 to return JSON/JSONB columns as raw strings
    (for backward compat with code that calls json.loads() explicitly).
    New code that wants the parsed object should call this instead.

    Returns the parsed value, or the original if it can't be parsed.
    """
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (ValueError, TypeError):
            return value
    return value
