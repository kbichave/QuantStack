"""Exponential backoff retry logic for transient failures."""

import functools
import logging
import random
import time
from typing import Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

RETRYABLE_EXCEPTIONS: tuple[type[Exception], ...] = (
    ConnectionError,
    TimeoutError,
    OSError,
)

NON_RETRYABLE_EXCEPTIONS: tuple[type[Exception], ...] = (
    ValueError,
    TypeError,
    KeyError,
    AssertionError,
    AttributeError,
)


def resilient_call(
    fn: Callable[..., T],
    *args,
    max_retries: int = 5,
    base_delay: float = 2.0,
    max_delay: float = 60.0,
    retryable_exceptions: tuple[type[Exception], ...] | None = None,
    **kwargs,
) -> T:
    """Call fn with exponential backoff + jitter on retryable errors.

    Non-retryable exceptions (ValueError, TypeError, etc.) re-raise immediately.
    """
    if retryable_exceptions is None:
        retryable_exceptions = RETRYABLE_EXCEPTIONS

    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except NON_RETRYABLE_EXCEPTIONS:
            raise
        except retryable_exceptions as exc:
            last_exc = exc
            if attempt >= max_retries:
                break
            delay = min(base_delay * (2 ** attempt) + random.uniform(0, 1), max_delay)
            logger.warning(
                "Attempt %d/%d failed (%s), retrying in %.1fs",
                attempt + 1, max_retries + 1, exc, delay,
            )
            time.sleep(delay)

    raise last_exc  # type: ignore[misc]


def db_reconnect_wrapper(fn: Callable[..., T]) -> Callable[..., T]:
    """Decorator that retries on OperationalError-like exceptions.

    Backoff: 2s, 4s, 8s, 16s, capped at 60s. Max 10 retries.
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs) -> T:
        return resilient_call(
            fn, *args,
            max_retries=10,
            base_delay=2.0,
            max_delay=60.0,
            retryable_exceptions=(Exception,),
            **kwargs,
        )
    return wrapper
