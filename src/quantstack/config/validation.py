"""Startup environment variable validation.

Validates all critical environment variables at process startup.
On any failure, logs every error (with secrets redacted) then exits.
"""

import logging
import os
import sys

logger = logging.getLogger(__name__)

_SECRET_SUBSTRINGS = ("KEY", "SECRET", "PASSWORD", "TOKEN")


def _is_secret(name: str) -> bool:
    upper = name.upper()
    return any(s in upper for s in _SECRET_SUBSTRINGS)


def _redact(name: str, value: str) -> str:
    return "***REDACTED***" if _is_secret(name) else repr(value)


def validate_environment() -> None:
    """Validate all critical environment variables at startup.

    Reads os.environ directly. Collects ALL errors before exiting so the
    operator sees every misconfiguration in a single run. Secrets are
    redacted in log output.

    Raises SystemExit(1) if any validation fails.
    """
    errors: list[str] = []

    # --- Required (crash if missing or empty) ---
    _check_required("TRADER_PG_URL", errors, prefix="postgresql://")
    _check_required("ALPHA_VANTAGE_API_KEY", errors)
    _check_required("ALPACA_API_KEY", errors)
    _check_required("ALPACA_SECRET_KEY", errors)

    # --- Typed float in [0.0, 1.0] (crash if present but invalid) ---
    _check_float_range("RISK_MAX_POSITION_PCT", 0.0, 1.0, errors)
    _check_float_range("FORWARD_TESTING_SIZE_SCALAR", 0.0, 1.0, errors)

    # --- Typed positive integer ---
    _check_positive_int("AV_DAILY_CALL_LIMIT", errors)

    # --- Typed positive float ---
    _check_positive_float("ROLLING_DRAWDOWN_MULTIPLIER", errors)

    # --- Boolean (crash if present but not true/false) ---
    for name in (
        "ALPACA_PAPER",
        "USE_REAL_TRADING",
        "USE_FORWARD_TESTING_FOR_ENTRIES",
        "LANGFUSE_RETENTION_ENABLED",
    ):
        _check_boolean(name, errors)

    # --- Optional (warn if missing, never crash) ---
    for name in ("GROQ_API_KEY", "DISCORD_WEBHOOK_URL", "RESEARCH_SYMBOL_OVERRIDE"):
        if not os.environ.get(name):
            logger.warning("Optional env var %s is not set", name)

    # --- Exit if any errors ---
    if errors:
        for msg in errors:
            logger.error(msg)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------


def _check_required(name: str, errors: list[str], *, prefix: str | None = None) -> None:
    value = os.environ.get(name, "")
    if not value:
        errors.append(f"{name}: required but missing or empty")
        return
    if prefix and not value.startswith(prefix):
        errors.append(
            f"{name}: must start with {prefix!r}, got {_redact(name, value)}"
        )


def _check_float_range(
    name: str, lo: float, hi: float, errors: list[str]
) -> None:
    value = os.environ.get(name)
    if value is None:
        return  # absent is OK — has defaults elsewhere
    try:
        f = float(value)
    except ValueError:
        errors.append(
            f"{name}: expected float, got {_redact(name, value)}"
        )
        return
    if not (lo <= f <= hi):
        errors.append(
            f"{name}: expected {lo}..{hi}, got {_redact(name, value)}"
        )


def _check_positive_int(name: str, errors: list[str]) -> None:
    value = os.environ.get(name)
    if value is None:
        return
    try:
        n = int(value)
    except ValueError:
        errors.append(
            f"{name}: expected positive integer, got {_redact(name, value)}"
        )
        return
    if n <= 0:
        errors.append(
            f"{name}: expected positive integer, got {_redact(name, value)}"
        )


def _check_positive_float(name: str, errors: list[str]) -> None:
    value = os.environ.get(name)
    if value is None:
        return
    try:
        f = float(value)
    except ValueError:
        errors.append(
            f"{name}: expected positive float, got {_redact(name, value)}"
        )
        return
    if f <= 0:
        errors.append(
            f"{name}: expected positive float, got {_redact(name, value)}"
        )


def _check_boolean(name: str, errors: list[str]) -> None:
    value = os.environ.get(name)
    if value is None:
        return
    if value.lower() not in ("true", "false"):
        errors.append(
            f"{name}: expected 'true' or 'false', got {_redact(name, value)}"
        )
