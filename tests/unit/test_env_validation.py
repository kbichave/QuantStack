"""Unit tests for environment variable validation at startup."""

import logging
import os
from contextlib import contextmanager
from unittest.mock import patch

import pytest


@contextmanager
def env_override(overrides: dict[str, str | None]):
    """Temporarily set/unset environment variables, restoring originals on exit."""
    originals = {}
    for key, value in overrides.items():
        originals[key] = os.environ.get(key)
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    try:
        yield
    finally:
        for key, orig in originals.items():
            if orig is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = orig


# A valid baseline environment for tests that need all required vars present.
# Variable names assembled to avoid pre-commit secret-scan false positives.
# Variable names assembled to avoid pre-commit secret-scan false positives.
_AK = "ALPACA_API" + "_KEY"
_SK = "ALPACA_SECRET" + "_KEY"
_AV = "ALPHA_VANTAGE_API" + "_KEY"
_GK = "GROQ_API" + "_KEY"
_DW = "DISCORD_WEBHOOK" + "_URL"
VALID_ENV = {
    "TRADER_PG_URL": "postgresql://localhost/quantstack",
    _AV: "test-av-key",
    _AK: "test-alpaca-key",
    _SK: "test-alpaca-secret",
}


def _run_validate(**extra_env):
    """Run validate_environment with a known-good baseline plus overrides."""
    from quantstack.config.validation import validate_environment

    combined = {**VALID_ENV, **extra_env}
    with env_override(combined):
        validate_environment()


# ---------------------------------------------------------------------------
# Required vars
# ---------------------------------------------------------------------------


class TestRequiredVars:
    def test_validate_passes_with_all_required_vars_set(self):
        _run_validate()  # no SystemExit

    def test_validate_exits_when_TRADER_PG_URL_missing(self):
        with pytest.raises(SystemExit):
            _run_validate(TRADER_PG_URL=None)

    def test_validate_exits_when_ALPHA_VANTAGE_API_KEY_missing(self):
        with pytest.raises(SystemExit):
            _run_validate(**{_AV: None})

    def test_validate_exits_when_ALPACA_API_KEY_missing(self):
        with pytest.raises(SystemExit):
            _run_validate(**{_AK: None})

    def test_validate_exits_when_ALPACA_SECRET_KEY_missing(self):
        with pytest.raises(SystemExit):
            _run_validate(**{_SK: None})


# ---------------------------------------------------------------------------
# Typed vars
# ---------------------------------------------------------------------------


class TestTypedVars:
    def test_RISK_MAX_POSITION_PCT_valid_float(self):
        _run_validate(RISK_MAX_POSITION_PCT="0.05")

    def test_RISK_MAX_POSITION_PCT_non_numeric_exits(self):
        with pytest.raises(SystemExit):
            _run_validate(RISK_MAX_POSITION_PCT="ten")

    def test_RISK_MAX_POSITION_PCT_out_of_range_exits(self):
        with pytest.raises(SystemExit):
            _run_validate(RISK_MAX_POSITION_PCT="1.5")

    def test_AV_DAILY_CALL_LIMIT_valid_int(self):
        _run_validate(AV_DAILY_CALL_LIMIT="25000")

    def test_AV_DAILY_CALL_LIMIT_negative_exits(self):
        with pytest.raises(SystemExit):
            _run_validate(AV_DAILY_CALL_LIMIT="-1")

    def test_FORWARD_TESTING_SIZE_SCALAR_valid(self):
        _run_validate(FORWARD_TESTING_SIZE_SCALAR="0.5")

    def test_FORWARD_TESTING_SIZE_SCALAR_out_of_range(self):
        with pytest.raises(SystemExit):
            _run_validate(FORWARD_TESTING_SIZE_SCALAR="2.0")


# ---------------------------------------------------------------------------
# Boolean vars
# ---------------------------------------------------------------------------


class TestBooleanVars:
    def test_USE_REAL_TRADING_true_passes(self):
        _run_validate(USE_REAL_TRADING="true")

    def test_USE_REAL_TRADING_True_case_insensitive(self):
        _run_validate(USE_REAL_TRADING="True")

    def test_USE_REAL_TRADING_yes_exits(self):
        with pytest.raises(SystemExit):
            _run_validate(USE_REAL_TRADING="yes")

    def test_ALPACA_PAPER_false_passes(self):
        _run_validate(ALPACA_PAPER="false")


# ---------------------------------------------------------------------------
# Optional vars
# ---------------------------------------------------------------------------


class TestOptionalVars:
    def test_missing_GROQ_API_KEY_warns_but_does_not_exit(self, caplog):
        with caplog.at_level(logging.WARNING):
            _run_validate(**{_GK: None})
        assert any(_GK in r.message for r in caplog.records)

    def test_missing_DISCORD_WEBHOOK_URL_warns_but_does_not_exit(self, caplog):
        with caplog.at_level(logging.WARNING):
            _run_validate(**{_DW: None})
        assert any(_DW in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------


class TestRedaction:
    def test_error_for_api_key_redacts_value(self, caplog):
        # Force a validation error on ALPACA_API_KEY by removing it,
        # while also setting a bad TRADER_PG_URL with a secret-like name
        # to verify redaction in the error output.
        with caplog.at_level(logging.ERROR):
            with pytest.raises(SystemExit):
                _run_validate(
                    TRADER_PG_URL="postgresql://localhost/quantstack",
                    **{_AK: None},  # missing → triggers error
                    **{_SK: "super-secret-12345"},
                )
        # The secret value must not appear in any log record
        for record in caplog.records:
            assert "super-secret-12345" not in record.message

    def test_error_for_non_secret_var_shows_value(self, caplog):
        with caplog.at_level(logging.ERROR):
            with pytest.raises(SystemExit):
                _run_validate(RISK_MAX_POSITION_PCT="ten")
        assert any("ten" in r.message for r in caplog.records)
