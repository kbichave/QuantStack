"""Tests for start.sh password validation logic.

Extracts the validate_password function from start.sh and tests it
in isolation via subprocess, avoiding any Docker dependency.
"""

import subprocess
import textwrap


# The validation snippet extracted from start.sh — tested in isolation
_VALIDATE_SCRIPT = textwrap.dedent("""\
    #!/usr/bin/env bash
    set -euo pipefail

    validate_password() {
        local var_name="$1"
        local default_value="$2"
        local value="${!var_name:-}"

        if [[ -z "$value" ]]; then
            echo "ERROR: ${var_name} is not set. Set a strong password (12+ characters) in .env" >&2
            return 1
        fi
        if [[ "$value" == "$default_value" ]]; then
            echo "ERROR: ${var_name} is using the insecure default value '${default_value}'. Change it in .env" >&2
            return 1
        fi
        if [[ ${#value} -lt 12 ]]; then
            echo "ERROR: ${var_name} is too short (${#value} chars). Use 12+ characters." >&2
            return 1
        fi
        return 0
    }

    PW_ERRORS=0
    validate_password "POSTGRES_PASSWORD" "quantstack" || ((PW_ERRORS++))
    validate_password "LANGFUSE_DB_PASSWORD" "langfuse" || ((PW_ERRORS++))
    validate_password "LANGFUSE_INIT_USER_PASSWORD" "quantstack123" || ((PW_ERRORS++))

    if [[ $PW_ERRORS -gt 0 ]]; then
        echo "ERROR: Fix the ${PW_ERRORS} password issue(s) above before starting." >&2
        exit 1
    fi
    echo "OK"
""")


def _run_validation(env_overrides: dict[str, str]) -> subprocess.CompletedProcess:
    """Run the password validation snippet with given env vars."""
    env = {
        "PATH": "/usr/bin:/bin:/usr/local/bin",
        "HOME": "/tmp",
    }
    env.update(env_overrides)
    return subprocess.run(
        ["bash", "-c", _VALIDATE_SCRIPT],
        capture_output=True,
        text=True,
        env=env,
    )


_VALID_ENV = {
    "POSTGRES_PASSWORD": "super_secure_pw1",
    "LANGFUSE_DB_PASSWORD": "another_secure1",
    "LANGFUSE_INIT_USER_PASSWORD": "yet_another_pw1",
}


class TestStartupPasswordValidation:

    def test_rejects_missing_postgres_password(self):
        env = {**_VALID_ENV}
        del env["POSTGRES_PASSWORD"]
        result = _run_validation(env)
        assert result.returncode == 1
        assert "POSTGRES_PASSWORD" in result.stderr
        assert "not set" in result.stderr

    def test_rejects_default_postgres_password(self):
        env = {**_VALID_ENV, "POSTGRES_PASSWORD": "quantstack"}
        result = _run_validation(env)
        assert result.returncode == 1
        assert "POSTGRES_PASSWORD" in result.stderr
        assert "default" in result.stderr.lower()

    def test_rejects_short_postgres_password(self):
        env = {**_VALID_ENV, "POSTGRES_PASSWORD": "short"}
        result = _run_validation(env)
        assert result.returncode == 1
        assert "POSTGRES_PASSWORD" in result.stderr
        assert "short" in result.stderr.lower() or "chars" in result.stderr.lower()

    def test_rejects_default_langfuse_db_password(self):
        env = {**_VALID_ENV, "LANGFUSE_DB_PASSWORD": "langfuse"}
        result = _run_validation(env)
        assert result.returncode == 1
        assert "LANGFUSE_DB_PASSWORD" in result.stderr

    def test_rejects_default_langfuse_init_password(self):
        env = {**_VALID_ENV, "LANGFUSE_INIT_USER_PASSWORD": "quantstack123"}
        result = _run_validation(env)
        assert result.returncode == 1
        assert "LANGFUSE_INIT_USER_PASSWORD" in result.stderr

    def test_accepts_valid_passwords(self):
        result = _run_validation(_VALID_ENV)
        assert result.returncode == 0
        assert "OK" in result.stdout
        assert "ERROR" not in result.stderr
