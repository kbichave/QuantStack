# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Tests for psycopg2 → psycopg3 migration (Section 01).

Covers: connection pool, PgConnection wrapper, JSON handling, regression checks.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Detect whether a real PostgreSQL database is available
# ---------------------------------------------------------------------------

try:
    import psycopg

    _pg_url = os.environ.get("TRADER_PG_URL", "postgresql://localhost/quantstack")
    _test_conn = psycopg.connect(_pg_url, autocommit=True)
    _test_conn.close()
    HAS_PG = True
except Exception:
    HAS_PG = False

requires_pg = pytest.mark.skipif(not HAS_PG, reason="PostgreSQL not available")

# Project root for lint/grep checks
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ============================================================================
# Connection Pool Behavior
# ============================================================================


class TestConnectionPool:
    """Verify psycopg_pool.ConnectionPool integration in db.py."""

    @requires_pg
    def test_pool_initializes_with_correct_sizes(self):
        """ConnectionPool uses min_size=min(4,max), max_size=20 (or PG_POOL_MAX override)."""
        from quantstack.db import _get_pg_pool, reset_pg_pool

        reset_pg_pool()
        pool = _get_pg_pool()
        expected_max = int(os.getenv("PG_POOL_MAX", "20"))
        assert pool.min_size == min(4, expected_max)
        assert pool.max_size == expected_max
        reset_pg_pool()

    @requires_pg
    def test_pool_max_size_respected(self):
        """Acquiring max_size+1 connections blocks or raises PoolTimeout."""
        from psycopg_pool import PoolTimeout

        from quantstack.db import _get_pg_pool, reset_pg_pool

        reset_pg_pool()
        with patch.dict(os.environ, {"PG_POOL_MAX": "2"}):
            reset_pg_pool()
            pool = _get_pg_pool()
            conns = []
            try:
                # Acquire max_size connections
                for _ in range(pool.max_size):
                    conns.append(pool.getconn(timeout=5.0))
                # Next one should timeout
                with pytest.raises(PoolTimeout):
                    pool.getconn(timeout=0.5)
            finally:
                for c in conns:
                    pool.putconn(c)
                reset_pg_pool()

    @requires_pg
    def test_context_manager_returns_connection_on_clean_exit(self):
        """pg_conn() returns connection to pool after clean exit."""
        from quantstack.db import _get_pg_pool, pg_conn, reset_pg_pool

        reset_pg_pool()
        pool = _get_pg_pool()

        with pg_conn() as conn:
            conn.execute("SELECT 1")
            assert conn.is_open

        # After context exit, connection should be released
        assert not conn.is_open

    @requires_pg
    def test_context_manager_returns_connection_on_exception(self):
        """pg_conn() returns connection to pool even when block raises."""
        from quantstack.db import pg_conn, reset_pg_pool

        reset_pg_pool()
        conn_ref = None
        with pytest.raises(ValueError):
            with pg_conn() as conn:
                conn_ref = conn
                conn.execute("SELECT 1")
                raise ValueError("intentional")

        # Connection should be released despite the exception
        assert conn_ref is not None
        assert not conn_ref.is_open


# ============================================================================
# PgConnection Wrapper
# ============================================================================


class TestPgConnection:
    """Verify PgConnection behaves correctly after psycopg3 migration."""

    @requires_pg
    def test_execute_with_percent_s_placeholders(self):
        """Parameterized INSERT with %s works, data round-trips correctly."""
        from quantstack.db import pg_conn

        with pg_conn() as conn:
            conn.execute("""
                CREATE TEMP TABLE _test_psycopg3 (
                    id SERIAL PRIMARY KEY,
                    name TEXT,
                    value DOUBLE PRECISION
                )
            """)
            conn.execute(
                "INSERT INTO _test_psycopg3 (name, value) VALUES (%s, %s)",
                ["test_row", 42.5],
            )
            conn.execute("SELECT name, value FROM _test_psycopg3 WHERE name = %s", ["test_row"])
            row = conn.fetchone()
            assert row is not None
            assert row["name"] == "test_row"
            assert row["value"] == 42.5

    @requires_pg
    def test_execute_with_question_mark_placeholders(self):
        """? placeholders are translated to %s for backward compatibility."""
        from quantstack.db import pg_conn

        with pg_conn() as conn:
            conn.execute("""
                CREATE TEMP TABLE _test_qmark (
                    id SERIAL PRIMARY KEY,
                    name TEXT
                )
            """)
            conn.execute("INSERT INTO _test_qmark (name) VALUES (?)", ["hello"])
            conn.execute("SELECT name FROM _test_qmark WHERE name = ?", ["hello"])
            row = conn.fetchone()
            assert row is not None
            assert row["name"] == "hello"

    @requires_pg
    def test_fetchone_returns_dict(self):
        """fetchone() returns a dict-like row with key access."""
        from quantstack.db import pg_conn

        with pg_conn() as conn:
            conn.execute("SELECT 1 AS num, 'hello' AS greeting")
            row = conn.fetchone()
            assert isinstance(row, dict)
            assert row["num"] == 1
            assert row["greeting"] == "hello"
            # Also supports integer indexing (backward compat)
            assert row[0] == 1
            assert row[1] == "hello"

    @requires_pg
    def test_fetchall_returns_list_of_dicts(self):
        """fetchall() returns list[dict] with tuple unpacking support."""
        from quantstack.db import pg_conn

        with pg_conn() as conn:
            conn.execute("SELECT generate_series(1, 3) AS n")
            rows = conn.fetchall()
            assert isinstance(rows, list)
            assert len(rows) == 3
            assert all(isinstance(r, dict) for r in rows)
            assert [r["n"] for r in rows] == [1, 2, 3]
            # Tuple unpacking works (backward compat with psycopg2)
            for (val,) in rows:
                assert isinstance(val, int)

    @requires_pg
    def test_fetchdf_returns_dataframe(self):
        """fetchdf() returns a pandas DataFrame with correct column names."""
        import pandas as pd

        from quantstack.db import pg_conn

        with pg_conn() as conn:
            conn.execute("SELECT 1 AS a, 2 AS b UNION ALL SELECT 3, 4")
            df = conn.fetchdf()
            assert isinstance(df, pd.DataFrame)
            assert list(df.columns) == ["a", "b"]
            assert len(df) == 2

    @requires_pg
    def test_operational_error_triggers_retry(self):
        """Broken connection is discarded and query retried on fresh connection."""
        from quantstack.db import PgConnection, _get_pg_pool, reset_pg_pool

        reset_pg_pool()
        pool = _get_pg_pool()
        conn = PgConnection(pool)
        try:
            # Force connection acquisition
            conn.execute("SELECT 1")

            # Simulate broken connection by closing the raw connection
            raw = conn._raw
            raw.close()

            # Next execute should detect broken conn, retry with fresh one
            conn.execute("SELECT 42 AS answer")
            row = conn.fetchone()
            assert row["answer"] == 42
        finally:
            conn.release()
            reset_pg_pool()


# ============================================================================
# JSON Handling
# ============================================================================


class TestJsonHandling:
    """Verify JSON/JSONB columns return raw strings (not parsed dicts)."""

    @requires_pg
    def test_jsonb_returns_raw_string(self):
        """SELECT on a JSONB column returns str, not dict."""
        from quantstack.db import pg_conn

        with pg_conn() as conn:
            conn.execute("""
                CREATE TEMP TABLE _test_json (
                    id SERIAL PRIMARY KEY,
                    data JSONB
                )
            """)
            conn.execute(
                "INSERT INTO _test_json (data) VALUES (%s::jsonb)",
                ['{"key": "value", "num": 42}'],
            )
            conn.execute("SELECT data FROM _test_json")
            row = conn.fetchone()
            assert row is not None
            # Must be a raw string, not a parsed dict
            assert isinstance(row["data"], str), (
                f"Expected str for JSONB column, got {type(row['data']).__name__}. "
                "set_json_loads(lambda x: x) may not be configured."
            )


# ============================================================================
# Regression / Lint Checks
# ============================================================================


class TestRegressionChecks:
    """Verify no psycopg2 remnants remain in the codebase."""

    def test_no_psycopg2_imports_in_src(self):
        """No 'import psycopg2' in src/ directory."""
        result = subprocess.run(
            ["grep", "-rI", "--include=*.py", "import psycopg2", str(PROJECT_ROOT / "src")],
            capture_output=True,
            text=True,
        )
        assert result.stdout.strip() == "", (
            f"psycopg2 imports found in src/:\n{result.stdout.strip()}"
        )

    def test_no_psycopg2_imports_in_scripts(self):
        """No 'import psycopg2' in scripts/ directory."""
        result = subprocess.run(
            ["grep", "-rI", "import psycopg2", str(PROJECT_ROOT / "scripts")],
            capture_output=True,
            text=True,
        )
        assert result.stdout.strip() == "", (
            f"psycopg2 imports found in scripts/:\n{result.stdout.strip()}"
        )

    def test_no_psycopg2_imports_in_tests(self):
        """No 'import psycopg2' in tests/ directory."""
        result = subprocess.run(
            [
                "grep", "-rI", "--include=*.py",
                "--exclude=test_psycopg3_migration.py",
                "import psycopg2",
                str(PROJECT_ROOT / "tests"),
            ],
            capture_output=True,
            text=True,
        )
        assert result.stdout.strip() == "", (
            f"psycopg2 imports found in tests/:\n{result.stdout.strip()}"
        )

    def test_no_psycopg2_in_pyproject_toml(self):
        """psycopg2-binary is removed from pyproject.toml dependencies."""
        content = (PROJECT_ROOT / "pyproject.toml").read_text()
        assert "psycopg2" not in content, "psycopg2-binary still listed in pyproject.toml"
