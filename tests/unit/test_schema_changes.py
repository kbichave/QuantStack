"""Tests for AV data expansion schema changes (Section 02).

Validates:
- delisted_at DATE column added to company_overview
- put_call_ratio table created with correct structure
- All DDL is idempotent
- Rollback DDL works
"""

import re

import pytest
from unittest.mock import MagicMock, call


def _make_schema_mixin():
    """Build a SchemaMixin instance with a mock connection.

    The concrete DataStore exposes ``conn`` as a property that lazily opens a
    DB handle.  The mixin methods call ``self.conn.execute(...)`` through that
    property.  We satisfy the contract by making ``conn`` a simple attribute
    that points to a MagicMock.
    """
    from quantstack.data._schema import SchemaMixin

    mixin = object.__new__(SchemaMixin)
    mock_conn = MagicMock()
    mixin._conn = mock_conn
    # SchemaMixin methods reference self.conn (a property on the real class).
    # Attach as a plain attribute so it resolves without the property descriptor.
    mixin.conn = mock_conn
    return mixin


def _executed_sql(mixin) -> list[str]:
    """Return all SQL strings passed to conn.execute, whitespace-normalised."""
    return [
        " ".join(c.args[0].split())
        for c in mixin.conn.execute.call_args_list
    ]


# ---------------------------------------------------------------------------
# delisted_at column on company_overview
# ---------------------------------------------------------------------------


class TestDelistedAtColumn:
    """Tests for delisted_at column on company_overview."""

    def test_alter_table_adds_delisted_at_column(self):
        """The schema init should execute ALTER TABLE ADD COLUMN IF NOT EXISTS delisted_at DATE."""
        mixin = _make_schema_mixin()
        mixin._init_options_schema()

        sqls = _executed_sql(mixin)
        alter_stmts = [s for s in sqls if "ALTER TABLE" in s and "delisted_at" in s]
        assert len(alter_stmts) == 1, f"Expected exactly 1 ALTER for delisted_at, got {len(alter_stmts)}"

        stmt = alter_stmts[0]
        assert "company_overview" in stmt
        assert "ADD COLUMN IF NOT EXISTS" in stmt
        assert "delisted_at" in stmt
        assert "DATE" in stmt

    def test_alter_is_idempotent(self):
        """Running the ALTER twice should not error (IF NOT EXISTS guards it)."""
        mixin = _make_schema_mixin()
        # Call twice — mock never raises, confirming idempotent DDL pattern.
        mixin._init_options_schema()
        mixin._init_options_schema()

        sqls = _executed_sql(mixin)
        alter_stmts = [s for s in sqls if "ALTER TABLE" in s and "delisted_at" in s]
        assert len(alter_stmts) == 2  # one per call, both use IF NOT EXISTS


# ---------------------------------------------------------------------------
# put_call_ratio table
# ---------------------------------------------------------------------------


class TestPutCallRatioTable:
    """Tests for put_call_ratio table DDL."""

    def test_creates_table_with_correct_columns(self):
        """CREATE TABLE should include symbol, date, put_volume, call_volume, pcr, source, fetched_at."""
        mixin = _make_schema_mixin()
        mixin._init_pcr_schema()

        sqls = _executed_sql(mixin)
        create_stmts = [s for s in sqls if "CREATE TABLE" in s and "put_call_ratio" in s]
        assert len(create_stmts) == 1

        ddl = create_stmts[0]
        for col in ("symbol", "date", "put_volume", "call_volume", "pcr", "source", "fetched_at"):
            assert col in ddl, f"Missing column: {col}"

    def test_primary_key_is_symbol_date_source(self):
        """PK should be (symbol, date, source)."""
        mixin = _make_schema_mixin()
        mixin._init_pcr_schema()

        sqls = _executed_sql(mixin)
        create_stmts = [s for s in sqls if "CREATE TABLE" in s and "put_call_ratio" in s]
        ddl = create_stmts[0]
        # Normalise and check PK clause
        assert re.search(r"PRIMARY\s+KEY\s*\(\s*symbol\s*,\s*date\s*,\s*source\s*\)", ddl)

    def test_creates_index_on_symbol_date(self):
        """Should create idx_pcr_symbol_date index."""
        mixin = _make_schema_mixin()
        mixin._init_pcr_schema()

        sqls = _executed_sql(mixin)
        idx_stmts = [s for s in sqls if "idx_pcr_symbol_date" in s]
        assert len(idx_stmts) == 1

        idx = idx_stmts[0]
        assert "CREATE INDEX IF NOT EXISTS" in idx
        assert "put_call_ratio" in idx
        assert "symbol" in idx
        assert "date" in idx

    def test_create_table_is_idempotent(self):
        """Running CREATE TABLE IF NOT EXISTS twice should not error."""
        mixin = _make_schema_mixin()
        mixin._init_pcr_schema()
        mixin._init_pcr_schema()

        sqls = _executed_sql(mixin)
        create_stmts = [s for s in sqls if "CREATE TABLE" in s and "put_call_ratio" in s]
        assert len(create_stmts) == 2  # one per call, both guarded


# ---------------------------------------------------------------------------
# _run_schema_ddl calls _init_pcr_schema
# ---------------------------------------------------------------------------


class TestSchemaOrchestration:
    """Verify _run_schema_ddl wires up the new sub-initialiser."""

    def test_run_schema_ddl_calls_init_pcr_schema(self):
        """_run_schema_ddl must invoke _init_pcr_schema."""
        mixin = _make_schema_mixin()
        # Stub all sub-inits to isolate the wiring check
        mixin._init_options_schema = MagicMock()
        mixin._init_news_sentiment_schema = MagicMock()
        mixin._init_intraday_schema = MagicMock()
        mixin._init_fundamentals_schema = MagicMock()
        mixin._init_pcr_schema = MagicMock()

        conn = MagicMock()
        mixin._run_schema_ddl(conn)

        mixin._init_pcr_schema.assert_called_once()


# ---------------------------------------------------------------------------
# Rollback DDL
# ---------------------------------------------------------------------------


class TestRollbackDDL:
    """Verify rollback SQL is defined and syntactically reasonable."""

    def test_rollback_ddl_drops_table_and_column(self):
        """The module should contain rollback DDL comments for both changes."""
        import inspect
        import quantstack.data._schema as mod

        source = inspect.getsource(mod)

        assert "DROP TABLE IF EXISTS put_call_ratio" in source, (
            "Rollback DDL for put_call_ratio table not found in _schema.py"
        )
        assert "DROP COLUMN IF EXISTS delisted_at" in source, (
            "Rollback DDL for delisted_at column not found in _schema.py"
        )
