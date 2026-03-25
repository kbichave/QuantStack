"""
Compatibility shim — DataStore is now backed by PostgreSQL via PgDataStore.

The legacy DuckDB-based implementation has been removed. All callers that import
DataStore continue to work; legacy constructor kwargs (db_path, read_only,
persistent) are accepted and silently ignored.
"""

from quantstack.data.pg_storage import PgDataStore


class DataStore(PgDataStore):
    """PostgreSQL-backed data store (replaces the former DataStore).

    Legacy kwargs ``db_path``, ``read_only``, and ``persistent`` are accepted
    for call-site compatibility but have no effect — PgDataStore manages its
    own connection pool via pg_conn().
    """

    def __init__(
        self,
        db_path: str | None = None,
        read_only: bool = False,
        persistent: bool = False,
    ) -> None:
        super().__init__()
