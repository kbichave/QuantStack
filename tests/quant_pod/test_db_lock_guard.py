# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
DuckDB lock-guard tests — removed.

The shared.duckdb_lock module was deleted as part of the DuckDB → PostgreSQL
migration.  PostgreSQL uses a connection pool (no file locks), so lock-guard
logic no longer exists.
"""

import pytest


@pytest.mark.skip(reason="duckdb_lock module removed — PostgreSQL has no file locks")
def test_placeholder():
    pass
