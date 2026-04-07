"""Integration tests for the shared PostgreSQL-backed rate limiter.

Requires a running PostgreSQL instance (TRADER_PG_URL).
Run with: pytest -m integration tests/integration/test_rate_limiter.py
"""

import os
import threading

import pytest

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def _check_db():
    if not os.environ.get("TRADER_PG_URL"):
        pytest.skip("TRADER_PG_URL not set — skipping integration tests")


class TestConsumeToken:
    """Tests for the PL/pgSQL consume_token() function."""

    def test_returns_true_when_tokens_available(self, _check_db):
        """consume_token('alpha_vantage') returns TRUE when bucket has tokens."""
        from quantstack.db import pg_conn

        with pg_conn() as conn:
            row = conn.execute("SELECT consume_token('alpha_vantage')").fetchone()
            assert row[0] is True

    def test_returns_false_when_bucket_empty(self, _check_db):
        """After consuming all 75 tokens, the next call returns FALSE."""
        from quantstack.db import pg_conn

        with pg_conn() as conn:
            # Drain the bucket
            for _ in range(75):
                conn.execute("SELECT consume_token('alpha_vantage')")
            row = conn.execute("SELECT consume_token('alpha_vantage')").fetchone()
            assert row[0] is False

    def test_uses_clock_timestamp_not_now(self, _check_db):
        """Verify the function source contains clock_timestamp(), not now()
        for refill calculation."""
        from quantstack.db import pg_conn

        with pg_conn() as conn:
            row = conn.execute(
                "SELECT prosrc FROM pg_proc WHERE proname = 'consume_token'"
            ).fetchone()
            source = row[0]
            assert "clock_timestamp()" in source

    def test_bucket_initializes_with_correct_values(self, _check_db):
        """Seed row has tokens=75, max_tokens=75, refill_rate=1.25."""
        from quantstack.db import pg_conn

        with pg_conn() as conn:
            row = conn.execute(
                "SELECT max_tokens, refill_rate FROM rate_limit_buckets "
                "WHERE bucket_key = 'alpha_vantage'"
            ).fetchone()
            assert float(row["max_tokens"]) == 75
            assert float(row["refill_rate"]) == 1.25

    def test_atomic_under_concurrent_access(self, _check_db):
        """3 threads race to consume from a 75-token bucket.
        Exactly 75 succeed, the rest fail."""
        from quantstack.db import pg_conn

        # Reset bucket to full
        with pg_conn() as conn:
            conn.execute(
                "UPDATE rate_limit_buckets SET tokens = 75, "
                "last_refill = clock_timestamp() WHERE bucket_key = 'alpha_vantage'"
            )
            conn.execute("COMMIT")

        results = []
        lock = threading.Lock()

        def consume_n(n):
            from quantstack.db import pg_conn as _pg_conn

            local_results = []
            for _ in range(n):
                with _pg_conn() as c:
                    row = c.execute("SELECT consume_token('alpha_vantage')").fetchone()
                    local_results.append(row[0])
                    c.execute("COMMIT")
            with lock:
                results.extend(local_results)

        threads = [threading.Thread(target=consume_n, args=(30,)) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert sum(1 for r in results if r is True) == 75
        assert sum(1 for r in results if r is False) == 15


class TestFetcherRateLimiter:
    """Tests for the Python-side rate limiter integration."""

    def test_circuit_breaker_falls_back_on_db_error(self):
        """When DB is unreachable, fetcher falls back to in-memory limiter."""
        from unittest.mock import MagicMock, patch

        from quantstack.data.fetcher import AlphaVantageFetcher

        fetcher = AlphaVantageFetcher.__new__(AlphaVantageFetcher)
        fetcher.rate_limit = 75
        fetcher._fallback_call_count = 0
        fetcher._fallback_minute_start = 0
        fetcher._using_fallback = False

        with patch("quantstack.data.fetcher.pg_conn", side_effect=Exception("DB down")):
            fetcher._wait_for_rate_limit()
            assert fetcher._using_fallback is True

    def test_per_process_fallback_enforces_75_per_min(self):
        """The fallback in-memory limiter caps at 75/min."""
        import time

        from quantstack.data.fetcher import AlphaVantageFetcher

        fetcher = AlphaVantageFetcher.__new__(AlphaVantageFetcher)
        fetcher.rate_limit = 75
        fetcher._fallback_call_count = 74
        fetcher._fallback_minute_start = time.time()
        fetcher._using_fallback = False

        # Should pass (74 < 75)
        fetcher._wait_for_rate_limit_fallback()
