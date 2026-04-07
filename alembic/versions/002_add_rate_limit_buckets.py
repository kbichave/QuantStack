"""Add rate_limit_buckets table and consume_token function.

Revision ID: 002
Revises: 001
Create Date: 2026-04-06

Shared rate limiter: a PostgreSQL-backed token bucket that coordinates
rate limiting across multiple Docker containers. Each API call atomically
checks, refills, and consumes tokens via a single PL/pgSQL function call.
"""

from typing import Sequence, Union

from alembic import op

revision: str = "002"
down_revision: Union[str, None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE IF NOT EXISTS rate_limit_buckets (
            bucket_key   TEXT        PRIMARY KEY,
            tokens       NUMERIC     NOT NULL,
            max_tokens   NUMERIC     NOT NULL,
            refill_rate  NUMERIC     NOT NULL,
            last_refill  TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """)

    # Seed the Alpha Vantage bucket: 75 capacity, 1.25 tokens/sec = 75/min
    op.execute("""
        INSERT INTO rate_limit_buckets (bucket_key, tokens, max_tokens, refill_rate, last_refill)
        VALUES ('alpha_vantage', 75, 75, 1.25, now())
        ON CONFLICT (bucket_key) DO NOTHING
    """)

    # PL/pgSQL token bucket function.
    # IMPORTANT: Uses clock_timestamp() (wall-clock time), NOT now() (transaction start time).
    # now() would freeze the refill calculation if called inside a larger transaction.
    op.execute("""
        CREATE OR REPLACE FUNCTION consume_token(p_key TEXT, p_cost NUMERIC DEFAULT 1)
        RETURNS BOOLEAN
        LANGUAGE plpgsql
        AS $$
        DECLARE
            v_tokens       NUMERIC;
            v_max_tokens   NUMERIC;
            v_refill_rate  NUMERIC;
            v_last_refill  TIMESTAMPTZ;
            v_elapsed      NUMERIC;
            v_new_tokens   NUMERIC;
        BEGIN
            -- Row lock: concurrent callers serialize on this row
            SELECT tokens, max_tokens, refill_rate, last_refill
              INTO v_tokens, v_max_tokens, v_refill_rate, v_last_refill
              FROM rate_limit_buckets
             WHERE bucket_key = p_key
               FOR UPDATE;

            IF NOT FOUND THEN
                RAISE EXCEPTION 'Unknown rate limit bucket: %', p_key;
            END IF;

            -- clock_timestamp() = wall-clock time at this instant
            -- (NOT now() which returns transaction start time)
            v_elapsed := EXTRACT(EPOCH FROM (clock_timestamp() - v_last_refill));
            v_new_tokens := LEAST(v_max_tokens, v_tokens + v_elapsed * v_refill_rate);

            IF v_new_tokens >= p_cost THEN
                UPDATE rate_limit_buckets
                   SET tokens = v_new_tokens - p_cost,
                       last_refill = clock_timestamp()
                 WHERE bucket_key = p_key;
                RETURN TRUE;
            ELSE
                -- Update refill state even on failure to prevent drift
                UPDATE rate_limit_buckets
                   SET tokens = v_new_tokens,
                       last_refill = clock_timestamp()
                 WHERE bucket_key = p_key;
                RETURN FALSE;
            END IF;
        END;
        $$
    """)


def downgrade() -> None:
    op.execute("DROP FUNCTION IF EXISTS consume_token(TEXT, NUMERIC)")
    op.execute("DROP TABLE IF EXISTS rate_limit_buckets")
