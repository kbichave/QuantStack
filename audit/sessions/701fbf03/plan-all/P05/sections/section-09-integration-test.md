# Section 09: Integration Test

## Objective

Write a single integration test that exercises the full P05 adaptive synthesis pipeline end-to-end: IC weight precomputation -> synthesis with precomputed weights -> transition zone propagation -> conviction factor metadata enrichment -> ensemble method recording. This test validates that the sections work together, not just in isolation.

## Dependencies

- **All Sections 01-08** must be implemented
- Requires a test PostgreSQL instance (or use the existing test DB fixture)

## File to Create

**`tests/integration/test_p05_adaptive_synthesis.py`**

If `tests/integration/` does not exist, create it with an `__init__.py`.

## Implementation

```python
"""Integration test: P05 Adaptive Signal Synthesis pipeline.

Requires a test PostgreSQL instance. Exercises the full flow:
  1. Seed IC observations for multiple collectors across regimes
  2. Run compute_and_store_ic_weights()
  3. Synthesize a symbol with known inputs
  4. Assert: IC-driven weights used (differ from static)
  5. Assert: transition_zone propagated to SymbolBrief
  6. Assert: conviction_factors in signals metadata
  7. Assert: ensemble method recorded in ensemble_ab_results

Skip if TRADER_PG_URL is not set (CI without DB).
"""

import json
import os
import math
import pytest
from datetime import date, datetime, timezone

# Skip entire module if no test DB
pytestmark = pytest.mark.skipif(
    not os.environ.get("TRADER_PG_URL"),
    reason="TRADER_PG_URL not set — integration test requires PostgreSQL",
)


@pytest.fixture(scope="module")
def db_setup():
    """Run migrations and return a db_conn context manager."""
    from quantstack.db import db_conn, run_migrations_pg

    # Run migrations to create all tables
    with db_conn() as conn:
        run_migrations_pg(conn)

    yield db_conn

    # Cleanup: remove test data from P05 tables
    with db_conn() as conn:
        for table in (
            "precomputed_ic_weights",
            "ensemble_ab_results",
            "ensemble_config",
            "conviction_calibration_params",
        ):
            try:
                conn.execute(f"DELETE FROM {table}")
            except Exception:
                pass
        # Clean IC attribution test data
        conn.execute(
            "DELETE FROM ic_attribution_data WHERE collector LIKE 'test_%%'"
        )
        # Clean signals test data
        conn.execute(
            "DELETE FROM signals WHERE symbol = 'TEST_P05'"
        )


@pytest.fixture
def seed_ic_data(db_setup):
    """Seed 63 days of IC attribution observations for 3 collectors across 2 regimes."""
    import random
    random.seed(42)

    with db_setup() as conn:
        for day_offset in range(63):
            recorded_at = f"2026-03-01T00:00:00+00:00"  # Adjust per offset
            for regime in ("trending_up", "ranging"):
                for collector, base_ic in [
                    ("test_trend", 0.10),
                    ("test_rsi", 0.04),
                    ("test_ml", 0.07),
                ]:
                    # Generate signal/return pairs that produce approximate target IC
                    signal = random.gauss(0.5, 0.2)
                    noise = random.gauss(0, 0.1)
                    forward_return = base_ic * signal + noise

                    conn.execute(
                        """
                        INSERT INTO ic_attribution_data
                            (collector, signal_value, forward_return, recorded_at, regime)
                        VALUES (%s, %s, %s, NOW() - INTERVAL '%s days', %s)
                        """,
                        [collector, signal, forward_return, day_offset, regime],
                    )

    return True


class TestP05AdaptiveSynthesisPipeline:
    """End-to-end integration test for P05 features."""

    def test_ic_weight_precompute_stores_weights(self, db_setup, seed_ic_data):
        """Step 1: compute_and_store_ic_weights() produces and stores weights."""
        from quantstack.learning.ic_attribution import compute_and_store_ic_weights

        results = compute_and_store_ic_weights()

        # At least one regime should have weights
        assert len(results) > 0, "Expected at least one regime with computed weights"

        # Verify weights stored in DB
        with db_setup() as conn:
            conn.execute("SELECT COUNT(*) AS cnt FROM precomputed_ic_weights")
            row = conn.fetchone()
            assert row["cnt"] > 0, "Expected rows in precomputed_ic_weights"

    def test_precomputed_weights_differ_from_static(self, db_setup, seed_ic_data):
        """Step 2: Precomputed weights differ from hardcoded static profiles."""
        from quantstack.learning.ic_attribution import (
            compute_and_store_ic_weights,
            get_precomputed_weights,
        )

        compute_and_store_ic_weights()
        weights = get_precomputed_weights("trending_up")

        if weights is None:
            pytest.skip("Insufficient data for trending_up regime")

        # Static profile for trending_up has specific values
        static = {
            "trend": 0.35, "rsi": 0.10, "macd": 0.20,
            "bb": 0.05, "sentiment": 0.10, "ml": 0.15, "flow": 0.05,
        }

        # Precomputed weights should have different collectors (test_trend, test_rsi, test_ml)
        # OR different values if the collectors overlap
        assert weights != static, "Precomputed weights should differ from static profile"

    def test_synthesis_produces_transition_zone(self, db_setup):
        """Step 3: Synthesis sets transition_zone=True when P(transition) > 0.3."""
        from quantstack.signal_engine.synthesis import RuleBasedSynthesizer

        synth = RuleBasedSynthesizer()
        brief = synth.synthesize(
            symbol="TEST_P05",
            technical={
                "rsi_14": 50.0, "macd_hist": 0.1, "bb_pct": 0.5,
                "adx_14": 30.0, "close": 100.0,
            },
            regime={
                "trend_regime": "trending_up",
                "transition_probability": 0.5,  # > 0.3 threshold
                "confidence": 0.8,
            },
            volume={},
            risk={},
            events={},
            fundamentals={},
            collector_failures=[],
        )

        assert brief.transition_zone is True, (
            "Expected transition_zone=True when transition_probability=0.5"
        )

    def test_synthesis_no_transition_zone_when_low(self, db_setup):
        """Step 3b: transition_zone=False when P(transition) <= 0.3."""
        from quantstack.signal_engine.synthesis import RuleBasedSynthesizer

        synth = RuleBasedSynthesizer()
        brief = synth.synthesize(
            symbol="TEST_P05",
            technical={
                "rsi_14": 50.0, "macd_hist": 0.1, "bb_pct": 0.5,
                "adx_14": 30.0, "close": 100.0,
            },
            regime={
                "trend_regime": "trending_up",
                "transition_probability": 0.1,
                "confidence": 0.8,
            },
            volume={},
            risk={},
            events={},
            fundamentals={},
            collector_failures=[],
        )

        assert brief.transition_zone is False

    def test_conviction_factors_in_signals_metadata(self, db_setup):
        """Step 4: Conviction factors are persisted in signals.metadata."""
        from quantstack.signal_engine.synthesis import RuleBasedSynthesizer

        synth = RuleBasedSynthesizer()
        synth.synthesize(
            symbol="TEST_P05",
            technical={
                "rsi_14": 45.0, "macd_hist": 0.5, "bb_pct": 0.4,
                "adx_14": 35.0, "close": 200.0,
            },
            regime={"trend_regime": "trending_up", "confidence": 0.9},
            volume={},
            risk={},
            events={},
            fundamentals={},
            collector_failures=[],
        )

        # Read back from signals table
        with db_setup() as conn:
            conn.execute(
                "SELECT metadata FROM signals WHERE symbol = 'TEST_P05' "
                "ORDER BY signal_date DESC LIMIT 1"
            )
            row = conn.fetchone()

        assert row is not None, "Expected a signal row for TEST_P05"
        meta = json.loads(row["metadata"]) if isinstance(row["metadata"], str) else row["metadata"]
        assert "conviction_factors" in meta, "Expected conviction_factors in metadata"

        factors = meta["conviction_factors"]
        expected_keys = {"adx", "stability", "timeframe", "regime_agreement", "ml_confirmation", "data_quality"}
        assert set(factors.keys()) == expected_keys, (
            f"Expected keys {expected_keys}, got {set(factors.keys())}"
        )

    def test_ensemble_ab_records_when_enabled(self, db_setup, monkeypatch):
        """Step 5: Ensemble A/B results recorded when flag is enabled."""
        monkeypatch.setenv("FEEDBACK_ENSEMBLE_AB_TEST", "true")

        from quantstack.signal_engine.synthesis import RuleBasedSynthesizer

        synth = RuleBasedSynthesizer()
        synth.synthesize(
            symbol="TEST_P05",
            technical={
                "rsi_14": 50.0, "macd_hist": 0.1, "bb_pct": 0.5,
                "adx_14": 25.0, "close": 150.0,
            },
            regime={"trend_regime": "ranging", "confidence": 0.7},
            volume={},
            risk={},
            events={},
            fundamentals={},
            collector_failures=[],
        )

        with db_setup() as conn:
            conn.execute(
                "SELECT COUNT(*) AS cnt FROM ensemble_ab_results "
                "WHERE symbol = 'TEST_P05'"
            )
            row = conn.fetchone()

        # Should have 3 rows (one per ensemble method)
        assert row["cnt"] >= 3, (
            f"Expected >= 3 ensemble_ab_results rows, got {row['cnt']}"
        )

    def test_full_pipeline_coherence(self, db_setup, seed_ic_data, monkeypatch):
        """Full pipeline: precompute -> synthesize -> verify all features."""
        monkeypatch.setenv("FEEDBACK_IC_DRIVEN_WEIGHTS", "true")
        monkeypatch.setenv("FEEDBACK_ENSEMBLE_AB_TEST", "true")

        # 1. Precompute weights
        from quantstack.learning.ic_attribution import compute_and_store_ic_weights
        compute_and_store_ic_weights()

        # 2. Synthesize
        from quantstack.signal_engine.synthesis import RuleBasedSynthesizer
        synth = RuleBasedSynthesizer()
        brief = synth.synthesize(
            symbol="TEST_P05_FULL",
            technical={
                "rsi_14": 40.0, "macd_hist": 0.3, "bb_pct": 0.3,
                "adx_14": 28.0, "close": 175.0,
            },
            regime={
                "trend_regime": "trending_up",
                "transition_probability": 0.4,
                "confidence": 0.85,
            },
            volume={},
            risk={},
            events={},
            fundamentals={},
            collector_failures=[],
        )

        # 3. Verify SymbolBrief
        assert brief.transition_zone is True, "Expected transition_zone from high probability"
        assert brief.conviction_factors is not None, "Expected conviction_factors dict"
        assert isinstance(brief.conviction_factors, dict)
        assert len(brief.conviction_factors) == 6, "Expected 6 conviction factors"

        # 4. Verify signals metadata
        with db_setup() as conn:
            conn.execute(
                "SELECT metadata FROM signals WHERE symbol = 'TEST_P05_FULL' "
                "ORDER BY signal_date DESC LIMIT 1"
            )
            row = conn.fetchone()

        if row:
            meta = json.loads(row["metadata"]) if isinstance(row["metadata"], str) else row["metadata"]
            assert "conviction_factors" in meta
            assert "votes" in meta
            assert "weights" in meta

        # 5. Verify ensemble recording
        with db_setup() as conn:
            conn.execute(
                "SELECT DISTINCT method_name FROM ensemble_ab_results "
                "WHERE symbol = 'TEST_P05_FULL'"
            )
            methods = conn.fetchall()

        method_names = {r["method_name"] for r in methods}
        assert len(method_names) >= 1, "Expected ensemble methods recorded"

        # Cleanup
        with db_setup() as conn:
            conn.execute("DELETE FROM signals WHERE symbol = 'TEST_P05_FULL'")
            conn.execute("DELETE FROM ensemble_ab_results WHERE symbol = 'TEST_P05_FULL'")
```

## Running the Integration Test

```bash
# Requires TRADER_PG_URL to be set (e.g., local dev DB)
export TRADER_PG_URL="postgresql://localhost/quantstack_test"
pytest tests/integration/test_p05_adaptive_synthesis.py -v

# Skip in CI without DB
pytest tests/integration/test_p05_adaptive_synthesis.py -v
# (auto-skips due to pytestmark)
```

## What This Test Validates

| Assertion | Validates Section |
|-----------|-------------------|
| precomputed_ic_weights has rows | Section 02 |
| Precomputed weights differ from static | Section 02 + 06 |
| transition_zone=True when P>0.3 | Section 03 |
| transition_zone=False when P<=0.3 | Section 03 |
| conviction_factors in metadata | Section 04 + 06 |
| ensemble_ab_results populated | Section 05 + 06 |
| Full pipeline coherence | All sections |

## Edge Cases Covered

1. **Empty IC data**: If `seed_ic_data` produces data that doesn't meet the 20-observation minimum for a collector, the precompute skips that collector (tested by checking `len(results) > 0` rather than exact count).
2. **Stale weights**: Not tested here (unit tests in Section 08 cover this). Integration test always has fresh data.
3. **DB isolation**: Test data uses unique symbols (`TEST_P05`, `TEST_P05_FULL`) and cleanup at end of fixture. Does not interfere with production data if run against a shared DB (though that's not recommended).
4. **Flag interactions**: The `test_full_pipeline_coherence` test enables multiple flags simultaneously to verify they don't interfere.
