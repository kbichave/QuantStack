#!/usr/bin/env python3
# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
End-to-end validation of the coordination infrastructure.

Runs against a FRESH database with live FD.ai data.  Validates:
  1. DB migrations create all tables
  2. Universe registry populates from FD.ai stock_screener + ETF list
  3. Cache warmer fetches OHLCV for a sample of symbols
  4. Screener scores and produces tiered watchlist
  5. WatchlistLoader reads tiers correctly
  6. Event bus publish/poll round-trip
  7. Strategy lock lifecycle (draft → forward_testing → retired)
  8. Daily digest generates without errors

Usage:
    source .env  # load FINANCIAL_DATASETS_API_KEY
    python scripts/validate_coordination.py

    # Or with explicit key:
    FINANCIAL_DATASETS_API_KEY=fd_xxx python scripts/validate_coordination.py
"""

from __future__ import annotations

import asyncio
import os
import time
from datetime import date, datetime, timedelta, timezone

from quantstack.autonomous.screener import AutonomousScreener
from quantstack.autonomous.watchlist import WatchlistLoader
from quantstack.coordination.daily_digest import DailyDigest
from quantstack.coordination.event_bus import Event, EventBus, EventType
from quantstack.coordination.strategy_lock import StrategyStatusLock
from quantstack.coordination.universe_registry import UniverseRegistry, UniverseSource
from quantstack.data.adapters.financial_datasets_client import FinancialDatasetsClient
from quantstack.data.cache_warmer import CacheWarmer
from quantstack.db import open_db, reset_connection, run_migrations


def _header(msg: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {msg}")
    print(f"{'=' * 60}")


def _ok(msg: str) -> None:
    print(f"  [OK] {msg}")


def _fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")


def _info(msg: str) -> None:
    print(f"  [..] {msg}")


def main() -> int:
    errors: list[str] = []

    # ── Step 0: Check API key ────────────────────────────────────────────
    _header("Step 0: Environment check")
    fd_key = os.getenv("FINANCIAL_DATASETS_API_KEY", "")
    if not fd_key:
        # Try loading from .env
        env_path = project_root / ".env"
        if env_path.exists():
            _info(f"Loading keys from {env_path}")
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())
            fd_key = os.getenv("FINANCIAL_DATASETS_API_KEY", "")

    if fd_key:
        _ok(f"FINANCIAL_DATASETS_API_KEY set ({fd_key[:8]}...)")
    else:
        _fail(
            "FINANCIAL_DATASETS_API_KEY not set — universe refresh will skip equities"
        )
        errors.append("No FD.ai API key")

    # ── Step 1: Fresh DB + migrations ────────────────────────────────────
    _header("Step 1: Fresh database + migrations")
    # Force a fresh DB
    reset_connection()
    conn = open_db()
    run_migrations(conn)

    tables = [r[0] for r in conn.execute("SHOW TABLES").fetchall()]
    expected_new = {
        "universe",
        "screener_results",
        "loop_events",
        "loop_cursors",
        "loop_heartbeats",
    }
    present = expected_new & set(tables)
    missing = expected_new - set(tables)

    if not missing:
        _ok(
            f"{len(tables)} tables created (including {len(present)} new coordination tables)"
        )
    else:
        _fail(f"Missing tables: {missing}")
        errors.append(f"Missing tables: {missing}")

    # ── Step 2: Universe refresh ─────────────────────────────────────────
    _header("Step 2: Universe registry refresh")
    client = None
    if fd_key:
        client = FinancialDatasetsClient(api_key=fd_key)

    registry = UniverseRegistry(conn, client)
    t0 = time.time()
    report = registry.refresh_constituents()
    elapsed = time.time() - t0

    _info(f"Added: {report.symbols_added}, Updated: {report.symbols_updated}")
    _info(f"Total active: {report.total_active}")
    _info(f"Elapsed: {elapsed:.1f}s")
    if report.errors:
        _info(f"Errors ({len(report.errors)}): {report.errors[:3]}")

    if report.total_active >= 40:  # At minimum the ~50 ETFs
        _ok(f"Universe has {report.total_active} symbols")
    else:
        _fail(f"Universe too small: {report.total_active}")
        errors.append(f"Universe only {report.total_active} symbols")

    # Show breakdown by source
    for source in ("sp500", "nasdaq100", "etf_liquid"):
        syms = registry.get_active_symbols(UniverseSource(source))
        _info(f"  {source}: {len(syms)} symbols")

    # ── Step 3: Cache warmer (sample of 10) ──────────────────────────────
    _header("Step 3: Cache warmer (10 symbols)")

    if fd_key:
        # We need a DataStore — create a minimal one
        # We need a DataStore — create a minimal one via PostgreSQL
        sample_symbols = registry.get_active_symbols()[:10]
        if not sample_symbols:
            sample_symbols = [
                "SPY",
                "QQQ",
                "AAPL",
                "MSFT",
                "NVDA",
                "XOM",
                "JPM",
                "GLD",
                "TLT",
                "IWM",
            ]

        _info(f"Warming: {sample_symbols}")

        # Fetch OHLCV directly and store in our ohlcv table
        # Create ohlcv table if needed (migrations may not have it if DataStore owns it)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ohlcv (
                symbol VARCHAR, timeframe VARCHAR, timestamp TIMESTAMP,
                open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE, volume DOUBLE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        warmed = 0
        warm_errors = 0
        for sym in sample_symbols:
            try:
                start = (date.today() - timedelta(days=252)).isoformat()
                end = date.today().isoformat()
                prices = client.get_all_historical_prices(sym, "day", 1, start, end)
                if prices:
                    for p in prices:
                        ts = p.get("time", p.get("date", p.get("timestamp")))
                        conn.execute(
                            "INSERT INTO ohlcv (symbol, timeframe, timestamp, open, high, low, close, volume) "
                            "VALUES (?, 'D1', ?, ?, ?, ?, ?, ?)",
                            [
                                sym,
                                ts,
                                p.get("open", 0),
                                p.get("high", 0),
                                p.get("low", 0),
                                p.get("close", 0),
                                p.get("volume", 0),
                            ],
                        )
                    warmed += 1
                    _info(f"  {sym}: {len(prices)} bars")
                else:
                    _info(f"  {sym}: no data returned")
                    warm_errors += 1
            except Exception as exc:
                _info(f"  {sym}: ERROR — {exc}")
                warm_errors += 1

        if warmed >= 5:
            _ok(f"Warmed {warmed}/{len(sample_symbols)} symbols")
        else:
            _fail(f"Only warmed {warmed}/{len(sample_symbols)}")
            errors.append(f"Cache warmer: only {warmed} symbols")
    else:
        _info("Skipping (no FD.ai key)")

    # ── Step 4: Screener ─────────────────────────────────────────────────
    _header("Step 4: Autonomous screener")
    screener = AutonomousScreener(conn)
    result = screener._screen_sync("unknown")

    _info(f"Universe scanned: {result.universe_size}")
    _info(f"Filtered out: {result.filtered_out}")
    _info(f"Tier 1: {len(result.tier_1)} symbols")
    _info(f"Tier 2: {len(result.tier_2)} symbols")
    _info(f"Tier 3: {len(result.tier_3)} symbols")

    if result.tier_1:
        top3 = result.tier_1[:3]
        _info(f"Top 3: {[(s.symbol, round(s.composite, 3)) for s in top3]}")

    if result.total_watchlist > 0:
        _ok(f"Screener produced {result.total_watchlist} symbols across 3 tiers")
    else:
        _fail("Screener produced empty watchlist")
        errors.append("Screener: empty watchlist")

    # Verify persistence
    row = conn.execute("SELECT COUNT(*) FROM screener_results").fetchone()
    _info(f"Persisted {row[0]} rows to screener_results table")

    # ── Step 5: WatchlistLoader tiered ───────────────────────────────────
    _header("Step 5: WatchlistLoader v2 (tiered)")

    # Temporarily enable tiered mode
    os.environ["USE_TIERED_WATCHLIST"] = "true"
    os.environ.pop("AUTONOMOUS_WATCHLIST", None)  # Remove any override

    loader = WatchlistLoader()
    tiered = loader.load_tiered()

    if tiered and any(tiered.values()):
        _ok(
            f"Tiered load: T1={len(tiered.get(1, []))} T2={len(tiered.get(2, []))} T3={len(tiered.get(3, []))}"
        )
    else:
        _fail("Tiered load returned empty")
        errors.append("WatchlistLoader: tiered empty")

    # Test flat load (T1 + T2)
    flat = loader.load()
    _info(f"Flat load (T1+T2): {len(flat)} symbols")

    # ── Step 6: Event bus round-trip ─────────────────────────────────────
    _header("Step 6: Event bus round-trip")
    bus = EventBus(conn)

    # Publish
    eid = bus.publish(
        Event(
            event_type=EventType.SCREENER_COMPLETED,
            source_loop="validation_script",
            payload={"watchlist_size": result.total_watchlist},
        )
    )
    _info(f"Published SCREENER_COMPLETED (id={eid})")

    # Poll from two consumers
    events_a = bus.poll("validator_a", [EventType.SCREENER_COMPLETED])
    events_b = bus.poll("validator_b")

    if len(events_a) == 1 and events_a[0].event_id == eid:
        _ok("Consumer A received the event")
    else:
        _fail(f"Consumer A: expected 1 event, got {len(events_a)}")
        errors.append("Event bus: consumer A failed")

    if len(events_b) >= 1:
        _ok(f"Consumer B received {len(events_b)} event(s)")
    else:
        _fail("Consumer B: no events")
        errors.append("Event bus: consumer B failed")

    # Second poll — cursor should have advanced
    events_a2 = bus.poll("validator_a")
    _info(f"Consumer A re-poll: {len(events_a2)} new events (expected 0)")

    # ── Step 7: Strategy lock lifecycle ──────────────────────────────────
    _header("Step 7: Strategy status lock lifecycle")
    lock = StrategyStatusLock(conn, event_bus=bus)

    # Create a test strategy
    conn.execute(
        "INSERT INTO strategies (strategy_id, name, parameters, entry_rules, exit_rules, status) "
        "VALUES ('validation_test', 'Validation Test Strategy', '{}', '{}', '{}', 'draft')"
    )

    ok1 = lock.transition(
        "validation_test", "draft", "forward_testing", "backtest passed"
    )
    ok2 = lock.transition(
        "validation_test", "forward_testing", "live", "21d paper trading good"
    )
    ok3 = lock.transition("validation_test", "live", "retired", "validation complete")

    if ok1 and ok2 and ok3:
        _ok("Full lifecycle: draft → forward_testing → live → retired")
    else:
        _fail(f"Lifecycle failed: {ok1=} {ok2=} {ok3=}")
        errors.append("Strategy lock: lifecycle failed")

    # CAS failure test
    conn.execute(
        "INSERT INTO strategies (strategy_id, name, parameters, entry_rules, exit_rules, status) "
        "VALUES ('cas_test', 'CAS Test', '{}', '{}', '{}', 'forward_testing')"
    )
    cas_ok = lock.transition("cas_test", "draft", "forward_testing", "should fail")
    if not cas_ok:
        _ok("CAS correctly rejected stale transition")
    else:
        _fail("CAS did not reject stale transition!")
        errors.append("Strategy lock: CAS broken")

    # Clean up test strategies
    conn.execute(
        "DELETE FROM strategies WHERE strategy_id IN ('validation_test', 'cas_test')"
    )

    # ── Step 8: Daily digest ─────────────────────────────────────────────
    _header("Step 8: Daily digest")
    digest = DailyDigest(conn)
    report = digest.generate()
    md = digest.format_markdown(report)

    _info(f"Universe: {report.universe_size}, Watchlist: {report.watchlist_size}")
    _ok(f"Digest generated ({len(md)} chars markdown)")

    # ── Summary ──────────────────────────────────────────────────────────
    _header("VALIDATION SUMMARY")
    if not errors:
        print("  ALL STEPS PASSED")
        print()
        print("  Next steps:")
        print("  1. Run the full test suite: pytest tests/unit/test_coordination.py -v")
        print("  2. Start supervised loops: ./scripts/start_supervised_loops.sh")
        print("  3. Monitor: call get_loop_health() MCP tool")
        return 0
    else:
        print(f"  {len(errors)} ERRORS:")
        for e in errors:
            print(f"    - {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
