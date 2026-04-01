#!/usr/bin/env python3
"""
One-shot DB repair script.

Run before restarting the trading system to:
  1. Delete 9 paper strategies with plain-text entry_rules (never evaluatable)
  2. Fix 2 forward_testing strategies with mechanism-dict entry_rules (demote + set proper rules)
  3. Clear 46 zombie research_queue tasks (null topic, never runnable)
  4. Promote qqq_iron_condor_vrp_swing_v1 from validated → forward_testing
  5. Seed regime_states for SPY, QQQ, IWM from live signal engine

Usage:
    python3 scripts/db_repair.py
"""

from __future__ import annotations

import asyncio
import sys

from loguru import logger

from quantstack.db import open_db


def repair_strategies(conn) -> None:
    # --- 1. Delete paper strategies with non-array entry_rules ---
    result = conn.execute(
        """
        DELETE FROM strategies
        WHERE status = 'paper'
          AND (
            entry_rules::text NOT LIKE '[%%'
            OR entry_rules::text LIKE '%%20-day high%%'
            OR entry_rules::text LIKE '%%ROE > 15%%'
          )
        """
    )
    deleted = conn.execute("SELECT COUNT(*) FROM strategies WHERE 1=0").fetchone()  # noqa
    # psycopg2 rowcount
    rows_deleted = conn._cur.rowcount if hasattr(conn, "_cur") else "?"
    conn.commit()
    logger.info(f"[repair] Deleted {rows_deleted} plain-text paper strategies")

    # --- 2. Fix forward_testing strategies with mechanism-dict entry_rules ---
    # These have {"mechanism": "..."} (single dict) instead of [{...}, ...] arrays.
    # Demote to draft and set structured rules so they can be re-validated.

    conn.execute(
        """
        UPDATE strategies SET
            status = 'draft',
            entry_rules = '[
                {"indicator": "close", "condition": "above", "value": "sma_50", "type": "confirmation"},
                {"indicator": "rsi", "condition": "below", "value": 60, "type": "confirmation"},
                {"indicator": "volume", "condition": "above", "value": 1.2, "type": "confirmation"}
            ]'::jsonb
        WHERE name = 'Directional Bullish Call Spread'
          AND entry_rules::text NOT LIKE '[%%'
        """
    )
    conn.commit()
    logger.info("[repair] Fixed 'Directional Bullish Call Spread' entry_rules → draft")

    conn.execute(
        """
        UPDATE strategies SET
            status = 'draft',
            entry_rules = '[
                {"indicator": "iv_percentile", "condition": "above", "value": 50, "type": "prerequisite"},
                {"indicator": "adx", "condition": "below", "value": 30, "type": "confirmation"},
                {"indicator": "bb_pct", "condition": "between", "lower": 0.3, "upper": 0.7, "type": "confirmation"}
            ]'::jsonb
        WHERE name = 'VRP Harvest - IV Compression'
          AND entry_rules::text NOT LIKE '[%%'
        """
    )
    conn.commit()
    logger.info("[repair] Fixed 'VRP Harvest - IV Compression' entry_rules → draft")

    # --- 3. Promote qqq_iron_condor_vrp_swing_v1 to forward_testing ---
    conn.execute(
        """
        UPDATE strategies SET status = 'forward_testing'
        WHERE name = 'qqq_iron_condor_vrp_swing_v1' AND status = 'validated'
        """
    )
    conn.commit()
    logger.info("[repair] Promoted qqq_iron_condor_vrp_swing_v1 → forward_testing")


def clear_zombie_queue(conn) -> None:
    result = conn.execute(
        "DELETE FROM research_queue WHERE status = 'pending' AND topic IS NULL"
    )
    rows_deleted = conn._cur.rowcount if hasattr(conn, "_cur") else "?"
    conn.commit()
    logger.info(f"[repair] Cleared {rows_deleted} zombie research_queue tasks (null topic)")


async def seed_regime_states(conn) -> None:
    from quantstack.signal_engine.engine import SignalEngine

    engine = SignalEngine(conn)
    symbols = ["SPY", "QQQ", "IWM"]

    for sym in symbols:
        try:
            brief = await engine.run(sym)
            signals = getattr(brief, "signals", {}) or {}
            regime_sig = signals.get("regime", {}) or {}
            tech_sig = signals.get("technical", {}) or {}

            trend = regime_sig.get("trend_regime") or brief.regime if hasattr(brief, "regime") else "unknown"
            vol_regime = regime_sig.get("volatility_regime", "normal")
            adx_val = float(tech_sig.get("adx", 0.0) or 0.0)
            confidence = float(getattr(brief, "regime_confidence", 0.5) or 0.5)

            conn.execute(
                """
                INSERT INTO regime_states
                  (timestamp, symbol, timeframe, trend_regime, volatility_regime,
                   adx, confidence, source_agent)
                VALUES (NOW(), %s, '1D', %s, %s, %s, %s, 'db_repair')
                ON CONFLICT DO NOTHING
                """,
                [sym, trend, vol_regime, adx_val, confidence],
            )
            conn.commit()
            logger.info(f"[repair] Seeded regime_states: {sym} → trend={trend}, vol={vol_regime}")
        except Exception as exc:
            logger.warning(f"[repair] Could not seed regime for {sym}: {exc} (non-fatal)")


def print_summary(conn) -> None:
    logger.info("\n=== POST-REPAIR SUMMARY ===")

    rows = conn.execute(
        "SELECT status, COUNT(*) FROM strategies GROUP BY status ORDER BY status"
    ).fetchall()
    logger.info("Strategies by status:")
    for r in rows:
        logger.info(f"  {r[0]}: {r[1]}")

    fwd = conn.execute(
        "SELECT name FROM strategies WHERE status = 'forward_testing'"
    ).fetchall()
    logger.info(f"Forward-testing strategies: {[r[0] for r in fwd]}")

    queue = conn.execute(
        "SELECT status, COUNT(*) FROM research_queue GROUP BY status"
    ).fetchall()
    logger.info(f"Research queue: {dict(queue)}")

    regime = conn.execute(
        "SELECT symbol, trend_regime, volatility_regime FROM regime_states"
    ).fetchall()
    logger.info(f"Regime states seeded: {[(r[0], r[1], r[2]) for r in regime]}")


def main() -> None:
    logger.info("[repair] Starting DB repair...")
    conn = open_db()

    try:
        repair_strategies(conn)
        clear_zombie_queue(conn)
        asyncio.run(seed_regime_states(conn))
        print_summary(conn)
        logger.info("[repair] Done. System is ready to start.")
    except Exception as exc:
        logger.error(f"[repair] Failed: {exc}")
        sys.exit(1)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
