# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""Cross-domain intelligence — surfaces signals from other research domains.

Each research domain (equity investment, equity swing, options) produces artifacts
that benefit the others. This tool queries across domains and returns structured
intel items with action suggestions and relevance scores.

Investment → Swing: fundamental floor, thesis status
Investment → Options: directional thesis, catalyst timeline
Swing → Investment: technical levels, momentum confirmation
Swing → Options: breakout levels, momentum timing
Options → Investment: IV surface, institutional flow
Options → Swing: GEX levels, vol regime

Tools:
  - get_cross_domain_intel — query cross-domain signals for a domain
"""

from datetime import datetime
from typing import Any

from loguru import logger

from quantstack.mcp._state import live_db_or_error
from quantstack.mcp.server import mcp
from quantstack.mcp.domains import Domain
from quantstack.mcp.tools._registry import domain


# ── Domain constants ─────────────────────────────────────────────────────

_DOMAINS = ("equity_investment", "equity_swing", "options")


# ── Intel item builders ──────────────────────────────────────────────────

def _build_fundamental_floor(alert: dict, target: str) -> dict:
    """Investment alert → fundamental stop floor for swing/options."""
    entry = alert.get("suggested_entry") or alert.get("current_price") or 0
    stop = alert.get("stop_price") or 0
    f_score = alert.get("piotroski_f_score")
    fcf = alert.get("fcf_yield_pct")

    return {
        "source_domain": "equity_investment",
        "target_domain": target,
        "intel_type": "fundamental_floor",
        "symbol": alert["symbol"],
        "headline": (
            f"{alert['symbol']}: fundamental floor at ${stop:.0f} "
            f"(F-Score={f_score or '?'}, FCF yield={fcf or '?'}%)"
        ),
        "data": {
            "stop_price": stop,
            "entry_price": entry,
            "piotroski_f_score": f_score,
            "fcf_yield_pct": fcf,
            "pe_ratio": alert.get("pe_ratio"),
            "alert_id": alert["id"],
        },
        "action_suggestion": (
            f"Use ${stop:.0f} as stop floor if between 1-2.5x ATR from current price"
        ),
        "relevance": 0.8 if f_score and f_score >= 7 else 0.5,
    }


def _build_thesis_status(alert: dict, thesis_status: str, target: str) -> dict:
    """Investment alert → thesis status for swing/options."""
    relevance_map = {"intact": 0.6, "strengthening": 0.7, "weakening": 0.9, "broken": 1.0}
    action_map = {
        "intact": "Fundamental thesis supports directional bias",
        "strengthening": "Thesis improving — higher conviction entries",
        "weakening": "Caution — tighten stops, reduce new position sizes by 50%",
        "broken": "AVOID new longs. Exit existing positions.",
    }
    return {
        "source_domain": "equity_investment",
        "target_domain": target,
        "intel_type": "thesis_status",
        "symbol": alert["symbol"],
        "headline": f"{alert['symbol']}: investment thesis is {thesis_status}",
        "data": {
            "thesis_status": thesis_status,
            "confidence": alert.get("confidence", 0),
            "thesis_summary": (alert.get("thesis") or "")[:200],
            "alert_id": alert["id"],
        },
        "action_suggestion": action_map.get(thesis_status, ""),
        "relevance": relevance_map.get(thesis_status, 0.5),
    }


def _build_fundamental_event(alert: dict, target: str) -> dict:
    """Investment alert with catalyst → event intel for swing/options."""
    return {
        "source_domain": "equity_investment",
        "target_domain": target,
        "intel_type": "fundamental_event",
        "symbol": alert["symbol"],
        "headline": f"{alert['symbol']}: catalyst — {alert.get('catalyst', 'unknown')}",
        "data": {
            "catalyst": alert.get("catalyst", ""),
            "thesis_direction": alert.get("action", ""),
            "confidence": alert.get("confidence", 0),
            "alert_id": alert["id"],
        },
        "action_suggestion": (
            "Check IV rank before options entry. "
            "Position swing trades before catalyst if thesis supports."
        ),
        "relevance": 0.75,
    }


def _build_technical_levels(alert: dict, target: str) -> dict:
    """Swing alert → technical levels for investment/options."""
    return {
        "source_domain": "equity_swing",
        "target_domain": target,
        "intel_type": "technical_levels",
        "symbol": alert["symbol"],
        "headline": (
            f"{alert['symbol']}: swing levels — "
            f"entry ${alert.get('suggested_entry', 0):.0f}, "
            f"stop ${alert.get('stop_price', 0):.0f}, "
            f"target ${alert.get('target_price', 0):.0f}"
        ),
        "data": {
            "entry": alert.get("suggested_entry"),
            "stop": alert.get("stop_price"),
            "target": alert.get("target_price"),
            "regime": alert.get("regime", "unknown"),
            "alert_id": alert["id"],
        },
        "action_suggestion": (
            "Use for entry timing (investment) or strike selection (options)"
        ),
        "relevance": 0.7 if alert.get("confidence", 0) > 0.6 else 0.4,
    }


def _build_momentum_signal(alert: dict, target: str) -> dict:
    """Swing alert → momentum signal for investment/options."""
    action = alert.get("action", "buy")
    direction = "bullish" if action == "buy" else "bearish"
    return {
        "source_domain": "equity_swing",
        "target_domain": target,
        "intel_type": "momentum_signal",
        "symbol": alert["symbol"],
        "headline": f"{alert['symbol']}: swing momentum is {direction} (conviction {alert.get('confidence', 0):.0%})",
        "data": {
            "direction": direction,
            "confidence": alert.get("confidence", 0),
            "regime": alert.get("regime", "unknown"),
            "alert_id": alert["id"],
        },
        "action_suggestion": (
            f"{'Confirms' if direction == 'bullish' else 'Contradicts'} bullish thesis. "
            f"{'Size up' if direction == 'bullish' else 'Delay entry or reduce size'}."
        ),
        "relevance": 0.65,
    }


def _compute_convergence(symbol: str, items: list[dict]) -> dict:
    """Analyze multi-domain alignment for a symbol."""
    domains = list({i["source_domain"] for i in items})
    directions = []
    for item in items:
        data = item.get("data", {})
        if item["intel_type"] == "thesis_status":
            status = data.get("thesis_status", "")
            if status in ("intact", "strengthening"):
                directions.append("bullish")
            elif status == "broken":
                directions.append("bearish")
        elif item["intel_type"] == "momentum_signal":
            directions.append(data.get("direction", "neutral"))

    bullish = sum(1 for d in directions if d == "bullish")
    bearish = sum(1 for d in directions if d == "bearish")

    if bullish > bearish:
        alignment = "bullish"
    elif bearish > bullish:
        alignment = "bearish"
    else:
        alignment = "mixed"

    conflicts = []
    if bullish > 0 and bearish > 0:
        conflicts.append(f"{bullish} bullish vs {bearish} bearish signals")

    return {
        "symbol": symbol,
        "domains_active": domains,
        "alignment": alignment,
        "signal_count": len(items),
        "conflicts": conflicts,
    }


# ── Main tool ────────────────────────────────────────────────────────────

@domain(Domain.INTEL)
@mcp.tool()
async def get_cross_domain_intel(
    symbol: str = "",
    requesting_domain: str = "",
    include_stale: bool = False,
) -> dict[str, Any]:
    """
    Query cross-domain intelligence — surfaces signals from other research domains.

    Each domain produces artifacts that benefit the others. This tool reads
    across domains and returns structured intel items with action suggestions.

    Args:
        symbol: Filter to one symbol. Empty = all active symbols.
        requesting_domain: "equity_investment", "equity_swing", "options", or "" (all).
                           Filters intel to what's relevant for THIS domain.
        include_stale: Include intel from alerts older than 14 days.

    Returns:
        Dict with intel_items (sorted by relevance), summary, and symbol_convergence.
    """
    ctx, err = live_db_or_error()
    if err:
        return err
    try:
        intel_items: list[dict] = []
        cutoff_days = 90 if include_stale else 14

        # Build symbol filter clause
        sym_clause = ""
        sym_params: list[Any] = []
        if symbol:
            sym_clause = " AND ea.symbol = ?"
            sym_params = [symbol.upper()]

        # ── 1. Investment → Swing/Options ────────────────────────────
        if requesting_domain in ("equity_swing", "options", ""):
            rows = ctx.db.execute(
                f"""
                SELECT ea.id, ea.symbol, ea.action, ea.confidence,
                       ea.current_price, ea.suggested_entry, ea.stop_price, ea.target_price,
                       ea.regime, ea.catalyst, ea.thesis, ea.key_risks,
                       ea.piotroski_f_score, ea.fcf_yield_pct, ea.pe_ratio,
                       ea.created_at
                FROM equity_alerts ea
                WHERE ea.time_horizon = 'investment'
                  AND ea.status IN ('pending', 'watching', 'acted')
                  AND ea.created_at >= CURRENT_TIMESTAMP - INTERVAL '{cutoff_days}' DAY
                  {sym_clause}
                ORDER BY ea.confidence DESC
                LIMIT 20
                """,
                sym_params,
            ).fetchall()

            inv_cols = [
                "id", "symbol", "action", "confidence",
                "current_price", "suggested_entry", "stop_price", "target_price",
                "regime", "catalyst", "thesis", "key_risks",
                "piotroski_f_score", "fcf_yield_pct", "pe_ratio", "created_at",
            ]

            for row in rows:
                alert = dict(zip(inv_cols, row))

                # Get latest thesis_status from alert_updates
                ts_row = ctx.db.execute(
                    "SELECT thesis_status FROM alert_updates WHERE alert_id = ? ORDER BY created_at DESC LIMIT 1",
                    [alert["id"]],
                ).fetchone()
                thesis_status = ts_row[0] if ts_row else "intact"

                targets = []
                if requesting_domain == "equity_swing":
                    targets = ["equity_swing"]
                elif requesting_domain == "options":
                    targets = ["options"]
                else:
                    targets = ["equity_swing", "options"]

                for tgt in targets:
                    if alert.get("stop_price") and alert["stop_price"] > 0:
                        intel_items.append(_build_fundamental_floor(alert, tgt))
                    intel_items.append(_build_thesis_status(alert, thesis_status, tgt))
                    if alert.get("catalyst"):
                        intel_items.append(_build_fundamental_event(alert, tgt))

        # ── 2. Swing → Investment/Options ────────────────────────────
        if requesting_domain in ("equity_investment", "options", ""):
            rows = ctx.db.execute(
                f"""
                SELECT ea.id, ea.symbol, ea.action, ea.confidence,
                       ea.current_price, ea.suggested_entry, ea.stop_price, ea.target_price,
                       ea.regime, ea.created_at
                FROM equity_alerts ea
                WHERE ea.time_horizon IN ('swing', 'position')
                  AND ea.status IN ('pending', 'watching', 'acted')
                  AND ea.created_at >= CURRENT_TIMESTAMP - INTERVAL '{cutoff_days}' DAY
                  {sym_clause}
                ORDER BY ea.confidence DESC
                LIMIT 20
                """,
                sym_params,
            ).fetchall()

            swing_cols = [
                "id", "symbol", "action", "confidence",
                "current_price", "suggested_entry", "stop_price", "target_price",
                "regime", "created_at",
            ]

            for row in rows:
                alert = dict(zip(swing_cols, row))
                targets = []
                if requesting_domain == "equity_investment":
                    targets = ["equity_investment"]
                elif requesting_domain == "options":
                    targets = ["options"]
                else:
                    targets = ["equity_investment", "options"]

                for tgt in targets:
                    if alert.get("stop_price") and alert.get("target_price"):
                        intel_items.append(_build_technical_levels(alert, tgt))
                    intel_items.append(_build_momentum_signal(alert, tgt))

        # ── 3. Options strategies → Investment/Swing ─────────────────
        # Options intel comes from strategies table (validated/live options strategies)
        if requesting_domain in ("equity_investment", "equity_swing", ""):
            try:
                strat_rows = ctx.db.execute(
                    """
                    SELECT strategy_id, name, status, regime_affinity
                    FROM strategies
                    WHERE instrument_type = 'options' AND status IN ('validated', 'live', 'forward_testing')
                    LIMIT 10
                    """
                ).fetchall()
                if strat_rows:
                    targets = []
                    if requesting_domain == "equity_investment":
                        targets = ["equity_investment"]
                    elif requesting_domain == "equity_swing":
                        targets = ["equity_swing"]
                    else:
                        targets = ["equity_investment", "equity_swing"]

                    for tgt in targets:
                        intel_items.append({
                            "source_domain": "options",
                            "target_domain": tgt,
                            "intel_type": "options_strategies_active",
                            "symbol": symbol or "portfolio",
                            "headline": f"{len(strat_rows)} active options strategies — check IV surface before equity entries",
                            "data": {
                                "strategy_count": len(strat_rows),
                                "strategies": [
                                    {"id": r[0], "name": r[1], "status": r[2]}
                                    for r in strat_rows[:5]
                                ],
                            },
                            "action_suggestion": (
                                "Active options strategies indicate vol awareness. "
                                "Check IV rank via get_iv_surface before sizing equity positions."
                            ),
                            "relevance": 0.5,
                        })
            except Exception as e:
                logger.warning(f"[cross_domain] options strategy query failed (schema may be missing instrument_type column): {e}")

        # ── 4. Convergence analysis ──────────────────────────────────
        symbol_groups: dict[str, list[dict]] = {}
        for item in intel_items:
            sym = item.get("symbol", "")
            if sym and sym != "portfolio":
                symbol_groups.setdefault(sym, []).append(item)

        convergence = []
        for sym, items in symbol_groups.items():
            domains = {i["source_domain"] for i in items}
            if len(domains) >= 2:
                convergence.append(_compute_convergence(sym, items))

        # ── 5. Summary ──────────────────────────────────────────────
        summary: dict[str, dict] = {}
        for domain_name, horizon_filter in [
            ("equity_investment", "= 'investment'"),
            ("equity_swing", "IN ('swing', 'position')"),
        ]:
            try:
                count_row = ctx.db.execute(
                    f"""
                    SELECT COUNT(*) FROM equity_alerts
                    WHERE time_horizon {horizon_filter}
                      AND status IN ('pending', 'watching', 'acted')
                    """
                ).fetchone()
                summary[domain_name] = {"active_alerts": count_row[0] if count_row else 0}
            except Exception as e:
                logger.warning(f"[cross_domain] alert count query failed for {domain_name}: {e}")
                summary[domain_name] = {"active_alerts": 0}

        try:
            opt_count = ctx.db.execute(
                "SELECT COUNT(*) FROM strategies WHERE instrument_type = 'options' AND status IN ('validated', 'live')"
            ).fetchone()
            summary["options"] = {"active_strategies": opt_count[0] if opt_count else 0}
        except Exception as e:
            logger.warning(f"[cross_domain] options strategy count query failed: {e}")
            summary["options"] = {"active_strategies": 0}

        # Sort by relevance
        intel_items.sort(key=lambda x: x.get("relevance", 0), reverse=True)

        return {
            "success": True,
            "symbol": symbol,
            "requesting_domain": requesting_domain,
            "generated_at": datetime.now().isoformat(),
            "intel_items": intel_items,
            "intel_count": len(intel_items),
            "summary": summary,
            "symbol_convergence": convergence,
        }
    except Exception as e:
        logger.error(f"[cross_domain] get_cross_domain_intel failed: {e}")
        return {"success": False, "error": str(e)}
