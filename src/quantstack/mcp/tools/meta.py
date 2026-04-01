"""Phase 5 — Meta Orchestration tools for the QuantPod MCP server.

Portfolio-level tools for regime-strategy allocation, signal conflict
resolution, strategy gap analysis, and automated promotion.

Tools:
  - get_regime_strategies        — get strategy allocations for a regime
  - set_regime_allocation        — set/update regime-strategy allocation matrix
  - resolve_portfolio_conflicts  — resolve signal conflicts across strategies
  - get_strategy_gaps            — analyze strategy registry for coverage gaps
  - promote_draft_strategies     — auto-promote drafts to forward_testing

Note: run_multi_analysis (CrewAI-based) was removed in v0.6.0.
      Use ``run_multi_signal_brief`` (SignalEngine) instead.
"""

import json
from datetime import date, datetime, timedelta, timezone
from typing import Any

from loguru import logger

from quantstack.db import pg_conn
from quantstack.features.enricher import FeatureEnricher
from quantstack.mcp._state import (
    _serialize,
    live_db_or_error,
    require_ctx,
    require_live_db,
)
from quantstack.mcp.allocation import resolve_conflicts
from quantstack.mcp.tools._tool_def import tool_def
from quantstack.mcp.domains import Domain
from quantstack.mcp.tools._registry import domain
from quantstack.strategies.signal_generator import (
    fetch_price_data as _fetch_price_data,
    generate_signals_from_rules as _generate_signals_from_rules,
)


@domain(Domain.RESEARCH)
@tool_def()
async def get_regime_strategies(regime: str) -> dict[str, Any]:
    """
    Get strategy allocations for a given regime from the matrix.

    Args:
        regime: Regime label (e.g., "trending_up", "ranging").

    Returns:
        Dict with list of (strategy_id, allocation_pct, confidence).
    """
    _, err = live_db_or_error()
    if err:
        return err
    try:
        with pg_conn() as conn:
            rows = conn.execute(
                "SELECT strategy_id, allocation_pct, confidence, last_updated "
                "FROM regime_strategy_matrix WHERE regime = ? ORDER BY allocation_pct DESC",
                [regime],
            ).fetchall()

        allocations = [
            {
                "strategy_id": r[0],
                "allocation_pct": r[1],
                "confidence": r[2],
                "last_updated": str(r[3]) if r[3] else None,
            }
            for r in rows
        ]
        return {
            "success": True,
            "regime": regime,
            "allocations": allocations,
            "total": len(allocations),
        }
    except Exception as e:
        logger.error(f"[quantpod_mcp] get_regime_strategies failed: {e}")
        return {"success": False, "error": str(e)}


@domain(Domain.RESEARCH)
@tool_def()
async def set_regime_allocation(
    regime: str,
    allocations: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Set or update strategy allocations for a regime.

    Upserts into the regime_strategy_matrix. This is how /reflect updates
    the matrix based on accumulated performance data.

    Args:
        regime: Regime label.
        allocations: List of dicts with strategy_id, allocation_pct, confidence (optional).

    Returns:
        Confirmation with the updated allocations.
    """
    _, err = live_db_or_error()
    if err:
        return err
    try:
        # Validate total allocation <= 1.0
        total = sum(a.get("allocation_pct", 0) for a in allocations)
        if total > 1.0:
            return {
                "success": False,
                "error": f"Total allocation {total:.0%} exceeds 100%. Reduce allocations.",
            }

        with pg_conn() as conn:
            for alloc in allocations:
                strategy_id = alloc.get("strategy_id")
                allocation_pct = alloc.get("allocation_pct", 0)
                confidence = alloc.get("confidence", 0.5)

                if not strategy_id:
                    continue

                # Upsert: try update, then insert
                conn.execute(
                    "UPDATE regime_strategy_matrix "
                    "SET allocation_pct = ?, confidence = ?, last_updated = CURRENT_TIMESTAMP "
                    "WHERE regime = ? AND strategy_id = ?",
                    [allocation_pct, confidence, regime, strategy_id],
                )

                # Check if row existed
                exists = conn.execute(
                    "SELECT 1 FROM regime_strategy_matrix WHERE regime = ? AND strategy_id = ?",
                    [regime, strategy_id],
                ).fetchone()

                if not exists:
                    conn.execute(
                        "INSERT INTO regime_strategy_matrix (regime, strategy_id, allocation_pct, confidence) "
                        "VALUES (?, ?, ?, ?)",
                        [regime, strategy_id, allocation_pct, confidence],
                    )

        logger.info(
            f"[quantpod_mcp] Updated regime matrix for '{regime}': {len(allocations)} strategies"
        )
        _fn = get_regime_strategies.fn if hasattr(get_regime_strategies, "fn") else get_regime_strategies
        return await _fn(regime)
    except Exception as e:
        logger.error(f"[quantpod_mcp] set_regime_allocation failed: {e}")
        return {"success": False, "error": str(e)}


@domain(Domain.RESEARCH)
@tool_def()
async def resolve_portfolio_conflicts(
    proposed_trades: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Resolve signal conflicts across multiple strategies for the same symbols.

    Rules:
      - Same symbol, different directions: high confidence wins, or SKIP if both high
      - Same symbol, same direction: merge with conservative sizing

    Args:
        proposed_trades: List of trade dicts, each with:
            symbol, action, confidence, strategy_id, capital_pct.

    Returns:
        Dict with resolved_trades, resolutions, conflicts_count.
    """
    try:
        result = resolve_conflicts(proposed_trades)
        return {"success": True, **result}
    except Exception as e:
        logger.error(f"[quantpod_mcp] resolve_portfolio_conflicts failed: {e}")
        return {"success": False, "error": str(e)}


# =============================================================================
# STRATEGY GAP ANALYSIS
# =============================================================================

# The five regime labels used in the regime-strategy matrix. A regime is
# considered "covered" when at least one strategy with status live or
# forward_testing has a regime_affinity entry for it.
_KNOWN_REGIMES = frozenset(
    {
        "trending_up",
        "trending_down",
        "ranging",
        "high_volatility",
        "unknown",
    }
)

# A strategy's trailing 30-day Sharpe below this value is flagged as degraded.
_DEGRADED_SHARPE_THRESHOLD = 0.3

# A regime with only one live strategy gets a "concentration" gap.
_MIN_STRATEGIES_PER_REGIME = 2


@domain(Domain.RESEARCH)
@tool_def()
async def get_strategy_gaps() -> dict[str, Any]:
    """
    Analyze the strategy registry for coverage gaps.

    Identifies regimes where:
    - No live or forward_testing strategy exists (critical)
    - The best strategy has trailing Sharpe < 0.3 (degraded)
    - Only one strategy covers the regime (concentration risk)

    Used by the Strategy Factory loop to target research at the regimes
    that need it most.

    Returns:
        {
            "success": True,
            "gaps": [{"regime": str, "gap_type": str, "severity": str, "details": str}],
            "coverage_summary": {"trending_up": 3, "ranging": 0, ...},
            "total_live": int,
            "total_forward_testing": int,
            "total_draft": int,
        }
    """
    _, err = live_db_or_error()
    if err:
        return err

    try:
        with pg_conn() as conn:
            # Count strategies by status
            status_counts = conn.execute(
                "SELECT status, COUNT(*) FROM strategies GROUP BY status"
            ).fetchall()
            counts_by_status = {row[0]: row[1] for row in status_counts}

            # Load all active strategies (live + forward_testing) with their regime_affinity
            active_rows = conn.execute(
                "SELECT strategy_id, name, status, regime_affinity "
                "FROM strategies WHERE status IN ('live', 'forward_testing')"
            ).fetchall()

            # Build regime → strategy mapping
            regime_strategies: dict[str, list[dict[str, Any]]] = {
                r: [] for r in _KNOWN_REGIMES
            }
            for row in active_rows:
                strategy_id, name, status, affinity_raw = row
                affinity = _parse_json_field(affinity_raw)
                if not isinstance(affinity, dict):
                    continue
                for regime_key in affinity:
                    normalized = regime_key.strip().lower().replace(" ", "_")
                    if normalized in regime_strategies:
                        regime_strategies[normalized].append(
                            {
                                "strategy_id": strategy_id,
                                "name": name,
                                "status": status,
                                "affinity_score": affinity[regime_key],
                            }
                        )

            # Compute trailing Sharpe per strategy from strategy_outcomes
            trailing_sharpe = _compute_trailing_sharpe(conn, days=30)

        # Identify gaps
        gaps: list[dict[str, Any]] = []
        coverage_summary: dict[str, int] = {}

        for regime, strategies in regime_strategies.items():
            coverage_summary[regime] = len(strategies)

            if len(strategies) == 0:
                gaps.append(
                    {
                        "regime": regime,
                        "gap_type": "no_strategy",
                        "severity": "critical",
                        "details": f"No live or forward_testing strategy covers '{regime}'.",
                    }
                )
                continue

            # Check for degradation — best strategy Sharpe below threshold.
            # Only strategies with enough data (Sharpe not None) are considered.
            known_sharpes = [
                s
                for s in (trailing_sharpe.get(s["strategy_id"]) for s in strategies)
                if s is not None
            ]
            if known_sharpes:
                best_sharpe = max(known_sharpes)
                if best_sharpe < _DEGRADED_SHARPE_THRESHOLD:
                    gaps.append(
                        {
                            "regime": regime,
                            "gap_type": "degraded",
                            "severity": "moderate",
                            "details": (
                                f"Best strategy for '{regime}' has trailing 30-day "
                                f"Sharpe={best_sharpe:.2f} (below {_DEGRADED_SHARPE_THRESHOLD})."
                            ),
                        }
                    )

            # Check for concentration risk
            if len(strategies) < _MIN_STRATEGIES_PER_REGIME:
                gaps.append(
                    {
                        "regime": regime,
                        "gap_type": "concentration",
                        "severity": "low",
                        "details": (
                            f"Only {len(strategies)} strategy covers '{regime}'. "
                            f"Recommend at least {_MIN_STRATEGIES_PER_REGIME}."
                        ),
                    }
                )

        return {
            "success": True,
            "gaps": gaps,
            "coverage_summary": coverage_summary,
            "total_live": counts_by_status.get("live", 0),
            "total_forward_testing": counts_by_status.get("forward_testing", 0),
            "total_draft": counts_by_status.get("draft", 0),
            "trailing_sharpe": {
                k: round(v, 3) for k, v in trailing_sharpe.items() if v is not None
            },
        }
    except Exception as e:
        logger.error(f"[quantpod_mcp] get_strategy_gaps failed: {e}")
        return {"success": False, "error": str(e)}


def _compute_trailing_sharpe(
    conn: Any,
    days: int = 30,
) -> dict[str, float | None]:
    """Compute trailing Sharpe ratio per strategy from strategy_outcomes.

    Returns {strategy_id: sharpe} for strategies with >= 3 closed outcomes in
    the lookback window. Returns None for strategies with insufficient data so
    callers can skip degradation checks rather than treating them as Sharpe=0.
    """
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    try:
        rows = conn.execute(
            "SELECT strategy_id, realized_pnl_pct "
            "FROM strategy_outcomes "
            "WHERE closed_at IS NOT NULL AND closed_at >= ?",
            [cutoff],
        ).fetchall()
    except Exception as exc:
        logger.debug(f"[meta] _compute_trailing_sharpe failed: {exc}")
        return {}

    # Group returns by strategy
    returns_by_strategy: dict[str, list[float]] = {}
    for strategy_id, pnl_pct in rows:
        if pnl_pct is not None:
            returns_by_strategy.setdefault(strategy_id, []).append(float(pnl_pct))

    sharpe_map: dict[str, float | None] = {}
    for strategy_id, returns in returns_by_strategy.items():
        if len(returns) < 3:
            sharpe_map[strategy_id] = None  # insufficient data — skip degradation check
            continue
        mean_ret = sum(returns) / len(returns)
        variance = sum((r - mean_ret) ** 2 for r in returns) / (len(returns) - 1)
        std_ret = variance**0.5
        sharpe_map[strategy_id] = (mean_ret / std_ret) if std_ret > 1e-9 else 0.0

    return sharpe_map


def _parse_json_field(value: Any) -> Any:
    """Parse a JSON field that may be a string, dict, or None."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return {}
    return {}


# =============================================================================
# AUTOMATED DRAFT PROMOTION
# =============================================================================


@domain(Domain.RESEARCH)
@tool_def()
async def promote_draft_strategies(
    min_oos_sharpe: float = 0.6,
    max_overfit_ratio: float = 2.0,
    max_age_days: int = 14,
) -> dict[str, Any]:
    """
    Evaluate all draft strategies for auto-promotion to forward_testing.

    Criteria (ALL must pass):
    1. OOS Sharpe mean >= min_oos_sharpe (from walk-forward summary)
    2. Overfit ratio < max_overfit_ratio
    3. Walk-forward degradation < 30%
    4. Created within max_age_days (stale drafts are retired instead)

    NEVER promotes to live. Only draft -> forward_testing.
    Live promotion requires a human /review session.

    Also retires draft strategies older than max_age_days that were never
    promoted — prevents DB bloat from accumulated discovery runs.

    Args:
        min_oos_sharpe: Minimum OOS Sharpe for promotion (default 0.6).
        max_overfit_ratio: Maximum IS/OOS Sharpe ratio (default 2.0).
        max_age_days: Drafts older than this are retired (default 14).

    Returns:
        {
            "success": True,
            "promoted": [{"strategy_id", "name", "oos_sharpe"}],
            "rejected": [{"strategy_id", "name", "reason"}],
            "retired": [{"strategy_id", "name", "reason"}],
        }
    """
    _, err = live_db_or_error()
    if err:
        return err

    try:
        with pg_conn() as conn:
            drafts = conn.execute(
                "SELECT strategy_id, name, walkforward_summary, backtest_summary, created_at "
                "FROM strategies WHERE status = 'draft'"
            ).fetchall()

        if not drafts:
            return {
                "success": True,
                "promoted": [],
                "rejected": [],
                "retired": [],
                "message": "No draft strategies to evaluate.",
            }

        promoted: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        retired: list[dict[str, Any]] = []
        # Collect status updates to apply in a single pooled connection
        status_updates: list[tuple[str, str]] = []  # (strategy_id, new_status)

        age_cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)

        for row in drafts:
            strategy_id, name, wf_raw, bt_raw, created_at = row

            # --- Age check: retire stale drafts ---
            if created_at is not None:
                created_dt = (
                    created_at
                    if isinstance(created_at, datetime)
                    else datetime.fromisoformat(str(created_at))
                )
                if created_dt.tzinfo is None:
                    created_dt = created_dt.replace(tzinfo=timezone.utc)
                if created_dt < age_cutoff:
                    status_updates.append((strategy_id, "retired"))
                    retired.append(
                        {
                            "strategy_id": strategy_id,
                            "name": name,
                            "reason": f"Stale draft — created {(datetime.now(timezone.utc) - created_dt).days} days ago, never promoted.",
                        }
                    )
                    continue

            # --- Walk-forward validation check ---
            wf_summary = _parse_json_field(wf_raw)
            if not wf_summary:
                rejected.append(
                    {
                        "strategy_id": strategy_id,
                        "name": name,
                        "reason": "No walk-forward summary — run run_walkforward() first.",
                    }
                )
                continue

            oos_sharpe = wf_summary.get("oos_sharpe_mean", 0.0)
            is_sharpe = wf_summary.get("is_sharpe_mean", 0.0)
            # Negative or near-zero OOS Sharpe → maximally overfit (reject)
            overfit_ratio = (
                (is_sharpe / oos_sharpe) if oos_sharpe > 1e-9 else float("inf")
            )
            if oos_sharpe <= 0:
                overfit_ratio = float("inf")

            if oos_sharpe < min_oos_sharpe:
                rejected.append(
                    {
                        "strategy_id": strategy_id,
                        "name": name,
                        "reason": f"OOS Sharpe {oos_sharpe:.2f} < {min_oos_sharpe}.",
                    }
                )
                continue

            if overfit_ratio > max_overfit_ratio:
                rejected.append(
                    {
                        "strategy_id": strategy_id,
                        "name": name,
                        "reason": f"Overfit ratio {overfit_ratio:.2f} > {max_overfit_ratio}.",
                    }
                )
                continue

            # Walk-forward degradation: (IS - OOS) / IS
            if is_sharpe > 0:
                degradation = (is_sharpe - oos_sharpe) / is_sharpe
                if degradation > 0.30:
                    rejected.append(
                        {
                            "strategy_id": strategy_id,
                            "name": name,
                            "reason": f"Walk-forward degradation {degradation:.0%} > 30%.",
                        }
                    )
                    continue

            # --- Promote ---
            status_updates.append((strategy_id, "forward_testing"))
            promoted.append(
                {
                    "strategy_id": strategy_id,
                    "name": name,
                    "oos_sharpe": round(oos_sharpe, 3),
                    "overfit_ratio": round(overfit_ratio, 2),
                }
            )
            logger.info(
                f"[quantpod_mcp] Auto-promoted '{name}' ({strategy_id}) "
                f"to forward_testing (OOS Sharpe={oos_sharpe:.2f})"
            )

        # Apply all status updates in a single pooled connection
        if status_updates:
            with pg_conn() as conn:
                for sid, new_status in status_updates:
                    _update_strategy_status(conn, sid, new_status)

        return {
            "success": True,
            "promoted": promoted,
            "rejected": rejected,
            "retired": retired,
        }
    except Exception as e:
        logger.error(f"[quantpod_mcp] promote_draft_strategies failed: {e}")
        return {"success": False, "error": str(e)}


def _update_strategy_status(conn: Any, strategy_id: str, new_status: str) -> None:
    """Update a strategy's status and timestamp in the DB."""
    conn.execute(
        "UPDATE strategies SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE strategy_id = ?",
        [new_status, strategy_id],
    )


# =============================================================================
# LIVE RULE EVALUATION
# =============================================================================


@domain(Domain.RESEARCH)
@tool_def()
async def check_strategy_rules(
    symbol: str,
    strategy_id: str,
) -> dict[str, Any]:
    """
    Evaluate a strategy's entry/exit rules against CURRENT market data.

    Uses the same FeatureEnricher and rule evaluator as backtesting — no
    train/serve skew. Loads the latest 252 daily bars, computes technical
    indicators + enriched features, then evaluates each rule on the latest bar.

    This is the critical bridge between backtested strategies and live
    execution: a strategy validated with "PE < 20 AND RSI < 30" now has
    those exact rules checked at trade time.

    Args:
        symbol: Ticker symbol to evaluate.
        strategy_id: Strategy to check (loads entry/exit rules from DB).

    Returns:
        {
            "success": True,
            "entry_triggered": True/False,
            "exit_triggered": True/False,
            "entry_rules_detail": [
                {"indicator": "fund_pe_ratio", "condition": "below", "value": 20,
                 "current_value": 18.5, "passed": True},
            ],
            "exit_rules_detail": [...],
            "features_loaded": ["fundamentals"],
        }
    """
    _, err = live_db_or_error()
    if err:
        return err

    try:
        with pg_conn() as conn:
            row = conn.execute(
                "SELECT entry_rules, exit_rules, parameters FROM strategies WHERE strategy_id = ?",
                [strategy_id],
            ).fetchone()

        if row is None:
            return {"success": False, "error": f"Strategy '{strategy_id}' not found"}

        entry_rules_raw, exit_rules_raw, params_raw = row
        entry_rules = _parse_json_field(entry_rules_raw)
        exit_rules = _parse_json_field(exit_rules_raw)
        parameters = _parse_json_field(params_raw)

        if not isinstance(entry_rules, list):
            entry_rules = []
        if not isinstance(exit_rules, list):
            exit_rules = []
        if not isinstance(parameters, dict):
            parameters = {}

        # Ensure symbol is in parameters for enricher
        parameters["symbol"] = symbol

        # Load latest 252 daily bars of OHLCV (approximately one trading year)
        end_date = date.today().isoformat()
        start_date = (date.today() - timedelta(days=365)).isoformat()
        price_data = _fetch_price_data(symbol, start_date=start_date, end_date=end_date)
        if price_data is None or price_data.empty:
            return {
                "success": False,
                "error": f"No price data available for {symbol}",
            }

        # Generate signals using the same pipeline as backtesting
        # This includes technical indicators + feature enrichment
        signals_df = _generate_signals_from_rules(
            price_data,
            entry_rules,
            exit_rules,
            parameters,
        )

        if signals_df.empty:
            return {
                "success": False,
                "error": "Signal generation produced empty result",
            }

        # Evaluate rules on the LATEST bar
        last_row = signals_df.iloc[-1]

        entry_detail = _evaluate_rules_detail(entry_rules, signals_df, last_row)
        exit_detail = _evaluate_rules_detail(exit_rules, signals_df, last_row)

        # Entry is triggered if the signal column is 1 on the last bar
        entry_triggered = bool(last_row.get("signal", 0) == 1)

        # For exit: check if any exit rule condition is met
        exit_triggered = any(
            r["passed"]
            for r in exit_detail
            if r.get("type") not in ("time_stop", "take_profit", "stop_loss")
        )

        # Detect which feature tiers were loaded
        enricher = FeatureEnricher()
        tiers = enricher.detect_needed_tiers(entry_rules + exit_rules)
        features_loaded = [
            name
            for name, active in [
                ("fundamentals", tiers.fundamentals),
                ("earnings", tiers.earnings),
                ("macro", tiers.macro),
                ("flow", tiers.flow),
            ]
            if active
        ]

        return {
            "success": True,
            "symbol": symbol,
            "strategy_id": strategy_id,
            "entry_triggered": entry_triggered,
            "exit_triggered": exit_triggered,
            "entry_rules_detail": entry_detail,
            "exit_rules_detail": exit_detail,
            "features_loaded": features_loaded,
            "bars_evaluated": len(signals_df),
            "latest_bar_date": str(signals_df.index[-1]),
        }

    except Exception as e:
        logger.error(f"[quantpod_mcp] check_strategy_rules failed: {e}")
        return {"success": False, "error": str(e)}


def _evaluate_rules_detail(
    rules: list[dict[str, Any]],
    df: Any,
    last_row: Any,
) -> list[dict[str, Any]]:
    """Evaluate each rule against the last bar and return per-rule detail."""
    details = []
    for rule in rules:
        indicator = rule.get("indicator", "")
        condition = rule.get("condition", "")
        value = rule.get("value")
        rule_type = rule.get("type", "plain")

        # Get current value of the indicator
        current_value = None
        if indicator in last_row.index:
            raw = last_row[indicator]
            try:
                current_value = float(raw) if not isinstance(raw, str) else raw
            except (ValueError, TypeError):
                current_value = raw

        # Simple pass/fail evaluation for the last bar
        passed = False
        if current_value is not None and value is not None:
            try:
                cv = float(current_value) if not isinstance(current_value, str) else 0.0
                v = float(value)
                if condition in ("below", "less_than"):
                    passed = cv < v
                elif condition in ("above", "greater_than"):
                    passed = cv > v
                elif condition == "equals":
                    passed = str(current_value).lower() == str(value).lower()
                elif condition == "between":
                    lower = float(rule.get("lower", 0))
                    upper = float(rule.get("upper", 100))
                    passed = lower <= cv <= upper
            except (ValueError, TypeError):
                passed = False

        details.append(
            {
                "indicator": indicator,
                "condition": condition,
                "value": value,
                "current_value": current_value,
                "passed": passed,
                "type": rule_type,
            }
        )

    return details


# ── Tool collection ──────────────────────────────────────────────────────────
from quantstack.mcp.tools._tool_def import collect_tools  # noqa: E402

TOOLS = collect_tools()
