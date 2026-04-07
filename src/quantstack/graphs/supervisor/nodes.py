"""Node functions for the Supervisor Graph.

Each node is an async function: (SupervisorState) -> dict
The return dict contains only the state fields the node updates.

Supervisor nodes use tools for real system introspection (heartbeats,
system status, strategy registry) rather than hallucinating health data.
"""

import json
import logging
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool

from quantstack.autonomous.ic_retirement import run_ic_retirement_sweep
from quantstack.core import ic_calculator as _ic_calculator_mod
from quantstack.core import signal_scorer as _signal_scorer_mod
from quantstack.core.execution.execution_quality import compute_quality_scalar
from quantstack.core.portfolio.mmc_scorer import (
    MIN_STRATEGIES_FOR_MMC,
    compute_mmc,
    compute_portfolio_signal,
    get_capital_weight_scalar,
)
from quantstack.core.attribution_engine import (
    SECTOR_ETF_MAP,
    _DEFAULT_SECTOR_ETF,
    decompose as attribution_decompose,
)
from quantstack.core.regime_detector import RegimeInputs, classify_regime
from quantstack.coordination.event_bus import Event, EventBus, EventType
from quantstack.db import db_conn
from quantstack.graphs.agent_executor import parse_json_response, run_agent
from quantstack.graphs.config import AgentConfig
from quantstack.graphs.state import SupervisorState

logger = logging.getLogger(__name__)


def make_health_check(llm: BaseChatModel, config: AgentConfig, tools: list[BaseTool] | None = None):
    """Create the health_check node with system introspection tools."""
    tools = tools or []

    async def health_check(state: SupervisorState) -> dict[str, Any]:
        try:
            prompt = (
                f"Cycle {state['cycle_number']}: Run a system health check.\n\n"
                "Use your tools to:\n"
                "1. Check overall system status (kill switch, services, data freshness)\n"
                "2. Check heartbeat for 'trading-graph' (max 120s stale)\n"
                "3. Check heartbeat for 'research-graph' (max 600s stale)\n\n"
                "Classify each service as healthy/degraded/critical.\n"
                'Return JSON: {"overall": "healthy|degraded|critical", "services": {...}}'
            )
            text = await run_agent(llm, tools, config, prompt)
            health_status = parse_json_response(text, {"overall": "unknown", "raw": text})

            # Collect operational health metrics and update Prometheus gauges
            try:
                from quantstack.graphs.supervisor.health_metrics import collect_health_metrics
                from quantstack.observability.metrics import (
                    record_cycle_error_count,
                    record_cycle_success_rate,
                    record_research_queue_depth,
                    record_strategy_generation,
                )

                metrics = await collect_health_metrics()
                record_cycle_success_rate("trading", metrics["trading_cycle_success_rate"])
                record_cycle_success_rate("research", metrics["research_cycle_success_rate"])
                record_cycle_error_count("trading", metrics["trading_cycle_error_count"])
                record_cycle_error_count("research", metrics["research_cycle_error_count"])
                record_strategy_generation(metrics["strategy_generation_7d"])
                record_research_queue_depth(metrics["research_queue_depth"])
            except Exception as exc:
                logger.warning("Health metrics collection failed: %s", exc)

            # Kill switch recovery and escalation checks
            try:
                from quantstack.execution.kill_switch import get_kill_switch
                from quantstack.execution.kill_switch_recovery import (
                    AutoRecoveryManager,
                    KillSwitchEscalationManager,
                )

                ks = get_kill_switch()
                AutoRecoveryManager(ks).check()
                KillSwitchEscalationManager(ks).check()
            except Exception as exc:
                logger.warning("Kill switch recovery/escalation check failed: %s", exc)

            # Factor exposure monitoring (section-04)
            factor_summary: dict[str, Any] = {}
            try:
                from quantstack.risk.factor_exposure import run_factor_exposure_check
                from quantstack.db import db_conn as _db_conn

                with _db_conn() as _conn:
                    pos_rows = _conn.execute(
                        "SELECT symbol, quantity, avg_cost, "
                        "quantity * avg_cost AS market_value "
                        "FROM positions"
                    ).fetchall()
                positions = [dict(r) for r in pos_rows]
                factor_summary = await run_factor_exposure_check(positions)
            except Exception as exc:
                logger.warning("Factor exposure check failed — continuing: %s", exc)
                factor_summary = {"error": str(exc)}

            health_status["factor_exposure"] = factor_summary

            # EventBus ACK monitoring (section-07)
            try:
                from quantstack.coordination.event_bus import check_missed_acks
                from quantstack.db import db_conn as _db_conn2

                with _db_conn2() as _ack_conn:
                    missed_ack_alerts = await check_missed_acks(_ack_conn)
                if missed_ack_alerts:
                    logger.warning("[Supervisor] %d missed ACK alerts raised", len(missed_ack_alerts))
                health_status["missed_ack_alerts"] = len(missed_ack_alerts)
            except Exception as exc:
                logger.warning("ACK monitoring failed — continuing: %s", exc)

            # LLM provider health monitoring (section-09)
            try:
                from quantstack.llm.provider import check_provider_health

                provider_health = await check_provider_health()
                health_status["llm_providers"] = provider_health

                # Emit alerts for degraded providers
                for pname, pinfo in provider_health.items():
                    if pinfo.get("status") == "error":
                        try:
                            from quantstack.tools.functions.system_alerts import emit_system_alert
                            emit_system_alert(
                                title=f"LLM provider {pname} unavailable",
                                category="service_failure",
                                severity="warning",
                                details=f"Error: {pinfo.get('error', 'unknown')}",
                            )
                        except Exception:
                            pass
            except Exception as exc:
                logger.warning("LLM provider health check failed — continuing: %s", exc)

            return {"health_status": health_status}
        except Exception as exc:
            logger.error("health_check failed: %s", exc)
            return {
                "health_status": {"error": str(exc), "overall": "unknown"},
                "errors": [f"health_check: {exc}"],
            }

    return health_check


def make_diagnose_issues(llm: BaseChatModel, config: AgentConfig, tools: list[BaseTool] | None = None):
    """Create the diagnose_issues node with diagnostic tools."""
    tools = tools or []

    async def diagnose_issues(state: SupervisorState) -> dict[str, Any]:
        health = state.get("health_status", {})
        try:
            prompt = (
                f"System health status:\n{json.dumps(health, indent=2, default=str)}\n\n"
                "Use your tools to:\n"
                "1. Check system status for detailed diagnostics\n"
                "2. Search knowledge base for similar past issues and resolutions\n\n"
                "Diagnose any degraded/critical services. For each, identify root cause "
                "and recommend a recovery action from the playbook.\n"
                "If all healthy, return an empty array.\n"
                'Return JSON: [{"service": ..., "diagnosis": ..., "recommended_action": ...}]'
            )
            text = await run_agent(llm, tools, config, prompt)
            issues = parse_json_response(text, [])
            if not isinstance(issues, list):
                issues = [issues] if issues else []
            return {"diagnosed_issues": issues}
        except Exception as exc:
            logger.error("diagnose_issues failed: %s", exc)
            return {
                "diagnosed_issues": [],
                "errors": [f"diagnose_issues: {exc}"],
            }

    return diagnose_issues


def make_execute_recovery(llm: BaseChatModel, config: AgentConfig, tools: list[BaseTool] | None = None):
    """Create the execute_recovery node."""
    tools = tools or []

    async def execute_recovery(state: SupervisorState) -> dict[str, Any]:
        issues = state.get("diagnosed_issues", [])
        if not issues:
            return {"recovery_actions": []}

        try:
            prompt = (
                f"Diagnosed issues:\n{json.dumps(issues, indent=2, default=str)}\n\n"
                "Execute recovery actions from the playbook:\n"
                "- Stale heartbeat: record the issue, watchdog handles restart\n"
                "- Ollama down: flag for restart, graphs operate degraded\n"
                "- LLM provider failure: trigger fallback chain\n"
                "- Database lost: exponential backoff reconnect\n"
                "- Data staleness: trigger data refresh\n"
                "- Multiple failures: consider kill switch\n\n"
                'Return JSON: [{"action": ..., "target": ..., "result": ...}]'
            )
            text = await run_agent(llm, tools, config, prompt)
            actions = parse_json_response(text, [])
            if not isinstance(actions, list):
                actions = [actions] if actions else []
            return {"recovery_actions": actions}
        except Exception as exc:
            logger.error("execute_recovery failed: %s", exc)
            return {
                "recovery_actions": [],
                "errors": [f"execute_recovery: {exc}"],
            }

    return execute_recovery


def make_strategy_pipeline():
    """Create the strategy_pipeline node — deterministic draft→backtested pass.

    Runs ``StrategyLifecycle.run_pipeline_pass()`` which backtests every draft
    strategy that has a symbol.  This is the mechanism that moves strategies
    from ``draft`` to ``backtested`` status.  It runs every supervisor cycle
    so drafts are picked up promptly.
    """

    async def strategy_pipeline(state: SupervisorState) -> dict[str, Any]:
        from quantstack.autonomous.strategy_lifecycle import StrategyLifecycle
        from quantstack.db import db_conn

        try:
            with db_conn() as conn:
                lifecycle = StrategyLifecycle(conn)
                report = await lifecycle.run_pipeline_pass()
            if report.skipped:
                logger.info("[strategy_pipeline] Skipped (concurrent run)")
            elif report.backtested:
                logger.info(
                    "[strategy_pipeline] Backtested %d strategies: %s",
                    len(report.backtested),
                    report.backtested,
                )
            if report.errors:
                logger.warning(
                    "[strategy_pipeline] %d errors: %s",
                    len(report.errors),
                    report.errors[:3],
                )
            return {
                "strategy_pipeline_report": {
                    "backtested": len(report.backtested),
                    "errors": len(report.errors),
                    "skipped": report.skipped,
                },
            }
        except Exception as exc:
            logger.error("strategy_pipeline failed: %s", exc)
            return {
                "strategy_pipeline_report": {"error": str(exc)},
                "errors": [f"strategy_pipeline: {exc}"],
            }

    return strategy_pipeline


def make_strategy_lifecycle(llm: BaseChatModel, config: AgentConfig, tools: list[BaseTool] | None = None):
    """Create the strategy_lifecycle node with registry access.

    Runs two phases:
    1. Deterministic: poll IC_DECAY events from EventBus and demote affected live strategies.
    2. LLM-based: review forward_testing strategies for promotion/retirement.
    """
    tools = tools or []

    async def strategy_lifecycle(state: SupervisorState) -> dict[str, Any]:
        demotion_actions: list[dict] = []

        # --- Phase 1: Deterministic IC_DECAY demotion ---
        try:
            with db_conn() as conn:
                bus = EventBus(conn)
                decay_events = bus.poll(
                    "strategy_lifecycle_ic_decay",
                    event_types=[EventType.IC_DECAY],
                )
                for event in decay_events:
                    strategy_id = event.payload.get("strategy_id")
                    if not strategy_id:
                        continue
                    rows_affected = conn.execute(
                        "UPDATE strategies SET status = 'forward_testing', updated_at = NOW() "
                        "WHERE strategy_id = %s AND status = 'live'",
                        (strategy_id,),
                    ).rowcount
                    if rows_affected:
                        logger.warning(
                            "[strategy_lifecycle] Demoted %s to forward_testing (IC_DECAY: "
                            "icir_21d=%.3f, icir_63d=%.3f)",
                            strategy_id,
                            event.payload.get("icir_21d", float("nan")),
                            event.payload.get("icir_63d", float("nan")),
                        )
                        demotion_actions.append({
                            "strategy_id": strategy_id,
                            "decision": "demote",
                            "reasoning": (
                                f"IC_DECAY: icir_21d={event.payload.get('icir_21d'):.3f}, "
                                f"icir_63d={event.payload.get('icir_63d'):.3f} — both below 0.3"
                            ),
                        })
        except Exception as exc:
            logger.warning("strategy_lifecycle: IC_DECAY poll/demotion failed: %s", exc)

        # --- Phase 2: LLM-based promotion/retirement review ---
        try:
            prompt = (
                f"Cycle {state['cycle_number']}: Review strategy lifecycle.\n\n"
                "Use your tools to:\n"
                "1. Fetch the strategy registry to see all strategies and their status\n"
                "2. Search knowledge base for past promotion/retirement lessons\n\n"
                "For each forward_testing strategy, evaluate performance evidence "
                "(P&L, win rate, drawdown, trade count, duration) and decide:\n"
                "- promote: sufficient evidence for live trading\n"
                "- extend: needs more testing time\n"
                "- retire: IS/OOS ratio diverged > 4x, win rate dropped > 20pts\n"
                "- no_change: still within testing window\n\n"
                'Return JSON: [{"strategy_id": ..., "decision": ..., "reasoning": ...}]'
            )
            text = await run_agent(llm, tools, config, prompt)
            llm_actions = parse_json_response(text, [])
            if not isinstance(llm_actions, list):
                llm_actions = [llm_actions] if llm_actions else []
            return {"strategy_lifecycle_actions": demotion_actions + llm_actions}
        except Exception as exc:
            logger.error("strategy_lifecycle failed: %s", exc)
            return {
                "strategy_lifecycle_actions": demotion_actions,
                "errors": [f"strategy_lifecycle: {exc}"],
            }

    return strategy_lifecycle


def _fetch_returns_for_attribution(symbol: str, as_of: date, lookback_days: int = 90) -> pd.Series:
    """
    Fetch daily log returns for `symbol` over the `lookback_days` calendar days ending on `as_of`.

    Returns a pd.Series of log returns (length ≤ 60 trading days) indexed by date,
    or an empty Series if no data is available.
    """
    window_start = as_of - timedelta(days=lookback_days)
    try:
        with db_conn() as conn:
            rows = conn.execute(
                """
                SELECT timestamp::date AS dt, close
                FROM ohlcv
                WHERE symbol = %s AND timeframe = 'daily'
                  AND timestamp >= %s AND timestamp <= %s
                ORDER BY timestamp
                """,
                (symbol, window_start, as_of),
            ).fetchall()
    except Exception as exc:
        logger.warning("_fetch_returns_for_attribution: OHLCV fetch failed for %s: %s", symbol, exc)
        return pd.Series(dtype=float)

    if not rows:
        return pd.Series(dtype=float)

    dates = [r[0] for r in rows]
    closes = np.array([float(r[1]) for r in rows])
    prices = pd.Series(closes, index=pd.DatetimeIndex(dates))
    return np.log(prices / prices.shift(1)).iloc[1:]


async def run_attribution() -> dict[str, Any]:
    """
    Nightly P&L attribution run.

    Watermark: reads MAX(date) FROM pnl_attribution. Only processes positions
    closed since that watermark date. Safe to re-run — ON CONFLICT DO NOTHING.

    Returns:
        dict with "positions_processed" and "rows_written" counts.
    """
    try:
        # 1. Read watermark
        with db_conn() as conn:
            row = conn.execute("SELECT MAX(date) FROM pnl_attribution").fetchone()
            watermark = row[0] if row and row[0] is not None else None

        # 2. Fetch closed positions since watermark
        with db_conn() as conn:
            if watermark is None:
                trades = conn.execute(
                    "SELECT symbol, strategy_id, opened_at, closed_at, realized_pnl "
                    "FROM closed_trades ORDER BY closed_at"
                ).fetchall()
            else:
                trades = conn.execute(
                    "SELECT symbol, strategy_id, opened_at, closed_at, realized_pnl "
                    "FROM closed_trades WHERE closed_at > %s ORDER BY closed_at",
                    (watermark,),
                ).fetchall()

        if not trades:
            logger.debug("run_attribution: no positions since watermark %s", watermark)
            return {"positions_processed": 0, "rows_written": 0}

        # 3. Read daily risk-free rate from system_state
        with db_conn() as conn:
            rf_row = conn.execute(
                "SELECT value FROM system_state WHERE key = 'risk_free_rate_daily'"
            ).fetchone()
        if rf_row is None:
            logger.warning("run_attribution: risk_free_rate_daily missing; defaulting to 0.0")
            rf = 0.0
        else:
            rf = float(rf_row[0])

        positions_processed = 0
        rows_written = 0

        for trade in trades:
            symbol, strategy_id, opened_at_raw, closed_at_raw, realized_pnl = trade

            # Normalize to date objects
            opened_dt = opened_at_raw.date() if hasattr(opened_at_raw, "date") else opened_at_raw
            closed_dt = closed_at_raw.date() if hasattr(closed_at_raw, "date") else closed_at_raw

            try:
                # Resolve sector ETF for symbol
                try:
                    with db_conn() as conn:
                        sector_row = conn.execute(
                            "SELECT sector FROM fundamentals WHERE symbol = %s LIMIT 1",
                            (symbol,),
                        ).fetchone()
                    sector = sector_row[0] if sector_row else None
                    sector_etf = SECTOR_ETF_MAP.get(sector, _DEFAULT_SECTOR_ETF) if sector else _DEFAULT_SECTOR_ETF
                except Exception:
                    sector_etf = _DEFAULT_SECTOR_ETF

                records = []
                # Iterate over every business day in [opened_dt, closed_dt]
                current = opened_dt
                while current <= closed_dt:
                    stock_returns = _fetch_returns_for_attribution(symbol, current)
                    spy_returns = _fetch_returns_for_attribution("SPY", current)
                    sector_returns = _fetch_returns_for_attribution(sector_etf, current)

                    position_notional = abs(realized_pnl) if realized_pnl else 1.0

                    record = attribution_decompose(
                        symbol=symbol,
                        strategy_id=strategy_id,
                        attr_date=current,
                        stock_returns=stock_returns,
                        spy_returns=spy_returns,
                        sector_returns=sector_returns,
                        risk_free_rate=rf,
                        position_notional=position_notional,
                        opened_at=opened_dt,
                        sector_etf=sector_etf,
                    )
                    records.append(record)

                    # Advance to next business day
                    current += timedelta(days=1)
                    while current.weekday() >= 5:
                        current += timedelta(days=1)

                # 5. Batch insert all records for this position
                if records:
                    with db_conn() as conn:
                        conn.executemany(
                            """
                            INSERT INTO pnl_attribution
                              (date, symbol, strategy_id, total_pnl, market_pnl, sector_pnl,
                               alpha_pnl, residual_pnl, beta_market, beta_sector, sector_etf, holding_day)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (date, symbol, strategy_id) DO NOTHING
                            """,
                            [
                                (r.date, r.symbol, r.strategy_id, r.total_pnl, r.market_pnl,
                                 r.sector_pnl, r.alpha_pnl, r.residual_pnl,
                                 r.beta_market, r.beta_sector, r.sector_etf, r.holding_day)
                                for r in records
                            ],
                        )
                    rows_written += len(records)

                positions_processed += 1

            except Exception as exc:
                logger.error("run_attribution: failed for %s/%s: %s", symbol, strategy_id, exc)
                continue

        return {"positions_processed": positions_processed, "rows_written": rows_written}

    except Exception as exc:
        logger.error("run_attribution: top-level failure: %s", exc)
        return {"positions_processed": 0, "rows_written": 0, "error": str(exc)}


def _fetch_market_data_for_scoring(symbol: str) -> dict:
    """
    Fetch the most recent OHLCV row for `symbol` and common indicators.

    Returns a flat dict suitable for passing to score_signal() as market_data.
    Returns empty dict if no data is available.
    """
    try:
        with db_conn() as conn:
            row = conn.execute(
                """
                SELECT close, open, high, low, volume, timestamp
                FROM ohlcv
                WHERE symbol = %s AND timeframe = 'daily'
                ORDER BY timestamp DESC LIMIT 1
                """,
                (symbol,),
            ).fetchone()
        if not row:
            return {}
        close, open_, high, low, volume, ts = row
        return {
            "close": float(close),
            "open": float(open_),
            "high": float(high),
            "low": float(low),
            "volume": float(volume) if volume is not None else 0.0,
            "_symbol": symbol,
        }
    except Exception as exc:
        logger.warning("_fetch_market_data_for_scoring: failed for %s: %s", symbol, exc)
        return {}


def _fetch_fwd_returns_for_ic(strategy_id: str, horizon_days: int) -> pd.DataFrame:
    """
    Fetch forward returns for all (signal_date, symbol) pairs belonging to this strategy.

    Returns a DataFrame with `signal_date` as index and `symbol` as columns,
    values = forward log return over `horizon_days` business days.
    Returns empty DataFrame if insufficient data.
    """
    try:
        with db_conn() as conn:
            rows = conn.execute(
                """
                SELECT s.signal_date, s.symbol,
                       LN(o_fwd.close / o_cur.close) AS fwd_return
                FROM signals s
                JOIN ohlcv o_cur ON o_cur.symbol = s.symbol
                    AND o_cur.timestamp::date = s.signal_date
                    AND o_cur.timeframe = 'daily'
                JOIN ohlcv o_fwd ON o_fwd.symbol = s.symbol
                    AND o_fwd.timeframe = 'daily'
                    AND o_fwd.timestamp::date = (
                        SELECT MIN(o2.timestamp::date) FROM ohlcv o2
                        WHERE o2.symbol = s.symbol AND o2.timeframe = 'daily'
                          AND o2.timestamp::date > s.signal_date
                        OFFSET %s - 1
                    )
                WHERE s.strategy_id = %s
                  AND s.signal_date >= CURRENT_DATE - INTERVAL '63 days'
                ORDER BY s.signal_date
                """,
                (horizon_days, strategy_id),
            ).fetchall()
    except Exception as exc:
        logger.warning("_fetch_fwd_returns_for_ic: failed for %s h=%d: %s", strategy_id, horizon_days, exc)
        return pd.DataFrame()

    if not rows:
        return pd.DataFrame()

    records = [(r[0], r[1], float(r[2])) for r in rows if r[2] is not None]
    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records, columns=["signal_date", "symbol", "fwd_return"])
    return df.pivot(index="signal_date", columns="symbol", values="fwd_return")


async def run_signal_scoring() -> dict[str, Any]:
    """
    Score all live and forward_testing strategies against current market data.

    Writes one row per (today, strategy_id, symbol) to the signals table.
    Uses ON CONFLICT UPDATE so re-runs on the same day overwrite stale values.

    Returns a summary dict: {strategies_scored, signals_written, errors}.
    """
    today = date.today()
    strategies_scored = 0
    signals_written = 0
    errors: list[str] = []

    try:
        with db_conn() as conn:
            strat_rows = conn.execute(
                "SELECT strategy_id, status, entry_rules FROM strategies "
                "WHERE status IN ('live', 'forward_testing')"
            ).fetchall()

            # Determine current regime
            regime_row = conn.execute(
                "SELECT regime FROM regime_state ORDER BY detected_at DESC LIMIT 1"
            ).fetchone()
            regime = regime_row[0] if regime_row else "unknown"

        for strat_row in strat_rows:
            strategy_id, status, entry_rules = strat_row
            if entry_rules is None:
                entry_rules = []
            elif isinstance(entry_rules, str):
                import json as _json
                try:
                    entry_rules = _json.loads(entry_rules)
                except Exception:
                    entry_rules = []

            # Symbol scope: union of closed_trades and positions
            try:
                with db_conn() as conn:
                    symbol_rows = conn.execute(
                        """
                        SELECT DISTINCT symbol FROM closed_trades WHERE strategy_id = %s
                        UNION
                        SELECT DISTINCT symbol FROM positions WHERE strategy_id = %s
                        """,
                        (strategy_id, strategy_id),
                    ).fetchall()
            except Exception as exc:
                logger.warning("run_signal_scoring: symbol fetch failed for %s: %s", strategy_id, exc)
                errors.append(f"{strategy_id}: symbol fetch failed: {exc}")
                continue

            symbols = [r[0] for r in symbol_rows]
            if not symbols:
                strategies_scored += 1
                continue

            batch_rows = []
            for symbol in symbols:
                try:
                    market_data = _fetch_market_data_for_scoring(symbol)
                    signal_value, confidence = _signal_scorer_mod.score_signal(entry_rules, market_data)
                    batch_rows.append((today, strategy_id, symbol, signal_value, confidence, regime))
                except Exception as exc:
                    logger.warning("run_signal_scoring: scoring failed for %s/%s: %s", strategy_id, symbol, exc)
                    errors.append(f"{strategy_id}/{symbol}: {exc}")

            if batch_rows:
                with db_conn() as conn:
                    conn.executemany(
                        """
                        INSERT INTO signals (signal_date, strategy_id, symbol, signal_value, confidence, regime, created_at)
                        VALUES (%s, %s, %s, %s, %s, %s, NOW())
                        ON CONFLICT (signal_date, strategy_id, symbol)
                        DO UPDATE SET signal_value = EXCLUDED.signal_value,
                                      confidence   = EXCLUDED.confidence,
                                      regime       = EXCLUDED.regime
                        """,
                        batch_rows,
                    )
                signals_written += len(batch_rows)

            strategies_scored += 1

    except Exception as exc:
        logger.error("run_signal_scoring: top-level failure: %s", exc)
        errors.append(f"top-level: {exc}")

    return {"strategies_scored": strategies_scored, "signals_written": signals_written, "errors": errors}


async def run_ic_computation() -> dict[str, Any]:
    """
    Compute cross-sectional Rank IC and rolling ICIR for each strategy
    that has sufficient signal history, then detect IC decay.

    Writes/upserts to signal_ic. Publishes IC_DECAY events for live
    strategies whose ICIR falls below 0.3 in both the 21D and 63D windows.

    Returns a summary dict: {strategies_computed, ic_decay_events, errors}.
    """
    strategies_computed = 0
    ic_decay_events = 0
    errors: list[str] = []

    try:
        with db_conn() as conn:
            eligible = conn.execute(
                """
                SELECT strategy_id
                FROM signals
                WHERE signal_date >= CURRENT_DATE - INTERVAL '63 days'
                GROUP BY strategy_id
                HAVING COUNT(DISTINCT symbol) >= 5
                   AND COUNT(DISTINCT signal_date) >= 21
                """
            ).fetchall()
    except Exception as exc:
        logger.error("run_ic_computation: eligibility query failed: %s", exc)
        return {"strategies_computed": 0, "ic_decay_events": 0, "errors": [str(exc)]}

    for (strategy_id,) in eligible:
        last_icir_21d: float | None = None
        last_icir_63d: float | None = None

        for horizon in (5, 10, 21):
            try:
                # Fetch signal rows for this strategy
                with db_conn() as conn:
                    sig_rows = conn.execute(
                        """
                        SELECT signal_date, symbol, signal_value FROM signals
                        WHERE strategy_id = %s AND signal_date >= CURRENT_DATE - INTERVAL '63 days'
                        ORDER BY signal_date
                        """,
                        (strategy_id,),
                    ).fetchall()

                if not sig_rows:
                    continue

                # Build signals DataFrame
                sig_df = pd.DataFrame(sig_rows, columns=["signal_date", "symbol", "signal_value"])
                sig_df["signal_date"] = pd.to_datetime(sig_df["signal_date"])
                signals_pivot = sig_df.pivot(index="signal_date", columns="symbol", values="signal_value")

                # Fetch forward returns
                fwd_df = _fetch_fwd_returns_for_ic(strategy_id, horizon)
                if fwd_df.empty:
                    continue

                # Normalize fwd index to Timestamps for intersection
                if not isinstance(fwd_df.index, pd.DatetimeIndex):
                    fwd_df.index = pd.to_datetime(fwd_df.index)

                # Align indices
                common_idx = signals_pivot.index.intersection(fwd_df.index)
                if len(common_idx) < 5:
                    continue

                s_aligned = signals_pivot.loc[common_idx]
                f_aligned = fwd_df.loc[common_idx]

                ic_series = _ic_calculator_mod.compute_cross_sectional_ic(s_aligned, f_aligned)
                icir_21d_series = _ic_calculator_mod.compute_rolling_icir(ic_series, window=21)
                icir_63d_series = _ic_calculator_mod.compute_rolling_icir(ic_series, window=63)

                # Most recent values
                rank_ic = float(ic_series.dropna().iloc[-1]) if not ic_series.dropna().empty else float("nan")
                icir_21d = float(icir_21d_series.dropna().iloc[-1]) if not icir_21d_series.dropna().empty else float("nan")
                icir_63d = float(icir_63d_series.dropna().iloc[-1]) if not icir_63d_series.dropna().empty else float("nan")
                n_symbols = int(sig_df["symbol"].nunique())

                valid_ic = ic_series.dropna()
                if len(valid_ic) > 1:
                    ic_std = float(valid_ic.std())
                    ic_tstat = float(valid_ic.mean() / (ic_std / (len(valid_ic) ** 0.5))) if ic_std > 0 else 0.0
                else:
                    ic_tstat = 0.0

                today = date.today()
                with db_conn() as conn:
                    conn.execute(
                        """
                        INSERT INTO signal_ic (date, strategy_id, horizon_days, rank_ic, icir_21d, icir_63d, ic_tstat, n_symbols, updated_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())
                        ON CONFLICT (date, strategy_id, horizon_days)
                        DO UPDATE SET rank_ic = EXCLUDED.rank_ic,
                                      icir_21d = EXCLUDED.icir_21d,
                                      icir_63d = EXCLUDED.icir_63d,
                                      ic_tstat = EXCLUDED.ic_tstat,
                                      n_symbols = EXCLUDED.n_symbols,
                                      updated_at = NOW()
                        """,
                        (today, strategy_id, horizon, rank_ic, icir_21d, icir_63d, ic_tstat, n_symbols),
                    )

                # Track 21-day window values for decay check
                if horizon == 21:
                    last_icir_21d = icir_21d
                    last_icir_63d = icir_63d

            except Exception as exc:
                logger.warning("run_ic_computation: failed for %s h=%d: %s", strategy_id, horizon, exc)
                errors.append(f"{strategy_id}/h={horizon}: {exc}")

        # IC decay check — only for horizon=21 and only for live strategies
        if last_icir_21d is not None and last_icir_63d is not None:
            import math
            if not (math.isnan(last_icir_21d) or math.isnan(last_icir_63d)):
                if last_icir_21d < 0.3 and last_icir_63d < 0.3:
                    # Check strategy status
                    try:
                        with db_conn() as conn:
                            status_row = conn.execute(
                                "SELECT status FROM strategies WHERE strategy_id = %s",
                                (strategy_id,),
                            ).fetchone()
                        strategy_status = status_row[0] if status_row else None

                        if strategy_status == "live":
                            with db_conn() as conn:
                                bus = EventBus(conn)
                                bus.publish(Event(
                                    event_type=EventType.IC_DECAY,
                                    source_loop="supervisor",
                                    payload={
                                        "strategy_id": strategy_id,
                                        "icir_21d": last_icir_21d,
                                        "icir_63d": last_icir_63d,
                                    },
                                ))
                            ic_decay_events += 1
                            logger.warning(
                                "IC_DECAY published for %s (icir_21d=%.3f, icir_63d=%.3f)",
                                strategy_id, last_icir_21d, last_icir_63d,
                            )
                    except Exception as exc:
                        logger.warning("run_ic_computation: IC_DECAY publish failed for %s: %s", strategy_id, exc)
                        errors.append(f"{strategy_id}/ic_decay: {exc}")

        strategies_computed += 1

    return {"strategies_computed": strategies_computed, "ic_decay_events": ic_decay_events, "errors": errors}


def run_execution_quality_scoring(conn: Any) -> dict[str, Any]:
    """Compute rolling 30-day execution quality scalars per symbol.

    Reads ``forecast_error_bps`` from ``tca_results`` for each symbol traded
    in the last 30 days, maps mean absolute error to a quality scalar via
    :func:`compute_quality_scalar`, and upserts to ``symbol_execution_quality``.

    Symbols with fewer than 3 trades are skipped (insufficient data).

    Args:
        conn: Active PostgreSQL connection (caller owns the db_conn context).

    Returns:
        Summary dict: {symbols_scored, symbols_skipped, errors}.
    """
    symbols_scored = 0
    symbols_skipped = 0
    errors: list[str] = []

    try:
        rows = conn.execute(
            """
            SELECT DISTINCT symbol FROM tca_results
            WHERE timestamp >= NOW() - INTERVAL '30 days'
              AND forecast_error_bps IS NOT NULL
            """
        ).fetchall()
    except Exception as exc:
        logger.error("run_execution_quality_scoring: symbol query failed: %s", exc)
        return {"symbols_scored": 0, "symbols_skipped": 0, "errors": [str(exc)]}

    for (symbol,) in rows:
        try:
            error_rows = conn.execute(
                """
                SELECT forecast_error_bps FROM tca_results
                WHERE symbol = %s
                  AND timestamp >= NOW() - INTERVAL '30 days'
                  AND forecast_error_bps IS NOT NULL
                """,
                (symbol,),
            ).fetchall()

            if len(error_rows) < 3:
                symbols_skipped += 1
                continue

            abs_errors = [abs(float(r[0])) for r in error_rows]
            mean_abs_error = sum(abs_errors) / len(abs_errors)
            quality_scalar = compute_quality_scalar(mean_abs_error)

            conn.execute(
                """
                INSERT INTO symbol_execution_quality
                    (symbol, week_ending, mean_abs_error_bps, quality_scalar, n_trades)
                VALUES (%s, CURRENT_DATE, %s, %s, %s)
                ON CONFLICT (symbol, week_ending) DO UPDATE SET
                    mean_abs_error_bps = EXCLUDED.mean_abs_error_bps,
                    quality_scalar = EXCLUDED.quality_scalar,
                    n_trades = EXCLUDED.n_trades
                """,
                (symbol, mean_abs_error, quality_scalar, len(error_rows)),
            )
            symbols_scored += 1

        except Exception as exc:
            logger.warning(
                "run_execution_quality_scoring: failed for %s: %s", symbol, exc,
            )
            errors.append(f"{symbol}: {exc}")

    return {
        "symbols_scored": symbols_scored,
        "symbols_skipped": symbols_skipped,
        "errors": errors,
    }


def run_mmc_computation(conn: Any) -> dict[str, Any]:
    """Compute Meta-Model Contribution scores for all active strategies.

    For each strategy with status ``live`` or ``forward_testing`` that has
    signal data in the last 21 days:

    * If total active strategies >= :data:`MIN_STRATEGIES_FOR_MMC` (20), uses
      full Gaussianised MMC via :func:`compute_mmc`.
    * Otherwise, falls back to Pearson correlation between the strategy signal
      and the aggregate portfolio signal.

    Writes one row per strategy to ``strategy_mmc``.

    Args:
        conn: Active PostgreSQL connection (caller owns the db_conn context).

    Returns:
        Summary dict: {strategies_scored, method, errors}.
    """
    strategies_scored = 0
    errors: list[str] = []

    # 1. Fetch eligible strategies (live or forward_testing with recent signals)
    try:
        strat_rows = conn.execute(
            """
            SELECT DISTINCT s.strategy_id, st.status
            FROM signals s
            JOIN strategies st ON st.strategy_id = s.strategy_id
            WHERE st.status IN ('live', 'forward_testing')
              AND s.signal_date >= CURRENT_DATE - INTERVAL '21 days'
            """
        ).fetchall()
    except Exception as exc:
        logger.error("run_mmc_computation: strategy query failed: %s", exc)
        return {"strategies_scored": 0, "method": "error", "errors": [str(exc)]}

    if not strat_rows:
        return {"strategies_scored": 0, "method": "none", "errors": []}

    strategy_ids = [r[0] for r in strat_rows]
    strategy_status_map = {r[0]: r[1] for r in strat_rows}

    # 2. Fetch signal data for all strategies
    strategy_signals: dict[str, pd.DataFrame] = {}
    for sid in strategy_ids:
        try:
            sig_rows = conn.execute(
                """
                SELECT signal_date, symbol, signal_value FROM signals
                WHERE strategy_id = %s
                  AND signal_date >= CURRENT_DATE - INTERVAL '21 days'
                ORDER BY signal_date
                """,
                (sid,),
            ).fetchall()
            if sig_rows:
                df = pd.DataFrame(sig_rows, columns=["signal_date", "symbol", "signal_value"])
                strategy_signals[sid] = df
        except Exception as exc:
            logger.warning("run_mmc_computation: signal fetch failed for %s: %s", sid, exc)
            errors.append(f"{sid}/signal_fetch: {exc}")

    if not strategy_signals:
        return {"strategies_scored": 0, "method": "no_signals", "errors": errors}

    # 3. Compute aggregate portfolio signal
    try:
        portfolio_signal_df = compute_portfolio_signal(strategy_signals)
    except Exception as exc:
        logger.error("run_mmc_computation: portfolio signal failed: %s", exc)
        return {"strategies_scored": 0, "method": "error", "errors": errors + [str(exc)]}

    n_active = len(strategy_signals)
    method = "mmc" if n_active >= MIN_STRATEGIES_FOR_MMC else "pearson"

    today = date.today()

    # 4. Score each strategy
    for sid, sig_df in strategy_signals.items():
        try:
            # Merge strategy signal with portfolio signal on (signal_date, symbol)
            merged = sig_df.merge(
                portfolio_signal_df,
                on=["signal_date", "symbol"],
                suffixes=("_strat", "_port"),
            )

            if len(merged) < 5:
                continue

            strat_vals = merged["signal_value_strat"].values.astype(float)
            port_vals = merged["signal_value_port"].values.astype(float)

            if method == "mmc":
                # Fetch realized returns for the same (signal_date, symbol) pairs
                realized = _fetch_realized_returns_for_mmc(conn, merged)
                if realized is None or len(realized) != len(strat_vals):
                    # Fall back to Pearson if returns unavailable
                    corr_matrix = np.corrcoef(strat_vals, port_vals)
                    correlation = float(corr_matrix[0, 1]) if not np.isnan(corr_matrix[0, 1]) else 0.0
                    mmc_value = None
                else:
                    mmc_value = compute_mmc(strat_vals, port_vals, realized)
                    # For capital weight, use correlation (not MMC directly)
                    corr_matrix = np.corrcoef(strat_vals, port_vals)
                    correlation = float(corr_matrix[0, 1]) if not np.isnan(corr_matrix[0, 1]) else 0.0
            else:
                # Pearson correlation path
                corr_matrix = np.corrcoef(strat_vals, port_vals)
                correlation = float(corr_matrix[0, 1]) if not np.isnan(corr_matrix[0, 1]) else 0.0
                mmc_value = None

            capital_weight_scalar = get_capital_weight_scalar(correlation)

            conn.execute(
                """
                INSERT INTO strategy_mmc
                    (date, strategy_id, mmc_value, correlation, capital_weight_scalar,
                     method, n_observations, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())
                ON CONFLICT (date, strategy_id) DO UPDATE SET
                    mmc_value = EXCLUDED.mmc_value,
                    correlation = EXCLUDED.correlation,
                    capital_weight_scalar = EXCLUDED.capital_weight_scalar,
                    method = EXCLUDED.method,
                    n_observations = EXCLUDED.n_observations,
                    updated_at = NOW()
                """,
                (today, sid, mmc_value, correlation, capital_weight_scalar, method, len(merged)),
            )
            strategies_scored += 1

        except Exception as exc:
            logger.warning("run_mmc_computation: failed for %s: %s", sid, exc)
            errors.append(f"{sid}: {exc}")

    return {"strategies_scored": strategies_scored, "method": method, "errors": errors}


def _fetch_realized_returns_for_mmc(
    conn: Any, merged: pd.DataFrame,
) -> np.ndarray | None:
    """Fetch 1-day forward returns for (signal_date, symbol) pairs in *merged*.

    Returns an ndarray of the same length as *merged* or ``None`` if data
    is insufficient.
    """
    returns = []
    for _, row in merged.iterrows():
        try:
            ret_row = conn.execute(
                """
                SELECT LN(o_fwd.close / o_cur.close)
                FROM ohlcv o_cur
                JOIN ohlcv o_fwd ON o_fwd.symbol = o_cur.symbol
                    AND o_fwd.timeframe = 'daily'
                    AND o_fwd.timestamp::date = (
                        SELECT MIN(o2.timestamp::date) FROM ohlcv o2
                        WHERE o2.symbol = o_cur.symbol AND o2.timeframe = 'daily'
                          AND o2.timestamp::date > o_cur.timestamp::date
                    )
                WHERE o_cur.symbol = %s
                  AND o_cur.timeframe = 'daily'
                  AND o_cur.timestamp::date = %s
                """,
                (row["symbol"], row["signal_date"]),
            ).fetchone()
            if ret_row and ret_row[0] is not None:
                returns.append(float(ret_row[0]))
            else:
                return None
        except Exception:
            return None

    return np.array(returns) if len(returns) == len(merged) else None


def _is_attribution_day() -> bool:
    """Attribution runs nightly on trading days (Monday–Friday)."""
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).weekday() < 5


def run_regime_detection(conn: Any) -> dict[str, Any]:
    """
    Fetch current market indicators, classify regime, write regime_state row,
    and publish REGIME_CHANGE event if regime has changed.

    Returns {"regime": str, "regime_change": bool, "confidence": float}.
    Falls back to regime="unknown" if any input fetch fails — never raises.

    Args:
        conn: Active PostgreSQL connection (caller owns the db_conn context).
    """
    # --- Fetch SPY OHLCV for 20d return and ADX ---
    adx = 22.0
    spy_20d_return = 0.0
    try:
        spy_rows = conn.execute(
            """
            SELECT close FROM ohlcv
            WHERE symbol = 'SPY' AND timeframe = 'daily'
            ORDER BY timestamp DESC LIMIT 50
            """,
        ).fetchall()
        if spy_rows and len(spy_rows) >= 21:
            closes = [float(r[0]) for r in reversed(spy_rows)]
            spy_20d_return = closes[-1] / closes[-21] - 1
            # Compute 14-period ADX from OHLCV (simplified: use log-return std as proxy)
            # For a proper ADX we need High/Low/Close; use available closes as proxy
            if len(closes) >= 28:
                import math
                rets = [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))]
                recent = rets[-14:]
                adx = float(min(100.0, abs(sum(recent)) / (sum(abs(r) for r in recent) + 1e-9) * 40 + 15))
    except Exception as exc:
        logger.warning("run_regime_detection: SPY fetch failed: %s", exc)

    # --- Fetch VIX ---
    vix_level = 20.0
    try:
        vix_row = conn.execute(
            "SELECT value FROM macro_indicators WHERE indicator = 'VIX' ORDER BY date DESC LIMIT 1"
        ).fetchone()
        if vix_row:
            vix_level = float(vix_row[0])
    except Exception as exc:
        logger.warning("run_regime_detection: VIX fetch failed: %s", exc)

    # --- Fetch breadth score ---
    breadth_score = 0.5
    try:
        breadth_rows = conn.execute(
            """
            SELECT symbol, close FROM ohlcv
            WHERE timeframe = 'daily'
              AND timestamp >= CURRENT_DATE - INTERVAL '55 days'
            ORDER BY symbol, timestamp
            """,
        ).fetchall()
        if breadth_rows:
            from collections import defaultdict
            sym_closes: dict = defaultdict(list)
            for sym, close in breadth_rows:
                sym_closes[sym].append(float(close))
            above = sum(1 for c in sym_closes.values() if len(c) >= 51 and c[-1] > sum(c[-51:-1]) / 50)
            total = sum(1 for c in sym_closes.values() if len(c) >= 51)
            if total >= 10:
                breadth_score = above / total
    except Exception as exc:
        logger.warning("run_regime_detection: breadth fetch failed: %s", exc)

    # --- Fetch previous regime ---
    previous_regime: str | None = None
    try:
        prev_row = conn.execute(
            "SELECT regime FROM regime_state ORDER BY detected_at DESC LIMIT 1"
        ).fetchone()
        if prev_row:
            previous_regime = prev_row[0]
    except Exception as exc:
        logger.warning("run_regime_detection: previous regime fetch failed: %s", exc)

    # --- Classify ---
    inputs = RegimeInputs(
        adx=adx,
        spy_20d_return=spy_20d_return,
        vix_level=vix_level,
        breadth_score=breadth_score,
        previous_regime=previous_regime,
    )
    classification = classify_regime(inputs)

    # --- Write to regime_state ---
    try:
        conn.execute(
            """
            INSERT INTO regime_state
              (detected_at, regime, adx, vix_level, spy_20d_return, breadth_score,
               confidence, previous_regime, regime_change)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (detected_at) DO UPDATE
              SET regime = EXCLUDED.regime,
                  adx = EXCLUDED.adx,
                  vix_level = EXCLUDED.vix_level,
                  spy_20d_return = EXCLUDED.spy_20d_return,
                  breadth_score = EXCLUDED.breadth_score,
                  confidence = EXCLUDED.confidence,
                  previous_regime = EXCLUDED.previous_regime,
                  regime_change = EXCLUDED.regime_change
            """,
            (
                classification.detected_at,
                classification.regime,
                adx,
                vix_level,
                spy_20d_return,
                breadth_score,
                classification.confidence,
                previous_regime,
                classification.regime_change,
            ),
        )
    except Exception as exc:
        logger.error("run_regime_detection: write to regime_state failed: %s", exc)

    # --- Publish REGIME_CHANGE event ---
    if classification.regime_change:
        try:
            bus = EventBus(conn)
            bus.publish(Event(
                event_type=EventType.REGIME_CHANGE,
                source_loop="supervisor",
                payload={
                    "regime": classification.regime,
                    "previous_regime": previous_regime,
                    "confidence": classification.confidence,
                    "adx": adx,
                    "spy_20d_return": spy_20d_return,
                },
            ))
        except Exception as exc:
            logger.warning("run_regime_detection: REGIME_CHANGE publish failed: %s", exc)

    return {
        "regime": classification.regime,
        "regime_change": classification.regime_change,
        "confidence": classification.confidence,
    }


def _is_trading_day_today() -> bool:
    """True if today is a weekday (Mon–Fri). Does not account for market holidays."""
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).weekday() < 5


def _is_nightly_functions_due() -> bool:
    """True if today is a trading day and nightly functions haven't run yet today."""
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    if now.weekday() >= 5:  # Saturday/Sunday
        return False

    today = now.strftime("%Y-%m-%d")
    try:
        with db_conn() as conn:
            row = conn.execute(
                "SELECT 1 FROM loop_heartbeats "
                "WHERE loop_name = 'nightly_functions' "
                "AND DATE(started_at) = %s AND status = 'completed' LIMIT 1",
                (today,),
            ).fetchone()
        return row is None
    except Exception:
        return True  # If we can't check, assume due


def _is_community_intel_due() -> bool:
    """Check if weekly community intel scan is due (Saturday, last run > 6 days ago)."""
    from datetime import datetime, timedelta, timezone

    now = datetime.now(timezone.utc)
    if now.weekday() != 5:  # 5 = Saturday
        return False

    try:
        from quantstack.db import db_conn

        six_days_ago = now - timedelta(days=6)
        with db_conn() as conn:
            row = conn.execute(
                "SELECT 1 FROM loop_heartbeats "
                "WHERE loop_name = 'community_intel' "
                "AND started_at > ? AND status = 'completed' LIMIT 1",
                [six_days_ago],
            ).fetchone()
        return row is None
    except Exception:
        return True  # If we can't check, assume it's due


def _is_execution_researcher_due() -> bool:
    """Check if monthly execution audit is due (1st business day, last run > 25 days ago)."""
    from datetime import datetime, timedelta, timezone

    now = datetime.now(timezone.utc)
    # 1st business day: day <= 3 and weekday < 5
    if now.day > 3 or now.weekday() >= 5:
        return False

    try:
        from quantstack.db import db_conn

        twenty_five_days_ago = now - timedelta(days=25)
        with db_conn() as conn:
            row = conn.execute(
                "SELECT 1 FROM loop_heartbeats "
                "WHERE loop_name = 'execution_researcher' "
                "AND started_at > ? AND status = 'completed' LIMIT 1",
                [twenty_five_days_ago],
            ).fetchone()
        return row is None
    except Exception:
        return True


def _is_weekly_task_due(task_name: str) -> bool:
    """Check if a weekly Saturday task is due (Saturday, last run > 6 days ago).

    Follows the same pattern as :func:`_is_community_intel_due` but is
    parameterised by *task_name* so multiple weekly tasks can reuse it.
    """
    from datetime import datetime, timedelta, timezone

    now = datetime.now(timezone.utc)
    if now.weekday() != 5:  # 5 = Saturday
        return False

    try:
        six_days_ago = now - timedelta(days=6)
        with db_conn() as conn:
            row = conn.execute(
                "SELECT 1 FROM loop_heartbeats "
                "WHERE loop_name = %s "
                "AND started_at > %s AND status = 'completed' LIMIT 1",
                (task_name, six_days_ago),
            ).fetchone()
        return row is None
    except Exception:
        return True  # If we can't check, assume due


def make_scheduled_tasks(
    llm: BaseChatModel,
    config: AgentConfig,
    tools: list[BaseTool] | None = None,
    *,
    community_llm: BaseChatModel | None = None,
    community_cfg: AgentConfig | None = None,
    community_tools: list[BaseTool] | None = None,
):
    """Create the scheduled_tasks node with concrete community_intel and execution_researcher.

    When community_llm/cfg/tools are provided, community_intel runs with its own
    dedicated agent config (including the 3-iteration backstory). Otherwise falls
    back to the generic llm/config.
    """
    tools = tools or []
    # Use dedicated community intel agent if provided, else fall back
    _ci_llm = community_llm or llm
    _ci_cfg = community_cfg or config
    _ci_tools = community_tools if community_tools is not None else tools

    async def scheduled_tasks(state: SupervisorState) -> dict[str, Any]:
        results: list[dict[str, Any]] = []

        # --- Community Intel (weekly, Saturday) ---
        community_due = _is_community_intel_due()
        community_result: dict[str, Any] = {
            "task": "community_intel", "was_due": community_due, "fired": False,
        }
        if community_due:
            try:
                community_prompt = (
                    "Run a weekly community intelligence scan using your 3-iteration protocol.\n\n"
                    "ITERATION 1: Broad scan across arXiv q-fin, GitHub trending, Reddit "
                    "r/algotrading + r/quant, QuantConnect forums, quant blogs, SSRN, Twitter.\n"
                    "ITERATION 2: Gap analysis — use get_strategy_gaps and fetch_strategy_registry "
                    "to find underrepresented areas, then targeted searches to fill gaps.\n"
                    "ITERATION 3: Refinement — cross-check knowledge base for previously failed ideas, "
                    "deduplicate, rank by actionability.\n\n"
                    "For each discovery: assess novelty, empirical validation, implementation feasibility.\n"
                    "Filter: skip duplicates, items > 90 days old, ideas without backtest evidence.\n\n"
                    'Return JSON: {"ideas": [{"title": "...", "source": "...", "url": "...", '
                    '"category": "...", "asset_class": "...", "summary": "...", '
                    '"empirical_evidence": "...", "implementation_path": "...", '
                    '"novelty_vs_registry": "...", "iteration_found": 1, '
                    '"relevance_score": 0.0-1.0}]}'
                )
                text = await run_agent(_ci_llm, _ci_tools, _ci_cfg, community_prompt)
                parsed = parse_json_response(text, {"ideas": []})
                ideas = parsed.get("ideas", [])

                # Publish IDEAS_DISCOVERED event
                if ideas:
                    try:
                        from quantstack.coordination.event_bus import Event, EventBus, EventType
                        from quantstack.db import db_conn

                        with db_conn() as conn:
                            bus = EventBus(conn)
                            bus.publish(Event(
                                event_type=EventType.IDEAS_DISCOVERED,
                                source_loop="supervisor",
                                payload={"ideas": ideas, "count": len(ideas)},
                            ))
                    except Exception as pub_exc:
                        logger.warning("Failed to publish IDEAS_DISCOVERED: %s", pub_exc)

                # Record heartbeat
                try:
                    from quantstack.tools.functions.system_functions import record_heartbeat

                    await record_heartbeat(
                        service="community_intel",
                        iteration=state["cycle_number"],
                        symbols_processed=len(ideas),
                        errors=0,
                        status="completed",
                    )
                except Exception as exc:
                    logger.warning("Failed to record heartbeat for community_intel: %s", exc)

                community_result["fired"] = True
                community_result["ideas_found"] = len(ideas)
            except Exception as exc:
                logger.error("community_intel task failed: %s", exc)
                community_result["error"] = str(exc)
        results.append(community_result)

        # --- Execution Quality Scoring (weekly, Saturday) ---
        eq_due = _is_weekly_task_due("execution_quality_scoring")
        eq_result: dict[str, Any] = {
            "task": "execution_quality_scoring", "was_due": eq_due, "fired": False,
        }
        if eq_due:
            try:
                with db_conn() as conn:
                    eq_summary = run_execution_quality_scoring(conn)
                eq_result["fired"] = True
                eq_result.update(eq_summary)
                logger.info(
                    "[scheduled_tasks] Execution quality scoring: %d scored, %d skipped",
                    eq_summary.get("symbols_scored", 0),
                    eq_summary.get("symbols_skipped", 0),
                )
                try:
                    from quantstack.tools.functions.system_functions import record_heartbeat

                    await record_heartbeat(
                        service="execution_quality_scoring",
                        iteration=state["cycle_number"],
                        symbols_processed=eq_summary.get("symbols_scored", 0),
                        errors=len(eq_summary.get("errors", [])),
                        status="completed",
                    )
                except Exception as exc:
                    logger.warning("Failed to record heartbeat for execution_quality_scoring: %s", exc)
            except Exception as exc:
                logger.error("execution_quality_scoring task failed: %s", exc)
                eq_result["error"] = str(exc)
        results.append(eq_result)

        # --- MMC Computation (weekly, Saturday) ---
        mmc_due = _is_weekly_task_due("mmc_computation")
        mmc_result: dict[str, Any] = {
            "task": "mmc_computation", "was_due": mmc_due, "fired": False,
        }
        if mmc_due:
            try:
                with db_conn() as conn:
                    mmc_summary = run_mmc_computation(conn)
                mmc_result["fired"] = True
                mmc_result.update(mmc_summary)
                logger.info(
                    "[scheduled_tasks] MMC computation: %d strategies scored (method=%s)",
                    mmc_summary.get("strategies_scored", 0),
                    mmc_summary.get("method", "unknown"),
                )
                try:
                    from quantstack.tools.functions.system_functions import record_heartbeat

                    await record_heartbeat(
                        service="mmc_computation",
                        iteration=state["cycle_number"],
                        symbols_processed=mmc_summary.get("strategies_scored", 0),
                        errors=len(mmc_summary.get("errors", [])),
                        status="completed",
                    )
                except Exception as exc:
                    logger.warning("Failed to record heartbeat for mmc_computation: %s", exc)
            except Exception as exc:
                logger.error("mmc_computation task failed: %s", exc)
                mmc_result["error"] = str(exc)
        results.append(mmc_result)

        # --- Memory Pruning (weekly, Saturday) ---
        prune_due = _is_weekly_task_due("memory_pruning")
        prune_result: dict[str, Any] = {
            "task": "memory_pruning", "was_due": prune_due, "fired": False,
        }
        if prune_due:
            try:
                from quantstack.memory.blackboard import Blackboard

                with db_conn() as conn:
                    bb = Blackboard(conn=conn)
                    archived = bb.archive_stale()
                prune_result["fired"] = True
                prune_result["archived_counts"] = archived
                total = sum(archived.values())
                logger.info(
                    "[scheduled_tasks] Memory pruning: archived %d entries %s",
                    total, archived,
                )
            except Exception as exc:
                logger.error("memory_pruning task failed: %s", exc)
                prune_result["error"] = str(exc)
        results.append(prune_result)

        # --- Execution Researcher (monthly, 1st business day) ---
        exec_due = _is_execution_researcher_due()
        exec_result: dict[str, Any] = {
            "task": "execution_researcher", "was_due": exec_due, "fired": False,
        }
        if exec_due:
            try:
                exec_prompt = (
                    "Run a monthly execution quality audit.\n\n"
                    "Use your tools to:\n"
                    "1. Fetch the portfolio and all fills from the past month\n"
                    "2. Compute TCA metrics: arrival shortfall per stock, per algo, per time-of-day\n"
                    "3. Identify worst-execution trades and systematic biases\n"
                    "4. Search knowledge base for past execution quality reports\n\n"
                    "Produce an execution quality report with:\n"
                    "- Average shortfall (bps)\n"
                    "- Best/worst execution stocks\n"
                    "- Time-of-day effects\n"
                    "- Recommendations for execution improvement\n\n"
                    'Return JSON: {"report": "...", "avg_shortfall_bps": ..., '
                    '"worst_executions": [...], "recommendations": [...]}'
                )
                text = await run_agent(llm, tools, config, exec_prompt)
                parsed = parse_json_response(text, {"report": text})

                # Store in knowledge base
                try:
                    from quantstack.knowledge.store import KnowledgeStore

                    ks = KnowledgeStore()
                    ks.add_entry(
                        category="execution_quality",
                        content=json.dumps(parsed, default=str),
                        metadata={"cycle": state["cycle_number"]},
                    )
                except Exception as kb_exc:
                    logger.warning("Failed to store execution report: %s", kb_exc)

                # Record heartbeat
                try:
                    from quantstack.tools.functions.system_functions import record_heartbeat

                    await record_heartbeat(
                        service="execution_researcher",
                        iteration=state["cycle_number"],
                        symbols_processed=0,
                        errors=0,
                        status="completed",
                    )
                except Exception as exc:
                    logger.warning("Failed to record heartbeat for execution_researcher: %s", exc)

                exec_result["fired"] = True
            except Exception as exc:
                logger.error("execution_researcher task failed: %s", exc)
                exec_result["error"] = str(exc)
        results.append(exec_result)

        # --- Attribution pipeline (nightly, trading days Mon-Fri) ---
        is_trading_day = _is_attribution_day()
        attribution_result: dict[str, Any] = {
            "task": "attribution", "was_due": is_trading_day, "fired": False,
        }
        if is_trading_day:
            try:
                summary = await run_attribution()
                attribution_result["fired"] = True
                attribution_result.update(summary)
            except Exception as exc:
                logger.error("attribution task failed: %s", exc)
                attribution_result["error"] = str(exc)
        results.append(attribution_result)

        # --- Nightly signal scoring, IC computation (trading days, once per day) ---
        nightly_due = _is_nightly_functions_due()
        nightly_result: dict[str, Any] = {
            "task": "nightly_functions", "was_due": nightly_due, "fired": False,
        }
        if nightly_due:
            try:
                scoring_summary = await run_signal_scoring()
                ic_summary = await run_ic_computation()

                # IC retirement sweep — retire forward_testing strategies
                # with persistently weak IC. Runs after IC computation so
                # signal_ic rows are fresh.
                ic_retired: list[str] = []
                try:
                    with db_conn() as conn:
                        ic_retired = run_ic_retirement_sweep(conn)
                    if ic_retired:
                        logger.info(
                            "[nightly] IC retirement sweep retired %d strategies: %s",
                            len(ic_retired),
                            ic_retired,
                        )
                except Exception as ic_ret_exc:
                    logger.error("[nightly] IC retirement sweep failed: %s", ic_ret_exc)

                nightly_result.update({
                    "fired": True,
                    "signal_scoring": scoring_summary,
                    "ic_computation": ic_summary,
                    "ic_retirement": {"retired": ic_retired},
                })
                try:
                    from quantstack.tools.functions.system_functions import record_heartbeat
                    await record_heartbeat(
                        service="nightly_functions",
                        iteration=state["cycle_number"],
                        symbols_processed=scoring_summary.get("signals_written", 0),
                        errors=0,
                        status="completed",
                    )
                except Exception as exc:
                    logger.warning("Failed to record heartbeat for nightly_functions: %s", exc)
            except Exception as exc:
                logger.error("nightly_functions failed: %s", exc)
                nightly_result["error"] = str(exc)
        results.append(nightly_result)

        # --- Regime Detection (daily, trading days) ---
        regime_due = _is_trading_day_today()
        regime_result: dict[str, Any] = {
            "task": "regime_detection", "was_due": regime_due, "fired": False,
        }
        if regime_due:
            try:
                with db_conn() as conn:
                    rd_result = run_regime_detection(conn)
                regime_result.update(rd_result)
                regime_result["fired"] = True
            except Exception as exc:
                logger.error("regime_detection task failed: %s", exc)
                regime_result["error"] = str(exc)
        results.append(regime_result)

        # --- Other scheduled tasks (existing LLM-based check) ---
        try:
            prompt = (
                f"Cycle {state['cycle_number']}: Check remaining scheduled tasks.\n\n"
                "Check due tasks: 30-min data freshness check, daily preflight, daily digest.\n"
                "Fire coordination events for due tasks.\n"
                'Return JSON: [{"task": ..., "was_due": true/false, "fired": true/false}]'
            )
            text = await run_agent(llm, tools, config, prompt)
            other_results = parse_json_response(text, [])
            if isinstance(other_results, list):
                results.extend(other_results)
        except Exception as exc:
            logger.error("scheduled_tasks (other) failed: %s", exc)
            results.append({"task": "other", "error": str(exc)})

        return {"scheduled_task_results": results}

    return scheduled_tasks


def make_eod_data_sync():
    """Create the eod_data_sync node (deterministic, no LLM).

    Runs once per supervisor cycle after market close. Fetches daily candles,
    options chains, fundamentals, and earnings calendar from Alpha Vantage.
    Skipped during market hours (intraday refresh handles that).
    """

    async def eod_data_sync(state: SupervisorState) -> dict[str, Any]:
        from quantstack.runners import is_market_hours

        if is_market_hours():
            return {
                "eod_refresh_summary": {"skipped": True, "reason": "market_open"},
            }

        # Only run once per day — check if we already ran today
        try:
            from datetime import datetime, timezone

            from quantstack.db import db_conn

            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            with db_conn() as conn:
                row = conn.execute(
                    "SELECT 1 FROM loop_heartbeats "
                    "WHERE loop_name = 'eod_data_sync' "
                    "AND DATE(started_at) = ? AND status = 'completed' LIMIT 1",
                    [today],
                ).fetchone()
            if row:
                return {
                    "eod_refresh_summary": {"skipped": True, "reason": "already_ran_today"},
                }
        except Exception as exc:
            logger.warning("Failed to check eod_data_sync run history: %s", exc)

        try:
            from quantstack.data.scheduled_refresh import run_eod_refresh

            report = await run_eod_refresh()
            summary = {
                "mode": report.mode,
                "symbols_refreshed": report.symbols_refreshed,
                "api_calls": report.api_calls,
                "errors": report.errors,
                "elapsed_seconds": round(report.elapsed_seconds, 1),
            }

            # Record that EOD sync completed today
            try:
                from quantstack.tools.functions.system_functions import record_heartbeat

                await record_heartbeat(
                    service="eod_data_sync",
                    iteration=int(datetime.now(timezone.utc).strftime("%Y%m%d")),
                    symbols_processed=report.symbols_refreshed,
                    errors=len(report.errors),
                    status="completed",
                )
            except Exception as hb_exc:
                logger.warning("Failed to record eod_data_sync heartbeat: %s", hb_exc)

            if report.errors:
                return {
                    "eod_refresh_summary": summary,
                    "errors": [f"eod_data_sync: {len(report.errors)} errors"],
                }
            return {"eod_refresh_summary": summary}
        except Exception as exc:
            logger.error("eod_data_sync failed: %s", exc)
            return {
                "eod_refresh_summary": {"error": str(exc)},
                "errors": [f"eod_data_sync: {exc}"],
            }

    return eod_data_sync
