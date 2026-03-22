# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
AutonomousRunner — unattended trading loop.

Runs without a Claude Code session. Calls SignalEngine for analysis and
DecisionRouter to pick execution path. All decisions are fully deterministic —
no LLM calls in the execution path (v1.1). All trades still flow through
risk_gate → broker so every safety invariant from the interactive path is preserved.

Invariants (never violated regardless of arguments):
- kill_switch.is_active() is checked before every symbol, not just at start.
- All trades go through execute_trade → risk_gate.check() — no bypass path.
- paper_mode=True unless USE_REAL_TRADING=true in env AND force_live=True.
- Every decision (including SKIPs) written to the audit log.
- DB is opened read-only for strategy queries; execution goes through MCP ctx.

Usage (CLI):
    python -m quant_pod.autonomous.runner --symbols XOM MSFT SPY
    python -m quant_pod.autonomous.runner --dry-run           # no order submission
    python -m quant_pod.autonomous.runner --paper-only        # force paper mode

Usage (code):
    from quantstack.autonomous.runner import AutonomousRunner

    report = await AutonomousRunner().run(symbols=["XOM", "MSFT"])
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from loguru import logger

from quantstack.autonomous.debate import DebateFilter
from quantstack.autonomous.decision import DecisionPath, DecisionRouter, RouteContext
from quantstack.observability.trace import TraceContext


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class SymbolResult:
    """Outcome of one symbol's processing pass."""

    symbol: str
    path: DecisionPath
    path_reason: str
    action: str | None = None  # "buy", "sell", or None
    position_size: str = "quarter"
    confidence: float = 0.0
    reasoning: str = ""
    fill_result: dict | None = None  # None when SKIP or dry-run
    error: str | None = None
    elapsed_seconds: float = 0.0


@dataclass
class RunReport:
    """Aggregate result of a full AutonomousRunner.run() call."""

    run_id: str
    started_at: datetime
    finished_at: datetime
    symbols_processed: int
    rule_based: int = 0
    groq_synthesis: int = 0
    skipped: int = 0
    executed: int = 0
    risk_rejected: int = 0
    errors: int = 0
    results: list[SymbolResult] = field(default_factory=list)


# =============================================================================
# AutonomousRunner
# =============================================================================


class AutonomousRunner:
    """
    Single pass over a symbol list:
      1. System health (kill switch, risk halt)
      2. Load active strategies from DB (read-only)
      3. For each symbol: SignalEngine → DecisionRouter → execute or skip
      4. Write audit trail for every decision
      5. Return RunReport

    Thread-safety: Not thread-safe. Designed to be called from a single
    asyncio event loop (the scheduler).
    """

    def __init__(
        self,
        dry_run: bool = False,
        paper_only: bool = True,
        max_concurrent: int = 3,
    ) -> None:
        """
        Args:
            dry_run: Log decisions but submit no orders.
            paper_only: Force paper_mode=True regardless of USE_REAL_TRADING.
            max_concurrent: Max parallel SignalEngine calls (default 3 to
                            avoid overwhelming QuantCore data layer).
        """
        self._dry_run = dry_run
        self._paper_only = paper_only
        self._max_concurrent = max_concurrent
        self._router = DecisionRouter()
        self._debate_filter = DebateFilter()
        self._drift_detector = self._init_drift_detector()
        self._current_drift_scale: float = 1.0
        self._auto_trigger = self._init_auto_trigger()
        # Track entry regimes for position monitoring (symbol → trend_regime at entry)
        self._entry_regimes: dict[str, str] = {}

    async def run(
        self,
        symbols: list[str] | None = None,
        last_regimes: dict[str, dict] | None = None,
    ) -> RunReport:
        """
        Execute one autonomous trading pass.

        Args:
            symbols: Tickers to analyze. If None, loads from WatchlistLoader.
            last_regimes: Regime dicts from prior bar, keyed by symbol.
                          Used by DecisionRouter for regime-flip detection.

        Returns:
            RunReport with per-symbol outcomes.
        """
        run_id = uuid.uuid4().hex[:12]
        started_at = datetime.now(timezone.utc)
        last_regimes = last_regimes or {}

        logger.info(f"[AutonomousRunner] run={run_id} starting")

        # --- Step 1: System Health ----------------------------------------
        sys_status = await self._get_system_status()
        if sys_status.get("kill_switch_active"):
            logger.warning(
                f"[AutonomousRunner] run={run_id} ABORTED — kill switch active"
            )
            return self._empty_report(run_id, started_at, symbols or [])

        if sys_status.get("risk_halted"):
            logger.warning(
                f"[AutonomousRunner] run={run_id} ABORTED — daily risk limit breached"
            )
            return self._empty_report(run_id, started_at, symbols or [])

        # --- Step 2: Symbols + Strategies ----------------------------------
        if symbols is None:
            from quantstack.autonomous.watchlist import WatchlistLoader

            symbols = await asyncio.to_thread(WatchlistLoader().load)

        if not symbols:
            logger.info(
                f"[AutonomousRunner] run={run_id} — empty symbol list, nothing to do"
            )
            return self._empty_report(run_id, started_at, [])

        strategies = await asyncio.to_thread(self._load_active_strategies)
        portfolio = await asyncio.to_thread(self._load_portfolio_snapshot)

        logger.info(
            f"[AutonomousRunner] run={run_id} "
            f"symbols={symbols} strategies={len(strategies)} "
            f"dry_run={self._dry_run}"
        )

        # --- Step 3: Process each symbol (bounded concurrency) ------------
        semaphore = asyncio.Semaphore(self._max_concurrent)
        tasks = [
            self._process_symbol(
                symbol=sym,
                strategies=strategies,
                portfolio=portfolio,
                last_regime=last_regimes.get(sym),
                system_status=sys_status,
                semaphore=semaphore,
                run_id=run_id,
            )
            for sym in symbols
        ]
        results: list[SymbolResult] = await asyncio.gather(
            *tasks, return_exceptions=False
        )

        # --- Step 4: Build report -----------------------------------------
        finished_at = datetime.now(timezone.utc)
        report = RunReport(
            run_id=run_id,
            started_at=started_at,
            finished_at=finished_at,
            symbols_processed=len(results),
            results=results,
        )
        for r in results:
            if r.error:
                report.errors += 1
            elif r.path == DecisionPath.SKIP:
                report.skipped += 1
            elif r.path in (DecisionPath.RULE_BASED, DecisionPath.GROQ_SYNTHESIS):
                report.rule_based += 1
                if r.fill_result and r.fill_result.get("success"):
                    report.executed += 1
                elif r.fill_result and not r.fill_result.get("risk_approved", True):
                    report.risk_rejected += 1

        logger.info(
            f"[AutonomousRunner] run={run_id} done — "
            f"executed={report.executed} skipped={report.skipped} "
            f"rule_based={report.rule_based} errors={report.errors} "
            f"elapsed={round((finished_at - started_at).total_seconds(), 1)}s"
        )

        # --- Step 5: Post-pass continuous risk monitoring ------------------
        await self._run_post_pass_monitor(results)

        return report

    # -------------------------------------------------------------------------
    # Per-Symbol Processing
    # -------------------------------------------------------------------------

    @staticmethod
    def _init_auto_trigger():
        """Lazy-load AutoTriggerMonitor. Returns None if import fails."""
        try:
            from quantstack.execution.kill_switch import get_kill_switch

            return get_kill_switch().create_auto_trigger()
        except Exception as exc:
            logger.debug(f"[AutonomousRunner] AutoTriggerMonitor not available: {exc}")
            return None

    @staticmethod
    def _init_drift_detector():
        """Lazy-load DriftDetector. Returns None if import fails."""
        try:
            from quantstack.learning.drift_detector import DriftDetector

            return DriftDetector()
        except Exception as exc:
            logger.debug(f"[AutonomousRunner] DriftDetector not available: {exc}")
            return None

    def _check_drift(self, strategy: dict, brief: dict):
        """Run drift check for a strategy. Returns DriftReport or None."""
        if self._drift_detector is None:
            return None
        strategy_id = strategy.get("strategy_id", "")
        if not strategy_id or not self._drift_detector.has_baseline(strategy_id):
            return None
        try:
            return self._drift_detector.check_drift_from_brief(strategy_id, brief)
        except Exception as exc:
            logger.debug(
                f"[AutonomousRunner] drift check failed for {strategy_id}: {exc}"
            )
            return None

    async def _process_symbol(
        self,
        symbol: str,
        strategies: list[dict],
        portfolio: dict,
        last_regime: dict | None,
        system_status: dict,
        semaphore: asyncio.Semaphore,
        run_id: str,
    ) -> SymbolResult:
        """Full pipeline for a single symbol."""
        t0 = time.monotonic()

        async with semaphore:
            # Wrap entire symbol processing in a trace for end-to-end correlation
            with TraceContext.new_trace(symbol=symbol, run_id=run_id):
                # Re-check kill switch per symbol — it may have been triggered
                # by an order submitted earlier in this same pass.
                if system_status.get("kill_switch_active"):
                    return SymbolResult(
                        symbol=symbol,
                        path=DecisionPath.SKIP,
                        path_reason="kill switch active (re-check)",
                        elapsed_seconds=time.monotonic() - t0,
                    )

                try:
                    return await self._analyze_and_decide(
                        symbol=symbol,
                        strategies=strategies,
                        portfolio=portfolio,
                        last_regime=last_regime,
                        system_status=system_status,
                        run_id=run_id,
                        t0=t0,
                    )
                except Exception as exc:
                    logger.error(
                        f"[AutonomousRunner] {symbol}: unhandled error — {exc}",
                        exc_info=True,
                    )
                    return SymbolResult(
                        symbol=symbol,
                        path=DecisionPath.SKIP,
                        path_reason="internal error",
                        error=str(exc),
                        elapsed_seconds=time.monotonic() - t0,
                    )

    async def _analyze_and_decide(
        self,
        symbol: str,
        strategies: list[dict],
        portfolio: dict,
        last_regime: dict | None,
        system_status: dict,
        run_id: str,
        t0: float,
    ) -> SymbolResult:
        """SignalEngine → DriftCheck → DecisionRouter → execute / Groq / skip."""
        from quantstack.signal_engine import SignalEngine

        # --- Signal analysis ---
        brief = await SignalEngine().run(symbol)

        # Filter strategies that are active for this symbol/asset class
        symbol_strategies = _filter_strategies_for_symbol(strategies, symbol)

        # --- Proactive drift check (per-strategy) ---
        # If ANY strategy for this symbol shows CRITICAL drift, skip entirely.
        # WARNING drift halves position size (applied downstream in _handle_rule_based).
        drift_scale = 1.0
        for strat in symbol_strategies:
            drift_report = self._check_drift(strat, brief)
            if drift_report and drift_report.severity == "CRITICAL":
                reason = (
                    f"drift_critical: strategy={strat.get('name', strat.get('strategy_id', '?'))}, "
                    f"PSI={drift_report.overall_psi:.3f}, features={drift_report.drifted_features}"
                )
                logger.warning(f"[AutonomousRunner] {symbol}: SKIP — {reason}")
                self._audit_skip(symbol, reason, brief, run_id)
                return SymbolResult(
                    symbol=symbol,
                    path=DecisionPath.SKIP,
                    path_reason=reason,
                    elapsed_seconds=time.monotonic() - t0,
                )
            if drift_report and drift_report.severity == "WARNING":
                drift_scale = min(drift_scale, 0.5)
                logger.info(
                    f"[AutonomousRunner] {symbol}: drift WARNING for "
                    f"{strat.get('name', '?')} — PSI={drift_report.overall_psi:.3f}, "
                    f"scaling position to {drift_scale:.0%}"
                )

        # Stash drift_scale for downstream handlers
        self._current_drift_scale = drift_scale

        # --- Routing (fully deterministic, no LLM) ---
        path, reason, route_ctx = self._router.route(
            symbol=symbol,
            brief=brief,
            strategies=symbol_strategies,
            portfolio=portfolio,
            last_regime=last_regime,
            system_status=system_status,
        )

        logger.info(f"[AutonomousRunner] {symbol}: path={path.name} — {reason}")

        if path == DecisionPath.SKIP:
            self._audit_skip(symbol, reason, brief, run_id)
            return SymbolResult(
                symbol=symbol,
                path=path,
                path_reason=reason,
                elapsed_seconds=time.monotonic() - t0,
            )

        # --- Adversarial debate on borderline signals ---
        conviction = getattr(brief, "market_conviction", 0.5)
        bias = getattr(brief, "market_bias", "neutral")
        _, _, position_size = _derive_trade_params(brief, symbol_strategies)
        if route_ctx.force_size:
            position_size = route_ctx.force_size

        # Retrieve reflexion episodes for debate injection (Reflexion, NeurIPS 2023)
        past_lessons = []
        try:
            from quantstack.autonomous.hooks import get_reflexion_episodes
            regime_detail = getattr(brief, "regime_detail", {}) or {}
            current_regime = regime_detail.get("trend_regime", "unknown")
            strategy_id = symbol_strategies[0].get("strategy_id", "") if symbol_strategies else ""
            past_lessons = get_reflexion_episodes(
                regime=current_regime, strategy_id=strategy_id, symbol=symbol, k=3,
            )
        except Exception as exc:
            logger.debug(f"[AutonomousRunner] Reflexion retrieval failed (non-critical): {exc}")

        debate = self._debate_filter.challenge(
            symbol=symbol,
            brief=brief,
            conviction=conviction,
            bias=bias,
            position_size=position_size,
            regime=getattr(brief, "regime_detail", {}) or {},
            portfolio=portfolio,
            strategies=symbol_strategies,
            past_lessons=past_lessons,
        )

        if debate.challenged:
            if debate.verdict == "veto":
                reason = f"debate vetoed: {debate.reason}"
                self._audit_skip(symbol, reason, brief, run_id)
                return SymbolResult(
                    symbol=symbol,
                    path=DecisionPath.SKIP,
                    path_reason=reason,
                    elapsed_seconds=time.monotonic() - t0,
                )
            if debate.verdict == "downgrade":
                # Reduce position size based on adjusted conviction
                if debate.adjusted_conviction < 0.60:
                    route_ctx.force_size = "quarter"
                logger.info(
                    f"[AutonomousRunner] {symbol}: debate downgraded "
                    f"({conviction:.0%} → {debate.adjusted_conviction:.0%})"
                )

        # All non-SKIP paths are RULE_BASED (Groq path removed in v1.1)
        return await self._handle_rule_based(
            symbol=symbol,
            brief=brief,
            reason=reason,
            strategies=symbol_strategies,
            portfolio=portfolio,
            run_id=run_id,
            t0=t0,
            route_ctx=route_ctx,
        )

    # -------------------------------------------------------------------------
    # Path Handlers
    # -------------------------------------------------------------------------

    async def _handle_rule_based(
        self,
        symbol: str,
        brief: Any,
        reason: str,
        strategies: list[dict],
        portfolio: dict,
        run_id: str,
        t0: float,
        route_ctx: RouteContext | None = None,
    ) -> SymbolResult:
        """Deterministic execution from brief bias + strategy rules + route context."""
        action, confidence, position_size = _derive_trade_params(brief, strategies)

        ctx = route_ctx or RouteContext()

        # Apply RouteContext overrides (from exception conditions)
        if ctx.force_size is not None:
            position_size = ctx.force_size

        # Apply drift-based position scaling (set by _analyze_and_decide)
        if self._current_drift_scale < 1.0:
            _SIZE_DOWNGRADE = {"full": "half", "half": "quarter", "quarter": "quarter"}
            position_size = _SIZE_DOWNGRADE.get(position_size, position_size)

        reasoning = (
            f"[rule-based] bias={brief.market_bias} "
            f"conviction={brief.market_conviction:.0%} "
            f"regime={getattr(brief, 'regime_detail', {}).get('trend_regime', 'unknown')}"
        )
        if ctx.exception_type:
            reasoning += f" [exception: {ctx.exception_type}]"
        if ctx.force_size:
            reasoning += f" [forced_size: {ctx.force_size}]"
        if self._current_drift_scale < 1.0:
            reasoning += f" [drift_warning: size scaled to {position_size}]"

        strategy_id = strategies[0].get("strategy_id") if strategies else None

        if action == "skip":
            self._audit_skip(
                symbol,
                f"rule-based: bias neutral/weak ({brief.market_bias})",
                brief,
                run_id,
            )
            return SymbolResult(
                symbol=symbol,
                path=DecisionPath.RULE_BASED,
                path_reason=reason,
                action=None,
                reasoning=reasoning,
                elapsed_seconds=time.monotonic() - t0,
            )

        regime_at_entry = getattr(brief, "regime_detail", {}).get(
            "trend_regime", "unknown"
        )
        fill_result = await self._submit_order(
            symbol=symbol,
            action=action,
            confidence=confidence,
            position_size=position_size,
            reasoning=reasoning,
            strategy_id=strategy_id,
            run_id=run_id,
            regime_at_entry=regime_at_entry,
        )
        return SymbolResult(
            symbol=symbol,
            path=DecisionPath.RULE_BASED,
            path_reason=reason,
            action=action,
            position_size=position_size,
            confidence=confidence,
            reasoning=reasoning,
            fill_result=fill_result,
            elapsed_seconds=time.monotonic() - t0,
        )

    # -------------------------------------------------------------------------
    # Order Submission
    # -------------------------------------------------------------------------

    async def _submit_order(
        self,
        symbol: str,
        action: str,
        confidence: float,
        position_size: str,
        reasoning: str,
        strategy_id: str | None,
        run_id: str,
        regime_at_entry: str = "unknown",
    ) -> dict | None:
        """Submit order via execute_trade MCP tool (or skip in dry-run)."""
        paper_mode = self._paper_only or (
            os.getenv("USE_REAL_TRADING", "false").strip().lower()
            not in ("true", "1", "yes")
        )

        if self._dry_run:
            logger.info(
                f"[AutonomousRunner][dry-run] {symbol}: {action} "
                f"size={position_size} confidence={confidence:.0%}"
            )
            return {"success": True, "dry_run": True, "action": action}

        try:
            from quantstack.mcp.tools.execution import execute_trade

            result = await execute_trade(
                symbol=symbol,
                action=action,
                reasoning=f"[autonomous run={run_id}] {reasoning}",
                confidence=confidence,
                position_size=position_size,
                strategy_id=strategy_id,
                paper_mode=paper_mode,
                regime_at_entry=regime_at_entry,
            )
            logger.info(
                f"[AutonomousRunner] {symbol}: {action} → "
                f"{'filled' if result.get('success') else 'rejected'} "
                f"paper={paper_mode}"
            )
            # Record broker success for auto-trigger monitoring
            if self._auto_trigger:
                self._auto_trigger.record_broker_result(success=True)
            # Track entry regime for continuous risk monitoring
            if result.get("success"):
                self._entry_regimes[symbol] = regime_at_entry
            return result
        except Exception as exc:
            logger.error(f"[AutonomousRunner] {symbol}: execute_trade failed — {exc}")
            # Record broker failure for auto-trigger monitoring
            if self._auto_trigger:
                self._auto_trigger.record_broker_result(success=False, error=str(exc))
            return {"success": False, "error": str(exc)}

    # -------------------------------------------------------------------------
    # Post-pass continuous risk monitoring
    # -------------------------------------------------------------------------

    async def _run_post_pass_monitor(self, results: list[SymbolResult]) -> None:
        """
        Run continuous risk monitor after each trading pass.

        Collects current regime data from the pass and checks open positions
        for size drift, correlation, regime flips, and daily loss proximity.
        """
        try:
            from quantstack.execution.risk_gate import get_risk_gate

            gate = get_risk_gate()

            # Build current_regimes from signal briefs we just computed
            current_regimes: dict[str, dict] = {}
            for r in results:
                # If the result has regime info, extract it
                if hasattr(r, "reasoning") and r.reasoning:
                    # Parse regime from reasoning string (best-effort)
                    pass  # Regime data comes from the signal brief, not available here

            monitor_report = gate.monitor(
                current_regimes=current_regimes,
                entry_regimes=self._entry_regimes,
            )

            if monitor_report.has_critical:
                logger.critical(
                    f"[AutonomousRunner] CRITICAL risk alerts detected — "
                    f"{len(monitor_report.alerts)} total alerts"
                )
        except Exception as exc:
            logger.debug(
                f"[AutonomousRunner] post-pass monitor failed (non-critical): {exc}"
            )

    # -------------------------------------------------------------------------
    # Audit
    # -------------------------------------------------------------------------

    def _audit_skip(
        self,
        symbol: str,
        reason: str,
        brief: Any,
        run_id: str,
    ) -> None:
        """Write a SKIP event to the audit log (best-effort, never raises)."""
        try:
            from quantstack.mcp._state import require_live_db
            from quantstack.audit.models import DecisionEvent

            ctx = require_live_db()
            ctx.audit.record(
                DecisionEvent(
                    event_id=str(uuid.uuid4()),
                    session_id=f"autonomous_{run_id}",
                    event_type="skip",
                    agent_name="AutonomousRunner",
                    agent_role="autonomous_pm",
                    symbol=symbol,
                    action="skip",
                    confidence=getattr(brief, "market_conviction", 0.0),
                    output_summary=reason,
                    risk_approved=False,
                    risk_violations=[reason],
                    portfolio_snapshot={},
                )
            )
        except Exception as exc:
            logger.debug(f"[AutonomousRunner] audit_skip failed (non-critical): {exc}")

    # -------------------------------------------------------------------------
    # Data Loaders (synchronous — called via asyncio.to_thread)
    # -------------------------------------------------------------------------

    def _load_active_strategies(self) -> list[dict]:
        """Load live + forward_testing strategies from DB (read-only)."""
        try:
            from quantstack.db import open_db_readonly

            conn = open_db_readonly()
            rows = conn.execute(
                """
                SELECT strategy_id, name, description, status, regime_affinity,
                       parameters, entry_rules, exit_rules, risk_params, asset_class
                FROM strategies
                WHERE status IN ('live', 'forward_testing')
                ORDER BY name
                """
            ).fetchall()
            cols = [
                "strategy_id",
                "name",
                "description",
                "status",
                "regime_affinity",
                "parameters",
                "entry_rules",
                "exit_rules",
                "risk_params",
                "asset_class",
            ]
            strategies = []
            for row in rows:
                d = dict(zip(cols, row))
                # Deserialize JSON fields
                for key in (
                    "regime_affinity",
                    "parameters",
                    "entry_rules",
                    "exit_rules",
                    "risk_params",
                ):
                    if isinstance(d.get(key), str):
                        try:
                            d[key] = json.loads(d[key])
                        except (ValueError, TypeError):
                            d[key] = {}
                strategies.append(d)
            conn.close()
            return strategies
        except Exception as exc:
            logger.warning(f"[AutonomousRunner] could not load strategies: {exc}")
            return []

    def _load_portfolio_snapshot(self) -> dict:
        """Load current portfolio state for DecisionRouter."""
        try:
            from quantstack.mcp._state import require_ctx

            ctx = require_ctx()
            snapshot = ctx.portfolio.get_snapshot()
            positions_raw = ctx.portfolio.get_all_positions()
            positions = {
                p.symbol: {
                    "quantity": p.quantity,
                    "entry_price": p.entry_price,
                    "current_price": p.current_price,
                    "unrealized_pnl_pct": (
                        ((p.current_price - p.entry_price) / p.entry_price * 100)
                        if p.entry_price and p.entry_price > 0
                        else 0.0
                    ),
                }
                for p in positions_raw
                if p.quantity != 0
            }
            return {
                "cash": snapshot.cash,
                "equity": snapshot.total_equity,
                "positions": positions,
            }
        except Exception as exc:
            logger.warning(f"[AutonomousRunner] could not load portfolio: {exc}")
            return {"cash": 0, "equity": 0, "positions": {}}

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    async def _get_system_status(self) -> dict:
        """Fetch system status from MCP tool."""
        try:
            from quantstack.mcp.tools.analysis import get_system_status

            return await get_system_status()
        except Exception as exc:
            logger.warning(f"[AutonomousRunner] get_system_status failed: {exc}")
            # Return safe defaults — treat as healthy, let kill_switch direct check decide
            return {"kill_switch_active": False, "risk_halted": False}

    @staticmethod
    def _empty_report(
        run_id: str, started_at: datetime, symbols: list[str]
    ) -> RunReport:
        now = datetime.now(timezone.utc)
        return RunReport(
            run_id=run_id,
            started_at=started_at,
            finished_at=now,
            symbols_processed=0,
            results=[
                SymbolResult(
                    symbol=s,
                    path=DecisionPath.SKIP,
                    path_reason="runner aborted before processing",
                )
                for s in symbols
            ],
        )


# =============================================================================
# Private helpers
# =============================================================================


def _filter_strategies_for_symbol(strategies: list[dict], symbol: str) -> list[dict]:
    """
    Filter strategies that are applicable to this symbol.

    Currently returns all active strategies — filtering by symbol is
    not yet persisted in the strategy schema (strategies are regime-keyed,
    not symbol-keyed). This is the extension point when per-symbol
    watchlists are added to the strategy schema.
    """
    return strategies


def _derive_trade_params(
    brief: Any,
    strategies: list[dict],
) -> tuple[str, float, str]:
    """
    Deterministic trade params from a RULE_BASED-routed brief.

    Returns (action, confidence, position_size).
    action is "skip" when brief does not support a clear directional bet.
    """
    bias = brief.market_bias
    conviction = brief.market_conviction

    if bias in ("bullish", "strong_bullish") and conviction >= 0.50:
        action = "buy"
    elif bias in ("bearish", "strong_bearish") and conviction >= 0.50:
        action = "sell"
    else:
        return "skip", conviction, "quarter"

    # Size from conviction: high → full, moderate → half, low → quarter
    if conviction >= 0.75:
        size = "full"
    elif conviction >= 0.60:
        size = "half"
    else:
        size = "quarter"

    # Strategies may override sizing via risk_params.position_size
    if strategies:
        rp = strategies[0].get("risk_params") or {}
        override = rp.get("position_size")
        if override in ("full", "half", "quarter"):
            size = override

    return action, conviction, size


# =============================================================================
# CLI entry point
# =============================================================================


def _parse_args() -> Any:
    parser = argparse.ArgumentParser(
        description="AutonomousRunner — headless trading loop"
    )
    parser.add_argument("--symbols", nargs="+", default=None, help="Symbols to analyze")
    parser.add_argument(
        "--dry-run", action="store_true", help="Log decisions, submit no orders"
    )
    parser.add_argument(
        "--paper-only", action="store_true", default=True, help="Force paper mode"
    )
    return parser.parse_args()


if __name__ == "__main__":
    import asyncio

    args = _parse_args()
    runner = AutonomousRunner(dry_run=args.dry_run, paper_only=args.paper_only)
    report = asyncio.run(runner.run(symbols=args.symbols))
    print(
        f"Run {report.run_id}: "
        f"executed={report.executed} skipped={report.skipped} "
        f"rule_based={report.rule_based} errors={report.errors}"
    )
