# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Trade execution hooks — fire-and-forget callbacks wired into the
execution pipeline. Non-blocking: failures are logged, never fatal.

Lives at L8 (learning/optimization peer). Registers callbacks into
execution.hook_registry (L7) at startup — dependency flows downward.

Hooks:
  - on_trade_close: fires ReflectionManager + ReflexionMemory after every position close
  - on_daily_close: fires daily reflection summary at market close
  - find_similar_situations: SQL query for pre-trade context (regime + symbol + strategy)
  - get_reflexion_episodes: structured episodic memory for debate filter injection
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from loguru import logger

from quantstack.execution.hook_registry import register
from quantstack.learning.outcome_tracker import OutcomeTracker
from quantstack.learning.prompt_tuner import PromptTuner
from quantstack.learning.reflection import ReflectionManager
from quantstack.optimization.credit_assignment import CreditAssigner
from quantstack.optimization.reflexion_memory import ReflexionMemory

# Module-level singletons — initialized lazily on first use
_reflection_mgr: ReflectionManager | None = None
_reflexion_mem: ReflexionMemory | None = None
_credit_assigner: CreditAssigner | None = None
_prompt_tuner: PromptTuner | None = None


def _get_prompt_tuner() -> PromptTuner | None:
    """Lazy-init PromptTuner singleton."""
    global _prompt_tuner
    if _prompt_tuner is None:
        _prompt_tuner = PromptTuner()
    return _prompt_tuner


def _get_reflection_manager() -> ReflectionManager | None:
    """Lazy-init ReflectionManager singleton."""
    global _reflection_mgr
    if _reflection_mgr is None:
        _reflection_mgr = ReflectionManager()
    return _reflection_mgr


def _get_reflexion_memory() -> ReflexionMemory | None:
    """Lazy-init ReflexionMemory singleton."""
    global _reflexion_mem
    if _reflexion_mem is None:
        _reflexion_mem = ReflexionMemory()
    return _reflexion_mem


def _get_credit_assigner() -> CreditAssigner | None:
    """Lazy-init CreditAssigner singleton."""
    global _credit_assigner
    if _credit_assigner is None:
        _credit_assigner = CreditAssigner()
    return _credit_assigner


def _update_skill_tracker(
    agent_name: str,
    prediction_correct: bool,
    realized_pnl: float,
) -> None:
    """Fire-and-forget SkillTracker update (Wire 4, Section 03).

    Never raises — tracking side-effects must not break trade hooks.
    """
    try:
        from quantstack.knowledge.store import KnowledgeStore
        from quantstack.learning.skill_tracker import SkillTracker
        store = KnowledgeStore()
        tracker = SkillTracker(store)
        tracker.update_agent_skill(agent_name, prediction_correct, signal_pnl=realized_pnl)
    except Exception as exc:
        logger.warning(f"[hooks] SkillTracker update failed for {agent_name}: {exc}")


def on_trade_close(
    symbol: str,
    strategy_id: str,
    action: str,
    entry_price: float,
    exit_price: float,
    realized_pnl_pct: float,
    holding_days: int = 0,
    regime_at_entry: str = "unknown",
    regime_at_exit: str = "unknown",
    conviction: float = 0.0,
    signals_summary: str = "",
    position_size: str = "half",
    debate_verdict: str = "",
    strategy_regime_affinity: float = 0.5,
    trade_id: int = 0,
    **_kwargs: Any,
) -> None:
    """Fire after every trade close. Non-blocking, best-effort.

    Called from PortfolioState.close_position or execute_trade.
    Records outcome in ReflectionManager (raw journal), creates a classified
    ReflexionEpisode for losses > 1%, and runs step-level credit assignment
    for P&L attribution.
    """
    # 1. Raw reflection (always)
    try:
        mgr = _get_reflection_manager()
        if mgr is None:
            return
        ref = mgr.record_outcome(
            symbol=symbol,
            strategy_id=strategy_id,
            action=action,
            entry_price=entry_price,
            exit_price=exit_price,
            realized_pnl_pct=realized_pnl_pct,
            holding_days=holding_days,
            regime_at_entry=regime_at_entry,
            regime_at_exit=regime_at_exit,
            conviction=conviction,
            signals_entry=signals_summary,
        )
    except Exception as exc:
        logger.warning(f"[hooks] on_trade_close reflection failed: {exc}")
        return

    # 2. research_queue INSERT for losses > 1% — feeds AutoResearchClaw bug_fix pipeline
    if realized_pnl_pct < -1.0:
        try:
            from quantstack.db import db_conn
            import json as _json
            with db_conn() as _conn:
                _conn.execute(
                    """
                    INSERT INTO research_queue (task_type, priority, context_json, source)
                    VALUES ('bug_fix', %s, %s, 'trade_reflector')
                    """,
                    [
                        7 if realized_pnl_pct < -3.0 else 5,
                        _json.dumps({
                            "symbol": symbol,
                            "strategy_id": strategy_id,
                            "realized_pnl_pct": realized_pnl_pct,
                            "regime_at_entry": regime_at_entry,
                            "regime_at_exit": regime_at_exit,
                            "holding_days": holding_days,
                            "conviction": conviction,
                            "debate_verdict": debate_verdict,
                        }),
                    ],
                )
        except Exception as exc:
            logger.warning(f"[hooks] research_queue insert failed: {exc}")

    # 3. PromptTuner outcome — losses only (≥ 5 same-pattern samples before surfacing)
    if realized_pnl_pct < -1.0:
        try:
            tuner = _get_prompt_tuner()
            if tuner is not None:
                direction = "bullish" if action == "buy" else "bearish"
                tuner.record_outcome(
                    desk="alpha-research",
                    prediction={"direction": direction, "conviction": conviction},
                    outcome={
                        "realized_return": realized_pnl_pct / 100.0,
                        "direction_correct": False,
                        "regime_at_exit": regime_at_exit,
                    },
                    symbol=symbol,
                )
        except Exception as exc:
            logger.warning(f"[hooks] on_trade_close prompt tuner failed: {exc}")

    # 3. Structured reflexion episode (losses only)
    if realized_pnl_pct < -1.0:
        try:
            mem = _get_reflexion_memory()
            if mem is not None:
                mem.record_episode(ref)
        except Exception as exc:
            logger.warning(f"[hooks] on_trade_close reflexion episode failed: {exc}")

    # 4. SkillTracker — record per-agent outcome (Wire 4, Section 03)
    _update_skill_tracker(
        agent_name=debate_verdict or "unknown_agent",
        prediction_correct=realized_pnl_pct > 0,
        realized_pnl=realized_pnl_pct,
    )

    # 3. Step-level credit assignment (every trade)
    try:
        assigner = _get_credit_assigner()
        if assigner is not None:
            trade_context = {
                "trade_id": trade_id,
                "realized_pnl_pct": realized_pnl_pct,
                "regime_at_entry": regime_at_entry,
                "regime_at_exit": regime_at_exit,
                "strategy_id": strategy_id,
                "conviction": conviction,
                "position_size": position_size,
                "debate_verdict": debate_verdict,
                "signals_present": bool(signals_summary),
                "strategy_regime_affinity": strategy_regime_affinity,
            }
            credits = assigner.assign_heuristic(trade_context)
            worst = assigner.get_worst_step(credits)
            if worst and worst.credit_score < -0.3:
                logger.info(
                    f"[hooks] Credit attribution: worst step={worst.step_type} "
                    f"score={worst.credit_score:.2f} — {worst.evidence}"
                )
    except Exception as exc:
        logger.warning(f"[hooks] on_trade_close credit assignment failed: {exc}")

    # 6. StrategyBreaker — record trade outcome for circuit breaker evaluation
    # (Wire 3b: feeds consecutive-loss and drawdown tracking)
    try:
        from quantstack.execution.strategy_breaker import StrategyBreaker
        breaker = StrategyBreaker()
        # Use realized_pnl_pct directly as pnl, synthetic 100.0 equity base.
        # StrategyBreaker tracks consecutive losses and drawdown percentages,
        # so the relative magnitude is what matters, not absolute dollars.
        breaker.record_trade(strategy_id, pnl=realized_pnl_pct, equity=100.0 + realized_pnl_pct)
    except Exception as exc:
        logger.warning(f"[hooks] StrategyBreaker record_trade failed: {exc}")

    # 7. ICAttributionTracker — record per-collector signal→return observations
    # (Wire 5a: data collection runs always, weight consumption is flag-gated)
    if signals_summary:
        try:
            import json as _json5
            collectors = _json5.loads(signals_summary) if isinstance(signals_summary, str) else {}
            if isinstance(collectors, dict) and collectors:
                from quantstack.learning.ic_attribution import ICAttributionTracker
                ic_tracker = ICAttributionTracker()
                fwd_return = realized_pnl_pct / 100.0
                for collector_name, signal_val in collectors.items():
                    if isinstance(signal_val, (int, float)):
                        ic_tracker.record(
                            symbol=symbol,
                            collector=collector_name,
                            signal_value=float(signal_val),
                            forward_return=fwd_return,
                            regime=regime_at_entry,
                        )
        except (_json5.JSONDecodeError, TypeError):
            pass  # signals_summary not JSON-parseable yet — IC data doesn't accumulate
        except Exception as exc:
            logger.warning(f"[hooks] ICAttributionTracker record failed: {exc}")

    # 8. P05 §5.3: Record conviction factor breakdown for empirical calibration
    conviction_factors = _kwargs.get("conviction_factors")
    if isinstance(conviction_factors, dict) and conviction_factors:
        try:
            from quantstack.db import db_conn
            fwd_return = realized_pnl_pct / 100.0
            with db_conn() as _conn:
                for factor_name, factor_value in conviction_factors.items():
                    if isinstance(factor_value, (int, float)):
                        _conn.execute(
                            """
                            INSERT INTO conviction_calibration
                                (factor_name, factor_value, forward_return, regime)
                            VALUES (%s, %s, %s, %s)
                            """,
                            [factor_name, float(factor_value), fwd_return, regime_at_entry],
                        )
        except Exception as exc:
            logger.warning(f"[hooks] conviction_calibration record failed: {exc}")


def on_daily_close(
    snapshot_date: date,
    daily_pnl: float = 0.0,
    daily_return_pct: float = 0.0,
    closed_trades: list[dict] | None = None,
    **_kwargs: Any,
) -> None:
    """Fire at market close (after equity snapshot). Non-blocking, best-effort.

    Called from EquityTracker.snapshot_daily.
    """
    try:
        mgr = _get_reflection_manager()
        if mgr is None:
            return
        mgr.daily_reflection(
            snapshot_date=snapshot_date,
            daily_pnl=daily_pnl,
            daily_return_pct=daily_return_pct,
            closed_trades=closed_trades,
        )
    except Exception as exc:
        logger.warning(f"[hooks] on_daily_close failed: {exc}")


def find_similar_situations(
    symbol: str,
    regime: str,
    signals: str,
    top_k: int = 3,
) -> list[dict]:
    """Query past trade situations similar to current context.

    Use before entering a trade to surface relevant lessons.
    Returns list of dicts with keys: symbol, pnl_pct, lesson, regime.
    """
    try:
        mgr = _get_reflection_manager()
        if mgr is None:
            return []
        matches = mgr.find_similar(symbol, regime, signals, top_k=top_k)
        return [
            {
                "symbol": m.symbol,
                "pnl_pct": m.realized_pnl_pct,
                "lesson": m.lesson,
                "regime": m.regime_at_entry,
                "strategy": m.strategy_id,
            }
            for m in matches
            if m.lesson  # Only return entries with lessons
        ]
    except Exception as exc:
        logger.warning(f"[hooks] find_similar_situations failed: {exc}")
        return []


def get_reflexion_episodes(
    regime: str,
    strategy_id: str = "",
    symbol: str = "",
    k: int = 3,
) -> list:
    """Retrieve structured reflexion episodes for debate filter injection.

    Returns list of ReflexionEpisode objects (or empty list on failure).
    Used by AutonomousRunner to condition the debate filter on past failures.
    """
    try:
        mem = _get_reflexion_memory()
        if mem is None:
            return []
        return mem.get_relevant(regime=regime, strategy_id=strategy_id, symbol=symbol, k=k)
    except Exception as exc:
        logger.warning(f"[hooks] get_reflexion_episodes failed: {exc}")
        return []


# ---------------------------------------------------------------------------
# Auto-registration: when this module is imported, wire callbacks into the
# execution-layer hook registry so portfolio_state / trade_service can fire
# them without knowing about learning/optimization.
# ---------------------------------------------------------------------------

def _update_tca_ewma(
    order_id: str,
    symbol: str,
    fill_timestamp: datetime,
    arrival_price: float,
    fill_price: float,
    fill_quantity: int,
    daily_volume: int,
) -> None:
    """Fire-and-forget TCA EWMA update (QS-E6).

    After each fill, decomposes realized costs into spread/impact and feeds
    into per-(symbol, time_bucket) EWMA parameters. Pre-trade forecasts then
    query these values so cost estimates calibrate from actual execution.

    Never raises — TCA side-effects must not break trade hooks.
    """
    try:
        from quantstack.db import db_conn
        from quantstack.execution.tca_ewma import update_ewma_after_fill

        with db_conn() as conn:
            update_ewma_after_fill(
                conn=conn,
                order_id=order_id,
                symbol=symbol,
                fill_timestamp=fill_timestamp,
                arrival_price=arrival_price,
                fill_price=fill_price,
                fill_quantity=fill_quantity,
                adv=float(daily_volume),
            )
    except Exception as exc:
        logger.warning(f"[hooks] TCA EWMA update failed for {symbol}: {exc}")


def _on_trade_fill(
    strategy_id: str,
    symbol: str,
    action: str,
    fill_price: float,
    fill_quantity: int,
    fill_timestamp: datetime,
    arrival_price: float,
    daily_volume: int,
    order_id: str,
    session_id: str = "",
    regime_at_entry: str = "unknown",
    **_kwargs: Any,
) -> None:
    """Outcome attribution hook — records entry/exit in OutcomeTracker.

    Also fires TCA EWMA recalibration on every fill.
    """
    try:
        tracker = OutcomeTracker()
        if action == "buy":
            tracker.record_entry(
                strategy_id, symbol, regime_at_entry, action, fill_price, session_id,
            )
        elif action == "sell":
            tracker.record_exit(strategy_id, symbol, fill_price)
            tracker.apply_learning(strategy_id)
    except Exception as exc:
        logger.warning(f"[hooks] trade_fill outcome attribution failed: {exc}")

    # TCA EWMA feedback — fire-and-forget (QS-E6)
    _update_tca_ewma(
        order_id=order_id,
        symbol=symbol,
        fill_timestamp=fill_timestamp,
        arrival_price=arrival_price,
        fill_price=fill_price,
        fill_quantity=fill_quantity,
        daily_volume=daily_volume,
    )


register("trade_close", on_trade_close)
register("daily_close", on_daily_close)
register("trade_fill", _on_trade_fill)
