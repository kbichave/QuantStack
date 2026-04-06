"""Node functions for the Trading Graph.

12 nodes + 2 conditional routers. Agent nodes use LLM reasoning with
tool access; tool nodes are deterministic data-gathering functions.

Each agent node uses the shared run_agent() executor which handles the
tool-calling loop: LLM decides to call tools → tools execute →
results fed back → LLM produces final answer.
"""

import json
import logging
from typing import Any

import numpy as np
import pandas as pd
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool

from quantstack.core.kelly_sizing import (
    KELLY_HARD_CEILING,
    KELLY_HARD_FLOOR,
    TARGET_VOL,
    VOL_SCALAR_CAP,
    compute_alpha_signals,
    compute_vol_state,
    regime_kelly_fraction,
)
from quantstack.core.portfolio.optimizer import covariance_matrix
from quantstack.core.signals.alt_data_normalizer import ALT_DATA_WEIGHT, get_alt_data_modifier
from quantstack.core.risk.position_sizing import ewma_volatility
from quantstack.portfolio.optimizer import optimize_portfolio
from quantstack.core.risk.safety_gate import RiskDecision, SafetyGate
from quantstack.db import db_conn
from quantstack.graphs.agent_executor import parse_json_response, run_agent
from quantstack.graphs.config import AgentConfig
from quantstack.graphs.state import TradingState
from quantstack.performance.trade_evaluator import create_trade_evaluator
from quantstack.runners import is_market_hours

logger = logging.getLogger(__name__)


def make_market_intel(llm: BaseChatModel, config: AgentConfig, tools: list[BaseTool] | None = None):
    """Create the market_intel node (agent with tool access).

    Runs pre-market (8:30-9:30 AM ET) to gather macro news, analyst actions,
    and position-specific alerts. Also triggered by MARKET_MOVE events
    (>2% intraday index move). Outside the pre-market window on normal days,
    returns empty context (no-op).
    """
    tools = tools or []

    async def market_intel(state: TradingState) -> dict[str, Any]:
        import zoneinfo
        from datetime import datetime

        try:
            et = zoneinfo.ZoneInfo("America/New_York")
            now_et = datetime.now(et)
            hour, minute = now_et.hour, now_et.minute
            current_minutes = hour * 60 + minute

            # Pre-market window: 8:30-9:30 AM ET
            pre_market_start = 8 * 60 + 30
            pre_market_end = 9 * 60 + 30

            # Check for event bus trigger (MARKET_MOVE with magnitude > 2%)
            event_triggered = False
            portfolio_ctx = state.get("portfolio_context", {})
            if portfolio_ctx.get("market_move_trigger"):
                event_triggered = True

            in_window = pre_market_start <= current_minutes <= pre_market_end
            if not in_window and not event_triggered:
                return {"market_context": {}}

            mode = "event_triggered" if event_triggered else "morning_briefing"
            prompt = (
                f"Mode: {mode}\n"
                f"Cycle {state['cycle_number']}, regime: {state.get('regime', 'unknown')}.\n"
                f"Portfolio: {json.dumps(portfolio_ctx, default=str)}\n\n"
                "Gather pre-market intelligence:\n"
                "1. Search for major overnight macro news, Fed announcements, geopolitical events\n"
                "2. Check sector news for held positions\n"
                "3. Identify any analyst rating changes or earnings surprises\n"
                "4. Assess market sentiment and key risk factors\n\n"
                "Return structured JSON:\n"
                '{"headlines": [...], "risk_alerts": [...], "event_calendar": [...], '
                '"sector_news": {...}, "sentiment": "bullish|neutral|bearish"}'
            )
            text = await run_agent(llm, tools, config, prompt)
            parsed = parse_json_response(text, {})
            return {
                "market_context": parsed,
                "decisions": [{"node": "market_intel", "mode": mode}],
            }
        except Exception as exc:
            logger.error("market_intel failed: %s", exc)
            return {
                "market_context": {},
                "errors": [f"market_intel: {exc}"],
            }

    return market_intel


def make_earnings_analysis(llm: BaseChatModel, config: AgentConfig, tools: list[BaseTool] | None = None):
    """Create the earnings_analysis node (agent with tool access).

    Triggered when plan_day detects positions or watchlist symbols with
    earnings within 14 days. Analyzes historical beat rate, estimate
    revisions, IV premium ratio, and recommends hold/exit/hedge.
    """
    tools = tools or []

    async def earnings_analysis(state: TradingState) -> dict[str, Any]:
        earnings_symbols = state.get("earnings_symbols", [])
        if not earnings_symbols:
            return {"earnings_analysis": {}}

        try:
            prompt = (
                f"Symbols with earnings within 14 days: {earnings_symbols}\n\n"
                "For each symbol, analyze:\n"
                "1. Historical beat rate from last 4-8 quarters\n"
                "2. Analyst estimate revision direction (up/down/flat)\n"
                "3. IV premium ratio (implied move / expected move from history)\n"
                "4. Current position status (if held)\n\n"
                "For held positions: recommend HOLD_THROUGH, EXIT_BEFORE, or HEDGE_WITH_OPTIONS\n"
                "For entry candidates: assess whether earnings creates opportunity or risk\n\n"
                "Return JSON:\n"
                '{"analyses": [{"symbol": "...", "beat_rate": ..., '
                '"iv_premium_ratio": ..., "recommendation": "...", '
                '"options_suggestion": "...", "reasoning": "..."}]}'
            )
            text = await run_agent(llm, tools, config, prompt)
            parsed = parse_json_response(text, {"analyses": []})
            return {
                "earnings_analysis": parsed,
                "decisions": [{"node": "earnings_analysis", "count": len(earnings_symbols)}],
            }
        except Exception as exc:
            logger.error("earnings_analysis failed: %s", exc)
            return {
                "earnings_analysis": {},
                "errors": [f"earnings_analysis: {exc}"],
            }

    return earnings_analysis


def make_data_refresh():
    """Create the data_refresh node (deterministic, no LLM).

    Runs scheduled_refresh.run_intraday_refresh() every cycle during market
    hours.  Outside market hours this is a no-op (the trading graph is paused
    anyway, but we guard defensively).
    """

    async def data_refresh(state: TradingState) -> dict[str, Any]:
        if not is_market_hours():
            return {
                "data_refresh_summary": {"skipped": True, "reason": "outside_market_hours"},
            }

        try:
            from quantstack.data.scheduled_refresh import run_intraday_refresh

            report = await run_intraday_refresh()
            summary = {
                "mode": report.mode,
                "symbols_refreshed": report.symbols_refreshed,
                "api_calls": report.api_calls,
                "errors": report.errors,
                "elapsed_seconds": round(report.elapsed_seconds, 1),
            }
            if report.errors:
                return {
                    "data_refresh_summary": summary,
                    "errors": [f"data_refresh: {len(report.errors)} errors"],
                }
            return {"data_refresh_summary": summary}
        except Exception as exc:
            logger.error("data_refresh failed: %s", exc)
            return {
                "data_refresh_summary": {"error": str(exc)},
                "errors": [f"data_refresh: {exc}"],
            }

    return data_refresh


def make_safety_check(llm: BaseChatModel, config: AgentConfig):
    """Create the safety_check node (deterministic, no retry, no tools)."""

    async def safety_check(state: TradingState) -> dict[str, Any]:
        try:
            response = await llm.ainvoke([
                SystemMessage(content=(
                    f"You are a {config.role}.\nGoal: {config.goal}\n"
                    "Always respond with valid JSON."
                )),
                HumanMessage(content=(
                    f"Cycle {state['cycle_number']}: Check system status. "
                    "Is the system halted or healthy? "
                    'Return JSON: {"halted": true/false, "reason": "..."}'
                )),
            ])
            parsed = parse_json_response(response.content, {"halted": False})
            halted = parsed.get("halted", False)
            update: dict[str, Any] = {
                "decisions": [{"node": "safety_check", "halted": halted}],
            }
            if halted:
                update["errors"] = [f"System halted: {parsed.get('reason', 'unknown')}"]
            return update
        except Exception as exc:
            logger.error("safety_check failed: %s", exc)
            return {
                "errors": [f"safety_check: {exc}"],
                "decisions": [{"node": "safety_check", "halted": True, "error": str(exc)}],
            }

    return safety_check


def make_daily_plan(llm: BaseChatModel, config: AgentConfig, tools: list[BaseTool] | None = None):
    """Create the daily_plan node (agent with tool access)."""
    tools = tools or []

    async def daily_plan(state: TradingState) -> dict[str, Any]:
        try:
            # Include market intelligence if available
            market_intel_section = ""
            market_ctx = state.get("market_context", {})
            if market_ctx:
                market_intel_section = (
                    f"\n--- Market Intelligence Briefing ---\n"
                    f"{json.dumps(market_ctx, default=str)}\n"
                    f"--- End Briefing ---\n\n"
                )

            prompt = (
                f"Cycle {state['cycle_number']}, regime: {state.get('regime', 'unknown')}.\n"
                f"Portfolio: {json.dumps(state.get('portfolio_context', {}), default=str)}\n"
                f"{market_intel_section}"
                "First, use your tools to gather real data:\n"
                "1. Fetch the current portfolio state\n"
                "2. Get signal briefs for key symbols in the portfolio and watchlist\n"
                "3. Search the knowledge base for recent lessons\n\n"
                "Then generate a daily trading plan with:\n"
                "- Regime assessment based on actual data\n"
                "- Ranked entry candidates with real signal scores\n"
                "- Exit recommendations for existing positions\n"
                "- Key events calendar\n"
                "- Flag any symbols with earnings within 14 days\n\n"
                'Return final JSON: {"plan": "...", "priorities": [...], '
                '"entry_candidates": [...], "exit_recommendations": [...], '
                '"earnings_within_14d": ["SYMBOL", ...]}'
            )
            text = await run_agent(llm, tools, config, prompt)
            parsed = parse_json_response(text, {"plan": text})

            # Extract earnings symbols from the plan
            earnings_symbols = parsed.get("earnings_within_14d", [])
            if not isinstance(earnings_symbols, list):
                earnings_symbols = []

            update: dict[str, Any] = {
                "daily_plan": parsed.get("plan", text),
                "earnings_symbols": earnings_symbols,
                "decisions": [{"node": "daily_plan", "action": "generated"}],
            }
            return update
        except Exception as exc:
            logger.error("daily_plan failed: %s", exc)
            return {
                "daily_plan": f"Plan generation failed: {exc}",
                "errors": [f"daily_plan: {exc}"],
            }

    return daily_plan


def make_position_review(
    llm: BaseChatModel,
    config: AgentConfig,
    tools: list[BaseTool] | None = None,
    *,
    exit_llm: BaseChatModel | None = None,
    exit_config: AgentConfig | None = None,
    exit_tools: list[BaseTool] | None = None,
):
    """Create the position_review node.

    Runs two sequential agents:
    1. position_monitor — produces a structured PositionAssessment per symbol (no writes).
    2. exit_evaluator  — consumes assessments, applies exit rules, writes exit signals.

    If exit_llm/exit_config are not provided, falls back to the monitor's llm/config.
    """
    tools = tools or []
    _exit_llm = exit_llm or llm
    _exit_cfg = exit_config or config
    _exit_tools = exit_tools if exit_tools is not None else tools

    async def position_review(state: TradingState) -> dict[str, Any]:
        # --- Phase 1: position_monitor — assessment only ---
        assessments_raw: str = ""
        try:
            monitor_prompt = (
                "Assess all open positions and return a structured PositionAssessment per symbol.\n\n"
                "For each position:\n"
                "1. Fetch portfolio and signal brief\n"
                "2. Check trading calendar for event flags\n"
                "3. Run alpha decay analysis if held > 5 days\n"
                "4. Search knowledge base for context\n\n"
                "Return JSON: [{\"symbol\": \"...\", \"thesis_intact\": true|false, "
                "\"signal_fresh\": true|false, \"regime_match\": true|false, "
                "\"calendar_flags\": [...], \"days_held\": 0, \"alpha_decay_flag\": false}]\n\n"
                "Do NOT recommend HOLD/CLOSE verdicts — only produce the assessment fields above."
            )
            assessments_raw = await run_agent(llm, tools, config, monitor_prompt)
        except Exception as exc:
            logger.error("position_review/monitor failed: %s", exc)
            return {
                "position_reviews": [],
                "errors": [f"position_monitor: {exc}"],
            }

        # --- Phase 2: exit_evaluator — verdict only ---
        try:
            evaluator_prompt = (
                f"PositionAssessments from position_monitor:\n{assessments_raw}\n\n"
                "Apply exit rules to each assessment and produce a binding verdict.\n\n"
                "Decision rules:\n"
                "- Hard exits: options DTE <= 2, loss > 2x stop distance\n"
                "- CLOSE: regime flipped, target >= 75%, holding exceeded signal half-life, "
                "alpha_decay_flag=True AND ICIR < 0.4\n"
                "- TRIM: profitable > 1x ATR, event within 48h\n"
                "- TIGHTEN: regime weakening, profitable with imminent event risk\n"
                "- HOLD: thesis intact, regime unchanged, within normal drawdown\n\n"
                "For CLOSE/TRIM: call create_exit_signal. For TIGHTEN: call update_position_stops.\n"
                "Query knowledge base for lessons from similar past exits.\n\n"
                'Return JSON: [{"symbol": "...", "action": "HOLD|TRIM|TIGHTEN|CLOSE", '
                '"reason": "...", "urgency": "low|medium|high"}]'
            )
            text = await run_agent(_exit_llm, _exit_tools, _exit_cfg, evaluator_prompt)
            reviews = parse_json_response(text, [])
            if not isinstance(reviews, list):
                reviews = [reviews] if reviews else []
            return {
                "position_reviews": reviews,
                "decisions": [{"node": "position_review", "count": len(reviews)}],
            }
        except Exception as exc:
            logger.error("position_review/exit_evaluator failed: %s", exc)
            return {
                "position_reviews": [],
                "errors": [f"exit_evaluator: {exc}"],
            }

    return position_review


def make_execute_exits(llm: BaseChatModel, config: AgentConfig, tools: list[BaseTool] | None = None):
    """Create the execute_exits node (tool, deterministic)."""
    tools = tools or []

    async def execute_exits(state: TradingState) -> dict[str, Any]:
        try:
            reviews = state.get("position_reviews", [])
            exits_needed = [r for r in reviews if r.get("action") in ("TRIM", "CLOSE")]
            if not exits_needed:
                return {
                    "exit_orders": [],
                    "decisions": [{"node": "execute_exits", "action": "no_exits"}],
                }
            prompt = (
                f"Execute exits for these positions:\n{json.dumps(exits_needed, default=str)}\n\n"
                "Use the execute_order tool for each exit. "
                'Return JSON: [{"symbol": "...", "order_id": "...", "action": "..."}]'
            )
            text = await run_agent(llm, tools, config, prompt)
            orders = parse_json_response(text, [])
            if not isinstance(orders, list):
                orders = [orders] if orders else []
            return {
                "exit_orders": orders,
                "decisions": [{"node": "execute_exits", "count": len(orders)}],
            }
        except Exception as exc:
            logger.error("execute_exits failed: %s", exc)
            return {
                "exit_orders": [],
                "errors": [f"execute_exits: {exc}"],
            }

    return execute_exits


def make_entry_scan(llm: BaseChatModel, config: AgentConfig, tools: list[BaseTool] | None = None):
    """Create the entry_scan node (agent with tool access)."""
    tools = tools or []

    async def entry_scan(state: TradingState) -> dict[str, Any]:
        try:
            prompt = (
                f"Regime: {state.get('regime', 'unknown')}\n"
                f"Daily plan: {state.get('daily_plan', 'none')}\n\n"
                "Use your tools to find entry candidates:\n"
                "1. Get multi-symbol signal briefs for potential candidates\n"
                "2. Fetch fundamentals for promising symbols\n"
                "3. Search knowledge base for lessons on candidate strategies\n\n"
                "Run a structured bull/bear debate for each candidate.\n"
                'Return JSON: [{"symbol": "...", "strategy": "...", "signal_strength": ..., '
                '"verdict": "ENTER|SKIP", "reasoning": "..."}]'
            )
            text = await run_agent(llm, tools, config, prompt)
            candidates = parse_json_response(text, [])
            if not isinstance(candidates, list):
                candidates = [candidates] if candidates else []
            return {
                "entry_candidates": candidates,
                "decisions": [{"node": "entry_scan", "count": len(candidates)}],
            }
        except Exception as exc:
            logger.error("entry_scan failed: %s", exc)
            return {
                "entry_candidates": [],
                "errors": [f"entry_scan: {exc}"],
            }

    return entry_scan


def merge_parallel(state: TradingState) -> dict[str, Any]:
    """No-op join node. Convergence point for parallel branches."""
    return {}


def merge_pre_execution(state: TradingState) -> dict[str, Any]:
    """No-op join node. Convergence for parallel portfolio_review + analyze_options."""
    return {}


_MIN_ANNUALIZED_VOL = 0.05  # 5% floor — prevents divide-by-zero in kelly_sizing


def make_risk_sizing():
    """Create the risk_sizing node (deterministic, no LLM).

    Computes IC-adjusted Kelly alpha signals for each entry candidate.
    SafetyGate validation runs downstream in portfolio_construction after
    the optimizer produces final weights.
    """
    async def risk_sizing(state: TradingState) -> dict[str, Any]:
        candidates = state.get("entry_candidates", [])
        if not candidates:
            return {
                "alpha_signals": [],
                "alpha_signal_candidates": [],
                "decisions": [{"node": "risk_sizing", "action": "no_candidates"}],
            }

        try:
            # --- Step 1: Query signal_ic for IC values per strategy ---
            strategy_ids = list({c.get("strategy_id", "") for c in candidates if c.get("strategy_id")})
            signal_ic_lookup: dict[str, float | None] = {}

            try:
                with db_conn() as conn:
                    for sid in strategy_ids:
                        row = conn.execute(
                            """
                            SELECT mean_rank_ic_21d FROM signal_ic
                            WHERE strategy_id = %s AND horizon_days = 21
                            ORDER BY date DESC LIMIT 1
                            """,
                            (sid,),
                        ).fetchone()
                        signal_ic_lookup[sid] = float(row[0]) if row and row[0] is not None else None
            except Exception as exc:
                logger.warning("signal_ic query failed (IC prior applies to all): %s", exc)

            # --- Step 2: Fetch 63-day returns for EWMA vol per symbol ---
            symbols = list({c.get("symbol", "") for c in candidates if c.get("symbol")})
            volatility_lookup: dict[str, float] = {}
            ewma_vols: list[float] = []  # For portfolio-level vol aggregation
            cold_start_symbols: set[str] = set()

            try:
                with db_conn() as conn:
                    for sym in symbols:
                        rows = conn.execute(
                            """
                            SELECT close FROM ohlcv
                            WHERE symbol = %s AND timeframe = 'daily'
                            ORDER BY timestamp DESC LIMIT 64
                            """,
                            (sym,),
                        ).fetchall()
                        closes = [float(r[0]) for r in rows if r[0] is not None]
                        if len(closes) < 5:
                            logger.warning("Insufficient OHLCV for vol on %s; using vol floor", sym)
                            volatility_lookup[sym] = _MIN_ANNUALIZED_VOL
                            continue

                        # Compute returns (oldest-first) and EWMA vol
                        prices = np.array(closes[::-1])
                        log_rets = pd.Series(np.diff(np.log(prices)))
                        ewma_vol = ewma_volatility(log_rets)

                        if ewma_vol is None:
                            cold_start_symbols.add(sym)
                            logger.info("Cold-start vol for %s (< 21 days); deferring position", sym)
                            continue

                        volatility_lookup[sym] = max(ewma_vol, _MIN_ANNUALIZED_VOL)
                        ewma_vols.append(ewma_vol)
            except Exception as exc:
                logger.warning("OHLCV vol query failed (vol floor applies): %s", exc)

            # --- Step 3: Dynamic Kelly via regime + vol state ---
            # Query current regime
            regime = "unknown"
            regime_confidence = 0.0
            try:
                with db_conn() as conn:
                    row = conn.execute(
                        "SELECT regime, confidence FROM regime_state ORDER BY updated_at DESC LIMIT 1"
                    ).fetchone()
                    if row:
                        regime = row[0] or "unknown"
                        regime_confidence = float(row[1]) if row[1] is not None else 0.0
            except Exception as exc:
                logger.warning("regime_state query failed; using unknown/0.0: %s", exc)

            # Portfolio-level EWMA vol (equal-weighted mean across symbols)
            if ewma_vols:
                portfolio_ewma_vol = float(np.mean(ewma_vols))
            else:
                portfolio_ewma_vol = TARGET_VOL  # No vol data → assume target

            # Vol state with hysteresis (use prior state from graph state)
            prior_vol_state = state.get("vol_state", "normal")
            # Use portfolio vol as proxy; vol_63d_mean approximated as TARGET_VOL
            # (a proper 63d rolling mean would require persisted history)
            vol_state = compute_vol_state(portfolio_ewma_vol, TARGET_VOL, prior_vol_state)

            # Regime-conditional Kelly multiplier
            regime_multiplier = regime_kelly_fraction(regime, vol_state, regime_confidence)

            # Vol-targeting scalar
            vol_scalar = min(TARGET_VOL / max(portfolio_ewma_vol, 0.10), VOL_SCALAR_CAP)

            # Raw kelly with hard ceiling/floor
            raw_kelly = regime_multiplier * vol_scalar
            kelly_fraction = max(min(raw_kelly, KELLY_HARD_CEILING), KELLY_HARD_FLOOR)

            logger.info(
                "Dynamic Kelly: regime=%s conf=%.2f vol_state=%s ewma_vol=%.3f "
                "regime_mult=%.3f vol_scalar=%.3f raw=%.4f final=%.4f",
                regime, regime_confidence, vol_state, portfolio_ewma_vol,
                regime_multiplier, vol_scalar, raw_kelly, kelly_fraction,
            )

            # --- Step 4: Filter cold-start symbols, compute alpha signals ---
            normalized_candidates = []
            filtered_candidates = []
            for c in candidates:
                sym = c.get("symbol", "")
                if sym in cold_start_symbols:
                    logger.info("Skipping %s: cold-start vol (insufficient history)", sym)
                    continue
                signal_value = c.get("signal_value", c.get("conviction", 0.5))
                normalized_candidates.append({
                    "symbol": sym,
                    "strategy_id": c.get("strategy_id", ""),
                    "signal_value": signal_value,
                })
                filtered_candidates.append(c)

            if not normalized_candidates:
                return {
                    "alpha_signals": [],
                    "alpha_signal_candidates": [],
                    "vol_state": vol_state,
                    "decisions": [{"node": "risk_sizing", "action": "all_cold_start"}],
                }

            alpha_signals = compute_alpha_signals(
                candidates=normalized_candidates,
                signal_ic_lookup=signal_ic_lookup,
                volatility_lookup=volatility_lookup,
                kelly_fraction=kelly_fraction,
                ic_prior=0.01,
            )

            # --- Step 5: Alt-data modifier (section-13) ---
            # Adjust alpha signals with alt-data (EDGAR, earnings).
            # Failure is non-fatal — falls back to price-only signals.
            try:
                with db_conn() as conn:
                    for i, nc in enumerate(normalized_candidates):
                        sym = nc["symbol"]
                        alt_mod = get_alt_data_modifier(sym, conn)
                        if alt_mod != 0.0:
                            alpha_signals[i] += ALT_DATA_WEIGHT * alt_mod
            except Exception as exc:
                logger.warning("Alt-data modifier failed (using price-only signals): %s", exc)

            return {
                "alpha_signals": alpha_signals.tolist(),
                "alpha_signal_candidates": filtered_candidates,
                "vol_state": vol_state,
                "decisions": [{
                    "node": "risk_sizing",
                    "n_candidates": len(filtered_candidates),
                    "kelly_fraction": round(kelly_fraction, 4),
                    "regime": regime,
                    "vol_state": vol_state,
                }],
            }

        except Exception as exc:
            logger.error("risk_sizing failed: %s", exc)
            return {
                "alpha_signals": [],
                "alpha_signal_candidates": [],
                "errors": [f"risk_sizing: {exc}"],
            }

    return risk_sizing


def make_portfolio_construction():
    """Create the portfolio_construction node (deterministic, no LLM).

    Reads alpha_signals and alpha_signal_candidates set by risk_sizing.
    Runs the MVO optimizer with Ledoit-Wolf shrunk covariance.
    Calls SafetyGate on the final optimizer weights (post-optimization).
    Stores covariance in state["last_covariance"] for stale-data fallback.
    """
    async def portfolio_construction(state: TradingState) -> dict[str, Any]:
        try:
            alpha_signals_list = state.get("alpha_signals", [])
            candidates = state.get("alpha_signal_candidates", [])

            if not candidates:
                return {
                    "portfolio_target_weights": {},
                    "risk_verdicts": [],
                    "last_covariance": state.get("last_covariance", []),
                    "decisions": [{"node": "portfolio_construction", "action": "no_candidates"}],
                }

            portfolio_ctx = state.get("portfolio_context", {})
            positions = portfolio_ctx.get("positions", [])

            # Build symbol list from candidates + existing positions
            candidate_symbols = [c.get("symbol", "") for c in candidates]
            position_symbols = [p.get("symbol", "") for p in positions]
            all_symbols = list(dict.fromkeys(candidate_symbols + position_symbols))

            if not all_symbols:
                return {
                    "portfolio_target_weights": {},
                    "risk_verdicts": [],
                    "last_covariance": state.get("last_covariance", []),
                    "decisions": [{"node": "portfolio_construction", "action": "no_symbols"}],
                }

            n = len(all_symbols)

            # Build current weights (from positions)
            total_equity = portfolio_ctx.get("total_equity", 100_000)
            current_weights = np.zeros(n)
            for p in positions:
                sym = p.get("symbol", "")
                if sym in all_symbols:
                    idx = all_symbols.index(sym)
                    mv = abs(p.get("quantity", 0) * p.get("current_price", 0))
                    current_weights[idx] = mv / total_equity if total_equity > 0 else 0

            # Build alpha signals array from state (set by risk_sizing via kelly_sizing)
            alpha_signals_array = np.array(alpha_signals_list) if alpha_signals_list else np.ones(n) * 0.5
            # Pad or trim to match n (position symbols may extend beyond candidates)
            if len(alpha_signals_array) < n:
                alpha_signals_array = np.concatenate([
                    alpha_signals_array,
                    np.zeros(n - len(alpha_signals_array)),
                ])

            # ------------------------------------------------------------------
            # Compute Ledoit-Wolf shrunk covariance from 120-day OHLCV history
            # ------------------------------------------------------------------
            cov_matrix_arr = _build_covariance(
                all_symbols=all_symbols,
                n=n,
                last_covariance=state.get("last_covariance", []),
            )

            sector_map = {sym: "default" for sym in all_symbols}
            strategy_map = {sym: "default" for sym in all_symbols}

            target_weights, meta = optimize_portfolio(
                cov_matrix_arr, alpha_signals_array, current_weights,
                sector_map, strategy_map,
            )

            # Build initial weight dict
            weight_dict = {sym: float(target_weights[i]) for i, sym in enumerate(all_symbols)}

            # ------------------------------------------------------------------
            # SafetyGate validation on final optimizer weights (post-optimization)
            # ------------------------------------------------------------------
            gate = SafetyGate()
            verdicts = []
            gate_errors = []
            for symbol, weight in list(weight_dict.items()):
                decision = RiskDecision(
                    symbol=symbol,
                    recommended_size_pct=weight * 100,
                    reasoning="IC-adjusted Kelly + MVO optimizer output",
                    confidence=1.0,
                )
                verdict = gate.validate(decision, portfolio_ctx)
                verdict_dict = {"symbol": symbol, "approved": verdict.approved, "size_pct": weight * 100}
                if not verdict.approved:
                    verdict_dict["violations"] = verdict.violations
                    gate_errors.append(f"SafetyGate rejected {symbol}: {', '.join(verdict.violations)}")
                    weight_dict.pop(symbol)
                verdicts.append(verdict_dict)

            update: dict[str, Any] = {
                "portfolio_target_weights": weight_dict,
                "risk_verdicts": verdicts,
                "last_covariance": cov_matrix_arr.tolist(),
                "decisions": [{
                    "node": "portfolio_construction",
                    "action": "optimized",
                    "n_assets": n,
                    "turnover": meta.get("turnover", 0),
                    "feasible": meta.get("feasible", True),
                }],
            }
            if gate_errors:
                update["errors"] = gate_errors
            return update

        except Exception as exc:
            logger.error("portfolio_construction failed: %s", exc)
            return {
                "portfolio_target_weights": {},
                "risk_verdicts": [],
                "errors": [f"portfolio_construction: {exc}"],
            }

    return portfolio_construction


def _build_covariance(
    all_symbols: list[str],
    n: int,
    last_covariance: list,
) -> np.ndarray:
    """
    Fetch 120-day OHLCV, compute Ledoit-Wolf shrunk annualized covariance.

    Falls back to prior covariance (from state["last_covariance"]) if:
    - Data is stale (most recent row > 2 trading days ago)
    - All symbols have < 60 days of clean history

    Falls back to np.eye(n) * 0.02 as last resort if no prior covariance.
    """
    import datetime

    symbol_placeholder = ",".join(f"'{s}'" for s in all_symbols)
    try:
        with db_conn() as conn:
            rows = conn.execute(
                f"""
                SELECT symbol, timestamp::date AS dt, close
                FROM ohlcv
                WHERE symbol IN ({symbol_placeholder})
                  AND timeframe = 'daily'
                ORDER BY symbol, dt
                """
            ).fetchall()
    except Exception as exc:
        logger.warning("[portfolio_construction] OHLCV fetch failed: %s — using fallback", exc)
        return _covariance_fallback(n, last_covariance)

    if not rows:
        logger.warning("[portfolio_construction] No OHLCV rows found — using fallback covariance")
        return _covariance_fallback(n, last_covariance)

    # Build prices DataFrame
    records = [(r[0], r[1], r[2]) for r in rows]
    prices_df = (
        pd.DataFrame(records, columns=["symbol", "dt", "close"])
        .pivot(index="dt", columns="symbol", values="close")
        .sort_index()
    )

    # Freshness check: most recent date must be within 2 trading days
    most_recent = prices_df.index[-1]
    today = datetime.date.today()
    delta_days = (today - most_recent).days
    if delta_days > 4:  # ~2 trading days buffer for weekends
        logger.warning(
            "[portfolio_construction] Stale OHLCV data (most recent: %s) — using prior covariance",
            most_recent,
        )
        return _covariance_fallback(n, last_covariance)

    # Limit to last 120 rows, compute log returns
    prices_df = prices_df.iloc[-121:]
    returns_df = np.log(prices_df / prices_df.shift(1)).iloc[1:]

    # Forward-fill small gaps, drop rows with > 20% missing
    returns_df = returns_df.ffill(limit=2)
    threshold = int(0.2 * len(returns_df.columns))
    returns_df = returns_df.dropna(thresh=len(returns_df.columns) - threshold)

    # Identify thin symbols (< 60 days of clean data)
    valid_counts = returns_df.notna().sum()
    thin_symbols = set(valid_counts[valid_counts < 60].index.tolist())
    thick_symbols = [s for s in all_symbols if s in returns_df.columns and s not in thin_symbols]

    if not thick_symbols:
        logger.warning("[portfolio_construction] All symbols have < 60 days of history — using fallback covariance")
        return _covariance_fallback(n, last_covariance)

    # Compute LW covariance on thick symbols
    thick_returns = returns_df[thick_symbols].dropna()
    cov_df = covariance_matrix(thick_returns, annualise=True, shrinkage=True)

    # Build n×n matrix; thin symbols use median diagonal variance
    median_var = float(np.median(np.diag(cov_df.values)))
    cov_full = np.eye(n) * median_var

    sym_to_idx = {s: i for i, s in enumerate(all_symbols)}
    thick_indices = [sym_to_idx[s] for s in thick_symbols if s in sym_to_idx]
    thick_cov_vals = cov_df.values

    for row_local, row_global in enumerate(thick_indices):
        for col_local, col_global in enumerate(thick_indices):
            cov_full[row_global, col_global] = thick_cov_vals[row_local, col_local]

    # Positive definite guard
    eigvals = np.linalg.eigvalsh(cov_full)
    if eigvals.min() <= 0:
        cov_full += 1e-6 * np.eye(n)
        logger.warning("[portfolio_construction] covariance not positive definite, applied 1e-6 perturbation")
        eigvals = np.linalg.eigvalsh(cov_full)

    # Condition number logging
    cond = eigvals.max() / eigvals.min()
    logger.debug("[portfolio_construction] covariance condition number: %.1f", cond)
    if cond > 500:
        logger.warning(
            "[portfolio_construction] high condition number %.1f — covariance may be near-singular", cond
        )

    return cov_full


def _covariance_fallback(n: int, last_covariance: list) -> np.ndarray:
    """Use prior covariance from state, or identity matrix as last resort."""
    if last_covariance:
        try:
            arr = np.array(last_covariance)
            if arr.shape == (n, n):
                return arr
        except Exception:
            pass
    # Last resort: identity (same as the original placeholder)
    return np.eye(n) * 0.02


def make_portfolio_review(llm: BaseChatModel, config: AgentConfig, tools: list[BaseTool] | None = None):
    """Create the portfolio_review node (agent: fund_manager with tool access)."""
    tools = tools or []

    async def portfolio_review(state: TradingState) -> dict[str, Any]:
        try:
            verdicts = state.get("risk_verdicts", [])
            approved = [v for v in verdicts if v.get("approved", False)]
            prompt = (
                f"Review these risk-approved candidates:\n{json.dumps(approved, default=str)}\n\n"
                "Use your tools to:\n"
                "1. Fetch the full portfolio for correlation analysis\n"
                "2. Compute risk metrics for the proposed additions\n"
                "3. Search knowledge base for past experiences with these symbols\n\n"
                "Assess portfolio-level risk: correlation, allocation, diversity.\n"
                'Return JSON: [{"symbol": "...", "decision": "APPROVED|REJECTED", "reason": "..."}]'
            )
            text = await run_agent(llm, tools, config, prompt)
            decisions = parse_json_response(text, [])
            if not isinstance(decisions, list):
                decisions = [decisions] if decisions else []
            return {
                "fund_manager_decisions": decisions,
                "decisions": [{"node": "portfolio_review", "count": len(decisions)}],
            }
        except Exception as exc:
            logger.error("portfolio_review failed: %s", exc)
            return {
                "fund_manager_decisions": [],
                "errors": [f"portfolio_review: {exc}"],
            }

    return portfolio_review


def make_options_analysis(llm: BaseChatModel, config: AgentConfig, tools: list[BaseTool] | None = None):
    """Create the options_analysis node (agent with tool access)."""
    tools = tools or []

    async def options_analysis(state: TradingState) -> dict[str, Any]:
        try:
            verdicts = state.get("risk_verdicts", [])
            approved = [v for v in verdicts if v.get("approved", False)]
            if not approved:
                return {
                    "options_analysis": [],
                    "decisions": [{"node": "options_analysis", "action": "no_candidates"}],
                }
            # Include earnings analysis context if available
            earnings_ctx = state.get("earnings_analysis", {})
            earnings_section = ""
            if earnings_ctx:
                earnings_section = (
                    f"\n--- Earnings Context ---\n"
                    f"{json.dumps(earnings_ctx, default=str)}\n"
                    "Adjust structure selection: prefer iron condors when IV premium ratio > 1.5, "
                    "directional options when IV premium < 1.2 with conviction.\n"
                    f"--- End Earnings ---\n\n"
                )

            prompt = (
                f"Analyze options structures for:\n{json.dumps(approved, default=str)}\n"
                f"{earnings_section}"
                "Use your tools to:\n"
                "1. Fetch options chains for each symbol\n"
                "2. Compute Greeks for candidate structures\n"
                "3. Search knowledge base for options lessons\n\n"
                "Select optimal options structures based on IV rank, regime, direction.\n"
                'Return JSON: [{"symbol": "...", "structure": "...", "legs": [...], '
                '"max_profit": ..., "max_loss": ..., "reasoning": "..."}]'
            )
            text = await run_agent(llm, tools, config, prompt)
            analysis = parse_json_response(text, [])
            if not isinstance(analysis, list):
                analysis = [analysis] if analysis else []
            return {
                "options_analysis": analysis,
                "decisions": [{"node": "options_analysis", "count": len(analysis)}],
            }
        except Exception as exc:
            logger.error("options_analysis failed: %s", exc)
            return {
                "options_analysis": [],
                "errors": [f"options_analysis: {exc}"],
            }

    return options_analysis


def make_execute_entries(llm: BaseChatModel, config: AgentConfig, tools: list[BaseTool] | None = None):
    """Create the execute_entries node (tool, deterministic)."""
    tools = tools or []

    async def execute_entries(state: TradingState) -> dict[str, Any]:
        try:
            fm_decisions = state.get("fund_manager_decisions", [])
            approved = [d for d in fm_decisions if d.get("decision") == "APPROVED"]
            if not approved:
                return {
                    "entry_orders": [],
                    "decisions": [{"node": "execute_entries", "action": "no_approved"}],
                }
            prompt = (
                f"Execute entries for approved candidates:\n{json.dumps(approved, default=str)}\n"
                f"Options analysis:\n{json.dumps(state.get('options_analysis', []), default=str)}\n\n"
                "Use the execute_order tool for each entry.\n"
                'Return JSON: [{"symbol": "...", "order_id": "...", "type": "..."}]'
            )
            text = await run_agent(llm, tools, config, prompt)
            orders = parse_json_response(text, [])
            if not isinstance(orders, list):
                orders = [orders] if orders else []
            return {
                "entry_orders": orders,
                "decisions": [{"node": "execute_entries", "count": len(orders)}],
            }
        except Exception as exc:
            logger.error("execute_entries failed: %s", exc)
            return {
                "entry_orders": [],
                "errors": [f"execute_entries: {exc}"],
            }

    return execute_entries


def _fetch_attribution_summary(symbol: str, strategy_id: str) -> dict | None:
    """
    Fetches aggregated P&L attribution for a closed position.

    Returns a dict with keys: alpha_pnl_sum, market_pnl_sum, sector_pnl_sum,
    residual_pnl_sum, total_pnl_sum, alpha_fraction.
    Returns None if no rows exist or total_pnl_sum is zero.
    """
    try:
        with db_conn() as conn:
            row = conn.execute(
                """
                SELECT
                    SUM(alpha_pnl)    AS alpha_pnl_sum,
                    SUM(market_pnl)   AS market_pnl_sum,
                    SUM(sector_pnl)   AS sector_pnl_sum,
                    SUM(residual_pnl) AS residual_pnl_sum,
                    SUM(total_pnl)    AS total_pnl_sum
                FROM pnl_attribution
                WHERE symbol = %s AND strategy_id = %s
                """,
                (symbol, strategy_id),
            ).fetchone()
    except Exception as exc:
        logger.warning("_fetch_attribution_summary failed for %s/%s: %s", symbol, strategy_id, exc)
        return None

    if row is None:
        return None

    alpha_pnl_sum, market_pnl_sum, sector_pnl_sum, residual_pnl_sum, total_pnl_sum = row

    if total_pnl_sum is None or float(total_pnl_sum) == 0.0:
        return None

    total = float(total_pnl_sum)
    alpha_fraction = round(float(alpha_pnl_sum or 0.0) / total, 4)

    return {
        "alpha_pnl_sum": float(alpha_pnl_sum or 0.0),
        "market_pnl_sum": float(market_pnl_sum or 0.0),
        "sector_pnl_sum": float(sector_pnl_sum or 0.0),
        "residual_pnl_sum": float(residual_pnl_sum or 0.0),
        "total_pnl_sum": total,
        "alpha_fraction": alpha_fraction,
    }


def _extract_entry_thesis(decisions: list[dict], symbol: str) -> str:
    """Find the entry thesis for a symbol from the accumulated decisions list."""
    for d in decisions:
        if d.get("symbol") == symbol and d.get("node") in ("entry_scan", "trade_debater"):
            thesis = d.get("thesis") or d.get("action") or ""
            if thesis:
                return thesis
    return "entry thesis unavailable"


def _persist_quality_scores(
    scores: list[dict], cycle_number: int, model_used: str
) -> None:
    """Write trade quality scores to the database."""
    if not scores:
        return
    try:
        with db_conn() as conn:
            for score in scores:
                conn.execute(
                    """
                    INSERT INTO trade_quality_scores
                        (trade_id, cycle_number, execution_quality, thesis_accuracy,
                         risk_management, timing_quality, sizing_quality,
                         overall_score, justification, model_used)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    [
                        score.get("trade_id"),
                        cycle_number,
                        score["execution_quality"],
                        score["thesis_accuracy"],
                        score["risk_management"],
                        score["timing_quality"],
                        score["sizing_quality"],
                        score["overall_score"],
                        score["justification"],
                        model_used,
                    ],
                )
    except Exception as exc:
        logger.error("Failed to persist quality scores: %s", exc)


def make_reflection(llm: BaseChatModel, config: AgentConfig, tools: list[BaseTool] | None = None):
    """Create the reflection node (agent with tool access)."""
    tools = tools or []

    async def reflection(state: TradingState) -> dict[str, Any]:
        try:
            prompt = (
                f"Reflect on this trading cycle:\n"
                f"Exits: {json.dumps(state.get('exit_orders', []), default=str)}\n"
                f"Entries: {json.dumps(state.get('entry_orders', []), default=str)}\n\n"
                "Use your tools to:\n"
                "1. Fetch portfolio to see current state after this cycle\n"
                "2. Search knowledge base for similar past outcomes\n\n"
                "Analyze what happened and extract lessons.\n"
                'Return JSON: {"reflection": "...", "lessons": [...]}'
            )
            text = await run_agent(llm, tools, config, prompt)
            parsed = parse_json_response(text, {"reflection": text})

            # --- Trade quality scoring for closed trades ---
            trade_quality_scores: list[dict] = []
            exit_orders = state.get("exit_orders", [])
            if exit_orders:
                try:
                    evaluator = create_trade_evaluator()
                    decisions = state.get("decisions", [])
                    for order in exit_orders:
                        symbol = order.get("symbol", "")
                        thesis = _extract_entry_thesis(decisions, symbol)
                        score = evaluator(
                            inputs={
                                "entry_thesis": thesis,
                                "symbol": symbol,
                                "signals": order.get("signals_at_entry", {}),
                            },
                            outputs={
                                "realized_pnl": order.get("realized_pnl", 0.0),
                                "exit_reason": order.get("exit_reason", ""),
                                "holding_days": order.get("holding_days", 0),
                            },
                        )
                        score["symbol"] = symbol
                        score["trade_id"] = order.get("trade_id")
                        trade_quality_scores.append(score)

                    _persist_quality_scores(
                        trade_quality_scores,
                        state.get("cycle_number", 0),
                        model_used="anthropic/claude-sonnet-4-20250514",
                    )
                except Exception as exc:
                    logger.error("trade quality evaluation failed: %s", exc)
                    trade_quality_scores = []

            # --- Attribution context for trade_reflector ---
            attribution_contexts: dict = {}
            for order in exit_orders:
                sym = order.get("symbol", "")
                strat = order.get("strategy_id", "")
                if sym and strat:
                    try:
                        summary = _fetch_attribution_summary(sym, strat)
                        if summary:
                            attribution_contexts[sym] = summary
                    except Exception as exc:
                        logger.warning("Failed to fetch attribution for %s: %s", sym, exc)

            return {
                "reflection": parsed.get("reflection", text),
                "trade_quality_scores": trade_quality_scores,
                "attribution_contexts": attribution_contexts,
                "decisions": [{"node": "reflection", "action": "completed"}],
            }
        except Exception as exc:
            logger.error("reflection failed: %s", exc)
            return {
                "reflection": f"Reflection failed: {exc}",
                "trade_quality_scores": [],
                "errors": [f"reflection: {exc}"],
            }

    return reflection
