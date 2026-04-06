"""Node functions for the Research Graph.

8 nodes + 1 conditional router. Agent nodes use LLM reasoning with
tool access to gather real data before making decisions.
"""

import logging
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from langgraph.types import Send

from quantstack.graphs.agent_executor import parse_json_response, run_agent
from quantstack.graphs.config import AgentConfig
from quantstack.graphs.state import ResearchState, SymbolValidationState

logger = logging.getLogger(__name__)


def make_context_load(llm: BaseChatModel, config: AgentConfig, tools: list[BaseTool] | None = None):
    """Create the context_load node (data gathering with tools)."""
    tools = tools or []

    async def context_load(state: ResearchState) -> dict[str, Any]:
        # Poll for IDEAS_DISCOVERED events from community intel scans
        community_ideas_section = ""
        queued_task_ids: list[str] = []
        try:
            from quantstack.coordination.event_bus import EventBus, EventType
            from quantstack.db import db_conn

            with db_conn() as conn:
                bus = EventBus(conn)
                events = bus.poll(
                    "research_graph",
                    event_types=[EventType.IDEAS_DISCOVERED],
                )
                if events:
                    all_ideas = []
                    for evt in events:
                        ideas = evt.payload.get("ideas", [])
                        all_ideas.extend(ideas)
                    if all_ideas:
                        import json as _json

                        community_ideas_section = (
                            f"\n--- Community Intelligence ({len(all_ideas)} ideas) ---\n"
                            f"{_json.dumps(all_ideas[:10], default=str)}\n"
                            "Consider these external ideas when selecting research priorities.\n"
                            "--- End Community Intel ---\n"
                        )
        except Exception as poll_exc:
            logger.debug("Failed to poll IDEAS_DISCOVERED events: %s", poll_exc)

        # Pull pending strategy_hypothesis items from research_queue
        queued_ideas_section = ""
        try:
            from quantstack.db import db_conn
            import json as _json

            with db_conn() as conn:
                rows = conn.execute(
                    "SELECT DISTINCT ON (topic) task_id, context_json "
                    "FROM research_queue "
                    "WHERE status = 'pending' AND task_type = 'strategy_hypothesis' "
                    "ORDER BY topic, priority DESC, created_at ASC "
                    "LIMIT 3"
                ).fetchall()

                if rows:
                    ideas = []
                    for row in rows:
                        task_id = row[0]
                        ctx = row[1] if isinstance(row[1], dict) else _json.loads(row[1] or "{}")
                        queued_task_ids.append(task_id)
                        ideas.append({
                            "title": ctx.get("title", "unknown"),
                            "summary": ctx.get("summary", ""),
                            "source_url": ctx.get("source_url", ""),
                            "implementation_notes": ctx.get("implementation_notes", ""),
                        })

                    # Mark them as running so other cycles don't double-pick
                    for task_id in queued_task_ids:
                        conn.execute(
                            "UPDATE research_queue SET status = 'running', started_at = NOW() "
                            "WHERE task_id = %s AND status = 'pending'",
                            [task_id],
                        )

                    queued_ideas_section = (
                        f"\n--- Research Queue ({len(ideas)} hypotheses to investigate) ---\n"
                        f"{_json.dumps(ideas, indent=2)}\n"
                        "PRIORITY: Investigate these queued hypotheses. They come from community "
                        "intelligence scans and have been waiting for research.\n"
                        "--- End Research Queue ---\n"
                    )
                    logger.info(
                        "context_load: claimed %d research_queue tasks: %s",
                        len(queued_task_ids), queued_task_ids,
                    )
        except Exception as rq_exc:
            logger.debug("Failed to pull research_queue items: %s", rq_exc)

        # Detect current market regime (deterministic, no LLM)
        detected_regime = "unknown"
        regime_detail = {}
        try:
            from quantstack.agents.regime_detector import RegimeDetectorAgent
            detector = RegimeDetectorAgent(symbols=["SPY"])
            result = detector.detect_regime("SPY")
            if result.get("success"):
                detected_regime = result["trend_regime"]
                regime_detail = {
                    "trend": result["trend_regime"],
                    "volatility": result.get("volatility_regime", "unknown"),
                    "adx": result.get("adx", 0),
                    "confidence": result.get("confidence", 0),
                }
                logger.info(
                    "context_load: detected regime=%s vol=%s adx=%.1f",
                    detected_regime, regime_detail["volatility"], regime_detail["adx"],
                )

                # Persist to regime_states table
                from quantstack.db import db_conn
                with db_conn() as conn:
                    conn.execute(
                        "INSERT INTO regime_states "
                        "(symbol, timeframe, trend_regime, volatility_regime, adx, "
                        " atr_percentile, confidence, source_agent) "
                        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                        [
                            "SPY", "1D", result["trend_regime"],
                            result.get("volatility_regime"), result.get("adx"),
                            result.get("atr_percentile"), result.get("confidence"),
                            "research_graph",
                        ],
                    )
        except Exception as regime_exc:
            logger.debug("Regime detection failed (non-fatal): %s", regime_exc)

        try:
            import asyncio
            import json as _json

            regime_section = (
                f"DETECTED REGIME: {detected_regime}"
                + (f" (vol={regime_detail.get('volatility','?')}, adx={regime_detail.get('adx',0):.0f})" if regime_detail else "")
            )

            # ------------------------------------------------------------------
            # Pre-fetch context data in parallel (CONTEXT_LOAD_MODE=parallel).
            # Falls back to tool-based agent if disabled.
            # ------------------------------------------------------------------
            import os
            parallel_mode = os.environ.get("CONTEXT_LOAD_MODE", "parallel") == "parallel"

            prefetched_context = ""
            if parallel_mode:
                try:
                    from quantstack.tools.langchain.learning_tools import search_knowledge_base
                    from quantstack.tools.langchain.strategy_tools import fetch_strategy_registry
                    from quantstack.tools.langchain.signal_tools import signal_brief

                    # Pick candidate symbols from focus list (active research/trading subset)
                    from quantstack.config.focus import get_focus_symbols
                    import random
                    _focus = list(get_focus_symbols())
                    candidate_symbols = random.sample(_focus, min(5, len(_focus))) if _focus else []

                    # Fire all fetches concurrently
                    knowledge_coro = search_knowledge_base.ainvoke(
                        {"query": "recent findings negative results failed hypotheses", "top_k": 5}
                    )
                    registry_coro = fetch_strategy_registry.ainvoke({"status": None})

                    brief_coros = [
                        signal_brief.ainvoke({"symbol": sym})
                        for sym in candidate_symbols[:3]
                    ]

                    results = await asyncio.gather(
                        knowledge_coro, registry_coro, *brief_coros,
                        return_exceptions=True,
                    )

                    knowledge_text = results[0] if not isinstance(results[0], Exception) else "unavailable"
                    registry_text = results[1] if not isinstance(results[1], Exception) else "unavailable"
                    brief_texts = []
                    for i, sym in enumerate(candidate_symbols[:3]):
                        r = results[2 + i]
                        if not isinstance(r, Exception):
                            # Truncate to keep prompt manageable
                            brief_texts.append(f"  {sym}: {str(r)[:800]}")
                        else:
                            brief_texts.append(f"  {sym}: unavailable")

                    prefetched_context = (
                        f"\n--- Knowledge Base (recent findings & negative results) ---\n"
                        f"{str(knowledge_text)[:2000]}\n"
                        f"\n--- Strategy Registry ---\n"
                        f"{str(registry_text)[:2000]}\n"
                        f"\n--- Signal Briefs (candidate symbols) ---\n"
                        + "\n".join(brief_texts) + "\n"
                    )
                    logger.info(
                        "context_load: pre-fetched context for %d symbols in parallel",
                        len(candidate_symbols[:3]),
                    )
                except Exception as pf_exc:
                    logger.warning("context_load: parallel pre-fetch failed, falling back to agent: %s", pf_exc)
                    prefetched_context = ""

            if prefetched_context:
                # Synthesis-only prompt — no tool calls needed
                prompt = (
                    f"Cycle {state['cycle_number']}, {regime_section}.\n"
                    f"{community_ideas_section}\n"
                    f"{queued_ideas_section}\n"
                    f"{prefetched_context}\n"
                    "All research context has been pre-fetched above.\n\n"
                    "IMPORTANT: Diversify research across individual stocks, not just ETFs/indices. "
                    "Explore momentum, quality, value, sector rotation, and trend-following — not just "
                    "mean-reversion. If regime is 'unknown', use price action and trend analysis instead.\n\n"
                    "Based on the data above, synthesize:\n"
                    "1. Current market regime assessment\n"
                    "2. Portfolio gaps (regimes/domains/symbols underrepresented)\n"
                    "3. Recent findings to build on or dead ends to avoid\n"
                    "4. Research priorities for this cycle\n\n"
                    'Return JSON: {"summary": "...", "regime": "...", "gaps": [...], "priorities": [...]}'
                )
            else:
                # Original tool-based prompt (fallback)
                prompt = (
                    f"Cycle {state['cycle_number']}, {regime_section}.\n"
                    f"{community_ideas_section}\n"
                    f"{queued_ideas_section}\n"
                    "Use your tools to gather the current research context:\n"
                    "1. Search the knowledge base for recent session findings and NEGATIVE results "
                    "(to avoid retesting failed hypotheses)\n"
                    "2. Use fetch_market_data for a few INDIVIDUAL STOCKS (not just indices) to see "
                    "which symbols have data available. Try 2-3 stocks you haven't researched recently.\n"
                    "3. Get signal briefs for the symbols that have data.\n"
                    "4. Fetch the strategy registry to identify portfolio gaps.\n\n"
                    "IMPORTANT: Diversify research across individual stocks, not just ETFs/indices. "
                    "Explore momentum, quality, value, sector rotation, and trend-following — not just "
                    "mean-reversion. If regime is 'unknown', use price action and trend analysis instead.\n\n"
                    "Summarize: market regime, portfolio gaps, recent findings, what to research next.\n"
                    'Return JSON: {"summary": "...", "regime": "...", "gaps": [...], "priorities": [...]}'
                )

            text = await run_agent(llm, tools, config, prompt)
            parsed = parse_json_response(text, {"summary": text})
            return {
                "context_summary": parsed.get("summary", text),
                "regime": detected_regime,
                "regime_detail": regime_detail,
                "decisions": [{"node": "context_load", "action": "loaded_context"}],
                "hypothesis_attempts": 0,
                "hypothesis_confidence": 0.0,
                "hypothesis_critique": "",
                "queued_task_ids": queued_task_ids,
            }
        except Exception as exc:
            logger.error("context_load failed: %s", exc)
            return {
                "context_summary": f"Context load failed: {exc}",
                "regime": detected_regime,
                "regime_detail": regime_detail,
                "errors": [f"context_load: {exc}"],
                "hypothesis_attempts": 0,
                "hypothesis_confidence": 0.0,
                "hypothesis_critique": "",
                "queued_task_ids": queued_task_ids,
            }

    return context_load


def make_domain_selection(llm: BaseChatModel, config: AgentConfig, tools: list[BaseTool] | None = None):
    """Create the domain_selection node (agent with tool access)."""
    tools = tools or []

    async def domain_selection(state: ResearchState) -> dict[str, Any]:
        try:
            prompt = (
                f"Context: {state.get('context_summary', 'no context')}\n\n"
                "Based on the context, select a research domain and symbols.\n"
                "Use your tools to:\n"
                "1. Search knowledge base for what domains need more coverage\n"
                "2. Use fetch_market_data to check which individual stocks have data available. "
                "Try stocks you haven't researched in recent cycles. Avoid indices (SPY/QQQ/IWM) "
                "for strategy research — focus on individual stocks.\n"
                "3. Get signal briefs for the stocks that have data.\n\n"
                "Select domain: 'swing', 'investment', or 'options'.\n"
                "Select 1-3 individual stocks that have data and show interesting setups.\n\n"
                "Strategy types to explore (rotate through these):\n"
                "- Momentum: 20-day breakouts, relative strength, earnings acceleration\n"
                "- Quality/Value: ROE improvement, P/E compression, buyback yield\n"
                "- Sector rotation: energy vs tech, cyclical vs defensive\n"
                "- Trend-following: moving average crossovers, ADX-based entries\n"
                "- Event-driven: earnings drift, analyst revision, institutional accumulation\n\n"
                'Return JSON: {"domain": "...", "symbols": ["..."], "reasoning": "..."}'
            )
            text = await run_agent(llm, tools, config, prompt)
            parsed = parse_json_response(text, {"domain": "swing", "symbols": []})
            return {
                "selected_domain": parsed.get("domain", "swing"),
                "selected_symbols": parsed.get("symbols", []),
                "decisions": [{"node": "domain_selection", "domain": parsed.get("domain"), "reasoning": parsed.get("reasoning")}],
            }
        except Exception as exc:
            logger.error("domain_selection failed: %s", exc)
            return {
                "selected_domain": "swing",
                "selected_symbols": [],
                "errors": [f"domain_selection: {exc}"],
            }

    return domain_selection


def make_hypothesis_generation(llm: BaseChatModel, config: AgentConfig, tools: list[BaseTool] | None = None):
    """Create the hypothesis_generation node (agent with tool access)."""
    tools = tools or []

    async def hypothesis_generation(state: ResearchState) -> dict[str, Any]:
        try:
            critique_section = ""
            if state.get("hypothesis_critique"):
                critique_section = (
                    "--- PREVIOUS ATTEMPT FEEDBACK ---\n"
                    f"Your previous hypotheses were critiqued:\n{state['hypothesis_critique']}\n"
                    "Address these weaknesses in your revised hypotheses.\n"
                    "--- END FEEDBACK ---\n\n"
                )

            prompt = (
                f"{critique_section}"
                f"Domain: {state.get('selected_domain', 'swing')}\n"
                f"Symbols: {state.get('selected_symbols', [])}\n\n"
                "Use your tools to:\n"
                "1. Fetch market data for the selected symbols\n"
                "2. Compute features (indicators, factors) to identify patterns\n"
                "3. Search knowledge base for past hypotheses on these symbols (avoid retesting dead ends)\n\n"
                "Generate a testable trading hypothesis with:\n"
                "- Directional prediction\n"
                "- Economic mechanism\n"
                "- Expected effect size\n"
                "- Falsification criteria\n"
                "- Signals to check\n"
                'Return JSON: {"hypothesis": "...", "mechanism": "...", "signals_to_check": [...], '
                '"expected_effect": "...", "falsification": "..."}'
            )
            text = await run_agent(llm, tools, config, prompt)
            parsed = parse_json_response(text, {"hypothesis": text})
            return {
                "hypothesis": parsed.get("hypothesis", text),
                "hypothesis_attempts": state.get("hypothesis_attempts", 0) + 1,
                "decisions": [{"node": "hypothesis_generation", "hypothesis": parsed.get("hypothesis")}],
            }
        except Exception as exc:
            logger.error("hypothesis_generation failed: %s", exc)
            return {
                "hypothesis": "",
                "hypothesis_attempts": state.get("hypothesis_attempts", 0) + 1,
                "errors": [f"hypothesis_generation: {exc}"],
            }

    return hypothesis_generation


def make_signal_validation(llm: BaseChatModel, config: AgentConfig, tools: list[BaseTool] | None = None):
    """Create the signal_validation node (tool-heavy)."""
    tools = tools or []

    async def signal_validation(state: ResearchState) -> dict[str, Any]:
        try:
            prompt = (
                f"Hypothesis: {state.get('hypothesis', '')}\n"
                f"Symbols: {state.get('selected_symbols', [])}\n\n"
                "Validate this hypothesis using real data:\n"
                "1. Get signal briefs for the target symbols\n"
                "2. Compute features relevant to the hypothesis\n"
                "3. Check for signal confluence across indicators\n\n"
                "Validation criteria (pass if MAJORITY met):\n"
                "- IC > 0.02 OR strong directional trend in price data\n"
                "- Signal half-life > intended holding period\n"
                "- At least 2 confirming indicators from: price trend, volume, momentum, "
                "mean-reversion, relative strength, fundamentals\n\n"
                "IMPORTANT: If regime is 'unknown' or regime confidence is 0%, do NOT reject "
                "solely for regime reasons. Use price action and technical indicators as regime "
                "proxies instead. A hypothesis can still pass if the price data and technical "
                "signals support it, even without a formal regime classification.\n\n"
                "Be constructive: if 2 of 3 criteria pass, approve for backtesting. "
                "The backtest will be the definitive test.\n\n"
                'Return JSON: {"passed": true/false, "signals": [...], "ic": ..., "reason": "..."}'
            )
            text = await run_agent(llm, tools, config, prompt)
            result = parse_json_response(text, {"passed": False, "reason": "Failed to parse"})
            decisions = [{"node": "signal_validation", "passed": result.get("passed", False)}]
            state_update: dict[str, Any] = {"validation_result": result, "decisions": decisions}
            if not result.get("passed", False):
                state_update["errors"] = [f"Signal validation failed: {result.get('reason', 'unknown')}"]
            return state_update
        except Exception as exc:
            logger.error("signal_validation failed: %s", exc)
            return {
                "validation_result": {"passed": False, "reason": str(exc)},
                "errors": [f"signal_validation: {exc}"],
            }

    return signal_validation


def route_after_validation(state: ResearchState) -> str:
    """Conditional router after signal_validation."""
    from langgraph.graph import END
    if state.get("validation_result", {}).get("passed", False):
        return "backtest_validation"
    return END


def make_backtest_validation(llm: BaseChatModel, config: AgentConfig, tools: list[BaseTool] | None = None):
    """Create the backtest_validation node (tool-heavy)."""
    tools = tools or []

    async def backtest_validation(state: ResearchState) -> dict[str, Any]:
        try:
            prompt = (
                f"Hypothesis: {state.get('hypothesis', '')}\n"
                f"Symbols: {state.get('selected_symbols', [])}\n"
                f"Validation: {state.get('validation_result', {})}\n\n"
                "Run a backtest for this validated hypothesis:\n"
                "1. Fetch market data for the symbols\n"
                "2. Compute features needed for the strategy\n"
                "3. Run the backtest with the strategy parameters\n\n"
                "Evaluate: Sharpe > 0.5, trades > 20, PF > 1.2, max DD < 20%.\n"
                'Return JSON: {"backtest_id": "...", "sharpe": ..., "win_rate": ..., '
                '"max_dd": ..., "trades": ..., "profit_factor": ..., "passed": true/false}'
            )
            text = await run_agent(llm, tools, config, prompt)
            parsed = parse_json_response(text, {"backtest_id": "bt-unknown"})
            return {
                "backtest_id": parsed.get("backtest_id", "bt-unknown"),
                "decisions": [{"node": "backtest_validation", "metrics": parsed}],
            }
        except Exception as exc:
            logger.error("backtest_validation failed: %s", exc)
            return {"backtest_id": "", "errors": [f"backtest_validation: {exc}"]}

    return backtest_validation


def make_ml_experiment(llm: BaseChatModel, config: AgentConfig, tools: list[BaseTool] | None = None):
    """Create the ml_experiment node (agent with ML tools)."""
    tools = tools or []

    async def ml_experiment(state: ResearchState) -> dict[str, Any]:
        try:
            import os
            draft_mode = os.environ.get("ML_TRAINING_MODE", "draft") == "draft"

            scope_instruction = ""
            if draft_mode:
                scope_instruction = (
                    "\nDRAFT MODE — use reduced scope for fast initial validation:\n"
                    "- Training window: 126 days (6 months) instead of 252\n"
                    "- Feature set: core features only (price, volume, momentum, volatility)\n"
                    "- Walk-forward: 3 folds instead of 5\n"
                    "- No hyperparameter grid search — use defaults\n"
                    "- training_days=126, n_folds=3 if your tool supports these params\n"
                    "This is a quick validation pass. Full training happens at promotion.\n"
                )

            prompt = (
                f"Hypothesis: {state.get('hypothesis', '')}\n"
                f"Backtest: {state.get('backtest_id', '')}\n"
                f"Symbols: {state.get('selected_symbols', [])}\n\n"
                "Design and run an ML experiment:\n"
                "1. Fetch market data for training\n"
                "2. Compute features for the ML model\n"
                "3. Train a model (LightGBM, XGBoost, or ensemble)\n"
                "4. Search knowledge base for past experiment results\n\n"
                f"{scope_instruction}"
                "Follow the Feature Quality Protocol: stationarity, redundancy, stability.\n"
                'Return JSON: {"experiment_id": "...", "model_type": "...", "ic": ..., '
                '"accuracy": ..., "feature_importance": [...], "passed": true/false}'
            )
            text = await run_agent(llm, tools, config, prompt)
            parsed = parse_json_response(text, {"experiment_id": "exp-unknown"})
            return {
                "ml_experiment_id": parsed.get("experiment_id", "exp-unknown"),
                "decisions": [{"node": "ml_experiment", "results": parsed}],
            }
        except Exception as exc:
            logger.error("ml_experiment failed: %s", exc)
            return {"ml_experiment_id": "", "errors": [f"ml_experiment: {exc}"]}

    return ml_experiment


def make_strategy_registration(llm: BaseChatModel, config: AgentConfig, tools: list[BaseTool] | None = None):
    """Create the strategy_registration node (tool)."""
    tools = tools or []

    async def strategy_registration(state: ResearchState) -> dict[str, Any]:
        try:
            regime = state.get("regime", "unknown")
            regime_detail = state.get("regime_detail", {})
            regime_instruction = ""
            if regime != "unknown":
                regime_instruction = (
                    f"\nCurrent market regime: {regime} "
                    f"(vol={regime_detail.get('volatility','?')}, adx={regime_detail.get('adx',0):.0f}).\n"
                    f"Set regime_affinity to match: e.g. '{regime}' should be in the strategy's "
                    f"target regimes. This ensures the strategy is deployed in the right conditions.\n"
                )
            prompt = (
                f"Register strategy from hypothesis: {state.get('hypothesis', '')}\n"
                f"Backtest: {state.get('backtest_id', '')}\n"
                f"ML experiment: {state.get('ml_experiment_id', '')}\n"
                f"{regime_instruction}\n"
                "Use fetch_strategy_registry to check current registry for duplicates.\n"
                "Then register this strategy with status 'paper_ready'.\n"
                "IMPORTANT: Use ONLY standard indicator-based entry rules. Supported indicators: "
                "rsi, adx, atr, bb_pct, stoch_k, stoch_d, cci, zscore, sma_fast, sma_slow, sma_200, "
                "volume, close, high, low. Conditions: above, below, crosses_above, crosses_below, "
                "greater_than, less_than. Do NOT use custom conditions like price_at_hvn_support.\n"
                'Return JSON: {"strategy_id": "...", "status": "paper_ready", "reasoning": "..."}'
            )
            text = await run_agent(llm, tools, config, prompt)
            parsed = parse_json_response(text, {"strategy_id": "strat-unknown"})
            return {
                "registered_strategy_id": parsed.get("strategy_id", "strat-unknown"),
                "decisions": [{"node": "strategy_registration", "strategy_id": parsed.get("strategy_id")}],
            }
        except Exception as exc:
            logger.error("strategy_registration failed: %s", exc)
            return {"registered_strategy_id": "", "errors": [f"strategy_registration: {exc}"]}

    return strategy_registration


def make_knowledge_update(llm: BaseChatModel, config: AgentConfig, tools: list[BaseTool] | None = None):
    """Create the knowledge_update node (tool)."""
    tools = tools or []

    async def knowledge_update(state: ResearchState) -> dict[str, Any]:
        try:
            prompt = (
                f"Update knowledge base with research findings:\n"
                f"Hypothesis: {state.get('hypothesis', '')}\n"
                f"Strategy: {state.get('registered_strategy_id', '')}\n"
                f"Validation: {state.get('validation_result', {})}\n\n"
                "Use search_knowledge_base to check for existing entries.\n"
                "Summarize what was learned (both positive and negative results).\n"
                'Return JSON: {"updated": true, "summary": "..."}'
            )
            text = await run_agent(llm, tools, config, prompt)

            # Mark consumed research_queue items as done
            queued_ids = state.get("queued_task_ids", [])
            if queued_ids:
                try:
                    from quantstack.db import db_conn
                    validation = state.get("validation_result", {})
                    passed = validation.get("passed", False)
                    rq_status = "done" if passed else "failed"
                    error_msg = "" if passed else validation.get("reason", "hypothesis not validated")

                    with db_conn() as conn:
                        for task_id in queued_ids:
                            conn.execute(
                                "UPDATE research_queue "
                                "SET status = %s, completed_at = NOW(), "
                                "    error_message = %s "
                                "WHERE task_id = %s AND status = 'running'",
                                [rq_status, error_msg[:2000] if error_msg else None, task_id],
                            )
                    logger.info(
                        "knowledge_update: marked %d queue tasks as %s",
                        len(queued_ids), rq_status,
                    )
                except Exception as rq_exc:
                    logger.warning("knowledge_update: failed to update research_queue: %s", rq_exc)

            return {
                "decisions": [{"node": "knowledge_update", "action": "updated"}],
            }
        except Exception as exc:
            logger.error("knowledge_update failed: %s", exc)

            # Best-effort: release claimed queue items back to pending on failure
            queued_ids = state.get("queued_task_ids", [])
            if queued_ids:
                try:
                    from quantstack.db import db_conn
                    with db_conn() as conn:
                        for task_id in queued_ids:
                            conn.execute(
                                "UPDATE research_queue SET status = 'pending', started_at = NULL "
                                "WHERE task_id = %s AND status = 'running'",
                                [task_id],
                            )
                except Exception:
                    pass

            return {"errors": [f"knowledge_update: {exc}"]}

    return knowledge_update


# ---------------------------------------------------------------------------
# WI-8: Self-critique loop
# ---------------------------------------------------------------------------


def make_hypothesis_critique(llm: BaseChatModel, config: AgentConfig, tools: list[BaseTool] | None = None):
    """Create the hypothesis_critique node that evaluates hypothesis quality."""
    tools = tools or []

    async def hypothesis_critique(state: ResearchState) -> dict[str, Any]:
        try:
            prompt = (
                f"Evaluate this hypothesis for quality before it enters the validation pipeline.\n\n"
                f"Hypothesis: {state.get('hypothesis', '')}\n"
                f"Domain: {state.get('selected_domain', 'swing')}\n"
                f"Symbols: {state.get('selected_symbols', [])}\n"
                f"Attempt: {state.get('hypothesis_attempts', 0)}\n\n"
                "Score 0-1 confidence on these criteria:\n"
                "- Specificity: clear entry/exit conditions, not vague directional bets\n"
                "- Testability: falsification criteria checkable with available data\n"
                "- Novelty: meaningfully different from existing strategies\n"
                "- Economic mechanism: WHY this edge exists\n"
                "- Risk/reward clarity: expected effect size and loss parameters\n\n"
                "Use search_knowledge_base and fetch_strategy_registry to check novelty.\n\n"
                "If confidence < 0.7, explain EXACTLY what is weak and how to improve it.\n"
                'Return JSON: {"confidence": 0.0-1.0, "critique": "...", '
                '"strengths": [...], "weaknesses": [...]}'
            )
            text = await run_agent(llm, tools, config, prompt)
            parsed = parse_json_response(text, {"confidence": 0.0, "critique": "Failed to parse"})
            confidence = float(parsed.get("confidence", 0.0))
            import os
            threshold = float(os.environ.get("HYPOTHESIS_CONFIDENCE_THRESHOLD", "0.7"))
            critique_text = parsed.get("critique", "") if confidence < threshold else ""
            return {
                "hypothesis_confidence": confidence,
                "hypothesis_critique": critique_text,
                "decisions": [{"node": "hypothesis_critique", "confidence": confidence, "attempt": state.get("hypothesis_attempts", 0)}],
            }
        except Exception as exc:
            logger.error("hypothesis_critique failed: %s", exc)
            return {
                "hypothesis_confidence": 0.0,
                "hypothesis_critique": f"Critique failed: {exc}",
                "errors": [f"hypothesis_critique: {exc}"],
            }

    return hypothesis_critique


def route_after_hypothesis(state: ResearchState) -> str:
    """Route after hypothesis critique: forward if confident, loop if not."""
    import os
    threshold = float(os.environ.get("HYPOTHESIS_CONFIDENCE_THRESHOLD", "0.7"))
    confidence = state.get("hypothesis_confidence", 0.0)
    attempts = state.get("hypothesis_attempts", 0)

    if confidence >= threshold or attempts >= 3:
        from quantstack.observability.tracing import trace_hypothesis_loop
        trace_hypothesis_loop(
            loop_count=attempts,
            final_confidence=confidence,
            max_attempts_hit=attempts >= 3,
        )
        return "signal_validation"
    return "hypothesis_generation"


# ---------------------------------------------------------------------------
# WI-7: Research fan-out via Send()
# ---------------------------------------------------------------------------


def fan_out_hypotheses(state: ResearchState) -> list[Send]:
    """Fan out per-symbol validation workers via Send()."""
    symbols = state.get("selected_symbols", [])
    hypothesis = state.get("hypothesis", "")
    domain = state.get("selected_domain", "swing")

    return [
        Send("validate_symbol", {
            "symbol_hypothesis": {
                "symbol": symbol,
                "hypothesis": hypothesis,
                "domain": domain,
            }
        })
        for symbol in symbols
    ]


def route_after_hypothesis_fanout(state: ResearchState) -> list[Send] | str:
    """Router that fans out to parallel workers when confidence is sufficient."""
    import os
    threshold = float(os.environ.get("HYPOTHESIS_CONFIDENCE_THRESHOLD", "0.7"))
    confidence = state.get("hypothesis_confidence", 0.0)
    attempts = state.get("hypothesis_attempts", 0)

    if confidence >= threshold or attempts >= 3:
        from quantstack.observability.tracing import trace_hypothesis_loop
        trace_hypothesis_loop(
            loop_count=attempts,
            final_confidence=confidence,
            max_attempts_hit=attempts >= 3,
        )
        return fan_out_hypotheses(state)
    return "hypothesis_generation"


def make_validate_symbol(
    quant_llm: BaseChatModel,
    ml_llm: BaseChatModel,
    quant_cfg: AgentConfig,
    ml_cfg: AgentConfig,
    quant_tools: list[BaseTool],
    ml_tools: list[BaseTool],
):
    """Factory that returns a validate_symbol node with LLMs captured via closure."""

    async def validate_symbol(state: SymbolValidationState) -> dict:
        import time
        hyp = state["symbol_hypothesis"]
        symbol = hyp["symbol"]
        hypothesis = hyp["hypothesis"]
        domain = hyp.get("domain", "swing")
        t0 = time.monotonic()

        try:
            # Signal validation
            val_prompt = (
                f"Validate hypothesis for {symbol}:\n{hypothesis}\n"
                f"Domain: {domain}\n"
                "Fetch data, compute features, check signal confluence.\n"
                'Return JSON: {{"passed": true/false, "signals": [...], "reason": "..."}}'
            )
            val_text = await run_agent(quant_llm, quant_tools, quant_cfg, val_prompt)
            val_result = parse_json_response(val_text, {"passed": False, "reason": "parse failed"})

            if not val_result.get("passed", False):
                return {"validation_results": [{"symbol": symbol, "passed": False, "reason": val_result.get("reason", "signal failed")}]}

            # Backtest
            bt_prompt = (
                f"Run backtest for {symbol} with hypothesis:\n{hypothesis}\n"
                'Return JSON: {{"backtest_id": "...", "sharpe": ..., "passed": true/false}}'
            )
            bt_text = await run_agent(quant_llm, quant_tools, quant_cfg, bt_prompt)
            bt_result = parse_json_response(bt_text, {"passed": False})

            # ML experiment
            ml_prompt = (
                f"Run ML experiment for {symbol} with hypothesis:\n{hypothesis}\n"
                'Return JSON: {{"experiment_id": "...", "ic": ..., "passed": true/false}}'
            )
            ml_text = await run_agent(ml_llm, ml_tools, ml_cfg, ml_prompt)
            ml_result = parse_json_response(ml_text, {"passed": False})

            duration = time.monotonic() - t0
            from quantstack.observability.tracing import trace_fanout_worker
            trace_fanout_worker(symbol=symbol, worker_index=0, duration_seconds=duration, success=True)
            return {
                "validation_results": [{
                    "symbol": symbol,
                    "passed": True,
                    "validation_details": val_result,
                    "backtest_results": bt_result,
                    "ml_results": ml_result,
                    "duration_seconds": duration,
                }]
            }
        except Exception as exc:
            duration = time.monotonic() - t0
            from quantstack.observability.tracing import trace_fanout_worker
            trace_fanout_worker(symbol=symbol, worker_index=0, duration_seconds=duration, success=False, error=str(exc))
            logger.error("validate_symbol failed for %s: %s", symbol, exc)
            return {
                "validation_results": [{
                    "symbol": symbol,
                    "passed": False,
                    "error": str(exc),
                    "duration_seconds": duration,
                }]
            }

    return validate_symbol


def make_filter_results():
    """Create the filter_results node that consolidates fan-out results."""

    async def filter_results(state: ResearchState) -> dict:
        results = state.get("validation_results", [])
        passed = [r for r in results if r.get("passed", False)]
        failed = [r for r in results if not r.get("passed", False)]

        for f in failed:
            logger.warning("Symbol %s failed validation: %s", f.get("symbol", "?"), f.get("error", f.get("reason", "unknown")))

        if passed:
            consolidated = {
                "passed": True,
                "symbols_passed": [r["symbol"] for r in passed],
                "symbols_failed": [r.get("symbol", "?") for r in failed],
                "results": passed,
                "reason": f"{len(passed)} of {len(results)} symbols passed validation",
            }
        else:
            consolidated = {
                "passed": False,
                "reason": f"All {len(results)} symbols failed validation",
                "symbols_failed": [r.get("symbol", "?") for r in failed],
            }

        return {
            "validation_result": consolidated,
            "decisions": [{"node": "filter_results", "passed": len(passed), "failed": len(failed)}],
        }

    return filter_results
