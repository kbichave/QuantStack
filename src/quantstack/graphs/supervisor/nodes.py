"""Node functions for the Supervisor Graph.

Each node is an async function: (SupervisorState) -> dict
The return dict contains only the state fields the node updates.

Supervisor nodes use tools for real system introspection (heartbeats,
system status, strategy registry) rather than hallucinating health data.
"""

import json
import logging
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool

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


def make_strategy_lifecycle(llm: BaseChatModel, config: AgentConfig, tools: list[BaseTool] | None = None):
    """Create the strategy_lifecycle node with registry access."""
    tools = tools or []

    async def strategy_lifecycle(state: SupervisorState) -> dict[str, Any]:
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
            actions = parse_json_response(text, [])
            if not isinstance(actions, list):
                actions = [actions] if actions else []
            return {"strategy_lifecycle_actions": actions}
        except Exception as exc:
            logger.error("strategy_lifecycle failed: %s", exc)
            return {
                "strategy_lifecycle_actions": [],
                "errors": [f"strategy_lifecycle: {exc}"],
            }

    return strategy_lifecycle


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


def make_scheduled_tasks(llm: BaseChatModel, config: AgentConfig, tools: list[BaseTool] | None = None):
    """Create the scheduled_tasks node with concrete community_intel and execution_researcher."""
    tools = tools or []

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
                    "Run a weekly community intelligence scan.\n\n"
                    "Search across these sources:\n"
                    "1. arXiv q-fin (quantitative finance papers, last 7 days)\n"
                    "2. GitHub trending repositories (quantitative, trading, ML)\n"
                    "3. Reddit r/algotrading and r/quant (top posts, last 7 days)\n"
                    "4. QuantConnect forums (recent strategy discussions)\n\n"
                    "For each discovery: assess novelty, empirical validation, implementation feasibility.\n"
                    "Filter: skip duplicates, items > 90 days old, ideas without backtest evidence.\n\n"
                    'Return JSON: {"ideas": [{"title": "...", "source": "...", "url": "...", '
                    '"relevance_score": 0.0-1.0, "novelty": "...", "implementation_path": "..."}]}'
                )
                text = await run_agent(llm, tools, config, community_prompt)
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
                except Exception:
                    pass

                community_result["fired"] = True
                community_result["ideas_found"] = len(ideas)
            except Exception as exc:
                logger.error("community_intel task failed: %s", exc)
                community_result["error"] = str(exc)
        results.append(community_result)

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
                except Exception:
                    pass

                exec_result["fired"] = True
            except Exception as exc:
                logger.error("execution_researcher task failed: %s", exc)
                exec_result["error"] = str(exc)
        results.append(exec_result)

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
        except Exception:
            pass  # Table may not exist yet; proceed with refresh

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
