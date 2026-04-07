"""Overnight autonomous research runner (20:00-04:00 ET).

Generates hypotheses, backtests them, scores results, and logs experiments
to the autoresearch_experiments table. Experiments that exceed an OOS IC
threshold of 0.02 are marked as 'winner' for morning validation.

Budget ceiling: $9.50 per night. Crash-safe via cumulative DB reads.
Each experiment has a 5-minute timeout to prevent runaway LLM calls.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timezone

from zoneinfo import ZoneInfo

from quantstack.coordination.event_bus import Event, EventBus, EventType
from quantstack.db import db_conn

logger = logging.getLogger(__name__)

# Operating window in US/Eastern
_TZ_ET = ZoneInfo("US/Eastern")

# Budget ceiling per night (USD)
BUDGET_CEILING_USD = 9.50

# OOS IC threshold for marking an experiment as a winner
WINNER_IC_THRESHOLD = 0.02

# Per-experiment timeout in seconds
EXPERIMENT_TIMEOUT_SECONDS = 300

# Approximate cost per 1000 tokens (conservative estimate)
_COST_PER_1K_TOKENS = 0.003


def _now_et() -> datetime:
    """Current time in US/Eastern."""
    return datetime.now(_TZ_ET)


def _is_within_operating_window() -> bool:
    """Check if current time is within 20:00-04:00 ET."""
    now = _now_et()
    hour = now.hour
    # 20:00 through 23:59 or 00:00 through 03:59
    return hour >= 20 or hour < 4


def _generate_experiment_id(night_date: str) -> str:
    """Generate a unique experiment ID with night_date prefix."""
    short_uuid = uuid.uuid4().hex[:8]
    return f"exp-{night_date}-{short_uuid}"


def get_nightly_budget_state(night_date: str) -> float:
    """Read cumulative cost from DB for crash recovery.

    Returns total USD spent on experiments for the given night_date.
    """
    with db_conn() as conn:
        row = conn.execute(
            "SELECT COALESCE(SUM(cost_usd), 0) FROM autoresearch_experiments "
            "WHERE night_date = %s",
            (night_date,),
        ).fetchone()
    return float(row[0]) if row else 0.0


def generate_hypothesis() -> dict:
    """Generate a trading hypothesis using an LLM.

    Returns a dict with entry_rules, exit_rules, and parameters.
    """
    from quantstack.llm.provider import get_chat_model

    model = get_chat_model("light")
    prompt = (
        "Generate a quantitative trading hypothesis. "
        "Return a JSON object with keys: "
        "'entry_rules' (list of conditions to enter a trade), "
        "'exit_rules' (list of conditions to exit a trade), "
        "'parameters' (dict of numeric parameters like lookback periods, thresholds), "
        "'rationale' (brief explanation of why this might work). "
        "Focus on technical or statistical edges. Be specific and testable."
    )
    from langchain_core.messages import HumanMessage

    response = model.invoke([HumanMessage(content=prompt)])
    content = response.content

    # Parse JSON from response — handle markdown code blocks
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0]
    elif "```" in content:
        content = content.split("```")[1].split("```")[0]

    try:
        hypothesis = json.loads(content.strip())
    except json.JSONDecodeError:
        # Fallback: wrap raw text as a hypothesis
        hypothesis = {
            "entry_rules": [content.strip()],
            "exit_rules": ["Stop loss at 2%"],
            "parameters": {},
            "rationale": "LLM-generated (unparsed)",
        }

    return hypothesis


def score_experiment(backtest_result: dict) -> dict:
    """Compute OOS IC and Sharpe from backtest result.

    Parameters
    ----------
    backtest_result : dict
        Must contain 'oos_ic' and 'sharpe' keys at minimum.

    Returns
    -------
    dict with 'oos_ic', 'sharpe', and 'status' ('winner' or 'tested').
    """
    oos_ic = backtest_result.get("oos_ic", 0.0)
    sharpe = backtest_result.get("sharpe", 0.0)
    status = "winner" if oos_ic > WINNER_IC_THRESHOLD else "tested"
    return {"oos_ic": oos_ic, "sharpe": sharpe, "status": status}


def _log_experiment(
    experiment_id: str,
    night_date: str,
    hypothesis: dict,
    hypothesis_source: str,
    oos_ic: float,
    sharpe: float,
    cost_tokens: int,
    cost_usd: float,
    duration_seconds: int,
    status: str,
    rejection_reason: str | None = None,
) -> None:
    """Insert experiment record into autoresearch_experiments table."""
    with db_conn() as conn:
        conn.execute(
            """
            INSERT INTO autoresearch_experiments
                (experiment_id, night_date, hypothesis, hypothesis_source,
                 oos_ic, sharpe, cost_tokens, cost_usd,
                 duration_seconds, status, rejection_reason, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                experiment_id,
                night_date,
                json.dumps(hypothesis),
                hypothesis_source,
                oos_ic,
                sharpe,
                cost_tokens,
                cost_usd,
                duration_seconds,
                status,
                rejection_reason,
                datetime.now(timezone.utc),
            ),
        )


def _publish_experiment_completed(experiment_id: str, status: str, oos_ic: float) -> None:
    """Publish EXPERIMENT_COMPLETED event to the event bus."""
    try:
        with db_conn() as conn:
            bus = EventBus(conn)
            bus.publish(
                Event(
                    event_type=EventType.EXPERIMENT_COMPLETED,
                    source_loop="overnight_research",
                    payload={
                        "experiment_id": experiment_id,
                        "status": status,
                        "oos_ic": oos_ic,
                    },
                )
            )
    except Exception as exc:
        logger.warning("Failed to publish EXPERIMENT_COMPLETED event: %s", exc)


async def run_single_experiment(
    hypothesis: dict,
    source: str,
    night_date: str,
) -> dict:
    """Generate, backtest, score, and log a single experiment.

    Parameters
    ----------
    hypothesis : dict
        The trading hypothesis (entry_rules, exit_rules, parameters).
    source : str
        Where the hypothesis came from (e.g., 'llm_generated', 'community_intel').
    night_date : str
        The night date string (YYYY-MM-DD) for grouping experiments.

    Returns
    -------
    dict with experiment_id, status, oos_ic, sharpe, cost_usd.
    """
    experiment_id = _generate_experiment_id(night_date)
    start_time = time.monotonic()

    try:
        # Run backtest (placeholder: in production this would call the backtesting engine)
        from quantstack.core.backtesting.patience import PatienceConfig

        # Simulated backtest — the real implementation would:
        # 1. Convert hypothesis rules into a strategy
        # 2. Run backtest over historical data
        # 3. Compute OOS IC and Sharpe
        backtest_result = _run_backtest(hypothesis)

        duration_seconds = int(time.monotonic() - start_time)
        scored = score_experiment(backtest_result)

        # Estimate token cost
        cost_tokens = backtest_result.get("tokens_used", 5000)
        cost_usd = cost_tokens * _COST_PER_1K_TOKENS / 1000

        _log_experiment(
            experiment_id=experiment_id,
            night_date=night_date,
            hypothesis=hypothesis,
            hypothesis_source=source,
            oos_ic=scored["oos_ic"],
            sharpe=scored["sharpe"],
            cost_tokens=cost_tokens,
            cost_usd=cost_usd,
            duration_seconds=duration_seconds,
            status=scored["status"],
        )

        _publish_experiment_completed(experiment_id, scored["status"], scored["oos_ic"])

        return {
            "experiment_id": experiment_id,
            "status": scored["status"],
            "oos_ic": scored["oos_ic"],
            "sharpe": scored["sharpe"],
            "cost_usd": cost_usd,
        }

    except asyncio.TimeoutError:
        duration_seconds = int(time.monotonic() - start_time)
        _log_experiment(
            experiment_id=experiment_id,
            night_date=night_date,
            hypothesis=hypothesis,
            hypothesis_source=source,
            oos_ic=0.0,
            sharpe=0.0,
            cost_tokens=0,
            cost_usd=0.0,
            duration_seconds=duration_seconds,
            status="timeout",
            rejection_reason="Exceeded 5-minute timeout",
        )
        return {
            "experiment_id": experiment_id,
            "status": "timeout",
            "oos_ic": 0.0,
            "sharpe": 0.0,
            "cost_usd": 0.0,
        }

    except Exception as exc:
        duration_seconds = int(time.monotonic() - start_time)
        _log_experiment(
            experiment_id=experiment_id,
            night_date=night_date,
            hypothesis=hypothesis,
            hypothesis_source=source,
            oos_ic=0.0,
            sharpe=0.0,
            cost_tokens=0,
            cost_usd=0.0,
            duration_seconds=duration_seconds,
            status="error",
            rejection_reason=str(exc)[:500],
        )
        logger.error("Experiment %s failed: %s", experiment_id, exc)
        return {
            "experiment_id": experiment_id,
            "status": "error",
            "oos_ic": 0.0,
            "sharpe": 0.0,
            "cost_usd": 0.0,
        }


def _run_backtest(hypothesis: dict) -> dict:
    """Run a backtest for the given hypothesis.

    This is a thin wrapper that delegates to the backtesting engine.
    Returns dict with oos_ic, sharpe, and tokens_used.
    """
    # In production this would call the real backtesting engine.
    # For now, return a stub that downstream code can override.
    raise NotImplementedError(
        "Backtest engine integration not yet wired. "
        "Override _run_backtest or provide a backtest_fn to run_single_experiment."
    )


async def run_overnight_loop() -> dict:
    """Main overnight research loop. Runs between 20:00-04:00 ET.

    Generates hypotheses, runs experiments, and stops when:
    - The operating window closes (04:00 ET)
    - The budget ceiling ($9.50) is reached
    - An unrecoverable error occurs

    Returns a summary dict with experiment counts and total cost.
    """
    night_date = _now_et().strftime("%Y-%m-%d")
    logger.info("Starting overnight research loop for night_date=%s", night_date)

    # Crash recovery: read how much we've already spent tonight
    cumulative_cost = get_nightly_budget_state(night_date)
    logger.info("Resuming with cumulative cost: $%.4f", cumulative_cost)

    results = {"winners": 0, "tested": 0, "errors": 0, "timeouts": 0, "total_cost": cumulative_cost}

    while _is_within_operating_window():
        # Budget check
        if cumulative_cost >= BUDGET_CEILING_USD:
            logger.info(
                "Budget ceiling reached ($%.2f >= $%.2f). Stopping.",
                cumulative_cost,
                BUDGET_CEILING_USD,
            )
            break

        try:
            hypothesis = generate_hypothesis()
        except Exception as exc:
            logger.error("Hypothesis generation failed: %s", exc)
            results["errors"] += 1
            await asyncio.sleep(30)
            continue

        try:
            result = await asyncio.wait_for(
                run_single_experiment(hypothesis, "llm_generated", night_date),
                timeout=EXPERIMENT_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            logger.warning("Experiment timed out after %ds", EXPERIMENT_TIMEOUT_SECONDS)
            results["timeouts"] += 1
            continue

        cumulative_cost += result.get("cost_usd", 0.0)
        status = result.get("status", "error")
        results[status] = results.get(status, 0) + 1
        results["total_cost"] = cumulative_cost

        logger.info(
            "Experiment %s: status=%s, oos_ic=%.4f, cost=$%.4f, cumulative=$%.4f",
            result["experiment_id"],
            status,
            result["oos_ic"],
            result["cost_usd"],
            cumulative_cost,
        )

        # Brief pause between experiments to avoid hammering the LLM
        await asyncio.sleep(5)

    logger.info(
        "Overnight loop complete: %d winners, %d tested, %d errors, %d timeouts, $%.4f total",
        results["winners"],
        results["tested"],
        results["errors"],
        results["timeouts"],
        results["total_cost"],
    )
    return results
