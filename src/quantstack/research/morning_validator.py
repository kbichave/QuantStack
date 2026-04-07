"""Morning validation of overnight experiment winners (04:00 ET).

Reads all experiments with status='winner' for a given night_date and
validates each using the 3-window patience protocol. Winners passing
3/3 windows become 'draft' strategies; 2/3 become 'draft' + provisional;
0-1/3 are rejected with logged reasons.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from quantstack.core.backtesting.patience import (
    PatienceConfig,
    WindowResult,
    evaluate_patience,
)
from quantstack.db import db_conn

logger = logging.getLogger(__name__)


def _fetch_winners(night_date: str) -> list[dict]:
    """Fetch all experiments with status='winner' for the given night_date."""
    with db_conn() as conn:
        rows = conn.execute(
            """
            SELECT experiment_id, hypothesis, hypothesis_source, oos_ic, sharpe
            FROM autoresearch_experiments
            WHERE night_date = %s AND status = 'winner'
            ORDER BY oos_ic DESC
            """,
            (night_date,),
        ).fetchall()

    experiments = []
    for row in rows:
        hypothesis_raw = row[1]
        if isinstance(hypothesis_raw, str):
            hypothesis = json.loads(hypothesis_raw)
        else:
            hypothesis = hypothesis_raw or {}
        experiments.append({
            "experiment_id": row[0],
            "hypothesis": hypothesis,
            "hypothesis_source": row[2],
            "oos_ic": row[3],
            "sharpe": row[4],
        })
    return experiments


def validate_winner(experiment: dict, patience_config: PatienceConfig | None = None) -> dict:
    """Validate a winner experiment using the 3-window patience protocol.

    Runs the hypothesis through three distinct market windows:
    - Full history (since full_start)
    - Recent (last N months)
    - Stressed period (e.g., COVID crash)

    Parameters
    ----------
    experiment : dict
        Must contain 'experiment_id', 'hypothesis', 'oos_ic', 'sharpe'.
    patience_config : PatienceConfig, optional
        Override default window configuration.

    Returns
    -------
    dict with 'status' ('draft', 'rejected'), 'provisional' (bool),
    'window_results' (list of WindowResult), and 'rejection_reason' (str or None).
    """
    if patience_config is None:
        patience_config = PatienceConfig()

    hypothesis = experiment["hypothesis"]

    # Run backtest across each patience window
    window_results = _run_patience_windows(hypothesis, patience_config)

    patience_verdict = evaluate_patience(window_results)

    if patience_verdict == "accepted":
        return {
            "status": "draft",
            "provisional": False,
            "window_results": window_results,
            "rejection_reason": None,
        }
    elif patience_verdict == "provisional":
        return {
            "status": "draft",
            "provisional": True,
            "window_results": window_results,
            "rejection_reason": None,
        }
    else:
        failed_windows = [r.window_name for r in window_results if not r.passed]
        reason = f"Failed patience windows: {', '.join(failed_windows)}"
        return {
            "status": "rejected",
            "provisional": False,
            "window_results": window_results,
            "rejection_reason": reason,
        }


def _run_patience_windows(
    hypothesis: dict,
    config: PatienceConfig,
) -> list[WindowResult]:
    """Run backtests across the three patience windows.

    In production, each window calls the backtesting engine with different
    date ranges. This implementation delegates to the backtest engine.
    """
    # Placeholder: real implementation would run three separate backtests.
    # The caller (or test) should mock this function.
    raise NotImplementedError(
        "Patience window backtesting not yet wired to backtest engine. "
        "Mock _run_patience_windows in tests."
    )


def register_draft_strategy(experiment: dict, validation: dict) -> str | None:
    """Insert a validated experiment into the strategies table as a draft.

    Parameters
    ----------
    experiment : dict
        The experiment record (experiment_id, hypothesis, oos_ic, sharpe, etc.).
    validation : dict
        The validation result from validate_winner (status, provisional, etc.).

    Returns
    -------
    The strategy_id if inserted, or None if validation.status != 'draft'.
    """
    if validation["status"] != "draft":
        return None

    strategy_id = f"strat-{experiment['experiment_id']}"
    hypothesis = experiment["hypothesis"]
    provisional = validation.get("provisional", False)

    with db_conn() as conn:
        conn.execute(
            """
            INSERT INTO strategies
                (strategy_id, name, status, hypothesis, oos_ic, sharpe,
                 provisional, source_experiment_id, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (strategy_id) DO NOTHING
            """,
            (
                strategy_id,
                f"AutoResearch: {experiment.get('hypothesis_source', 'unknown')}",
                "draft",
                json.dumps(hypothesis),
                experiment["oos_ic"],
                experiment["sharpe"],
                provisional,
                experiment["experiment_id"],
                datetime.now(timezone.utc),
            ),
        )

    logger.info(
        "Registered draft strategy %s (provisional=%s, oos_ic=%.4f)",
        strategy_id,
        provisional,
        experiment["oos_ic"],
    )
    return strategy_id


def _update_experiment_status(
    experiment_id: str,
    new_status: str,
    rejection_reason: str | None = None,
) -> None:
    """Update the status of an experiment in autoresearch_experiments."""
    with db_conn() as conn:
        conn.execute(
            """
            UPDATE autoresearch_experiments
            SET status = %s, rejection_reason = %s
            WHERE experiment_id = %s
            """,
            (new_status, rejection_reason, experiment_id),
        )


async def run_morning_validation(night_date: str) -> dict:
    """Validate all winner experiments from the overnight run.

    Parameters
    ----------
    night_date : str
        The night date string (YYYY-MM-DD) identifying which experiments to validate.

    Returns
    -------
    dict with counts: drafted, rejected, total_winners, strategies_registered.
    """
    winners = _fetch_winners(night_date)
    logger.info("Morning validation: %d winners found for %s", len(winners), night_date)

    if not winners:
        logger.info("No winners to validate for %s", night_date)
        return {
            "total_winners": 0,
            "drafted": 0,
            "rejected": 0,
            "strategies_registered": [],
        }

    drafted = 0
    rejected = 0
    strategies_registered = []

    for experiment in winners:
        exp_id = experiment["experiment_id"]
        try:
            validation = validate_winner(experiment)
        except Exception as exc:
            logger.error("Validation failed for %s: %s", exp_id, exc)
            _update_experiment_status(exp_id, "rejected", f"Validation error: {str(exc)[:300]}")
            rejected += 1
            continue

        if validation["status"] == "draft":
            strategy_id = register_draft_strategy(experiment, validation)
            _update_experiment_status(exp_id, "draft")
            if strategy_id:
                strategies_registered.append(strategy_id)
            drafted += 1
            logger.info(
                "Winner %s passed validation -> draft (provisional=%s)",
                exp_id,
                validation.get("provisional", False),
            )
        else:
            _update_experiment_status(exp_id, "rejected", validation.get("rejection_reason"))
            rejected += 1
            logger.info(
                "Winner %s rejected: %s",
                exp_id,
                validation.get("rejection_reason", "unknown"),
            )

    summary = {
        "total_winners": len(winners),
        "drafted": drafted,
        "rejected": rejected,
        "strategies_registered": strategies_registered,
    }
    logger.info("Morning validation complete: %s", summary)
    return summary
