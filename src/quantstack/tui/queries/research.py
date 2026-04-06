"""Research queries: WIP, queue, ML experiments, alpha programs, reflections, bugs, concept drift."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from loguru import logger

from quantstack.db import PgConnection


@dataclass
class ResearchWip:
    symbol: str
    domain: str
    agent_id: str
    started_at: datetime
    duration_minutes: float


@dataclass
class ResearchQueueItem:
    task_type: str
    status: str
    topic: str
    priority: int


@dataclass
class MlExperiment:
    experiment_id: str
    created_at: datetime
    model_type: str
    symbol: str
    test_auc: float | None
    feature_count: int
    verdict: str


@dataclass
class AlphaProgram:
    thesis: str
    status: str
    experiments_run: int
    last_result_summary: str


@dataclass
class Breakthrough:
    feature_name: str
    importance: float


@dataclass
class TradeReflection:
    symbol: str
    realized_pnl_pct: float
    lesson: str
    created_at: datetime


@dataclass
class BugRecord:
    bug_id: str
    tool_name: str
    status: str
    error_message: str
    created_at: datetime


@dataclass
class ConceptDrift:
    symbol: str
    recent_auc: float
    historical_auc: float
    drift_magnitude: float


def fetch_research_wip(conn: PgConnection) -> list[ResearchWip]:
    """Return active research work-in-progress."""
    try:
        conn.execute(
            "SELECT symbol, domain, agent_id, started_at, "
            "EXTRACT(EPOCH FROM (NOW() - COALESCE(heartbeat_at, started_at))) / 60 "
            "FROM research_wip ORDER BY started_at DESC"
        )
        return [
            ResearchWip(
                symbol=r[0], domain=r[1], agent_id=r[2],
                started_at=r[3], duration_minutes=float(r[4] or 0),
            )
            for r in conn.fetchall()
        ]
    except Exception:
        logger.warning("fetch_research_wip failed", exc_info=True)
        return []


def fetch_research_queue(conn: PgConnection) -> list[ResearchQueueItem]:
    """Return pending research queue items."""
    try:
        conn.execute(
            "SELECT task_type, status, topic, priority "
            "FROM research_queue WHERE status = 'pending' "
            "ORDER BY priority"
        )
        return [
            ResearchQueueItem(
                task_type=r[0], status=r[1], topic=r[2], priority=int(r[3]),
            )
            for r in conn.fetchall()
        ]
    except Exception:
        logger.warning("fetch_research_queue failed", exc_info=True)
        return []


def fetch_ml_experiments(conn: PgConnection, limit: int = 10) -> list[MlExperiment]:
    """Return recent ML experiments."""
    try:
        conn.execute(
            "SELECT experiment_id, created_at, model_type, symbol, test_auc, "
            "COALESCE(n_features_filtered, 0), COALESCE(verdict, 'pending') "
            "FROM ml_experiments ORDER BY created_at DESC LIMIT %s",
            (limit,),
        )
        return [
            MlExperiment(
                experiment_id=r[0], created_at=r[1], model_type=r[2], symbol=r[3],
                test_auc=float(r[4]) if r[4] is not None else None,
                feature_count=int(r[5]), verdict=r[6],
            )
            for r in conn.fetchall()
        ]
    except Exception:
        logger.warning("fetch_ml_experiments failed", exc_info=True)
        return []


def fetch_alpha_programs(conn: PgConnection) -> list[AlphaProgram]:
    """Return active alpha research programs."""
    try:
        conn.execute(
            "SELECT thesis, status, COALESCE(experiments_run, 0), "
            "COALESCE(last_result_summary, '') "
            "FROM alpha_research_program WHERE status = 'active'"
        )
        return [
            AlphaProgram(
                thesis=r[0], status=r[1],
                experiments_run=int(r[2]), last_result_summary=r[3],
            )
            for r in conn.fetchall()
        ]
    except Exception:
        logger.warning("fetch_alpha_programs failed", exc_info=True)
        return []


def fetch_breakthroughs(conn: PgConnection) -> list[Breakthrough]:
    """Return breakthrough features ordered by importance."""
    try:
        conn.execute(
            "SELECT feature_name, avg_shap_importance FROM breakthrough_features "
            "ORDER BY avg_shap_importance DESC"
        )
        return [
            Breakthrough(feature_name=r[0], importance=float(r[1]))
            for r in conn.fetchall()
        ]
    except Exception:
        logger.warning("fetch_breakthroughs failed", exc_info=True)
        return []


def fetch_reflections(conn: PgConnection, limit: int = 10) -> list[TradeReflection]:
    """Return recent trade reflections."""
    try:
        conn.execute(
            "SELECT symbol, realized_pnl_pct, lesson, created_at "
            "FROM trade_reflections ORDER BY created_at DESC LIMIT %s",
            (limit,),
        )
        return [
            TradeReflection(
                symbol=r[0], realized_pnl_pct=float(r[1]),
                lesson=r[2], created_at=r[3],
            )
            for r in conn.fetchall()
        ]
    except Exception:
        logger.warning("fetch_reflections failed", exc_info=True)
        return []


def fetch_bugs(conn: PgConnection) -> list[BugRecord]:
    """Return open and in-progress bugs."""
    try:
        conn.execute(
            "SELECT bug_id, tool_name, status, error_message, created_at "
            "FROM bugs WHERE status IN ('open', 'in_progress') "
            "ORDER BY created_at DESC"
        )
        return [
            BugRecord(
                bug_id=r[0], tool_name=r[1], status=r[2],
                error_message=r[3], created_at=r[4],
            )
            for r in conn.fetchall()
        ]
    except Exception:
        logger.warning("fetch_bugs failed", exc_info=True)
        return []


def fetch_concept_drift(
    conn: PgConnection, window_days: int = 14, threshold: float = 0.05,
) -> list[ConceptDrift]:
    """Return symbols where recent model AUC drifts from historical average."""
    try:
        conn.execute(
            "SELECT recent.symbol, recent.avg_auc, hist.avg_auc, "
            "ABS(recent.avg_auc - hist.avg_auc) AS drift "
            "FROM ("
            "  SELECT symbol, AVG(test_auc) AS avg_auc FROM ml_experiments "
            "  WHERE created_at >= NOW() - make_interval(days => %s) "
            "  AND test_auc IS NOT NULL GROUP BY symbol"
            ") recent "
            "JOIN ("
            "  SELECT symbol, AVG(test_auc) AS avg_auc FROM ml_experiments "
            "  WHERE test_auc IS NOT NULL GROUP BY symbol"
            ") hist ON hist.symbol = recent.symbol "
            "WHERE ABS(recent.avg_auc - hist.avg_auc) > %s "
            "ORDER BY drift DESC",
            (window_days, threshold),
        )
        return [
            ConceptDrift(
                symbol=r[0], recent_auc=float(r[1]),
                historical_auc=float(r[2]), drift_magnitude=float(r[3]),
            )
            for r in conn.fetchall()
        ]
    except Exception:
        logger.warning("fetch_concept_drift failed", exc_info=True)
        return []
