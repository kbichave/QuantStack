"""Baseline: capture all existing tables from db.py _migrate_*_pg() functions.

Revision ID: 001
Revises: None
Create Date: 2026-04-06

This baseline migration delegates to the existing _migrate_*_pg() functions
in quantstack.db to guarantee identical schema output. Every function uses
CREATE TABLE IF NOT EXISTS / ADD COLUMN IF NOT EXISTS, making the upgrade
safe to run against databases that already have the tables.

When the legacy migration path is eventually removed, the SQL from these
functions should be inlined here.
"""

from typing import Sequence, Union

from alembic import op

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


class _AlembicConnAdapter:
    """Minimal adapter so _migrate_*_pg(conn) calls route to op.execute().

    The _migrate_*_pg() functions call conn.execute(sql). This adapter
    translates those calls to Alembic's op.execute().
    """

    def execute(self, sql: str, params: object = None) -> "_AlembicConnAdapter":
        op.execute(sql)
        return self

    def _ensure_raw(self):
        return self


def upgrade() -> None:
    from quantstack.db import (
        _migrate_analytics_pg,
        _migrate_attribution_pg,
        _migrate_audit_pg,
        _migrate_broker_pg,
        _migrate_bugs_pg,
        _migrate_capital_allocation_pg,
        _migrate_conversations_pg,
        _migrate_coordination_pg,
        _migrate_equity_alerts_pg,
        _migrate_ewf_pg,
        _migrate_hnsw_index_pg,
        _migrate_institutional_gaps_pg,
        _migrate_learning_pg,
        _migrate_loop_context_pg,
        _migrate_market_data_pg,
        _migrate_market_holidays_pg,
        _migrate_memory_pg,
        _migrate_ml_pipeline_pg,
        _migrate_phase4_coordination_pg,
        _migrate_pnl_attribution_pg,
        _migrate_portfolio_pg,
        _migrate_regime_matrix_pg,
        _migrate_regime_state_pg,
        _migrate_research_queue_pg,
        _migrate_research_wip_pg,
        _migrate_risk_monitoring_pg,
        _migrate_screener_pg,
        _migrate_signal_history_pg,
        _migrate_signal_ic_pg,
        _migrate_signals_pg,
        _migrate_stat_arb_pg,
        _migrate_strategies_pg,
        _migrate_strategy_outcomes_pg,
        _migrate_system_pg,
        _migrate_tool_search_metrics_pg,
        _migrate_trade_quality_pg,
        _migrate_universe_pg,
    )

    conn = _AlembicConnAdapter()

    # Extensions
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # Same order as run_migrations_pg()
    _migrate_portfolio_pg(conn)
    _migrate_broker_pg(conn)
    _migrate_audit_pg(conn)
    _migrate_learning_pg(conn)
    _migrate_memory_pg(conn)
    _migrate_signals_pg(conn)
    _migrate_system_pg(conn)
    _migrate_strategies_pg(conn)
    _migrate_regime_matrix_pg(conn)
    _migrate_strategy_outcomes_pg(conn)
    _migrate_universe_pg(conn)
    _migrate_screener_pg(conn)
    _migrate_coordination_pg(conn)
    _migrate_research_wip_pg(conn)
    _migrate_conversations_pg(conn)
    _migrate_attribution_pg(conn)
    _migrate_equity_alerts_pg(conn)
    _migrate_market_data_pg(conn)
    _migrate_analytics_pg(conn)
    _migrate_research_queue_pg(conn)
    _migrate_loop_context_pg(conn)
    _migrate_bugs_pg(conn)
    _migrate_ml_pipeline_pg(conn)
    _migrate_capital_allocation_pg(conn)
    _migrate_risk_monitoring_pg(conn)
    _migrate_stat_arb_pg(conn)
    _migrate_tool_search_metrics_pg(conn)
    _migrate_trade_quality_pg(conn)
    _migrate_market_holidays_pg(conn)
    _migrate_signal_history_pg(conn)
    _migrate_signal_ic_pg(conn)
    _migrate_pnl_attribution_pg(conn)
    _migrate_regime_state_pg(conn)
    _migrate_institutional_gaps_pg(conn)
    _migrate_ewf_pg(conn)
    _migrate_hnsw_index_pg(conn)
    _migrate_phase4_coordination_pg(conn)


def downgrade() -> None:
    # WARNING: destroys all data. Never run in production.
    # Drop tables in reverse dependency order.
    from sqlalchemy import text

    bind = op.get_bind()
    result = bind.execute(text(
        "SELECT tablename FROM pg_tables WHERE schemaname = 'public' "
        "AND tablename != 'alembic_version'"
    ))
    tables = [row[0] for row in result]
    for table in tables:
        op.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
