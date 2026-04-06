"""Query modules for the TUI dashboard.

Each module contains typed query functions that accept a PgConnection
and return dataclasses. All functions degrade gracefully on error.
"""
from quantstack.tui.queries.agents import (
    AgentSkill,
    CalibrationRecord,
    CycleHistory,
    GraphActivity,
    PromptVersion,
    fetch_agent_skills,
    fetch_calibration,
    fetch_cycle_history,
    fetch_graph_activity,
    fetch_prompt_versions,
)
from quantstack.tui.queries.calendar import (
    EarningsEvent,
    fetch_earnings_calendar,
)
from quantstack.tui.queries.data_health import (
    DataFreshness,
    fetch_insider_freshness,
    fetch_macro_freshness,
    fetch_news_freshness,
    fetch_ohlcv_freshness,
    fetch_options_freshness,
    fetch_sentiment_freshness,
)
from quantstack.tui.queries.portfolio import (
    BenchmarkPoint,
    ClosedTrade,
    EquityPoint,
    EquitySummary,
    Position,
    StrategyPnl,
    SymbolPnl,
    fetch_benchmark,
    fetch_closed_trades,
    fetch_equity_curve,
    fetch_equity_summary,
    fetch_pnl_by_strategy,
    fetch_pnl_by_symbol,
    fetch_positions,
)
from quantstack.tui.queries.research import (
    AlphaProgram,
    Breakthrough,
    BugRecord,
    ConceptDrift,
    MlExperiment,
    ResearchQueueItem,
    ResearchWip,
    TradeReflection,
    fetch_alpha_programs,
    fetch_breakthroughs,
    fetch_bugs,
    fetch_concept_drift,
    fetch_ml_experiments,
    fetch_reflections,
    fetch_research_queue,
    fetch_research_wip,
)
from quantstack.tui.queries.risk import (
    EquityAlert,
    RiskEvent,
    RiskSnapshot,
    fetch_equity_alerts,
    fetch_risk_events,
    fetch_risk_snapshot,
)
from quantstack.tui.queries.signals import (
    Signal,
    SignalBrief,
    fetch_active_signals,
    fetch_signal_brief,
)
from quantstack.tui.queries.system import (
    AgentEvent,
    GraphCheckpoint,
    Heartbeat,
    RegimeState,
    fetch_agent_events,
    fetch_av_calls,
    fetch_graph_checkpoints,
    fetch_heartbeats,
    fetch_kill_switch,
    fetch_regime,
)

__all__ = [
    # system
    "fetch_kill_switch", "fetch_av_calls", "fetch_regime",
    "fetch_graph_checkpoints", "fetch_heartbeats", "fetch_agent_events",
    "RegimeState", "GraphCheckpoint", "Heartbeat", "AgentEvent",
    # portfolio
    "fetch_equity_summary", "fetch_positions", "fetch_closed_trades",
    "fetch_equity_curve", "fetch_benchmark", "fetch_pnl_by_strategy", "fetch_pnl_by_symbol",
    "EquitySummary", "Position", "ClosedTrade", "EquityPoint", "BenchmarkPoint",
    "StrategyPnl", "SymbolPnl",
    # data_health
    "fetch_ohlcv_freshness", "fetch_news_freshness", "fetch_sentiment_freshness",
    "fetch_options_freshness", "fetch_insider_freshness",
    "fetch_macro_freshness", "DataFreshness",
    # signals
    "fetch_active_signals", "fetch_signal_brief", "Signal", "SignalBrief",
    # calendar
    "fetch_earnings_calendar", "EarningsEvent",
    # agents
    "fetch_graph_activity", "fetch_cycle_history", "fetch_agent_skills",
    "fetch_calibration", "fetch_prompt_versions",
    "GraphActivity", "CycleHistory", "AgentSkill", "CalibrationRecord", "PromptVersion",
    # research
    "fetch_research_wip", "fetch_research_queue", "fetch_ml_experiments",
    "fetch_alpha_programs", "fetch_breakthroughs", "fetch_reflections",
    "fetch_bugs", "fetch_concept_drift",
    "ResearchWip", "ResearchQueueItem", "MlExperiment", "AlphaProgram",
    "Breakthrough", "TradeReflection", "BugRecord", "ConceptDrift",
    # risk
    "fetch_risk_snapshot", "fetch_risk_events", "fetch_equity_alerts",
    "RiskSnapshot", "RiskEvent", "EquityAlert",
]
