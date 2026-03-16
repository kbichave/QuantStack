# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
CrewAI Trading Crew - Hierarchical Multi-Agent Trading System.

Architecture:
    SuperTrader → Assistant → Pod Managers → ICs

- ICs (Individual Contributors): Fetch data, compute metrics, return RAW outputs
- Pod Managers: Coordinate ICs, compile findings, forward to Assistant
- Assistant: Synthesizes all pod outputs into 1-pager for SuperTrader
- SuperTrader: Consumes only the 1-pager, makes final decisions

All agents have reasoning enabled. NO FALLBACKS - agents produce output or fail.

Agent prompts are loaded from JSON files in prompts/ directory.

Usage:
    from quant_pod.crews import TradingCrew

    crew = TradingCrew()
    result = crew.crew().kickoff(inputs={
        "symbol": "SPY",
        "current_date": date.today(),
        "regime": {"trend": "bullish", "volatility": "normal"},
        "portfolio": {...},
        "historical_context": "...",
    })
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

from loguru import logger

from quant_pod.crewai_compat import (
    Agent,
    BaseAgent,
    Crew,
    CrewBase,
    Process,
    Task,
    after_kickoff,
    agent,
    before_kickoff,
    crew,
    task,
)
from quant_pod.crews.assembler import CrewAssembler, PodSelection
from quant_pod.crews.schemas import (
    DailyBrief,
    TaskEnvelope,
    TradeDecision,
)
from quant_pod.crews.tools import (
    # RL Tools
    RL_TOOLS_AVAILABLE,
    analyze_liquidity_tool,
    analyze_option_structure_tool,
    analyze_volume_profile_tool,
    check_risk_limits_tool,
    compute_all_features_tool,
    compute_alpha_decay_tool,
    compute_greeks_tool,
    compute_implied_vol_tool,
    # Technical Analysis
    compute_indicators_tool,
    compute_information_coefficient_tool,
    compute_max_drawdown_tool,
    compute_option_chain_tool,
    compute_portfolio_stats_tool,
    compute_position_size_tool,
    # Risk
    compute_var_tool,
    # Market Data
    fetch_market_data_tool,
    # Trade
    generate_trade_template_tool,
    # Calendar
    get_event_calendar_tool,
    get_market_regime_snapshot_tool,
    get_rl_tools,
    get_symbol_snapshot_tool,
    get_trading_calendar_tool,
    list_stored_symbols_tool,
    load_market_data_tool,
    # Options
    price_option_tool,
    # Statistical
    run_adf_test_tool,
    score_trade_structure_tool,
    simulate_trade_outcome_tool,
    stress_test_portfolio_tool,
    validate_trade_tool,
)
from quant_pod.llm_config import get_llm_for_agent, log_llm_config_summary
from quant_pod.prompts import get_prompt_loader

# =============================================================================
# TOOL IMPORTS
# =============================================================================
from quant_pod.tools.alphavantage_tools import (
    fetch_company_overview_tool,
    fetch_news_sentiment_tool,
    fetch_upcoming_earnings_tool,
)
from quant_pod.tools.memory_tools import (
    get_recent_memory_tool,
    search_memory_tool,
    store_memory_tool,
)
from quant_pod.tools.options_flow_tools import OptionsFlowTool, PutCallRatioTool

# =============================================================================
# TOOL REGISTRY
# =============================================================================

TOOL_REGISTRY = {
    # Market Data
    "fetch_market_data": fetch_market_data_tool,
    "load_market_data": load_market_data_tool,
    "list_stored_symbols": list_stored_symbols_tool,
    "get_symbol_snapshot": get_symbol_snapshot_tool,
    # Technical Analysis
    "compute_indicators": compute_indicators_tool,
    "compute_all_features": compute_all_features_tool,
    "get_market_regime_snapshot": get_market_regime_snapshot_tool,
    "analyze_volume_profile": analyze_volume_profile_tool,
    # Risk
    "compute_var": compute_var_tool,
    "compute_position_size": compute_position_size_tool,
    "check_risk_limits": check_risk_limits_tool,
    "stress_test_portfolio": stress_test_portfolio_tool,
    "compute_max_drawdown": compute_max_drawdown_tool,
    "compute_portfolio_stats": compute_portfolio_stats_tool,
    "analyze_liquidity": analyze_liquidity_tool,
    # Statistical
    "run_adf_test": run_adf_test_tool,
    "compute_information_coefficient": compute_information_coefficient_tool,
    "compute_alpha_decay": compute_alpha_decay_tool,
    # Options
    "price_option": price_option_tool,
    "compute_greeks": compute_greeks_tool,
    "compute_implied_vol": compute_implied_vol_tool,
    "analyze_option_structure": analyze_option_structure_tool,
    "compute_option_chain": compute_option_chain_tool,
    # Calendar
    "get_event_calendar": get_event_calendar_tool,
    "get_trading_calendar": get_trading_calendar_tool,
    # Trade
    "generate_trade_template": generate_trade_template_tool,
    "validate_trade": validate_trade_tool,
    "score_trade_structure": score_trade_structure_tool,
    "simulate_trade_outcome": simulate_trade_outcome_tool,
    # Alpha Vantage — news, earnings, fundamentals
    "fetch_news_sentiment": fetch_news_sentiment_tool,
    "fetch_upcoming_earnings": fetch_upcoming_earnings_tool,
    "fetch_company_overview": fetch_company_overview_tool,
    # Options flow — UOA detection
    "options_flow_analysis": lambda: OptionsFlowTool(),
    "put_call_ratio": lambda: PutCallRatioTool(),
    # Memory (Mem0 — cross-session observations and regime context)
    "search_memory": search_memory_tool,
    "store_memory": store_memory_tool,
    "get_recent_memory": get_recent_memory_tool,
}

# Agent ordering for deterministic roster construction
IC_AGENT_ORDER = [
    "data_ingestion_ic",
    "market_snapshot_ic",
    "regime_detector_ic",
    "trend_momentum_ic",
    "volatility_ic",
    "structure_levels_ic",
    "statarb_ic",
    "options_vol_ic",
    "risk_limits_ic",
    "calendar_events_ic",
    # Alpha signals ICs — run last in wave (serialize Alpha Vantage calls)
    "news_sentiment_ic",
    "options_flow_ic",
    "fundamentals_ic",
]

POD_MANAGER_ORDER = [
    "data_pod_manager",
    "market_monitor_pod_manager",
    "technicals_pod_manager",
    "quant_pod_manager",
    "risk_pod_manager",
    "alpha_signals_pod_manager",
]


def get_tools_for_agent(tool_names: list[str]) -> list:
    """Get tool instances for an agent based on tool names from config."""
    tools = []
    for name in tool_names:
        if name in TOOL_REGISTRY:
            tools.append(TOOL_REGISTRY[name]())
        else:
            logger.warning(f"Tool not found in registry: {name}")
    return tools


# =============================================================================
# TRADING CREW - Hierarchical Agent System
# =============================================================================


@CrewBase
class TradingCrew:
    """
    CrewAI-native Hierarchical Trading Crew.

    Orchestrates agents through a hierarchical workflow:

    Layer 1 - ICs (Individual Contributors):
        - data_ingestion_ic
        - market_snapshot_ic, regime_detector_ic
        - trend_momentum_ic, volatility_ic, structure_levels_ic
        - statarb_ic, options_vol_ic
        - risk_limits_ic, calendar_events_ic

    Layer 2 - Pod Managers:
        - data_pod_manager
        - market_monitor_pod_manager
        - technicals_pod_manager
        - quant_pod_manager
        - risk_pod_manager

    Layer 3 - Assistant:
        - trading_assistant

    Layer 4 - SuperTrader:
        - super_trader

    Agent prompts loaded from prompts/*.json files.
    NO FALLBACKS - All agents must produce output or explicitly fail.
    """

    # Type hints for CrewBase
    agents: list[BaseAgent]
    tasks: list[Task]

    # Tasks config still from YAML (task descriptions stay centralized)
    tasks_config = "config/tasks.yaml"

    def __init__(self):
        """Initialize the trading crew."""
        # Load prompt configs
        self._prompt_loader = get_prompt_loader()
        self._assembler = CrewAssembler()
        self._last_roster: PodSelection | None = None

        # Verify tasks config exists
        config_dir = Path(__file__).parent / "config"
        if not (config_dir / "tasks.yaml").exists():
            raise FileNotFoundError(f"tasks.yaml not found in {config_dir}")

        logger.info("TradingCrew initialized with JSON prompts")
        log_llm_config_summary()

    def _create_agent_from_config(self, name: str, tools: list = None) -> Agent:
        """
        Create an Agent from JSON config.

        Args:
            name: Agent name matching JSON filename
            tools: Optional list of tools to override config

        Returns:
            Configured Agent instance
        """
        config = self._prompt_loader.load_agent(name)
        settings = config.get("settings", {})

        # Get tools from config if not provided
        if tools is None:
            tool_names = config.get("tools", [])
            tools = get_tools_for_agent(tool_names)

        return Agent(
            role=config.get("role", ""),
            goal=config.get("goal", ""),
            backstory=config.get("backstory", ""),
            llm=get_llm_for_agent(name),
            verbose=settings.get("verbose", True),
            allow_delegation=settings.get("allow_delegation", False),
            max_iter=settings.get("max_iter", 20),
            respect_context_window=settings.get("respect_context_window", True),
            tools=tools,
        )

    def _ic_agent_factories(self) -> dict[str, Callable[[], Agent]]:
        return {
            "data_ingestion_ic": self.data_ingestion_ic,
            "market_snapshot_ic": self.market_snapshot_ic,
            "regime_detector_ic": self.regime_detector_ic,
            "trend_momentum_ic": self.trend_momentum_ic,
            "volatility_ic": self.volatility_ic,
            "structure_levels_ic": self.structure_levels_ic,
            "statarb_ic": self.statarb_ic,
            "options_vol_ic": self.options_vol_ic,
            "risk_limits_ic": self.risk_limits_ic,
            "calendar_events_ic": self.calendar_events_ic,
            "news_sentiment_ic": self.news_sentiment_ic,
            "options_flow_ic": self.options_flow_ic,
            "fundamentals_ic": self.fundamentals_ic,
        }

    def _pod_manager_factories(self) -> dict[str, Callable[[], Agent]]:
        return {
            "data_pod_manager": self.data_pod_manager,
            "market_monitor_pod_manager": self.market_monitor_pod_manager,
            "technicals_pod_manager": self.technicals_pod_manager,
            "quant_pod_manager": self.quant_pod_manager,
            "risk_pod_manager": self.risk_pod_manager,
            "alpha_signals_pod_manager": self.alpha_signals_pod_manager,
        }

    def _ic_task_factories(self) -> dict[str, Callable[[], Task]]:
        return {
            "data_ingestion_ic": self.fetch_data_task,
            "market_snapshot_ic": self.snapshot_task,
            "regime_detector_ic": self.regime_task,
            "trend_momentum_ic": self.trend_momentum_task,
            "volatility_ic": self.volatility_task,
            "structure_levels_ic": self.structure_task,
            "statarb_ic": self.statarb_task,
            "options_vol_ic": self.options_task,
            "risk_limits_ic": self.risk_limits_task,
            "calendar_events_ic": self.events_task,
            "news_sentiment_ic": self.news_sentiment_task,
            "options_flow_ic": self.options_flow_task,
            "fundamentals_ic": self.fundamentals_task,
        }

    def _pod_task_factories(self) -> dict[str, Callable[[], Task]]:
        return {
            "data_pod_manager": self.data_pod_compile_task,
            "market_monitor_pod_manager": self.market_monitor_compile_task,
            "technicals_pod_manager": self.technicals_compile_task,
            "quant_pod_manager": self.quant_compile_task,
            "risk_pod_manager": self.risk_compile_task,
            "alpha_signals_pod_manager": self.alpha_signals_compile_task,
        }

    def _assemble_roster(
        self,
        envelope: TaskEnvelope,
        llm_decider: Callable[[str], str] | None = None,
    ) -> PodSelection:
        roster = self._assembler.assemble(envelope=envelope, llm_decider=llm_decider)
        self._last_roster = roster
        return roster

    def _build_agents(self, roster: PodSelection, stop_at_assistant: bool = False) -> list[Agent]:
        agents: list[Agent] = []
        ic_factories = self._ic_agent_factories()
        pod_factories = self._pod_manager_factories()

        for name in IC_AGENT_ORDER:
            if name in roster.ic_agents and name in ic_factories:
                agents.append(ic_factories[name]())

        for name in POD_MANAGER_ORDER:
            if name in roster.pod_managers and name in pod_factories:
                agents.append(pod_factories[name]())

        agents.append(self.trading_assistant())
        if not stop_at_assistant:
            agents.append(self.super_trader())
        return agents

    def _build_tasks(self, roster: PodSelection, stop_at_assistant: bool = False) -> list[Task]:
        tasks: list[Task] = []
        ic_task_factories = self._ic_task_factories()
        pod_task_factories = self._pod_task_factories()

        for name in IC_AGENT_ORDER:
            if name in roster.ic_agents and name in ic_task_factories:
                tasks.append(ic_task_factories[name]())

        for name in POD_MANAGER_ORDER:
            if name in roster.pod_managers and name in pod_task_factories:
                tasks.append(pod_task_factories[name]())

        tasks.append(self.assistant_synthesis_task())
        if not stop_at_assistant:
            tasks.append(self.trade_decision_task())
        return tasks

    # =========================================================================
    # HOOKS - Pre/Post Kickoff Processing
    # =========================================================================

    @before_kickoff
    def prepare_inputs(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Prepare and validate inputs before crew kickoff."""
        logger.info(
            "TradingCrew kickoff",
            extra={
                "symbol": inputs.get("symbol", "UNKNOWN"),
                "asset_class": inputs.get("asset_class"),
            },
        )

        envelope = TaskEnvelope.from_inputs(inputs)
        inputs["task_envelope"] = envelope.model_dump()
        inputs.setdefault("asset_class", envelope.asset_class)
        inputs.setdefault("instrument_type", envelope.instrument_type)
        inputs.setdefault("task_intent", envelope.task_intent)
        inputs["task_scope"] = (
            f"{envelope.asset_class}/{envelope.instrument_type}:{envelope.task_intent}"
        )

        # Set defaults — always stringify: CrewAI template interpolation rejects date objects
        if "current_date" not in inputs or inputs["current_date"] is None:
            inputs["current_date"] = str(date.today())
        else:
            inputs["current_date"] = str(inputs["current_date"])

        if "regime" not in inputs or inputs["regime"] is None:
            inputs["regime"] = {
                "trend": "unknown",
                "volatility": "normal",
                "confidence": 0.5,
            }

        # Convert regime dict to string for prompts
        regime = inputs.get("regime", {})
        inputs["regime_str"] = (
            f"Trend: {regime.get('trend', 'unknown')}, "
            f"Volatility: {regime.get('volatility', 'normal')}, "
            f"Confidence: {regime.get('confidence', 0.5):.0%}"
        )

        inputs.setdefault("portfolio", {})
        inputs.setdefault("historical_context", "")

        logger.debug(f"Prepared inputs: {list(inputs.keys())}")
        return inputs

    @after_kickoff
    def process_result(self, result: Any) -> Any:
        """Process crew result after kickoff."""
        logger.info("TradingCrew completed")
        if hasattr(result, "raw"):
            logger.info(f"Crew result: {str(result.raw)[:500]}...")
        return result

    # =========================================================================
    # LAYER 1: IC AGENTS - Individual Contributors
    # =========================================================================

    @agent
    def data_ingestion_ic(self) -> Agent:
        """Data ingestion specialist IC."""
        return self._create_agent_from_config("data_ingestion_ic")

    @agent
    def market_snapshot_ic(self) -> Agent:
        """Market snapshot specialist IC."""
        return self._create_agent_from_config("market_snapshot_ic")

    @agent
    def regime_detector_ic(self) -> Agent:
        """Regime detection specialist IC."""
        return self._create_agent_from_config("regime_detector_ic")

    @agent
    def trend_momentum_ic(self) -> Agent:
        """Trend and momentum analyst IC."""
        return self._create_agent_from_config("trend_momentum_ic")

    @agent
    def volatility_ic(self) -> Agent:
        """Volatility analyst IC."""
        return self._create_agent_from_config("volatility_ic")

    @agent
    def structure_levels_ic(self) -> Agent:
        """Support/resistance analyst IC."""
        return self._create_agent_from_config("structure_levels_ic")

    @agent
    def statarb_ic(self) -> Agent:
        """Statistical analysis specialist IC."""
        return self._create_agent_from_config("statarb_ic")

    @agent
    def options_vol_ic(self) -> Agent:
        """Options and vol surface analyst IC."""
        return self._create_agent_from_config("options_vol_ic")

    @agent
    def risk_limits_ic(self) -> Agent:
        """Risk metrics specialist IC."""
        return self._create_agent_from_config("risk_limits_ic")

    @agent
    def calendar_events_ic(self) -> Agent:
        """Event calendar specialist IC."""
        return self._create_agent_from_config("calendar_events_ic")

    @agent
    def news_sentiment_ic(self) -> Agent:
        """News sentiment and earnings risk specialist IC."""
        return self._create_agent_from_config("news_sentiment_ic")

    @agent
    def options_flow_ic(self) -> Agent:
        """Unusual options activity and institutional flow specialist IC."""
        return self._create_agent_from_config("options_flow_ic")

    @agent
    def fundamentals_ic(self) -> Agent:
        """Company fundamentals analyst IC."""
        return self._create_agent_from_config("fundamentals_ic")

    # =========================================================================
    # LAYER 2: POD MANAGER AGENTS
    # =========================================================================

    @agent
    def data_pod_manager(self) -> Agent:
        """Data ingestion pod manager."""
        return self._create_agent_from_config("data_pod_manager")

    @agent
    def market_monitor_pod_manager(self) -> Agent:
        """Market monitor pod manager."""
        return self._create_agent_from_config("market_monitor_pod_manager")

    @agent
    def technicals_pod_manager(self) -> Agent:
        """Technical analysis pod manager."""
        return self._create_agent_from_config("technicals_pod_manager")

    @agent
    def quant_pod_manager(self) -> Agent:
        """Quantitative analysis pod manager."""
        return self._create_agent_from_config("quant_pod_manager")

    @agent
    def risk_pod_manager(self) -> Agent:
        """Risk and execution pod manager."""
        agent = self._create_agent_from_config("risk_pod_manager")
        # Risk pod manager can use RL execution strategy tool — it approves trade execution
        # ICs do NOT get RL tools (preserves no-compounding-errors invariant)
        if RL_TOOLS_AVAILABLE:
            rl = get_rl_tools()
            if rl:
                agent.tools = (agent.tools or []) + rl
        return agent

    @agent
    def alpha_signals_pod_manager(self) -> Agent:
        """Alpha signals pod manager — synthesises news, flow, fundamentals, statarb, vol."""
        return self._create_agent_from_config("alpha_signals_pod_manager")

    # =========================================================================
    # LAYER 3: ASSISTANT AGENT
    # =========================================================================

    @agent
    def trading_assistant(self) -> Agent:
        """Trading assistant - synthesizes pod outputs into 1-pager."""
        return self._create_agent_from_config("trading_assistant")

    # =========================================================================
    # LAYER 4: SUPER TRADER AGENT
    # =========================================================================

    @agent
    def super_trader(self) -> Agent:
        """Portfolio manager and final decision maker."""
        agent = self._create_agent_from_config("super_trader")
        # SuperTrader is the final decision maker — all three RL tools are available to it.
        # ICs do NOT get RL tools (preserves no-compounding-errors invariant in agent_hardening.py)
        if RL_TOOLS_AVAILABLE:
            rl = get_rl_tools()
            if rl:
                agent.tools = (agent.tools or []) + rl
        return agent

    # =========================================================================
    # LAYER 1 TASKS: IC Tasks - Raw Data Gathering
    # =========================================================================

    # =========================================================================
    # LAYER 1 TASKS: IC Tasks - Raw Data Gathering
    # All IC tasks run asynchronously (async_execution=True) because they are
    # independent of each other.  data_ingestion_ic runs first and its output
    # is available as context; the remaining ICs can overlap with it because
    # they rely only on already-fetched cached data.
    # =========================================================================

    @task
    def fetch_data_task(self) -> Task:
        """Data fetching task — runs first (ICs depend on data being available)."""
        # Keep synchronous so data is on disk before the parallel IC wave starts.
        return Task(config=self.tasks_config["fetch_data_task"])

    @task
    def snapshot_task(self) -> Task:
        """Market snapshot task."""
        return Task(config=self.tasks_config["snapshot_task"], async_execution=True)

    @task
    def regime_task(self) -> Task:
        """Regime detection task."""
        return Task(config=self.tasks_config["regime_task"], async_execution=True)

    @task
    def trend_momentum_task(self) -> Task:
        """Trend and momentum task."""
        return Task(config=self.tasks_config["trend_momentum_task"], async_execution=True)

    @task
    def volatility_task(self) -> Task:
        """Volatility metrics task."""
        return Task(config=self.tasks_config["volatility_task"], async_execution=True)

    @task
    def structure_task(self) -> Task:
        """Structure/levels task."""
        return Task(config=self.tasks_config["structure_task"], async_execution=True)

    @task
    def statarb_task(self) -> Task:
        """Statistical arbitrage task."""
        return Task(config=self.tasks_config["statarb_task"], async_execution=True)

    @task
    def options_task(self) -> Task:
        """Options metrics task."""
        return Task(config=self.tasks_config["options_task"], async_execution=True)

    @task
    def risk_limits_task(self) -> Task:
        """Risk limits task."""
        return Task(config=self.tasks_config["risk_limits_task"], async_execution=True)

    @task
    def events_task(self) -> Task:
        """Calendar events task."""
        return Task(config=self.tasks_config["events_task"], async_execution=True)

    @task
    def news_sentiment_task(self) -> Task:
        """News sentiment task — synchronous to serialise Alpha Vantage calls (5/min rate limit)."""
        return Task(config=self.tasks_config["news_sentiment_task"])

    @task
    def options_flow_task(self) -> Task:
        """Options flow (UOA) task — synchronous to serialise Alpha Vantage calls."""
        return Task(config=self.tasks_config["options_flow_task"])

    @task
    def fundamentals_task(self) -> Task:
        """Company fundamentals task — synchronous to serialise Alpha Vantage calls."""
        return Task(config=self.tasks_config["fundamentals_task"])

    # =========================================================================
    # LAYER 2 TASKS: Pod Manager Tasks - Compilation
    # =========================================================================

    @task
    def data_pod_compile_task(self) -> Task:
        """Data pod compilation task."""
        return Task(config=self.tasks_config["data_pod_compile_task"])

    @task
    def market_monitor_compile_task(self) -> Task:
        """Market monitor pod compilation task."""
        return Task(config=self.tasks_config["market_monitor_compile_task"])

    @task
    def technicals_compile_task(self) -> Task:
        """Technicals pod compilation task."""
        return Task(config=self.tasks_config["technicals_compile_task"])

    @task
    def quant_compile_task(self) -> Task:
        """Quant pod compilation task."""
        return Task(config=self.tasks_config["quant_compile_task"])

    @task
    def risk_compile_task(self) -> Task:
        """Risk pod compilation task."""
        return Task(config=self.tasks_config["risk_compile_task"])

    @task
    def alpha_signals_compile_task(self) -> Task:
        """Alpha signals pod compilation task — news, flow, fundamentals, statarb, vol confluence."""
        return Task(config=self.tasks_config["alpha_signals_compile_task"])

    # =========================================================================
    # LAYER 3 TASK: Assistant Synthesis
    # =========================================================================

    @task
    def assistant_synthesis_task(self) -> Task:
        """Assistant synthesis task - produces 1-pager."""
        return Task(
            config=self.tasks_config["assistant_synthesis_task"],
            output_pydantic=DailyBrief,
        )

    # =========================================================================
    # LAYER 4 TASK: SuperTrader Decision
    # =========================================================================

    @task
    def trade_decision_task(self) -> Task:
        """Final trade decision task."""
        return Task(
            config=self.tasks_config["trade_decision_task"],
            output_pydantic=TradeDecision,
        )

    # =========================================================================
    # CREW - Main Orchestration
    # =========================================================================

    @crew
    def crew(
        self,
        task_envelope: Any | None = None,
        llm_decider: Callable[[str], str] | None = None,
        stop_at_assistant: bool = False,
    ) -> Crew:
        """
        Create and configure the hierarchical trading crew.

        Uses sequential process - tasks execute in defined order.
        Context flows through task dependencies defined in tasks.yaml.

        Args:
            stop_at_assistant: If True, the crew stops after the Trading
                Assistant's DailyBrief synthesis — the SuperTrader agent and
                trade_decision_task are excluded.  The crew output will be a
                validated DailyBrief instead of a TradeDecision.
        """
        envelope_inputs: dict[str, Any] = (
            {"task_envelope": task_envelope} if task_envelope is not None else {}
        )
        envelope = TaskEnvelope.from_inputs(envelope_inputs)
        roster = self._assemble_roster(envelope=envelope, llm_decider=llm_decider)
        agents = self._build_agents(roster, stop_at_assistant=stop_at_assistant)
        tasks = self._build_tasks(roster, stop_at_assistant=stop_at_assistant)

        logger.info(
            "Crew roster finalized",
            extra={
                "asset_class": envelope.asset_class,
                "instrument_type": envelope.instrument_type,
                "roster": roster.as_log_dict(),
            },
        )

        return Crew(
            agents=agents,
            tasks=tasks,
            process=Process.sequential,
            verbose=True,
            cache=True,
        )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def create_trading_crew() -> TradingCrew:
    """Factory function to create a TradingCrew instance."""
    return TradingCrew()


def run_trading_analysis(
    symbol: str,
    regime: dict[str, Any] | None = None,
    portfolio: dict | None = None,
    historical_context: str = "",
    current_date: date | None = None,
    task_envelope: Any | None = None,
) -> Any:
    """
    Convenience function to run trading analysis for a symbol.

    Args:
        symbol: Trading symbol (e.g., "SPY")
        regime: Optional regime dict with trend/volatility/confidence
        portfolio: Optional portfolio state dict
        historical_context: Optional historical context string
        current_date: Optional date (defaults to today)
        task_envelope: Optional pre-built TaskEnvelope for asset-class routing

    Returns:
        Crew execution result with TradeDecision
    """
    crew = TradingCrew()

    # Pass raw inputs — the @before_kickoff prepare_inputs hook handles all
    # normalization, envelope construction, and default-setting.
    inputs = {
        "symbol": symbol,
        "current_date": str(current_date or date.today()),
        "regime": regime or {"trend": "unknown", "volatility": "normal", "confidence": 0.5},
        "portfolio": portfolio or {},
        "historical_context": historical_context,
    }

    result = crew.crew(task_envelope=task_envelope).kickoff(inputs=inputs)
    return result


def run_analysis_only(
    symbol: str,
    regime: dict[str, Any] | None = None,
    portfolio: dict | None = None,
    historical_context: str = "",
    strategy_context: str = "",
    session_notes: str = "",
    current_date: date | None = None,
    task_envelope: Any | None = None,
) -> Any:
    """
    Run crew analysis and return DailyBrief without SuperTrader decision.

    Identical to run_trading_analysis but stops after the Trading Assistant's
    synthesis, returning a DailyBrief instead of a TradeDecision.

    Args:
        strategy_context: Active strategies from .claude/memory/strategy_registry.md
                          injected so the crew knows what strategies are live.
        session_notes: Recent session handoff notes from session_handoffs.md
                       injected so the crew has cross-session awareness.
    """
    crew = TradingCrew()
    inputs = {
        "symbol": symbol,
        "current_date": str(current_date or date.today()),
        "regime": regime or {"trend": "unknown", "volatility": "normal", "confidence": 0.5},
        "portfolio": portfolio or {},
        "historical_context": historical_context,
        "strategy_context": strategy_context,
        "session_notes": session_notes,
    }
    return crew.crew(task_envelope=task_envelope, stop_at_assistant=True).kickoff(inputs=inputs)


def list_available_agents() -> dict[str, list[str]]:
    """List all available agents by category."""
    return get_prompt_loader().list_all_agents()


__all__ = [
    "TradingCrew",
    "create_trading_crew",
    "run_trading_analysis",
    "run_analysis_only",
    "list_available_agents",
]
