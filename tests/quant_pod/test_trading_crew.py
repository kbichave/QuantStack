# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Tests for TradingCrew CrewAI Hierarchical Implementation with JSON Prompts.

These tests verify:
- TradingCrew initialization and JSON prompt loading
- Schema validation for inputs and outputs
- Hierarchical agent creation (ICs → Pod Managers → Assistant → SuperTrader)
- Task creation and context flow
- Prompt loader functionality

Note: Tests that require API keys are skipped unless the key is set.
"""

import os
import json
import pytest
from datetime import date
from unittest.mock import MagicMock, patch
from pathlib import Path

# Import crew and schemas
from quant_pod.crews import (
    TradingCrew,
    AnalysisNote,
    PodResearchNote,
    DailyBrief,
    SymbolBrief,
    TradeDecision,
    RiskVerdict,
    KeyLevel,
    list_available_agents,
)
from quant_pod.prompts import (
    PromptLoader,
    load_agent_config,
    load_all_ics,
    load_all_pod_managers,
)

# Check if OpenAI API key is available
HAS_OPENAI_KEY = bool(os.environ.get("OPENAI_API_KEY"))
requires_openai_key = pytest.mark.skipif(
    not HAS_OPENAI_KEY, reason="OPENAI_API_KEY not set"
)


class TestSchemas:
    """Test Pydantic schema validation."""

    def test_key_level_schema(self):
        """Test KeyLevel schema."""
        level = KeyLevel(
            price=100.0, level_type="support", strength=0.8, source="SMA_200"
        )
        assert level.price == 100.0
        assert level.level_type == "support"
        assert level.strength == 0.8

    def test_analysis_note_schema(self):
        """Test AnalysisNote schema."""
        note = AnalysisNote(
            symbol="SPY",
            analyst_name="trend_momentum_ic",
            analysis_type="trend",
            current_state="uptrend with pullback",
            bias="bullish",
            bias_conviction=0.75,
            observations=["SMA 20 > SMA 50", "ADX at 28"],
            risk_factors=["Approaching resistance"],
            rationale="Strong trend with healthy pullback",
            technical_metrics={"rsi_14": 55.0, "adx_14": 28.0},
        )
        assert note.symbol == "SPY"
        assert note.bias == "bullish"
        assert note.bias_conviction == 0.75

    def test_trade_decision_schema(self):
        """Test TradeDecision schema."""
        decision = TradeDecision(
            action="buy",
            symbol="SPY",
            confidence=0.8,
            position_size="half",
            holding_period="swing",
            entry_type="limit",
            limit_price=450.0,
            stop_loss=440.0,
            take_profit=470.0,
            reasoning="Strong trend alignment with risk-managed entry",
            signals_used=["trend", "momentum"],
        )
        assert decision.action == "buy"
        assert decision.confidence == 0.8
        assert decision.stop_loss == 440.0

    def test_risk_verdict_schema(self):
        """Test RiskVerdict schema."""
        verdict = RiskVerdict(
            status="SCALE",
            scale_factor=0.5,
            risk_score=0.6,
            reasoning="Position size exceeds volatility-adjusted limits",
            concerns=["High ATR", "Sector concentration"],
            recommendations=["Reduce size by 50%"],
        )
        assert verdict.status == "SCALE"
        assert verdict.scale_factor == 0.5

    def test_daily_brief_schema(self):
        """Test DailyBrief schema."""
        brief = DailyBrief(
            date=date.today(),
            market_overview="Markets showing strength",
            market_bias="bullish",
            market_conviction=0.7,
            risk_environment="normal",
            top_opportunities=["SPY", "QQQ"],
            key_risks=["Fed meeting next week"],
            strategic_notes="Focus on large-cap tech",
        )
        assert brief.market_bias == "bullish"
        assert len(brief.top_opportunities) == 2


class TestPromptLoader:
    """Test JSON prompt loading functionality."""

    def test_prompts_directory_exists(self):
        """Test that prompts directory exists."""
        prompts_dir = Path(__file__).parent.parent / "src/quant_pod/prompts"
        assert prompts_dir.exists()

    def test_ic_json_files_exist(self):
        """Test that IC JSON config files exist."""
        prompts_dir = Path(__file__).parent.parent / "src/quant_pod/prompts"
        ics_dir = prompts_dir / "ics"

        # Check some IC files exist
        assert (ics_dir / "data/data_ingestion_ic.json").exists()
        assert (ics_dir / "market_monitor/market_snapshot_ic.json").exists()
        assert (ics_dir / "technicals/trend_momentum_ic.json").exists()

    def test_pod_manager_json_files_exist(self):
        """Test that pod manager JSON config files exist."""
        prompts_dir = Path(__file__).parent.parent / "src/quant_pod/prompts"
        pm_dir = prompts_dir / "pod_managers"

        assert (pm_dir / "data_pod_manager.json").exists()
        assert (pm_dir / "technicals_pod_manager.json").exists()

    def test_assistant_json_exists(self):
        """Test that assistant JSON config exists."""
        prompts_dir = Path(__file__).parent.parent / "src/quant_pod/prompts"
        assert (prompts_dir / "assistant/trading_assistant.json").exists()

    def test_supertrader_json_exists(self):
        """Test that supertrader JSON config exists."""
        prompts_dir = Path(__file__).parent.parent / "src/quant_pod/prompts"
        assert (prompts_dir / "supertrader/super_trader.json").exists()

    def test_load_agent_config(self):
        """Test loading a single agent config."""
        config = load_agent_config("data_ingestion_ic")

        assert "name" in config
        assert "role" in config
        assert "goal" in config
        assert "backstory" in config
        assert "settings" in config
        assert config["name"] == "data_ingestion_ic"

    def test_load_all_ics(self):
        """Test loading all IC configs."""
        ics = load_all_ics()

        assert len(ics) == 10  # 10 ICs
        assert "data_ingestion_ic" in ics
        assert "trend_momentum_ic" in ics

    def test_load_all_pod_managers(self):
        """Test loading all pod manager configs."""
        managers = load_all_pod_managers()

        assert len(managers) == 5  # 5 pod managers
        assert "data_pod_manager" in managers
        assert "technicals_pod_manager" in managers

    def test_list_all_agents(self):
        """Test listing all available agents."""
        agents = list_available_agents()

        assert "ics" in agents
        assert "pod_managers" in agents
        assert "assistant" in agents
        assert "supertrader" in agents

        assert len(agents["ics"]) == 10
        assert len(agents["pod_managers"]) == 5
        assert len(agents["assistant"]) == 1
        assert len(agents["supertrader"]) == 1

    def test_json_schema_has_required_fields(self):
        """Test that JSON configs have required fields."""
        loader = PromptLoader()

        # Check a sample IC
        config = loader.load_agent("data_ingestion_ic")
        assert "role" in config
        assert "goal" in config
        assert "backstory" in config
        assert "settings" in config

        settings = config["settings"]
        assert "llm" in settings
        assert "reasoning" in settings
        assert "verbose" in settings


class TestTradingCrewInit:
    """Test TradingCrew initialization."""

    @requires_openai_key
    def test_crew_init(self):
        """Test basic crew initialization."""
        crew = TradingCrew()
        assert crew is not None
        assert crew._prompt_loader is not None

    def test_tasks_config_exists(self):
        """Test that tasks YAML config exists."""
        config_dir = Path(__file__).parent.parent / "src/quant_pod/crews/config"
        assert (config_dir / "tasks.yaml").exists()


class TestTradingCrewHooks:
    """Test before/after kickoff hooks."""

    @requires_openai_key
    def test_prepare_inputs_adds_defaults(self):
        """Test that prepare_inputs adds default values."""
        crew = TradingCrew()

        inputs = {"symbol": "SPY"}
        processed = crew.prepare_inputs(inputs)

        assert "current_date" in processed
        assert "regime" in processed
        assert "portfolio" in processed
        assert "regime_str" in processed

    @requires_openai_key
    def test_prepare_inputs_preserves_provided(self):
        """Test that prepare_inputs preserves provided values."""
        crew = TradingCrew()

        regime = {"trend": "bullish", "volatility": "low", "confidence": 0.9}
        inputs = {
            "symbol": "SPY",
            "current_date": date(2024, 1, 15),
            "regime": regime,
        }
        processed = crew.prepare_inputs(inputs)

        assert processed["current_date"] == date(2024, 1, 15)
        assert processed["regime"]["trend"] == "bullish"


class TestTradingCrewICAgents:
    """Test IC (Individual Contributor) agent creation."""

    @requires_openai_key
    def test_ic_agents_created(self):
        """Test that all IC agents are created."""
        crew = TradingCrew()

        # Test IC agent methods exist and return agents
        assert crew.data_ingestion_ic() is not None
        assert crew.market_snapshot_ic() is not None
        assert crew.regime_detector_ic() is not None
        assert crew.trend_momentum_ic() is not None
        assert crew.volatility_ic() is not None
        assert crew.structure_levels_ic() is not None
        assert crew.statarb_ic() is not None
        assert crew.options_vol_ic() is not None
        assert crew.risk_limits_ic() is not None
        assert crew.calendar_events_ic() is not None


class TestTradingCrewPodManagers:
    """Test Pod Manager agent creation."""

    @requires_openai_key
    def test_pod_manager_agents_created(self):
        """Test that all Pod Manager agents are created."""
        crew = TradingCrew()

        # Test Pod Manager agent methods
        assert crew.data_pod_manager() is not None
        assert crew.market_monitor_pod_manager() is not None
        assert crew.technicals_pod_manager() is not None
        assert crew.quant_pod_manager() is not None
        assert crew.risk_pod_manager() is not None


class TestTradingCrewTopLevel:
    """Test Assistant and SuperTrader agent creation."""

    @requires_openai_key
    def test_assistant_and_super_trader_created(self):
        """Test that Assistant and SuperTrader are created."""
        crew = TradingCrew()

        # Test top-level agents
        assert crew.trading_assistant() is not None
        assert crew.super_trader() is not None


class TestTradingCrewTasks:
    """Test task creation."""

    @requires_openai_key
    def test_ic_tasks_created(self):
        """Test that all IC tasks are created."""
        crew = TradingCrew()

        # IC tasks
        assert crew.fetch_data_task() is not None
        assert crew.snapshot_task() is not None
        assert crew.regime_task() is not None
        assert crew.trend_momentum_task() is not None
        assert crew.volatility_task() is not None
        assert crew.structure_task() is not None
        assert crew.risk_limits_task() is not None
        assert crew.events_task() is not None

    @requires_openai_key
    def test_pod_manager_tasks_created(self):
        """Test that all Pod Manager tasks are created."""
        crew = TradingCrew()

        # Pod Manager tasks
        assert crew.data_pod_compile_task() is not None
        assert crew.market_monitor_compile_task() is not None
        assert crew.technicals_compile_task() is not None
        assert crew.quant_compile_task() is not None
        assert crew.risk_compile_task() is not None

    @requires_openai_key
    def test_synthesis_and_decision_tasks_created(self):
        """Test that Assistant and SuperTrader tasks are created."""
        crew = TradingCrew()

        # Top-level tasks
        assert crew.assistant_synthesis_task() is not None
        assert crew.trade_decision_task() is not None


class TestTradingCrewCrew:
    """Test crew creation."""

    @requires_openai_key
    def test_crew_created(self):
        """Test that crew is created with agents and tasks."""
        crew_instance = TradingCrew()
        crew = crew_instance.crew()

        assert crew is not None
        assert len(crew.agents) > 0
        assert len(crew.tasks) > 0


class TestTradingCrewHierarchy:
    """Test the hierarchical structure."""

    @requires_openai_key
    def test_hierarchy_layer_count(self):
        """Test that we have the correct hierarchy layers."""
        crew = TradingCrew()

        # Layer 1: ICs - should be 10
        ic_agents = [
            crew.data_ingestion_ic,
            crew.market_snapshot_ic,
            crew.regime_detector_ic,
            crew.trend_momentum_ic,
            crew.volatility_ic,
            crew.structure_levels_ic,
            crew.statarb_ic,
            crew.options_vol_ic,
            crew.risk_limits_ic,
            crew.calendar_events_ic,
        ]
        assert len(ic_agents) == 10

        # Layer 2: Pod Managers - should be 5
        pm_agents = [
            crew.data_pod_manager,
            crew.market_monitor_pod_manager,
            crew.technicals_pod_manager,
            crew.quant_pod_manager,
            crew.risk_pod_manager,
        ]
        assert len(pm_agents) == 5

        # Layer 3: Assistant - should be 1
        assert crew.trading_assistant is not None

        # Layer 4: SuperTrader - should be 1
        assert crew.super_trader is not None


class TestTradingDayFlowIntegration:
    """Test integration with TradingDayFlow."""

    @requires_openai_key
    def test_flow_initialization(self):
        """Test that flow initializes correctly."""
        from quant_pod.flows.trading_day_flow import TradingDayFlow

        flow = TradingDayFlow()
        assert flow._trading_crew is None  # Lazy loaded
        assert flow.trading_crew is not None  # Created on access


class TestNoFallbacks:
    """Test that deprecated functions raise errors (no fallbacks)."""

    def test_create_all_pods_raises(self):
        """Test that create_all_pods raises NotImplementedError."""
        from quant_pod.agents import create_all_pods

        with pytest.raises(NotImplementedError):
            create_all_pods()

    def test_get_super_trader_raises(self):
        """Test that get_super_trader raises NotImplementedError."""
        from quant_pod.agents import get_super_trader

        with pytest.raises(NotImplementedError):
            get_super_trader()


# Run tests with pytest
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
