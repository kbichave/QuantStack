# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Knowledge Tools - CrewAI tools for reading/writing shared knowledge.

Provides tools for agents to share observations, signals, and scenarios.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Type

from loguru import logger
from pydantic import BaseModel, Field
from quant_pod.crewai_compat import BaseTool

from quant_pod.knowledge.store import KnowledgeStore
from quant_pod.knowledge.models import (
    MarketObservation,
    TradingSignal,
    TradeDirection,
    WaveScenario,
    WavePosition,
)


# Global knowledge store instance
_store: Optional[KnowledgeStore] = None


def get_store() -> KnowledgeStore:
    """Get or create the global knowledge store instance."""
    global _store
    if _store is None:
        _store = KnowledgeStore()
    return _store


# =============================================================================
# INPUT SCHEMAS
# =============================================================================


class SaveObservationInput(BaseModel):
    """Input for save_observation tool."""

    symbol: str = Field(..., description="Symbol observed")
    observation_type: str = Field(
        ..., description="Type: PRICE_ALERT, VOLUME_SPIKE, GAP, TECHNICAL"
    )
    current_price: float = Field(..., description="Current price")
    alert_message: str = Field(..., description="Description of the observation")
    price_change_pct: Optional[float] = Field(
        None, description="Price change percentage"
    )
    volume_ratio: Optional[float] = Field(None, description="Volume vs average")
    severity: str = Field("INFO", description="INFO, WARNING, ALERT, CRITICAL")


class GetObservationsInput(BaseModel):
    """Input for get_observations tool."""

    symbol: Optional[str] = Field(None, description="Filter by symbol")
    hours: int = Field(24, description="Lookback hours")
    unprocessed_only: bool = Field(False, description="Only unprocessed observations")


class SaveSignalInput(BaseModel):
    """Input for save_signal tool."""

    symbol: str = Field(..., description="Symbol for signal")
    direction: str = Field(..., description="LONG or SHORT")
    signal_type: str = Field(
        ..., description="Signal type: WAVE_TARGET, REGIME_SHIFT, TECHNICAL"
    )
    strength: float = Field(0.5, description="Signal strength 0-1")
    confidence: float = Field(0.5, description="Confidence 0-1")
    rationale: str = Field(..., description="Why this signal was generated")
    entry_price: Optional[float] = Field(None, description="Suggested entry price")
    target_price: Optional[float] = Field(None, description="Target price")
    stop_loss: Optional[float] = Field(None, description="Stop loss level")


class GetSignalsInput(BaseModel):
    """Input for get_signals tool."""

    symbol: Optional[str] = Field(None, description="Filter by symbol")
    unprocessed_only: bool = Field(False, description="Only unprocessed signals")


class SaveWaveScenarioInput(BaseModel):
    """Input for save_wave_scenario tool."""

    symbol: str = Field(..., description="Symbol")
    timeframe: str = Field(..., description="Timeframe: 1h, 4h, daily, weekly")
    wave_position: str = Field(
        ..., description="Current wave: WAVE_1, WAVE_2, ..., WAVE_C"
    )
    wave_degree: str = Field(
        ..., description="Wave degree: Primary, Intermediate, Minor"
    )
    confidence: float = Field(0.5, description="Confidence 0-1")
    invalidation_level: float = Field(
        ..., description="Price that invalidates this count"
    )
    scenario_type: str = Field(..., description="BULLISH, BEARISH, NEUTRAL")
    description: str = Field(..., description="Scenario description")
    primary_target: Optional[float] = Field(None, description="Primary price target")
    secondary_target: Optional[float] = Field(None, description="Secondary target")


class GetWaveScenariosInput(BaseModel):
    """Input for get_wave_scenarios tool."""

    symbol: Optional[str] = Field(None, description="Filter by symbol")
    timeframe: Optional[str] = Field(None, description="Filter by timeframe")


# =============================================================================
# TOOL CLASSES
# =============================================================================


class SaveObservationTool(BaseTool):
    """Tool to save a market observation to knowledge store."""

    name: str = "save_observation"
    description: str = (
        "Save a market observation (price alert, volume spike, etc.) for other agents."
    )
    args_schema: Type[BaseModel] = SaveObservationInput

    def _run(
        self,
        symbol: str,
        observation_type: str,
        current_price: float,
        alert_message: str,
        price_change_pct: Optional[float] = None,
        volume_ratio: Optional[float] = None,
        severity: str = "INFO",
    ) -> str:
        """Save observation to knowledge store."""
        store = get_store()

        obs = MarketObservation(
            symbol=symbol.upper(),
            observation_type=observation_type,
            current_price=current_price,
            alert_message=alert_message,
            price_change_pct=price_change_pct,
            volume_ratio=volume_ratio,
            severity=severity,
            source_agent="agent_tool",
        )

        obs_id = store.save_observation(obs)

        return json.dumps(
            {
                "success": True,
                "observation_id": obs_id,
                "message": f"Observation saved: {alert_message}",
            }
        )


class GetObservationsTool(BaseTool):
    """Tool to get recent market observations."""

    name: str = "get_observations"
    description: str = "Get recent market observations from the knowledge store."
    args_schema: Type[BaseModel] = GetObservationsInput

    def _run(
        self,
        symbol: Optional[str] = None,
        hours: int = 24,
        unprocessed_only: bool = False,
    ) -> str:
        """Get observations from knowledge store."""
        store = get_store()

        observations = store.get_recent_observations(
            symbol=symbol.upper() if symbol else None,
            hours=hours,
            unprocessed_only=unprocessed_only,
        )

        return json.dumps(
            {
                "success": True,
                "count": len(observations),
                "observations": [obs.model_dump() for obs in observations],
            },
            default=str,
        )


class SaveSignalTool(BaseTool):
    """Tool to save a trading signal."""

    name: str = "save_signal"
    description: str = "Save a trading signal for other agents to process."
    args_schema: Type[BaseModel] = SaveSignalInput

    def _run(
        self,
        symbol: str,
        direction: str,
        signal_type: str,
        strength: float = 0.5,
        confidence: float = 0.5,
        rationale: str = "",
        entry_price: Optional[float] = None,
        target_price: Optional[float] = None,
        stop_loss: Optional[float] = None,
    ) -> str:
        """Save signal to knowledge store."""
        store = get_store()

        signal = TradingSignal(
            symbol=symbol.upper(),
            direction=TradeDirection(direction.upper()),
            signal_type=signal_type,
            strength=strength,
            confidence=confidence,
            rationale=rationale,
            entry_price=entry_price,
            target_price=target_price,
            stop_loss=stop_loss,
            source_agent="agent_tool",
        )

        signal_id = store.save_signal(signal)

        return json.dumps(
            {
                "success": True,
                "signal_id": signal_id,
                "message": f"Signal saved: {direction} {symbol} ({confidence:.0%} confidence)",
            }
        )


class GetSignalsTool(BaseTool):
    """Tool to get active trading signals."""

    name: str = "get_signals"
    description: str = "Get active trading signals from the knowledge store."
    args_schema: Type[BaseModel] = GetSignalsInput

    def _run(
        self,
        symbol: Optional[str] = None,
        unprocessed_only: bool = False,
    ) -> str:
        """Get signals from knowledge store."""
        store = get_store()

        signals = store.get_active_signals(
            symbol=symbol.upper() if symbol else None,
            unprocessed_only=unprocessed_only,
        )

        return json.dumps(
            {
                "success": True,
                "count": len(signals),
                "signals": [sig.model_dump() for sig in signals],
            },
            default=str,
        )


class SaveWaveScenarioTool(BaseTool):
    """Tool to save an Elliott Wave scenario."""

    name: str = "save_wave_scenario"
    description: str = (
        "Save an Elliott Wave scenario with targets and invalidation levels."
    )
    args_schema: Type[BaseModel] = SaveWaveScenarioInput

    def _run(
        self,
        symbol: str,
        timeframe: str,
        wave_position: str,
        wave_degree: str,
        confidence: float,
        invalidation_level: float,
        scenario_type: str,
        description: str,
        primary_target: Optional[float] = None,
        secondary_target: Optional[float] = None,
    ) -> str:
        """Save wave scenario to knowledge store."""
        store = get_store()

        scenario = WaveScenario(
            symbol=symbol.upper(),
            timeframe=timeframe,
            wave_position=WavePosition(wave_position.upper()),
            wave_degree=wave_degree,
            confidence=confidence,
            invalidation_level=invalidation_level,
            scenario_type=scenario_type,
            description=description,
            primary_target=primary_target,
            secondary_target=secondary_target,
            source_agent="wave_analyst",
        )

        scenario_id = store.save_wave_scenario(scenario)

        return json.dumps(
            {
                "success": True,
                "scenario_id": scenario_id,
                "message": f"Wave scenario saved: {symbol} {wave_position} targeting {primary_target}",
            }
        )


class GetWaveScenariosTool(BaseTool):
    """Tool to get active wave scenarios."""

    name: str = "get_wave_scenarios"
    description: str = "Get active Elliott Wave scenarios from the knowledge store."
    args_schema: Type[BaseModel] = GetWaveScenariosInput

    def _run(
        self,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
    ) -> str:
        """Get wave scenarios from knowledge store."""
        store = get_store()

        scenarios = store.get_active_wave_scenarios(
            symbol=symbol.upper() if symbol else None,
            timeframe=timeframe,
        )

        return json.dumps(
            {
                "success": True,
                "count": len(scenarios),
                "scenarios": [s.model_dump() for s in scenarios],
            },
            default=str,
        )


# Tool factory functions
def save_observation_tool() -> SaveObservationTool:
    """Get the save observation tool instance."""
    return SaveObservationTool()


def get_observations_tool() -> GetObservationsTool:
    """Get the get observations tool instance."""
    return GetObservationsTool()


def save_signal_tool() -> SaveSignalTool:
    """Get the save signal tool instance."""
    return SaveSignalTool()


def get_signals_tool() -> GetSignalsTool:
    """Get the get signals tool instance."""
    return GetSignalsTool()


def save_wave_scenario_tool() -> SaveWaveScenarioTool:
    """Get the save wave scenario tool instance."""
    return SaveWaveScenarioTool()


def get_wave_scenarios_tool() -> GetWaveScenariosTool:
    """Get the get wave scenarios tool instance."""
    return GetWaveScenariosTool()
