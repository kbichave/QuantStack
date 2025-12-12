# Copyright 2024 QuantArena Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Historical simulation configuration.

Defines all parameters for running a historical QuantArena simulation:
- Symbol universe
- Date range
- Capital and leverage limits
- Transaction costs
- Policy update frequency
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


@dataclass
class HistoricalConfig:
    """
    Configuration for historical QuantArena simulation.

    Attributes:
        symbols: List of logical symbols to trade (mapped via universe.py)
        start_date: Simulation start date (None = earliest available)
        end_date: Simulation end date (None = today)
        initial_equity: Starting capital in USD
        max_leverage: Maximum portfolio leverage (1.0 = no leverage)
        commission_per_share: Commission per share traded
        slippage_bps: Slippage in basis points
        policy_update_frequency: How often to update strategy weights
        active_pods: List of strategy pods to run
        max_position_pct: Maximum position size as % of equity
        max_drawdown_halt_pct: Halt trading if drawdown exceeds this
        db_path: Path to experience store DuckDB (None = default)

    Usage:
        config = HistoricalConfig(
            symbols=["SPY", "QQQ", "IWM", "WTI", "BRENT"],
            initial_equity=100_000,
        )
    """

    # Symbol universe (logical names, mapped in universe.py)
    symbols: List[str] = field(
        default_factory=lambda: ["SPY", "QQQ", "IWM", "WTI", "BRENT"]
    )

    # Date range
    start_date: Optional[str] = None  # YYYY-MM-DD or None for earliest
    end_date: Optional[str] = None  # YYYY-MM-DD or None for today

    # Capital settings
    initial_equity: float = 100_000.0
    max_leverage: float = 1.0  # 1.0 = no leverage (equity only)

    # Transaction costs
    commission_per_share: float = 0.005
    slippage_bps: float = 5.0

    # Policy/learning settings
    policy_update_frequency: Literal["monthly", "quarterly", "never"] = "monthly"
    enable_learning: bool = True  # Enable MetaOrchestrator, lesson injection, etc.

    # Multi-timeframe settings
    enable_mtf: bool = True  # Enable multi-timeframe analysis
    execution_timeframe: Literal["daily", "4h", "1h"] = "4h"  # Execution timeframe
    use_super_trader: bool = True  # Enable SuperTrader aggregator
    use_fast_llm: bool = False  # Use gpt-4o-mini for faster execution
    use_llm: bool = True  # LLM agents required - no rule-based fallbacks allowed

    # CrewAI integration
    use_crewai_flow: bool = True  # Use CrewAI Flows for orchestration (recommended)

    # Active strategy pods
    active_pods: List[str] = field(
        default_factory=lambda: [
            "trend_following",
            "mean_reversion",
            "momentum",
            "breakout",
            "volatility",
        ]
    )

    # Risk limits
    max_position_pct: float = 0.20  # Max 20% in single position
    max_drawdown_halt_pct: float = 0.15  # Halt at 15% drawdown
    max_daily_loss_pct: float = 0.05  # Max 5% daily loss

    # Storage
    db_path: Optional[str] = None  # Path to DuckDB, None = default location

    def get_start_date(self) -> Optional[date]:
        """Parse start_date string to date object."""
        if self.start_date is None:
            return None
        return date.fromisoformat(self.start_date)

    def get_end_date(self) -> Optional[date]:
        """Parse end_date string to date object."""
        if self.end_date is None:
            return None
        return date.fromisoformat(self.end_date)

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "symbols": self.symbols,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "initial_equity": self.initial_equity,
            "max_leverage": self.max_leverage,
            "commission_per_share": self.commission_per_share,
            "slippage_bps": self.slippage_bps,
            "policy_update_frequency": self.policy_update_frequency,
            "active_pods": self.active_pods,
            "max_position_pct": self.max_position_pct,
            "max_drawdown_halt_pct": self.max_drawdown_halt_pct,
            "max_daily_loss_pct": self.max_daily_loss_pct,
            "db_path": self.db_path,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "HistoricalConfig":
        """Create from dictionary."""
        return cls(**data)


class SimulationContext(BaseModel):
    """
    Context passed to agents for each simulation day.

    Contains all information needed for agents to make decisions:
    - Current date
    - Market data for all symbols
    - Computed features
    - Current portfolio state
    - Active policy weights
    - Multi-timeframe data (if MTF enabled)
    """

    date: date
    market_data: Dict[str, Dict] = Field(
        default_factory=dict,
        description="OHLCV data per symbol: {symbol: {open, high, low, close, volume}}",
    )
    features: Dict[str, Dict] = Field(
        default_factory=dict, description="Computed features per symbol from QuantCore"
    )
    portfolio: Dict = Field(
        default_factory=dict,
        description="Current portfolio state: {equity, cash, positions, drawdown}",
    )
    policy: Dict = Field(
        default_factory=dict, description="Current policy weights: {pod_name: weight}"
    )
    regimes: Dict[str, Dict] = Field(
        default_factory=dict,
        description="Regime state per symbol: {symbol: {trend, volatility}}",
    )

    # Multi-timeframe fields (optional, used when MTF is enabled)
    mtf_data: Optional[Dict] = Field(
        default=None,
        description="Multi-timeframe data per symbol: {symbol: {timeframe: DataFrame}}",
    )
    bar_hour: Optional[int] = Field(
        default=None,
        description="Current bar hour for intraday processing (e.g., 10, 14)",
    )
    execution_timeframe: Optional[str] = Field(
        default=None, description="Execution timeframe (daily, 4h, 1h)"
    )

    class Config:
        arbitrary_types_allowed = True


class DayResult(BaseModel):
    """
    Result from running agents for a single day.

    Contains:
    - Approved trades to execute
    - Agent log messages for timeline
    - Detected regimes
    - Signal funnel metrics
    """

    trades: List[Dict] = Field(
        default_factory=list, description="List of trades to execute"
    )
    agent_logs: List[Dict] = Field(
        default_factory=list, description="Agent messages for chat timeline"
    )
    regimes: Dict[str, Dict] = Field(
        default_factory=dict, description="Regime classification per symbol"
    )
    signals: List[Dict] = Field(
        default_factory=list, description="Generated signals (for logging)"
    )
    signal_funnel: Dict[str, int] = Field(
        default_factory=dict,
        description="Signal funnel metrics: generated, validated, approved, executed",
    )
