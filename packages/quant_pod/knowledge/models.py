# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Pydantic models for knowledge store schemas.

Defines the structured data types stored in DuckDB and shared between agents.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# =============================================================================
# ENUMS
# =============================================================================


class TradeDirection(str, Enum):
    """Trade direction."""

    LONG = "LONG"
    SHORT = "SHORT"


class StructureType(str, Enum):
    """Option structure types."""

    LONG_CALL = "LONG_CALL"
    LONG_PUT = "LONG_PUT"
    SHORT_CALL = "SHORT_CALL"
    SHORT_PUT = "SHORT_PUT"
    CALL_SPREAD = "CALL_SPREAD"
    PUT_SPREAD = "PUT_SPREAD"
    IRON_CONDOR = "IRON_CONDOR"
    IRON_BUTTERFLY = "IRON_BUTTERFLY"
    CALENDAR = "CALENDAR"
    DIAGONAL = "DIAGONAL"
    STRADDLE = "STRADDLE"
    STRANGLE = "STRANGLE"
    BUTTERFLY = "BUTTERFLY"
    STOCK = "STOCK"


class TradeStatus(str, Enum):
    """Trade lifecycle status."""

    PENDING = "PENDING"
    OPEN = "OPEN"
    PARTIAL = "PARTIAL"
    CLOSED = "CLOSED"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"


class RegimeType(str, Enum):
    """Market regime classification."""

    TRENDING_UP = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    RANGING = "RANGING"
    VOLATILE = "VOLATILE"
    UNKNOWN = "UNKNOWN"


class VolatilityRegime(str, Enum):
    """Volatility regime classification."""

    LOW = "LOW"
    NORMAL = "NORMAL"
    HIGH = "HIGH"
    EXTREME = "EXTREME"


class WavePosition(str, Enum):
    """Elliott Wave position."""

    WAVE_1 = "WAVE_1"
    WAVE_2 = "WAVE_2"
    WAVE_3 = "WAVE_3"
    WAVE_4 = "WAVE_4"
    WAVE_5 = "WAVE_5"
    WAVE_A = "WAVE_A"
    WAVE_B = "WAVE_B"
    WAVE_C = "WAVE_C"
    UNKNOWN = "UNKNOWN"


# =============================================================================
# TRADE MODELS
# =============================================================================


class TradeLeg(BaseModel):
    """Single leg of a trade."""

    symbol: str
    option_type: Optional[str] = None  # CALL or PUT
    strike: Optional[float] = None
    expiration: Optional[str] = None
    action: str  # BUY_TO_OPEN, SELL_TO_CLOSE, etc.
    quantity: int
    fill_price: Optional[float] = None


class TradeRecord(BaseModel):
    """
    Complete trade record for journal.

    Stored in DuckDB trade_journal table.
    """

    id: Optional[int] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    # Trade identification
    symbol: str
    direction: TradeDirection
    structure_type: StructureType
    status: TradeStatus = TradeStatus.PENDING

    # Position details
    legs: List[TradeLeg] = []
    entry_price: Optional[float] = None  # Net premium paid/received
    exit_price: Optional[float] = None
    quantity: int = 1

    # P&L
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    max_profit_potential: Optional[float] = None
    max_loss_potential: Optional[float] = None

    # Context
    wave_scenario_id: Optional[str] = None
    regime_at_entry: Optional[str] = None
    volatility_at_entry: Optional[float] = None

    # Decision tracking
    confidence_score: float = 0.5
    agent_rationale: Optional[str] = None
    research_score: Optional[float] = None
    arena_rank: Optional[int] = None

    # Outcome analysis
    outcome_correct: Optional[bool] = None
    lessons_learned: Optional[str] = None
    tags: List[str] = []

    # Order IDs
    entry_order_id: Optional[str] = None
    exit_order_id: Optional[str] = None


# =============================================================================
# OBSERVATION MODELS
# =============================================================================


class MarketObservation(BaseModel):
    """
    Market observation from monitoring agents.

    Used for price alerts, volume spikes, gap detection, etc.
    """

    id: Optional[int] = None
    timestamp: datetime = Field(default_factory=datetime.now)

    symbol: str
    observation_type: str  # PRICE_ALERT, VOLUME_SPIKE, GAP, TECHNICAL

    # Observation data
    current_price: float
    price_change_pct: Optional[float] = None
    volume: Optional[int] = None
    volume_ratio: Optional[float] = None  # vs average

    # Technical levels
    support_level: Optional[float] = None
    resistance_level: Optional[float] = None

    # Alert details
    alert_message: str
    severity: str = "INFO"  # INFO, WARNING, ALERT, CRITICAL

    # Agent info
    source_agent: str = "market_monitor"
    processed: bool = False


class WaveScenario(BaseModel):
    """
    Elliott Wave scenario from wave analyst.
    """

    id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)

    symbol: str
    timeframe: str  # 1h, 4h, daily, weekly

    # Wave count
    wave_position: WavePosition
    wave_degree: str  # Primary, Intermediate, Minor, etc.
    confidence: float = 0.5

    # Scenario projections
    primary_target: Optional[float] = None
    secondary_target: Optional[float] = None
    invalidation_level: float

    # Scenario description
    scenario_type: str  # BULLISH, BEARISH, NEUTRAL
    description: str

    # Tracking
    is_active: bool = True
    invalidated_at: Optional[datetime] = None
    target_hit_at: Optional[datetime] = None

    source_agent: str = "wave_analyst"


class RegimeState(BaseModel):
    """
    Market regime state from regime detector.
    """

    id: Optional[int] = None
    timestamp: datetime = Field(default_factory=datetime.now)

    symbol: str
    timeframe: str

    # Regime classification
    trend_regime: RegimeType
    volatility_regime: VolatilityRegime

    # Metrics
    atr: Optional[float] = None
    atr_percentile: Optional[float] = None  # vs history
    adx: Optional[float] = None
    correlation_to_spy: Optional[float] = None

    # Regime change detection
    regime_changed: bool = False
    previous_trend_regime: Optional[RegimeType] = None
    previous_volatility_regime: Optional[VolatilityRegime] = None

    confidence: float = 0.5
    source_agent: str = "regime_detector"


# =============================================================================
# AGENT COMMUNICATION MODELS
# =============================================================================


class AgentMessage(BaseModel):
    """
    Inter-agent message for coordination.
    """

    id: Optional[int] = None
    timestamp: datetime = Field(default_factory=datetime.now)

    from_agent: str
    to_agent: Optional[str] = None  # None = broadcast
    message_type: str  # ALERT, SIGNAL, REQUEST, RESPONSE

    subject: str
    content: str
    data: Dict[str, Any] = {}

    priority: int = 5  # 1 (highest) to 10 (lowest)
    requires_response: bool = False
    response_deadline: Optional[datetime] = None

    acknowledged: bool = False
    acknowledged_at: Optional[datetime] = None


class TradingSignal(BaseModel):
    """
    Trading signal generated by analysis.
    """

    id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)

    symbol: str
    direction: TradeDirection
    signal_type: str  # WAVE_TARGET, REGIME_SHIFT, TECHNICAL, etc.

    # Signal strength
    strength: float = 0.5  # 0 to 1
    confidence: float = 0.5

    # Price targets
    entry_price: Optional[float] = None
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None

    # Supporting evidence
    wave_scenario_id: Optional[str] = None
    regime_state_id: Optional[int] = None
    observation_ids: List[int] = []

    # Rationale
    rationale: str

    # Status
    is_active: bool = True
    processed: bool = False
    trade_id: Optional[int] = None  # If converted to trade

    source_agent: str


class PerformanceMetrics(BaseModel):
    """
    Agent/strategy performance metrics.
    """

    timestamp: datetime = Field(default_factory=datetime.now)

    # Identification
    entity_type: str  # AGENT, STRUCTURE, REGIME, STRATEGY
    entity_name: str

    # Trade statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0

    # P&L metrics
    total_pnl: float = 0.0
    avg_pnl_per_trade: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0

    # Risk metrics
    sharpe_ratio: Optional[float] = None
    sortino_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    profit_factor: Optional[float] = None

    # Time period
    period_start: datetime
    period_end: datetime
