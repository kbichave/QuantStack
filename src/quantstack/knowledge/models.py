# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Pydantic models for knowledge store schemas.

Defines the structured data types stored in PostgreSQL and shared between agents.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

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
    option_type: str | None = None  # CALL or PUT
    strike: float | None = None
    expiration: str | None = None
    action: str  # BUY_TO_OPEN, SELL_TO_CLOSE, etc.
    quantity: int
    fill_price: float | None = None


class TradeRecord(BaseModel):
    """
    Complete trade record for journal.

    Stored in the trade_journal table.
    """

    id: int | None = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    # Trade identification
    symbol: str
    direction: TradeDirection
    structure_type: StructureType
    status: TradeStatus = TradeStatus.PENDING

    # Position details
    legs: list[TradeLeg] = []
    entry_price: float | None = None  # Net premium paid/received
    exit_price: float | None = None
    quantity: int = 1

    # P&L
    pnl: float | None = None
    pnl_pct: float | None = None
    max_profit_potential: float | None = None
    max_loss_potential: float | None = None

    # Context
    wave_scenario_id: str | None = None
    regime_at_entry: str | None = None
    volatility_at_entry: float | None = None

    # Decision tracking
    confidence_score: float = 0.5
    agent_rationale: str | None = None
    research_score: float | None = None
    arena_rank: int | None = None

    # Outcome analysis
    outcome_correct: bool | None = None
    lessons_learned: str | None = None
    tags: list[str] = []

    # Order IDs
    entry_order_id: str | None = None
    exit_order_id: str | None = None


# =============================================================================
# OBSERVATION MODELS
# =============================================================================


class MarketObservation(BaseModel):
    """
    Market observation from monitoring agents.

    Used for price alerts, volume spikes, gap detection, etc.
    """

    id: int | None = None
    timestamp: datetime = Field(default_factory=datetime.now)

    symbol: str
    observation_type: str  # PRICE_ALERT, VOLUME_SPIKE, GAP, TECHNICAL

    # Observation data
    current_price: float
    price_change_pct: float | None = None
    volume: int | None = None
    volume_ratio: float | None = None  # vs average

    # Technical levels
    support_level: float | None = None
    resistance_level: float | None = None

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

    id: str | None = None
    timestamp: datetime = Field(default_factory=datetime.now)

    symbol: str
    timeframe: str  # 1h, 4h, daily, weekly

    # Wave count
    wave_position: WavePosition
    wave_degree: str  # Primary, Intermediate, Minor, etc.
    confidence: float = 0.5

    # Scenario projections
    primary_target: float | None = None
    secondary_target: float | None = None
    invalidation_level: float

    # Scenario description
    scenario_type: str  # BULLISH, BEARISH, NEUTRAL
    description: str

    # Tracking
    is_active: bool = True
    invalidated_at: datetime | None = None
    target_hit_at: datetime | None = None

    source_agent: str = "wave_analyst"


class RegimeState(BaseModel):
    """
    Market regime state from regime detector.
    """

    id: int | None = None
    timestamp: datetime = Field(default_factory=datetime.now)

    symbol: str
    timeframe: str

    # Regime classification
    trend_regime: RegimeType
    volatility_regime: VolatilityRegime

    # Metrics
    atr: float | None = None
    atr_percentile: float | None = None  # vs history
    adx: float | None = None
    correlation_to_spy: float | None = None

    # Regime change detection
    regime_changed: bool = False
    previous_trend_regime: RegimeType | None = None
    previous_volatility_regime: VolatilityRegime | None = None

    confidence: float = 0.5
    source_agent: str = "regime_detector"


# =============================================================================
# AGENT COMMUNICATION MODELS
# =============================================================================


class AgentMessage(BaseModel):
    """
    Inter-agent message for coordination.
    """

    id: int | None = None
    timestamp: datetime = Field(default_factory=datetime.now)

    from_agent: str
    to_agent: str | None = None  # None = broadcast
    message_type: str  # ALERT, SIGNAL, REQUEST, RESPONSE

    subject: str
    content: str
    data: dict[str, Any] = {}

    priority: int = 5  # 1 (highest) to 10 (lowest)
    requires_response: bool = False
    response_deadline: datetime | None = None

    acknowledged: bool = False
    acknowledged_at: datetime | None = None


class TradingSignal(BaseModel):
    """
    Trading signal generated by analysis.
    """

    id: str | None = None
    timestamp: datetime = Field(default_factory=datetime.now)

    symbol: str
    direction: TradeDirection
    signal_type: str  # WAVE_TARGET, REGIME_SHIFT, TECHNICAL, etc.

    # Signal strength
    strength: float = 0.5  # 0 to 1
    confidence: float = 0.5

    # Price targets
    entry_price: float | None = None
    target_price: float | None = None
    stop_loss: float | None = None

    # Supporting evidence
    wave_scenario_id: str | None = None
    regime_state_id: int | None = None
    observation_ids: list[int] = []

    # Rationale
    rationale: str

    # Status
    is_active: bool = True
    processed: bool = False
    trade_id: int | None = None  # If converted to trade

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
    sharpe_ratio: float | None = None
    sortino_ratio: float | None = None
    max_drawdown: float | None = None
    profit_factor: float | None = None

    # Time period
    period_start: datetime
    period_end: datetime
