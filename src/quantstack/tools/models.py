# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Pydantic I/O models for QuantStack tool implementations.

Each tool has typed inputs and outputs defined here.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

# =============================================================================
# run_analysis
# =============================================================================


class RunAnalysisInput(BaseModel):
    """Input for the run_analysis tool."""

    symbol: str = Field(description="Ticker symbol (e.g., 'SPY', 'AAPL')")
    regime: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Pre-computed regime dict with keys: trend_regime, volatility_regime, "
            "confidence. If None, the server detects regime automatically."
        ),
    )
    include_historical_context: bool = Field(
        default=True,
        description="Whether to include blackboard historical context in the analysis.",
    )


class RunAnalysisOutput(BaseModel):
    """Output from the run_analysis tool."""

    success: bool
    daily_brief: dict[str, Any] | None = None
    error: str | None = None
    regime_used: dict[str, Any] = Field(default_factory=dict)
    elapsed_seconds: float = 0.0


# =============================================================================
# get_portfolio_state
# =============================================================================


class PortfolioStateOutput(BaseModel):
    """Output from the get_portfolio_state tool."""

    snapshot: dict[str, Any] = Field(
        description="PortfolioSnapshot: cash, positions_value, total_equity, daily_pnl, etc."
    )
    positions: list[dict[str, Any]] = Field(
        default_factory=list,
        description="List of open Position objects.",
    )
    context_string: str = Field(
        default="",
        description="Human-readable markdown summary of portfolio state.",
    )


# =============================================================================
# get_regime
# =============================================================================


class GetRegimeOutput(BaseModel):
    """Output from the get_regime tool."""

    success: bool
    symbol: str = ""
    trend_regime: str = ""
    volatility_regime: str = ""
    confidence: float = 0.0
    adx: float = 0.0
    atr: float = 0.0
    atr_percentile: float = 0.0
    error: str | None = None


# =============================================================================
# get_recent_decisions
# =============================================================================


class RecentDecisionSummary(BaseModel):
    """Summary of a single audit event."""

    event_id: str
    event_type: str
    agent_name: str
    symbol: str = ""
    action: str = ""
    confidence: float = 0.0
    output_summary: str = ""
    created_at: datetime | None = None


class RecentDecisionsOutput(BaseModel):
    """Output from the get_recent_decisions tool."""

    decisions: list[RecentDecisionSummary] = Field(default_factory=list)
    total: int = 0


# =============================================================================
# get_system_status
# =============================================================================


class SystemStatusOutput(BaseModel):
    """Output from the get_system_status tool."""

    kill_switch_active: bool = False
    kill_switch_reason: str | None = None
    risk_halted: bool = False
    broker_mode: str = "paper"
    session_id: str = ""


# =============================================================================
# Phase 2: Strategy Registry + Backtesting
# =============================================================================


class StrategyDefinition(BaseModel):
    """
    Input for registering a new strategy.

    entry_rules and exit_rules are lists of rule dicts, e.g.:
      [{"indicator": "rsi_14", "condition": "crosses_below", "value": 30}]
    parameters holds indicator settings:
      {"rsi_period": 14, "atr_period": 14, "sma_fast": 10, "sma_slow": 50}
    risk_params holds sizing/stop config:
      {"stop_loss_atr": 2.0, "take_profit_atr": 3.0, "position_pct": 0.05}
    regime_affinity maps regime labels to suitability scores (0-1):
      {"trending_up": 0.8, "ranging": 0.3}
    """

    name: str = Field(description="Unique human-readable strategy name")
    description: str = ""
    asset_class: str = "equities"
    regime_affinity: dict[str, float] = Field(default_factory=dict)
    parameters: dict[str, Any] = Field(default_factory=dict)
    entry_rules: list[dict[str, Any]] = Field(default_factory=list)
    exit_rules: list[dict[str, Any]] = Field(default_factory=list)
    risk_params: dict[str, Any] = Field(default_factory=dict)
    source: str = "manual"


class StrategyRecord(BaseModel):
    """Full strategy row as returned from the DB."""

    strategy_id: str
    name: str
    description: str = ""
    asset_class: str = "equities"
    regime_affinity: dict[str, float] = Field(default_factory=dict)
    parameters: dict[str, Any] = Field(default_factory=dict)
    entry_rules: list[dict[str, Any]] = Field(default_factory=list)
    exit_rules: list[dict[str, Any]] = Field(default_factory=list)
    risk_params: dict[str, Any] = Field(default_factory=dict)
    backtest_summary: dict[str, Any] | None = None
    walkforward_summary: dict[str, Any] | None = None
    status: str = "draft"
    source: str = "manual"
    created_at: str | None = None
    updated_at: str | None = None
    created_by: str = "claude_code"


class BacktestRequest(BaseModel):
    """Input for run_backtest tool."""

    strategy_id: str = Field(description="Strategy to backtest (from registry)")
    symbol: str = Field(description="Ticker symbol for price data")
    start_date: str | None = Field(
        default=None, description="Start date (YYYY-MM-DD). None = earliest available."
    )
    end_date: str | None = Field(
        default=None, description="End date (YYYY-MM-DD). None = latest available."
    )
    initial_capital: float = 100_000.0
    position_size_pct: float = 0.10
    commission: float = 1.0
    slippage_pct: float = 0.001


class BacktestResult(BaseModel):
    """Output from run_backtest tool."""

    success: bool
    strategy_id: str = ""
    symbol: str = ""
    total_trades: int = 0
    win_rate: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    total_return_pct: float = 0.0
    profit_factor: float = 0.0
    calmar_ratio: float = 0.0
    avg_trade_pnl: float = 0.0
    start_date: str | None = None
    end_date: str | None = None
    bars_tested: int = 0
    error: str | None = None


class WalkForwardRequest(BaseModel):
    """Input for run_walkforward tool."""

    strategy_id: str
    symbol: str
    n_splits: int = Field(default=5, ge=2, le=20)
    test_size: int = Field(
        default=252, description="Bars per test fold (~1 year daily)"
    )
    min_train_size: int = Field(
        default=504, description="Minimum training bars (~2 years)"
    )
    gap: int = Field(
        default=0, description="Legacy embargo bars (ignored when use_purged_cv=True)"
    )
    expanding: bool = True
    initial_capital: float = 100_000.0
    position_size_pct: float = 0.10
    embargo_pct: float = Field(
        default=0.01,
        ge=0.0,
        le=0.10,
        description="Fraction of data to embargo between train/test (purged CV). Default 1%.",
    )
    use_purged_cv: bool = Field(
        default=True,
        description="Use purged walk-forward CV with embargo (default True). "
        "Set False for legacy TimeSeriesSplit behavior.",
    )
    start_date: str | None = Field(
        default=None,
        description="Earliest date to include (YYYY-MM-DD). Filters out pre-QE data "
        "that poisons ETF walk-forward folds (e.g. '2010-01-01' for QQQ).",
    )
    end_date: str | None = Field(
        default=None,
        description="Latest date to include (YYYY-MM-DD).",
    )


class WalkForwardResult(BaseModel):
    """Output from run_walkforward tool."""

    success: bool
    strategy_id: str = ""
    symbol: str = ""
    n_folds: int = 0
    fold_results: list[dict[str, Any]] = Field(default_factory=list)
    is_sharpe_mean: float = 0.0
    oos_sharpe_mean: float = 0.0
    is_sharpe_std: float = 0.0
    oos_sharpe_std: float = 0.0
    overfit_ratio: float = 0.0
    oos_positive_folds: int = 0
    oos_degradation_pct: float = 0.0
    cv_method: str = Field(
        default="purged_walk_forward",
        description="CV method used: 'purged_walk_forward' or 'legacy_time_series_split'",
    )
    embargo_pct: float | None = Field(
        default=None,
        description="Embargo percentage used (None if legacy mode)",
    )
    error: str | None = None


# =============================================================================
# Phase 3: Execution
# =============================================================================


class TradeOrder(BaseModel):
    """Input for execute_trade tool."""

    symbol: str = Field(description="Ticker symbol")
    action: str = Field(description="'buy' or 'sell'")
    quantity: int | None = Field(
        default=None, description="Shares. Auto-calculated from position_size if None."
    )
    position_size: str = Field(
        default="quarter",
        description="'full', 'half', or 'quarter' — used if quantity is None.",
    )
    order_type: str = Field(default="market", description="'market' or 'limit'")
    limit_price: float | None = None
    reasoning: str = Field(
        description="REQUIRED — audit trail reasoning for this trade"
    )
    confidence: float = Field(ge=0, le=1, description="REQUIRED — 0-1 confidence score")
    strategy_id: str | None = Field(
        default=None, description="Links to strategy registry"
    )
    paper_mode: bool = Field(
        default=True, description="Must be explicitly False for live trading"
    )


class TradeResult(BaseModel):
    """Output from execute_trade / close_position tools."""

    success: bool
    order_id: str | None = None
    fill_price: float | None = None
    filled_quantity: int | None = None
    slippage_bps: float | None = None
    commission: float | None = None
    risk_approved: bool = False
    risk_violations: list[str] = Field(default_factory=list)
    error: str | None = None
    broker_mode: str = "paper"


class RiskMetrics(BaseModel):
    """Output from get_risk_metrics tool."""

    cash: float = 0.0
    total_equity: float = 0.0
    positions_value: float = 0.0
    position_count: int = 0
    daily_pnl: float = 0.0
    daily_loss_pct: float = 0.0
    daily_loss_limit_pct: float = 0.0
    daily_headroom_pct: float = 0.0
    gross_exposure: float = 0.0
    gross_exposure_pct: float = 0.0
    max_gross_exposure_pct: float = 0.0
    largest_position_pct: float = 0.0
    max_position_pct: float = 0.0
    kill_switch_active: bool = False
    risk_halted: bool = False


# =============================================================================
# Phase 4: Decoder
# =============================================================================


class DecodedStrategy(BaseModel):
    """Output from the decoder — inferred strategy specification."""

    source_trader: str = ""
    sample_size: int = 0
    date_range: str = ""
    style: str = ""
    entry_trigger: str = ""
    exit_trigger: str = ""
    timing_pattern: str = ""
    avg_holding_minutes: float = 0.0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    best_regime: str = ""
    regime_affinity: dict[str, float] = Field(default_factory=dict)
    edge_hypothesis: str = ""
    confidence: float = 0.0


# =============================================================================
# Phase 5: Meta Orchestration
# =============================================================================


class StrategyAllocation(BaseModel):
    """A single strategy's allocation within the portfolio."""

    strategy_id: str
    strategy_name: str = ""
    capital_pct: float = Field(ge=0, le=1.0, description="Fraction of equity allocated")
    symbols: list[str] = Field(default_factory=list)
    mode: str = "paper"  # "paper" or "live"
    regime_score: float = 0.0
    ranking_sharpe: float = 0.0
    reasoning: str = ""


class AllocationPlan(BaseModel):
    """Output from the allocation engine — portfolio-level capital plan."""

    regime: str = ""
    regime_confidence: float = 0.0
    allocations: list[StrategyAllocation] = Field(default_factory=list)
    total_allocated_pct: float = 0.0
    unallocated_pct: float = 1.0
    conflicts_resolved: int = 0
    warnings: list[str] = Field(default_factory=list)


class ConflictResolution(BaseModel):
    """Result of resolving a signal conflict for a single symbol."""

    symbol: str
    action: str = ""  # "keep", "skip", "adjust"
    kept_strategy: str | None = None
    skipped_strategies: list[str] = Field(default_factory=list)
    reasoning: str = ""
    original_trades: list[dict[str, Any]] = Field(default_factory=list)
    resolved_trade: dict[str, Any] | None = None
