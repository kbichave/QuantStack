# Copyright 2024 QuantCore Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Unified strategy interface for trading systems.

This module provides the abstract base class for all trading strategies,
along with supporting data structures for market state and target positions.

The strategy interface follows an event-driven paradigm where strategies
receive market state updates and output desired positions.

Example
-------
>>> from quantcore.strategy.base import Strategy, MarketState, TargetPosition
>>>
>>> class MyStrategy(Strategy):
...     def on_bar(self, state):
...         if state.features.get("rsi", 50) < 30:
...             return [TargetPosition(
...                 symbol=state.symbol,
...                 direction=PositionDirection.LONG,
...                 confidence=0.8
...             )]
...         return []
...
...     def get_required_data(self):
...         return DataRequirements(timeframes=["1D"])
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


class PositionDirection(Enum):
    """
    Position direction enumeration.

    Attributes
    ----------
    LONG : str
        Long position (buy).
    SHORT : str
        Short position (sell).
    FLAT : str
        No position (close all).
    """

    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


@dataclass
class RegimeState:
    """
    Current market regime state.

    Parameters
    ----------
    trend_regime : str
        Current trend regime: "BULL", "BEAR", or "SIDEWAYS".
    vol_regime : str
        Current volatility regime: "LOW", "MEDIUM", or "HIGH".
    trend_confidence : float, default 0.5
        Confidence in trend regime classification [0, 1].
    vol_confidence : float, default 0.5
        Confidence in volatility regime classification [0, 1].

    Examples
    --------
    >>> regime = RegimeState(
    ...     trend_regime="BULL",
    ...     vol_regime="LOW",
    ...     trend_confidence=0.8,
    ...     vol_confidence=0.7
    ... )
    """

    trend_regime: str  # BULL, BEAR, SIDEWAYS
    vol_regime: str  # LOW, MEDIUM, HIGH
    trend_confidence: float = 0.5
    vol_confidence: float = 0.5


@dataclass
class MarketState:
    """
    Complete market state passed to strategies.

    Contains all information a strategy needs to make trading decisions,
    including price data, computed features, regime context, and current
    position information.

    Parameters
    ----------
    timestamp : datetime
        Current bar timestamp.
    bar_index : int
        Sequential bar index from start of data.
    symbol : str
        Symbol being traded.
    open : float
        Bar open price.
    high : float
        Bar high price.
    low : float
        Bar low price.
    close : float
        Bar close price.
    volume : float
        Bar volume.
    features : Dict[str, float], optional
        Computed features dictionary (200+ from FeatureFactory).
    regime : RegimeState, optional
        Current regime classification.
    iv_rank : float, optional
        Implied volatility rank (0-100).
    iv_percentile : float, optional
        Implied volatility percentile (0-100).
    days_to_earnings : int, optional
        Days until next earnings announcement.
    atm_iv : float, optional
        At-the-money implied volatility.
    current_delta : float, default 0.0
        Current portfolio delta exposure.
    current_gamma : float, default 0.0
        Current portfolio gamma exposure.
    current_theta : float, default 0.0
        Current portfolio theta exposure.
    current_vega : float, default 0.0
        Current portfolio vega exposure.
    unrealized_pnl : float, default 0.0
        Current unrealized P&L.
    portfolio_equity : float, default 100000.0
        Current portfolio equity value.
    drawdown_pct : float, default 0.0
        Current drawdown from peak (percentage).

    Examples
    --------
    >>> state = MarketState(
    ...     timestamp=datetime.now(),
    ...     bar_index=100,
    ...     symbol="AAPL",
    ...     open=150.0,
    ...     high=152.0,
    ...     low=149.0,
    ...     close=151.0,
    ...     volume=1000000,
    ...     features={"rsi": 45.0, "macd": 0.5}
    ... )
    >>> state.to_feature_vector().shape
    (20,)

    See Also
    --------
    Strategy.on_bar : Method that receives MarketState.
    TargetPosition : Output from strategy decisions.
    """

    # Time
    timestamp: datetime
    bar_index: int

    # Underlying data
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float

    # Features (200+ from MultiTimeframeFeatureFactory)
    features: Dict[str, float] = field(default_factory=dict)

    # Regime context (must be explicit)
    regime: Optional[RegimeState] = None

    # Options-specific
    iv_rank: Optional[float] = None
    iv_percentile: Optional[float] = None
    days_to_earnings: Optional[int] = None
    atm_iv: Optional[float] = None

    # Current position state
    current_delta: float = 0.0
    current_gamma: float = 0.0
    current_theta: float = 0.0
    current_vega: float = 0.0
    unrealized_pnl: float = 0.0

    # Portfolio context
    portfolio_equity: float = 100000.0
    drawdown_pct: float = 0.0

    def to_feature_vector(self) -> np.ndarray:
        """
        Convert market state to feature vector for ML/RL models.

        Returns
        -------
        np.ndarray
            Float32 array containing all features in consistent order:
            1. All features from features dict
            2. Regime one-hot encoding (if available)
            3. Options features
            4. Position state features

        Notes
        -----
        This method is primarily used for reinforcement learning agents
        that require a fixed-size observation vector.
        """
        # Base features
        vector = list(self.features.values())

        # Add regime one-hot
        if self.regime:
            vector.extend(
                [
                    1.0 if self.regime.trend_regime == "BULL" else 0.0,
                    1.0 if self.regime.trend_regime == "BEAR" else 0.0,
                    1.0 if self.regime.trend_regime == "SIDEWAYS" else 0.0,
                    1.0 if self.regime.vol_regime == "LOW" else 0.0,
                    1.0 if self.regime.vol_regime == "MEDIUM" else 0.0,
                    1.0 if self.regime.vol_regime == "HIGH" else 0.0,
                    self.regime.trend_confidence,
                    self.regime.vol_confidence,
                ]
            )

        # Add options features
        vector.extend(
            [
                self.iv_rank or 50.0,
                self.iv_percentile or 50.0,
                self.days_to_earnings or 999,
                self.atm_iv or 0.2,
            ]
        )

        # Add position state
        vector.extend(
            [
                self.current_delta,
                self.current_gamma,
                self.current_theta,
                self.current_vega,
                (
                    self.unrealized_pnl / self.portfolio_equity
                    if self.portfolio_equity > 0
                    else 0
                ),
            ]
        )

        return np.array(vector, dtype=np.float32)


@dataclass
class TargetPosition:
    """
    Target position specification from strategy.

    Strategies output this to indicate desired positions. The execution
    layer interprets these targets and generates appropriate orders.

    Parameters
    ----------
    symbol : str
        Symbol for the position.
    direction : PositionDirection
        Desired position direction (LONG, SHORT, FLAT).
    confidence : float
        Signal confidence in [-1, +1] range. Used for position sizing.
    target_delta : float, optional
        Target delta exposure (for options strategies).
    max_premium : float, optional
        Maximum premium to pay (for options).
    min_dte : int, default 20
        Minimum days to expiration (for options).
    max_dte : int, default 45
        Maximum days to expiration (for options).
    stop_loss_pct : float, optional
        Stop loss as percentage of entry price.
    take_profit_pct : float, optional
        Take profit as percentage of entry price.
    reason : str, default ""
        Human-readable reason for the signal.
    signal_strength : float, default 0.0
        Raw signal strength before confidence scaling.

    Examples
    --------
    >>> target = TargetPosition(
    ...     symbol="AAPL",
    ...     direction=PositionDirection.LONG,
    ...     confidence=0.8,
    ...     stop_loss_pct=0.02,
    ...     take_profit_pct=0.04,
    ...     reason="RSI oversold with bullish divergence"
    ... )
    """

    symbol: str
    direction: PositionDirection
    confidence: float  # [-1, +1] for sizing

    # Optional specifics
    target_delta: Optional[float] = None
    max_premium: Optional[float] = None
    min_dte: int = 20
    max_dte: int = 45

    # Risk parameters
    stop_loss_pct: Optional[float] = None
    take_profit_pct: Optional[float] = None

    # Metadata
    reason: str = ""
    signal_strength: float = 0.0


@dataclass
class DataRequirements:
    """
    Data requirements specification for a strategy.

    Strategies declare their data requirements so the system can ensure
    all necessary data is available before running the strategy.

    Parameters
    ----------
    timeframes : List[str], default ["1H", "4H", "1D", "1W"]
        Required timeframes for multi-timeframe analysis.
    need_options_chain : bool, default False
        Whether options chain data is required.
    need_earnings_calendar : bool, default False
        Whether earnings calendar is required.
    need_news_sentiment : bool, default False
        Whether news sentiment data is required.
    lookback_bars : int, default 252
        Number of historical bars required for warmup.
    symbols : List[str], default []
        Specific symbols required (for multi-symbol strategies).

    Examples
    --------
    >>> requirements = DataRequirements(
    ...     timeframes=["1H", "1D"],
    ...     need_options_chain=True,
    ...     lookback_bars=100
    ... )
    """

    timeframes: List[str] = field(default_factory=lambda: ["1H", "4H", "1D", "1W"])
    need_options_chain: bool = False
    need_earnings_calendar: bool = False
    need_news_sentiment: bool = False
    lookback_bars: int = 252
    symbols: List[str] = field(default_factory=list)


class Strategy(ABC):
    """
    Abstract base class for all trading strategies.

    Strategies receive MarketState and output TargetPosition objects.
    This follows an event-driven paradigm where on_bar is called for
    each new bar of data.

    Parameters
    ----------
    name : str, default "BaseStrategy"
        Strategy name for logging and reporting.

    Attributes
    ----------
    name : str
        Strategy name.
    _is_initialized : bool
        Whether initialize() has been called.

    Methods
    -------
    on_bar(state)
        Process new bar and generate target positions.
    get_required_data()
        Specify data requirements for this strategy.
    initialize(**kwargs)
        Initialize strategy with any needed state.
    on_fill(symbol, direction, quantity, fill_price, timestamp)
        Callback when an order is filled.
    on_position_close(symbol, pnl, reason, timestamp)
        Callback when a position is closed.
    get_state()
        Get strategy state for serialization.
    set_state(state)
        Restore strategy state from serialization.

    Examples
    --------
    >>> class MeanReversionStrategy(Strategy):
    ...     def __init__(self, zscore_threshold=2.0):
    ...         super().__init__("MeanReversion")
    ...         self.threshold = zscore_threshold
    ...
    ...     def on_bar(self, state):
    ...         zscore = state.features.get("close_zscore_20", 0)
    ...         if zscore < -self.threshold:
    ...             return [TargetPosition(
    ...                 symbol=state.symbol,
    ...                 direction=PositionDirection.LONG,
    ...                 confidence=min(1.0, abs(zscore) / 3.0),
    ...                 reason=f"Z-score {zscore:.2f} below threshold"
    ...             )]
    ...         elif zscore > self.threshold:
    ...             return [TargetPosition(
    ...                 symbol=state.symbol,
    ...                 direction=PositionDirection.SHORT,
    ...                 confidence=min(1.0, abs(zscore) / 3.0),
    ...                 reason=f"Z-score {zscore:.2f} above threshold"
    ...             )]
    ...         return []
    ...
    ...     def get_required_data(self):
    ...         return DataRequirements(timeframes=["1D"])

    Notes
    -----
    Strategy implementations should be stateless where possible. If state
    is required (e.g., for tracking positions), implement get_state() and
    set_state() for proper serialization.

    The on_bar method is called at bar close. Any trades generated will
    execute at the next bar's open to avoid lookahead bias.

    See Also
    --------
    CompositeStrategy : Combines multiple strategies.
    MarketState : Input to on_bar method.
    TargetPosition : Output from on_bar method.
    """

    def __init__(self, name: str = "BaseStrategy") -> None:
        """
        Initialize strategy.

        Parameters
        ----------
        name : str, default "BaseStrategy"
            Strategy name for logging and reporting.
        """
        self.name = name
        self._is_initialized = False

    @abstractmethod
    def on_bar(self, state: MarketState) -> List[TargetPosition]:
        """
        Process new bar and generate target positions.

        This is the main decision method. It is called at bar T close,
        and any trades will execute at T+1 open.

        Parameters
        ----------
        state : MarketState
            Current market state including price data, features, and
            regime information.

        Returns
        -------
        List[TargetPosition]
            List of target positions. Empty list indicates no action.
            Multiple positions can be returned for multi-leg strategies.

        Notes
        -----
        Implementations should be idempotent - calling on_bar multiple
        times with the same state should return the same result.
        """
        pass

    @abstractmethod
    def get_required_data(self) -> DataRequirements:
        """
        Specify data requirements for this strategy.

        Returns
        -------
        DataRequirements
            Specification of required data including timeframes,
            lookback period, and any optional data sources.
        """
        pass

    def initialize(self, **kwargs: Any) -> None:
        """
        Initialize strategy with any needed state.

        Called once before the first on_bar call. Override to perform
        any one-time setup (loading models, warming up indicators, etc.).

        Parameters
        ----------
        **kwargs : Any
            Additional initialization parameters.
        """
        self._is_initialized = True

    def on_fill(
        self,
        symbol: str,
        direction: PositionDirection,
        quantity: int,
        fill_price: float,
        timestamp: datetime,
    ) -> None:
        """
        Callback when an order is filled.

        Override to track fills for position management.

        Parameters
        ----------
        symbol : str
            Symbol that was filled.
        direction : PositionDirection
            Direction of the fill.
        quantity : int
            Number of shares/contracts filled.
        fill_price : float
            Average fill price.
        timestamp : datetime
            Time of fill.
        """
        pass

    def on_position_close(
        self,
        symbol: str,
        pnl: float,
        reason: str,
        timestamp: datetime,
    ) -> None:
        """
        Callback when a position is closed.

        Override to track closed positions for analysis.

        Parameters
        ----------
        symbol : str
            Symbol that was closed.
        pnl : float
            Realized P&L from the position.
        reason : str
            Reason for close (e.g., "stop_loss", "take_profit", "signal").
        timestamp : datetime
            Time of close.
        """
        pass

    def get_state(self) -> Dict[str, Any]:
        """
        Get strategy state for serialization.

        Override to save strategy-specific state that needs to persist
        across sessions.

        Returns
        -------
        Dict[str, Any]
            Dictionary of state to serialize.
        """
        return {"name": self.name, "initialized": self._is_initialized}

    def set_state(self, state: Dict[str, Any]) -> None:
        """
        Restore strategy state from serialization.

        Override to restore strategy-specific state.

        Parameters
        ----------
        state : Dict[str, Any]
            Previously serialized state from get_state().
        """
        self._is_initialized = state.get("initialized", False)


class CompositeStrategy(Strategy):
    """
    Combines multiple strategies and aggregates their signals.

    This enables ensemble strategies where multiple sub-strategies vote
    on positions and signals are aggregated according to a specified rule.

    Parameters
    ----------
    strategies : List[Strategy]
        List of strategies to combine.
    name : str, default "CompositeStrategy"
        Strategy name.
    aggregation : str, default "majority"
        Aggregation method: "majority", "unanimous", or "any".

        - "majority": Signal fires if >50% of strategies agree
        - "unanimous": Signal fires only if all strategies agree
        - "any": Signal fires if any strategy generates it

    Examples
    --------
    >>> from quantcore.strategy.base import CompositeStrategy
    >>>
    >>> composite = CompositeStrategy(
    ...     strategies=[MomentumStrategy(), MeanReversionStrategy()],
    ...     aggregation="majority"
    ... )

    See Also
    --------
    Strategy : Base class for individual strategies.
    """

    def __init__(
        self,
        strategies: List[Strategy],
        name: str = "CompositeStrategy",
        aggregation: str = "majority",
    ) -> None:
        """
        Initialize composite strategy.

        Parameters
        ----------
        strategies : List[Strategy]
            List of strategies to combine.
        name : str, default "CompositeStrategy"
            Strategy name.
        aggregation : str, default "majority"
            How to combine signals: "majority", "unanimous", or "any".
        """
        super().__init__(name)
        self.strategies = strategies
        self.aggregation = aggregation

    def on_bar(self, state: MarketState) -> List[TargetPosition]:
        """
        Aggregate signals from all sub-strategies.

        Parameters
        ----------
        state : MarketState
            Current market state.

        Returns
        -------
        List[TargetPosition]
            Aggregated target positions.
        """
        all_positions = []

        for strategy in self.strategies:
            positions = strategy.on_bar(state)
            all_positions.extend(positions)

        # Group by symbol
        by_symbol: Dict[str, List[TargetPosition]] = {}
        for pos in all_positions:
            if pos.symbol not in by_symbol:
                by_symbol[pos.symbol] = []
            by_symbol[pos.symbol].append(pos)

        # Aggregate
        result = []
        for symbol, positions in by_symbol.items():
            aggregated = self._aggregate_positions(positions)
            if aggregated:
                result.append(aggregated)

        return result

    def _aggregate_positions(
        self, positions: List[TargetPosition]
    ) -> Optional[TargetPosition]:
        """
        Aggregate positions for the same symbol.

        Parameters
        ----------
        positions : List[TargetPosition]
            Positions to aggregate.

        Returns
        -------
        Optional[TargetPosition]
            Aggregated position, or None if no consensus.
        """
        if not positions:
            return None

        # Count directions
        long_count = sum(1 for p in positions if p.direction == PositionDirection.LONG)
        short_count = sum(
            1 for p in positions if p.direction == PositionDirection.SHORT
        )
        flat_count = sum(1 for p in positions if p.direction == PositionDirection.FLAT)

        total = len(positions)

        if self.aggregation == "majority":
            if long_count > total / 2:
                direction = PositionDirection.LONG
            elif short_count > total / 2:
                direction = PositionDirection.SHORT
            else:
                direction = PositionDirection.FLAT
        elif self.aggregation == "unanimous":
            if long_count == total:
                direction = PositionDirection.LONG
            elif short_count == total:
                direction = PositionDirection.SHORT
            else:
                direction = PositionDirection.FLAT
        else:  # any
            if long_count > 0:
                direction = PositionDirection.LONG
            elif short_count > 0:
                direction = PositionDirection.SHORT
            else:
                direction = PositionDirection.FLAT

        # Average confidence for matching direction
        matching = [p for p in positions if p.direction == direction]
        avg_confidence = np.mean([p.confidence for p in matching]) if matching else 0.0

        return TargetPosition(
            symbol=positions[0].symbol,
            direction=direction,
            confidence=float(avg_confidence),
            reason=f"Aggregated from {len(positions)} strategies",
        )

    def get_required_data(self) -> DataRequirements:
        """
        Combine data requirements from all sub-strategies.

        Returns
        -------
        DataRequirements
            Union of all sub-strategy requirements.
        """
        all_timeframes = set()
        need_options = False
        need_earnings = False
        need_news = False
        max_lookback = 0
        all_symbols = set()

        for strategy in self.strategies:
            req = strategy.get_required_data()
            all_timeframes.update(req.timeframes)
            need_options = need_options or req.need_options_chain
            need_earnings = need_earnings or req.need_earnings_calendar
            need_news = need_news or req.need_news_sentiment
            max_lookback = max(max_lookback, req.lookback_bars)
            all_symbols.update(req.symbols)

        return DataRequirements(
            timeframes=list(all_timeframes),
            need_options_chain=need_options,
            need_earnings_calendar=need_earnings,
            need_news_sentiment=need_news,
            lookback_bars=max_lookback,
            symbols=list(all_symbols),
        )

    def initialize(self, **kwargs: Any) -> None:
        """
        Initialize all sub-strategies.

        Parameters
        ----------
        **kwargs : Any
            Initialization parameters passed to all sub-strategies.
        """
        for strategy in self.strategies:
            strategy.initialize(**kwargs)
        super().initialize(**kwargs)
