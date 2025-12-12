"""
Execution environment for RL.

Simulates order execution with market impact.

MATURITY: STABLE (with data), EXPERIMENTAL (without data)
- Works well with provided OHLCV data
- Falls back to random walk when data not available (not recommended)

Use cases:
- Learning optimal execution strategies (TWAP-like, IS minimization)
- Testing market impact sensitivity
- Order slicing optimization

Data requirements:
- OHLCV DataFrame with columns: close, high, low, volume
"""

from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
import numpy as np
import pandas as pd
from loguru import logger

from quantcore.rl.base import RLEnvironment, State, Action, Reward


@dataclass
class ExecutionOrder:
    """Order to be executed."""

    symbol: str
    direction: str  # "BUY" or "SELL"
    total_quantity: float
    time_horizon: int  # Bars to complete execution
    arrival_price: float


@dataclass
class ExecutionState:
    """Current execution state."""

    remaining_quantity: float
    remaining_time: int
    current_price: float
    spread: float
    volatility: float
    volume: float
    vwap: float
    shortfall: float  # Implementation shortfall so far


@dataclass
class ExecutionDataRequirements:
    """
    Documents data requirements for ExecutionEnvironment.

    Required for production use:
        data: DataFrame with 'close', 'high', 'low', 'volume' columns

    Without data:
        - Environment uses random walk simulation (EXPERIMENTAL)
        - Results may not reflect real market dynamics
    """

    required_columns: List[str] = None

    def __post_init__(self):
        self.required_columns = ["close", "high", "low", "volume"]


class ExecutionEnvironment(RLEnvironment):
    """
    Execution environment for order slicing optimization.

    MATURITY: STABLE (with OHLCV data)

    State space (8 features):
    - Remaining quantity (fraction)
    - Remaining time (fraction)
    - Price deviation from arrival (%)
    - Current spread (%)
    - Volatility (normalized)
    - Volume ratio (vs average)
    - VWAP deviation (%)
    - Implementation shortfall so far (%)

    Action space (discrete, 5 actions):
    - 0: Wait (no execution)
    - 1: Small limit order (10% of remaining)
    - 2: Medium limit order (25% of remaining)
    - 3: Large limit order (50% of remaining)
    - 4: Market order (aggressive, full execution)

    Reward:
    - Negative implementation shortfall
    - Penalty for incomplete execution
    """

    # Maturity depends on data availability
    MATURITY = "STABLE"  # When data provided
    DATA_REQUIREMENTS = ExecutionDataRequirements()

    # Action definitions
    ACTION_WAIT = 0
    ACTION_SMALL_LIMIT = 1
    ACTION_MEDIUM_LIMIT = 2
    ACTION_LARGE_LIMIT = 3
    ACTION_MARKET = 4

    # Execution fractions per action
    EXECUTION_FRACTIONS = {
        0: 0.0,  # Wait
        1: 0.10,  # Small
        2: 0.25,  # Medium
        3: 0.50,  # Large
        4: 1.00,  # Market (full)
    }

    def __init__(
        self,
        data: Optional[pd.DataFrame] = None,
        market_impact_coef: float = 0.1,
        spread_bps: float = 5.0,
        volatility_mult: float = 1.0,
        seed: Optional[int] = None,
    ):
        """
        Initialize execution environment.

        Args:
            data: Historical OHLCV data for simulation
                Required columns: close, high, low, volume
            market_impact_coef: Market impact coefficient
            spread_bps: Bid-ask spread in basis points
            volatility_mult: Volatility multiplier
            seed: Random seed for reproducibility
        """
        super().__init__()

        self.data = data
        self.market_impact_coef = market_impact_coef
        self.spread_bps = spread_bps
        self.volatility_mult = volatility_mult
        self.seed = seed

        # Random state for reproducibility
        self._rng = np.random.RandomState(seed)

        # Current order being executed
        self.order: Optional[ExecutionOrder] = None
        self.exec_state: Optional[ExecutionState] = None

        # Simulation state
        self.data_idx = 0
        self.executed_qty = 0.0
        self.executed_value = 0.0
        self.avg_fill_price = 0.0

        # Warning flags
        self._warned_no_data = False
        self._warned_random_walk = False

        # Validate data
        if self.data is not None:
            self._validate_data()
        else:
            logger.warning(
                "ExecutionEnvironment: No data provided. "
                "Using random walk simulation (EXPERIMENTAL). "
                "Provide OHLCV data for realistic execution simulation."
            )

    def _validate_data(self) -> None:
        """Validate input data has required columns."""
        required = ["close", "volume"]
        missing = [col for col in required if col not in self.data.columns]

        if missing:
            logger.warning(
                f"ExecutionEnvironment: Missing columns {missing}. "
                "Some features may use fallback values."
            )

        # Check for NaN in critical columns
        for col in ["close", "volume"]:
            if col in self.data.columns:
                nan_count = self.data[col].isna().sum()
                if nan_count > 0:
                    logger.warning(
                        f"ExecutionEnvironment: {nan_count} NaN values in '{col}'. "
                        "Consider forward-filling."
                    )

    def set_order(self, order: ExecutionOrder) -> None:
        """Set order to execute."""
        self.order = order

    def reset(self) -> State:
        """Reset environment for new execution."""
        self.current_step = 0
        self.episode_reward = 0.0
        self.done = False

        # Reset RNG
        if self.seed is not None:
            self._rng = np.random.RandomState(
                self.seed
                + (self.episode_count if hasattr(self, "episode_count") else 0)
            )

        # Reset execution state
        self.executed_qty = 0.0
        self.executed_value = 0.0
        self.avg_fill_price = 0.0

        # If no order set, create default
        if self.order is None:
            self.order = ExecutionOrder(
                symbol="WTI",
                direction="BUY",
                total_quantity=1000,
                time_horizon=20,
                arrival_price=self._get_current_price(),
            )

        # Initialize execution state
        self.exec_state = ExecutionState(
            remaining_quantity=self.order.total_quantity,
            remaining_time=self.order.time_horizon,
            current_price=self.order.arrival_price,
            spread=self.spread_bps / 10000,
            volatility=self._get_volatility(),
            volume=self._get_volume(),
            vwap=self.order.arrival_price,
            shortfall=0.0,
        )

        # Random starting point in data
        if self.data is not None and len(self.data) > self.order.time_horizon:
            self.data_idx = self._rng.randint(
                0, len(self.data) - self.order.time_horizon - 1
            )

        return self._get_state()

    def step(self, action: Action) -> Tuple[State, Reward, bool, Dict[str, Any]]:
        """
        Execute one step.

        Args:
            action: Action to take

        Returns:
            (next_state, reward, done, info)
        """
        action_idx = int(action.value) if action.is_discrete else int(action.value)

        # Get execution fraction
        exec_fraction = self.EXECUTION_FRACTIONS.get(action_idx, 0.0)
        exec_qty = self.exec_state.remaining_quantity * exec_fraction

        # Simulate execution
        fill_price, market_impact = self._simulate_execution(exec_qty, action_idx)

        # Update state
        if exec_qty > 0:
            self.executed_qty += exec_qty
            self.executed_value += exec_qty * fill_price
            self.avg_fill_price = (
                self.executed_value / self.executed_qty if self.executed_qty > 0 else 0
            )

        # Update execution state
        self.exec_state.remaining_quantity -= exec_qty
        self.exec_state.remaining_time -= 1

        # Update market state
        self._update_market_state()

        # Calculate implementation shortfall
        shortfall = self._calculate_shortfall()
        self.exec_state.shortfall = shortfall

        # Check if done
        execution_complete = self.exec_state.remaining_quantity <= 0
        time_expired = self.exec_state.remaining_time <= 0
        self.done = execution_complete or time_expired

        # Calculate reward
        reward = self._calculate_reward(
            exec_qty, fill_price, market_impact, execution_complete, time_expired
        )

        # Advance data index
        self.data_idx += 1
        self.current_step += 1

        info = {
            "executed_qty": exec_qty,
            "fill_price": fill_price,
            "market_impact": market_impact,
            "shortfall": shortfall,
            "remaining_qty": self.exec_state.remaining_quantity,
            "remaining_time": self.exec_state.remaining_time,
            "using_real_data": self.data is not None,
        }

        return self._get_state(), reward, self.done, info

    def _get_state(self) -> State:
        """Get current state."""
        features = np.array(
            [
                self.exec_state.remaining_quantity / self.order.total_quantity,
                self.exec_state.remaining_time / self.order.time_horizon,
                (self.exec_state.current_price - self.order.arrival_price)
                / self.order.arrival_price,
                self.exec_state.spread,
                self.exec_state.volatility,
                self.exec_state.volume,
                (
                    (self.exec_state.current_price - self.exec_state.vwap)
                    / self.exec_state.vwap
                    if self.exec_state.vwap > 0
                    else 0
                ),
                self.exec_state.shortfall,
            ],
            dtype=np.float32,
        )

        return State(features=features, metadata={"exec_state": self.exec_state})

    def _simulate_execution(
        self, quantity: float, action_idx: int
    ) -> Tuple[float, float]:
        """
        Simulate order execution with market impact.

        Returns:
            (fill_price, market_impact)
        """
        if quantity <= 0:
            return self.exec_state.current_price, 0.0

        # Base price
        base_price = self.exec_state.current_price

        # Market impact (temporary and permanent)
        # Impact = coef * sqrt(quantity / avg_volume) * volatility
        volume_fraction = quantity / (
            self.exec_state.volume * self.order.total_quantity + 1e-8
        )
        temp_impact = (
            self.market_impact_coef
            * np.sqrt(volume_fraction)
            * self.exec_state.volatility
        )

        # Market orders have higher impact
        if action_idx == self.ACTION_MARKET:
            temp_impact *= 2.0

        # Spread cost for market orders
        spread_cost = 0.0
        if action_idx == self.ACTION_MARKET:
            spread_cost = self.exec_state.spread / 2

        # Direction adjustment
        if self.order.direction == "BUY":
            fill_price = base_price * (1 + temp_impact + spread_cost)
        else:
            fill_price = base_price * (1 - temp_impact - spread_cost)

        return fill_price, temp_impact

    def _update_market_state(self) -> None:
        """Update market state from data or simulation."""
        if self.data is not None and self.data_idx < len(self.data):
            row = self.data.iloc[self.data_idx]

            if "close" in row.index:
                self.exec_state.current_price = row["close"]

            self.exec_state.volatility = self._get_volatility()
            self.exec_state.volume = self._get_volume()

            # Update VWAP
            if self.data_idx > 0 and all(
                col in self.data.columns for col in ["high", "low", "close", "volume"]
            ):
                recent = self.data.iloc[max(0, self.data_idx - 10) : self.data_idx + 1]
                tp = (recent["high"] + recent["low"] + recent["close"]) / 3
                vol_sum = recent["volume"].sum()
                if vol_sum > 0:
                    self.exec_state.vwap = (tp * recent["volume"]).sum() / vol_sum
        else:
            # Log warning once for random walk
            if not self._warned_random_walk:
                logger.debug(
                    "ExecutionEnvironment: Using random walk for price simulation. "
                    "Provide OHLCV data for realistic execution."
                )
                self._warned_random_walk = True

            # Simulate random walk (with seeded RNG)
            returns = self._rng.normal(0, self.exec_state.volatility)
            self.exec_state.current_price *= 1 + returns

    def _calculate_shortfall(self) -> float:
        """Calculate implementation shortfall."""
        if self.executed_qty <= 0:
            return 0.0

        if self.order.direction == "BUY":
            shortfall = (
                self.avg_fill_price - self.order.arrival_price
            ) / self.order.arrival_price
        else:
            shortfall = (
                self.order.arrival_price - self.avg_fill_price
            ) / self.order.arrival_price

        return shortfall

    def _calculate_reward(
        self,
        exec_qty: float,
        fill_price: float,
        market_impact: float,
        execution_complete: bool,
        time_expired: bool,
    ) -> Reward:
        """Calculate reward."""
        components = {}

        # Base reward: negative market impact
        impact_cost = -market_impact * 100  # Scale to percentage
        components["market_impact"] = impact_cost

        # Completion bonus
        if execution_complete:
            components["completion_bonus"] = 1.0
        else:
            components["completion_bonus"] = 0.0

        # Time penalty for incomplete execution
        if time_expired and not execution_complete:
            unfilled_fraction = (
                self.exec_state.remaining_quantity / self.order.total_quantity
            )
            components["time_penalty"] = -unfilled_fraction * 5.0  # Heavy penalty
        else:
            components["time_penalty"] = 0.0

        # Progress reward (encourage execution)
        if exec_qty > 0:
            progress = exec_qty / self.order.total_quantity
            components["progress"] = progress * 0.1
        else:
            components["progress"] = -0.01  # Small penalty for waiting

        total_reward = sum(components.values())

        return Reward(value=total_reward, components=components)

    def _get_current_price(self) -> float:
        """Get current price from data or default."""
        if (
            self.data is not None
            and len(self.data) > 0
            and "close" in self.data.columns
        ):
            idx = min(self.data_idx, len(self.data) - 1)
            return float(self.data.iloc[idx]["close"])
        return 100.0

    def _get_volatility(self) -> float:
        """Get current volatility estimate from data."""
        if (
            self.data is not None
            and "close" in self.data.columns
            and self.data_idx >= 20
        ):
            returns = (
                self.data["close"].pct_change().iloc[self.data_idx - 20 : self.data_idx]
            )
            vol = returns.std()
            if not np.isnan(vol):
                return float(vol) * self.volatility_mult
        return 0.02 * self.volatility_mult

    def _get_volume(self) -> float:
        """Get normalized volume from data."""
        if (
            self.data is not None
            and "volume" in self.data.columns
            and self.data_idx >= 20
        ):
            recent_vol = self.data["volume"].iloc[self.data_idx]
            avg_vol = (
                self.data["volume"].iloc[self.data_idx - 20 : self.data_idx].mean()
            )
            if avg_vol > 0 and not np.isnan(recent_vol):
                return float(recent_vol / avg_vol)
        return 1.0

    def get_state_dim(self) -> int:
        """Return state dimension."""
        return 8

    def get_action_dim(self) -> int:
        """Return action dimension."""
        return 5
