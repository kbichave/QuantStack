"""
Position sizing environment for RL.

Simulates position sizing decisions with risk management.

MATURITY: EXPERIMENTAL
- Works best with real price data and pre-generated signals
- Falls back to synthetic signal generation if signals not provided
- Returns simulation uses real data when available

Use cases:
- Research into dynamic position sizing
- Testing risk management approaches

Limitations:
- Synthetic signals don't represent real alpha quality
- Return simulation simplified when no data provided
"""

from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
import numpy as np
import pandas as pd
from loguru import logger

from quantcore.rl.base import RLEnvironment, State, Action, Reward


@dataclass
class TradingSignal:
    """Trading signal from alpha."""

    direction: str  # "LONG", "SHORT", "NEUTRAL"
    confidence: float  # 0-1
    expected_return: float
    alpha_name: str


@dataclass
class PortfolioState:
    """Current portfolio state."""

    equity: float
    position: float  # Current position size
    unrealized_pnl: float
    realized_pnl: float
    drawdown: float
    max_drawdown: float
    volatility: float
    risk_budget_used: float


@dataclass
class SizingDataRequirements:
    """
    Documents data requirements for SizingEnvironment.

    Recommended:
        data: DataFrame with 'close' column for return simulation
        signals: List[TradingSignal] - pre-generated trading signals

    If not provided:
        - Returns are simulated with random walk
        - Signals are generated synthetically
    """

    required_columns: List[str] = None

    def __post_init__(self):
        self.required_columns = ["close"]


class SizingEnvironment(RLEnvironment):
    """
    Position sizing environment.

    MATURITY: EXPERIMENTAL

    State space (10 features):
    - Signal confidence
    - Signal direction encoded (-1, 0, 1)
    - Volatility regime (normalized)
    - Current drawdown (%)
    - Risk budget used (%)
    - Recent Sharpe ratio
    - Current position (% of equity)
    - Time since last trade
    - Regime indicator (encoded)
    - Rolling win rate

    Action space (continuous, 1 dimension):
    - Position scale factor (0.0 to 1.0)

    Reward:
    - Risk-adjusted return (Sharpe improvement)
    - Drawdown penalty
    - Consistency bonus
    """

    # Mark maturity
    MATURITY = "EXPERIMENTAL"
    DATA_REQUIREMENTS = SizingDataRequirements()

    def __init__(
        self,
        initial_equity: float = 100000,
        max_position_pct: float = 0.2,
        max_drawdown_limit: float = 0.15,
        risk_free_rate: float = 0.02,
        data: Optional[pd.DataFrame] = None,
        signals: Optional[List[TradingSignal]] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize sizing environment.

        Args:
            initial_equity: Starting equity
            max_position_pct: Maximum position as % of equity
            max_drawdown_limit: Max drawdown before forced liquidation
            risk_free_rate: Risk-free rate for Sharpe calculation
            data: Historical price data (requires 'close' column)
            signals: Pre-generated signals for simulation
            seed: Random seed for reproducibility
        """
        super().__init__()

        self.initial_equity = initial_equity
        self.max_position_pct = max_position_pct
        self.max_drawdown_limit = max_drawdown_limit
        self.risk_free_rate = risk_free_rate
        self.data = data
        self.signals = signals
        self.seed = seed

        # Random state for reproducibility
        self._rng = np.random.RandomState(seed)

        # State
        self.portfolio: Optional[PortfolioState] = None
        self.current_signal: Optional[TradingSignal] = None
        self.data_idx = 0

        # History
        self.equity_curve: List[float] = []
        self.returns: List[float] = []
        self.positions: List[float] = []
        self.trades: List[Dict] = []

        # Warning flags
        self._warned_synthetic_signals = False
        self._warned_synthetic_returns = False

        # Validate data if provided
        if self.data is not None:
            self._validate_data()

    def _validate_data(self) -> None:
        """Validate input data."""
        if "close" not in self.data.columns:
            logger.warning(
                "SizingEnvironment: 'close' column not found in data. "
                "Return simulation will use synthetic returns."
            )

    def reset(self) -> State:
        """Reset environment."""
        self.current_step = 0
        self.episode_reward = 0.0
        self.done = False

        # Reset RNG
        if self.seed is not None:
            self._rng = np.random.RandomState(
                self.seed
                + (self.episode_count if hasattr(self, "episode_count") else 0)
            )

        # Reset portfolio
        self.portfolio = PortfolioState(
            equity=self.initial_equity,
            position=0.0,
            unrealized_pnl=0.0,
            realized_pnl=0.0,
            drawdown=0.0,
            max_drawdown=0.0,
            volatility=0.02,
            risk_budget_used=0.0,
        )

        # Reset history
        self.equity_curve = [self.initial_equity]
        self.returns = []
        self.positions = []
        self.trades = []

        # Reset data index with bounds checking
        if self.data is not None and len(self.data) > 200:
            min_idx = 50
            max_idx = len(self.data) - 100
            if max_idx > min_idx:
                self.data_idx = self._rng.randint(min_idx, max_idx)
            else:
                self.data_idx = min_idx
        else:
            self.data_idx = 0

        # Generate initial signal
        self.current_signal = self._get_signal()

        return self._get_state()

    def step(self, action: Action) -> Tuple[State, Reward, bool, Dict[str, Any]]:
        """
        Take position sizing action.

        Args:
            action: Position scale factor (0-1)

        Returns:
            (next_state, reward, done, info)
        """
        # Get scale factor from action
        if isinstance(action.value, np.ndarray):
            scale = float(np.clip(action.value[0], 0, 1))
        else:
            scale = float(np.clip(action.value, 0, 1))

        # Calculate target position
        signal_direction = (
            1
            if self.current_signal.direction == "LONG"
            else (-1 if self.current_signal.direction == "SHORT" else 0)
        )

        max_position = self.portfolio.equity * self.max_position_pct
        target_position = signal_direction * scale * max_position

        # Execute position change
        old_position = self.portfolio.position
        position_change = target_position - old_position

        # Simulate market move
        price_return = self._simulate_return()

        # Update portfolio
        pnl = self.portfolio.position * price_return
        self.portfolio.equity += pnl
        self.portfolio.position = target_position
        self.portfolio.unrealized_pnl = 0  # Simplified: mark-to-market
        self.portfolio.realized_pnl += pnl

        # Update drawdown
        peak_equity = max(self.equity_curve)
        self.portfolio.drawdown = (peak_equity - self.portfolio.equity) / peak_equity
        self.portfolio.max_drawdown = max(
            self.portfolio.max_drawdown, self.portfolio.drawdown
        )

        # Update history
        self.equity_curve.append(self.portfolio.equity)
        period_return = (
            (self.portfolio.equity - self.equity_curve[-2]) / self.equity_curve[-2]
            if len(self.equity_curve) > 1
            else 0
        )
        self.returns.append(period_return)
        self.positions.append(target_position)

        # Update volatility estimate
        if len(self.returns) >= 20:
            self.portfolio.volatility = np.std(self.returns[-20:]) * np.sqrt(252)

        # Update risk budget
        position_risk = abs(target_position) / self.portfolio.equity
        self.portfolio.risk_budget_used = position_risk / self.max_position_pct

        # Get next signal
        self.data_idx += 1
        self.current_signal = self._get_signal()

        # Check termination
        breached_drawdown = self.portfolio.drawdown >= self.max_drawdown_limit
        equity_depleted = self.portfolio.equity <= self.initial_equity * 0.5
        max_steps_reached = self.current_step >= 250  # ~1 year

        self.done = breached_drawdown or equity_depleted or max_steps_reached

        # Calculate reward
        reward = self._calculate_reward(
            pnl=pnl,
            scale=scale,
            breached_drawdown=breached_drawdown,
        )

        self.current_step += 1

        info = {
            "equity": self.portfolio.equity,
            "position": target_position,
            "pnl": pnl,
            "drawdown": self.portfolio.drawdown,
            "scale": scale,
            "is_synthetic_signal": self.signals is None,
        }

        return self._get_state(), reward, self.done, info

    def _get_state(self) -> State:
        """Get current state."""
        # Signal features
        signal_direction = (
            1
            if self.current_signal.direction == "LONG"
            else (-1 if self.current_signal.direction == "SHORT" else 0)
        )

        # Calculate rolling Sharpe
        if len(self.returns) >= 20:
            rolling_sharpe = (
                (np.mean(self.returns[-20:]) - self.risk_free_rate / 252)
                / (np.std(self.returns[-20:]) + 1e-8)
                * np.sqrt(252)
            )
        else:
            rolling_sharpe = 0.0

        # Calculate win rate
        if len(self.returns) >= 10:
            win_rate = np.mean([r > 0 for r in self.returns[-20:]])
        else:
            win_rate = 0.5

        # Time since last trade
        time_since_trade = 0
        for i in range(len(self.positions) - 1, -1, -1):
            if i > 0 and self.positions[i] != self.positions[i - 1]:
                break
            time_since_trade += 1

        # Regime indicator (from volatility)
        regime = 0  # Normal
        if self.portfolio.volatility > 0.25:
            regime = 1  # High vol
        elif self.portfolio.volatility < 0.1:
            regime = -1  # Low vol

        features = np.array(
            [
                self.current_signal.confidence,
                signal_direction,
                self.portfolio.volatility / 0.3,  # Normalize
                self.portfolio.drawdown / self.max_drawdown_limit,
                self.portfolio.risk_budget_used,
                rolling_sharpe / 3,  # Normalize
                self.portfolio.position
                / (self.portfolio.equity * self.max_position_pct + 1e-8),
                min(time_since_trade / 10, 1.0),
                regime,
                win_rate,
            ],
            dtype=np.float32,
        )

        return State(
            features=features,
            metadata={
                "signal": self.current_signal,
                "portfolio": self.portfolio,
            },
        )

    def _get_signal(self) -> TradingSignal:
        """Get current trading signal."""
        # Use provided signals if available
        if self.signals is not None and self.current_step < len(self.signals):
            return self.signals[self.current_step]

        # Log warning once for synthetic signals
        if not self._warned_synthetic_signals and self.signals is None:
            logger.warning(
                "SizingEnvironment: No signals provided. "
                "Generating synthetic signals. These do not represent real alpha quality."
            )
            self._warned_synthetic_signals = True

        # Generate synthetic signal (with seeded RNG for reproducibility)
        direction = self._rng.choice(["LONG", "SHORT", "NEUTRAL"], p=[0.4, 0.4, 0.2])
        confidence = self._rng.beta(2, 2)  # Concentrated around 0.5
        expected_return = (
            self._rng.normal(0.001, 0.002) if direction != "NEUTRAL" else 0
        )

        return TradingSignal(
            direction=direction,
            confidence=confidence,
            expected_return=expected_return,
            alpha_name="synthetic",
        )

    def _simulate_return(self) -> float:
        """Simulate market return."""
        # Use real data if available
        if self.data is not None and "close" in self.data.columns:
            if self.data_idx > 0 and self.data_idx < len(self.data):
                current = self.data.iloc[self.data_idx]["close"]
                prev = self.data.iloc[self.data_idx - 1]["close"]
                return float((current - prev) / prev)

        # Log warning once for synthetic returns
        if not self._warned_synthetic_returns:
            logger.debug(
                "SizingEnvironment: Using synthetic returns. "
                "Provide data with 'close' column for realistic simulation."
            )
            self._warned_synthetic_returns = True

        # Synthetic return with slight drift (seeded RNG)
        return self._rng.normal(0.0005, 0.015)

    def _calculate_reward(
        self,
        pnl: float,
        scale: float,
        breached_drawdown: bool,
    ) -> Reward:
        """Calculate reward."""
        components = {}

        # Risk-adjusted return
        if len(self.returns) >= 2:
            recent_sharpe = np.mean(self.returns[-5:]) / (
                np.std(self.returns[-5:]) + 1e-8
            )
            components["risk_adjusted_return"] = recent_sharpe
        else:
            components["risk_adjusted_return"] = pnl / (self.initial_equity * 0.01)

        # Drawdown penalty
        dd_penalty = -self.portfolio.drawdown * 10
        components["drawdown_penalty"] = dd_penalty

        # Consistency bonus (reward for stable returns)
        if len(self.returns) >= 10:
            consistency = 1 - np.std(self.returns[-10:]) / (
                np.mean(np.abs(self.returns[-10:])) + 1e-8
            )
            components["consistency"] = consistency * 0.5
        else:
            components["consistency"] = 0

        # Appropriate sizing bonus
        # Reward for scaling with confidence
        if self.current_signal.confidence > 0.7 and scale > 0.5:
            components["sizing_bonus"] = 0.2
        elif self.current_signal.confidence < 0.3 and scale < 0.3:
            components["sizing_bonus"] = 0.2
        else:
            components["sizing_bonus"] = 0

        # Breach penalty
        if breached_drawdown:
            components["breach_penalty"] = -5.0
        else:
            components["breach_penalty"] = 0.0

        total = sum(components.values())

        return Reward(value=total, components=components)

    def get_state_dim(self) -> int:
        """Return state dimension."""
        return 10

    def get_action_dim(self) -> int:
        """Return action dimension."""
        return 1  # Continuous scale factor
