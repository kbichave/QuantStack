"""
Spread trading environment for RL.

Simulates WTI-Brent spread trading.

MATURITY: EXPERIMENTAL
- Core spread features (z-score, momentum, percentile) work with real data
- Volatility regime computed from spread data
- Correlation requires wti_data and brent_data columns
- USD regime and curve shape require additional data (stubbed with neutral values)
"""

from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
import numpy as np
import pandas as pd
from loguru import logger

from quantcore.rl.base import RLEnvironment, State, Action, Reward


@dataclass
class SpreadPosition:
    """Current spread position."""

    direction: int  # 1 = long spread (long WTI, short Brent), -1 = short, 0 = flat
    size: float  # Position size as fraction of max
    entry_spread: float  # Spread at entry
    entry_zscore: float  # Z-score at entry
    unrealized_pnl: float
    bars_held: int


@dataclass
class SpreadDataRequirements:
    """
    Documents what data the SpreadEnvironment needs.

    Required:
        spread_data: DataFrame with 'spread' column and DatetimeIndex

    Optional (for richer features):
        - 'wti' column: WTI price for correlation calculation
        - 'brent' column: Brent price for correlation calculation
        - 'usd' column: USD index for USD regime
        - 'curve' column: Futures curve shape indicator
    """

    required_columns: List[str] = None
    optional_columns: List[str] = None

    def __post_init__(self):
        self.required_columns = ["spread"]
        self.optional_columns = ["wti", "brent", "usd", "curve"]


class SpreadEnvironment(RLEnvironment):
    """
    WTI-Brent spread trading environment.

    MATURITY: EXPERIMENTAL

    State space (12 features):
    - Spread z-score (computed from data)
    - Spread momentum (5-bar) (computed from data)
    - Spread momentum (20-bar) (computed from data)
    - Spread percentile rank (computed from data)
    - Current position direction
    - Current position size
    - Unrealized PnL
    - Bars held
    - Volatility regime (computed from spread returns)
    - Correlation (WTI-Brent) (computed if wti/brent columns present, else neutral)
    - USD regime indicator (computed if usd column present, else neutral)
    - Curve shape (contango/backwardation) (computed if curve column present, else neutral)

    Action space (discrete, 5 actions):
    - 0: Close position
    - 1: Small long spread (25%)
    - 2: Full long spread (100%)
    - 3: Small short spread (25%)
    - 4: Full short spread (100%)

    Reward:
    - Spread convergence profit
    - Mean reversion success
    - Risk-adjusted return
    """

    ACTION_CLOSE = 0
    ACTION_SMALL_LONG = 1
    ACTION_FULL_LONG = 2
    ACTION_SMALL_SHORT = 3
    ACTION_FULL_SHORT = 4

    # Data requirements documentation
    DATA_REQUIREMENTS = SpreadDataRequirements()

    def __init__(
        self,
        spread_data: Optional[pd.DataFrame] = None,
        max_position_value: float = 100000,
        transaction_cost_bps: float = 5,
        zscore_lookback: int = 60,
        max_holding_bars: int = 50,
        volatility_lookback: int = 20,
        correlation_lookback: int = 60,
    ):
        """
        Initialize spread environment.

        Args:
            spread_data: DataFrame with spread time series.
                Required columns: 'spread'
                Optional columns: 'wti', 'brent', 'usd', 'curve'
            max_position_value: Maximum position value
            transaction_cost_bps: Transaction cost in bps
            zscore_lookback: Lookback for z-score calculation
            max_holding_bars: Max bars to hold position
            volatility_lookback: Lookback for volatility regime calculation
            correlation_lookback: Lookback for correlation calculation
        """
        super().__init__()

        self.spread_data = spread_data
        self.max_position_value = max_position_value
        self.transaction_cost = transaction_cost_bps / 10000
        self.zscore_lookback = zscore_lookback
        self.max_holding_bars = max_holding_bars
        self.volatility_lookback = volatility_lookback
        self.correlation_lookback = correlation_lookback

        # State
        self.position: Optional[SpreadPosition] = None
        self.data_idx = 0

        # History
        self.equity_curve: List[float] = []
        self.trades: List[Dict] = []

        # Generated data if not provided
        self.generated_spread: List[float] = []

        # Track warnings to avoid spamming logs
        self._warned_missing_wti_brent = False
        self._warned_missing_usd = False
        self._warned_missing_curve = False
        self._warned_synthetic_data = False

        # Precompute volatility percentiles for efficiency
        self._volatility_cache: Optional[pd.Series] = None
        self._correlation_cache: Optional[pd.Series] = None

        # Validate and precompute if data provided
        if self.spread_data is not None:
            self._validate_spread_data()
            self._precompute_features()

    def _validate_spread_data(self) -> None:
        """Validate spread data has required columns."""
        if self.spread_data is None:
            return

        if "spread" not in self.spread_data.columns:
            raise ValueError(
                "spread_data must contain 'spread' column. "
                f"Got columns: {list(self.spread_data.columns)}"
            )

        # Check for NaN in spread column
        nan_count = self.spread_data["spread"].isna().sum()
        if nan_count > 0:
            nan_pct = nan_count / len(self.spread_data) * 100
            logger.warning(
                f"SpreadEnvironment: spread column has {nan_count} NaN values ({nan_pct:.1f}%). "
                "Consider forward-filling or dropping these rows."
            )

        # Check data is sorted by index
        if isinstance(self.spread_data.index, pd.DatetimeIndex):
            if not self.spread_data.index.is_monotonic_increasing:
                logger.warning(
                    "SpreadEnvironment: spread_data index is not sorted. "
                    "This may cause unexpected behavior."
                )

    def _precompute_features(self) -> None:
        """Precompute rolling features for efficiency."""
        if self.spread_data is None:
            return

        # Precompute rolling volatility (as percentile rank)
        spread_returns = (
            self.spread_data["spread"].pct_change(fill_method=None).fillna(0)
        )
        rolling_vol = spread_returns.rolling(window=self.volatility_lookback).std()

        # Convert to percentile rank over full history (expanding window)
        def vol_percentile(vol_series):
            """Compute expanding percentile rank of volatility."""
            result = np.zeros(len(vol_series))
            for i in range(len(vol_series)):
                if i < self.volatility_lookback:
                    result[i] = 0.5  # Neutral for insufficient data
                else:
                    historical = vol_series.iloc[: i + 1].dropna()
                    if len(historical) > 0:
                        current_vol = vol_series.iloc[i]
                        result[i] = (historical <= current_vol).mean()
                    else:
                        result[i] = 0.5
            return pd.Series(result, index=vol_series.index)

        self._volatility_cache = vol_percentile(rolling_vol)

        # Precompute rolling correlation if wti/brent columns exist
        if "wti" in self.spread_data.columns and "brent" in self.spread_data.columns:
            wti_returns = self.spread_data["wti"].pct_change().fillna(0)
            brent_returns = self.spread_data["brent"].pct_change().fillna(0)
            self._correlation_cache = (
                wti_returns.rolling(window=self.correlation_lookback)
                .corr(brent_returns)
                .fillna(0.9)
            )  # Default to high correlation

    def reset(self) -> State:
        """Reset environment."""
        self.current_step = 0
        self.episode_reward = 0.0
        self.done = False

        # Reset position
        self.position = SpreadPosition(
            direction=0,
            size=0.0,
            entry_spread=0.0,
            entry_zscore=0.0,
            unrealized_pnl=0.0,
            bars_held=0,
        )

        # Reset history
        self.equity_curve = [self.max_position_value]
        self.trades = []

        # Reset data index
        if (
            self.spread_data is not None
            and len(self.spread_data) > self.zscore_lookback + 150
        ):
            min_idx = self.zscore_lookback + 10
            max_idx = len(self.spread_data) - 100
            if max_idx > min_idx:
                self.data_idx = np.random.randint(min_idx, max_idx)
            else:
                self.data_idx = min_idx
        else:
            self.data_idx = self.zscore_lookback + 10
            self._generate_spread_series()
            if not self._warned_synthetic_data:
                logger.warning(
                    "SpreadEnvironment: No spread_data provided or insufficient data. "
                    "Using synthetic mean-reverting spread. Results may not reflect real market dynamics."
                )
                self._warned_synthetic_data = True

        return self._get_state()

    def step(self, action: Action) -> Tuple[State, Reward, bool, Dict[str, Any]]:
        """
        Take trading action.

        Args:
            action: Trading action

        Returns:
            (next_state, reward, done, info)
        """
        action_idx = int(action.value) if action.is_discrete else int(action.value)

        old_position = self.position
        transaction_cost = 0.0

        # Execute action
        if action_idx == self.ACTION_CLOSE:
            if self.position.direction != 0:
                transaction_cost = self._close_position()
        elif action_idx == self.ACTION_SMALL_LONG:
            transaction_cost = self._open_position(1, 0.25)
        elif action_idx == self.ACTION_FULL_LONG:
            transaction_cost = self._open_position(1, 1.0)
        elif action_idx == self.ACTION_SMALL_SHORT:
            transaction_cost = self._open_position(-1, 0.25)
        elif action_idx == self.ACTION_FULL_SHORT:
            transaction_cost = self._open_position(-1, 1.0)

        # Advance market
        self.data_idx += 1
        self.current_step += 1

        # Update position PnL
        spread_change = self._get_spread_change()
        if self.position.direction != 0:
            self.position.unrealized_pnl += (
                self.position.direction
                * self.position.size
                * spread_change
                * self.max_position_value
            )
            self.position.bars_held += 1

        # Update equity
        equity_change = self.position.unrealized_pnl - transaction_cost
        self.equity_curve.append(self.equity_curve[-1] + equity_change)

        # Check termination
        max_bars_reached = self.current_step >= 200
        position_timeout = self.position.bars_held >= self.max_holding_bars

        if position_timeout and self.position.direction != 0:
            self._close_position()  # Force close

        self.done = max_bars_reached

        # Calculate reward
        reward = self._calculate_reward(
            action_idx,
            old_position,
            spread_change,
            transaction_cost,
        )

        info = {
            "position_direction": self.position.direction,
            "position_size": self.position.size,
            "unrealized_pnl": self.position.unrealized_pnl,
            "spread_zscore": self._get_spread_zscore(),
            "equity": self.equity_curve[-1],
        }

        return self._get_state(), reward, self.done, info

    def _get_state(self) -> State:
        """Get current state."""
        zscore = self._get_spread_zscore()
        spread = self._get_current_spread()

        # Spread momentum
        mom_5 = self._get_spread_momentum(5)
        mom_20 = self._get_spread_momentum(20)

        # Percentile rank
        percentile = self._get_spread_percentile()

        # Market features - computed from real data where available
        volatility = self._get_volatility_regime()
        correlation = self._get_correlation()
        usd_regime = self._get_usd_regime()
        curve_shape = self._get_curve_shape()

        features = np.array(
            [
                zscore / 3,  # Normalize
                mom_5 * 100,  # Scale
                mom_20 * 100,
                percentile,
                self.position.direction,
                self.position.size,
                self.position.unrealized_pnl
                / (self.max_position_value * 0.01),  # Normalize
                min(self.position.bars_held / self.max_holding_bars, 1.0),
                volatility,
                correlation,
                usd_regime,
                curve_shape,
            ],
            dtype=np.float32,
        )

        return State(
            features=features,
            metadata={
                "spread": spread,
                "zscore": zscore,
                "position": self.position,
            },
        )

    def _open_position(self, direction: int, size: float) -> float:
        """Open or adjust position."""
        current_spread = self._get_current_spread()
        current_zscore = self._get_spread_zscore()

        # Calculate transaction cost
        position_change = abs(
            direction * size - self.position.direction * self.position.size
        )
        cost = position_change * self.max_position_value * self.transaction_cost

        # Update position
        self.position = SpreadPosition(
            direction=direction,
            size=size,
            entry_spread=current_spread,
            entry_zscore=current_zscore,
            unrealized_pnl=0.0,
            bars_held=0,
        )

        return cost

    def _close_position(self) -> float:
        """Close current position."""
        if self.position.direction == 0:
            return 0.0

        # Transaction cost
        cost = self.position.size * self.max_position_value * self.transaction_cost

        # Record trade
        pnl = self.position.unrealized_pnl - cost
        self.trades.append(
            {
                "direction": self.position.direction,
                "size": self.position.size,
                "entry_zscore": self.position.entry_zscore,
                "exit_zscore": self._get_spread_zscore(),
                "pnl": pnl,
                "bars_held": self.position.bars_held,
            }
        )

        # Reset position
        self.position = SpreadPosition(
            direction=0,
            size=0.0,
            entry_spread=0.0,
            entry_zscore=0.0,
            unrealized_pnl=0.0,
            bars_held=0,
        )

        return cost

    def _calculate_reward(
        self,
        action_idx: int,
        old_position: SpreadPosition,
        spread_change: float,
        transaction_cost: float,
    ) -> Reward:
        """Calculate reward."""
        components = {}

        # PnL reward
        pnl = self.position.unrealized_pnl - transaction_cost
        components["pnl"] = pnl / (self.max_position_value * 0.001)  # Normalize

        # Mean reversion reward
        zscore = self._get_spread_zscore()
        if self.position.direction != 0:
            # Reward for z-score moving toward zero
            zscore_change = abs(self.position.entry_zscore) - abs(zscore)
            components["mean_reversion"] = zscore_change * 0.5
        else:
            components["mean_reversion"] = 0.0

        # Correct timing reward
        if action_idx in [self.ACTION_SMALL_LONG, self.ACTION_FULL_LONG]:
            if zscore < -1.5:
                components["timing"] = 0.3  # Good to long when oversold
            else:
                components["timing"] = -0.1
        elif action_idx in [self.ACTION_SMALL_SHORT, self.ACTION_FULL_SHORT]:
            if zscore > 1.5:
                components["timing"] = 0.3  # Good to short when overbought
            else:
                components["timing"] = -0.1
        else:
            components["timing"] = 0.0

        # Hold reward (small penalty for holding too long)
        if self.position.bars_held > 20:
            components["hold_penalty"] = -0.05 * (self.position.bars_held - 20) / 10
        else:
            components["hold_penalty"] = 0.0

        # Transaction cost penalty
        components["cost_penalty"] = -transaction_cost / (
            self.max_position_value * 0.001
        )

        total = sum(components.values())

        return Reward(value=total, components=components)

    def _generate_spread_series(self) -> None:
        """Generate synthetic spread series (fallback when no data provided)."""
        # Mean-reverting spread using Ornstein-Uhlenbeck process
        n = 500
        spread = [0.0]
        mean = 0.0

        # Use fixed seed for reproducibility within episode
        rng = np.random.RandomState(42 + self.current_step)

        for _ in range(n - 1):
            # OU process
            theta = 0.1  # Mean reversion speed
            sigma = 0.02  # Volatility

            ds = theta * (mean - spread[-1]) + sigma * rng.normal()
            spread.append(spread[-1] + ds)

        self.generated_spread = spread

    def _get_current_spread(self) -> float:
        """Get current spread value."""
        if self.spread_data is not None and self.data_idx < len(self.spread_data):
            return float(self.spread_data.iloc[self.data_idx].get("spread", 0))
        elif self.generated_spread and self.data_idx < len(self.generated_spread):
            return self.generated_spread[self.data_idx]
        return 0.0

    def _get_spread_change(self) -> float:
        """Get spread change from previous bar."""
        if self.spread_data is not None:
            if self.data_idx > 0 and self.data_idx < len(self.spread_data):
                current = self.spread_data.iloc[self.data_idx].get("spread", 0)
                prev = self.spread_data.iloc[self.data_idx - 1].get("spread", 0)
                return float(current - prev)
        elif self.generated_spread:
            if self.data_idx > 0 and self.data_idx < len(self.generated_spread):
                return (
                    self.generated_spread[self.data_idx]
                    - self.generated_spread[self.data_idx - 1]
                )
        return 0.0

    def _get_spread_zscore(self) -> float:
        """Get spread z-score."""
        if self.spread_data is not None and self.data_idx >= self.zscore_lookback:
            recent = self.spread_data.iloc[
                self.data_idx - self.zscore_lookback : self.data_idx + 1
            ]["spread"]
            mean = recent.mean()
            std = recent.std()
            current = self.spread_data.iloc[self.data_idx]["spread"]
            return float((current - mean) / (std + 1e-8))
        elif self.generated_spread and self.data_idx >= self.zscore_lookback:
            recent = self.generated_spread[
                self.data_idx - self.zscore_lookback : self.data_idx + 1
            ]
            mean = np.mean(recent)
            std = np.std(recent)
            return (self.generated_spread[self.data_idx] - mean) / (std + 1e-8)
        return 0.0

    def _get_spread_momentum(self, lookback: int) -> float:
        """Get spread momentum."""
        if self.spread_data is not None and self.data_idx >= lookback:
            current = self.spread_data.iloc[self.data_idx]["spread"]
            past = self.spread_data.iloc[self.data_idx - lookback]["spread"]
            return float(current - past)
        elif self.generated_spread and self.data_idx >= lookback:
            return (
                self.generated_spread[self.data_idx]
                - self.generated_spread[self.data_idx - lookback]
            )
        return 0.0

    def _get_spread_percentile(self) -> float:
        """Get spread percentile rank."""
        if self.spread_data is not None and self.data_idx >= self.zscore_lookback:
            recent = self.spread_data.iloc[
                self.data_idx - self.zscore_lookback : self.data_idx + 1
            ]["spread"]
            current = self.spread_data.iloc[self.data_idx]["spread"]
            return float((recent <= current).mean())
        return 0.5

    def _get_volatility_regime(self) -> float:
        """
        Get volatility regime indicator.

        Returns a value in [0, 1] representing the percentile rank of current
        volatility relative to historical volatility. Higher values indicate
        higher volatility regimes.

        Computed from rolling standard deviation of spread returns.
        """
        # Use precomputed cache if available
        if self._volatility_cache is not None and self.data_idx < len(
            self._volatility_cache
        ):
            return float(self._volatility_cache.iloc[self.data_idx])

        # Compute on-the-fly for synthetic data
        if self.generated_spread and self.data_idx >= self.volatility_lookback:
            recent = self.generated_spread[
                self.data_idx - self.volatility_lookback : self.data_idx + 1
            ]
            returns = np.diff(recent) / (np.array(recent[:-1]) + 1e-8)
            current_vol = np.std(returns)

            # Compare to longer history for percentile
            if self.data_idx >= self.volatility_lookback * 2:
                historical = self.generated_spread[: self.data_idx + 1]
                historical_returns = np.diff(historical) / (
                    np.array(historical[:-1]) + 1e-8
                )
                # Rolling volatility
                vol_series = (
                    pd.Series(historical_returns)
                    .rolling(self.volatility_lookback)
                    .std()
                    .dropna()
                )
                if len(vol_series) > 0:
                    return float((vol_series <= current_vol).mean())
            return 0.5  # Neutral for insufficient history

        return 0.5  # Neutral default

    def _get_correlation(self) -> float:
        """
        Get WTI-Brent correlation.

        Returns rolling correlation between WTI and Brent returns.
        If wti/brent columns not available, returns neutral value (0.9)
        representing the typical high correlation between these benchmarks.
        """
        # Use precomputed cache if available
        if self._correlation_cache is not None and self.data_idx < len(
            self._correlation_cache
        ):
            return float(self._correlation_cache.iloc[self.data_idx])

        # Log warning once if data not available
        if self.spread_data is not None:
            has_wti = "wti" in self.spread_data.columns
            has_brent = "brent" in self.spread_data.columns

            if not (has_wti and has_brent) and not self._warned_missing_wti_brent:
                logger.warning(
                    "SpreadEnvironment: 'wti' and/or 'brent' columns not found. "
                    "Correlation feature will use neutral value (0.9). "
                    "Add these columns for more accurate correlation computation."
                )
                self._warned_missing_wti_brent = True

        # Return typical high correlation as neutral value
        return 0.9

    def _get_usd_regime(self) -> float:
        """
        Get USD regime indicator.

        Returns value in [-1, 1] indicating USD strength regime.
        Requires 'usd' column in spread_data (e.g., DXY or UUP).

        If not available, returns neutral value (0.0).
        """
        if self.spread_data is not None and "usd" in self.spread_data.columns:
            if self.data_idx >= self.zscore_lookback:
                usd = self.spread_data["usd"]
                recent = usd.iloc[
                    self.data_idx - self.zscore_lookback : self.data_idx + 1
                ]
                current = usd.iloc[self.data_idx]
                mean = recent.mean()
                std = recent.std()
                if std > 1e-8:
                    zscore = (current - mean) / std
                    # Clip to [-1, 1] range
                    return float(np.clip(zscore / 2, -1, 1))
            return 0.0

        # Log warning once
        if not self._warned_missing_usd and self.spread_data is not None:
            logger.debug(
                "SpreadEnvironment: 'usd' column not found. "
                "USD regime feature will use neutral value (0.0). "
                "Add 'usd' column (e.g., DXY index) for this feature."
            )
            self._warned_missing_usd = True

        return 0.0  # Neutral

    def _get_curve_shape(self) -> float:
        """
        Get curve shape (contango/backwardation) indicator.

        Returns value in [-1, 1] where:
        - Positive: Contango (futures > spot)
        - Negative: Backwardation (futures < spot)

        Requires 'curve' column in spread_data.
        If not available, returns neutral value (0.0).
        """
        if self.spread_data is not None and "curve" in self.spread_data.columns:
            curve_val = self.spread_data.iloc[self.data_idx]["curve"]
            # Clip to [-1, 1] range
            return float(np.clip(curve_val, -1, 1))

        # Log warning once
        if not self._warned_missing_curve and self.spread_data is not None:
            logger.debug(
                "SpreadEnvironment: 'curve' column not found. "
                "Curve shape feature will use neutral value (0.0). "
                "Add 'curve' column for contango/backwardation signals."
            )
            self._warned_missing_curve = True

        return 0.0  # Neutral

    def get_state_dim(self) -> int:
        """Return state dimension."""
        return 12

    def get_action_dim(self) -> int:
        """Return action dimension."""
        return 5
