"""
Alpha Selection environment for RL.

Simulates selecting which alpha to follow based on regime.

MATURITY: EXPERIMENTAL
This environment is designed for meta-learning research and generates
SYNTHETIC alpha returns. It does not use real market data.

Use cases:
- Research into alpha combination strategies
- Testing meta-learning approaches
- Educational purposes

NOT suitable for:
- Live trading decisions
- Backtesting with real data
- Production deployment
"""

from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
import numpy as np
import pandas as pd
from loguru import logger

from quantcore.rl.base import RLEnvironment, State, Action, Reward


@dataclass
class AlphaPerformance:
    """Performance metrics for an alpha."""

    name: str
    recent_sharpe: float
    recent_return: float
    hit_rate: float
    regime_alignment: float  # How well alpha aligns with current regime


@dataclass
class AlphaSelectionDataRequirements:
    """
    Documents what data this environment needs.

    EXPERIMENTAL: This environment generates synthetic data for all features.
    Real alpha returns and market data can be provided but are optional.

    Optional data:
        - market_data: DataFrame with 'volatility', 'vix' columns
        - alpha_returns: Dict[str, pd.Series] of historical alpha returns
    """

    synthetic_by_design: bool = True


class AlphaSelectionEnvironment(RLEnvironment):
    """
    Alpha selection environment.

    MATURITY: EXPERIMENTAL

    WARNING: This environment generates SYNTHETIC alpha returns for research
    purposes. The regime-dependent performance characteristics are simulated,
    not derived from real market data.

    State space (variable, depends on n_alphas):
    - Regime features (4): HMM state probs, changepoint prob, volatility, trend
    - Per-alpha features (4 each): recent Sharpe, recent return, hit rate, regime alignment
    - Market features (4): volatility, correlation regime, USD regime, VIX level

    Action space (discrete):
    - Select one of N alphas
    - Or "no trade" option

    Reward:
    - Return of selected alpha
    - Regime consistency bonus
    """

    # Mark as experimental
    MATURITY = "EXPERIMENTAL"
    DATA_REQUIREMENTS = AlphaSelectionDataRequirements()

    def __init__(
        self,
        alpha_names: Optional[List[str]] = None,
        lookback: int = 20,
        include_no_trade: bool = True,
        market_data: Optional[pd.DataFrame] = None,
        alpha_returns: Optional[Dict[str, pd.Series]] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize alpha selection environment.

        EXPERIMENTAL: Generates synthetic alpha returns for meta-learning research.

        Args:
            alpha_names: Names of alphas to select from
            lookback: Lookback for performance calculation
            include_no_trade: Include "no trade" as an option
            market_data: Optional DataFrame with 'volatility', 'vix' columns
            alpha_returns: Optional dict of historical alpha returns
            seed: Random seed for reproducibility
        """
        super().__init__()

        # Log experimental warning
        logger.warning(
            "AlphaSelectionEnvironment is EXPERIMENTAL. "
            "It generates synthetic alpha returns for research purposes. "
            "Do not use for production trading decisions."
        )

        self.alpha_names = alpha_names or [
            "WTI_BRENT_SPREAD",
            "CRACK_SPREAD",
            "EIA_INVENTORY",
            "MICROSTRUCTURE",
            "COMMODITY_REGIME",
            "CROSS_ASSET",
            "MACRO",
        ]
        self.n_alphas = len(self.alpha_names)
        self.lookback = lookback
        self.include_no_trade = include_no_trade
        self.market_data = market_data
        self.alpha_returns_data = alpha_returns
        self.seed = seed

        # Random state for reproducibility
        self._rng = np.random.RandomState(seed)

        # Total actions
        self.n_actions = self.n_alphas + (1 if include_no_trade else 0)

        # State dimension
        # Regime features (4) + per-alpha features (4 * n_alphas) + market features (4)
        self.state_dim_calc = 4 + 4 * self.n_alphas + 4

        # Simulation state
        self.alpha_returns: Dict[str, List[float]] = {
            name: [] for name in self.alpha_names
        }
        self.alpha_signals: Dict[str, List[int]] = {
            name: [] for name in self.alpha_names
        }
        self.selected_alpha_history: List[int] = []
        self.regime_history: List[int] = []

        # Track data idx for market_data
        self.data_idx = 0

        # Warning flags
        self._warned_synthetic = False

    def reset(self) -> State:
        """Reset environment."""
        self.current_step = 0
        self.episode_reward = 0.0
        self.done = False
        self.data_idx = 0

        # Reset RNG for reproducibility
        if self.seed is not None:
            self._rng = np.random.RandomState(
                self.seed + self.episode_count
                if hasattr(self, "episode_count")
                else self.seed
            )

        # Reset histories
        self.alpha_returns = {name: [] for name in self.alpha_names}
        self.alpha_signals = {name: [] for name in self.alpha_names}
        self.selected_alpha_history = []
        self.regime_history = []

        # Generate initial history
        self._generate_initial_history()

        return self._get_state()

    def step(self, action: Action) -> Tuple[State, Reward, bool, Dict[str, Any]]:
        """
        Select alpha and observe outcome.

        Args:
            action: Alpha index to select

        Returns:
            (next_state, reward, done, info)
        """
        action_idx = int(action.value) if action.is_discrete else int(action.value)

        # Generate new alpha returns for this step
        alpha_returns = self._generate_alpha_returns()

        # Store returns
        for name, ret in alpha_returns.items():
            self.alpha_returns[name].append(ret)

        # Get return of selected alpha
        if self.include_no_trade and action_idx == self.n_alphas:
            # No trade selected
            selected_return = 0.0
            selected_alpha = "NO_TRADE"
        else:
            selected_alpha = self.alpha_names[action_idx]
            selected_return = alpha_returns[selected_alpha]

        self.selected_alpha_history.append(action_idx)

        # Calculate reward
        reward = self._calculate_reward(selected_return, action_idx, alpha_returns)

        # Update regime
        self.regime_history.append(self._get_current_regime())

        # Advance data index
        self.data_idx += 1

        # Check termination
        self.current_step += 1
        self.done = self.current_step >= 100  # ~100 trading days

        info = {
            "selected_alpha": selected_alpha,
            "selected_return": selected_return,
            "all_returns": alpha_returns,
            "best_alpha": max(alpha_returns, key=alpha_returns.get),
            "best_return": max(alpha_returns.values()),
            "is_synthetic": True,  # Always mark as synthetic
        }

        return self._get_state(), reward, self.done, info

    def _get_state(self) -> State:
        """Get current state."""
        features = []

        # Regime features (4)
        regime = self._get_current_regime()
        regime_one_hot = [0, 0, 0, 0]
        regime_one_hot[regime] = 1
        features.extend(regime_one_hot)

        # Per-alpha features (4 * n_alphas)
        for name in self.alpha_names:
            returns = self.alpha_returns[name]

            # Recent Sharpe
            if len(returns) >= self.lookback:
                recent = returns[-self.lookback :]
                sharpe = np.mean(recent) / (np.std(recent) + 1e-8) * np.sqrt(252)
            else:
                sharpe = 0.0

            # Recent return
            recent_return = np.sum(returns[-self.lookback :]) if returns else 0.0

            # Hit rate
            if returns:
                hit_rate = np.mean([r > 0 for r in returns[-self.lookback :]])
            else:
                hit_rate = 0.5

            # Regime alignment (synthetic)
            regime_alignment = self._get_regime_alignment(name, regime)

            features.extend(
                [
                    sharpe / 3,  # Normalize
                    recent_return * 10,  # Scale
                    hit_rate,
                    regime_alignment,
                ]
            )

        # Market features (4) - computed from data if available, else neutral
        features.extend(
            [
                self._get_volatility(),
                self._get_correlation_regime(),
                self._get_usd_regime(),
                self._get_vix_level(),
            ]
        )

        return State(
            features=np.array(features, dtype=np.float32),
            metadata={
                "regime": regime,
                "step": self.current_step,
                "is_synthetic": True,
            },
        )

    def _generate_initial_history(self) -> None:
        """Generate initial performance history."""
        for _ in range(self.lookback):
            alpha_returns = self._generate_alpha_returns()
            for name, ret in alpha_returns.items():
                self.alpha_returns[name].append(ret)
            self.regime_history.append(self._get_current_regime())

    def _generate_alpha_returns(self) -> Dict[str, float]:
        """
        Generate alpha returns for current step.

        NOTE: These are SYNTHETIC returns with regime-dependent characteristics.
        They do not represent real market behavior.
        """
        # Use real alpha returns if provided
        if self.alpha_returns_data is not None:
            returns = {}
            for name in self.alpha_names:
                if name in self.alpha_returns_data:
                    series = self.alpha_returns_data[name]
                    if self.data_idx < len(series):
                        returns[name] = float(series.iloc[self.data_idx])
                    else:
                        returns[name] = self._rng.normal(0.001, 0.01)
                else:
                    returns[name] = self._rng.normal(0.001, 0.01)
            return returns

        # Generate synthetic returns
        regime = self._get_current_regime()
        returns = {}

        # Different alpha characteristics by regime
        for name in self.alpha_names:
            # Base return with some noise (use seeded RNG)
            base_return = self._rng.normal(0.001, 0.01)

            # Regime-dependent adjustment
            if regime == 0:  # Low vol bull
                if name in ["COMMODITY_REGIME", "MACRO"]:
                    base_return += 0.002
                elif name in ["MICROSTRUCTURE"]:
                    base_return -= 0.001
            elif regime == 1:  # High vol bull
                if name in ["EIA_INVENTORY", "MICROSTRUCTURE"]:
                    base_return += 0.003
            elif regime == 2:  # Low vol bear
                if name in ["WTI_BRENT_SPREAD", "CRACK_SPREAD"]:
                    base_return += 0.002
            elif regime == 3:  # High vol bear
                if name in ["CROSS_ASSET", "MACRO"]:
                    base_return += 0.003
                elif name in ["COMMODITY_REGIME"]:
                    base_return -= 0.002

            returns[name] = base_return

        return returns

    def _get_current_regime(self) -> int:
        """
        Get current regime (0-3).

        Uses Markov chain with transition matrix for regime simulation.
        """
        # Simulate regime transitions with seeded RNG
        if len(self.regime_history) > 0:
            prev_regime = self.regime_history[-1]
            # Small probability of regime change
            if self._rng.random() < 0.05:
                return self._rng.randint(4)
            return prev_regime
        return self._rng.randint(4)

    def _get_regime_alignment(self, alpha_name: str, regime: int) -> float:
        """Get how well an alpha aligns with current regime."""
        alignments = {
            0: {"COMMODITY_REGIME": 0.8, "MACRO": 0.7, "WTI_BRENT_SPREAD": 0.6},
            1: {"EIA_INVENTORY": 0.8, "MICROSTRUCTURE": 0.7, "CROSS_ASSET": 0.6},
            2: {"WTI_BRENT_SPREAD": 0.8, "CRACK_SPREAD": 0.7, "MACRO": 0.5},
            3: {"CROSS_ASSET": 0.8, "MACRO": 0.7, "MICROSTRUCTURE": 0.6},
        }
        return alignments.get(regime, {}).get(alpha_name, 0.4)

    def _get_volatility(self) -> float:
        """
        Get normalized volatility.

        Uses market_data if available, otherwise returns neutral value.
        """
        if self.market_data is not None and "volatility" in self.market_data.columns:
            if self.data_idx < len(self.market_data):
                vol = self.market_data.iloc[self.data_idx]["volatility"]
                # Normalize to [0, 1] range
                return float(np.clip(vol / 0.5, 0, 1))

        # Return neutral value (0.3 = moderate volatility)
        return 0.3

    def _get_correlation_regime(self) -> float:
        """
        Get correlation regime indicator.

        Returns neutral value (0.0) as this requires complex multi-asset data.
        """
        return 0.0  # Neutral

    def _get_usd_regime(self) -> float:
        """
        Get USD regime indicator.

        Uses market_data if available, otherwise returns neutral value.
        """
        if self.market_data is not None and "usd" in self.market_data.columns:
            if self.data_idx < len(self.market_data):
                # Assume z-score in [-3, 3], normalize to [-1, 1]
                usd = self.market_data.iloc[self.data_idx]["usd"]
                return float(np.clip(usd / 3, -1, 1))

        return 0.0  # Neutral

    def _get_vix_level(self) -> float:
        """
        Get normalized VIX level.

        Uses market_data if available, otherwise returns neutral value.
        """
        if self.market_data is not None and "vix" in self.market_data.columns:
            if self.data_idx < len(self.market_data):
                vix = self.market_data.iloc[self.data_idx]["vix"]
                # Normalize: VIX 10-50 maps to 0-1
                return float(np.clip((vix - 10) / 40, 0, 1))

        # Return neutral value (0.3 = VIX ~22)
        return 0.3

    def _calculate_reward(
        self,
        selected_return: float,
        action_idx: int,
        all_returns: Dict[str, float],
    ) -> Reward:
        """Calculate reward."""
        components = {}

        # Base reward: return of selected alpha
        components["return"] = selected_return * 100  # Scale

        # Regret penalty (vs best alpha)
        best_return = max(all_returns.values())
        regret = best_return - selected_return
        components["regret_penalty"] = -regret * 50

        # Regime consistency bonus
        regime = self._get_current_regime()
        if self.include_no_trade and action_idx == self.n_alphas:
            alignment = 0.3  # Neutral for no trade
        else:
            alpha_name = self.alpha_names[action_idx]
            alignment = self._get_regime_alignment(alpha_name, regime)
        components["regime_bonus"] = alignment * 0.5

        # Switching penalty (discourage frequent changes)
        if len(self.selected_alpha_history) >= 2:
            if self.selected_alpha_history[-1] != self.selected_alpha_history[-2]:
                components["switching_penalty"] = -0.1
            else:
                components["switching_penalty"] = 0.0
        else:
            components["switching_penalty"] = 0.0

        total = sum(components.values())

        return Reward(value=total, components=components)

    def get_state_dim(self) -> int:
        """Return state dimension."""
        return self.state_dim_calc

    def get_action_dim(self) -> int:
        """Return action dimension."""
        return self.n_actions
