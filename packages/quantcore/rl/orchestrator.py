"""
RL Orchestrator - Coordinates all RL layers.

Integrates Execution, Position Sizing, Alpha Selection, and Spread RL.
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd
from loguru import logger

from quantcore.rl.base import State, Action, Reward
from quantcore.rl.execution.agent import ExecutionRLAgent
from quantcore.rl.execution.environment import ExecutionEnvironment, ExecutionOrder
from quantcore.rl.sizing.agent import SizingRLAgent
from quantcore.rl.sizing.environment import SizingEnvironment, TradingSignal
from quantcore.rl.meta.agent import AlphaSelectionAgent
from quantcore.rl.meta.environment import AlphaSelectionEnvironment
from quantcore.rl.spread.agent import SpreadArbitrageAgent
from quantcore.rl.spread.environment import SpreadEnvironment

# AlphaSignal is defined inline to avoid circular imports
from dataclasses import dataclass as _dataclass
from typing import Protocol


class AlphaSignal(Protocol):
    """Protocol for alpha signals."""

    name: str
    value: float
    confidence: float


from quantcore.data.base import AssetClass


@dataclass
class RLDecision:
    """Decision from RL orchestrator."""

    selected_alpha: str
    alpha_weight: float
    position_scale: float
    execution_strategy: str
    spread_action: Optional[str]
    confidence: float
    metadata: Dict[str, Any]


class RLOrchestrator:
    """
    Orchestrates all RL layers for trading decisions.

    Flow:
    1. Alpha Selection RL chooses which alpha(s) to follow
    2. Position Sizing RL determines position size
    3. Spread RL handles spread-specific decisions
    4. Execution RL optimizes order execution

    The orchestrator coordinates these layers and provides
    unified trading decisions.
    """

    def __init__(
        self,
        alpha_names: Optional[List[str]] = None,
        enable_execution_rl: bool = True,
        enable_sizing_rl: bool = True,
        enable_meta_rl: bool = True,
        enable_spread_rl: bool = True,
        device: str = "cpu",
    ):
        """
        Initialize RL orchestrator.

        Args:
            alpha_names: Names of available alphas
            enable_execution_rl: Enable execution optimization
            enable_sizing_rl: Enable position sizing
            enable_meta_rl: Enable alpha selection
            enable_spread_rl: Enable spread trading
            device: Device for RL agents
        """
        self.alpha_names = alpha_names or [
            "WTI_BRENT_SPREAD",
            "CRACK_SPREAD",
            "EIA_INVENTORY",
            "MICROSTRUCTURE",
            "COMMODITY_REGIME",
            "CROSS_ASSET",
            "MACRO",
        ]
        self.device = device

        # Feature flags
        self.enable_execution_rl = enable_execution_rl
        self.enable_sizing_rl = enable_sizing_rl
        self.enable_meta_rl = enable_meta_rl
        self.enable_spread_rl = enable_spread_rl

        # Initialize agents
        self._init_agents()

        # State tracking
        self.current_regime: Optional[Dict] = None
        self.alpha_performance: Dict[str, List[float]] = {
            name: [] for name in self.alpha_names
        }

    def _init_agents(self) -> None:
        """Initialize RL agents."""
        # Meta agent (alpha selection)
        if self.enable_meta_rl:
            meta_env = AlphaSelectionEnvironment(self.alpha_names)
            self.meta_agent = AlphaSelectionAgent(
                state_dim=meta_env.get_state_dim(),
                action_dim=meta_env.get_action_dim(),
                device=self.device,
            )
            self.meta_env = meta_env
        else:
            self.meta_agent = None
            self.meta_env = None

        # Sizing agent
        if self.enable_sizing_rl:
            self.sizing_agent = SizingRLAgent(
                state_dim=10,
                action_dim=1,
                device=self.device,
            )
            self.sizing_env = SizingEnvironment()
        else:
            self.sizing_agent = None
            self.sizing_env = None

        # Execution agent
        if self.enable_execution_rl:
            self.execution_agent = ExecutionRLAgent(
                state_dim=8,
                action_dim=5,
                device=self.device,
            )
            self.execution_env = ExecutionEnvironment()
        else:
            self.execution_agent = None
            self.execution_env = None

        # Spread agent
        if self.enable_spread_rl:
            self.spread_agent = SpreadArbitrageAgent(
                state_dim=12,
                action_dim=5,
                device=self.device,
            )
            self.spread_env = SpreadEnvironment()
        else:
            self.spread_agent = None
            self.spread_env = None

    def decide(
        self,
        alpha_signals: List[AlphaSignal],
        market_features: Dict[str, float],
        regime_info: Optional[Dict] = None,
        spread_data: Optional[pd.DataFrame] = None,
    ) -> RLDecision:
        """
        Make trading decision using all RL layers.

        Args:
            alpha_signals: Signals from all alphas
            market_features: Current market features
            regime_info: Current regime information
            spread_data: Spread data for spread RL

        Returns:
            RLDecision with coordinated decision
        """
        self.current_regime = regime_info

        # Step 1: Alpha Selection
        selected_alpha, alpha_weight = self._select_alpha(
            alpha_signals, market_features, regime_info
        )

        # Step 2: Get signal from selected alpha
        selected_signal = self._get_signal_for_alpha(alpha_signals, selected_alpha)

        # Step 3: Position Sizing
        position_scale = self._determine_position_size(
            selected_signal, market_features, regime_info
        )

        # Step 4: Spread Decision (if applicable)
        spread_action = None
        if self.enable_spread_rl and spread_data is not None:
            spread_action = self._get_spread_action(spread_data, market_features)

        # Step 5: Execution Strategy
        execution_strategy = self._determine_execution_strategy(
            position_scale, market_features
        )

        # Calculate overall confidence
        confidence = self._calculate_confidence(
            selected_signal, alpha_weight, position_scale
        )

        return RLDecision(
            selected_alpha=selected_alpha,
            alpha_weight=alpha_weight,
            position_scale=position_scale,
            execution_strategy=execution_strategy,
            spread_action=spread_action,
            confidence=confidence,
            metadata={
                "regime": regime_info,
                "n_signals": len(alpha_signals),
                "selected_signal": (
                    selected_signal.__dict__ if selected_signal else None
                ),
            },
        )

    def _select_alpha(
        self,
        alpha_signals: List[AlphaSignal],
        market_features: Dict[str, float],
        regime_info: Optional[Dict],
    ) -> Tuple[str, float]:
        """Select alpha using Meta RL."""
        if not self.enable_meta_rl or self.meta_agent is None:
            # Fallback: use highest confidence signal
            if alpha_signals:
                best = max(alpha_signals, key=lambda s: s.confidence)
                return best.alpha_name, best.confidence
            return self.alpha_names[0], 0.5

        # Build state for meta agent
        state = self._build_meta_state(alpha_signals, market_features, regime_info)

        # Get action
        action = self.meta_agent.select_action(state, explore=False)
        action_idx = int(action.value)

        # Map to alpha name
        if action_idx < len(self.alpha_names):
            selected_alpha = self.alpha_names[action_idx]
        else:
            selected_alpha = "NO_TRADE"

        # Get weight (confidence)
        alpha_weight = self._get_alpha_weight(selected_alpha, alpha_signals)

        return selected_alpha, alpha_weight

    def _build_meta_state(
        self,
        alpha_signals: List[AlphaSignal],
        market_features: Dict[str, float],
        regime_info: Optional[Dict],
    ) -> State:
        """Build state for meta agent."""
        features = []

        # Regime features (4)
        if regime_info:
            regime_type = regime_info.get("regime", "MACRO_DRIVEN")
            regime_map = {
                "INVENTORY_DRIVEN": [1, 0, 0, 0],
                "MACRO_DRIVEN": [0, 1, 0, 0],
                "USD_DRIVEN": [0, 0, 1, 0],
                "VOLATILITY_DRIVEN": [0, 0, 0, 1],
            }
            features.extend(regime_map.get(regime_type, [0, 0, 0, 1]))
        else:
            features.extend([0, 0, 0, 1])

        # Per-alpha features (4 each)
        signal_map = {s.alpha_name: s for s in alpha_signals}

        for name in self.alpha_names:
            if name in signal_map:
                signal = signal_map[name]
                # Recent Sharpe estimate
                sharpe = self._get_alpha_sharpe(name)
                # Recent return estimate
                recent_return = self._get_alpha_recent_return(name)
                # Hit rate estimate
                hit_rate = self._get_alpha_hit_rate(name)
                # Regime alignment
                alignment = self._get_regime_alignment(name, regime_info)

                features.extend([sharpe, recent_return, hit_rate, alignment])
            else:
                features.extend([0, 0, 0.5, 0.5])

        # Market features (4)
        features.extend(
            [
                market_features.get("volatility", 0.5),
                market_features.get("correlation_regime", 0),
                market_features.get("usd_regime", 0),
                market_features.get("vix_level", 0.5),
            ]
        )

        return State(features=np.array(features, dtype=np.float32))

    def _determine_position_size(
        self,
        signal: Optional[AlphaSignal],
        market_features: Dict[str, float],
        regime_info: Optional[Dict],
    ) -> float:
        """Determine position size using Sizing RL."""
        if not self.enable_sizing_rl or self.sizing_agent is None or signal is None:
            # Fallback: use signal confidence
            return signal.confidence if signal else 0.0

        # Build state for sizing agent
        state = self._build_sizing_state(signal, market_features, regime_info)

        # Get action
        action = self.sizing_agent.select_action(state, explore=False)

        if isinstance(action.value, np.ndarray):
            scale = float(np.clip(action.value[0], 0, 1))
        else:
            scale = float(np.clip(action.value, 0, 1))

        return scale

    def _build_sizing_state(
        self,
        signal: AlphaSignal,
        market_features: Dict[str, float],
        regime_info: Optional[Dict],
    ) -> State:
        """Build state for sizing agent."""
        direction = (
            1
            if signal.direction == "LONG"
            else (-1 if signal.direction == "SHORT" else 0)
        )

        features = np.array(
            [
                signal.confidence,
                direction,
                market_features.get("volatility", 0.5),
                market_features.get("drawdown", 0),
                market_features.get("risk_budget_used", 0),
                market_features.get("rolling_sharpe", 0),
                market_features.get("current_position", 0),
                market_features.get("time_since_trade", 0),
                self._get_regime_indicator(regime_info),
                market_features.get("win_rate", 0.5),
            ],
            dtype=np.float32,
        )

        return State(features=features)

    def _get_spread_action(
        self,
        spread_data: pd.DataFrame,
        market_features: Dict[str, float],
    ) -> str:
        """Get spread trading action."""
        if self.spread_agent is None:
            return "HOLD"

        # Build spread state
        state = self._build_spread_state(spread_data, market_features)

        # Get action
        action = self.spread_agent.select_action(state, explore=False)
        action_idx = int(action.value)

        action_map = {
            0: "CLOSE",
            1: "SMALL_LONG",
            2: "FULL_LONG",
            3: "SMALL_SHORT",
            4: "FULL_SHORT",
        }

        return action_map.get(action_idx, "HOLD")

    def _build_spread_state(
        self,
        spread_data: pd.DataFrame,
        market_features: Dict[str, float],
    ) -> State:
        """Build state for spread agent."""
        # Extract spread features from data
        if spread_data is not None and len(spread_data) > 0:
            current = spread_data.iloc[-1]
            zscore = float(current.get("spread_zscore", 0))
            mom_5 = float(current.get("spread_roc_5", 0))
            mom_20 = float(current.get("spread_roc_10", 0))
            percentile = float(current.get("spread_percentile", 0.5))
        else:
            zscore, mom_5, mom_20, percentile = 0, 0, 0, 0.5

        features = np.array(
            [
                zscore / 3,
                mom_5 * 100,
                mom_20 * 100,
                percentile,
                0,  # position direction (would track)
                0,  # position size
                0,  # unrealized pnl
                0,  # bars held
                market_features.get("volatility", 0.5),
                market_features.get("correlation", 0.9),
                market_features.get("usd_regime", 0),
                market_features.get("curve_shape", 0),
            ],
            dtype=np.float32,
        )

        return State(features=features)

    def _determine_execution_strategy(
        self,
        position_scale: float,
        market_features: Dict[str, float],
    ) -> str:
        """Determine execution strategy."""
        volatility = market_features.get("volatility", 0.5)

        if position_scale < 0.1:
            return "NO_TRADE"

        # High urgency or low volatility: more aggressive
        if position_scale > 0.8 and volatility < 0.3:
            return "AGGRESSIVE"

        # High volatility: more passive
        if volatility > 0.7:
            return "PASSIVE"

        return "BALANCED"

    def _calculate_confidence(
        self,
        signal: Optional[AlphaSignal],
        alpha_weight: float,
        position_scale: float,
    ) -> float:
        """Calculate overall decision confidence."""
        if signal is None:
            return 0.0

        return signal.confidence * alpha_weight * position_scale

    def _get_signal_for_alpha(
        self,
        signals: List[AlphaSignal],
        alpha_name: str,
    ) -> Optional[AlphaSignal]:
        """Get signal for specific alpha."""
        for signal in signals:
            if signal.alpha_name == alpha_name:
                return signal
        return None

    def _get_alpha_weight(
        self,
        alpha_name: str,
        signals: List[AlphaSignal],
    ) -> float:
        """Get weight for selected alpha."""
        for signal in signals:
            if signal.alpha_name == alpha_name:
                return signal.confidence
        return 0.5

    def _get_alpha_sharpe(self, alpha_name: str) -> float:
        """Get estimated Sharpe for alpha."""
        returns = self.alpha_performance.get(alpha_name, [])
        if len(returns) >= 20:
            return (
                np.mean(returns[-20:]) / (np.std(returns[-20:]) + 1e-8) * np.sqrt(252)
            )
        return 0.0

    def _get_alpha_recent_return(self, alpha_name: str) -> float:
        """Get recent return for alpha."""
        returns = self.alpha_performance.get(alpha_name, [])
        if returns:
            return sum(returns[-20:])
        return 0.0

    def _get_alpha_hit_rate(self, alpha_name: str) -> float:
        """Get hit rate for alpha."""
        returns = self.alpha_performance.get(alpha_name, [])
        if returns:
            return np.mean([r > 0 for r in returns[-20:]])
        return 0.5

    def _get_regime_alignment(
        self,
        alpha_name: str,
        regime_info: Optional[Dict],
    ) -> float:
        """Get regime alignment score for alpha."""
        if regime_info is None:
            return 0.5

        regime = regime_info.get("regime", "MACRO_DRIVEN")

        alignments = {
            "INVENTORY_DRIVEN": {
                "EIA_INVENTORY": 0.9,
                "MICROSTRUCTURE": 0.7,
                "COMMODITY_REGIME": 0.6,
            },
            "MACRO_DRIVEN": {
                "MACRO": 0.9,
                "CROSS_ASSET": 0.8,
                "COMMODITY_REGIME": 0.6,
            },
            "USD_DRIVEN": {
                "CROSS_ASSET": 0.8,
                "MACRO": 0.7,
                "WTI_BRENT_SPREAD": 0.5,
            },
            "VOLATILITY_DRIVEN": {
                "MICROSTRUCTURE": 0.8,
                "EIA_INVENTORY": 0.6,
                "CRACK_SPREAD": 0.5,
            },
        }

        return alignments.get(regime, {}).get(alpha_name, 0.4)

    def _get_regime_indicator(self, regime_info: Optional[Dict]) -> float:
        """Get regime indicator as float."""
        if regime_info is None:
            return 0.0

        regime = regime_info.get("regime", "MACRO_DRIVEN")
        regime_map = {
            "INVENTORY_DRIVEN": -1.0,
            "MACRO_DRIVEN": 0.0,
            "USD_DRIVEN": 0.5,
            "VOLATILITY_DRIVEN": 1.0,
        }
        return regime_map.get(regime, 0.0)

    def update_alpha_performance(
        self,
        alpha_name: str,
        return_value: float,
    ) -> None:
        """Update alpha performance tracking."""
        if alpha_name in self.alpha_performance:
            self.alpha_performance[alpha_name].append(return_value)
            # Keep only recent history
            if len(self.alpha_performance[alpha_name]) > 100:
                self.alpha_performance[alpha_name] = self.alpha_performance[alpha_name][
                    -100:
                ]

    def execute_order(
        self,
        order: ExecutionOrder,
        market_data: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        Execute order using Execution RL.

        Args:
            order: Order to execute
            market_data: Market data for execution

        Returns:
            Execution results
        """
        if not self.enable_execution_rl or self.execution_agent is None:
            return {"status": "SIMULATED", "fill_price": order.arrival_price}

        # Set up execution environment
        self.execution_env.set_order(order)
        if market_data is not None:
            self.execution_env.data = market_data

        state = self.execution_env.reset()

        # Run execution episode
        total_executed = 0
        total_value = 0
        execution_log = []

        while not self.execution_env.done:
            action = self.execution_agent.select_action(state, explore=False)
            state, reward, done, info = self.execution_env.step(action)

            execution_log.append(
                {
                    "action": action.value,
                    "executed_qty": info["executed_qty"],
                    "fill_price": info["fill_price"],
                    "shortfall": info["shortfall"],
                }
            )

            total_executed += info["executed_qty"]
            if info["executed_qty"] > 0:
                total_value += info["executed_qty"] * info["fill_price"]

        avg_fill = (
            total_value / total_executed if total_executed > 0 else order.arrival_price
        )

        return {
            "status": "EXECUTED",
            "total_quantity": total_executed,
            "avg_fill_price": avg_fill,
            "implementation_shortfall": (avg_fill - order.arrival_price)
            / order.arrival_price,
            "execution_log": execution_log,
        }

    def train(self) -> None:
        """Set all agents to training mode."""
        if self.meta_agent:
            self.meta_agent.train()
        if self.sizing_agent:
            self.sizing_agent.train()
        if self.execution_agent:
            self.execution_agent.train()
        if self.spread_agent:
            self.spread_agent.train()

    def eval(self) -> None:
        """Set all agents to evaluation mode."""
        if self.meta_agent:
            self.meta_agent.eval()
        if self.sizing_agent:
            self.sizing_agent.eval()
        if self.execution_agent:
            self.execution_agent.eval()
        if self.spread_agent:
            self.spread_agent.eval()

    def save(self, path: str) -> None:
        """Save all agents."""
        import os

        os.makedirs(path, exist_ok=True)

        if self.meta_agent:
            self.meta_agent.save(os.path.join(path, "meta_agent.pt"))
        if self.sizing_agent:
            self.sizing_agent.save(os.path.join(path, "sizing_agent.pt"))
        if self.execution_agent:
            self.execution_agent.save(os.path.join(path, "execution_agent.pt"))
        if self.spread_agent:
            self.spread_agent.save(os.path.join(path, "spread_agent.pt"))

        logger.info(f"All agents saved to {path}")

    def load(self, path: str) -> None:
        """Load all agents."""
        import os

        if self.meta_agent and os.path.exists(os.path.join(path, "meta_agent.pt")):
            self.meta_agent.load(os.path.join(path, "meta_agent.pt"))
        if self.sizing_agent and os.path.exists(os.path.join(path, "sizing_agent.pt")):
            self.sizing_agent.load(os.path.join(path, "sizing_agent.pt"))
        if self.execution_agent and os.path.exists(
            os.path.join(path, "execution_agent.pt")
        ):
            self.execution_agent.load(os.path.join(path, "execution_agent.pt"))
        if self.spread_agent and os.path.exists(os.path.join(path, "spread_agent.pt")):
            self.spread_agent.load(os.path.join(path, "spread_agent.pt"))

        logger.info(f"All agents loaded from {path}")
