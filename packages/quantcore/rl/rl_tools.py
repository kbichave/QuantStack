"""
RL Agent CrewAI Tools — exposes trained RL agents as first-class crew tools.

These three tools make RL agents available to any agent in the TradingCrew,
just like the existing 24 MCP bridge tools. The SuperTrader and risk_pod_manager
can call them to get ML-optimized recommendations:

    rl_position_size  → SizingRLAgent.select_action()
    rl_execution_strategy → ExecutionRLAgent.select_action()
    rl_alpha_weight   → AlphaSelectionAgent.select_action()

Shadow mode: when shadow_mode=True in config (the default), tool output is
tagged [SHADOW – not yet validated]. The LLM agent can read the recommendation
but it does not override decisions until the agent passes PromotionGate.

Graceful degradation: if no checkpoint exists or PyTorch is unavailable,
tools return {} and log a warning. Crew execution continues normally.

Usage in crews/tools.py:
    from quantcore.rl.rl_tools import get_rl_tools
    rl_tools = get_rl_tools(cfg)

Usage in trading_crew.py:
    super_trader.tools = get_execution_tools() + get_risk_tools() + get_rl_tools(cfg)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger
from pydantic import BaseModel, Field
from quant_pod.crewai_compat import BaseTool

from quantcore.rl.config import RLProductionConfig, get_rl_config
from quantcore.rl.features import RLFeatureExtractor

# ---------------------------------------------------------------------------
# Module-level pre-trade snapshot registry
# Keys: tool_name (one snapshot kept per tool per session — at most one
# rl_position_size and one rl_execution_strategy call per trading day).
# Written by RL tools at inference time; consumed by PostTradeRLAdapter
# in TradingDayFlow._run_post_trade_learning() via pop().
# ---------------------------------------------------------------------------
_PRETRADE_SNAPSHOTS: dict[str, Any] = {}


def save_pretrade_snapshot(tool_name: str, snapshot: dict[str, Any]) -> None:
    """Store a pre-trade snapshot for later retrieval by PostTradeRLAdapter."""
    _PRETRADE_SNAPSHOTS[tool_name] = snapshot


def pop_pretrade_snapshot(tool_name: str) -> dict[str, Any] | None:
    """Retrieve and remove the snapshot for a given tool. Returns None if absent."""
    return _PRETRADE_SNAPSHOTS.pop(tool_name, None)


# ---------------------------------------------------------------------------
# Input schemas (pydantic models for tool argument validation)
# ---------------------------------------------------------------------------


class RLPositionSizeInput(BaseModel):
    signal_confidence: float = Field(
        ..., ge=0.0, le=1.0, description="IC/pod manager signal confidence (0–1)"
    )
    signal_direction: str = Field(..., description="Signal direction: LONG, SHORT, or NEUTRAL")
    regime: str = Field(default="normal", description="Current market regime label")
    current_drawdown: float = Field(
        default=0.0, ge=0.0, description="Current portfolio drawdown fraction (e.g. 0.05)"
    )
    current_position_pct: float = Field(
        default=0.0, description="Current position as fraction of max allowed"
    )
    risk_budget_used: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Fraction of risk budget already used"
    )
    win_rate: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Rolling win rate over recent trades"
    )
    rolling_sharpe: float = Field(default=0.0, description="Rolling 20-period Sharpe ratio")
    time_since_trade: int = Field(default=0, ge=0, description="Number of bars since last trade")


class RLExecutionStrategyInput(BaseModel):
    symbol: str = Field(..., description="Trading symbol (e.g. SPY)")
    quantity: float = Field(..., gt=0, description="Order quantity (shares)")
    urgency: str = Field(default="normal", description="Urgency level: low, normal, high")
    arrival_price: float = Field(
        default=0.0, description="Price at decision time (0 = use last known)"
    )
    time_horizon_bars: int = Field(
        default=10, ge=1, description="Number of bars available to execute"
    )


class RLAlphaWeightInput(BaseModel):
    regime: str = Field(..., description="Current market regime (e.g. trending_up, ranging)")
    competing_signals: list[str] = Field(..., description="List of IC/alpha signal names to weight")
    market_volatility: float = Field(
        default=0.3, ge=0.0, description="Current normalized volatility"
    )
    vix_normalized: float = Field(
        default=0.3, ge=0.0, le=1.0, description="Normalized VIX level (0=10, 1=50)"
    )


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


class RLPositionSizeTool(BaseTool):
    """
    RL-recommended position scale factor (0.0–1.0).

    Calls the trained PPO SizingRLAgent with current regime, signal confidence,
    and portfolio context to recommend how large a position to take.

    Returns JSON with:
        scale (float 0-1): multiply against your base position size
        confidence (float): agent's confidence in this recommendation
        shadow (bool): True = recommendation is not yet validated, treat as advisory
        reasoning (str): human-readable explanation
    """

    name: str = "rl_position_size"
    description: str = (
        "Get RL-recommended position scale factor (0.0–1.0) based on current "
        "signal confidence, market regime, drawdown, and risk budget. "
        "Use this to dynamically size positions beyond fixed Kelly fractions. "
        "Input: signal_confidence (0-1), signal_direction (LONG/SHORT/NEUTRAL), "
        "regime (string), current_drawdown (fraction), risk_budget_used (0-1). "
        "Output: JSON with scale, confidence, shadow flag."
    )
    args_schema: type[BaseModel] = RLPositionSizeInput

    # Lazy-loaded agent (initialized once per session on first call)
    _agent: Any | None = None
    _config: RLProductionConfig | None = None

    def _run(
        self,
        signal_confidence: float,
        signal_direction: str = "LONG",
        regime: str = "normal",
        current_drawdown: float = 0.0,
        current_position_pct: float = 0.0,
        risk_budget_used: float = 0.0,
        win_rate: float = 0.5,
        rolling_sharpe: float = 0.0,
        time_since_trade: int = 0,
    ) -> str:
        cfg = self._get_config()

        if not cfg.enable_sizing_rl:
            return json.dumps(
                {
                    "scale": signal_confidence,
                    "shadow": True,
                    "reasoning": "Sizing RL disabled in config.",
                }
            )

        agent = self._load_agent(cfg)
        if agent is None:
            return json.dumps(
                {
                    "scale": signal_confidence,
                    "shadow": True,
                    "reasoning": "Sizing agent not loaded — no checkpoint.",
                }
            )

        # Build feature vector using canonical extractor
        features = RLFeatureExtractor.sizing_features(
            signal_confidence=signal_confidence,
            signal_direction=signal_direction,
            returns_window=[],  # no live returns at inference time — use defaults
            current_position_pct=current_position_pct,
            drawdown=current_drawdown,
            risk_budget_used=risk_budget_used,
            time_since_trade=time_since_trade,
            regime_label=regime,
            win_rate=win_rate,
            rolling_sharpe=rolling_sharpe,
        )

        from quantcore.rl.base import State

        state = State(features=features)

        try:
            agent.eval()
            action = agent.select_action(state, explore=False)
            if isinstance(action.value, np.ndarray):
                scale = float(np.clip(action.value[0], 0.0, 1.0))
            else:
                scale = float(np.clip(action.value, 0.0, 1.0))
        except Exception as exc:
            logger.warning(f"[RLPositionSizeTool] Inference failed: {exc}")
            return json.dumps(
                {"scale": signal_confidence, "shadow": True, "reasoning": f"Inference error: {exc}"}
            )

        is_shadow = cfg.sizing_shadow

        # Save pre-trade snapshot for PostTradeRLAdapter
        save_pretrade_snapshot(
            "rl_position_size",
            {
                "tool_name": "rl_position_size",
                "state_vector": features.tolist(),
                "action_value": scale,
                "volatility_at_entry": current_drawdown,  # best proxy available at call time
                "signal_confidence": signal_confidence,
            },
        )

        result = {
            "scale": round(scale, 4),
            "confidence": round(float(signal_confidence * scale), 4),
            "shadow": is_shadow,
            "reasoning": (
                f"RL sizing agent recommends {scale:.0%} of max position. "
                f"{'[SHADOW – not yet validated] ' if is_shadow else ''}"
                f"Based on confidence={signal_confidence:.2f}, "
                f"drawdown={current_drawdown:.1%}, regime={regime}."
            ),
        }
        return json.dumps(result)

    def _get_config(self) -> RLProductionConfig:
        if self._config is None:
            object.__setattr__(self, "_config", get_rl_config())
        return self._config

    def _load_agent(self, cfg: RLProductionConfig) -> Any | None:
        """Lazy-load sizing agent from checkpoint (once per session)."""
        if self._agent is not None:
            return self._agent

        checkpoint = cfg.sizing_checkpoint_path
        if not Path(checkpoint).exists():
            logger.debug(f"[RLPositionSizeTool] No checkpoint at {checkpoint}.")
            return None

        try:
            from quantcore.rl.sizing.agent import SizingRLAgent

            agent = SizingRLAgent(state_dim=cfg.sizing_state_dim, action_dim=1)
            agent.load(str(checkpoint))
            agent.eval()
            object.__setattr__(self, "_agent", agent)
            logger.info("[RLPositionSizeTool] Sizing agent loaded from checkpoint.")
            return agent
        except Exception as exc:
            logger.warning(f"[RLPositionSizeTool] Failed to load checkpoint: {exc}")
            return None


class RLExecutionStrategyTool(BaseTool):
    """
    RL-recommended order execution strategy.

    Calls the trained DQN ExecutionRLAgent to recommend how to slice and
    time an order to minimize implementation shortfall.

    Returns JSON with:
        strategy: AGGRESSIVE | BALANCED | PASSIVE | TWAP | NO_TRADE
        order_fraction: fraction to execute immediately (0–1)
        confidence (float): agent's confidence
        shadow (bool): True = not yet validated
        reasoning (str)
    """

    name: str = "rl_execution_strategy"
    description: str = (
        "Get RL-recommended order execution strategy to minimize implementation shortfall. "
        "The agent recommends how aggressively to execute: AGGRESSIVE (market order), "
        "BALANCED (medium limit), PASSIVE (small limit/TWAP), or NO_TRADE. "
        "Input: symbol (str), quantity (float), urgency (low/normal/high), "
        "arrival_price (float, optional), time_horizon_bars (int, default 10). "
        "Output: JSON with strategy, order_fraction, confidence, shadow flag."
    )
    args_schema: type[BaseModel] = RLExecutionStrategyInput

    _agent: Any | None = None
    _config: RLProductionConfig | None = None

    _STRATEGY_MAP = {
        0: ("PASSIVE", 0.0),  # Wait
        1: ("PASSIVE", 0.10),  # Small limit
        2: ("BALANCED", 0.25),  # Medium limit
        3: ("BALANCED", 0.50),  # Large limit
        4: ("AGGRESSIVE", 1.0),  # Market order
    }

    def _run(
        self,
        symbol: str,
        quantity: float,
        urgency: str = "normal",
        arrival_price: float = 0.0,
        time_horizon_bars: int = 10,
    ) -> str:
        cfg = self._get_config()

        if not cfg.enable_execution_rl:
            strategy = {"low": "PASSIVE", "normal": "BALANCED", "high": "AGGRESSIVE"}.get(
                urgency, "BALANCED"
            )
            return json.dumps(
                {
                    "strategy": strategy,
                    "order_fraction": 0.25,
                    "shadow": True,
                    "reasoning": "Execution RL disabled in config.",
                }
            )

        agent = self._load_agent(cfg)
        if agent is None:
            return json.dumps(
                {
                    "strategy": "BALANCED",
                    "order_fraction": 0.25,
                    "shadow": True,
                    "reasoning": "Execution agent not loaded — no checkpoint.",
                }
            )

        # Urgency-adjusted timing approximation
        remaining_time = {
            "low": time_horizon_bars,
            "normal": max(1, time_horizon_bars // 2),
            "high": 1,
        }.get(urgency, max(1, time_horizon_bars // 2))

        features = RLFeatureExtractor.execution_features(
            remaining_qty=quantity,
            total_qty=quantity,
            remaining_time=remaining_time,
            time_horizon=time_horizon_bars,
            current_price=arrival_price if arrival_price > 0 else 100.0,
            arrival_price=arrival_price if arrival_price > 0 else 100.0,
            spread_bps=5.0,
            volatility=0.015,  # conservative default; real value from market data
            volume_ratio=1.0,
            vwap=arrival_price if arrival_price > 0 else 100.0,
            shortfall=0.0,
        )

        from quantcore.rl.base import State

        state = State(features=features)

        try:
            agent.eval()
            action = agent.select_action(state, explore=False)
            action_idx = int(action.value)
        except Exception as exc:
            logger.warning(f"[RLExecutionStrategyTool] Inference failed: {exc}")
            return json.dumps(
                {
                    "strategy": "BALANCED",
                    "order_fraction": 0.25,
                    "shadow": True,
                    "reasoning": f"Inference error: {exc}",
                }
            )

        strategy_label, order_fraction = self._STRATEGY_MAP.get(action_idx, ("BALANCED", 0.25))
        is_shadow = cfg.execution_shadow

        # Save pre-trade snapshot for PostTradeRLAdapter (AC reward needs these)
        save_pretrade_snapshot(
            "rl_execution_strategy",
            {
                "tool_name": "rl_execution_strategy",
                "state_vector": features.tolist(),
                "action_value": action_idx,
                "volatility_at_entry": float(features[4]),  # volatility from state vector
                "daily_volume": 0.0,  # populated by caller if available
                "signal_confidence": 0.7 if not is_shadow else 0.5,
            },
        )

        result = {
            "strategy": strategy_label,
            "order_fraction": round(order_fraction, 2),
            "confidence": 0.7 if not is_shadow else 0.5,
            "shadow": is_shadow,
            "reasoning": (
                f"RL execution agent recommends {strategy_label} for {symbol}. "
                f"{'[SHADOW – not yet validated] ' if is_shadow else ''}"
                f"Execute {order_fraction:.0%} of {quantity:.0f} shares immediately."
            ),
        }
        return json.dumps(result)

    def _get_config(self) -> RLProductionConfig:
        if self._config is None:
            object.__setattr__(self, "_config", get_rl_config())
        return self._config

    def _load_agent(self, cfg: RLProductionConfig) -> Any | None:
        if self._agent is not None:
            return self._agent

        checkpoint = cfg.execution_checkpoint_path
        if not Path(checkpoint).exists():
            logger.debug(f"[RLExecutionStrategyTool] No checkpoint at {checkpoint}.")
            return None

        try:
            from quantcore.rl.execution.agent import ExecutionRLAgent

            agent = ExecutionRLAgent(state_dim=cfg.execution_state_dim, action_dim=5)
            agent.load(str(checkpoint))
            agent.eval()
            object.__setattr__(self, "_agent", agent)
            logger.info("[RLExecutionStrategyTool] Execution agent loaded from checkpoint.")
            return agent
        except Exception as exc:
            logger.warning(f"[RLExecutionStrategyTool] Failed to load checkpoint: {exc}")
            return None


class RLAlphaWeightTool(BaseTool):
    """
    RL-recommended weights for competing IC/alpha signals.

    Calls the trained contextual bandit AlphaSelectionAgent to recommend
    which IC signal to prioritize given current market regime.

    Returns JSON with:
        selected_alpha: str — recommended alpha to follow
        weights: Dict[str, float] — softmax weights for each competing alpha
        confidence (float): agent's confidence
        shadow (bool): True = not yet validated
        reasoning (str)
    """

    name: str = "rl_alpha_weight"
    description: str = (
        "Get RL-recommended weights for competing IC/alpha signals based on regime. "
        "The contextual bandit agent has learned which signal type performs best "
        "in each market regime from historical trade data. "
        "Input: regime (str), competing_signals (list of signal names), "
        "market_volatility (float 0-1, optional), vix_normalized (float 0-1, optional). "
        "Output: JSON with selected_alpha, weights dict, confidence, shadow flag."
    )
    args_schema: type[BaseModel] = RLAlphaWeightInput

    _agent: Any | None = None
    _alpha_names: list[str] | None = None
    _config: RLProductionConfig | None = None

    _REGIME_MAP: dict[str, int] = {
        "trending_up": 0,
        "low_vol_bull": 0,
        "trending_down": 2,
        "low_vol_bear": 2,
        "high_vol": 1,
        "high_vol_bull": 1,
        "high_vol_bear": 3,
        "ranging": 1,
        "volatile": 3,
    }

    def _run(
        self,
        regime: str,
        competing_signals: list[str],
        market_volatility: float = 0.3,
        vix_normalized: float = 0.3,
    ) -> str:
        cfg = self._get_config()

        if not cfg.enable_meta_rl:
            # Fallback: equal weights
            equal_w = 1.0 / len(competing_signals) if competing_signals else 0.0
            return json.dumps(
                {
                    "selected_alpha": competing_signals[0] if competing_signals else "NONE",
                    "weights": {s: round(equal_w, 4) for s in competing_signals},
                    "confidence": 0.5,
                    "shadow": True,
                    "reasoning": "Alpha selection RL disabled in config — using equal weights.",
                }
            )

        agent = self._load_agent(cfg, competing_signals)
        if agent is None:
            equal_w = 1.0 / len(competing_signals) if competing_signals else 0.0
            return json.dumps(
                {
                    "selected_alpha": competing_signals[0] if competing_signals else "NONE",
                    "weights": {s: round(equal_w, 4) for s in competing_signals},
                    "confidence": 0.5,
                    "shadow": True,
                    "reasoning": "Alpha selection agent not loaded — using equal weights.",
                }
            )

        # Map regime string to index
        regime_idx = self._REGIME_MAP.get(regime.lower().replace(" ", "_"), 1)

        # Build alpha return history from in-memory agent tracking
        alpha_histories: dict[str, list[float]] = {name: [] for name in competing_signals}
        alpha_alignments: dict[str, float] = dict.fromkeys(competing_signals, 0.5)

        features = RLFeatureExtractor.alpha_selection_features(
            regime_idx=regime_idx,
            alpha_names=competing_signals,
            alpha_returns_history=alpha_histories,
            alpha_regime_alignments=alpha_alignments,
            market_volatility=market_volatility,
            vix_normalized=vix_normalized,
        )

        from quantcore.rl.base import State

        state = State(features=features)

        try:
            agent.eval()
            action = agent.select_action(state, explore=False)
            action_idx = int(action.value)
        except Exception as exc:
            logger.warning(f"[RLAlphaWeightTool] Inference failed: {exc}")
            equal_w = 1.0 / len(competing_signals) if competing_signals else 0.0
            return json.dumps(
                {
                    "selected_alpha": competing_signals[0] if competing_signals else "NONE",
                    "weights": {s: round(equal_w, 4) for s in competing_signals},
                    "confidence": 0.5,
                    "shadow": True,
                    "reasoning": f"Inference error: {exc}",
                }
            )

        # Build output
        n = len(competing_signals)
        if 0 <= action_idx < n:
            selected = competing_signals[action_idx]
            # Weighted: selected gets 0.5, others share 0.5
            weights = {s: round(0.5 / max(n - 1, 1), 4) for s in competing_signals}
            weights[selected] = 0.5
        else:
            selected = competing_signals[0] if competing_signals else "NONE"
            weights = {s: round(1.0 / n, 4) for s in competing_signals}

        is_shadow = cfg.meta_shadow
        result = {
            "selected_alpha": selected,
            "weights": weights,
            "confidence": 0.65 if not is_shadow else 0.5,
            "shadow": is_shadow,
            "reasoning": (
                f"RL alpha selector recommends {selected} for regime={regime}. "
                f"{'[SHADOW – not yet validated] ' if is_shadow else ''}"
                f"Regime={regime}, volatility={market_volatility:.2f}."
            ),
        }
        return json.dumps(result)

    def _get_config(self) -> RLProductionConfig:
        if self._config is None:
            object.__setattr__(self, "_config", get_rl_config())
        return self._config

    def _load_agent(
        self,
        cfg: RLProductionConfig,
        signal_names: list[str],
    ) -> Any | None:
        if self._agent is not None:
            return self._agent

        checkpoint = cfg.meta_checkpoint_path
        if not Path(checkpoint).exists():
            logger.debug(f"[RLAlphaWeightTool] No checkpoint at {checkpoint}.")
            return None

        try:
            from quantcore.rl.meta.agent import AlphaSelectionAgent

            # Use signal names to compute correct state dim
            n_alphas = len(signal_names)
            state_dim = 4 + 4 * n_alphas + 4
            action_dim = n_alphas + 1  # +1 for no-trade

            agent = AlphaSelectionAgent(state_dim=state_dim, action_dim=action_dim)
            agent.load(str(checkpoint))
            agent.eval()
            object.__setattr__(self, "_agent", agent)
            object.__setattr__(self, "_alpha_names", signal_names)
            logger.info("[RLAlphaWeightTool] Alpha selection agent loaded from checkpoint.")
            return agent
        except Exception as exc:
            logger.warning(f"[RLAlphaWeightTool] Failed to load checkpoint: {exc}")
            return None


# ---------------------------------------------------------------------------
# Factory function (mirrors pattern in crews/tools.py)
# ---------------------------------------------------------------------------


def get_rl_tools(config: RLProductionConfig | None = None) -> list[BaseTool]:
    """
    Get RL agent tools for crew registration.

    Returns only tools whose corresponding enable_* flag is True.
    All tools start in shadow mode until PromotionGate passes.

    Usage:
        from quantcore.rl.rl_tools import get_rl_tools
        tools = get_rl_tools(cfg)  # add to super_trader tool list
    """
    cfg = config or get_rl_config()
    tools = []

    if cfg.enable_sizing_rl:
        tools.append(RLPositionSizeTool())
    if cfg.enable_execution_rl:
        tools.append(RLExecutionStrategyTool())
    if cfg.enable_meta_rl:
        tools.append(RLAlphaWeightTool())

    logger.debug(f"[RL] {len(tools)} RL tools registered (shadow={cfg.shadow_mode_enabled}).")
    return tools


def rl_position_size_tool() -> RLPositionSizeTool:
    """Factory function — consistent with mcp_bridge.py pattern."""
    return RLPositionSizeTool()


def rl_execution_strategy_tool() -> RLExecutionStrategyTool:
    return RLExecutionStrategyTool()


def rl_alpha_weight_tool() -> RLAlphaWeightTool:
    return RLAlphaWeightTool()
