"""
RL Production Configuration.

Single versioned configuration for the entire RL system.
All numeric thresholds and feature flags live here — no magic numbers scattered
across individual files.

Version this alongside model checkpoints: if config changes, retrain.

Usage:
    from quantcore.rl.config import get_rl_config, RLProductionConfig

    cfg = get_rl_config()
    if cfg.enable_execution_rl:
        ...
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class RLProductionConfig(BaseSettings):
    """
    Production configuration for all RL agents.

    Loaded from environment variables with QUANTRL_ prefix.
    Defaults are conservative — safe to run out of the box.
    """

    # -------------------------------------------------------------------------
    # Versioning (must match checkpoint metadata to use saved models)
    # -------------------------------------------------------------------------
    config_version: str = "1.0.0"

    # -------------------------------------------------------------------------
    # Agent feature flags
    # -------------------------------------------------------------------------
    # Execution RL: lowest risk, most mature. Enable first.
    enable_execution_rl: bool = True
    # Sizing RL: medium risk, needs real signal history.
    enable_sizing_rl: bool = True
    # Alpha selection RL: highest risk, needs real alpha return history.
    # Gated by KnowledgeStoreRLBridge.get_alpha_return_history() having >= 20 rows.
    enable_meta_rl: bool = True
    # Spread RL: deferred — requires reliable spread data not available in AlphaVantage.
    enable_spread_rl: bool = False

    # -------------------------------------------------------------------------
    # Shadow mode (all agents start in shadow — output tagged [SHADOW])
    # -------------------------------------------------------------------------
    # When True, RL tools return recommendations tagged [SHADOW – not yet validated].
    # LLM agents can read them but they do NOT change execution decisions.
    # Set to False per-agent only after PromotionGate passes.
    shadow_mode_enabled: bool = True
    execution_shadow: bool = True
    sizing_shadow: bool = True
    meta_shadow: bool = True

    # -------------------------------------------------------------------------
    # Online learning safety bounds
    # -------------------------------------------------------------------------
    # Max number of online updates per agent per trading day.
    max_updates_per_day: int = 5
    # Minimum replay buffer size before any online update is attempted.
    min_replay_buffer_size: int = 100
    # Catastrophic forgetting guard: parameter gradient norm cap.
    # If the norm of a proposed weight update exceeds this multiple of baseline,
    # the update is skipped and a warning is logged.
    max_param_change_norm: float = 0.10
    # Minimum bars (trading periods) between online updates.
    update_cooldown_bars: int = 20
    # If rolling 20-trade eval reward drops below this fraction of best seen,
    # online updates are paused and a retraining alert is logged.
    degradation_threshold: float = 0.80
    # Enable online learning updates (set False to disable feedback loop entirely).
    enable_online_updates: bool = True

    # -------------------------------------------------------------------------
    # Feature dimensions (must match neural network input layers)
    # Changing these requires retraining all affected agents.
    # -------------------------------------------------------------------------
    execution_state_dim: int = 8
    sizing_state_dim: int = 10
    # Alpha selection: 4 (regime) + 4*n_alphas + 4 (market). Default 7 alphas → 36.
    alpha_selection_state_dim: int = 36

    # -------------------------------------------------------------------------
    # Checkpoint paths
    # -------------------------------------------------------------------------
    checkpoint_base_path: str = "~/.quant_pod/rl_checkpoints"
    shadow_log_path: str = "~/.quant_pod/rl_shadow"

    # -------------------------------------------------------------------------
    # Shadow promotion thresholds (used by PromotionGate)
    # -------------------------------------------------------------------------
    # Minimum number of shadow observations before promotion is allowed.
    min_shadow_observations_execution: int = 63  # ~3 months trading days
    min_shadow_observations_sizing: int = 63
    min_shadow_observations_meta: int = 126  # ~6 months — regime diversity required

    # Simulated Sharpe lower bound (Lo 2002 CI) required for promotion.
    min_promo_sharpe_sizing: float = 0.5
    min_promo_sharpe_meta: float = 0.5

    # Execution: simulated implementation shortfall must be 20% below baseline.
    min_execution_shortfall_improvement: float = 0.20

    # Walk-forward: minimum fraction of positive folds.
    min_wf_positive_folds: float = 0.60
    # Walk-forward: maximum Sharpe degradation (train → OOS).
    max_wf_sharpe_degradation: float = 0.30

    # Monte Carlo: p-value must be below this threshold.
    max_monte_carlo_pvalue: float = 0.05

    # Max simulated drawdown for any agent.
    max_promo_drawdown: float = 0.12

    # Alpha selection cannot be promoted until execution + sizing have been
    # non-shadow for at least this many days.
    meta_requires_live_days: int = 30

    model_config = {"env_prefix": "QUANTRL_", "extra": "ignore"}

    @field_validator("checkpoint_base_path", "shadow_log_path", mode="before")
    @classmethod
    def expand_paths(cls, v: str) -> str:
        return str(Path(v).expanduser())

    @property
    def execution_checkpoint_path(self) -> Path:
        return Path(self.checkpoint_base_path) / "execution_agent.pt"

    @property
    def sizing_checkpoint_path(self) -> Path:
        return Path(self.checkpoint_base_path) / "sizing_agent.pt"

    @property
    def meta_checkpoint_path(self) -> Path:
        return Path(self.checkpoint_base_path) / "meta_agent.pt"


_cached_config: Optional[RLProductionConfig] = None


def get_rl_config() -> RLProductionConfig:
    """Get singleton RL production config (loaded once from env)."""
    global _cached_config
    if _cached_config is None:
        _cached_config = RLProductionConfig()
    return _cached_config


def reset_rl_config() -> None:
    """Reset cached config (for tests)."""
    global _cached_config
    _cached_config = None
