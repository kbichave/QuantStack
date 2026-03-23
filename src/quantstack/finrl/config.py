"""
FinRL configuration — model paths, shadow mode, promotion thresholds.

Replaces quantstack.rl.config with a simpler, FinRL-focused config.
All numeric thresholds live here — no magic numbers scattered across files.

Usage:
    from quantstack.finrl.config import get_finrl_config

    cfg = get_finrl_config()
    if cfg.shadow_mode_enabled:
        ...
"""

from __future__ import annotations

from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings


class FinRLConfig(BaseSettings):
    """
    Configuration for the FinRL-based RL system.

    Loaded from environment variables with FINRL_ prefix.
    """

    # ── Versioning ──
    config_version: str = "2.0.0"

    # ── Shadow mode ──
    # New models start in shadow — predictions tagged [SHADOW].
    # Promotion requires passing PromotionGate statistical tests.
    shadow_mode_enabled: bool = True

    # ── Model storage ──
    checkpoint_base_path: str = "~/.quant_pod/finrl_models"
    shadow_log_path: str = "~/.quant_pod/finrl_shadow"

    # ── Training defaults ──
    default_algorithm: str = "ppo"
    default_total_timesteps: int = 100_000
    default_learning_rate: float = 3e-4
    default_batch_size: int = 64
    default_net_arch: list[int] = [256, 256]

    # ── Ensemble defaults ──
    ensemble_algorithms: list[str] = ["ppo", "a2c", "ddpg"]
    ensemble_validation_months: int = 3
    ensemble_rebalance_months: int = 1

    # ── Promotion thresholds (used by PromotionGate) ──
    min_shadow_observations: int = 63  # ~3 months trading days
    min_promo_sharpe: float = 0.5
    max_promo_drawdown: float = 0.12
    min_wf_positive_folds: float = 0.60
    max_wf_sharpe_degradation: float = 0.30
    max_monte_carlo_pvalue: float = 0.05
    min_direction_agreement: float = 0.55

    # ── Environment defaults ──
    default_initial_capital: float = 100_000
    default_transaction_cost: float = 0.001

    # ── Technical indicators for stock trading envs ──
    default_indicators: list[str] = [
        "macd",
        "boll_ub",
        "boll_lb",
        "rsi_30",
        "cci_30",
        "dx_30",
        "close_30_sma",
        "close_60_sma",
        "vix",
        "turbulence",
    ]

    model_config = {"env_prefix": "FINRL_", "extra": "ignore"}

    @field_validator("checkpoint_base_path", "shadow_log_path", mode="before")
    @classmethod
    def expand_paths(cls, v: str) -> str:
        return str(Path(v).expanduser())


_cached_config: FinRLConfig | None = None


def get_finrl_config() -> FinRLConfig:
    """Get singleton FinRL config (loaded once from env)."""
    global _cached_config
    if _cached_config is None:
        _cached_config = FinRLConfig()
    return _cached_config


def reset_finrl_config() -> None:
    """Reset cached config (for tests)."""
    global _cached_config
    _cached_config = None
