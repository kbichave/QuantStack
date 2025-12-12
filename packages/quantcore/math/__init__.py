"""
Mathematical Models for Quantitative Finance.

Provides implementations of core stochastic processes and filtering techniques:
- Brownian motion and geometric Brownian motion
- Ito processes and numerical integration
- Stochastic volatility models (Heston)
- Kalman filter for state-space models
- Particle filter for nonlinear filtering
- Optimization utilities
"""

from quantcore.math.brownian_motion import (
    brownian_motion,
    geometric_brownian_motion,
    simulate_gbm_paths,
)
from quantcore.math.ito_processes import (
    euler_maruyama,
    milstein,
    ornstein_uhlenbeck,
)
from quantcore.math.stochastic_vol import (
    HestonModel,
    simulate_heston,
)
from quantcore.math.kalman_filter import (
    KalmanFilter,
    LocalLevelModel,
)
from quantcore.math.particle_filter import (
    ParticleFilter,
    systematic_resample,
)
from quantcore.math.optimizer_utils import (
    portfolio_optimize,
    minimum_variance_portfolio,
    maximum_sharpe_portfolio,
)

__all__ = [
    "brownian_motion",
    "geometric_brownian_motion",
    "simulate_gbm_paths",
    "euler_maruyama",
    "milstein",
    "ornstein_uhlenbeck",
    "HestonModel",
    "simulate_heston",
    "KalmanFilter",
    "LocalLevelModel",
    "ParticleFilter",
    "systematic_resample",
    "portfolio_optimize",
    "minimum_variance_portfolio",
    "maximum_sharpe_portfolio",
]
