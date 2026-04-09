"""Strategy module for mean-reversion rules, signal generation, and options strategies."""

from quantstack.core.strategy.condor_harvesting import (
    CondorConfig,
    CondorHarvestingStrategy,
)
from quantstack.core.strategy.dispersion import (
    DispersionConfig,
    DispersionStrategy,
)
from quantstack.core.strategy.filters import (
    CombinedFilter,
    RRGFilter,
    SwingFilter,
    WaveFilter,
)
from quantstack.core.strategy.gamma_scalping import (
    GammaScalpConfig,
    GammaScalpingStrategy,
    GammaScalpPnLTracker,
)
from quantstack.core.strategy.options_ensemble import (
    EnsembleSignal,
    EnsembleSignalGenerator,
    EnsembleWeights,
    SignalDirection,
    create_ensemble_from_config,
)
from quantstack.core.strategy.rules import EntrySignal, MeanReversionRules
from quantstack.core.strategy.signals import GeneratedSignal, SignalGenerator
from quantstack.core.strategy.vol_arb_engine import VolArbConfig, VolArbStrategy

__all__ = [
    "MeanReversionRules",
    "EntrySignal",
    "RRGFilter",
    "SwingFilter",
    "WaveFilter",
    "CombinedFilter",
    "SignalGenerator",
    "GeneratedSignal",
    # Ensemble signal combiner
    "EnsembleSignalGenerator",
    "EnsembleWeights",
    "EnsembleSignal",
    "SignalDirection",
    "create_ensemble_from_config",
    # P08: Options market-making strategies
    "VolArbConfig",
    "VolArbStrategy",
    "DispersionConfig",
    "DispersionStrategy",
    "GammaScalpConfig",
    "GammaScalpingStrategy",
    "GammaScalpPnLTracker",
    "CondorConfig",
    "CondorHarvestingStrategy",
]
