"""Feature engineering module for multi-timeframe analysis."""

from quantcore.features.base import FeatureBase
from quantcore.features.factory import MultiTimeframeFeatureFactory
from quantcore.features.market_structure import MarketStructureFeatures
from quantcore.features.momentum import MomentumFeatures
from quantcore.features.rrg import RRGFeatures
from quantcore.features.trend import TrendFeatures
from quantcore.features.volatility import VolatilityFeatures
from quantcore.features.volume import VolumeFeatures
from quantcore.features.waves import (
    SwingDetector,
    WaveConfig,
    WaveFeatures,
    WaveLabeler,
    WaveRole,
)

__all__ = [
    "FeatureBase",
    "TrendFeatures",
    "MomentumFeatures",
    "VolatilityFeatures",
    "VolumeFeatures",
    "MarketStructureFeatures",
    "RRGFeatures",
    "WaveFeatures",
    "WaveLabeler",
    "SwingDetector",
    "WaveConfig",
    "WaveRole",
    "MultiTimeframeFeatureFactory",
]
