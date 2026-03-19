"""Feature engineering module for multi-timeframe analysis."""

from quantcore.features.base import FeatureBase
from quantcore.features.factory import MultiTimeframeFeatureFactory
from quantcore.features.market_structure import MarketStructureFeatures
from quantcore.features.earnings_signals import AnalystRevisionSignals, EarningsSurpriseSignals
from quantcore.features.fundamental import (
    AssetGrowthAnomaly,
    FCFYield,
    NovyMarxGP,
    OperatingLeverage,
    RevenueAcceleration,
    SloanAccruals,
)
from quantcore.features.microstructure import (
    AmihudIlliquidity,
    CorwinSchultzSpread,
    OvernightGapPersistence,
    RealizedVarianceDecomposition,
    RollImpliedSpread,
    VWAPSessionDeviation,
)
from quantcore.features.momentum import MomentumFeatures, PercentRExhaustion
from quantcore.features.rates import DualMomentum, YieldCurveFeatures
from quantcore.features.rrg import RRGFeatures
from quantcore.features.trend import HullMovingAverage, IchimokuCloud, SupertrendIndicator, TrendFeatures
from quantcore.features.volatility import VolatilityFeatures, WilliamsVIXFix
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
    # Trend
    "TrendFeatures",
    "SupertrendIndicator",
    "IchimokuCloud",
    "HullMovingAverage",
    # Momentum
    "MomentumFeatures",
    "PercentRExhaustion",
    # Volatility
    "VolatilityFeatures",
    "WilliamsVIXFix",
    # Volume / Market Structure
    "VolumeFeatures",
    "MarketStructureFeatures",
    # Earnings / Fundamental (quantamental)
    "EarningsSurpriseSignals",
    "AnalystRevisionSignals",
    "SloanAccruals",
    "NovyMarxGP",
    "AssetGrowthAnomaly",
    "FCFYield",
    "RevenueAcceleration",
    "OperatingLeverage",
    # Microstructure (OHLCV-derived)
    "AmihudIlliquidity",
    "RollImpliedSpread",
    "CorwinSchultzSpread",
    "RealizedVarianceDecomposition",
    "VWAPSessionDeviation",
    "OvernightGapPersistence",
    # Rates / Macro
    "YieldCurveFeatures",
    "DualMomentum",
    # Other
    "RRGFeatures",
    "WaveFeatures",
    "WaveLabeler",
    "SwingDetector",
    "WaveConfig",
    "WaveRole",
    "MultiTimeframeFeatureFactory",
]
