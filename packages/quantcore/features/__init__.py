"""Feature engineering module for multi-timeframe analysis."""

from quantcore.features.base import FeatureBase
from quantcore.features.factory import MultiTimeframeFeatureFactory
from quantcore.features.market_structure import MarketStructureFeatures
from quantcore.features.earnings_signals import AnalystRevisionSignals, EarningsSurpriseSignals
from quantcore.features.fundamental import (
    BeneishMScore,
    PiotroskiFScore,
    AssetGrowthAnomaly,
    FCFYield,
    NovyMarxGP,
    OperatingLeverage,
    RevenueAcceleration,
    SloanAccruals,
)
from quantcore.features.insider_signals import InsiderSignals
from quantcore.features.institutional_signals import InstitutionalConcentration, LSVHerding
from quantcore.features.microstructure import (
    AmihudIlliquidity,
    CorwinSchultzSpread,
    OvernightGapPersistence,
    RealizedVarianceDecomposition,
    RollImpliedSpread,
    VWAPSessionDeviation,
)
from quantcore.features.momentum import LaguerreRSI, MomentumFeatures, PercentRExhaustion
from quantcore.features.rates import DualMomentum, YieldCurveFeatures
from quantcore.features.sec_nlp import EightKClassifier, MDADeltaAnalyzer, RiskFactorDeltaAnalyzer
from quantcore.features.smart_money import (
    EqualHighsLows,
    FairValueGapDetector,
    ICTKillZones,
    ICTPowerOfThree,
    OrderBlockDetector,
    OTELevels,
    StructureAnalysis,
)
from quantcore.features.rrg import RRGFeatures
from quantcore.features.trend import HullMovingAverage, IchimokuCloud, SupertrendIndicator, TrendFeatures
from quantcore.features.volatility import VolatilityFeatures, WilliamsVIXFix
from quantcore.features.volume import AnchoredVWAP, VolumeFeatures, VolumePointOfControl
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
    "LaguerreRSI",
    # Volatility
    "VolatilityFeatures",
    "WilliamsVIXFix",
    # Volume / Market Structure
    "VolumeFeatures",
    "VolumePointOfControl",
    "AnchoredVWAP",
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
    "PiotroskiFScore",
    "BeneishMScore",
    # Insider / Institutional
    "InsiderSignals",
    "LSVHerding",
    "InstitutionalConcentration",
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
    # SEC NLP Signals
    "EightKClassifier",
    "MDADeltaAnalyzer",
    "RiskFactorDeltaAnalyzer",
    # ICT Smart Money Concepts
    "FairValueGapDetector",
    "OrderBlockDetector",
    "StructureAnalysis",
    "EqualHighsLows",
    "OTELevels",
    "ICTKillZones",
    "ICTPowerOfThree",
    # Other
    "RRGFeatures",
    "WaveFeatures",
    "WaveLabeler",
    "SwingDetector",
    "WaveConfig",
    "WaveRole",
    "MultiTimeframeFeatureFactory",
]
