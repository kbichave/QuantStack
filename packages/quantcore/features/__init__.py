"""Feature engineering module for multi-timeframe analysis."""

from quantcore.features.base import FeatureBase
from quantcore.features.factory import MultiTimeframeFeatureFactory
from quantcore.features.market_structure import MarketStructureFeatures
from quantcore.features.earnings_signals import AnalystRevisionSignals, EarningsImpliedMove, EarningsSurpriseSignals
from quantcore.features.fundamental import (
    BeneishMScore,
    EarningsMomentumComposite,
    PiotroskiFScore,
    QualityMomentumComposite,
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
from quantcore.features.carry import COTSignals, EquityCarry, FuturesBasis
from quantcore.features.rates import DualMomentum, SpreadSignals, YieldCurveFeatures
from quantcore.features.sec_nlp import EightKClassifier, MDADeltaAnalyzer, RiskFactorDeltaAnalyzer
from quantcore.features.smart_money import (
    BreakerBlockDetector,
    EqualHighsLows,
    FairValueGapDetector,
    ICTKillZones,
    ICTPowerOfThree,
    MMXMCycle,
    OrderBlockDetector,
    OTELevels,
    SilverBullet,
    SMTDivergence,
    StructureAnalysis,
)
from quantcore.features.alternative_data import (
    BorrowRateSignals,
    DarkPoolSignals,
    EarningsTranscriptNLP,
    ShortInterestSignals,
)
from quantcore.features.flow import CumulativeVolumeDelta, FootprintApproximation, VPIN, HawkesIntensity
from quantcore.features.koncorde import Koncorde
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
    "EarningsImpliedMove",
    "SloanAccruals",
    "NovyMarxGP",
    "AssetGrowthAnomaly",
    "FCFYield",
    "RevenueAcceleration",
    "OperatingLeverage",
    "PiotroskiFScore",
    "BeneishMScore",
    "QualityMomentumComposite",
    "EarningsMomentumComposite",
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
    # Rates / Macro / Carry
    "YieldCurveFeatures",
    "DualMomentum",
    "SpreadSignals",
    "EquityCarry",
    "FuturesBasis",
    "COTSignals",
    # SEC NLP Signals
    "EightKClassifier",
    "MDADeltaAnalyzer",
    "RiskFactorDeltaAnalyzer",
    # ICT Smart Money Concepts
    "FairValueGapDetector",
    "OrderBlockDetector",
    "BreakerBlockDetector",
    "StructureAnalysis",
    "EqualHighsLows",
    "OTELevels",
    "ICTKillZones",
    "ICTPowerOfThree",
    "SilverBullet",
    "MMXMCycle",
    "SMTDivergence",
    # Order Flow Approximations
    "CumulativeVolumeDelta",
    "FootprintApproximation",
    "VPIN",
    "HawkesIntensity",
    # Koncorde composite
    "Koncorde",
    # Alternative Data Framework Stubs
    "DarkPoolSignals",
    "BorrowRateSignals",
    "ShortInterestSignals",
    "EarningsTranscriptNLP",
    # Other
    "RRGFeatures",
    "WaveFeatures",
    "WaveLabeler",
    "SwingDetector",
    "WaveConfig",
    "WaveRole",
    "MultiTimeframeFeatureFactory",
]
