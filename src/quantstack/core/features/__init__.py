"""Feature engineering module for multi-timeframe analysis."""

from quantstack.core.features.base import FeatureBase
from quantstack.core.features.factory import MultiTimeframeFeatureFactory
from quantstack.core.features.market_structure import MarketStructureFeatures
from quantstack.core.features.earnings_signals import (
    AnalystRevisionSignals,
    EarningsImpliedMove,
    EarningsSurpriseSignals,
)
from quantstack.core.features.fundamental import (
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
from quantstack.core.features.insider_signals import InsiderSignals
from quantstack.core.features.institutional_signals import (
    InstitutionalConcentration,
    LSVHerding,
)
from quantstack.core.features.microstructure import (
    AmihudIlliquidity,
    CorwinSchultzSpread,
    OvernightGapPersistence,
    RealizedVarianceDecomposition,
    RollImpliedSpread,
    VWAPSessionDeviation,
)
from quantstack.core.features.momentum import (
    LaguerreRSI,
    MomentumFeatures,
    PercentRExhaustion,
)
from quantstack.core.features.carry import (
    COTSignals,
    CTAPositioningModel,
    EquityCarry,
    FuturesBasis,
)
from quantstack.core.features.rates import DualMomentum, SpreadSignals, YieldCurveFeatures
from quantstack.core.features.sec_nlp import (
    EightKClassifier,
    MDADeltaAnalyzer,
    RiskFactorDeltaAnalyzer,
)
from quantstack.core.features.smart_money import (
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
from quantstack.core.features.alternative_data import (
    BorrowRateSignals,
    DarkPoolSignals,
    EarningsTranscriptNLP,
    ShortInterestSignals,
)
from quantstack.core.features.flow import (
    CumulativeVolumeDelta,
    FootprintApproximation,
    VPIN,
    HawkesIntensity,
)
from quantstack.core.features.statistical import (
    AutocorrelationSpectrum,
    EntropyFeatures,
    HurstExponent,
    OUHalfLife,
    VarianceRatioTest,
    YangZhangVolatility,
)
from quantstack.core.features.momentum_factors import (
    CrossSectionalDispersion,
    InstitutionalMomentumFactors,
)
from quantstack.core.features.calendar_features import FourierCalendarFeatures
from quantstack.core.features.preprocessing import FractionalDifferentiator
from quantstack.core.features.macro_features import (
    CopperGoldRatio,
    CreditSpreadFeatures,
    DXYMomentum,
    EquityBondCorrelation,
    MOVEIndex,
    RealYieldFeatures,
    VolOfVol,
)
from quantstack.core.features.koncorde import Koncorde
from quantstack.core.features.rrg import RRGFeatures
from quantstack.core.features.trend import (
    HullMovingAverage,
    IchimokuCloud,
    SupertrendIndicator,
    TrendFeatures,
)
from quantstack.core.features.volatility import VolatilityFeatures, WilliamsVIXFix
from quantstack.core.features.volume import (
    AnchoredVWAP,
    VolumeFeatures,
    VolumePointOfControl,
)
from quantstack.core.features.waves import (
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
    "CTAPositioningModel",
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
    # Statistical (hedge fund grade)
    "YangZhangVolatility",
    "HurstExponent",
    "VarianceRatioTest",
    "OUHalfLife",
    "AutocorrelationSpectrum",
    "EntropyFeatures",
    # Institutional Momentum
    "InstitutionalMomentumFactors",
    "CrossSectionalDispersion",
    # Calendar
    "FourierCalendarFeatures",
    # ML Preprocessing
    "FractionalDifferentiator",
    # Macro (FRED-sourced)
    "RealYieldFeatures",
    "CreditSpreadFeatures",
    "CopperGoldRatio",
    "DXYMomentum",
    "MOVEIndex",
    "EquityBondCorrelation",
    "VolOfVol",
]
