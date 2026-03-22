# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
SignalBrief — DailyBrief-compatible output from SignalEngine.

SignalBrief is a strict superset of DailyBrief: every field present in
DailyBrief exists here with an identical name, type, and semantic.

Backward-compat invariant (enforced by tests/test_signal_brief_schema.py):
    DailyBrief.model_validate(signal_brief.model_dump())  # must not raise
"""

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

# Import shared schema types — do NOT redefine them here.
from quantstack.crews.schemas import DailyBrief, KeyLevel, SymbolBrief  # noqa: F401


class SignalBrief(BaseModel):
    """
    Output of SignalEngine.run(). Drop-in replacement for DailyBrief.

    The DailyBrief fields are reproduced verbatim so that any code
    consuming a DailyBrief dict can consume a SignalBrief dict without
    change.  Extra fields (engine_version, collection_duration_ms,
    collector_failures) are additive and ignored by DailyBrief consumers.
    """

    # ------------------------------------------------------------------ #
    # DailyBrief fields — identical names, types, defaults                #
    # ------------------------------------------------------------------ #

    date: date

    market_overview: str
    market_bias: Literal["bullish", "bearish", "neutral"]
    market_conviction: float = Field(ge=0, le=1, default=0.5)

    risk_environment: Literal["low", "normal", "elevated", "high"]

    symbol_briefs: list[SymbolBrief] = Field(default_factory=list)
    top_opportunities: list[str] = Field(default_factory=list)
    key_risks: list[str] = Field(default_factory=list)
    strategic_notes: str = ""

    pods_reporting: int = 0
    total_analyses: int = 0
    overall_confidence: float = Field(ge=0, le=1, default=0.5)

    # ------------------------------------------------------------------ #
    # SignalEngine-specific additions (not in DailyBrief)                 #
    # ------------------------------------------------------------------ #

    engine_version: str = "signal_engine_v1"
    collection_duration_ms: float = 0.0

    # Names of collectors that timed out or raised an exception.
    # Consumers should lower conviction when this is non-empty.
    collector_failures: list[str] = Field(default_factory=list)

    # Raw RegimeContext dict for advanced consumers (AutonomousRunner, /reflect).
    regime_detail: dict | None = None

    # Sentiment signal from SentimentCollector (news-scored via Groq).
    # Defaults to neutral (0.5) when no headlines or Groq is unavailable.
    sentiment_score: float = Field(ge=0.0, le=1.0, default=0.5)
    dominant_sentiment: str = "neutral"  # "positive" | "negative" | "neutral"

    # ── Phase 3 additions (v1.0) ─────────────────────────────────────
    # All optional — missing collectors return {} and these stay at defaults.

    # Macro: yield curve + rate momentum (macro collector)
    macro_rate_regime: str = "unknown"  # "rising" | "falling" | "stable" | "unknown"
    yield_curve_slope: float | None = None  # 10Y - 2Y spread

    # Sector: relative strength + rotation (sector collector)
    sector_signal: str = "unknown"  # "leading" | "lagging" | "inline" | "unknown"
    rotation_signal: str = (
        "unknown"  # "growth_to_value" | "defensive_shift" | "broad_rally" | etc.
    )
    breadth_positive_sectors: int | None = None

    # Flow: institutional + insider (flow collector)
    flow_signal: float | None = None  # -1 (distributing) to +1 (accumulating)
    insider_direction: str = "unknown"  # "buying" | "selling" | "neutral" | "unknown"

    # Cross-asset: risk-on/off (cross_asset collector)
    cross_asset_regime: str = "unknown"  # "risk_on" | "risk_off" | "mixed" | "unknown"
    risk_on_score: float | None = None  # 0-1

    # Quality: earnings quality (quality collector)
    quality_score: float | None = None  # 0-1 composite

    # ML: trained model predictions (ml_signal collector)
    ml_prediction: float | None = None  # 0-1 probability
    ml_direction: str = "unknown"  # "bullish" | "bearish" | "neutral" | "unknown"

    # StatArb: pairs signals (statarb collector)
    statarb_signal: str = (
        "unknown"  # "long_spread" | "short_spread" | "neutral" | "unknown"
    )
    spread_zscore: float | None = None

    # Options flow: dealer positioning signals (options_flow collector)
    opt_gex: float | None = None  # Net Gamma Exposure (positive = mean-reverting)
    opt_gamma_flip: float | None = None  # Strike where GEX crosses zero (key S/R)
    opt_above_gamma_flip: int | None = None  # 1 if spot > gamma flip, 0 if below
    opt_dex: float | None = None  # Net Delta Exposure (directional bias)
    opt_max_pain: float | None = None  # Max pain strike
    opt_iv_skew: float | None = None  # OTM put IV - OTM call IV
    opt_iv_skew_zscore: float | None = None  # Skew z-score vs history
    opt_vrp: float | None = None  # Vol Risk Premium (IV - RV)
    opt_charm: float | None = None  # Aggregate delta decay
    opt_vanna: float | None = None  # Aggregate dDelta/dVol
    opt_ehd: float | None = None  # Expected Hedging Demand

    def to_daily_brief(self) -> DailyBrief:
        """Return a DailyBrief-compatible view of this brief (drops extra fields)."""
        return DailyBrief.model_validate(self.model_dump())
