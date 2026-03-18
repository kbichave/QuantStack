# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Portfolio-level risk analytics — correlation, factor exposure, concentration.

Sits above the per-trade RiskGate. RiskGate checks individual position limits;
this module checks portfolio-level coherence: are positions correlated? Is factor
exposure concentrated? Is sector exposure balanced?

Usage:
    from quant_pod.risk.portfolio_risk import PortfolioRiskAnalyzer

    analyzer = PortfolioRiskAnalyzer()
    report = analyzer.analyze(positions, returns_df)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from loguru import logger

# =============================================================================
# SECTOR MAPPING
# =============================================================================

# Hardcoded sector mapping for ~50 major symbols. Same approach as the sector
# collector — avoids an API call for a lookup that changes infrequently.
# Positions with unmapped symbols default to "Unknown".
SYMBOL_TO_SECTOR: dict[str, str] = {
    # Technology
    "AAPL": "Technology",
    "MSFT": "Technology",
    "GOOGL": "Technology",
    "GOOG": "Technology",
    "META": "Technology",
    "NVDA": "Technology",
    "AMD": "Technology",
    "INTC": "Technology",
    "CRM": "Technology",
    "ADBE": "Technology",
    "ORCL": "Technology",
    "CSCO": "Technology",
    "AVGO": "Technology",
    "QCOM": "Technology",
    "TSM": "Technology",
    # Financials
    "JPM": "Financials",
    "BAC": "Financials",
    "GS": "Financials",
    "MS": "Financials",
    "WFC": "Financials",
    "C": "Financials",
    "BLK": "Financials",
    "SCHW": "Financials",
    # Healthcare
    "JNJ": "Healthcare",
    "UNH": "Healthcare",
    "PFE": "Healthcare",
    "ABBV": "Healthcare",
    "MRK": "Healthcare",
    "LLY": "Healthcare",
    "TMO": "Healthcare",
    # Consumer Discretionary
    "AMZN": "Consumer Discretionary",
    "TSLA": "Consumer Discretionary",
    "HD": "Consumer Discretionary",
    "NKE": "Consumer Discretionary",
    "MCD": "Consumer Discretionary",
    "SBUX": "Consumer Discretionary",
    # Consumer Staples
    "PG": "Consumer Staples",
    "KO": "Consumer Staples",
    "PEP": "Consumer Staples",
    "WMT": "Consumer Staples",
    "COST": "Consumer Staples",
    # Energy
    "XOM": "Energy",
    "CVX": "Energy",
    "COP": "Energy",
    "SLB": "Energy",
    # Industrials
    "CAT": "Industrials",
    "BA": "Industrials",
    "UPS": "Industrials",
    "HON": "Industrials",
    "GE": "Industrials",
    # Utilities
    "NEE": "Utilities",
    "DUK": "Utilities",
    "SO": "Utilities",
    # Communication Services
    "DIS": "Communication Services",
    "NFLX": "Communication Services",
    "CMCSA": "Communication Services",
    "T": "Communication Services",
    "VZ": "Communication Services",
    # ETFs — classified as "ETF" (not a true sector, but prevents false concentration alerts)
    "SPY": "ETF",
    "QQQ": "ETF",
    "IWM": "ETF",
    "DIA": "ETF",
    "XLF": "ETF",
    "XLE": "ETF",
    "XLK": "ETF",
    "GLD": "ETF",
    "TLT": "ETF",
    "HYG": "ETF",
}

# Maximum sector weight before flagging a breach
_SECTOR_BREACH_THRESHOLD = 0.30

# Correlation thresholds
_CORR_REDUCE_THRESHOLD = 0.80  # Recommend reducing size at this correlation
_CORR_SKIP_THRESHOLD = 0.90    # Recommend skipping entirely at this level

# Rolling correlation window (trading days)
_CORR_WINDOW = 60


# =============================================================================
# DATACLASSES
# =============================================================================


@dataclass
class CorrelationCheck:
    """Result of checking a proposed position against existing holdings."""

    max_correlation: float
    correlated_with: str | None
    recommendation: str  # "ok", "reduce_50pct", "skip"
    detail: str


@dataclass
class FactorExposure:
    """Simplified factor exposure of the current portfolio."""

    market_beta: float
    size_tilt: str  # "large_cap", "mid_cap", "small_cap", "mixed"
    momentum_tilt: str  # "chasing", "balanced", "contrarian"
    sector_concentration: float  # Herfindahl 0-1
    dominant_sector: str | None


@dataclass
class ConcentrationReport:
    """Position and sector concentration metrics."""

    herfindahl_index: float
    largest_position_pct: float
    top3_pct: float
    sector_weights: dict[str, float]
    sector_breach: bool
    position_count: int


@dataclass
class PortfolioRiskReport:
    """Complete portfolio-level risk assessment."""

    correlation_matrix: pd.DataFrame | None
    factor_exposure: FactorExposure
    concentration: ConcentrationReport
    risk_score: int  # 1-10, higher = more risky
    warnings: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)


# =============================================================================
# ANALYZER
# =============================================================================


class PortfolioRiskAnalyzer:
    """
    Portfolio-level risk analytics.

    Checks that the RiskGate cannot: cross-position correlation, factor
    concentration, and sector balance. The RiskGate enforces per-trade limits;
    this module enforces portfolio coherence.
    """

    def analyze(
        self,
        positions: list[dict],
        returns_df: pd.DataFrame | None = None,
    ) -> PortfolioRiskReport:
        """
        Run full portfolio risk analysis.

        Args:
            positions: List of dicts with keys: symbol, quantity, market_value.
                       Optional keys: sector, beta.
            returns_df: Historical daily returns DataFrame with symbols as columns.
                        Used for correlation analysis. If None, correlation is skipped.

        Returns:
            PortfolioRiskReport with all analytics and an aggregate risk score.
        """
        if not positions:
            empty_factor = FactorExposure(
                market_beta=0.0,
                size_tilt="mixed",
                momentum_tilt="balanced",
                sector_concentration=0.0,
                dominant_sector=None,
            )
            empty_concentration = ConcentrationReport(
                herfindahl_index=0.0,
                largest_position_pct=0.0,
                top3_pct=0.0,
                sector_weights={},
                sector_breach=False,
                position_count=0,
            )
            return PortfolioRiskReport(
                correlation_matrix=None,
                factor_exposure=empty_factor,
                concentration=empty_concentration,
                risk_score=1,
                warnings=[],
                recommendations=["No open positions — portfolio is flat."],
            )

        # Compute each component
        corr_matrix = None
        if returns_df is not None and len(returns_df) >= _CORR_WINDOW:
            try:
                corr_matrix = self.compute_correlation_matrix(returns_df)
            except Exception as exc:
                logger.warning(f"[PORTFOLIO_RISK] Correlation computation failed: {exc}")

        factor_exposure = self.compute_factor_exposure(positions)
        concentration = self.compute_concentration_risk(positions)

        # Aggregate risk score
        warnings: list[str] = []
        recommendations: list[str] = []
        risk_score = self._compute_risk_score(
            factor_exposure, concentration, warnings, recommendations
        )

        report = PortfolioRiskReport(
            correlation_matrix=corr_matrix,
            factor_exposure=factor_exposure,
            concentration=concentration,
            risk_score=risk_score,
            warnings=warnings,
            recommendations=recommendations,
        )

        logger.info(
            f"[PORTFOLIO_RISK] Analysis complete | risk_score={risk_score}/10 "
            f"| positions={concentration.position_count} "
            f"| herfindahl={concentration.herfindahl_index:.3f} "
            f"| beta={factor_exposure.market_beta:.2f} "
            f"| warnings={len(warnings)}"
        )
        return report

    # -------------------------------------------------------------------------
    # Correlation
    # -------------------------------------------------------------------------

    def compute_correlation_matrix(self, returns_df: pd.DataFrame) -> pd.DataFrame:
        """
        Rolling 60-day correlation matrix across all columns in returns_df.

        Args:
            returns_df: DataFrame with symbols as columns, daily returns as rows.
                        Index should be datetime-like.

        Returns:
            NxN correlation DataFrame using the trailing 60 trading days.
        """
        window = returns_df.tail(_CORR_WINDOW)
        if window.shape[0] < 20:
            logger.warning(
                f"[PORTFOLIO_RISK] Only {window.shape[0]} days of data for correlation "
                f"(need {_CORR_WINDOW} ideal, 20 minimum)"
            )
        return window.corr()

    def check_correlation_risk(
        self,
        proposed_symbol: str,
        positions: list[dict],
        returns_df: pd.DataFrame,
    ) -> CorrelationCheck:
        """
        Check if a proposed new position is dangerously correlated with existing holdings.

        Args:
            proposed_symbol: Ticker of the proposed new position.
            positions: Current open positions (list of dicts with 'symbol' key).
            returns_df: Historical returns DataFrame (symbols as columns).

        Returns:
            CorrelationCheck with recommendation: "ok", "reduce_50pct", or "skip".
        """
        held_symbols = [p["symbol"] for p in positions if p["symbol"] != proposed_symbol]

        if not held_symbols:
            return CorrelationCheck(
                max_correlation=0.0,
                correlated_with=None,
                recommendation="ok",
                detail="No existing positions to correlate against.",
            )

        # Require both proposed and at least one held symbol in the returns data
        available = set(returns_df.columns)
        if proposed_symbol not in available:
            return CorrelationCheck(
                max_correlation=0.0,
                correlated_with=None,
                recommendation="ok",
                detail=f"No return data for {proposed_symbol} — correlation check skipped.",
            )

        overlap = [s for s in held_symbols if s in available]
        if not overlap:
            return CorrelationCheck(
                max_correlation=0.0,
                correlated_with=None,
                recommendation="ok",
                detail="No return data for existing positions — correlation check skipped.",
            )

        # Compute pairwise correlations with the proposed symbol
        window = returns_df[[proposed_symbol] + overlap].tail(_CORR_WINDOW)
        corr_series = window.corr()[proposed_symbol].drop(proposed_symbol)

        max_corr = corr_series.abs().max()
        max_corr_symbol = corr_series.abs().idxmax()
        raw_corr = corr_series[max_corr_symbol]

        if max_corr >= _CORR_SKIP_THRESHOLD:
            recommendation = "skip"
            detail = (
                f"{proposed_symbol} has {raw_corr:.2f} correlation with {max_corr_symbol} — "
                f"adding this doubles down on the same risk factor. Skip or hedge."
            )
        elif max_corr >= _CORR_REDUCE_THRESHOLD:
            recommendation = "reduce_50pct"
            detail = (
                f"{proposed_symbol} has {raw_corr:.2f} correlation with {max_corr_symbol} — "
                f"high overlap. Recommend 50% position size."
            )
        else:
            recommendation = "ok"
            detail = (
                f"Max correlation with existing holdings is {raw_corr:.2f} "
                f"({max_corr_symbol}) — within acceptable range."
            )

        return CorrelationCheck(
            max_correlation=float(raw_corr),
            correlated_with=str(max_corr_symbol),
            recommendation=recommendation,
            detail=detail,
        )

    # -------------------------------------------------------------------------
    # Factor exposure
    # -------------------------------------------------------------------------

    def compute_factor_exposure(self, positions: list[dict]) -> FactorExposure:
        """
        Estimate portfolio factor tilts.

        Uses simplified Fama-French-like decomposition:
        - market_beta: weighted-average beta across positions
        - size_tilt: based on market cap classification of held symbols
        - momentum_tilt: fraction of positions near recent highs vs lows
        - sector_concentration: Herfindahl index on sector weights

        Args:
            positions: List of dicts with keys: symbol, quantity, market_value.
                       Optional keys: beta, market_cap, near_20d_high, near_20d_low.
        """
        total_value = sum(abs(p.get("market_value", 0)) for p in positions)
        if total_value == 0:
            return FactorExposure(
                market_beta=1.0,
                size_tilt="mixed",
                momentum_tilt="balanced",
                sector_concentration=0.0,
                dominant_sector=None,
            )

        # -- Market beta: weighted average
        weighted_beta = 0.0
        for p in positions:
            weight = abs(p.get("market_value", 0)) / total_value
            beta = p.get("beta", 1.0)
            weighted_beta += weight * beta
        market_beta = round(weighted_beta, 3)

        # -- Size tilt: count large/mid/small by market cap
        large_weight = 0.0
        mid_weight = 0.0
        small_weight = 0.0
        for p in positions:
            weight = abs(p.get("market_value", 0)) / total_value
            cap = p.get("market_cap", 100e9)  # Default to large cap if unknown
            if cap >= 10e9:
                large_weight += weight
            elif cap >= 2e9:
                mid_weight += weight
            else:
                small_weight += weight

        if large_weight >= 0.7:
            size_tilt = "large_cap"
        elif mid_weight >= 0.5:
            size_tilt = "mid_cap"
        elif small_weight >= 0.5:
            size_tilt = "small_cap"
        else:
            size_tilt = "mixed"

        # -- Momentum tilt: fraction near 20-day high vs low
        near_high_count = sum(1 for p in positions if p.get("near_20d_high", False))
        near_low_count = sum(1 for p in positions if p.get("near_20d_low", False))
        total_positions = len(positions)

        if total_positions > 0 and near_high_count / total_positions >= 0.5:
            momentum_tilt = "chasing"
        elif total_positions > 0 and near_low_count / total_positions >= 0.5:
            momentum_tilt = "contrarian"
        else:
            momentum_tilt = "balanced"

        # -- Sector concentration (Herfindahl on sector weights)
        sector_weights = self._compute_sector_weights(positions, total_value)
        sector_hhi = sum(w ** 2 for w in sector_weights.values()) if sector_weights else 0.0

        dominant_sector = max(sector_weights, key=sector_weights.get) if sector_weights else None

        return FactorExposure(
            market_beta=market_beta,
            size_tilt=size_tilt,
            momentum_tilt=momentum_tilt,
            sector_concentration=round(sector_hhi, 4),
            dominant_sector=dominant_sector,
        )

    # -------------------------------------------------------------------------
    # Concentration
    # -------------------------------------------------------------------------

    def compute_concentration_risk(self, positions: list[dict]) -> ConcentrationReport:
        """
        Compute position and sector concentration metrics.

        Args:
            positions: List of dicts with keys: symbol, quantity, market_value.
                       Optional key: sector.
        """
        total_value = sum(abs(p.get("market_value", 0)) for p in positions)
        if total_value == 0:
            return ConcentrationReport(
                herfindahl_index=0.0,
                largest_position_pct=0.0,
                top3_pct=0.0,
                sector_weights={},
                sector_breach=False,
                position_count=len(positions),
            )

        # Position weights
        weights = [abs(p.get("market_value", 0)) / total_value for p in positions]
        weights_sorted = sorted(weights, reverse=True)

        herfindahl_index = float(np.sum(np.array(weights) ** 2))
        largest_position_pct = weights_sorted[0] * 100.0 if weights_sorted else 0.0
        top3_pct = sum(weights_sorted[:3]) * 100.0

        # Sector weights
        sector_weights = self._compute_sector_weights(positions, total_value)
        sector_breach = any(w > _SECTOR_BREACH_THRESHOLD for w in sector_weights.values())

        # Convert sector_weights to percentage for readability
        sector_weights_pct = {k: round(v * 100.0, 1) for k, v in sector_weights.items()}

        return ConcentrationReport(
            herfindahl_index=round(herfindahl_index, 4),
            largest_position_pct=round(largest_position_pct, 2),
            top3_pct=round(top3_pct, 2),
            sector_weights=sector_weights_pct,
            sector_breach=sector_breach,
            position_count=len(positions),
        )

    # -------------------------------------------------------------------------
    # Risk scoring
    # -------------------------------------------------------------------------

    def _compute_risk_score(
        self,
        factor: FactorExposure,
        concentration: ConcentrationReport,
        warnings: list[str],
        recommendations: list[str],
    ) -> int:
        """
        Aggregate risk score from 1 (safe) to 10 (dangerous).

        Mutates warnings and recommendations lists in place.
        """
        score = 1

        if concentration.herfindahl_index > 0.3:
            score += 2
            warnings.append(
                f"High position concentration (HHI={concentration.herfindahl_index:.3f}). "
                f"Portfolio is sensitive to a single-name move."
            )
            recommendations.append(
                "Diversify: spread capital across more names or add uncorrelated positions."
            )

        if concentration.sector_breach:
            score += 2
            breached = [
                f"{sector} ({pct:.1f}%)"
                for sector, pct in concentration.sector_weights.items()
                if pct > _SECTOR_BREACH_THRESHOLD * 100
            ]
            warnings.append(f"Sector concentration breach: {', '.join(breached)} exceed 30%.")
            recommendations.append(
                "Reduce overweight sector exposure or add positions in underweight sectors."
            )

        if factor.market_beta > 1.5:
            score += 2
            warnings.append(
                f"Portfolio beta is {factor.market_beta:.2f} — highly directional. "
                f"A 1% market move implies ~{factor.market_beta:.1f}% portfolio move."
            )
            recommendations.append(
                "Reduce beta by trimming high-beta names or adding low-beta / hedging positions."
            )

        if factor.momentum_tilt == "chasing":
            score += 1
            warnings.append(
                "Momentum tilt is 'chasing' — majority of positions are near 20-day highs. "
                "Vulnerable to mean-reversion."
            )

        if concentration.largest_position_pct > 12.0:
            score += 1
            warnings.append(
                f"Largest position is {concentration.largest_position_pct:.1f}% of portfolio — "
                f"single-name risk is elevated."
            )
            recommendations.append(
                f"Trim the largest position to below 10% or spread capital across more names."
            )

        if concentration.position_count > 8:
            score += 1
            warnings.append(
                f"Tracking {concentration.position_count} positions — "
                f"may exceed attention capacity for active management."
            )
            recommendations.append(
                "Consider closing the weakest positions to focus on highest-conviction ideas."
            )

        # Cap at 10
        score = min(score, 10)
        return score

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _compute_sector_weights(
        positions: list[dict], total_value: float
    ) -> dict[str, float]:
        """
        Compute sector weights as fractions (0-1) from position list.

        Uses the SYMBOL_TO_SECTOR mapping. Unmapped symbols go to "Unknown".
        Positions may provide their own 'sector' key to override the default mapping.
        """
        if total_value == 0:
            return {}

        sector_totals: dict[str, float] = {}
        for p in positions:
            sector = p.get("sector") or SYMBOL_TO_SECTOR.get(p["symbol"], "Unknown")
            value = abs(p.get("market_value", 0))
            sector_totals[sector] = sector_totals.get(sector, 0.0) + value

        return {sector: val / total_value for sector, val in sector_totals.items()}
