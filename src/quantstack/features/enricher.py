# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Unified feature enricher — shared pipeline for backtest, live, and ML.

Adds fundamental, earnings, macro, and flow features to an OHLCV DataFrame.
Each feature tier is loaded on-demand: a purely technical strategy pays zero
overhead. Rules referencing ``fund_pe_ratio`` trigger only the fundamentals
tier, not macro or flow.

Lookahead protection: all data is forward-filled from its public availability
date (report_period for fundamentals, release date for macro, filing date for
13F), never from the fiscal period end.

Usage:
    enricher = FeatureEnricher()
    tiers = enricher.detect_needed_tiers(strategy_entry_rules)
    df_enriched = enricher.enrich(ohlcv_df, symbol="AAPL", tiers=tiers)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from loguru import logger

from quantstack.core.features.fundamental_features import EarningsFeatures
from quantstack.core.features.economic_features import EconomicFeatureEngineer
from quantstack.data.storage import DataStore
from quantstack.features.flow_features import (
    compute_insider_flow,
    compute_institutional_flow,
)

# ---------------------------------------------------------------------------
# Feature tier configuration
# ---------------------------------------------------------------------------

# Prefix → tier mapping. Used by detect_needed_tiers() to scan rule indicators.
_PREFIX_TO_TIER: dict[str, str] = {
    "fund_": "fundamentals",
    "earn_": "earnings",
    "treasury_": "macro",
    "yield_curve_": "macro",
    "fed_": "macro",
    "recession_": "macro",
    "high_inflation": "macro",
    "strong_growth": "macro",
    "unemployment_": "macro",
    "flow_": "flow",
}

# Indicators that are exact matches (no prefix) → tier
_EXACT_TO_TIER: dict[str, str] = {
    "recession_risk": "macro",
    "high_inflation": "macro",
    "fed_tightening": "macro",
    "fed_easing": "macro",
    "strong_growth": "macro",
}


@dataclass
class FeatureTiers:
    """Which feature tiers to compute. Only loads what rules reference."""

    fundamentals: bool = False
    earnings: bool = False
    macro: bool = False
    flow: bool = False

    def any_active(self) -> bool:
        return self.fundamentals or self.earnings or self.macro or self.flow


# ---------------------------------------------------------------------------
# FeatureEnricher
# ---------------------------------------------------------------------------


class FeatureEnricher:
    """
    Enrich an OHLCV DataFrame with fundamental, macro, earnings, and flow
    features. Loads ONLY the tiers requested.

    Each tier fails gracefully — if data is unavailable (no API key, no cached
    data, DataStore not initialized), the original DataFrame is returned with
    NaN columns. Rules evaluated against NaN columns return False (safe default
    in ``_evaluate_rule()``).
    """

    def enrich(
        self,
        df: pd.DataFrame,
        symbol: str,
        tiers: FeatureTiers | None = None,
    ) -> pd.DataFrame:
        """
        Add requested feature columns to *df*.

        Args:
            df: OHLCV DataFrame with DatetimeIndex.
            symbol: Ticker symbol (required for fundamentals/earnings/flow).
            tiers: Which feature tiers to load. If None, loads all available.

        Returns:
            DataFrame with additional feature columns. Original columns untouched.
        """
        if tiers is None:
            tiers = FeatureTiers(
                fundamentals=True,
                earnings=True,
                macro=True,
                flow=True,
            )

        if not tiers.any_active() or df.empty:
            return df

        result = df

        if tiers.fundamentals:
            result = self._merge_fundamentals(result, symbol)

        if tiers.earnings:
            result = self._merge_earnings(result, symbol)

        if tiers.macro:
            result = self._merge_macro(result)

        if tiers.flow:
            result = self._merge_flow(result, symbol)

        return result

    def detect_needed_tiers(self, rules: list[dict[str, Any]]) -> FeatureTiers:
        """
        Scan strategy rules to determine which feature tiers they reference.

        A rule ``{"indicator": "fund_pe_ratio", "condition": "below", "value": 20}``
        triggers the ``fundamentals`` tier.

        Args:
            rules: Combined entry_rules + exit_rules list.

        Returns:
            FeatureTiers with True for each tier that has at least one rule.
        """
        tiers = FeatureTiers()
        for rule in rules:
            indicator = str(rule.get("indicator", ""))
            if not indicator:
                continue

            # Check exact matches first
            if indicator in _EXACT_TO_TIER:
                tier_name = _EXACT_TO_TIER[indicator]
                setattr(tiers, tier_name, True)
                continue

            # Check prefix matches
            for prefix, tier_name in _PREFIX_TO_TIER.items():
                if indicator.startswith(prefix):
                    setattr(tiers, tier_name, True)
                    break

        return tiers

    # ------------------------------------------------------------------
    # Tier loaders — each returns df unchanged on failure
    # ------------------------------------------------------------------

    def _merge_fundamentals(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Load fundamental metrics from PostgreSQL cache and forward-fill to daily."""
        try:
            store = DataStore()
            metrics = store.load_financial_metrics(symbol)

            if metrics is None or metrics.empty:
                logger.debug(f"[enricher] No fundamental metrics cached for {symbol}")
                return df

            # Select columns we need, indexed by date
            metric_cols = [
                "pe_ratio",
                "pb_ratio",
                "ps_ratio",
                "ev_to_ebitda",
                "roe",
                "roa",
                "gross_margin",
                "operating_margin",
                "net_margin",
                "debt_to_equity",
                "current_ratio",
                "dividend_yield",
                "revenue_growth",
                "earnings_growth",
            ]

            # Ensure date column exists and is datetime
            if "date" in metrics.columns:
                metrics = metrics.copy()
                metrics["date"] = pd.to_datetime(metrics["date"])
                metrics = metrics.set_index("date").sort_index()
            else:
                logger.debug(f"[enricher] No 'date' column in metrics for {symbol}")
                return df

            # Keep only columns that exist
            available = [c for c in metric_cols if c in metrics.columns]
            if not available:
                return df

            fund_df = metrics[available]

            # Rename with fund_ prefix
            fund_df = fund_df.rename(columns={c: f"fund_{c}" for c in available})

            # Forward-fill to daily bars (aligns from report date, no lookahead)
            fund_daily = fund_df.reindex(df.index, method="ffill")

            # Merge into result
            result = df.copy()
            for col in fund_daily.columns:
                result[col] = fund_daily[col].values

            logger.debug(
                f"[enricher] Merged {len(available)} fundamental features for {symbol}"
            )
            return result

        except Exception as exc:
            logger.debug(f"[enricher] Fundamentals failed for {symbol}: {exc}")
            return df

    def _merge_earnings(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Load earnings features using CombinedFundamentalFeatures."""
        try:
            earner = EarningsFeatures()
            result = earner.compute(df, symbol=symbol)
            logger.debug(f"[enricher] Merged earnings features for {symbol}")
            return result

        except Exception as exc:
            logger.debug(f"[enricher] Earnings features failed for {symbol}: {exc}")
            return df

    def _merge_macro(self, df: pd.DataFrame) -> pd.DataFrame:
        """Load economic features from EconomicFeatureEngineer."""
        try:
            engineer = EconomicFeatureEngineer()
            macro_df = engineer.create_daily_features(df.index)

            if macro_df.empty:
                logger.debug("[enricher] No macro data available")
                return df

            result = df.copy()
            for col in macro_df.columns:
                if col not in result.columns:
                    result[col] = macro_df[col].values

            logger.debug(f"[enricher] Merged {len(macro_df.columns)} macro features")
            return result

        except Exception as exc:
            logger.debug(f"[enricher] Macro features failed: {exc}")
            return df

    def _merge_flow(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Load insider and institutional flow features from PostgreSQL cache."""
        try:
            store = DataStore()
            result = df

            # Insider flow
            insider = store.load_insider_trades(symbol)
            if insider is not None and not insider.empty:
                result = compute_insider_flow(result, insider)

            # Institutional flow
            institutional = store.load_institutional_ownership(symbol)
            if institutional is not None and not institutional.empty:
                result = compute_institutional_flow(result, institutional)

            logger.debug(f"[enricher] Merged flow features for {symbol}")
            return result

        except Exception as exc:
            logger.debug(f"[enricher] Flow features failed for {symbol}: {exc}")
            return df
