"""
Feature engineering for economic indicators.

Forward-fills economic indicators to make them available for each trading day.
IMPORTANT: Only uses data available as-of each date (no lookahead).
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
from loguru import logger

from quantcore.data.economic_storage import EconomicStorage


class EconomicFeatureEngineer:
    """Creates daily features from economic indicators."""

    def __init__(self, storage: Optional[EconomicStorage] = None):
        """Initialize feature engineer.

        Args:
            storage: Economic storage instance. If None, creates new one.
        """
        self.storage = storage or EconomicStorage()

    def create_daily_features(
        self,
        date_range: pd.DatetimeIndex,
        indicators: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Create daily features by forward-filling indicators.

        Args:
            date_range: DatetimeIndex of trading days to create features for
            indicators: List of indicator names. If None, uses all available.

        Returns:
            DataFrame with date index and indicator columns
        """
        if date_range.empty:
            return pd.DataFrame()

        start_date = date_range.min()
        end_date = date_range.max()

        # Get all indicator data
        if indicators:
            # Fetch specific indicators
            indicator_dfs = {}
            for ind in indicators:
                df = self.storage.get_indicator(ind, start_date, end_date)
                if not df.empty:
                    indicator_dfs[ind] = df.set_index("date")["value"]
        else:
            # Fetch all indicators
            wide_df = self.storage.get_all_indicators(start_date, end_date)
            if wide_df.empty:
                logger.warning("No economic indicator data found")
                return pd.DataFrame(index=date_range)

            indicator_dfs = {col: wide_df[col] for col in wide_df.columns}

        if not indicator_dfs:
            return pd.DataFrame(index=date_range)

        # Create DataFrame with trading days as index
        result = pd.DataFrame(index=date_range)

        # Forward-fill each indicator
        for ind_name, ind_series in indicator_dfs.items():
            # Reindex to trading days and forward-fill
            # This ensures we only use data available as-of each date
            result[ind_name] = ind_series.reindex(date_range, method="ffill")

        # Add change features (rate of change)
        result = self._add_change_features(result)

        # Add yield curve features
        result = self._add_yield_curve_features(result)

        return result

    def _add_change_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rate-of-change features for indicators.

        Args:
            df: DataFrame with indicator columns

        Returns:
            DataFrame with additional change columns
        """
        # Monthly changes for monthly indicators
        monthly_indicators = [
            "fed_funds_rate",
            "cpi",
            "retail_sales",
            "durables",
            "unemployment",
            "nonfarm_payroll",
        ]

        for ind in monthly_indicators:
            if ind in df.columns:
                # 1-month change
                df[f"{ind}_mom"] = df[ind].pct_change(periods=21)  # ~1 month
                # 3-month change
                df[f"{ind}_qoq"] = df[ind].pct_change(periods=63)  # ~3 months
                # 12-month change
                df[f"{ind}_yoy"] = df[ind].pct_change(periods=252)  # ~1 year

        # Quarterly changes for quarterly indicators
        quarterly_indicators = ["real_gdp_quarterly", "real_gdp_per_capita"]

        for ind in quarterly_indicators:
            if ind in df.columns:
                # Quarter-over-quarter
                df[f"{ind}_qoq"] = df[ind].pct_change(periods=63)  # ~3 months
                # Year-over-year
                df[f"{ind}_yoy"] = df[ind].pct_change(periods=252)  # ~1 year

        # Treasury yield changes
        treasury_yields = ["treasury_3m", "treasury_2y", "treasury_10y", "treasury_30y"]

        for ind in treasury_yields:
            if ind in df.columns:
                # Daily change in basis points
                df[f"{ind}_change_bp"] = df[ind].diff() * 100
                # 1-week change
                df[f"{ind}_change_1w"] = df[ind].diff(periods=5)
                # 1-month change
                df[f"{ind}_change_1m"] = df[ind].diff(periods=21)

        return df

    def _add_yield_curve_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add yield curve slope features.

        Args:
            df: DataFrame with treasury yield columns

        Returns:
            DataFrame with yield curve features
        """
        # 10y-3m spread (classic recession indicator)
        if "treasury_10y" in df.columns and "treasury_3m" in df.columns:
            df["yield_curve_10y3m"] = df["treasury_10y"] - df["treasury_3m"]

        # 10y-2y spread
        if "treasury_10y" in df.columns and "treasury_2y" in df.columns:
            df["yield_curve_10y2y"] = df["treasury_10y"] - df["treasury_2y"]

        # 30y-10y spread
        if "treasury_30y" in df.columns and "treasury_10y" in df.columns:
            df["yield_curve_30y10y"] = df["treasury_30y"] - df["treasury_10y"]

        # Overall curve steepness (30y-3m)
        if "treasury_30y" in df.columns and "treasury_3m" in df.columns:
            df["yield_curve_steepness"] = df["treasury_30y"] - df["treasury_3m"]

        return df

    def create_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create regime-based features from economic indicators.

        Args:
            df: DataFrame with economic indicator features

        Returns:
            DataFrame with regime classification features
        """
        result = df.copy()

        # Recession indicator (inverted yield curve)
        if "yield_curve_10y3m" in result.columns:
            result["recession_risk"] = (result["yield_curve_10y3m"] < 0).astype(int)

        # Inflation regime
        if "cpi_yoy" in result.columns:
            result["high_inflation"] = (result["cpi_yoy"] > 0.03).astype(int)  # >3% YoY
            result["deflation_risk"] = (result["cpi_yoy"] < 0).astype(int)

        # Unemployment trend
        if "unemployment_mom" in result.columns:
            result["unemployment_rising"] = (result["unemployment_mom"] > 0).astype(int)

        # Fed policy stance
        if "fed_funds_rate" in result.columns:
            # Rate of change in Fed funds
            result["fed_tightening"] = (
                result["fed_funds_rate"].diff(periods=63) > 0.5
            ).astype(
                int
            )  # Hiking >50bp in 3mo
            result["fed_easing"] = (
                result["fed_funds_rate"].diff(periods=63) < -0.5
            ).astype(
                int
            )  # Cutting >50bp in 3mo

        # Economic growth regime
        if "real_gdp_quarterly_yoy" in result.columns:
            result["strong_growth"] = (result["real_gdp_quarterly_yoy"] > 0.03).astype(
                int
            )  # >3% YoY
            result["recession"] = (result["real_gdp_quarterly_yoy"] < 0).astype(
                int
            )  # Negative

        # Retail strength
        if "retail_sales_yoy" in result.columns:
            result["retail_strong"] = (result["retail_sales_yoy"] > 0.05).astype(
                int
            )  # >5% YoY

        return result

    def merge_with_market_data(
        self, market_df: pd.DataFrame, economic_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge economic features with market data.

        Args:
            market_df: DataFrame with market data (OHLCV, features)
            economic_df: DataFrame with economic features

        Returns:
            Merged DataFrame with both market and economic features
        """
        if market_df.empty or economic_df.empty:
            logger.warning("Empty DataFrame passed to merge")
            return market_df

        # Ensure both have date index
        if not isinstance(market_df.index, pd.DatetimeIndex):
            if "date" in market_df.columns:
                market_df = market_df.set_index("date")
            else:
                logger.error("market_df must have date index or date column")
                return market_df

        if not isinstance(economic_df.index, pd.DatetimeIndex):
            if "date" in economic_df.columns:
                economic_df = economic_df.set_index("date")
            else:
                logger.error("economic_df must have date index or date column")
                return market_df

        # Left join to preserve all market data
        result = market_df.join(economic_df, how="left")

        # Forward-fill any missing economic data
        economic_cols = economic_df.columns.tolist()
        result[economic_cols] = result[economic_cols].fillna(method="ffill")

        logger.info(
            "Merged {} market rows with {} economic features",
            len(market_df),
            len(economic_cols),
        )

        return result

    def get_feature_importance_groups(self) -> Dict[str, List[str]]:
        """Get logical groupings of economic features for analysis.

        Returns:
            Dict mapping group name to list of feature names
        """
        return {
            "yield_curve": [
                "treasury_3m",
                "treasury_2y",
                "treasury_10y",
                "treasury_30y",
                "yield_curve_10y3m",
                "yield_curve_10y2y",
                "yield_curve_30y10y",
                "yield_curve_steepness",
            ],
            "inflation": [
                "cpi",
                "cpi_mom",
                "cpi_qoq",
                "cpi_yoy",
                "inflation",
                "high_inflation",
                "deflation_risk",
            ],
            "labor": [
                "unemployment",
                "unemployment_mom",
                "unemployment_yoy",
                "unemployment_rising",
                "nonfarm_payroll",
                "nonfarm_payroll_mom",
                "nonfarm_payroll_yoy",
            ],
            "growth": [
                "real_gdp_quarterly",
                "real_gdp_quarterly_qoq",
                "real_gdp_quarterly_yoy",
                "real_gdp_per_capita",
                "real_gdp_per_capita_yoy",
                "strong_growth",
                "recession",
            ],
            "consumption": [
                "retail_sales",
                "retail_sales_mom",
                "retail_sales_yoy",
                "retail_strong",
                "durables",
                "durables_mom",
                "durables_yoy",
            ],
            "monetary_policy": [
                "fed_funds_rate",
                "fed_funds_rate_mom",
                "fed_funds_rate_yoy",
                "fed_tightening",
                "fed_easing",
            ],
            "recession_signals": [
                "recession_risk",
                "recession",
                "yield_curve_10y3m",
                "unemployment_rising",
            ],
        }


def create_economic_features_for_symbol(
    symbol: str,
    market_df: pd.DataFrame,
    storage: Optional[EconomicStorage] = None,
) -> pd.DataFrame:
    """Convenience function to create economic features for a symbol.

    Args:
        symbol: Stock symbol
        market_df: DataFrame with market data for the symbol
        storage: Economic storage instance

    Returns:
        DataFrame with market data and economic features
    """
    engineer = EconomicFeatureEngineer(storage)

    # Get date range from market data
    if isinstance(market_df.index, pd.DatetimeIndex):
        date_range = market_df.index
    elif "date" in market_df.columns:
        date_range = pd.DatetimeIndex(market_df["date"])
    else:
        raise ValueError("market_df must have date index or date column")

    # Create economic features
    economic_df = engineer.create_daily_features(date_range)

    # Add regime features
    economic_df = engineer.create_regime_features(economic_df)

    # Merge with market data
    result = engineer.merge_with_market_data(market_df, economic_df)

    logger.info("Created economic features for {}", symbol)

    return result
