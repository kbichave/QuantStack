# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Fundamentals collector — reads from the local QuantCore fundamentals store.

No network calls in the live trading path.  Returns {} (not a failure) when
data isn't loaded yet.  Fundamentals are refreshed separately (nightly) via
data fetchers.

Quantamental signals computed here (all degrade gracefully when data absent):
  NovyMarxGP, AssetGrowthAnomaly, RevenueAcceleration, OperatingLeverage —
      from quarterly income / balance sheet statements
  AnalystRevisionSignals — from stored analyst estimates
  InsiderSignals — cluster buy, adjusted buy/sell ratio — from Form 4 data
  LSVHerding — institutional herding measure — from 13F ownership data
"""

import asyncio
from typing import Any

import pandas as pd
from loguru import logger

from quantcore.data.storage import DataStore


async def collect_fundamentals(symbol: str, store: DataStore) -> dict[str, Any]:
    """
    Load stored fundamental metrics and compute quantamental signals for *symbol*.

    Returns a dict with keys (subset of what's available):
        pe_ratio                : float | None
        eps_ttm                 : float | None
        revenue_growth          : float | None
        gross_margin            : float | None
        debt_to_equity          : float | None
        beta                    : float | None
        market_cap              : float | None
        fundamentals_age_days   : int | None
        --- quantamental signals (absent when data not yet loaded) ---
        novy_marx_gp            : float | None — gross profit / total assets
        asset_growth            : float | None — QoQ asset growth rate
        revenue_acceleration    : float | None — QoQ revenue growth delta
        operating_leverage      : float | None — DOL estimate
        analyst_revision_momentum : float | None
        analyst_estimate_dispersion : float | None
        insider_cluster_buy     : int | None — 1 when 3+ insiders buy within 30d
        insider_adj_ratio       : float | None — open-market buy / (buy+sell)
        inst_herding_measure    : float | None — LSV herding H
        inst_herding_buy_bias   : int | None — 1 buying, -1 selling, 0 neutral
        inst_herding_high       : int | None — 1 when H > 0.05
    """
    try:
        return await asyncio.to_thread(_collect_fundamentals_sync, symbol, store)
    except Exception as exc:
        logger.debug(f"[fundamentals] {symbol}: {exc} — skipping (non-critical)")
        return {}


# ---------------------------------------------------------------------------
# Sync implementation
# ---------------------------------------------------------------------------

def _collect_fundamentals_sync(symbol: str, store: DataStore) -> dict[str, Any]:
    """Load fundamentals and quantamental signals from local store."""
    # --- Baseline pre-computed metrics ---
    try:
        row = store.get_fundamentals(symbol) if hasattr(store, "get_fundamentals") else None
    except Exception as exc:
        logger.debug(f"[fundamentals] {symbol}: store.get_fundamentals raised: {exc}")
        row = None

    if row is None:
        # Still attempt quantamental signals even without the pre-computed row
        row = {}

    result: dict[str, Any] = {
        "pe_ratio":             _sf(row.get("pe_ratio")),
        "eps_ttm":              _sf(row.get("eps_ttm")),
        "revenue_growth":       _sf(row.get("revenue_growth")),
        "gross_margin":         _sf(row.get("gross_margin")),
        "debt_to_equity":       _sf(row.get("debt_to_equity")),
        "beta":                 _sf(row.get("beta")),
        "market_cap":           _sf(row.get("market_cap")),
        "fundamentals_age_days": row.get("age_days"),
    }

    # Remove None values for the baseline so a missing fundamentals row
    # doesn't pollute the output with nulls — callers expect {} or values.
    if not any(v is not None for v in result.values()):
        result = {}

    # --- Quantamental signals from financial statements ---
    _add_statement_signals(symbol, store, result)

    # --- Analyst revision signals ---
    _add_analyst_signals(symbol, store, result)

    # --- Insider signals ---
    _add_insider_signals(symbol, store, result)

    # --- Institutional herding signals ---
    _add_herding_signals(symbol, store, result)

    return result


# ---------------------------------------------------------------------------
# Quantamental helpers
# ---------------------------------------------------------------------------

def _add_statement_signals(symbol: str, store: DataStore, result: dict) -> None:
    """Add NovyMarxGP, AssetGrowthAnomaly, RevenueAcceleration, OperatingLeverage."""
    try:
        if not hasattr(store, "load_financial_statements"):
            return

        df = store.load_financial_statements(
            symbol, statement_type="income", period_type="quarterly", limit=12
        )
        if df is None or df.empty or len(df) < 2:
            return

        # Rename to match feature class convention
        if "report_period" in df.columns:
            df = df.rename(columns={"report_period": "period_end"})

        df = df.sort_values("period_end").reset_index(drop=True)

        # NovyMarxGP: gross_profit / total_assets
        if "gross_profit" in df.columns and "total_assets" in df.columns:
            from quantcore.features.fundamental import NovyMarxGP
            gp_df = NovyMarxGP().compute(df)
            result["novy_marx_gp"] = _sf(gp_df["gp_ratio"].iloc[-1])

        # AssetGrowthAnomaly: total_assets growth rate
        if "total_assets" in df.columns:
            from quantcore.features.fundamental import AssetGrowthAnomaly
            ag_df = AssetGrowthAnomaly().compute(df)
            result["asset_growth"] = _sf(ag_df["asset_growth"].iloc[-1])

        # RevenueAcceleration: QoQ revenue growth delta
        if "revenue" in df.columns and len(df) >= 4:
            from quantcore.features.fundamental import RevenueAcceleration
            ra_df = RevenueAcceleration().compute(df)
            result["revenue_acceleration"] = _sf(ra_df["revenue_acceleration"].iloc[-1])

        # OperatingLeverage: DOL = %Δ operating_income / %Δ revenue
        if "operating_income" in df.columns and "revenue" in df.columns and len(df) >= 4:
            from quantcore.features.fundamental import OperatingLeverage
            ol_df = OperatingLeverage().compute(df)
            result["operating_leverage"] = _sf(ol_df["dol"].iloc[-1])

    except Exception as exc:
        logger.debug(f"[fundamentals] {symbol}: statement signals failed: {exc}")


def _add_analyst_signals(symbol: str, store: DataStore, result: dict) -> None:
    """Add analyst revision momentum and estimate dispersion."""
    try:
        if not hasattr(store, "load_analyst_estimates"):
            return

        est_df = store.load_analyst_estimates(symbol)
        if est_df is None or est_df.empty:
            return

        # The store schema: ticker, fiscal_date, period_type, metric, consensus,
        # high, low, num_analysts.  Filter for EPS estimates only.
        eps_col = None
        for col in ("metric",):
            if col in est_df.columns:
                est_df = est_df[est_df[col].str.upper().str.contains("EPS", na=False)]
                break

        if est_df.empty or len(est_df) < 4:
            return

        # Map to expected schema: estimate_date, eps_estimate, analyst_count
        rename_map = {}
        for cand in ("fiscal_date", "date", "period"):
            if cand in est_df.columns:
                rename_map[cand] = "estimate_date"
                break
        for cand in ("consensus", "eps_estimate", "estimate"):
            if cand in est_df.columns:
                rename_map[cand] = "eps_estimate"
                break
        for cand in ("num_analysts", "analyst_count", "count"):
            if cand in est_df.columns:
                rename_map[cand] = "analyst_count"
                break

        if "estimate_date" not in rename_map.values() or "eps_estimate" not in rename_map.values():
            return

        mapped = est_df.rename(columns=rename_map)
        if "analyst_count" not in mapped.columns:
            mapped["analyst_count"] = 1  # fallback

        from quantcore.features.earnings_signals import AnalystRevisionSignals
        rev_df = AnalystRevisionSignals().compute(mapped)

        rev_mom = rev_df["revision_momentum"].dropna()
        if not rev_mom.empty:
            result["analyst_revision_momentum"] = _sf(rev_mom.iloc[-1])

        dispersion = rev_df["estimate_dispersion"].dropna()
        if not dispersion.empty:
            result["analyst_estimate_dispersion"] = _sf(dispersion.iloc[-1])

    except Exception as exc:
        logger.debug(f"[fundamentals] {symbol}: analyst signals failed: {exc}")


def _add_insider_signals(symbol: str, store: DataStore, result: dict) -> None:
    """Add insider cluster buy and adjusted buy/sell ratio."""
    try:
        if not hasattr(store, "load_insider_trades"):
            return

        insider_df = store.load_insider_trades(symbol, limit=100)
        if insider_df is None or insider_df.empty or len(insider_df) < 2:
            return

        # Map store schema → InsiderSignals expected schema
        # Store: transaction_date, owner_name, owner_title, transaction_type,
        #        shares, price_per_share, total_value, shares_owned_after
        rename_map: dict[str, str] = {}
        for cand in ("owner_name", "insider_name", "name"):
            if cand in insider_df.columns:
                rename_map[cand] = "insider_name"
                break
        for cand in ("owner_title", "insider_role", "title"):
            if cand in insider_df.columns:
                rename_map[cand] = "insider_role"
                break
        for cand in ("price_per_share", "price"):
            if cand in insider_df.columns:
                rename_map[cand] = "price"
                break

        mapped = insider_df.rename(columns=rename_map)

        # Fill required columns with safe defaults if missing
        if "insider_name" not in mapped.columns:
            mapped["insider_name"] = "unknown"
        if "insider_role" not in mapped.columns:
            mapped["insider_role"] = "unknown"
        if "price" not in mapped.columns:
            mapped["price"] = 1.0
        if "is_plan_trade" not in mapped.columns:
            mapped["is_plan_trade"] = False

        from quantcore.features.insider_signals import InsiderSignals
        ins_df = InsiderSignals().compute(mapped)

        result["insider_cluster_buy"] = int(ins_df["cluster_buy"].iloc[-1])
        result["insider_adj_ratio"] = _sf(ins_df["adj_buy_sell_ratio"].iloc[-1])

    except Exception as exc:
        logger.debug(f"[fundamentals] {symbol}: insider signals failed: {exc}")


def _add_herding_signals(symbol: str, store: DataStore, result: dict) -> None:
    """Add LSV institutional herding measure."""
    try:
        if not hasattr(store, "load_institutional_ownership"):
            return

        raw_df = store.load_institutional_ownership(symbol, limit=500)
        if raw_df is None or raw_df.empty:
            return

        # Aggregate per-investor rows into per-period holder counts
        # Store schema: ticker, investor_name, report_date, shares_held,
        #               market_value, portfolio_pct, change_shares, change_pct
        date_col = None
        for cand in ("report_date", "period_end", "date"):
            if cand in raw_df.columns:
                date_col = cand
                break
        if date_col is None:
            return

        raw_df[date_col] = pd.to_datetime(raw_df[date_col])
        change_col = "change_shares" if "change_shares" in raw_df.columns else None

        agg: dict[str, Any] = {"total_holders": ("investor_name", "nunique")} if "investor_name" in raw_df.columns else {"total_holders": (raw_df.columns[0], "count")}

        period_df = raw_df.groupby(date_col).agg(
            total_holders=pd.NamedAgg(column=raw_df.columns[1] if "investor_name" not in raw_df.columns else "investor_name", aggfunc="nunique"),
        ).reset_index()

        if change_col:
            inc = raw_df.groupby(date_col).apply(lambda g: (g[change_col] > 0).sum()).reset_index(name="holders_increased")
            dec = raw_df.groupby(date_col).apply(lambda g: (g[change_col] < 0).sum()).reset_index(name="holders_decreased")
            period_df = period_df.merge(inc, on=date_col, how="left")
            period_df = period_df.merge(dec, on=date_col, how="left")
        else:
            period_df["holders_increased"] = 0
            period_df["holders_decreased"] = 0

        period_df = period_df.rename(columns={date_col: "period_end"})
        period_df = period_df.sort_values("period_end")

        if len(period_df) < 4:
            return

        from quantcore.features.institutional_signals import LSVHerding
        herding_df = LSVHerding().compute(period_df)

        last = herding_df.iloc[-1]
        result["inst_herding_measure"] = _sf(last.get("herding_measure"))
        hbias = last.get("herding_buy_bias")
        result["inst_herding_buy_bias"] = int(hbias) if pd.notna(hbias) else None
        hhigh = last.get("herding_high")
        result["inst_herding_high"] = int(hhigh) if pd.notna(hhigh) else None

    except Exception as exc:
        logger.debug(f"[fundamentals] {symbol}: herding signals failed: {exc}")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _sf(v: Any) -> float | None:
    """Safe float conversion with NaN guard."""
    if v is None:
        return None
    try:
        f = float(v)
        return None if f != f else f  # NaN guard
    except (TypeError, ValueError):
        return None
