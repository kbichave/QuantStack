# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Macro collector — yield curve, rate momentum, cross-asset risk appetite.

Fetches treasury rate data and crypto prices from FinancialDatasets.ai to
compute macro regime context.  Returns {} if no API key is configured —
macro signals are supplementary, never blocking.
"""

import asyncio
import os
from datetime import date, timedelta
from typing import Any

import pandas as pd
from loguru import logger

from quantstack.data.adapters.financial_datasets_client import FinancialDatasetsClient
from quantstack.data.storage import DataStore
from quantstack.core.features.macro_features import CreditSpreadFeatures, VolOfVol
from quantstack.core.features.rates import SpreadSignals, YieldCurveFeatures
from quantstack.data.fred_fetcher import FREDFetcher


_TIMEOUT_SECONDS = 10.0
_RATE_LOOKBACK_DAYS = 30  # enough for 20-day trend classification


async def collect_macro(symbol: str, store: DataStore) -> dict[str, Any]:
    """Compute macro signals: yield curve, rate momentum, risk appetite. Returns {} on failure."""
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(_collect_macro_sync, symbol, store),
            timeout=_TIMEOUT_SECONDS,
        )
    except (asyncio.TimeoutError, Exception) as exc:
        logger.debug(f"[macro] {symbol}: {type(exc).__name__} — returning empty")
        return {}


def _collect_macro_sync(symbol: str, store: DataStore) -> dict[str, Any]:
    api_key = os.environ.get("FINANCIAL_DATASETS_API_KEY", "")
    if not api_key:
        logger.debug("[macro] FINANCIAL_DATASETS_API_KEY not set — skipping")
        return {}

    end = date.today()
    start = end - timedelta(days=_RATE_LOOKBACK_DAYS)

    with FinancialDatasetsClient(api_key=api_key) as client:
        rates_resp = client.get_interest_rates_historical(
            start_date=start.isoformat(),
            end_date=end.isoformat(),
        )
        crypto_resp = client.get_crypto_prices(
            ticker="BTC-USD",
            interval="day",
            interval_multiplier=1,
            start_date=(end - timedelta(days=20)).isoformat(),
            end_date=end.isoformat(),
        )

    result: dict[str, Any] = {}

    # --- Yield curve and rate momentum ---
    rates = _extract_rate_series(rates_resp)
    if rates:
        latest = rates[-1]
        ten_yr = _safe_float(latest.get("ten_year") or latest.get("10_year"))
        two_yr = _safe_float(latest.get("two_year") or latest.get("2_year"))

        if ten_yr is not None and two_yr is not None:
            result["yield_curve_slope"] = round(ten_yr - two_yr, 4)
        else:
            result["yield_curve_slope"] = None

        # Rate momentum: 5-day change in 10Y
        if ten_yr is not None and len(rates) >= 6:
            prior_10y = _safe_float(
                rates[-6].get("ten_year") or rates[-6].get("10_year")
            )
            if prior_10y is not None:
                result["rate_momentum_5d"] = round(ten_yr - prior_10y, 4)
            else:
                result["rate_momentum_5d"] = None
        else:
            result["rate_momentum_5d"] = None

        # Rate regime: direction of 10Y over last 20 observations
        result["rate_regime"] = _classify_rate_regime(rates)

        # YieldCurveFeatures — 3m10y spread, inversion flag, smoothed slope
        try:
            dates = [r.get("date") or r.get("report_date") for r in rates]
            idx = pd.to_datetime(dates)
            s10y = pd.Series(
                [_safe_float(r.get("ten_year") or r.get("10_year")) for r in rates],
                index=idx,
                dtype=float,
            )
            s2y = pd.Series(
                [_safe_float(r.get("two_year") or r.get("2_year")) for r in rates],
                index=idx,
                dtype=float,
            )
            s3m = pd.Series(
                [_safe_float(r.get("three_month") or r.get("3_month")) for r in rates],
                index=idx,
                dtype=float,
            )
            if s10y.notna().sum() >= 5 and s2y.notna().sum() >= 5:
                ycf_df = YieldCurveFeatures(smooth_period=5).compute(s2y, s3m, s10y)
                result["yc_2s10s"] = _safe_float(ycf_df["spread_2s10s"].iloc[-1])
                result["yc_3m10y"] = _safe_float(ycf_df["spread_3m10y"].iloc[-1])
                result["yc_inverted"] = int(ycf_df["inverted"].iloc[-1])
                result["yc_slope_smooth"] = _safe_float(ycf_df["slope_smooth"].iloc[-1])
        except Exception as exc:
            logger.debug(f"[macro] YieldCurveFeatures failed: {exc}")

        # SpreadSignals — TED spread proxy (3M T-bill vs fed funds / overnight rate)
        # Uses 3M Treasury as the "risky" short rate and the 1-month or fed-funds
        # equivalent as overnight.  If the 1M series is available use it; else fall
        # back to the 2Y as a conservative short-term proxy.
        try:
            s1m_vals = [
                _safe_float(r.get("one_month") or r.get("1_month")) for r in rates
            ]
            s1m = pd.Series(s1m_vals, index=idx, dtype=float)
            overnight_proxy = s1m if s1m.notna().sum() >= 5 else s2y
            if s3m.notna().sum() >= 5 and overnight_proxy.notna().sum() >= 5:
                sp_df = SpreadSignals(smooth_period=5).compute(s3m, overnight_proxy)
                result["ted_spread"] = _safe_float(sp_df["ted_spread"].iloc[-1])
                result["ted_spread_zscore"] = _safe_float(
                    sp_df["ted_spread_zscore"].iloc[-1]
                )
                result["credit_stress"] = int(sp_df["credit_stress"].iloc[-1])
                result["spread_widening"] = int(sp_df["spread_widening"].iloc[-1])
        except Exception as exc:
            logger.debug(f"[macro] SpreadSignals failed: {exc}")
    else:
        result["yield_curve_slope"] = None
        result["rate_momentum_5d"] = None
        result["rate_regime"] = "unknown"

    # --- Risk appetite proxy from BTC momentum ---
    # Note: cross_asset.py independently computes risk_on_score from SPY/TLT/GLD.
    # Both signals are intentional — BTC provides a faster-moving risk appetite
    # read that leads equities by 1-2 days; cross_asset is equity-anchored.
    # If FINANCIAL_DATASETS_API_KEY is not set, this returns None and SignalBrief
    # stores None for risk_appetite_proxy, which is expected and not an error.
    result["risk_appetite_proxy"] = _compute_risk_appetite(crypto_resp)

    # --- FRED-sourced macro signals (optional, degrades gracefully) ---
    try:
        fred = FREDFetcher()
        hy_data = fred.fetch("hy_oas", days=365)
        if not hy_data.empty and len(hy_data) >= 30:
            oas = pd.Series(
                hy_data["value"].values,
                index=pd.to_datetime(hy_data["date"]),
                dtype=float,
            )
            cs_df = CreditSpreadFeatures().compute(oas)
            result["hy_oas"] = _safe_float(cs_df["hy_oas"].iloc[-1])
            result["hy_oas_zscore"] = _safe_float(cs_df["hy_oas_zscore"].iloc[-1])
            result["credit_regime_fred"] = cs_df["credit_regime"].iloc[-1]
    except Exception as exc:
        logger.debug(f"[macro] FRED credit signals skipped: {exc}")

    return result


def _extract_rate_series(resp: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Pull the rate observations list from the API response."""
    if resp is None:
        return []
    # API returns {"interest_rates": [...]} or {"rates": [...]}
    rates = resp.get("interest_rates") or resp.get("rates") or []
    if not isinstance(rates, list):
        return []
    return rates


def _classify_rate_regime(rates: list[dict[str, Any]]) -> str:
    """Classify rate regime from recent 10Y observations.

    Uses up to the last 20 data points.  Linear direction:
    - "rising"  if 10Y end > 10Y start by more than 10 bps
    - "falling" if 10Y end < 10Y start by more than 10 bps
    - "stable"  otherwise
    """
    window = rates[-20:] if len(rates) >= 20 else rates
    if len(window) < 3:
        return "unknown"

    first_10y = _safe_float(window[0].get("ten_year") or window[0].get("10_year"))
    last_10y = _safe_float(window[-1].get("ten_year") or window[-1].get("10_year"))

    if first_10y is None or last_10y is None:
        return "unknown"

    delta = last_10y - first_10y
    if delta > 0.10:
        return "rising"
    if delta < -0.10:
        return "falling"
    return "stable"


def _compute_risk_appetite(crypto_resp: dict[str, Any] | None) -> float | None:
    """BTC 10-day momentum normalized to 0-1 as a risk-on/off proxy.

    None if crypto data unavailable — caller can ignore gracefully.
    """
    if crypto_resp is None:
        return None
    prices = crypto_resp.get("prices") or []
    if len(prices) < 10:
        return None

    recent = prices[-10:]
    first_close = _safe_float(recent[0].get("close"))
    last_close = _safe_float(recent[-1].get("close"))
    if first_close is None or last_close is None or first_close <= 0:
        return None

    # Momentum as pct change, then sigmoid-style squeeze to [0, 1]
    pct_change = (last_close - first_close) / first_close
    # Clamp to [-0.2, 0.2] range then map to [0, 1]
    clamped = max(-0.20, min(0.20, pct_change))
    score = round((clamped + 0.20) / 0.40, 3)
    return score


def _safe_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
        return None if f != f else f  # NaN guard
    except (TypeError, ValueError):
        return None
