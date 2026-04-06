# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Earnings quality collector — composite quality score from fundamentals.

Fetches financial metrics, income statements, and cash flow statements via
FinancialDatasetsClient to compute accrual ratio, FCF conversion, revenue
consistency, and a composite quality score.  Returns {} when the API key
is missing or data is unavailable — never raises.
"""

import asyncio
import os
import statistics
from typing import Any

from loguru import logger

from quantstack.data.adapters.financial_datasets_client import FinancialDatasetsClient
from quantstack.data.storage import DataStore


async def collect_quality(symbol: str, store: DataStore) -> dict[str, Any]:
    """
    Compute earnings quality metrics for *symbol*.

    Returns a dict with keys:
        roe                         : float | None — return on equity
        fcf_yield                   : float | None — free cash flow yield
        debt_to_equity              : float | None
        accrual_ratio               : float | None — (net_income - operating_cf) / total_assets
        revenue_growth_consistency  : float 0-1 — inverted std of quarterly rev growth
        fcf_conversion              : float | None — operating_cf / net_income
        quality_score               : float 0-1 — composite quality score

    Returns {} if API key is missing or data cannot be fetched.
    """
    try:
        return await asyncio.to_thread(_collect_quality_sync, symbol, store)
    except Exception as exc:
        logger.warning(f"[quality] {symbol}: {exc} — returning empty")
        return {}


def _collect_quality_sync(symbol: str, store: DataStore) -> dict[str, Any]:
    """Synchronous quality collection — called via asyncio.to_thread."""
    api_key = os.environ.get("FINANCIAL_DATASETS_API_KEY", "")
    if not api_key:
        logger.debug("[quality] FINANCIAL_DATASETS_API_KEY not set — skipping")
        return {}

    with FinancialDatasetsClient(api_key=api_key) as client:
        metrics_resp = client.get_financial_metrics(symbol, period="annual", limit=1)
        income_resp = client.get_income_statements(symbol, period="quarterly", limit=8)
        cashflow_resp = client.get_cash_flow_statements(
            symbol, period="quarterly", limit=4
        )

    # --- Financial metrics (latest annual snapshot) ---
    metrics_list = (metrics_resp or {}).get("financial_metrics", [])
    metrics = metrics_list[0] if metrics_list else {}

    roe = _safe_float(metrics.get("return_on_equity"))
    fcf_yield = _safe_float(metrics.get("free_cash_flow_yield"))
    debt_to_equity = _safe_float(metrics.get("debt_to_equity"))

    # --- Accrual ratio: (net_income - operating_cash_flow) / total_assets ---
    accrual_ratio = _compute_accrual_ratio(income_resp, cashflow_resp)

    # --- Revenue growth consistency (last 8 quarters) ---
    revenue_growth_consistency = _compute_revenue_consistency(income_resp)

    # --- FCF conversion: operating_cash_flow / net_income ---
    fcf_conversion = _compute_fcf_conversion(income_resp, cashflow_resp)

    # --- Composite quality score ---
    quality_score = _compute_quality_score(
        roe, debt_to_equity, accrual_ratio, fcf_conversion
    )

    return {
        "roe": roe,
        "fcf_yield": fcf_yield,
        "debt_to_equity": debt_to_equity,
        "accrual_ratio": _round_or_none(accrual_ratio, 4),
        "revenue_growth_consistency": _round_or_none(revenue_growth_consistency, 4),
        "fcf_conversion": _round_or_none(fcf_conversion, 4),
        "quality_score": _round_or_none(quality_score, 4),
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _compute_accrual_ratio(
    income_resp: dict[str, Any] | None,
    cashflow_resp: dict[str, Any] | None,
) -> float | None:
    """(net_income - operating_cash_flow) / total_assets.

    High accruals = low earnings quality. Uses the most recent quarter from
    each statement set.
    """
    income_list = (income_resp or {}).get("income_statements", [])
    cf_list = (cashflow_resp or {}).get("cash_flow_statements", [])

    if not income_list or not cf_list:
        return None

    latest_income = income_list[0]
    latest_cf = cf_list[0]

    net_income = _safe_float(latest_income.get("net_income"))
    operating_cf = _safe_float(latest_cf.get("operating_cash_flow"))
    total_assets = _safe_float(latest_income.get("total_assets"))

    # total_assets might not be on the income statement — try balance-sheet-like
    # fields that FinancialDatasets sometimes nests.
    if total_assets is None or total_assets == 0:
        total_assets = _safe_float(latest_cf.get("total_assets"))
    if total_assets is None or total_assets == 0:
        return None

    if net_income is None or operating_cf is None:
        return None

    return (net_income - operating_cf) / total_assets


def _compute_revenue_consistency(income_resp: dict[str, Any] | None) -> float | None:
    """Inverted normalised std of quarter-over-quarter revenue growth.

    Returns a 0-1 score where 1 = perfectly consistent growth.
    Needs at least 3 quarters to be meaningful.
    """
    income_list = (income_resp or {}).get("income_statements", [])
    if len(income_list) < 3:
        return None

    revenues = [_safe_float(q.get("revenue")) for q in income_list]
    # Filter out None / zero values
    revenues = [r for r in revenues if r is not None and r > 0]
    if len(revenues) < 3:
        return None

    # Quarter-over-quarter growth rates (oldest to newest)
    revenues = list(reversed(revenues))  # chronological order
    growth_rates = [
        (revenues[i] - revenues[i - 1]) / revenues[i - 1]
        for i in range(1, len(revenues))
    ]

    if not growth_rates:
        return None

    std = statistics.pstdev(growth_rates)
    # Invert: low std = high consistency. Cap at 1.0.
    # A std of 0.5 (50% variation) maps to ~0.0; std of 0 maps to 1.0.
    return max(0.0, min(1.0, 1.0 - std * 2))


def _compute_fcf_conversion(
    income_resp: dict[str, Any] | None,
    cashflow_resp: dict[str, Any] | None,
) -> float | None:
    """operating_cash_flow / net_income.

    >1 means cash earnings exceed accrual earnings (good quality).
    """
    income_list = (income_resp or {}).get("income_statements", [])
    cf_list = (cashflow_resp or {}).get("cash_flow_statements", [])

    if not income_list or not cf_list:
        return None

    net_income = _safe_float(income_list[0].get("net_income"))
    operating_cf = _safe_float(cf_list[0].get("operating_cash_flow"))

    if net_income is None or operating_cf is None or net_income == 0:
        return None

    return operating_cf / net_income


def _compute_quality_score(
    roe: float | None,
    debt_to_equity: float | None,
    accrual_ratio: float | None,
    fcf_conversion: float | None,
) -> float | None:
    """Composite 0-1 quality score.

    Components (equal weight when available):
        - ROE score: higher is better, capped at 30%
        - Debt score: lower is better, capped at 2.0
        - Accrual score: lower (more negative or near zero) is better
        - FCF conversion score: higher is better, capped at 2.0
    """
    scores: list[float] = []

    if roe is not None:
        # ROE: 0% -> 0.0, 15% -> 0.5, 30%+ -> 1.0
        scores.append(max(0.0, min(1.0, roe / 0.30)))

    if debt_to_equity is not None:
        # D/E: 0 -> 1.0, 1.0 -> 0.5, 2.0+ -> 0.0
        scores.append(max(0.0, min(1.0, 1.0 - debt_to_equity / 2.0)))

    if accrual_ratio is not None:
        # Accrual: -0.1 -> 1.0, 0 -> 0.75, 0.1 -> 0.5, 0.5+ -> 0.0
        scores.append(max(0.0, min(1.0, 0.75 - accrual_ratio * 2.5 + 0.25)))

    if fcf_conversion is not None:
        # FCF conversion: 0 -> 0.0, 1.0 -> 0.5, 2.0+ -> 1.0
        scores.append(max(0.0, min(1.0, fcf_conversion / 2.0)))

    if not scores:
        return None

    return sum(scores) / len(scores)


def _safe_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
        return None if f != f else f  # NaN guard
    except (TypeError, ValueError):
        return None


def _round_or_none(v: float | None, digits: int) -> float | None:
    return round(v, digits) if v is not None else None
