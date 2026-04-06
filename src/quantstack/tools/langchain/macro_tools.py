"""Macro and market breadth tools for LangGraph agents."""

import json

from langchain_core.tools import tool


@tool
async def get_credit_market_signals() -> str:
    """Retrieves credit market stress indicators and macro regime classification using ETF price proxies. Use when gating bottom-fishing or buy-the-dip entries with macro context, assessing risk-on vs risk-off conditions, or monitoring credit spread dynamics. Computes high-yield/investment-grade spread ratio (HYG/LQD), yield curve slope proxy (TLT/SHY), US dollar direction (UUP), gold-bond divergence for inflation fear detection, and an overall credit regime label (widening, stable, contracting). Returns JSON with credit_regime, hy_spread_zscore, yield_curve_slope, dollar_direction, risk_on_score, bottom_signal gate, and underlying ETF price details."""
    result = {"error": "Tool pending implementation", "status": "not_available"}
    return json.dumps(result, default=str)


@tool
async def get_market_breadth() -> str:
    """Computes market breadth and sector participation width using 15 ETF proxies (11 sector ETFs plus SPY, QQQ, IWM, MDY). Use when assessing overall market health, detecting breadth divergences that precede selloffs or bottoms, or identifying sector rotation patterns. Calculates the percentage of ETFs above their 50-day SMA, 5-day breadth trend direction (rising/falling/stable), price-breadth divergence signals, count of sectors above all key moving averages (20d/50d/200d), and ranks strongest and weakest sectors by relative performance. Returns JSON with breadth_score, breadth_trend, breadth_divergence flag, sector_breakdown, and bottom_signal assessment."""
    result = {"error": "Tool pending implementation", "status": "not_available"}
    return json.dumps(result, default=str)
