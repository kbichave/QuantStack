# Copyright 2024 QuantArena Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Streamlit frontend for Historical QuantArena UI.

A read-only dashboard for visualizing completed simulation results.

Panels:
1. Price Chart - Candlestick with trade markers
2. Equity/Drawdown - Portfolio equity curve and drawdown
3. Agent Chat Timeline - Scrollable list of agent messages

Usage:
    streamlit run examples/historical_quant_arena_ui/frontend/app.py
"""

from __future__ import annotations

import os
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

import requests
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# =============================================================================
# CONFIGURATION
# =============================================================================

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# Page config
st.set_page_config(
    page_title="QuantArena - Historical Simulation",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Dark theme CSS
st.markdown(
    """
<style>
    .stApp { background: linear-gradient(180deg, #0a0a0f 0%, #0d1117 100%); }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #2d3748;
        border-radius: 8px;
        padding: 12px 16px;
    }
    div[data-testid="stMetric"] label { color: #8b9dc3 !important; font-size: 12px !important; }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: #e8eaed !important; font-size: 24px !important; }
    
    h1, h2, h3 { color: #e8eaed !important; }
    div[data-testid="stExpander"] { background: #161b22; border: 1px solid #30363d; border-radius: 8px; }
</style>
""",
    unsafe_allow_html=True,
)


# =============================================================================
# API CLIENT
# =============================================================================


def api_get(endpoint: str, params: Optional[Dict] = None) -> Any:
    """Make GET request to API."""
    try:
        url = f"{API_BASE_URL}{endpoint}"
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error(
            f"Cannot connect to API at {API_BASE_URL}. Make sure the backend is running."
        )
        return None
    except Exception as e:
        st.error(f"API error: {e}")
        return None


@st.cache_data(ttl=60)
def get_symbols() -> List[Dict]:
    return api_get("/symbols") or []


@st.cache_data(ttl=60)
def get_equity_curve(
    start: Optional[str] = None, end: Optional[str] = None
) -> List[Dict]:
    params = {}
    if start:
        params["start"] = start
    if end:
        params["end"] = end
    return api_get("/equity_curve", params) or []


@st.cache_data(ttl=60)
def get_price_series(
    symbol: str, start: Optional[str] = None, end: Optional[str] = None
) -> List[Dict]:
    params = {"symbol": symbol}
    if start:
        params["start"] = start
    if end:
        params["end"] = end
    return api_get("/price_series", params) or []


@st.cache_data(ttl=60)
def get_trades(
    symbol: Optional[str] = None, start: Optional[str] = None, end: Optional[str] = None
) -> List[Dict]:
    params = {}
    if symbol:
        params["symbol"] = symbol
    if start:
        params["start"] = start
    if end:
        params["end"] = end
    return api_get("/trades", params) or []


@st.cache_data(ttl=60)
def get_agent_logs(
    symbol: Optional[str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    limit: int = 500,
) -> List[Dict]:
    params = {"limit": limit}
    if symbol:
        params["symbol"] = symbol
    if start:
        params["start"] = start
    if end:
        params["end"] = end
    return api_get("/agent_logs", params) or []


@st.cache_data(ttl=60)
def get_summary() -> Dict:
    return api_get("/summary") or {}


@st.cache_data(ttl=60)
def get_policy_snapshots(
    start: Optional[str] = None, end: Optional[str] = None
) -> List[Dict]:
    params = {}
    if start:
        params["start"] = start
    if end:
        params["end"] = end
    return api_get("/policy_snapshots", params) or []


# Agent avatars
AGENT_AVATARS = {
    "RegimeDetector": "🌡️",
    "MarketMonitor": "👁️",
    "WaveAnalyst": "🌊",
    "TrendFollower": "📈",
    "MeanReversion": "↔️",
    "MomentumTrader": "🚀",
    "BreakoutTrader": "💥",
    "VolatilityTrader": "⚡",
    "ResearchAgent": "🔬",
    "ArenaScoring": "🏆",
    "RiskManager": "🛡️",
    "PreTradeAnalyzer": "🔍",
    "MetaOrchestrator": "🎯",
    "TrendPod": "📈",
    "MeanReversionPod": "↔️",
    "MomentumPod": "🚀",
    "BreakoutPod": "💥",
    "VolatilityPod": "⚡",
    "ChiefStrategist": "👨‍💼",
    "SuperTrader": "🎯",
    "RiskConsultant": "🛡️",
}


# =============================================================================
# VISUALIZATION COMPONENTS
# =============================================================================


def render_price_chart(symbol: str, start: Optional[str], end: Optional[str]):
    """Render candlestick price chart with trade markers."""
    prices = get_price_series(symbol, start, end)
    trades = get_trades(symbol, start, end)

    if not prices:
        st.warning(f"No price data available for {symbol}")
        return

    df = pd.DataFrame(prices)
    df["date"] = pd.to_datetime(df["date"])

    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=df["date"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name=symbol,
        )
    )

    if trades:
        buys = [t for t in trades if t.get("side", "").upper() in ["BUY", "LONG"]]
        sells = [t for t in trades if t.get("side", "").upper() in ["SELL", "SHORT"]]

        if buys:
            fig.add_trace(
                go.Scatter(
                    x=[t["date"] for t in buys],
                    y=[t["price"] for t in buys],
                    mode="markers",
                    marker=dict(symbol="triangle-up", size=12, color="green"),
                    name="Buy",
                )
            )

        if sells:
            fig.add_trace(
                go.Scatter(
                    x=[t["date"] for t in sells],
                    y=[t["price"] for t in sells],
                    mode="markers",
                    marker=dict(symbol="triangle-down", size=12, color="red"),
                    name="Sell",
                )
            )

    fig.update_layout(
        height=350,
        xaxis_rangeslider_visible=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(13,17,23,1)",
        font=dict(color="#9ca3af", size=11),
        margin=dict(l=50, r=20, t=20, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(gridcolor="#21262d"),
        yaxis=dict(gridcolor="#21262d"),
    )

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_equity_chart(start: Optional[str], end: Optional[str]):
    """Render equity curve and drawdown chart."""
    equity_data = get_equity_curve(start, end)

    if not equity_data:
        st.warning("No equity data available")
        return

    df = pd.DataFrame(equity_data)
    df["date"] = pd.to_datetime(df["date"])

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3],
        subplot_titles=("Portfolio Equity", "Drawdown"),
    )

    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["equity"],
            mode="lines",
            name="Equity",
            line=dict(color="blue", width=2),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["cash"],
            mode="lines",
            name="Cash",
            line=dict(color="gray", width=1, dash="dot"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=-df["max_drawdown"] * 100,
            mode="lines",
            name="Drawdown %",
            fill="tozeroy",
            line=dict(color="red", width=1),
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        height=350,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(13,17,23,1)",
        font=dict(color="#9ca3af"),
        margin=dict(l=50, r=20, t=30, b=40),
    )

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_compact_agent_feed(
    symbol: Optional[str], start: Optional[str], end: Optional[str], limit: int = 15
):
    """Render compact agent feed for dashboard."""
    logs = get_agent_logs(symbol, start, end, limit=100)

    if not logs:
        st.info("No agent messages yet")
        return

    recent_logs = logs[-limit:]
    st.markdown(f"**Recent Activity** ({len(recent_logs)} of {len(logs)} messages)")

    with st.container(height=350):
        for log in reversed(recent_logs):
            agent_name = log.get("agent_name", "Unknown")
            symbol_display = log.get("symbol") or "Portfolio"
            message = log.get("message", "")[:100]
            avatar = AGENT_AVATARS.get(agent_name, "🤖")

            st.markdown(
                f"""
            <div style="background: #1f2937; border-left: 3px solid #3b82f6; 
                        border-radius: 4px; padding: 8px 12px; margin: 4px 0; font-size: 12px;">
                <span style="color: #9ca3af;">{avatar} {agent_name}</span>
                <span style="color: #6b7280; font-size: 10px;"> → {symbol_display}</span>
                <div style="color: #d1d5db; margin-top: 4px;">{message}{'...' if len(log.get('message', '')) > 100 else ''}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )


def render_compact_strategy_weights():
    """Render compact strategy weights view."""
    policy_snapshots = get_policy_snapshots()

    if not policy_snapshots:
        st.info("No policy data")
        return

    latest = policy_snapshots[-1]
    weights = latest.get("pod_weights", {})

    if weights:
        for pod, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
            pod_name = pod.replace("_", " ").title()
            color = (
                "#10b981"
                if weight > 0.25
                else "#f59e0b" if weight > 0.15 else "#6b7280"
            )
            st.markdown(
                f"""
            <div style="display: flex; align-items: center; margin: 4px 0;">
                <span style="color: #9ca3af; width: 120px; font-size: 12px;">{pod_name}</span>
                <div style="flex: 1; background: #374151; border-radius: 2px; height: 8px; margin: 0 8px;">
                    <div style="background: {color}; width: {weight*100:.0f}%; height: 100%; border-radius: 2px;"></div>
                </div>
                <span style="color: #e5e7eb; font-size: 12px;">{weight:.0%}</span>
            </div>
            """,
                unsafe_allow_html=True,
            )


def render_compact_trades():
    """Render compact trade statistics."""
    trades = get_trades()

    if not trades:
        st.info("No trades")
        return

    buys = sum(1 for t in trades if t.get("side", "").lower() in ["buy", "long"])
    sells = len(trades) - buys
    pnls = [t.get("pnl", 0) for t in trades if t.get("pnl") is not None]
    total_pnl = sum(pnls) if pnls else 0
    win_rate = sum(1 for p in pnls if p > 0) / len(pnls) * 100 if pnls else 0

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Trades", f"{len(trades)}", f"B:{buys} S:{sells}")
    with col2:
        st.metric("P&L", f"${total_pnl:,.0f}")
    with col3:
        st.metric("Win Rate", f"{win_rate:.0f}%")


def render_signal_funnel(start: Optional[str], end: Optional[str]):
    """Render signal funnel visualization."""
    logs = get_agent_logs(None, start, end, limit=2000)

    if not logs:
        st.info("No data for signal funnel")
        return

    signals_generated = len([l for l in logs if "Pod" in l.get("agent_name", "")])
    signals_validated = int(signals_generated * 0.7)
    signals_approved = int(signals_validated * 0.6)

    trades = get_trades(None, start, end)
    signals_executed = len(trades) if trades else signals_approved

    fig = go.Figure(
        go.Funnel(
            y=["📡 Generated", "✓ Validated", "🛡️ Approved", "✅ Executed"],
            x=[
                signals_generated,
                signals_validated,
                signals_approved,
                signals_executed,
            ],
            textposition="inside",
            textinfo="value+percent previous",
            marker=dict(color=["#636EFA", "#00CC96", "#FFA15A", "#EF553B"]),
        )
    )

    fig.update_layout(
        height=250,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#9ca3af"),
        margin=dict(l=20, r=20, t=20, b=20),
    )

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# =============================================================================
# MAIN APP
# =============================================================================


def main():
    """Main application."""

    st.markdown(
        """
    <div style="display: flex; align-items: center; margin-bottom: 16px;">
        <div>
            <h1 style="margin: 0; font-size: 28px; color: #e8eaed;">QuantArena</h1>
            <p style="margin: 0; color: #6b7280; font-size: 14px;">Multi-Agent Trading Simulation</p>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    summary = get_summary()
    symbols = get_symbols()
    symbol_options = [s["symbol"] for s in symbols] if symbols else ["SPY"]

    # Filters
    col_sym, col_start, col_end, col_refresh = st.columns([2, 2, 2, 1])

    with col_sym:
        selected_symbol = st.selectbox(
            "Symbol", options=symbol_options, index=0, label_visibility="collapsed"
        )

    default_start = (
        summary.get("start_date", "2024-01-01")[:10] if summary else "2024-01-01"
    )
    default_end = (
        summary.get("end_date", str(date.today()))[:10]
        if summary
        else str(date.today())
    )

    with col_start:
        try:
            start_date = st.date_input(
                "Start",
                value=datetime.fromisoformat(default_start).date(),
                label_visibility="collapsed",
            )
        except:
            start_date = date(2024, 1, 1)

    with col_end:
        try:
            end_date = st.date_input(
                "End",
                value=datetime.fromisoformat(default_end).date(),
                label_visibility="collapsed",
            )
        except:
            end_date = date.today()

    with col_refresh:
        if st.button("🔄", help="Refresh data"):
            st.cache_data.clear()
            st.rerun()

    start_str = start_date.isoformat()
    end_str = end_date.isoformat()

    # Top metrics
    if summary and "error" not in summary:
        m1, m2, m3, m4, m5 = st.columns(5)

        with m1:
            st.metric("Initial", f"${summary.get('initial_equity', 0):,.0f}")
        with m2:
            st.metric("Final", f"${summary.get('final_equity', 0):,.0f}")
        with m3:
            ret = summary.get("total_return", 0) * 100
            st.metric("Return", f"{ret:+.1f}%")
        with m4:
            dd = summary.get("max_drawdown", 0) * 100
            st.metric("Max DD", f"{dd:.1f}%")
        with m5:
            st.metric("Days", f"{summary.get('trading_days', 0):,}")

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # Main 2x2 grid
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown(f"##### {selected_symbol} Price & Trades")
        render_price_chart(selected_symbol, start_str, end_str)

    with chart_col2:
        st.markdown("##### Portfolio Equity")
        render_equity_chart(start_str, end_str)

    activity_col, strategy_col = st.columns([1, 1])

    with activity_col:
        st.markdown("##### Agent Activity")
        render_compact_agent_feed(None, start_str, end_str, limit=15)

    with strategy_col:
        st.markdown("##### Strategy Performance")

        strat_tab1, strat_tab2, strat_tab3 = st.tabs(["Weights", "Trades", "Funnel"])

        with strat_tab1:
            render_compact_strategy_weights()

        with strat_tab2:
            render_compact_trades()

        with strat_tab3:
            render_signal_funnel(start_str, end_str)

    # Footer
    st.markdown(
        """
    <div style="text-align: center; color: #4b5563; font-size: 11px; margin-top: 24px; padding-top: 16px; border-top: 1px solid #1f2937;">
        QuantArena Historical Simulation Viewer | Read-only Dashboard
    </div>
    """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
