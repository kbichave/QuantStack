# Copyright 2024 QuantArena Contributors
# SPDX-License-Identifier: Apache-2.0

"""
FastAPI backend for Historical QuantArena UI.

Provides READ-ONLY endpoints to visualize completed simulation results.
No write operations - this is purely a viewer for the experience store.

Endpoints:
    GET /symbols           - List of symbols in universe
    GET /equity_curve      - Portfolio equity over time
    GET /price_series      - OHLCV data for a symbol
    GET /trades            - Trade history
    GET /agent_logs        - Agent chat timeline
    GET /regimes           - Regime history for a symbol
    GET /policy_snapshots  - Policy evolution history

Usage:
    uvicorn examples.historical_quant_arena_ui.backend.api:app --reload

    # Or run directly
    python examples/historical_quant_arena_ui/backend/api.py
"""

from __future__ import annotations

import json
import os
from datetime import date, datetime
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Ensure repo src is on path for quantcore imports
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PACKAGES_PATH = PROJECT_ROOT / "packages"
if str(PACKAGES_PATH) not in sys.path:
    sys.path.insert(0, str(PACKAGES_PATH))


# =============================================================================
# RESPONSE MODELS
# =============================================================================


class SymbolInfo(BaseModel):
    """Symbol information."""

    symbol: str
    ticker: str
    description: str
    asset_class: str


class EquityPoint(BaseModel):
    """Single point on equity curve."""

    date: str
    equity: float
    cash: float
    max_drawdown: float
    regime_summary: Optional[str] = None


class PriceBar(BaseModel):
    """OHLCV price bar."""

    date: str
    open: float
    high: float
    low: float
    close: float
    volume: float


class TradeRecord(BaseModel):
    """Trade record for visualization."""

    date: str
    symbol: str
    side: str
    quantity: int
    price: float
    pnl: Optional[float] = None


class AgentLogEntry(BaseModel):
    """Agent log entry for chat timeline."""

    date: str
    agent_name: str
    symbol: Optional[str] = None
    message: str
    role: str


class RegimeEntry(BaseModel):
    """Regime state entry."""

    date: str
    symbol: str
    trend: str
    volatility: str


class PolicyEntry(BaseModel):
    """Policy snapshot entry."""

    effective_date: str
    pod_weights: Dict[str, float]
    comment: str


# =============================================================================
# API APPLICATION
# =============================================================================

app = FastAPI(
    title="Historical QuantArena API",
    description="Read-only API for viewing historical simulation results",
    version="0.1.0",
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to specific origins
    allow_credentials=True,
    allow_methods=["GET"],  # Read-only
    allow_headers=["*"],
)


def get_db_connection(retries: int = 3, delay: float = 0.5):
    """
    Get a fresh DuckDB connection for read-only access.
    Returns None if database is not available.

    Uses retries to handle temporary lock contention with writer process.
    """
    import time
    import duckdb

    db_path = os.getenv("ALPHA_ARENA_DB_PATH", "~/.quant_pod/knowledge.duckdb")
    expanded_path = os.path.expanduser(db_path)

    if not os.path.exists(expanded_path):
        return None

    last_error = None
    for attempt in range(retries):
        try:
            conn = duckdb.connect(expanded_path, read_only=True)
            return conn
        except Exception as e:
            last_error = e
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))  # Exponential backoff

    print(f"DB connection error after {retries} attempts: {last_error}")
    return None


def safe_query(
    query: str, params: tuple = (), fallback_table: Optional[str] = None
) -> List[Dict]:
    """Execute query safely, returning empty list on error. Optionally retry with fallback table."""
    conn = get_db_connection()
    if conn is None:
        return []
    try:
        result = conn.execute(query, params).fetchall()
        columns = [desc[0] for desc in conn.description]
        return [dict(zip(columns, row)) for row in result]
    except Exception as e:
        # Retry with fallback table if provided
        if fallback_table and "does not exist" in str(e):
            try:
                q = query.replace(
                    "FROM " + query.split("FROM")[1].split()[0],
                    f"FROM {fallback_table}",
                )
                result = conn.execute(q, params).fetchall()
                columns = [desc[0] for desc in conn.description]
                return [dict(zip(columns, row)) for row in result]
            except Exception:
                pass
        print(f"Query error: {e}")
        return []
    finally:
        try:
            conn.close()
        except:
            pass


# =============================================================================
# ENDPOINTS
# =============================================================================


@app.get("/")
async def root():
    """API root - health check."""
    return {
        "status": "ok",
        "service": "Historical QuantArena API",
        "version": "0.1.0",
        "read_only": True,
    }


@app.get("/symbols", response_model=List[SymbolInfo])
async def get_symbols():
    """Get list of symbols in the universe."""
    symbols = [
        SymbolInfo(
            symbol="SPY",
            ticker="SPY",
            description="SPDR S&P 500 ETF Trust",
            asset_class="equity_index",
        ),
        SymbolInfo(
            symbol="QQQ",
            ticker="QQQ",
            description="Invesco QQQ Trust (Nasdaq 100)",
            asset_class="equity_index",
        ),
        SymbolInfo(
            symbol="IWM",
            ticker="IWM",
            description="iShares Russell 2000 ETF",
            asset_class="equity_index",
        ),
    ]
    return symbols


@app.get("/equity_curve", response_model=List[EquityPoint])
async def get_equity_curve(
    start: Optional[str] = Query(None, description="Start date YYYY-MM-DD"),
    end: Optional[str] = Query(None, description="End date YYYY-MM-DD"),
):
    """Get portfolio equity curve."""
    query = """
        SELECT date, equity, cash, max_drawdown, regime_summary
        FROM equity_curve
        WHERE 1=1
    """
    params = []
    if start:
        query += " AND date >= ?"
        params.append(start)
    if end:
        query += " AND date <= ?"
        params.append(end)
    query += " ORDER BY date"

    data = safe_query(query, tuple(params), fallback_table="portfolio_equity")

    return [
        EquityPoint(
            date=str(d.get("date", "")),
            equity=d.get("equity", 0) or 0,
            cash=d.get("cash", 0) or 0,
            max_drawdown=d.get("max_drawdown", 0) or 0,
            regime_summary=d.get("regime_summary"),
        )
        for d in data
    ]


@app.get("/price_series", response_model=List[PriceBar])
async def get_price_series(
    symbol: str = Query(..., description="Symbol to get prices for"),
    start: Optional[str] = Query(None, description="Start date YYYY-MM-DD"),
    end: Optional[str] = Query(None, description="End date YYYY-MM-DD"),
):
    """Get OHLCV price series for a symbol."""
    try:
        from quantcore.data.storage import DataStore

        data_store = DataStore()
        df = data_store.load(symbol, "daily")

        if df is not None and not df.empty:
            if start:
                df = df[df.index >= start]
            if end:
                df = df[df.index <= end]

            return [
                PriceBar(
                    date=str(idx.date()) if hasattr(idx, "date") else str(idx),
                    open=float(row.get("open", row.get("Open", 0))),
                    high=float(row.get("high", row.get("High", 0))),
                    low=float(row.get("low", row.get("Low", 0))),
                    close=float(row.get("close", row.get("Close", 0))),
                    volume=float(row.get("volume", row.get("Volume", 0))),
                )
                for idx, row in df.iterrows()
            ]
    except Exception as e:
        print(f"Price series error: {e}")

    return []


@app.get("/trades", response_model=List[TradeRecord])
async def get_trades(
    symbol: Optional[str] = Query(None, description="Filter by symbol"),
    start: Optional[str] = Query(None, description="Start date YYYY-MM-DD"),
    end: Optional[str] = Query(None, description="End date YYYY-MM-DD"),
    limit: int = Query(100, description="Maximum records to return"),
):
    """Get trade history."""
    query = """
        SELECT created_at, symbol, direction, quantity, entry_price, pnl
        FROM trades
        WHERE 1=1
    """
    params = []
    if symbol:
        query += " AND symbol = ?"
        params.append(symbol)
    if start:
        query += " AND DATE(created_at) >= ?"
        params.append(start)
    if end:
        query += " AND DATE(created_at) <= ?"
        params.append(end)
    query += f" ORDER BY created_at DESC LIMIT {limit}"

    data = safe_query(query, tuple(params), fallback_table="trade_journal")

    return [
        TradeRecord(
            date=str(t.get("created_at", ""))[:10],
            symbol=t.get("symbol", ""),
            side=t.get("direction", ""),
            quantity=t.get("quantity", 0) or 0,
            price=t.get("entry_price", 0) or 0,
            pnl=t.get("pnl"),
        )
        for t in data
    ]


@app.get("/agent_logs", response_model=List[AgentLogEntry])
async def get_agent_logs(
    symbol: Optional[str] = Query(None, description="Filter by symbol"),
    agent: Optional[str] = Query(None, description="Filter by agent name"),
    start: Optional[str] = Query(None, description="Start date YYYY-MM-DD"),
    end: Optional[str] = Query(None, description="End date YYYY-MM-DD"),
    limit: int = Query(1000, description="Maximum records to return"),
):
    """Get agent chat timeline from blackboard.md file."""
    import re

    # Read from blackboard.md file (doesn't have lock issues)
    blackboard_path = os.path.expanduser("~/.quant_pod/blackboard.md")

    results = []

    if os.path.exists(blackboard_path):
        try:
            with open(blackboard_path, "r") as f:
                content = f.read()

            # Parse markdown blocks: ### [timestamp] AgentName
            blocks = re.split(r"\n---\n", content)

            for block in blocks:
                if not block.strip() or block.startswith("# Blackboard"):
                    continue

                # Parse header: ### [2024-01-31 10:00:00] AgentName
                header_match = re.search(r"###\s*\[([^\]]+)\]\s*(\S+)", block)
                if not header_match:
                    continue

                timestamp = header_match.group(1)
                agent_name = header_match.group(2)

                # Parse symbol
                symbol_match = re.search(r"\*\*Symbol:\*\*\s*(\S+)", block)
                log_symbol = symbol_match.group(1) if symbol_match else None

                # Get message (everything after symbol line)
                lines = block.split("\n")
                message_lines = []
                capture = False
                for line in lines:
                    if capture and line.strip() and not line.startswith("---"):
                        message_lines.append(line)
                    if "**Symbol:**" in line:
                        capture = True

                message = " ".join(message_lines).strip()
                if not message:
                    # Fallback: get all non-header content
                    message = re.sub(r"###\s*\[[^\]]+\]\s*\S+", "", block)
                    message = re.sub(r"\*\*Symbol:\*\*\s*\S+", "", message)
                    message = message.strip()

                # Apply filters
                if symbol and log_symbol and symbol.lower() != log_symbol.lower():
                    continue
                if agent and agent.lower() not in agent_name.lower():
                    continue

                # Date filter
                try:
                    log_date = timestamp[:10]
                    if start and log_date < start:
                        continue
                    if end and log_date > end:
                        continue
                except:
                    pass

                results.append(
                    AgentLogEntry(
                        date=timestamp,
                        agent_name=agent_name,
                        symbol=log_symbol,
                        message=message[:500] if message else "Analysis complete",
                        role="analysis",
                    )
                )
        except Exception as e:
            print(f"Blackboard read error: {e}")

    # Sort by date descending and limit
    results.sort(key=lambda x: x.date, reverse=True)
    return results[:limit]


@app.get("/regimes", response_model=List[RegimeEntry])
async def get_regimes(
    symbol: str = Query(..., description="Symbol to get regimes for"),
    start: Optional[str] = Query(None, description="Start date YYYY-MM-DD"),
    end: Optional[str] = Query(None, description="End date YYYY-MM-DD"),
):
    """Get regime history for a symbol."""
    query = """
        SELECT timestamp, symbol, trend_regime, volatility_regime
        FROM regime_states
        WHERE symbol = ?
    """
    params = [symbol]

    if start:
        query += " AND DATE(timestamp) >= ?"
        params.append(start)
    if end:
        query += " AND DATE(timestamp) <= ?"
        params.append(end)

    query += " ORDER BY timestamp ASC"

    data = safe_query(query, tuple(params))

    return [
        RegimeEntry(
            date=str(row.get("timestamp", ""))[:10],
            symbol=symbol,
            trend=row.get("trend_regime", "unknown"),
            volatility=row.get("volatility_regime", "normal"),
        )
        for row in data
    ]


@app.get("/policy_snapshots", response_model=List[PolicyEntry])
async def get_policy_snapshots(
    start: Optional[str] = Query(None, description="Start date YYYY-MM-DD"),
    end: Optional[str] = Query(None, description="End date YYYY-MM-DD"),
):
    """Get policy evolution history."""
    query = """
        SELECT effective_date, pod_weights, comment
        FROM policy_snapshots
        WHERE 1=1
    """
    params = []
    if start:
        query += " AND DATE(effective_date) >= ?"
        params.append(start)
    if end:
        query += " AND DATE(effective_date) <= ?"
        params.append(end)
    query += " ORDER BY effective_date"

    data = safe_query(query, tuple(params))

    results = []
    for s in data:
        pod_weights = s.get("pod_weights", {})
        if isinstance(pod_weights, str):
            try:
                pod_weights = json.loads(pod_weights)
            except:
                pod_weights = {}
        results.append(
            PolicyEntry(
                effective_date=str(s.get("effective_date", "")),
                pod_weights=pod_weights if isinstance(pod_weights, dict) else {},
                comment=s.get("comment", ""),
            )
        )
    return results


@app.get("/summary")
async def get_summary():
    """Get simulation summary statistics."""
    equity_data = safe_query(
        """
        SELECT date, equity, cash, max_drawdown
        FROM equity_curve
        ORDER BY date
    """
    )

    if not equity_data:
        return {
            "start_date": "",
            "end_date": "",
            "trading_days": 0,
            "initial_equity": 100000,
            "final_equity": 100000,
            "total_return": 0,
            "max_drawdown": 0,
            "total_agent_messages": 0,
            "status": "No simulation data yet - simulation in progress",
        }

    first = equity_data[0]
    last = equity_data[-1]

    initial_equity = first.get("equity", 100000) or 100000
    final_equity = last.get("equity", 100000) or 100000
    total_return = (
        (final_equity - initial_equity) / initial_equity if initial_equity else 0
    )

    max_dd = max((d.get("max_drawdown", 0) or 0) for d in equity_data)

    log_count = safe_query("SELECT COUNT(*) as cnt FROM agent_logs")
    total_logs = log_count[0]["cnt"] if log_count else 0

    return {
        "start_date": str(first.get("date", "")),
        "end_date": str(last.get("date", "")),
        "trading_days": len(equity_data),
        "initial_equity": initial_equity,
        "final_equity": final_equity,
        "total_return": total_return,
        "max_drawdown": max_dd,
        "total_agent_messages": total_logs,
    }


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
