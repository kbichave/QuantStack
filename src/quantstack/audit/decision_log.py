# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Compliance audit trail — append-only log of all agent decisions.

Every IC analysis, pod synthesis, assistant brief, SuperTrader decision,
and execution/rejection event is written here in full. No deletes.

Key properties:
  - Append-only: no UPDATE or DELETE operations, ever
  - Queryable: full SQL queries on the log via PostgreSQL
  - Human-readable: output_summary is plain text, auditable without tools
  - Immutable context hash: SHA256 of inputs so data integrity is provable

Usage:
    log = get_decision_log()

    # Record an IC analysis
    log.record(DecisionEvent(
        event_id=str(uuid.uuid4()),
        session_id=session_id,
        event_type="ic_analysis",
        agent_name="TrendIC",
        agent_role="ic",
        symbol="SPY",
        confidence=0.72,
        tool_calls=[ToolCall(tool_name="compute_indicators", ...)],
        output_summary="Strong uptrend detected (ADX=35, EMA12 > EMA26)",
    ))

    # Query all AAPL decisions from today
    events = log.query(AuditQuery(symbol="AAPL"))

    # Get full decision trace for a SuperTrader decision
    trace = log.get_decision_trace(event_id="...")
"""

from __future__ import annotations

import hashlib
import json
import uuid
from threading import Lock

from loguru import logger

from quantstack.audit.models import (
    AuditQuery,
    DecisionEvent,
    IndicatorAttribution,
    ToolCall,
)
from quantstack.db import PgConnection, open_db_readonly

# =============================================================================
# DECISION LOG
# =============================================================================


class DecisionLog:
    """
    Append-only compliance audit trail.

    Stores every agent decision with full context for regulatory review,
    explainability, and debugging.

    Preferred construction is via TradingContext which injects a shared
    PostgreSQL connection.  Schema is created by run_migrations() at startup.
    """

    def __init__(
        self,
        conn: PgConnection | None = None,
    ):
        # Instance-level lock so multiple DecisionLog objects don't share state
        self._lock = Lock()
        # Connection is injected; schema already managed by run_migrations()
        self._conn: PgConnection | None = conn
        logger.info("DecisionLog initialized")

    @property
    def conn(self) -> PgConnection:
        if self._conn is None:
            raise RuntimeError(
                "DecisionLog requires an injected PgConnection. "
                "Construct via TradingContext or pass conn= explicitly."
            )
        return self._conn

    def _init_schema(self) -> None:
        with self._lock:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS decision_events (
                    event_id                VARCHAR PRIMARY KEY,
                    session_id              VARCHAR NOT NULL,
                    event_type              VARCHAR NOT NULL,
                    agent_name              VARCHAR NOT NULL,
                    agent_role              VARCHAR NOT NULL,
                    symbol                  VARCHAR,
                    action                  VARCHAR,
                    confidence              DOUBLE PRECISION,
                    input_context_hash      VARCHAR,
                    market_data_snapshot    JSON,
                    portfolio_snapshot      JSON,
                    tool_calls              JSON,
                    output_summary          TEXT,
                    output_structured       JSON,
                    risk_approved           BOOLEAN,
                    risk_violations         JSON,
                    created_at              TIMESTAMP NOT NULL,
                    decision_latency_ms     INTEGER,
                    parent_event_ids        JSON
                )
            """
            )
            # Indices for common query patterns
            self.conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_events_session
                ON decision_events (session_id)
            """
            )
            self.conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_events_symbol_date
                ON decision_events (symbol, created_at)
            """
            )

    # -------------------------------------------------------------------------
    # Write (append-only)
    # -------------------------------------------------------------------------

    def record(self, event: DecisionEvent) -> str:
        """
        Record a decision event. Returns the event_id.

        This is the ONLY write method — no updates, no deletes.
        """
        if not event.event_id:
            event.event_id = str(uuid.uuid4())

        with self._lock:
            self.conn.execute(
                """
                INSERT INTO decision_events VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                [
                    event.event_id,
                    event.session_id,
                    event.event_type,
                    event.agent_name,
                    event.agent_role,
                    event.symbol,
                    event.action,
                    event.confidence,
                    event.input_context_hash,
                    json.dumps(event.market_data_snapshot),
                    json.dumps(event.portfolio_snapshot),
                    json.dumps([tc.model_dump() for tc in event.tool_calls]),
                    event.output_summary,
                    json.dumps(event.output_structured),
                    event.risk_approved,
                    json.dumps(event.risk_violations),
                    event.created_at,
                    event.decision_latency_ms,
                    json.dumps(event.parent_event_ids),
                ],
            )

        logger.debug(
            f"[AUDIT] Recorded {event.event_type} | {event.agent_name} | "
            f"{event.symbol or 'N/A'} | {event.action or 'N/A'}"
        )
        return event.event_id

    # -------------------------------------------------------------------------
    # Read
    # -------------------------------------------------------------------------

    def query(self, q: AuditQuery) -> list[DecisionEvent]:
        """Query the audit log with filters."""
        conditions = []
        params = []

        if q.symbol:
            conditions.append("symbol = ?")
            params.append(q.symbol)
        if q.agent_name:
            conditions.append("agent_name = ?")
            params.append(q.agent_name)
        if q.event_type:
            conditions.append("event_type = ?")
            params.append(q.event_type)
        if q.action:
            conditions.append("action = ?")
            params.append(q.action)
        if q.session_id:
            conditions.append("session_id = ?")
            params.append(q.session_id)
        if q.from_date:
            conditions.append("created_at >= ?")
            params.append(q.from_date)
        if q.to_date:
            conditions.append("created_at <= ?")
            params.append(q.to_date)

        where = " AND ".join(conditions) if conditions else "1=1"
        params.append(q.limit)

        rows = self.conn.execute(
            f"SELECT * FROM decision_events WHERE {where} ORDER BY created_at DESC LIMIT ?",
            params,
        ).fetchall()

        return [self._row_to_event(r) for r in rows]

    def get_event(self, event_id: str) -> DecisionEvent | None:
        """Fetch a single event by ID."""
        row = self.conn.execute(
            "SELECT * FROM decision_events WHERE event_id = ?", [event_id]
        ).fetchone()
        return self._row_to_event(row) if row else None

    def get_decision_trace(self, event_id: str) -> list[DecisionEvent]:
        """
        Return the full decision chain leading to an event.

        Walks parent_event_ids recursively to produce a trace from
        IC inputs → pod synthesis → assistant → super_trader decision.
        """
        trace = []
        visited = set()
        queue = [event_id]

        while queue:
            eid = queue.pop(0)
            if eid in visited:
                continue
            visited.add(eid)

            event = self.get_event(eid)
            if event:
                trace.append(event)
                queue.extend(event.parent_event_ids)

        return sorted(trace, key=lambda e: e.created_at)

    def get_session_summary(self, session_id: str) -> dict:
        """
        High-level summary of a trading session's decisions.

        Useful for compliance review and daily audit reports.
        """
        row = self.conn.execute(
            """
            SELECT
                COUNT(*) as total_events,
                COUNT(DISTINCT agent_name) as agents_used,
                COUNT(DISTINCT symbol) as symbols_analyzed,
                SUM(CASE WHEN action IN ('buy', 'sell') THEN 1 ELSE 0 END) as trade_decisions,
                SUM(CASE WHEN risk_approved = FALSE THEN 1 ELSE 0 END) as risk_rejections,
                MIN(created_at) as session_start,
                MAX(created_at) as session_end
            FROM decision_events
            WHERE session_id = ?
            """,
            [session_id],
        ).fetchone()

        if not row:
            return {}

        return {
            "session_id": session_id,
            "total_events": row[0],
            "agents_used": row[1],
            "symbols_analyzed": row[2],
            "trade_decisions": row[3],
            "risk_rejections": row[4],
            "session_start": row[5],
            "session_end": row[6],
        }

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _row_to_event(self, row) -> DecisionEvent:
        tool_calls_raw = json.loads(row[11]) if row[11] else []
        return DecisionEvent(
            event_id=row[0],
            session_id=row[1],
            event_type=row[2],
            agent_name=row[3],
            agent_role=row[4],
            symbol=row[5],
            action=row[6],
            confidence=row[7],
            input_context_hash=row[8] or "",
            market_data_snapshot=json.loads(row[9]) if row[9] else {},
            portfolio_snapshot=json.loads(row[10]) if row[10] else {},
            tool_calls=[ToolCall(**tc) for tc in tool_calls_raw],
            output_summary=row[12] or "",
            output_structured=json.loads(row[13]) if row[13] else {},
            risk_approved=row[14],
            risk_violations=json.loads(row[15]) if row[15] else [],
            created_at=row[16],
            decision_latency_ms=row[17],
            parent_event_ids=json.loads(row[18]) if row[18] else [],
        )

    @staticmethod
    def hash_context(context: dict) -> str:
        """SHA256 hash of input context for integrity verification."""
        serialized = json.dumps(context, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()


# -------------------------------------------------------------------------
# Convenience builder functions
# -------------------------------------------------------------------------


def make_trade_event(
    session_id: str,
    agent_name: str,
    agent_role: str,
    symbol: str,
    action: str,
    confidence: float,
    reasoning: str,
    output_structured: dict,
    parent_event_ids: list[str] | None = None,
    risk_approved: bool | None = None,
    risk_violations: list[str] | None = None,
    latency_ms: int | None = None,
) -> DecisionEvent:
    """Build a DecisionEvent for a trade decision."""
    return DecisionEvent(
        event_id=str(uuid.uuid4()),
        session_id=session_id,
        event_type="super_trader_decision",
        agent_name=agent_name,
        agent_role=agent_role,
        symbol=symbol,
        action=action,
        confidence=confidence,
        output_summary=reasoning[:500],
        output_structured=output_structured,
        parent_event_ids=parent_event_ids or [],
        risk_approved=risk_approved,
        risk_violations=risk_violations or [],
        decision_latency_ms=latency_ms,
    )


def make_analysis_event(
    session_id: str,
    agent_name: str,
    agent_role: str,
    symbol: str,
    summary: str,
    output_structured: dict,
    parent_event_ids: list[str] | None = None,
    confidence: float | None = None,
) -> DecisionEvent:
    """Build a DecisionEvent for an IC analysis or pod synthesis."""
    return DecisionEvent(
        event_id=str(uuid.uuid4()),
        session_id=session_id,
        event_type="ic_analysis" if agent_role == "ic" else "pod_synthesis",
        agent_name=agent_name,
        agent_role=agent_role,
        symbol=symbol,
        confidence=confidence,
        output_summary=summary[:500],
        output_structured=output_structured,
        parent_event_ids=parent_event_ids or [],
    )


def extract_indicator_attributions(
    market_data_snapshot: dict,
    action: str | None = None,
) -> list[IndicatorAttribution]:
    """
    Derive SHAP-style indicator attributions from a market data snapshot.

    Walks known indicator keys in the snapshot, applies domain rules to
    classify signal direction (bullish/bearish/neutral), and computes a
    weight (0..1) based on how far the value is from the neutral zone.

    The weight is signed relative to the action taken: if action="buy"
    and an indicator is bearish, its weight is low (it opposed the decision).

    Rules are intentionally simple and deterministic — no ML required.
    Purpose is explainability, not prediction.
    """
    attrs: list[IndicatorAttribution] = []

    def _attr(
        ind: str,
        value: float,
        signal: str,
        weight: float,
        threshold: float | None = None,
    ) -> None:
        attrs.append(
            IndicatorAttribution(
                indicator=ind,
                value=round(value, 4),
                signal=signal,
                weight=round(min(max(weight, 0.0), 1.0), 4),
                threshold=threshold,
            )
        )

    # RSI (14 or any period)
    for key in ("RSI_14", "RSI_9", "RSI_21", "RSI"):
        val = market_data_snapshot.get(key)
        if val is not None:
            if val > 70:
                _attr(key, val, "bearish", (val - 70) / 30, threshold=70.0)
            elif val < 30:
                _attr(key, val, "bullish", (30 - val) / 30, threshold=30.0)
            else:
                _attr(key, val, "neutral", 0.1)
            break

    # MACD histogram / signal
    for key in ("MACD_hist", "MACD_signal", "MACD"):
        val = market_data_snapshot.get(key)
        if val is not None:
            signal = "bullish" if val > 0 else ("bearish" if val < 0 else "neutral")
            weight = min(abs(val) / max(abs(val), 0.01), 1.0) * 0.8
            _attr(key, val, signal, weight)
            break

    # ADX (trend strength — not directional)
    adx = market_data_snapshot.get("ADX")
    if adx is not None:
        signal = "trending" if adx > 25 else "ranging"
        _attr(
            "ADX",
            adx,
            signal,
            min((adx - 20) / 30, 1.0) if adx > 20 else 0.1,
            threshold=25.0,
        )

    # Price vs EMA cross
    for ema_key in ("EMA_12", "EMA_20", "EMA_50"):
        ema = market_data_snapshot.get(ema_key)
        price = market_data_snapshot.get("close") or market_data_snapshot.get("price")
        if ema is not None and price is not None and price > 0:
            pct_diff = (price - ema) / price
            signal = "bullish" if pct_diff > 0 else "bearish"
            weight = min(abs(pct_diff) * 10, 1.0)
            _attr(f"price_vs_{ema_key}", pct_diff, signal, weight)
            break

    # Volume ratio (vs average)
    vol_ratio = market_data_snapshot.get("volume_ratio") or market_data_snapshot.get(
        "vol_ratio"
    )
    if vol_ratio is not None:
        if vol_ratio > 1.5:
            _attr(
                "volume_ratio",
                vol_ratio,
                "bullish",
                min((vol_ratio - 1) / 2, 1.0),
                threshold=1.5,
            )
        elif vol_ratio < 0.5:
            _attr("volume_ratio", vol_ratio, "bearish", (1 - vol_ratio), threshold=0.5)
        else:
            _attr("volume_ratio", vol_ratio, "neutral", 0.1)

    # ATR percentile (volatility — inform position sizing, not direction)
    atr_pct = market_data_snapshot.get("ATR_pct") or market_data_snapshot.get(
        "atr_percentile"
    )
    if atr_pct is not None:
        regime = (
            "high_vol"
            if atr_pct > 75
            else ("low_vol" if atr_pct < 25 else "normal_vol")
        )
        _attr("ATR_percentile", atr_pct, regime, atr_pct / 100, threshold=75.0)

    return attrs


# Singleton
_decision_log: DecisionLog | None = None


def get_decision_log(conn: PgConnection | None = None) -> DecisionLog:
    """Get the singleton DecisionLog instance.

    Args:
        conn: PostgreSQL connection to inject. If None, the singleton must have
              been previously initialized with a connection.
    """
    global _decision_log
    if _decision_log is None:
        _decision_log = DecisionLog(conn=conn or open_db_readonly())
    return _decision_log


# Read-only singleton — for processes that query the audit trail without writing.
_decision_log_ro: DecisionLog | None = None


def get_decision_log_readonly() -> DecisionLog:
    """
    Get a read-only DecisionLog singleton backed by a PostgreSQL connection.

    Use this in processes (FastAPI, scripts) that run alongside the MCP server
    and only need to QUERY the audit trail.  The returned instance uses the
    shared PostgreSQL pool, which supports unlimited concurrent readers.
    """
    global _decision_log_ro
    if _decision_log_ro is None:
        _decision_log_ro = DecisionLog(conn=open_db_readonly())
    return _decision_log_ro
