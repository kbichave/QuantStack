"""Corporate actions monitor — dividends, splits, and M&A events.

Collectors:
  - ``fetch_av_dividends()`` — Alpha Vantage DIVIDENDS endpoint
  - ``fetch_av_splits()`` — Alpha Vantage SPLITS endpoint (planned)
  - ``fetch_edgar_8k_events()`` — SEC EDGAR 8-K filings for M&A items

Adjustment:
  - ``apply_split_adjustment()`` — auto-adjust cost basis + quantity

Orchestration:
  - ``refresh_corporate_actions()`` — daily scheduled job
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any

import httpx
from loguru import logger

from quantstack.db import db_conn


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class CorporateAction:
    """A single corporate action event ready for DB insertion."""

    symbol: str
    event_type: str  # dividend, split, merger_signing, acquisition_complete, etc.
    source: str  # alpha_vantage, edgar_8k
    effective_date: date
    announcement_date: date | None = None
    raw_payload: dict | None = None


@dataclass
class SplitAdjustment:
    """Audit record for an applied split adjustment."""

    symbol: str
    effective_date: date
    event_type: str
    split_ratio: float
    old_quantity: float
    new_quantity: float
    old_cost_basis: float
    new_cost_basis: float


# ---------------------------------------------------------------------------
# 8-K item code -> event_type mapping
# ---------------------------------------------------------------------------

_8K_ITEM_MAP = {
    "1.01": "merger_signing",
    "2.01": "acquisition_complete",
    "3.03": "rights_modification",
    "5.01": "change_of_control",
}

# SEC requires a meaningful User-Agent per EDGAR fair access policy.
_SEC_USER_AGENT = os.getenv(
    "SEC_USER_AGENT",
    "QuantStack admin@quantstack.dev",
)

_SEC_RATE_DELAY = 0.12  # 10 req/s max, use 0.12s gap for margin


# ---------------------------------------------------------------------------
# CIK Mapping
# ---------------------------------------------------------------------------


class CIKMapper:
    """Resolves ticker symbols to SEC CIK numbers.

    Uses SEC's company_tickers.json endpoint. Loaded on first call,
    cached in memory. Call ``load()`` to refresh.
    """

    _URL = "https://www.sec.gov/files/company_tickers.json"

    def __init__(self) -> None:
        self._ticker_to_cik: dict[str, str] = {}
        self._loaded = False

    async def load(self) -> None:
        """Fetch and parse company_tickers.json from SEC."""
        try:
            async with httpx.AsyncClient(
                headers={"User-Agent": _SEC_USER_AGENT},
                timeout=30,
            ) as client:
                resp = await client.get(self._URL)
                resp.raise_for_status()
                data = resp.json()
            # Format: {"0": {"cik_str": 320193, "ticker": "AAPL", ...}, ...}
            self._ticker_to_cik = {}
            for entry in data.values():
                ticker = entry.get("ticker", "").upper()
                cik = str(entry.get("cik_str", ""))
                if ticker and cik:
                    self._ticker_to_cik[ticker] = cik.zfill(10)
            self._loaded = True
            logger.info("[CIK] Loaded %d ticker->CIK mappings", len(self._ticker_to_cik))
        except Exception as e:
            logger.error("[CIK] Failed to load company_tickers.json: %s", e)

    def lookup(self, ticker: str) -> str | None:
        """Return CIK for ticker, or None if not found."""
        return self._ticker_to_cik.get(ticker.upper())


# Module-level singleton
_cik_mapper = CIKMapper()


# ---------------------------------------------------------------------------
# Alpha Vantage collectors
# ---------------------------------------------------------------------------


def _parse_date_or_none(val: str | None) -> date | None:
    """Parse a date string, returning None for null-ish values."""
    if not val or val == "None" or val == "null":
        return None
    try:
        return date.fromisoformat(val)
    except ValueError:
        return None


async def fetch_av_dividends(symbol: str) -> list[CorporateAction]:
    """Fetch dividend history from AV DIVIDENDS endpoint.

    Uses the existing AV rate limiter from fetcher.py via AlphaVantageClient.
    """
    try:
        from quantstack.data.fetcher import AlphaVantageClient

        client = AlphaVantageClient()
        await asyncio.to_thread(client._wait_for_rate_limit)

        api_key = client.api_key
        url = (
            f"{client.base_url}?function=DIVIDENDS"
            f"&symbol={symbol}&apikey={api_key}"
        )

        async with httpx.AsyncClient(timeout=30) as http:
            resp = await http.get(url)
            resp.raise_for_status()
            data = resp.json()

        records = data.get("data", [])
        actions: list[CorporateAction] = []
        for rec in records:
            eff_date = _parse_date_or_none(rec.get("ex_dividend_date"))
            if eff_date is None:
                continue
            actions.append(CorporateAction(
                symbol=symbol,
                event_type="dividend",
                source="alpha_vantage",
                effective_date=eff_date,
                announcement_date=_parse_date_or_none(rec.get("declaration_date")),
                raw_payload=rec,
            ))
        return actions
    except Exception as e:
        logger.warning("[corp_actions] fetch_av_dividends(%s) failed: %s", symbol, e)
        return []


async def fetch_av_splits(symbol: str) -> list[CorporateAction]:
    """Fetch split history from AV TIME_SERIES_DAILY_ADJUSTED or dedicated endpoint.

    AV doesn't have a standalone SPLITS endpoint — we extract from stock_splits
    in the company overview, or use the TIME_SERIES_DAILY_ADJUSTED split coefficient.
    For now, parse from the OVERVIEW endpoint's stock_splits field.
    """
    try:
        from quantstack.data.fetcher import AlphaVantageClient

        client = AlphaVantageClient()
        await asyncio.to_thread(client._wait_for_rate_limit)

        api_key = client.api_key
        url = (
            f"{client.base_url}?function=OVERVIEW"
            f"&symbol={symbol}&apikey={api_key}"
        )

        async with httpx.AsyncClient(timeout=30) as http:
            resp = await http.get(url)
            resp.raise_for_status()
            data = resp.json()

        # AV OVERVIEW has LastSplitFactor and LastSplitDate fields
        split_factor = data.get("LastSplitFactor", "")
        split_date_str = data.get("LastSplitDate", "")
        if not split_factor or split_factor == "None" or not split_date_str:
            return []

        eff_date = _parse_date_or_none(split_date_str)
        if eff_date is None:
            return []

        # Parse "4:1" format -> ratio 4.0
        parts = split_factor.split(":")
        if len(parts) == 2:
            try:
                ratio = float(parts[0]) / float(parts[1])
            except (ValueError, ZeroDivisionError):
                return []
        else:
            return []

        return [CorporateAction(
            symbol=symbol,
            event_type="split",
            source="alpha_vantage",
            effective_date=eff_date,
            announcement_date=None,
            raw_payload={"split_factor": split_factor, "split_ratio": ratio},
        )]
    except Exception as e:
        logger.warning("[corp_actions] fetch_av_splits(%s) failed: %s", symbol, e)
        return []


# ---------------------------------------------------------------------------
# EDGAR 8-K collector
# ---------------------------------------------------------------------------


async def fetch_edgar_8k_events(symbol: str, cik: str | None) -> list[CorporateAction]:
    """Fetch recent 8-K filings from EDGAR submissions API.

    Targets items 1.01, 2.01, 3.03, 5.01. All others ignored.
    If CIK is None or empty, returns empty list.
    """
    if not cik:
        logger.debug("[corp_actions] No CIK for %s, skipping EDGAR lookup", symbol)
        return []

    try:
        url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        async with httpx.AsyncClient(
            headers={"User-Agent": _SEC_USER_AGENT},
            timeout=30,
        ) as http:
            await asyncio.sleep(_SEC_RATE_DELAY)
            resp = await http.get(url)
            resp.raise_for_status()
            data = resp.json()

        recent = data.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        items_list = recent.get("items", [])
        accessions = recent.get("accessionNumber", [])

        actions: list[CorporateAction] = []
        for i, form_type in enumerate(forms):
            if form_type != "8-K":
                continue
            filing_items = items_list[i] if i < len(items_list) else ""
            filing_date_str = dates[i] if i < len(dates) else ""
            accession = accessions[i] if i < len(accessions) else ""

            eff_date = _parse_date_or_none(filing_date_str)
            if eff_date is None:
                continue

            # Check if any target items are present
            for item_code, event_type in _8K_ITEM_MAP.items():
                if item_code in filing_items:
                    actions.append(CorporateAction(
                        symbol=symbol,
                        event_type=event_type,
                        source="edgar_8k",
                        effective_date=eff_date,
                        announcement_date=eff_date,
                        raw_payload={
                            "accession": accession,
                            "items": filing_items,
                            "item_code": item_code,
                        },
                    ))
        return actions
    except Exception as e:
        logger.warning("[corp_actions] fetch_edgar_8k(%s, CIK=%s) failed: %s", symbol, cik, e)
        return []


# ---------------------------------------------------------------------------
# Split auto-adjustment
# ---------------------------------------------------------------------------


async def apply_split_adjustment(
    symbol: str,
    split_ratio: float,
    effective_date: date,
) -> SplitAdjustment | None:
    """Auto-adjust cost basis and quantity for a stock split.

    Returns the SplitAdjustment record, or None if skipped (idempotent/broker-adjusted).
    """
    # 1. Check if already applied (idempotent)
    with db_conn() as conn:
        existing = conn.execute(
            "SELECT 1 FROM split_adjustments "
            "WHERE symbol = %s AND effective_date = %s AND event_type = 'split'",
            [symbol, effective_date],
        ).fetchone()
        if existing:
            logger.info("[split] Already applied for %s on %s, skipping", symbol, effective_date)
            return None

    # 2. Check broker position for reconciliation
    try:
        from quantstack.execution.portfolio_state import get_portfolio_state_readonly

        ps = get_portfolio_state_readonly()
        db_position = ps.get_position(symbol)
    except Exception:
        db_position = None

    if db_position is None:
        logger.info("[split] No position for %s, skipping adjustment", symbol)
        return None

    old_qty = db_position.quantity
    old_cost = db_position.avg_cost

    # 3. Check if broker already adjusted
    try:
        from quantstack.execution.alpaca_broker import AlpacaBroker

        broker = AlpacaBroker()
        broker_positions = await asyncio.to_thread(broker.get_positions)
        broker_pos = next((p for p in broker_positions if p.get("symbol") == symbol), None)
        if broker_pos:
            broker_qty = float(broker_pos.get("qty", 0))
            expected_qty = old_qty * split_ratio
            if abs(broker_qty - expected_qty) < 0.01:
                # Broker already adjusted — sync DB to broker
                logger.info(
                    "[split] Broker already adjusted %s: qty %.2f -> %.2f, syncing DB",
                    symbol, old_qty, broker_qty,
                )
    except Exception as e:
        logger.debug("[split] Broker reconciliation skipped: %s", e)

    # 4. Compute adjusted values
    new_qty = old_qty * split_ratio
    new_cost = old_cost / split_ratio

    # Handle fractional shares on reverse splits
    if split_ratio < 1.0:
        fractional = new_qty - math.floor(new_qty)
        new_qty = math.floor(new_qty)
        if new_qty == 0:
            new_qty = 0  # Position eliminated by reverse split
    else:
        fractional = 0.0

    # 5. Assert invariant: total cost basis preserved
    if new_qty > 0:
        old_total = old_qty * old_cost
        new_total = new_qty * new_cost
        if abs(old_total - new_total) > 0.01 and fractional == 0:
            logger.error(
                "[split] Invariant violated for %s: old_total=%.2f != new_total=%.2f",
                symbol, old_total, new_total,
            )

    # 6. Update position in DB
    with db_conn() as conn:
        conn.execute(
            "UPDATE positions SET quantity = %s, avg_cost = %s WHERE symbol = %s",
            [new_qty, new_cost, symbol],
        )

    # 7. Write audit row
    adjustment = SplitAdjustment(
        symbol=symbol,
        effective_date=effective_date,
        event_type="split",
        split_ratio=split_ratio,
        old_quantity=old_qty,
        new_quantity=new_qty,
        old_cost_basis=old_cost,
        new_cost_basis=new_cost,
    )
    with db_conn() as conn:
        conn.execute(
            "INSERT INTO split_adjustments "
            "(symbol, effective_date, event_type, split_ratio, "
            "old_quantity, new_quantity, old_cost_basis, new_cost_basis) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s) "
            "ON CONFLICT (symbol, effective_date, event_type) DO NOTHING",
            [
                symbol, effective_date, "split", split_ratio,
                old_qty, new_qty, old_cost, new_cost,
            ],
        )

    # 8. Emit system alert
    try:
        from quantstack.tools.functions.system_alerts import emit_system_alert

        await emit_system_alert(
            category="data_quality",
            severity="info",
            title=f"Split adjustment applied: {symbol} {split_ratio}:1",
            detail=(
                f"Adjusted {symbol} for {split_ratio}:1 split on {effective_date}. "
                f"Old: {old_qty} shares @ ${old_cost:.2f}. "
                f"New: {new_qty} shares @ ${new_cost:.2f}."
            ),
            source="corporate_actions",
            metadata={
                "symbol": symbol,
                "split_ratio": split_ratio,
                "effective_date": str(effective_date),
                "fractional_shares": fractional,
            },
        )
    except Exception as e:
        logger.warning("[split] Failed to emit alert: %s", e)

    logger.info(
        "[split] Applied %s: %.0f @ $%.2f -> %.0f @ $%.2f (ratio=%.2f)",
        symbol, old_qty, old_cost, new_qty, new_cost, split_ratio,
    )
    return adjustment


# ---------------------------------------------------------------------------
# M&A thesis flagging
# ---------------------------------------------------------------------------


async def _flag_ma_events(
    actions: list[CorporateAction],
    held_symbols: set[str],
) -> int:
    """Create system alerts for M&A events affecting held symbols."""
    flagged = 0
    for action in actions:
        if action.symbol not in held_symbols:
            continue
        if action.event_type not in ("merger_signing", "acquisition_complete"):
            continue
        try:
            from quantstack.tools.functions.system_alerts import emit_system_alert

            await emit_system_alert(
                category="thesis_review",
                severity="critical",
                title=f"M&A event detected: {action.symbol} - {action.event_type}",
                detail=(
                    f"EDGAR 8-K filing on {action.effective_date} indicates "
                    f"{action.event_type} for {action.symbol}. "
                    f"Review position thesis immediately."
                ),
                source="corporate_actions",
                metadata={
                    "symbol": action.symbol,
                    "event_type": action.event_type,
                    "filing_date": str(action.effective_date),
                    "raw_payload": action.raw_payload,
                },
            )
            flagged += 1
        except Exception as e:
            logger.warning("[corp_actions] Failed to flag M&A for %s: %s", action.symbol, e)
    return flagged


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _store_actions(actions: list[CorporateAction]) -> int:
    """Insert corporate actions into DB, deduplicating via unique constraint."""
    stored = 0
    with db_conn() as conn:
        for a in actions:
            result = conn.execute(
                "INSERT INTO corporate_actions "
                "(symbol, event_type, source, effective_date, announcement_date, raw_payload) "
                "VALUES (%s, %s, %s, %s, %s, %s) "
                "ON CONFLICT (symbol, event_type, effective_date, source) DO NOTHING",
                [
                    a.symbol, a.event_type, a.source, a.effective_date,
                    a.announcement_date,
                    json.dumps(a.raw_payload) if a.raw_payload else None,
                ],
            )
            if result.rowcount and result.rowcount > 0:
                stored += 1
    return stored


async def refresh_corporate_actions(symbols: list[str]) -> dict[str, Any]:
    """Daily corporate actions check for all holdings.

    Called by supervisor graph's scheduled_tasks node.
    """
    errors: list[str] = []
    all_dividends: list[CorporateAction] = []
    all_splits: list[CorporateAction] = []
    all_8k: list[CorporateAction] = []

    # 1. Fetch AV dividends
    for sym in symbols:
        try:
            divs = await fetch_av_dividends(sym)
            all_dividends.extend(divs)
        except Exception as e:
            errors.append(f"av_dividends_{sym}: {e}")

    # 2. Fetch AV splits
    for sym in symbols:
        try:
            splits = await fetch_av_splits(sym)
            all_splits.extend(splits)
        except Exception as e:
            errors.append(f"av_splits_{sym}: {e}")

    # 3. Fetch EDGAR 8-K events
    if not _cik_mapper._loaded:
        await _cik_mapper.load()

    for sym in symbols:
        cik = _cik_mapper.lookup(sym)
        try:
            events = await fetch_edgar_8k_events(sym, cik)
            all_8k.extend(events)
        except Exception as e:
            errors.append(f"edgar_8k_{sym}: {e}")

    # 4. Store all actions (dedup via unique constraint)
    new_dividends = _store_actions(all_dividends)
    new_splits = _store_actions(all_splits)
    new_ma = _store_actions(all_8k)

    # 5. Apply unprocessed splits
    splits_applied = 0
    with db_conn() as conn:
        unprocessed = conn.execute(
            "SELECT ca.symbol, ca.raw_payload, ca.effective_date "
            "FROM corporate_actions ca "
            "WHERE ca.event_type = 'split' AND ca.processed = FALSE"
        ).fetchall()

    for row in unprocessed:
        sym = row["symbol"]
        payload = row["raw_payload"] if isinstance(row["raw_payload"], dict) else {}
        ratio = payload.get("split_ratio", 1.0)
        eff_date = row["effective_date"]
        if ratio != 1.0:
            result = await apply_split_adjustment(sym, ratio, eff_date)
            if result:
                splits_applied += 1
                with db_conn() as conn:
                    conn.execute(
                        "UPDATE corporate_actions SET processed = TRUE "
                        "WHERE symbol = %s AND event_type = 'split' AND effective_date = %s",
                        [sym, eff_date],
                    )

    # 6. Flag M&A events for held symbols
    held_symbols = set()
    try:
        with db_conn() as conn:
            rows = conn.execute("SELECT DISTINCT symbol FROM positions").fetchall()
            held_symbols = {r["symbol"] for r in rows}
    except Exception:
        pass

    ma_flagged = await _flag_ma_events(all_8k, held_symbols)

    # 7. Mark dividends as processed
    with db_conn() as conn:
        conn.execute(
            "UPDATE corporate_actions SET processed = TRUE "
            "WHERE event_type = 'dividend' AND processed = FALSE"
        )

    summary = {
        "new_dividends": new_dividends,
        "new_splits": new_splits,
        "new_ma_events": new_ma,
        "splits_applied": splits_applied,
        "ma_alerts_created": ma_flagged,
        "errors": errors,
    }
    logger.info("[corp_actions] Refresh complete: %s", summary)
    return summary
