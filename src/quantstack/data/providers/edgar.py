"""SEC EDGAR data provider via edgartools.

Capabilities:
- fetch_insider_transactions: Form 4 filings -> insider_trades schema
- fetch_sec_filings: Filing metadata (10-K, 10-Q, 8-K)
- fetch_fundamentals: XBRL Company Facts (future)
- fetch_institutional_holdings: 13F filings (future)

All other DataProvider methods raise NotImplementedError (EDGAR does not
provide OHLCV, options, news, or macro data).

Rate limit: 10 req/sec. Enforced via minimum 0.1s gap between requests.
"""

from __future__ import annotations

import os
import time

import pandas as pd
from edgar import Company, set_identity
from loguru import logger

from quantstack.data.providers.base import ConfigurationError, DataProvider

# Transaction code mapping: EDGAR -> QuantStack
_TX_TYPE_MAP = {"A": "buy", "D": "sell"}


class EDGARProvider(DataProvider):
    """SEC EDGAR data provider.

    Requires EDGAR_USER_AGENT environment variable (SEC policy: company + email).
    """

    def __init__(self) -> None:
        user_agent = os.environ.get("EDGAR_USER_AGENT", "").strip()
        if not user_agent:
            raise ConfigurationError(
                "EDGAR_USER_AGENT environment variable is required "
                "(format: 'CompanyName admin@company.com')"
            )
        set_identity(user_agent)
        self._cik_cache: dict[str, str | None] = {}
        self._last_request_at: float = 0.0

    def name(self) -> str:
        return "edgar"

    def _throttle(self) -> None:
        """Enforce 10 req/sec rate limit (0.1s minimum gap)."""
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < 0.1:
            time.sleep(0.1 - elapsed)
        self._last_request_at = time.monotonic()

    def _get_company(self, symbol: str) -> Company:
        """Get Company object for ticker, with throttling."""
        self._throttle()
        return Company(symbol)

    def fetch_insider_transactions(self, symbol: str) -> pd.DataFrame | None:
        """Fetch Form 4 insider transactions from EDGAR.

        Returns DataFrame with columns:
            ticker, transaction_date, owner_name, transaction_type, shares, price_per_share
        """
        try:
            company = self._get_company(symbol)
            filings = company.get_filings()
            form4s = filings.filter(form="4")

            if not form4s:
                return None

            rows = []
            for filing in form4s:
                try:
                    self._throttle()
                    parsed = filing.obj()
                    if not hasattr(parsed, "transactions") or not parsed.transactions:
                        continue
                    for tx in parsed.transactions:
                        tx_type = _TX_TYPE_MAP.get(
                            getattr(tx, "acquisition_disposition", ""), "unknown"
                        )
                        shares = abs(float(getattr(tx, "transaction_shares", 0) or 0))
                        price = float(getattr(tx, "transaction_price_per_share", 0) or 0)
                        rows.append({
                            "ticker": symbol,
                            "transaction_date": getattr(tx, "transaction_date", None),
                            "owner_name": getattr(tx, "owner_name", "Unknown"),
                            "transaction_type": tx_type,
                            "shares": shares,
                            "price_per_share": price,
                        })
                except Exception as exc:
                    logger.debug("[EDGAR] Failed to parse Form 4 for %s: %s", symbol, exc)
                    continue

            if not rows:
                return None

            df = pd.DataFrame(rows)
            df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="coerce")
            logger.debug("[EDGAR] Insider transactions for %s: %d rows", symbol, len(df))
            return df

        except Exception as exc:
            logger.warning("[EDGAR] fetch_insider_transactions(%s) failed: %s", symbol, exc)
            return None

    def fetch_sec_filings(
        self, symbol: str, form_types: list[str] | None = None
    ) -> pd.DataFrame | None:
        """Fetch SEC filing metadata.

        Returns DataFrame with columns:
            accession_number, symbol, form_type, filing_date, period_of_report, primary_doc_url
        """
        try:
            company = self._get_company(symbol)
            filings = company.get_filings()

            if form_types:
                all_filtered = []
                for ft in form_types:
                    all_filtered.extend(filings.filter(form=ft))
                filings_list = all_filtered
            else:
                filings_list = list(filings)

            if not filings_list:
                return None

            rows = []
            for f in filings_list:
                rows.append({
                    "accession_number": getattr(f, "accession_no", ""),
                    "symbol": symbol,
                    "form_type": getattr(f, "form", ""),
                    "filing_date": getattr(f, "filing_date", None),
                    "period_of_report": getattr(f, "report_date", None),
                    "primary_doc_url": getattr(f, "primary_doc_url", ""),
                })

            df = pd.DataFrame(rows)
            logger.debug("[EDGAR] SEC filings for %s: %d rows", symbol, len(df))
            return df

        except Exception as exc:
            logger.warning("[EDGAR] fetch_sec_filings(%s) failed: %s", symbol, exc)
            return None

    def fetch_fundamentals(self, symbol: str) -> dict | None:
        """Fetch XBRL financial statements from EDGAR.

        Returns dict with key financial metrics, or None if unavailable.
        """
        try:
            company = self._get_company(symbol)
            self._throttle()
            facts = company.get_facts()

            if facts is None:
                return None

            # Extract key US-GAAP concepts
            result = {"symbol": symbol}
            gaap_concepts = {
                "revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax"],
                "net_income": ["NetIncomeLoss"],
                "total_assets": ["Assets"],
                "total_liabilities": ["Liabilities"],
                "stockholders_equity": ["StockholdersEquity"],
                "eps_basic": ["EarningsPerShareBasic"],
            }

            for key, concept_names in gaap_concepts.items():
                for concept_name in concept_names:
                    try:
                        concept = facts.get(f"us-gaap:{concept_name}")
                        if concept is not None:
                            # Get the most recent value
                            df = concept.to_dataframe() if hasattr(concept, "to_dataframe") else None
                            if df is not None and not df.empty:
                                result[key] = float(df.iloc[-1].get("val", 0))
                                break
                    except Exception:
                        continue

            if len(result) <= 1:  # Only has 'symbol'
                return None

            logger.debug("[EDGAR] Fundamentals for %s: %d fields", symbol, len(result) - 1)
            return result

        except Exception as exc:
            logger.warning("[EDGAR] fetch_fundamentals(%s) failed: %s", symbol, exc)
            return None

    def fetch_institutional_holdings(self, symbol: str) -> pd.DataFrame | None:
        """Fetch 13F institutional holdings from EDGAR.

        Returns DataFrame with institutional ownership data, or None.
        """
        try:
            company = self._get_company(symbol)
            filings = company.get_filings()
            thirteenfs = filings.filter(form="13-F")

            if not thirteenfs:
                return None

            # Parse the most recent 13F
            self._throttle()
            latest = thirteenfs[0]
            parsed = latest.obj()

            if not hasattr(parsed, "infotable") or parsed.infotable is None:
                return None

            df = parsed.infotable
            if hasattr(df, "to_dataframe"):
                df = df.to_dataframe()

            if df is None or (hasattr(df, "empty") and df.empty):
                return None

            logger.debug("[EDGAR] 13F holdings for %s: %d rows", symbol, len(df))
            return df

        except Exception as exc:
            logger.warning("[EDGAR] fetch_institutional_holdings(%s) failed: %s", symbol, exc)
            return None
