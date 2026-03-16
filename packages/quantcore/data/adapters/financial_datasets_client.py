"""
FinancialDatasetsClient — low-level HTTP client for financialdatasets.ai.

Shared by both the OHLCV adapter and the FundamentalsProvider.  Owns the
connection pool, rate limiter, and retry logic.

Auth: ``X-API-KEY`` header.
Rate limit: sliding-window token bucket (default 1000 req/min for Developer tier).
Retry: exponential backoff on 429 and 5xx responses.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any

import httpx
from loguru import logger

# Default timeout for individual API calls (seconds).
_DEFAULT_TIMEOUT = 30.0

# Retry configuration.
_MAX_RETRIES = 3
_RETRY_BACKOFF_BASE = 1.0  # seconds; doubles each retry


class _SlidingWindowRateLimiter:
    """Thread-safe sliding-window rate limiter.

    Tracks timestamps of recent requests in a deque.  When the window is
    full, ``acquire()`` sleeps until the oldest request falls out of the window.
    """

    def __init__(self, max_requests: int, window_seconds: float = 60.0) -> None:
        self._max = max_requests
        self._window = window_seconds
        self._timestamps: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            # Evict timestamps outside the window.
            while self._timestamps and self._timestamps[0] <= now - self._window:
                self._timestamps.popleft()
            if len(self._timestamps) >= self._max:
                sleep_until = self._timestamps[0] + self._window
                sleep_for = sleep_until - now
                if sleep_for > 0:
                    logger.debug(f"[FinancialDatasets] Rate limit reached, sleeping {sleep_for:.1f}s")
                    time.sleep(sleep_for)
            self._timestamps.append(time.monotonic())


class FinancialDatasetsClient:
    """Low-level HTTP client for the FinancialDatasets.ai REST API.

    Usage::

        with FinancialDatasetsClient(api_key="...") as client:
            prices = client.get_historical_prices("AAPL", "day", 1, "2024-01-01", "2024-12-31")
            income = client.get_income_statements("AAPL")

    All public methods return the parsed JSON response as a ``dict``.
    On HTTP or network errors (after retries) they return ``None``.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.financialdatasets.ai",
        rate_limit_rpm: int = 1000,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._rate_limiter = _SlidingWindowRateLimiter(
            max_requests=rate_limit_rpm, window_seconds=60.0
        )
        self._client = httpx.Client(
            headers={"X-API-KEY": api_key},
            timeout=timeout,
        )

    # ── Context manager ────────────────────────────────────────────────────

    def __enter__(self) -> FinancialDatasetsClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    # ── Price data ─────────────────────────────────────────────────────────

    def get_historical_prices(
        self,
        ticker: str,
        interval: str = "day",
        interval_multiplier: int = 1,
        start_date: str | None = None,
        end_date: str | None = None,
        limit: int = 5000,
    ) -> dict[str, Any] | None:
        """Fetch historical OHLCV price bars.

        Args:
            ticker: Symbol (e.g. "AAPL").
            interval: One of "second", "minute", "hour", "day", "week", "month".
            interval_multiplier: Multiplier for interval (e.g. 5 with "minute" = 5-min bars).
            start_date: YYYY-MM-DD start (inclusive).
            end_date: YYYY-MM-DD end (inclusive).
            limit: Max records per page (default 5000, API max 5000).

        Returns:
            ``{"prices": [...], "next_page_url": "..."|null}`` or None on failure.
        """
        params: dict[str, Any] = {
            "ticker": ticker,
            "interval": interval,
            "interval_multiplier": interval_multiplier,
            "limit": limit,
        }
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        return self._get("/prices/", params)

    def get_all_historical_prices(
        self,
        ticker: str,
        interval: str = "day",
        interval_multiplier: int = 1,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch all pages of historical prices, following ``next_page_url``.

        Returns a flat list of price dicts (may be empty on failure).
        """
        all_prices: list[dict[str, Any]] = []
        resp = self.get_historical_prices(
            ticker, interval, interval_multiplier, start_date, end_date
        )
        while resp:
            all_prices.extend(resp.get("prices") or [])
            next_url = resp.get("next_page_url")
            if not next_url:
                break
            resp = self._get_absolute(next_url)
        return all_prices

    def get_price_snapshot(self, ticker: str) -> dict[str, Any] | None:
        """Get latest price snapshot for a ticker."""
        return self._get("/prices/snapshot/", {"ticker": ticker})

    # ── Financial statements ───────────────────────────────────────────────

    def get_income_statements(
        self, ticker: str, period: str = "annual", limit: int = 10
    ) -> dict[str, Any] | None:
        return self._get(
            "/financials/income-statements/",
            {"ticker": ticker, "period": period, "limit": limit},
        )

    def get_balance_sheets(
        self, ticker: str, period: str = "annual", limit: int = 10
    ) -> dict[str, Any] | None:
        return self._get(
            "/financials/balance-sheets/",
            {"ticker": ticker, "period": period, "limit": limit},
        )

    def get_cash_flow_statements(
        self, ticker: str, period: str = "annual", limit: int = 10
    ) -> dict[str, Any] | None:
        return self._get(
            "/financials/cash-flow-statements/",
            {"ticker": ticker, "period": period, "limit": limit},
        )

    def get_all_financial_statements(
        self, ticker: str, period: str = "annual", limit: int = 10
    ) -> dict[str, Any] | None:
        return self._get(
            "/financials/all-financial-statements/",
            {"ticker": ticker, "period": period, "limit": limit},
        )

    def get_segmented_revenues(
        self, ticker: str, period: str = "annual", limit: int = 10
    ) -> dict[str, Any] | None:
        return self._get(
            "/financials/segmented-revenues/",
            {"ticker": ticker, "period": period, "limit": limit},
        )

    # ── Financial metrics ──────────────────────────────────────────────────

    def get_financial_metrics(
        self, ticker: str, period: str = "annual", limit: int = 10
    ) -> dict[str, Any] | None:
        return self._get(
            "/financial-metrics/historical/",
            {"ticker": ticker, "period": period, "limit": limit},
        )

    def get_financial_metrics_snapshot(self, ticker: str) -> dict[str, Any] | None:
        return self._get("/financial-metrics/snapshot/", {"ticker": ticker})

    # ── Earnings ───────────────────────────────────────────────────────────

    def get_earnings(self, ticker: str, limit: int = 20) -> dict[str, Any] | None:
        return self._get("/earnings/earnings/", {"ticker": ticker, "limit": limit})

    def get_earnings_press_releases(
        self, ticker: str, limit: int = 10
    ) -> dict[str, Any] | None:
        return self._get(
            "/earnings/press-releases/", {"ticker": ticker, "limit": limit}
        )

    # ── Insider trades ─────────────────────────────────────────────────────

    def get_insider_trades(
        self, ticker: str, limit: int = 100
    ) -> dict[str, Any] | None:
        return self._get(
            "/insider-trades/insider-trades/", {"ticker": ticker, "limit": limit}
        )

    # ── Institutional ownership ────────────────────────────────────────────

    def get_institutional_ownership_by_ticker(
        self, ticker: str, limit: int = 50
    ) -> dict[str, Any] | None:
        return self._get(
            "/institutional-ownership/ticker/", {"ticker": ticker, "limit": limit}
        )

    def get_institutional_ownership_by_investor(
        self, investor: str, limit: int = 50
    ) -> dict[str, Any] | None:
        return self._get(
            "/institutional-ownership/investor/",
            {"investor": investor, "limit": limit},
        )

    # ── Analyst estimates ──────────────────────────────────────────────────

    def get_analyst_estimates(self, ticker: str) -> dict[str, Any] | None:
        return self._get("/analyst-estimates/ticker/", {"ticker": ticker})

    # ── SEC filings ────────────────────────────────────────────────────────

    def get_sec_filings(
        self,
        ticker: str,
        limit: int = 20,
        filing_type: str | None = None,
    ) -> dict[str, Any] | None:
        params: dict[str, Any] = {"ticker": ticker, "limit": limit}
        if filing_type:
            params["filing_type"] = filing_type
        return self._get("/filings/", params)

    def get_sec_filing_items(
        self, accession_number: str, section: str | None = None
    ) -> dict[str, Any] | None:
        params: dict[str, Any] = {"accession_number": accession_number}
        if section:
            params["section"] = section
        return self._get("/filings/items/", params)

    # ── Company data ───────────────────────────────────────────────────────

    def get_company_facts(self, ticker: str) -> dict[str, Any] | None:
        return self._get("/company/facts/ticker/", {"ticker": ticker})

    def get_company_news(self, ticker: str, limit: int = 50) -> dict[str, Any] | None:
        return self._get("/news/", {"ticker": ticker, "limit": limit})

    # ── Macro ──────────────────────────────────────────────────────────────

    def get_interest_rates_historical(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict[str, Any] | None:
        params: dict[str, Any] = {}
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        return self._get("/macro/interest-rates/historical/", params)

    def get_interest_rates_snapshot(self) -> dict[str, Any] | None:
        return self._get("/macro/interest-rates/snapshot/", {})

    # ── Search / screener ──────────────────────────────────────────────────

    def search_financials(self, filters: dict[str, Any]) -> dict[str, Any] | None:
        """Search financial statements by line items (POST endpoint)."""
        return self._post("/financials/search-by-line-items/", filters)

    def stock_screener(self, filters: dict[str, Any]) -> dict[str, Any] | None:
        """Screen stocks by financial criteria (POST endpoint)."""
        return self._post("/financials/search-screener/", filters)

    # ── Crypto ─────────────────────────────────────────────────────────────

    def get_crypto_prices(
        self,
        ticker: str,
        interval: str = "day",
        interval_multiplier: int = 1,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict[str, Any] | None:
        params: dict[str, Any] = {
            "ticker": ticker,
            "interval": interval,
            "interval_multiplier": interval_multiplier,
        }
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        return self._get("/crypto/prices/", params)

    def get_crypto_price_snapshot(self, ticker: str) -> dict[str, Any] | None:
        return self._get("/crypto/prices/snapshot/", {"ticker": ticker})

    def get_crypto_tickers(self) -> dict[str, Any] | None:
        return self._get("/crypto/prices/tickers", {})

    # ── Internal HTTP methods ──────────────────────────────────────────────

    def _get(self, path: str, params: dict[str, Any]) -> dict[str, Any] | None:
        url = f"{self._base_url}{path}"
        return self._request("GET", url, params=params)

    def _get_absolute(self, url: str) -> dict[str, Any] | None:
        """GET with an absolute URL (for pagination next_page_url)."""
        return self._request("GET", url)

    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any] | None:
        url = f"{self._base_url}{path}"
        return self._request("POST", url, json_body=body)

    def _request(
        self,
        method: str,
        url: str,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Execute an HTTP request with rate limiting and retries."""
        last_exc: Exception | None = None

        for attempt in range(_MAX_RETRIES):
            self._rate_limiter.acquire()
            try:
                if method == "GET":
                    resp = self._client.get(url, params=params)
                else:
                    resp = self._client.post(url, json=json_body)

                if resp.status_code == 429:
                    retry_after = float(resp.headers.get("Retry-After", _RETRY_BACKOFF_BASE))
                    wait = max(retry_after, _RETRY_BACKOFF_BASE * (2**attempt))
                    logger.warning(
                        f"[FinancialDatasets] 429 rate limited on {url}, "
                        f"retry {attempt + 1}/{_MAX_RETRIES} after {wait:.1f}s"
                    )
                    time.sleep(wait)
                    continue

                if resp.status_code >= 500:
                    wait = _RETRY_BACKOFF_BASE * (2**attempt)
                    logger.warning(
                        f"[FinancialDatasets] {resp.status_code} on {url}, "
                        f"retry {attempt + 1}/{_MAX_RETRIES} after {wait:.1f}s"
                    )
                    time.sleep(wait)
                    continue

                resp.raise_for_status()
                return resp.json()

            except httpx.HTTPStatusError as exc:
                last_exc = exc
                logger.warning(f"[FinancialDatasets] HTTP {exc.response.status_code}: {url}")
                break  # Non-retryable 4xx
            except (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout) as exc:
                last_exc = exc
                wait = _RETRY_BACKOFF_BASE * (2**attempt)
                logger.warning(
                    f"[FinancialDatasets] {type(exc).__name__} on {url}, "
                    f"retry {attempt + 1}/{_MAX_RETRIES} after {wait:.1f}s"
                )
                time.sleep(wait)

        logger.error(f"[FinancialDatasets] All retries exhausted for {url}: {last_exc}")
        return None
