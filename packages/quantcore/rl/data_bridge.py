"""
KnowledgeStore RL Bridge — feeds real data to RL environments.

This module is the fix for the synthetic returns production blocker.

Problem: AlphaSelectionEnvironment, SizingEnvironment, and ExecutionEnvironment
all have optional real-data parameters (alpha_returns, signals, data) but they
are never populated from real data at construction time. Instead they fall back
to generating synthetic noise, which produces agents that learn nothing useful.

Solution: This bridge queries the KnowledgeStore DuckDB and formats historical
data into exactly what each environment expects.

Also provides bootstrap_from_alphavantage() for when the KnowledgeStore is
sparse (fresh system or first run). This is idempotent: it skips symbols/dates
already present.

Usage:
    from quantcore.rl.data_bridge import KnowledgeStoreRLBridge

    bridge = KnowledgeStoreRLBridge.from_knowledge_store(store)

    # Fix for meta environment (replaces synthetic returns)
    alpha_returns = bridge.get_alpha_return_history(alpha_names, lookback_days=252)

    # Fix for sizing environment
    signals = bridge.get_signal_history(lookback_days=90)

    # Fix for execution environment
    ohlcv = bridge.get_ohlcv_for_execution("SPY", lookback_days=60)
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta

import pandas as pd
from loguru import logger
from quant_pod.knowledge.store import KnowledgeStore


class KnowledgeStoreRLBridge:
    """
    Extracts and formats KnowledgeStore data for RL environment initialization.

    All methods return data in the format expected by the corresponding
    RL environment constructors. None is never returned without an explicit
    warning — callers can choose to disable the agent when data is insufficient.
    """

    # AlphaVantage free tier: 5 requests per minute
    _ALPHAVANTAGE_RATE_LIMIT_CALLS = 5
    _ALPHAVANTAGE_RATE_LIMIT_WINDOW_S = 60.0

    def __init__(self, store: KnowledgeStore):
        self.store = store
        self._av_call_timestamps: list[float] = []

    @classmethod
    def from_knowledge_store(cls, store: KnowledgeStore) -> KnowledgeStoreRLBridge:
        """Standard factory. Prefer this over direct __init__."""
        return cls(store)

    # -------------------------------------------------------------------------
    # Alpha selection data — for AlphaSelectionEnvironment
    # -------------------------------------------------------------------------

    def get_alpha_return_history(
        self,
        alpha_names: list[str],
        lookback_days: int = 252,
    ) -> dict[str, pd.Series]:
        """
        Build per-alpha return series from closed trade journal entries.

        Maps alpha_names to signal_type (the signal field in trade_journal).
        Returns a dict of {alpha_name: pd.Series of daily pnl_pct}.

        If an alpha has fewer than 5 entries its series is still returned —
        callers that need a minimum sample should check series length.

        Returns {} (empty) if the trade_journal table has no closed trades at all.
        """
        cutoff = datetime.utcnow() - timedelta(days=lookback_days)

        try:
            rows = self.store.conn.execute(
                """
                SELECT
                    COALESCE(signal_type, direction) AS alpha_tag,
                    DATE_TRUNC('day', created_at) AS trade_date,
                    AVG(pnl_pct) AS daily_pnl_pct
                FROM trade_journal
                WHERE status = 'CLOSED'
                  AND created_at >= ?
                  AND pnl_pct IS NOT NULL
                GROUP BY 1, 2
                ORDER BY 1, 2
                """,
                [cutoff],
            ).fetchall()
        except Exception as exc:
            logger.warning(f"[DataBridge] get_alpha_return_history failed: {exc}")
            return {}

        if not rows:
            logger.debug("[DataBridge] No closed trades found in trade_journal.")
            return {}

        # Build a DataFrame for easy pivoting
        df = pd.DataFrame(rows, columns=["alpha_tag", "trade_date", "daily_pnl_pct"])
        df["trade_date"] = pd.to_datetime(df["trade_date"])

        result: dict[str, pd.Series] = {}
        for alpha_name in alpha_names:
            # Match on signal_type prefix — e.g. "trend_momentum_ic" matches "trend"
            mask = df["alpha_tag"].str.contains(
                alpha_name.replace("_", ".*"), case=False, regex=True, na=False
            )
            subset = df[mask].sort_values("trade_date").set_index("trade_date")
            if not subset.empty:
                result[alpha_name] = subset["daily_pnl_pct"].rename(alpha_name)
            else:
                result[alpha_name] = pd.Series(dtype=float, name=alpha_name)

        n_valid = sum(1 for s in result.values() if len(s) >= 5)
        logger.debug(
            f"[DataBridge] Alpha return history: {n_valid}/{len(alpha_names)} alphas "
            f"have >= 5 observations."
        )
        return result

    # -------------------------------------------------------------------------
    # Sizing agent data — for SizingEnvironment
    # -------------------------------------------------------------------------

    def get_signal_history(
        self,
        lookback_days: int = 90,
    ) -> list[_TradingSignalLike]:  # noqa: F821
        """
        Fetch TradingSignal records for SizingEnvironment initialization.

        Returns a list of _TradingSignalLike objects with the same interface
        as quantcore.rl.sizing.environment.TradingSignal.

        These replace the synthetic signal generation fallback in SizingEnvironment.
        Returns [] if no signals exist (caller should fall back to synthetic or disable).
        """
        from quantcore.rl.sizing.environment import TradingSignal

        cutoff = datetime.utcnow() - timedelta(days=lookback_days)

        try:
            rows = self.store.conn.execute(
                """
                SELECT direction, confidence, signal_type
                FROM trading_signals
                WHERE created_at >= ?
                  AND is_active = FALSE  -- historical, no longer active
                ORDER BY created_at ASC
                """,
                [cutoff],
            ).fetchall()
        except Exception as exc:
            logger.warning(f"[DataBridge] get_signal_history failed: {exc}")
            return []

        if not rows:
            logger.debug("[DataBridge] No historical trading signals found.")
            return []

        signals = []
        for direction, confidence, signal_type in rows:
            # Map to RL TradingSignal format
            rl_direction = direction if direction in ("LONG", "SHORT", "NEUTRAL") else "NEUTRAL"
            signals.append(
                TradingSignal(
                    direction=rl_direction,
                    confidence=float(confidence or 0.5),
                    expected_return=0.0,  # not stored in trading_signals
                    alpha_name=str(signal_type or "unknown"),
                )
            )

        logger.debug(f"[DataBridge] Loaded {len(signals)} historical signals.")
        return signals

    # -------------------------------------------------------------------------
    # Execution agent data — for ExecutionEnvironment
    # -------------------------------------------------------------------------

    def get_ohlcv_for_execution(
        self,
        symbol: str,
        lookback_days: int = 60,
    ) -> pd.DataFrame:
        """
        Build OHLCV DataFrame from market_observations for ExecutionEnvironment.

        market_observations stores current_price (close) and volume.
        high and low are approximated from the stored price when unavailable —
        execution RL only strictly requires close and volume.

        Returns empty DataFrame if no observations exist (caller should bootstrap).
        """
        cutoff = datetime.utcnow() - timedelta(days=lookback_days)

        try:
            rows = self.store.conn.execute(
                """
                SELECT
                    timestamp,
                    current_price AS close,
                    COALESCE(volume, 0) AS volume
                FROM market_observations
                WHERE symbol = ?
                  AND timestamp >= ?
                  AND current_price IS NOT NULL
                ORDER BY timestamp ASC
                """,
                [symbol, cutoff],
            ).fetchall()
        except Exception as exc:
            logger.warning(f"[DataBridge] get_ohlcv_for_execution failed: {exc}")
            return pd.DataFrame()

        if not rows:
            logger.debug(f"[DataBridge] No market observations for {symbol}.")
            return pd.DataFrame()

        df = pd.DataFrame(rows, columns=["timestamp", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")

        # Approximate high/low when unavailable
        df["high"] = df["close"] * 1.002  # +20 bps
        df["low"] = df["close"] * 0.998  # -20 bps

        logger.debug(f"[DataBridge] Loaded {len(df)} bars for {symbol} execution env.")
        return df

    # -------------------------------------------------------------------------
    # Realized returns for sizing context
    # -------------------------------------------------------------------------

    def get_realized_returns(
        self,
        lookback_days: int = 252,
    ) -> list[float]:
        """
        Fetch daily realized P&L % from closed trades as a return series.

        Used by PostTradeRLAdapter to compute rolling Sharpe and drawdown
        for the sizing agent's state at inference time.
        """
        cutoff = datetime.utcnow() - timedelta(days=lookback_days)

        try:
            rows = self.store.conn.execute(
                """
                SELECT pnl_pct
                FROM trade_journal
                WHERE status = 'CLOSED'
                  AND created_at >= ?
                  AND pnl_pct IS NOT NULL
                ORDER BY created_at ASC
                """,
                [cutoff],
            ).fetchall()
        except Exception as exc:
            logger.warning(f"[DataBridge] get_realized_returns failed: {exc}")
            return []

        return [float(r[0]) for r in rows]

    # -------------------------------------------------------------------------
    # AlphaVantage bootstrap — for fresh systems with empty KnowledgeStore
    # -------------------------------------------------------------------------

    def bootstrap_from_alphavantage(
        self,
        symbols: list[str],
        start_date: str = "2022-01-01",
        api_key: str | None = None,
        function: str = "TIME_SERIES_DAILY",
    ) -> None:
        """
        Fetch historical OHLCV from AlphaVantage and store into market_observations.

        Idempotent: checks existing rows before fetching to avoid duplicate inserts.
        Rate-limited to 5 calls/min (AlphaVantage free tier).

        Args:
            symbols: Symbols to fetch (e.g. ["SPY", "QQQ", "AAPL"])
            start_date: ISO date string, data fetched from this date forward
            api_key: AlphaVantage API key. Falls back to ALPHA_VANTAGE_API_KEY env var.
            function: AlphaVantage function (TIME_SERIES_DAILY or TIME_SERIES_WEEKLY)
        """
        import os

        import requests

        key = api_key or os.getenv("ALPHA_VANTAGE_API_KEY", "demo")
        if key == "demo":
            logger.warning(
                "[DataBridge] Using AlphaVantage 'demo' key — rate limits apply. "
                "Set ALPHA_VANTAGE_API_KEY env var for production use."
            )

        base_url = "https://www.alphavantage.co/query"
        start_dt = pd.Timestamp(start_date)

        for symbol in symbols:
            # Check how many observations already exist for this symbol
            try:
                existing = self.store.conn.execute(
                    "SELECT COUNT(*) FROM market_observations WHERE symbol = ?",
                    [symbol],
                ).fetchone()[0]
            except Exception:
                existing = 0

            if existing > 50:
                logger.debug(
                    f"[DataBridge] {symbol} already has {existing} observations — skipping."
                )
                continue

            # Rate limit enforcement
            self._enforce_rate_limit()

            logger.info(f"[DataBridge] Bootstrapping {symbol} from AlphaVantage...")

            try:
                resp = requests.get(
                    base_url,
                    params={
                        "function": function,
                        "symbol": symbol,
                        "outputsize": "full",
                        "apikey": key,
                    },
                    timeout=30,
                )
                resp.raise_for_status()
                data = resp.json()

                # AlphaVantage rate limit signal
                if "Note" in data or "Information" in data:
                    logger.warning(
                        f"[DataBridge] AlphaVantage rate limit hit for {symbol}. Sleeping 60s..."
                    )
                    time.sleep(61.0)
                    continue

                key_map = {
                    "TIME_SERIES_DAILY": "Time Series (Daily)",
                    "TIME_SERIES_WEEKLY": "Weekly Time Series",
                }
                ts_key = key_map.get(function, "Time Series (Daily)")
                time_series = data.get(ts_key, {})

                if not time_series:
                    logger.warning(f"[DataBridge] No time series data for {symbol}.")
                    continue

                inserted = 0
                for date_str, ohlcv in time_series.items():
                    dt = pd.Timestamp(date_str)
                    if dt < start_dt:
                        continue

                    close = float(ohlcv.get("4. close", ohlcv.get("close", 0)))
                    volume = int(float(ohlcv.get("5. volume", ohlcv.get("volume", 0))))

                    if close <= 0:
                        continue

                    try:
                        self.store.conn.execute(
                            """
                            INSERT INTO market_observations
                                (timestamp, symbol, observation_type,
                                 current_price, volume, source_agent)
                            VALUES (?, ?, 'OHLCV_BOOTSTRAP', ?, ?, 'bootstrap')
                            """,
                            [dt, symbol, close, volume],
                        )
                        inserted += 1
                    except Exception:
                        # Skip duplicates gracefully
                        pass

                logger.info(f"[DataBridge] Inserted {inserted} observations for {symbol}.")

            except requests.RequestException as exc:
                logger.error(f"[DataBridge] Failed to fetch {symbol}: {exc}")

    def _enforce_rate_limit(self) -> None:
        """Enforce AlphaVantage free-tier rate limit (5 calls / 60s)."""
        now = time.time()
        # Remove timestamps older than 60 seconds
        self._av_call_timestamps = [
            t for t in self._av_call_timestamps if now - t < self._ALPHAVANTAGE_RATE_LIMIT_WINDOW_S
        ]
        if len(self._av_call_timestamps) >= self._ALPHAVANTAGE_RATE_LIMIT_CALLS:
            wait = self._ALPHAVANTAGE_RATE_LIMIT_WINDOW_S - (now - self._av_call_timestamps[0])
            if wait > 0:
                logger.debug(f"[DataBridge] Rate limit — sleeping {wait:.1f}s...")
                time.sleep(wait + 0.5)
        self._av_call_timestamps.append(time.time())

    # -------------------------------------------------------------------------
    # Sufficiency checks used by orchestrator / rl_tools
    # -------------------------------------------------------------------------

    def has_sufficient_alpha_history(
        self,
        alpha_names: list[str],
        min_observations: int = 20,
    ) -> bool:
        """
        Returns True if enough real alpha return data exists to train the meta agent.

        Used as a gate in RLOrchestrator._init_agents() to decide whether to
        enable meta RL or disable it pending more data.
        """
        history = self.get_alpha_return_history(alpha_names, lookback_days=365)
        valid = sum(1 for s in history.values() if len(s) >= min_observations)
        return valid >= max(1, len(alpha_names) // 2)  # at least half must have history

    def has_sufficient_signal_history(
        self,
        min_signals: int = 30,
    ) -> bool:
        """Returns True if enough historical signals exist to train the sizing agent."""
        signals = self.get_signal_history(lookback_days=180)
        return len(signals) >= min_signals
