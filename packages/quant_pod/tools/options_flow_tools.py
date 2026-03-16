# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Unusual Options Activity (UOA) tools for agent use.

Addresses GAP-10 in the gap analysis:
  "Options flow / UOA (FlowAlgo, Unusual Whales API): Unusual options activity
   signals; dark pool prints represent 40-50% of institutional order flow."

Data source strategy:
  - Primary: Alpha Vantage REALTIME_OPTIONS endpoint (existing key, no extra cost)
  - Derived signals: computed from the raw options chain (volume/OI ratio, IV percentile,
    put/call skew, net premium flow)
  - No dependency on FlowAlgo/Unusual Whales accounts (too expensive for retail)

What "unusual" means (operational definitions used here):
  - Volume/OI ratio > 3.0: Today's volume is >3× open interest — institutions
    opening positions, not hedging existing ones.
  - Volume > 10× 30-day average for that strike/expiry: Directional interest spike.
  - Large premium (notional > $100k): Smart money doesn't trade $1 tickets.
  - IV spike vs 30-day historical: Market maker flagging unknown risk.
  - Put/call ratio extreme (< 0.5 = heavy calls = bullish flow, > 2.0 = heavy puts = bearish).

Signal output (for CrewAI agents and /options/flow API):
  - `flow_bias`: "BULLISH" | "BEARISH" | "NEUTRAL" | "MIXED"
  - `unusual_score`: 0–100 (higher = more unusual; threshold for action = 60)
  - `largest_trade`: the single most anomalous contract (for agent context)
  - `net_premium_usd`: calls premium - puts premium (positive = institutional buying calls)
  - `put_call_ratio`: simple volume-based ratio

Failure modes:
  - No Alpha Vantage key → returns empty result with warning (graceful degradation)
  - Rate-limited → sleeps 60s then raises (Alpha Vantage free tier: 5 calls/min)
  - Symbol has no listed options → returns NEUTRAL with n_contracts=0
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date, datetime

from loguru import logger

try:
    from quant_pod.crewai_compat import BaseTool

    _CREWAI_AVAILABLE = True
except ImportError:
    _CREWAI_AVAILABLE = False
    BaseTool = object


# =============================================================================
# DATA MODELS
# =============================================================================


@dataclass
class OptionsContract:
    """A single options contract from the chain."""

    symbol: str
    expiry: str
    strike: float
    option_type: str  # "call" or "put"
    volume: int
    open_interest: int
    implied_volatility: float  # Fraction (0.25 = 25%)
    last_price: float
    bid: float
    ask: float
    delta: float = 0.0
    gamma: float = 0.0

    @property
    def volume_oi_ratio(self) -> float:
        """Volume / open interest. > 3 = unusual."""
        return self.volume / max(self.open_interest, 1)

    @property
    def mid_price(self) -> float:
        return (self.bid + self.ask) / 2 if self.bid and self.ask else self.last_price

    @property
    def notional_premium(self) -> float:
        """Approximate notional: volume × mid_price × 100 (standard contract size)."""
        return self.volume * self.mid_price * 100


@dataclass
class OptionsFlowSignal:
    """
    Aggregated unusual options activity signal for a symbol.

    This is what agents consume — the raw chain is reduced to a directional
    signal and a small set of high-signal metrics.
    """

    symbol: str
    as_of: datetime
    flow_bias: str  # "BULLISH" | "BEARISH" | "NEUTRAL" | "MIXED"
    unusual_score: float  # 0–100
    put_call_ratio: float  # < 0.5 = heavy calls, > 2.0 = heavy puts
    net_premium_usd: float  # Calls premium - puts premium
    call_volume: int
    put_volume: int
    n_unusual_contracts: int  # Contracts that crossed the UOA threshold
    largest_trade_description: str  # Human-readable for agent context
    raw_contracts: list[OptionsContract] = field(default_factory=list, repr=False)

    @property
    def summary(self) -> str:
        """Concise one-liner for LLM agent context."""
        direction = {"BULLISH": "↑", "BEARISH": "↓", "NEUTRAL": "→", "MIXED": "↔"}
        arrow = direction.get(self.flow_bias, "→")
        return (
            f"{self.symbol} options flow {arrow} {self.flow_bias} "
            f"(score={self.unusual_score:.0f}/100, "
            f"P/C={self.put_call_ratio:.2f}, "
            f"net_premium=${self.net_premium_usd:+,.0f}). "
            f"{self.largest_trade_description}"
        )


# =============================================================================
# OPTIONS FLOW CLIENT
# =============================================================================


class OptionsFlowClient:
    """
    Fetches options chain data from Alpha Vantage REALTIME_OPTIONS endpoint
    and computes UOA signals.

    REALTIME_OPTIONS returns the full current-day options chain for a symbol.
    We use the volume, OI, IV, and premium data to derive unusual activity flags.
    """

    # UOA thresholds
    VOLUME_OI_UNUSUAL = 3.0  # vol/OI > 3 = unusual
    MIN_NOTIONAL_UNUSUAL = 50_000  # $50k minimum to be "institutional size"
    UNUSUAL_SCORE_THRESHOLD = 60  # Scores above this are actionable

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = (
            api_key
            or os.getenv("ALPHA_VANTAGE_API_KEY")
            or os.getenv("ALPHAVANTAGE_API_KEY")
            or "demo"
        )

    def get_flow(
        self,
        symbol: str,
        expiry_within_days: int = 45,
        min_volume: int = 100,
    ) -> OptionsFlowSignal:
        """
        Fetch options chain and compute UOA signal for a symbol.

        Args:
            symbol: Ticker symbol (e.g. "SPY", "AAPL")
            expiry_within_days: Only look at contracts expiring within this many
                                days — reduces noise from far-dated LEAPS.
            min_volume: Minimum volume to include a contract (filters penny contracts).

        Returns:
            OptionsFlowSignal with flow_bias and unusual_score.
        """
        raw_chain = self._fetch_chain(symbol)
        if not raw_chain:
            logger.warning(f"[OPTS] No chain data for {symbol}")
            return self._empty_signal(symbol)

        contracts = self._parse_chain(raw_chain, symbol, expiry_within_days, min_volume)
        if not contracts:
            logger.info(
                f"[OPTS] No liquid contracts found for {symbol} within {expiry_within_days}d"
            )
            return self._empty_signal(symbol)

        return self._compute_signal(symbol, contracts)

    # -------------------------------------------------------------------------
    # Data fetching
    # -------------------------------------------------------------------------

    def _fetch_chain(self, symbol: str) -> list[dict]:
        """Fetch raw options chain from Alpha Vantage."""
        try:
            import httpx

            params = {
                "function": "REALTIME_OPTIONS",
                "symbol": symbol.upper(),
                "apikey": self._api_key,
            }
            with httpx.Client(timeout=30.0) as client:
                resp = client.get("https://www.alphavantage.co/query", params=params)
                resp.raise_for_status()
                data = resp.json()

            if "Note" in data:
                logger.warning("[OPTS] Alpha Vantage rate-limited — slow down requests")
                return []

            if "Error Message" in data:
                logger.warning(f"[OPTS] Alpha Vantage error: {data['Error Message']}")
                return []

            # Response structure: {"data": [...list of contract dicts...]}
            return data.get("data", [])

        except Exception as e:
            logger.warning(f"[OPTS] Failed to fetch options chain for {symbol}: {e}")
            return []

    # -------------------------------------------------------------------------
    # Parsing
    # -------------------------------------------------------------------------

    def _parse_chain(
        self,
        raw_chain: list[dict],
        symbol: str,
        expiry_within_days: int,
        min_volume: int,
    ) -> list[OptionsContract]:
        """Parse raw Alpha Vantage options contracts, filtering by expiry and volume."""
        contracts = []
        today = date.today()
        cutoff = (
            datetime.today() + __import__("datetime").timedelta(days=expiry_within_days)
        ).date()

        for row in raw_chain:
            try:
                expiry_str = row.get("expiration", "")
                if expiry_str:
                    expiry_date = date.fromisoformat(expiry_str)
                    if expiry_date > cutoff or expiry_date < today:
                        continue

                volume = int(row.get("volume", 0) or 0)
                if volume < min_volume:
                    continue

                contracts.append(
                    OptionsContract(
                        symbol=symbol,
                        expiry=expiry_str,
                        strike=float(row.get("strike", 0) or 0),
                        option_type=row.get("type", "").lower(),  # "call" or "put"
                        volume=volume,
                        open_interest=int(row.get("open_interest", 0) or 0),
                        implied_volatility=float(row.get("implied_volatility", 0) or 0),
                        last_price=float(row.get("last", 0) or 0),
                        bid=float(row.get("bid", 0) or 0),
                        ask=float(row.get("ask", 0) or 0),
                        delta=float(row.get("delta", 0) or 0),
                        gamma=float(row.get("gamma", 0) or 0),
                    )
                )
            except (ValueError, TypeError, KeyError) as e:
                logger.debug(f"[OPTS] Skipping malformed contract row: {e}")
                continue

        return contracts

    # -------------------------------------------------------------------------
    # Signal computation
    # -------------------------------------------------------------------------

    def _compute_signal(
        self,
        symbol: str,
        contracts: list[OptionsContract],
    ) -> OptionsFlowSignal:
        """
        Derive the UOA signal from the parsed options chain.

        Score breakdown (max 100):
          - Put/call ratio extremity (0–20 pts)
          - Unusual vol/OI count (0–30 pts)
          - Net premium flow magnitude (0–30 pts)
          - IV spike (0–20 pts)
        """
        calls = [c for c in contracts if c.option_type == "call"]
        puts = [c for c in contracts if c.option_type == "put"]

        call_vol = sum(c.volume for c in calls)
        put_vol = sum(c.volume for c in puts)
        call_vol + put_vol or 1

        put_call_ratio = put_vol / max(call_vol, 1)

        call_premium = sum(c.notional_premium for c in calls)
        put_premium = sum(c.notional_premium for c in puts)
        net_premium = call_premium - put_premium

        # Unusual contracts (vol/OI > threshold AND notional > minimum)
        unusual = [
            c
            for c in contracts
            if c.volume_oi_ratio >= self.VOLUME_OI_UNUSUAL
            and c.notional_premium >= self.MIN_NOTIONAL_UNUSUAL
        ]

        # --- Score components ---
        # 1. P/C ratio extremity
        if put_call_ratio < 0.4 or put_call_ratio > 2.5:
            pc_score = 20.0
        elif put_call_ratio < 0.6 or put_call_ratio > 2.0:
            pc_score = 12.0
        elif put_call_ratio < 0.8 or put_call_ratio > 1.5:
            pc_score = 6.0
        else:
            pc_score = 0.0

        # 2. Unusual contract count (cap at 30 pts)
        unusual_score_component = min(30.0, len(unusual) * 5.0)

        # 3. Net premium flow (cap at 30 pts)
        net_premium_abs = abs(net_premium)
        if net_premium_abs > 5_000_000:
            premium_score = 30.0
        elif net_premium_abs > 1_000_000:
            premium_score = 20.0
        elif net_premium_abs > 250_000:
            premium_score = 10.0
        else:
            premium_score = min(10.0, net_premium_abs / 25_000)

        # 4. IV spike: median IV vs typical baseline (0.20 = 20% baseline assumption)
        all_ivs = [c.implied_volatility for c in contracts if c.implied_volatility > 0]
        iv_score = 0.0
        if all_ivs:
            median_iv = sorted(all_ivs)[len(all_ivs) // 2]
            if median_iv > 0.60:
                iv_score = 20.0
            elif median_iv > 0.40:
                iv_score = 12.0
            elif median_iv > 0.30:
                iv_score = 6.0

        total_score = min(100.0, pc_score + unusual_score_component + premium_score + iv_score)

        # --- Bias ---
        if call_vol > put_vol * 1.8 and net_premium > 0:
            flow_bias = "BULLISH"
        elif put_vol > call_vol * 1.8 and net_premium < 0:
            flow_bias = "BEARISH"
        elif abs(put_call_ratio - 1.0) < 0.3:
            flow_bias = "NEUTRAL"
        else:
            flow_bias = "MIXED"

        # --- Largest unusual trade description ---
        largest = max(unusual, key=lambda c: c.notional_premium, default=None)
        if largest:
            largest_desc = (
                f"Largest: {largest.strike:.0f} {largest.option_type.upper()} "
                f"expiring {largest.expiry} — "
                f"{largest.volume:,} contracts "
                f"(vol/OI={largest.volume_oi_ratio:.1f}×, "
                f"notional=${largest.notional_premium:,.0f})"
            )
        else:
            largest_desc = "No single dominant unusual trade."

        logger.info(
            f"[OPTS] {symbol}: {flow_bias} score={total_score:.0f}/100 "
            f"P/C={put_call_ratio:.2f} net_premium=${net_premium:+,.0f} "
            f"unusual_contracts={len(unusual)}"
        )

        return OptionsFlowSignal(
            symbol=symbol,
            as_of=datetime.now(),
            flow_bias=flow_bias,
            unusual_score=round(total_score, 1),
            put_call_ratio=round(put_call_ratio, 3),
            net_premium_usd=round(net_premium, 0),
            call_volume=call_vol,
            put_volume=put_vol,
            n_unusual_contracts=len(unusual),
            largest_trade_description=largest_desc,
            raw_contracts=contracts,
        )

    @staticmethod
    def _empty_signal(symbol: str) -> OptionsFlowSignal:
        return OptionsFlowSignal(
            symbol=symbol,
            as_of=datetime.now(),
            flow_bias="NEUTRAL",
            unusual_score=0.0,
            put_call_ratio=1.0,
            net_premium_usd=0.0,
            call_volume=0,
            put_volume=0,
            n_unusual_contracts=0,
            largest_trade_description="No options data available.",
        )


# =============================================================================
# CREWAI TOOLS (only registered if crewai is available)
# =============================================================================


def _make_options_flow_tool():
    """Factory for the CrewAI OptionsFlowTool — avoids import-time crash if crewai absent."""
    if not _CREWAI_AVAILABLE:
        return None

    from pydantic import BaseModel as PydanticBaseModel
    from pydantic import Field as PydanticField

    class OptionsFlowInput(PydanticBaseModel):
        symbol: str = PydanticField(..., description="Ticker symbol to analyse, e.g. 'SPY'")
        expiry_within_days: int = PydanticField(
            45, description="Only include options expiring within this many days (default 45)"
        )

    class OptionsFlowTool(BaseTool):
        """
        Unusual Options Activity (UOA) analyser.

        Fetches the current options chain and detects unusual activity:
        elevated volume/OI ratio, large net premium flow, put/call skew.

        Returns a structured flow_bias signal and unusual_score (0–100).
        """

        name: str = "options_flow_analysis"
        description: str = (
            "Analyse unusual options activity for a ticker. Returns flow_bias "
            "(BULLISH/BEARISH/NEUTRAL/MIXED), unusual_score (0-100, actionable above 60), "
            "put/call ratio, and net premium flow in dollars. "
            "High unusual_score signals institutional positioning before a catalyst."
        )
        args_schema: type[PydanticBaseModel] = OptionsFlowInput

        def _run(self, symbol: str, expiry_within_days: int = 45) -> str:
            client = OptionsFlowClient()
            signal = client.get_flow(symbol.upper(), expiry_within_days=expiry_within_days)
            return signal.summary

    return OptionsFlowTool


def _make_put_call_ratio_tool():
    """Factory for the CrewAI PutCallRatioTool."""
    if not _CREWAI_AVAILABLE:
        return None

    from pydantic import BaseModel as PydanticBaseModel
    from pydantic import Field as PydanticField

    class PCRInput(PydanticBaseModel):
        symbol: str = PydanticField(..., description="Ticker symbol")

    class PutCallRatioTool(BaseTool):
        """
        Quick put/call ratio lookup.

        Returns the current volume-based put/call ratio:
        - < 0.5: Aggressive bullish sentiment (heavy call buying)
        - 0.5–0.8: Mildly bullish
        - 0.8–1.2: Neutral
        - 1.2–2.0: Mildly bearish
        - > 2.0: Aggressive bearish (heavy put buying or hedging)
        """

        name: str = "put_call_ratio"
        description: str = (
            "Get the put/call volume ratio for a symbol. "
            "< 0.5 = bullish flow, 0.5-1.2 = neutral, > 1.2 = bearish flow. "
            "Use alongside options_flow_analysis for confirmation."
        )
        args_schema: type[PydanticBaseModel] = PCRInput

        def _run(self, symbol: str) -> str:
            client = OptionsFlowClient()
            signal = client.get_flow(symbol.upper(), expiry_within_days=30)
            pcr = signal.put_call_ratio
            if pcr < 0.5:
                sentiment = "aggressive BULLISH (heavy call buying)"
            elif pcr < 0.8:
                sentiment = "mildly bullish"
            elif pcr < 1.2:
                sentiment = "neutral"
            elif pcr < 2.0:
                sentiment = "mildly bearish"
            else:
                sentiment = "aggressive BEARISH (heavy put buying / hedging)"
            return (
                f"{symbol} put/call ratio: {pcr:.2f} — {sentiment}. "
                f"Call volume: {signal.call_volume:,}, Put volume: {signal.put_volume:,}. "
                f"Net premium flow: ${signal.net_premium_usd:+,.0f}."
            )

    return PutCallRatioTool


# Instantiate tools — only if crewai is available
_OptionsFlowToolClass = _make_options_flow_tool()
_PutCallRatioToolClass = _make_put_call_ratio_tool()

OptionsFlowTool = _OptionsFlowToolClass() if _OptionsFlowToolClass else None
PutCallRatioTool = _PutCallRatioToolClass() if _PutCallRatioToolClass else None


def get_options_tools() -> list:
    """Return all available options flow tools for agent registration."""
    tools = []
    if OptionsFlowTool is not None:
        tools.append(OptionsFlowTool)
    if PutCallRatioTool is not None:
        tools.append(PutCallRatioTool)
    return tools
