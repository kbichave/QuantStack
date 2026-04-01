# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Trading Sheet Generator — per-symbol daily playbook.

Produces a structured trading plan for each symbol with:
- Current regime + HMM state probabilities
- Technical levels (support, resistance, SMA, ATR stops)
- Options positioning (GEX, gamma flip, DEX, IV skew, VRP)
- Event calendar (earnings, FOMC, CPI proximity)
- ML signal (direction probability from trained model)
- Active strategies for current regime
- Specific trade plan with entry/exit/sizing rules

This is what an HRT desk has for every name in the book.

Usage:
    generator = TradingSheetGenerator()
    sheets = await generator.generate_all(["SPY", "QQQ", "IWM", "TSLA", "NVDA"])
    for sheet in sheets:
        print(sheet.to_markdown())
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from loguru import logger

from quantstack.config.timeframes import Timeframe
from quantstack.data.storage import DataStore
from quantstack.signal_engine.engine import SignalEngine


@dataclass
class TradingSheet:
    """Complete trading playbook for a single symbol."""

    symbol: str
    generated_at: datetime = field(default_factory=datetime.now)
    as_of_date: date = field(default_factory=date.today)

    # Regime
    trend_regime: str = "unknown"
    volatility_regime: str = "normal"
    regime_confidence: float = 0.0
    regime_source: str = "unknown"
    hmm_stability: float | None = None
    hmm_expected_duration: float | None = None

    # Technical levels
    close_price: float = 0.0
    sma_20: float | None = None
    sma_50: float | None = None
    sma_200: float | None = None
    rsi_14: float | None = None
    adx_14: float | None = None
    atr_14: float | None = None
    bb_upper: float | None = None
    bb_lower: float | None = None
    macd_hist: float | None = None
    weekly_trend: str = "unknown"

    # Options positioning
    opt_gex: float | None = None
    opt_gamma_flip: float | None = None
    opt_above_gamma_flip: int | None = None
    opt_dex: float | None = None
    opt_max_pain: float | None = None
    opt_iv_skew: float | None = None
    opt_vrp: float | None = None
    opt_call_oi: int | None = None
    opt_put_oi: int | None = None

    # Events
    has_earnings_7d: bool = False
    next_event: str = "none"
    has_fomc_24h: bool = False
    has_macro_event: bool = False

    # Signal
    consensus_bias: str = "neutral"
    consensus_conviction: float = 0.0
    ml_direction: str = "unknown"
    ml_probability: float | None = None
    ml_confidence: float | None = None

    # Strategy
    active_strategies: list[str] = field(default_factory=list)
    recommended_action: str = "HOLD"
    position_size: str = "quarter"
    stop_loss: float | None = None
    take_profit: float | None = None
    trade_rationale: str = ""

    def to_markdown(self) -> str:
        """Generate a readable markdown trading sheet."""
        lines = [
            f"# {self.symbol} — Trading Sheet ({self.as_of_date})",
            "",
            "## Regime",
            f"- Trend: **{self.trend_regime}** (confidence {self.regime_confidence:.0%}, source: {self.regime_source})",
            f"- Volatility: **{self.volatility_regime}**",
        ]
        if self.hmm_stability is not None:
            lines.append(
                f"- HMM stability: {self.hmm_stability:.0%}, expected duration: {self.hmm_expected_duration:.0f} bars"
            )

        lines.extend(
            [
                "",
                "## Technical Levels",
                f"- Close: **${self.close_price:.2f}**",
                f"- RSI(14): {self.rsi_14:.1f}" if self.rsi_14 else "- RSI(14): N/A",
                f"- ADX(14): {self.adx_14:.1f}" if self.adx_14 else "- ADX(14): N/A",
                (
                    f"- MACD hist: {self.macd_hist:+.3f}"
                    if self.macd_hist is not None
                    else "- MACD hist: N/A"
                ),
                f"- ATR(14): ${self.atr_14:.2f}" if self.atr_14 else "- ATR(14): N/A",
                (
                    f"- SMA 20/50/200: ${self.sma_20:.2f} / ${self.sma_50:.2f} / ${self.sma_200:.2f}"
                    if self.sma_200
                    else "- SMAs: N/A"
                ),
                (
                    f"- Bollinger: ${self.bb_lower:.2f} — ${self.bb_upper:.2f}"
                    if self.bb_upper
                    else "- Bollinger: N/A"
                ),
                f"- Weekly trend: {self.weekly_trend}",
            ]
        )

        if self.opt_gex is not None:
            lines.extend(
                [
                    "",
                    "## Options Positioning",
                    f"- GEX: {self.opt_gex:,.0f} ({'mean-reverting' if self.opt_gex > 0 else 'amplifying'})",
                    (
                        f"- Gamma flip: ${self.opt_gamma_flip:.2f} (spot {'above' if self.opt_above_gamma_flip else 'below'})"
                        if self.opt_gamma_flip
                        else "- Gamma flip: N/A"
                    ),
                    f"- DEX: {self.opt_dex:,.0f}" if self.opt_dex else "- DEX: N/A",
                    (
                        f"- Max pain: ${self.opt_max_pain:.2f}"
                        if self.opt_max_pain
                        else "- Max pain: N/A"
                    ),
                    (
                        f"- IV skew: {self.opt_iv_skew:.4f}"
                        if self.opt_iv_skew
                        else "- IV skew: N/A"
                    ),
                    f"- VRP: {self.opt_vrp:.4f}" if self.opt_vrp else "- VRP: N/A",
                    (
                        f"- Put/Call OI: {self.opt_put_oi:,}/{self.opt_call_oi:,}"
                        if self.opt_call_oi
                        else "- Put/Call OI: N/A"
                    ),
                ]
            )

        lines.extend(
            [
                "",
                "## Events",
                f"- Next event: {self.next_event}",
                f"- Earnings within 7d: {'YES — CAUTION' if self.has_earnings_7d else 'No'}",
                f"- FOMC within 24h: {'YES — AVOID NEW POSITIONS' if self.has_fomc_24h else 'No'}",
                f"- Macro event (CPI/NFP): {'YES — REDUCE SIZE' if self.has_macro_event else 'No'}",
            ]
        )

        lines.extend(
            [
                "",
                "## Signal",
                f"- Bias: **{self.consensus_bias}** (conviction {self.consensus_conviction:.0%})",
                (
                    f"- ML: {self.ml_direction} (prob={self.ml_probability:.2f}, conf={self.ml_confidence:.0%})"
                    if self.ml_probability
                    else "- ML: no model trained"
                ),
            ]
        )

        lines.extend(
            [
                "",
                "## Trade Plan",
                f"- Action: **{self.recommended_action}**",
                f"- Size: {self.position_size}",
                (
                    f"- Stop loss: ${self.stop_loss:.2f}"
                    if self.stop_loss
                    else "- Stop loss: N/A"
                ),
                (
                    f"- Take profit: ${self.take_profit:.2f}"
                    if self.take_profit
                    else "- Take profit: N/A"
                ),
                f"- Strategies: {', '.join(self.active_strategies) if self.active_strategies else 'none active'}",
                f"- Rationale: {self.trade_rationale}",
            ]
        )

        return "\n".join(lines)


class TradingSheetGenerator:
    """Generate trading sheets from SignalEngine output."""

    async def generate_all(
        self,
        symbols: list[str],
        as_of_date: date | None = None,
    ) -> list[TradingSheet]:
        """Generate trading sheets for all symbols."""
        sheets = []
        for symbol in symbols:
            try:
                sheet = await self.generate(symbol, as_of_date)
                sheets.append(sheet)
            except Exception as exc:
                logger.warning(f"[TradingSheet] Failed for {symbol}: {exc}")
                sheets.append(TradingSheet(symbol=symbol))
        return sheets

    async def generate(
        self,
        symbol: str,
        as_of_date: date | None = None,
    ) -> TradingSheet:
        """Generate a complete trading sheet for one symbol."""
        sheet = TradingSheet(symbol=symbol, as_of_date=as_of_date or date.today())

        # Run SignalEngine for full analysis
        engine = SignalEngine()
        brief = await engine.run(symbol)

        # Extract from SignalBrief
        sb = brief.symbol_briefs[0] if brief.symbol_briefs else None
        regime = brief.regime_detail or {}

        # Regime
        sheet.trend_regime = regime.get("trend_regime", "unknown")
        sheet.volatility_regime = regime.get("volatility_regime", "normal")
        sheet.regime_confidence = regime.get("confidence", 0.0)
        sheet.regime_source = regime.get("regime_source", "unknown")
        sheet.hmm_stability = regime.get("hmm_stability")
        sheet.hmm_expected_duration = regime.get("hmm_expected_duration")

        # Technical — extract from the brief's observations
        if sb:
            sheet.consensus_bias = sb.consensus_bias
            sheet.consensus_conviction = sb.consensus_conviction

        # Get raw technical data from a fresh collector run
        try:
            store = DataStore(read_only=True)
            df = store.load_ohlcv(symbol, Timeframe.D1)
            if df is not None and len(df) >= 200:
                sheet.close_price = float(df["close"].iloc[-1])
                sheet.sma_20 = float(df["close"].rolling(20).mean().iloc[-1])
                sheet.sma_50 = float(df["close"].rolling(50).mean().iloc[-1])
                sheet.sma_200 = float(df["close"].rolling(200).mean().iloc[-1])

                # RSI
                delta = df["close"].diff()
                gain = delta.clip(lower=0).rolling(14).mean()
                loss = (-delta.clip(upper=0)).rolling(14).mean()
                rs = gain / loss.replace(0, float("nan"))
                sheet.rsi_14 = (
                    float(100 - 100 / (1 + rs.iloc[-1]))
                    if rs.iloc[-1] == rs.iloc[-1]
                    else None
                )

                # ATR
                tr = (
                    (df["high"] - df["low"])
                    .combine((df["high"] - df["close"].shift()).abs(), max)
                    .combine((df["low"] - df["close"].shift()).abs(), max)
                )
                sheet.atr_14 = float(tr.rolling(14).mean().iloc[-1])

                # Stop loss / take profit from ATR
                if sheet.atr_14:
                    sheet.stop_loss = round(sheet.close_price - 2 * sheet.atr_14, 2)
                    sheet.take_profit = round(sheet.close_price + 3 * sheet.atr_14, 2)
        except Exception:
            pass

        # Options
        sheet.opt_gex = brief.opt_gex
        sheet.opt_gamma_flip = brief.opt_gamma_flip
        sheet.opt_above_gamma_flip = brief.opt_above_gamma_flip
        sheet.opt_dex = brief.opt_dex
        sheet.opt_max_pain = brief.opt_max_pain
        sheet.opt_iv_skew = brief.opt_iv_skew
        sheet.opt_vrp = brief.opt_vrp

        # Events
        if sb:
            for obs in sb.risk_factors:
                if "earnings" in obs.lower():
                    sheet.has_earnings_7d = True
                if "fomc" in obs.lower():
                    sheet.has_fomc_24h = True
                if "macro" in obs.lower() or "cpi" in obs.lower():
                    sheet.has_macro_event = True

        # ML
        sheet.ml_direction = brief.ml_direction
        sheet.ml_probability = brief.ml_prediction
        if brief.ml_prediction is not None:
            sheet.ml_confidence = abs(brief.ml_prediction - 0.5) * 2

        # Trade plan
        sheet.recommended_action = self._derive_action(sheet)
        sheet.trade_rationale = self._derive_rationale(sheet)
        sheet.position_size = self._derive_size(sheet)

        return sheet

    def _derive_action(self, sheet: TradingSheet) -> str:
        """Derive recommended action from signals."""
        if sheet.has_fomc_24h or sheet.has_macro_event:
            return "HOLD — macro event"
        if sheet.has_earnings_7d and sheet.consensus_conviction < 0.7:
            return "HOLD — earnings risk"
        if (
            sheet.consensus_bias in ("bullish", "strong_bullish")
            and sheet.consensus_conviction >= 0.5
        ):
            return "BUY"
        if (
            sheet.consensus_bias in ("bearish", "strong_bearish")
            and sheet.consensus_conviction >= 0.5
        ):
            return "SELL"
        return "HOLD"

    def _derive_rationale(self, sheet: TradingSheet) -> str:
        """Build human-readable trade rationale."""
        parts = []
        parts.append(f"Regime {sheet.trend_regime} ({sheet.regime_confidence:.0%})")
        if sheet.rsi_14:
            if sheet.rsi_14 < 35:
                parts.append(f"RSI oversold ({sheet.rsi_14:.1f})")
            elif sheet.rsi_14 > 65:
                parts.append(f"RSI overbought ({sheet.rsi_14:.1f})")
        if sheet.opt_gex is not None:
            parts.append(
                f"GEX {'positive (dampening)' if sheet.opt_gex > 0 else 'negative (amplifying)'}"
            )
        if sheet.ml_direction != "unknown":
            parts.append(f"ML signal: {sheet.ml_direction}")
        return " | ".join(parts)

    def _derive_size(self, sheet: TradingSheet) -> str:
        """Derive position size from conviction + regime."""
        if "HOLD" in sheet.recommended_action:
            return "none"
        if sheet.volatility_regime in ("high", "extreme"):
            return "quarter"
        if sheet.consensus_conviction >= 0.75:
            return "full"
        if sheet.consensus_conviction >= 0.6:
            return "half"
        return "quarter"
