# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Agent reliability hardening.

Addresses three failure modes:
  1. Prompt injection via market data
     → Sanitize all external text before it reaches agent context
     → Only structured (OHLC, computed indicators) goes into prompts
     → Raw news/headlines are embedded as vectors, never as text

  2. Compounding errors across agent chain
     → ICs see only raw market data, never other ICs' interpretations
     → Execution size scales DOWN when ICs disagree (not averaged)
     → Dissent signals are explicitly flagged

  3. Context window exhaustion in long sessions
     → Portfolio state always injected FIRST (critical, cannot be forgotten)
     → Old trades summarized not detailed
     → Regime context compressed to key metrics

Usage:
    hardener = AgentHardener()

    # Sanitize market data text before injecting into prompts
    safe_text = hardener.sanitize_market_text(raw_news_headline)

    # Build agent context with mandatory safety scaffolding
    context = hardener.build_safe_context(
        symbol="SPY",
        portfolio_state=portfolio.as_context_string(),
        market_data={"close": 450.0, "volume": 80_000_000},
        indicators={"rsi_14": 65.2, "adx_14": 35.0},
        regime={"trend": "trending_up", "volatility": "normal"},
    )

    # Measure IC agreement (scale down execution on disagreement)
    agreement = hardener.measure_ic_agreement(ic_outputs)
    position_scale = hardener.position_scale_from_agreement(agreement)
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from loguru import logger


# Tokens that could be used for prompt injection
_INJECTION_PATTERNS = [
    r"ignore (previous|all) instructions",
    r"disregard (previous|all|your) (instructions|guidelines)",
    r"you are now",
    r"new instructions?:",
    r"system prompt:",
    r"<\|im_start\|>",
    r"<\|im_end\|>",
    r"\[INST\]",
    r"<s>",
    r"override",
]

_INJECTION_RE = re.compile(
    "|".join(_INJECTION_PATTERNS), re.IGNORECASE
)


class AgentHardener:
    """
    Utility class for hardening agent context against reliability failures.
    """

    # Maximum characters of raw text allowed in any prompt field
    MAX_TEXT_FIELD_LEN = 300

    # Threshold below which ICs are considered in strong disagreement
    AGREEMENT_SCALE_THRESHOLD = 0.5

    # -------------------------------------------------------------------------
    # 1. Prompt injection prevention
    # -------------------------------------------------------------------------

    def sanitize_market_text(self, text: str) -> str:
        """
        Sanitize free-form text (news headlines, analyst comments, etc.)
        before it is included in any agent prompt.

        - Strips injection patterns
        - Truncates to MAX_TEXT_FIELD_LEN
        - Strips special tokens

        Returns sanitized string, or "[REDACTED]" if injection detected.
        """
        if not text:
            return ""

        # Check for injection attempts
        if _INJECTION_RE.search(text):
            logger.warning(
                f"[HARDENING] Potential prompt injection detected in market text: "
                f"{text[:100]!r} — redacted"
            )
            return "[REDACTED: possible injection attempt]"

        # Strip special LLM tokens
        cleaned = re.sub(r"<\|[^|>]+\|>", "", text)
        cleaned = re.sub(r"\[/?INST\]", "", cleaned)
        cleaned = re.sub(r"</?s>", "", cleaned)

        # Truncate
        if len(cleaned) > self.MAX_TEXT_FIELD_LEN:
            cleaned = cleaned[: self.MAX_TEXT_FIELD_LEN] + "…"

        return cleaned.strip()

    def sanitize_indicators(self, indicators: Dict[str, Any]) -> Dict[str, float]:
        """
        Sanitize a dict of indicator values, keeping only numeric scalars.

        Agents should only receive computed numeric indicators — never raw
        free-form data that could contain injection payloads.
        """
        safe = {}
        for key, val in indicators.items():
            # Only allow numeric values
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                safe[str(key)[:50]] = round(float(val), 6)
            else:
                logger.debug(
                    f"[HARDENING] Dropped non-numeric indicator '{key}': {type(val)}"
                )
        return safe

    # -------------------------------------------------------------------------
    # 2. Safe context builder
    # -------------------------------------------------------------------------

    def build_safe_context(
        self,
        symbol: str,
        portfolio_state_str: str,
        market_data: Dict[str, Any],
        indicators: Dict[str, Any],
        regime: Dict[str, Any],
        recent_trades_summary: str = "",
        session_block: int = 1,
    ) -> str:
        """
        Build a hardened agent context string.

        Portfolio state is ALWAYS first so it cannot be forgotten even if
        the context window fills up from the tail.

        Never includes raw news text — only structured numeric data.
        """
        safe_indicators = self.sanitize_indicators(indicators)
        safe_market = self.sanitize_indicators(
            {k: v for k, v in market_data.items() if isinstance(v, (int, float))}
        )

        lines = [
            # Portfolio state is always first and flagged as immutable
            portfolio_state_str,
            "",
            f"## Analysis Target: {symbol}",
            f"Session Block: {session_block} (context resets every 4 blocks)",
            "",
            "### Market Data",
            "```",
        ]
        for k, v in safe_market.items():
            lines.append(f"{k}: {v}")
        lines += [
            "```",
            "",
            "### Key Indicators",
            "```",
        ]
        for k, v in safe_indicators.items():
            lines.append(f"{k}: {v}")
        lines += [
            "```",
            "",
            "### Market Regime",
            f"- Trend: {regime.get('trend', 'unknown')}",
            f"- Volatility: {regime.get('volatility', 'unknown')}",
            f"- Confidence: {regime.get('confidence', 0.0):.0%}",
        ]

        if recent_trades_summary:
            safe_summary = self.sanitize_market_text(recent_trades_summary)
            lines += [
                "",
                "### Recent Trade History (summarized)",
                safe_summary,
            ]

        lines += [
            "",
            "---",
            "**IMPORTANT:** You are analyzing the symbol above. Do not recommend "
            "a trade that duplicates an existing position shown in the portfolio "
            "state at the top of this context without explicit justification.",
        ]

        return "\n".join(lines)

    # -------------------------------------------------------------------------
    # 3. IC agreement measurement
    # -------------------------------------------------------------------------

    def measure_ic_agreement(
        self, ic_outputs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Measure agreement level across IC outputs.

        Returns:
            {
                agreement_score: 0.0–1.0,
                consensus_bias: "bullish" | "bearish" | "neutral" | "conflicted",
                dissenting_ics: [list of IC names that disagree with consensus],
                scale_factor: 0.25–1.0 (how much to scale position size)
            }
        """
        if not ic_outputs:
            return {
                "agreement_score": 0.0,
                "consensus_bias": "neutral",
                "dissenting_ics": [],
                "scale_factor": 0.0,
            }

        bias_map = {"bullish": 1, "bearish": -1, "neutral": 0}
        biases = []
        names = []

        for ic in ic_outputs:
            bias = str(ic.get("bias", "neutral")).lower()
            score = bias_map.get(bias, 0)
            biases.append(score)
            names.append(ic.get("analyst_name", ic.get("agent_name", "unknown")))

        if not biases:
            return {
                "agreement_score": 0.5,
                "consensus_bias": "neutral",
                "dissenting_ics": [],
                "scale_factor": 0.5,
            }

        avg_bias = sum(biases) / len(biases)
        # Agreement = fraction that agree with the majority direction
        majority_dir = 1 if avg_bias > 0 else (-1 if avg_bias < 0 else 0)

        if majority_dir == 0:
            agreement_score = 0.3  # Neutral consensus = weak signal
        else:
            agreeing = sum(1 for b in biases if b == majority_dir)
            agreement_score = agreeing / len(biases)

        # Identify dissenters
        dissenting = [
            names[i]
            for i, b in enumerate(biases)
            if majority_dir != 0 and b != majority_dir
        ]

        # Scale factor: full size only when all ICs agree
        if agreement_score >= 0.9:
            scale = 1.0
        elif agreement_score >= 0.7:
            scale = 0.75
        elif agreement_score >= 0.5:
            scale = 0.50
        else:
            scale = 0.25  # Strong disagreement → minimal position

        bias_label = (
            "bullish" if avg_bias > 0.3
            else "bearish" if avg_bias < -0.3
            else "conflicted" if len(set(biases)) > 1
            else "neutral"
        )

        result = {
            "agreement_score": round(agreement_score, 3),
            "consensus_bias": bias_label,
            "dissenting_ics": dissenting,
            "scale_factor": scale,
        }

        if dissenting:
            logger.info(
                f"[HARDENING] IC disagreement: {dissenting} dissent from "
                f"{bias_label} consensus (scale={scale:.0%})"
            )

        return result

    # -------------------------------------------------------------------------
    # 4. Context reset helper
    # -------------------------------------------------------------------------

    def should_reset_context(self, session_block: int) -> bool:
        """
        True if the agent context should be reset.

        Context resets every 4 blocks to prevent context window exhaustion.
        Each block = one trading analysis cycle.
        """
        return session_block % 4 == 0

    def summarize_trades_for_context(
        self, trades: List[Dict[str, Any]], max_detailed: int = 3
    ) -> str:
        """
        Compress trade history for context injection.

        Shows last `max_detailed` trades in detail, older trades summarized.
        Prevents context window exhaustion in long trading sessions.
        """
        if not trades:
            return "No recent trades."

        recent = trades[-max_detailed:]
        older = trades[:-max_detailed]

        lines = []

        if older:
            pnls = [t.get("pnl", 0) or 0 for t in older]
            avg_pnl = sum(pnls) / len(pnls) if pnls else 0
            wins = sum(1 for p in pnls if p > 0)
            lines.append(
                f"**Older trades ({len(older)} total):** "
                f"avg P&L ${avg_pnl:+.2f}, {wins}/{len(older)} wins"
            )

        for t in recent:
            symbol = t.get("symbol", "?")
            side = t.get("side", t.get("action", "?")).upper()
            qty = t.get("quantity", 0)
            price = t.get("fill_price", t.get("price", 0))
            pnl = t.get("pnl")
            pnl_str = f" | P&L: ${pnl:+.2f}" if pnl is not None else ""
            lines.append(f"- {side} {qty} {symbol} @ ${price:.2f}{pnl_str}")

        return "\n".join(lines)


    # -------------------------------------------------------------------------
    # 5. Agent output anomaly detection (TradeTrap: direct prompt injection)
    # -------------------------------------------------------------------------

    # Maximum position size fraction any agent is allowed to recommend
    MAX_RECOMMENDED_POSITION_PCT = 0.20  # 20% of equity
    # Minimum confidence score to consider a recommendation actionable
    MIN_ACTIONABLE_CONFIDENCE = 0.10

    def validate_agent_output(
        self,
        output: Dict[str, Any],
        portfolio_equity: float = 100_000.0,
    ) -> Dict[str, Any]:
        """
        Validate agent output before it reaches the execution layer.

        Detects anomalous recommendations that could indicate prompt injection
        or hallucinated outputs (TradeTrap failure modes).

        Checks:
          - position_size_pct is within bounds
          - confidence is in [0, 1]
          - action is a known valid value
          - symbol is a non-empty string without injection markers
          - price targets are numerically plausible

        Returns:
            Dict with keys:
              is_valid (bool)
              violations (list of str)
              sanitized_output (dict — same as output but with anomalous fields zeroed)
        """
        violations = []
        sanitized = dict(output)

        # --- Action ---
        action = str(output.get("action", "")).upper()
        valid_actions = {"BUY", "SELL", "HOLD", "CLOSE", "REDUCE", ""}
        if action and action not in valid_actions:
            violations.append(f"Unknown action '{action}' — expected one of {valid_actions}")
            sanitized["action"] = "HOLD"

        # --- Confidence ---
        conf = output.get("confidence")
        if conf is not None:
            try:
                conf_f = float(conf)
            except (ValueError, TypeError):
                conf_f = 0.0
            if not (0.0 <= conf_f <= 1.0):
                violations.append(f"Confidence {conf_f} outside [0, 1]")
                sanitized["confidence"] = max(0.0, min(1.0, conf_f))

        # --- Position size ---
        pos_pct = output.get("position_size_pct") or output.get("position_pct")
        if pos_pct is not None:
            try:
                pos_pct_f = float(pos_pct)
            except (ValueError, TypeError):
                pos_pct_f = 0.0
            if pos_pct_f > self.MAX_RECOMMENDED_POSITION_PCT:
                violations.append(
                    f"position_size_pct {pos_pct_f:.1%} exceeds max "
                    f"{self.MAX_RECOMMENDED_POSITION_PCT:.0%} — clamped"
                )
                sanitized["position_size_pct"] = self.MAX_RECOMMENDED_POSITION_PCT
            if pos_pct_f < 0:
                violations.append(f"Negative position_size_pct {pos_pct_f}")
                sanitized["position_size_pct"] = 0.0

        # --- Symbol ---
        symbol = output.get("symbol", "")
        if symbol:
            if not isinstance(symbol, str) or len(symbol) > 10:
                violations.append(f"Suspicious symbol value: {symbol!r}")
                sanitized["symbol"] = ""
            elif _INJECTION_RE.search(symbol):
                violations.append(f"Injection pattern in symbol field: {symbol!r}")
                sanitized["symbol"] = "[REDACTED]"

        # --- Price targets ---
        for price_field in ("entry_price", "stop_loss", "take_profit", "target_price"):
            val = output.get(price_field)
            if val is None:
                continue
            try:
                val_f = float(val)
            except (ValueError, TypeError):
                violations.append(f"Non-numeric {price_field}: {val!r}")
                sanitized[price_field] = None
                continue
            if not (0.0 < val_f < 1_000_000):
                violations.append(f"{price_field}={val_f} out of plausible price range")
                sanitized[price_field] = None

        if violations:
            for v in violations:
                logger.warning(f"[HARDENING] Agent output anomaly: {v}")

        return {
            "is_valid": len(violations) == 0,
            "violations": violations,
            "sanitized_output": sanitized,
        }

    # -------------------------------------------------------------------------
    # 6. Blackboard / PortfolioState reconciliation (TradeTrap: memory poisoning)
    # -------------------------------------------------------------------------

    def reconcile_blackboard_with_portfolio(
        self,
        blackboard_positions: List[Dict[str, Any]],
        portfolio_positions: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Detect divergence between agent beliefs (Blackboard) and ground truth
        (PortfolioState) before execution.

        TradeTrap demonstrated that memory poisoning can cause agents to
        hallucinate positions they believe they hold. This check flags:
          - Positions the agent believes it has but portfolio doesn't
          - Positions the portfolio has that the agent doesn't know about
          - Quantity mismatches above a tolerance threshold

        Args:
            blackboard_positions: List of {symbol, quantity, ...} from Blackboard.
            portfolio_positions: List of {symbol, quantity, ...} from PortfolioState
                                 (ground truth).

        Returns:
            {
                is_reconciled: bool,
                phantom_positions: list,     # BB says exists, portfolio doesn't
                unknown_positions: list,     # Portfolio has, BB doesn't know
                quantity_mismatches: list,   # Both have, quantity differs
                should_halt_execution: bool, # True when divergence is material
            }
        """
        # Build lookup dicts by symbol
        bb_map = {str(p.get("symbol", "")).upper(): p for p in blackboard_positions if p.get("symbol")}
        port_map = {str(p.get("symbol", "")).upper(): p for p in portfolio_positions if p.get("symbol")}

        bb_symbols = set(bb_map.keys())
        port_symbols = set(port_map.keys())

        phantom = []      # In BB but not in portfolio
        unknown = []      # In portfolio but not in BB
        mismatches = []   # In both but quantity differs

        for sym in bb_symbols - port_symbols:
            phantom.append({"symbol": sym, "bb_quantity": bb_map[sym].get("quantity", 0)})

        for sym in port_symbols - bb_symbols:
            unknown.append({"symbol": sym, "portfolio_quantity": port_map[sym].get("quantity", 0)})

        for sym in bb_symbols & port_symbols:
            bb_qty = float(bb_map[sym].get("quantity", 0) or 0)
            port_qty = float(port_map[sym].get("quantity", 0) or 0)
            if port_qty != 0 and abs(bb_qty - port_qty) / abs(port_qty) > 0.05:
                mismatches.append({
                    "symbol": sym,
                    "bb_quantity": bb_qty,
                    "portfolio_quantity": port_qty,
                    "pct_diff": abs(bb_qty - port_qty) / abs(port_qty),
                })

        issues = phantom + unknown + mismatches
        is_reconciled = len(issues) == 0
        # Halt if agent believes it has positions that don't exist — classic TradeTrap
        should_halt = len(phantom) > 0

        if phantom:
            logger.error(
                f"[HARDENING] Blackboard/Portfolio DIVERGENCE — phantom positions detected: "
                f"{[p['symbol'] for p in phantom]}. Halting execution until reconciled."
            )
        if mismatches:
            logger.warning(
                f"[HARDENING] Quantity mismatches between Blackboard and PortfolioState: "
                f"{[m['symbol'] for m in mismatches]}"
            )
        if unknown:
            logger.warning(
                f"[HARDENING] Portfolio has positions unknown to Blackboard: "
                f"{[u['symbol'] for u in unknown]}"
            )

        return {
            "is_reconciled": is_reconciled,
            "phantom_positions": phantom,
            "unknown_positions": unknown,
            "quantity_mismatches": mismatches,
            "should_halt_execution": should_halt,
        }


# Singleton
_hardener: Optional[AgentHardener] = None


def get_hardener() -> AgentHardener:
    """Get the singleton AgentHardener instance."""
    global _hardener
    if _hardener is None:
        _hardener = AgentHardener()
    return _hardener
