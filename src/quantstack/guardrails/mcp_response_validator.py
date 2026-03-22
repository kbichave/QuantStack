"""
MCP Response Validator.

Addresses the TradeTrap attack vectors (arxiv:2512.02261) that can corrupt
the agent decision pipeline through malicious or malformed tool responses:

  - Tool hijacking: API response substituted with fabricated signals.
    → Numerical bounds validation on every MCP tool response.
    → Cross-field consistency checks (high >= low, volume >= 0, etc.).

  - Data fabrication: Fake market data injected into the pipeline.
    → Statistical plausibility checks (price change, volume spike).
    → Comparison against last-known-good values when available.

  - State tampering: Corrupted portfolio state passed to agents.
    → Portfolio-level sanity checks (total notional, position counts).

All checks are non-blocking by default (the trade is rejected, not the
process) — validation failures increment a counter, log a structured
warning, and return a ValidationResult with is_valid=False.

Usage::

    validator = get_mcp_validator()

    raw_response = call_mcp_tool("get_quote", {"symbol": "AAPL"})
    result = validator.validate_quote_response(raw_response)

    if not result.is_valid:
        logger.error(f"MCP response invalid: {result.violations}")
        return fallback_value
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from loguru import logger

# ---------------------------------------------------------------------------
# Bounds configuration
# ---------------------------------------------------------------------------

# Price bounds — applies to all equity/ETF symbols
_PRICE_MIN = 0.01
_PRICE_MAX = 100_000.0

# Single-bar price change limit (fraction)
# E.g., 0.25 = 25% max move in one bar — catches injected 10x prices
_MAX_SINGLE_BAR_CHANGE = 0.25

# Volume bounds
_VOLUME_MIN = 0
_VOLUME_MAX = 5_000_000_000  # 5 billion shares — catches overflow / injection

# Ratio sanity (high/low max spread as fraction of close)
_MAX_HL_SPREAD_FRACTION = 0.30  # 30% intraday spread is extreme but possible

# Options bounds
_IV_MIN = 0.01  # 1% IV
_IV_MAX = 20.0  # 2000% IV — very high but possible for deep OTM

# Portfolio bounds
_MAX_POSITION_NOTIONAL = 1_000_000.0  # $1M per position hard ceiling for validation
_MAX_PORTFOLIO_NOTIONAL = 5_000_000.0  # $5M total notional ceiling for validation
_MAX_POSITION_COUNT = 50  # More than 50 open positions is suspicious


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class Violation:
    field: str
    value: Any
    reason: str

    def __str__(self) -> str:
        return f"[{self.field}={self.value}] {self.reason}"


@dataclass
class ValidationResult:
    is_valid: bool
    tool_name: str
    violations: list[Violation] = field(default_factory=list)

    def add(self, field: str, value: Any, reason: str) -> None:
        self.violations.append(Violation(field, value, reason))
        self.is_valid = False

    def log(self) -> None:
        if self.is_valid:
            return
        for v in self.violations:
            logger.warning(f"[MCP_VALIDATOR] {self.tool_name}: {v}")


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


class MCPResponseValidator:
    """
    Validates all MCP tool responses before they reach agent context.

    Maintains per-symbol last-known-good prices to detect suspicious jumps.
    """

    def __init__(self) -> None:
        # symbol -> last validated close price
        self._last_close: dict[str, float] = {}
        self._violation_counts: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Quote / market data
    # ------------------------------------------------------------------

    def validate_quote_response(self, response: dict[str, Any]) -> ValidationResult:
        """Validate a get_quote MCP tool response."""
        result = ValidationResult(is_valid=True, tool_name="get_quote")

        if not isinstance(response, dict):
            result.add(
                "response", type(response).__name__, "Expected dict, got non-dict"
            )
            result.log()
            return result

        symbol = str(response.get("symbol", "UNKNOWN"))

        for price_field in (
            "price",
            "last",
            "bid",
            "ask",
            "close",
            "open",
            "high",
            "low",
        ):
            if price_field not in response:
                continue
            val = response[price_field]
            if not isinstance(val, (int, float)):
                result.add(price_field, val, "Non-numeric price field")
                continue
            if not (_PRICE_MIN <= val <= _PRICE_MAX):
                result.add(
                    price_field,
                    val,
                    f"Price out of bounds [{_PRICE_MIN}, {_PRICE_MAX}]",
                )

        # Cross-field: high >= low
        high = response.get("high")
        low = response.get("low")
        close = response.get("price") or response.get("last") or response.get("close")

        if isinstance(high, (int, float)) and isinstance(low, (int, float)):
            if high < low:
                result.add("high/low", f"{high}/{low}", "high < low — impossible OHLC")
            spread_frac = (high - low) / max(low, 0.01)
            if spread_frac > _MAX_HL_SPREAD_FRACTION:
                result.add(
                    "hl_spread",
                    f"{spread_frac:.1%}",
                    f"Intraday H-L spread > {_MAX_HL_SPREAD_FRACTION:.0%} — suspicious",
                )

        # Temporal continuity: flag if price > 25% different from last known
        if isinstance(close, (int, float)) and close > 0:
            last = self._last_close.get(symbol)
            if last and last > 0:
                change = abs(close - last) / last
                if change > _MAX_SINGLE_BAR_CHANGE:
                    result.add(
                        "close",
                        f"{close} (was {last})",
                        f"Single-bar change {change:.1%} > {_MAX_SINGLE_BAR_CHANGE:.0%} threshold",
                    )
            if result.is_valid:
                self._last_close[symbol] = close  # Only update cache on valid response

        # Volume
        volume = response.get("volume")
        if volume is not None:
            if not isinstance(volume, (int, float)):
                result.add("volume", volume, "Non-numeric volume")
            elif not (_VOLUME_MIN <= volume <= _VOLUME_MAX):
                result.add("volume", volume, f"Volume out of bounds [0, {_VOLUME_MAX}]")

        self._record_result(symbol, result)
        result.log()
        return result

    def validate_ohlcv_response(
        self,
        response: dict[str, Any],
        symbol: str = "UNKNOWN",
    ) -> ValidationResult:
        """Validate an OHLCV bar dict (open, high, low, close, volume)."""
        result = ValidationResult(is_valid=True, tool_name="ohlcv")

        if not isinstance(response, dict):
            result.add("response", type(response).__name__, "Expected dict")
            result.log()
            return result

        required = ("open", "high", "low", "close")
        for field_name in required:
            val = response.get(field_name)
            if val is None:
                result.add(
                    field_name, None, f"Missing required OHLCV field: {field_name}"
                )
                continue
            if not isinstance(val, (int, float)):
                result.add(field_name, val, "Non-numeric price")
                continue
            if not (_PRICE_MIN <= val <= _PRICE_MAX):
                result.add(
                    field_name, val, f"Price out of bounds [{_PRICE_MIN}, {_PRICE_MAX}]"
                )

        o = response.get("open", 0)
        h = response.get("high", 0)
        lo = response.get("low", float("inf"))
        c = response.get("close", 0)

        if all(isinstance(x, (int, float)) for x in (o, h, lo, c)):
            if h < lo:
                result.add("high/low", f"{h}/{lo}", "high < low")
            if h < o:
                result.add("high/open", f"{h}/{o}", "high < open")
            if h < c:
                result.add("high/close", f"{h}/{c}", "high < close")
            if lo > o:
                result.add("low/open", f"{lo}/{o}", "low > open")
            if lo > c:
                result.add("low/close", f"{lo}/{c}", "low > close")

        vol = response.get("volume")
        if vol is not None and not (_VOLUME_MIN <= vol <= _VOLUME_MAX):
            result.add("volume", vol, "Volume out of bounds")

        self._record_result(symbol, result)
        result.log()
        return result

    # ------------------------------------------------------------------
    # Options
    # ------------------------------------------------------------------

    def validate_options_response(self, response: dict[str, Any]) -> ValidationResult:
        """Validate an options pricing MCP response."""
        result = ValidationResult(is_valid=True, tool_name="options")

        if not isinstance(response, dict):
            result.add("response", type(response).__name__, "Expected dict")
            result.log()
            return result

        iv = response.get("implied_volatility") or response.get("iv")
        if iv is not None:
            if not isinstance(iv, (int, float)):
                result.add("iv", iv, "Non-numeric IV")
            elif not (_IV_MIN <= iv <= _IV_MAX):
                result.add(
                    "iv", iv, f"IV out of plausible range [{_IV_MIN}, {_IV_MAX}]"
                )

        for greek in ("delta", "gamma", "theta", "vega", "rho"):
            val = response.get(greek)
            if val is None:
                continue
            if not isinstance(val, (int, float)):
                result.add(greek, val, f"Non-numeric {greek}")
            # Delta must be in [-1, 1]
            elif greek == "delta" and not (-1.0 <= val <= 1.0):
                result.add(greek, val, "Delta outside [-1, 1]")

        option_price = response.get("price") or response.get("option_price")
        if option_price is not None:
            if not isinstance(option_price, (int, float)):
                result.add("option_price", option_price, "Non-numeric")
            elif not (0.0 < option_price <= _PRICE_MAX):
                result.add(
                    "option_price",
                    option_price,
                    f"Option price out of range (0, {_PRICE_MAX}]",
                )

        result.log()
        return result

    # ------------------------------------------------------------------
    # Portfolio / account state
    # ------------------------------------------------------------------

    def validate_portfolio_response(self, response: dict[str, Any]) -> ValidationResult:
        """Validate portfolio/account state from eTrade or PortfolioState."""
        result = ValidationResult(is_valid=True, tool_name="portfolio")

        if not isinstance(response, dict):
            result.add("response", type(response).__name__, "Expected dict")
            result.log()
            return result

        positions = response.get("positions", [])
        if not isinstance(positions, list):
            result.add(
                "positions", type(positions).__name__, "Expected list of positions"
            )
            result.log()
            return result

        if len(positions) > _MAX_POSITION_COUNT:
            result.add(
                "position_count",
                len(positions),
                f"More than {_MAX_POSITION_COUNT} positions — suspicious state",
            )

        total_notional = 0.0
        for pos in positions:
            if not isinstance(pos, dict):
                continue
            qty = pos.get("quantity", 0)
            price = pos.get("current_price") or pos.get("entry_price", 0)
            if isinstance(qty, (int, float)) and isinstance(price, (int, float)):
                notional = abs(qty) * price
                total_notional += notional
                if notional > _MAX_POSITION_NOTIONAL:
                    sym = pos.get("symbol", "?")
                    result.add(
                        f"position_{sym}_notional",
                        notional,
                        f"Single position notional ${notional:,.0f} > ${_MAX_POSITION_NOTIONAL:,.0f}",
                    )

        if total_notional > _MAX_PORTFOLIO_NOTIONAL:
            result.add(
                "total_notional",
                total_notional,
                f"Total portfolio notional ${total_notional:,.0f} > ${_MAX_PORTFOLIO_NOTIONAL:,.0f}",
            )

        result.log()
        return result

    # ------------------------------------------------------------------
    # Generic / catch-all
    # ------------------------------------------------------------------

    def validate_generic_response(
        self,
        response: Any,
        tool_name: str,
    ) -> ValidationResult:
        """
        Minimal validation for tool responses without a dedicated validator.

        Ensures the response is not None and is either a dict, list, or scalar.
        Detects common injection markers in string responses.
        """
        result = ValidationResult(is_valid=True, tool_name=tool_name)

        if response is None:
            result.add(
                "response", None, "Tool returned None — possible tool call failure"
            )
            result.log()
            return result

        # Check for obvious injection markers embedded in string responses
        if isinstance(response, str):
            injection_markers = [
                "ignore previous instructions",
                "disregard your",
                "new instructions:",
                "you are now",
                "<|im_start|>",
            ]
            lowered = response.lower()
            for marker in injection_markers:
                if marker in lowered:
                    result.add(
                        "response_text",
                        response[:100],
                        f"Possible prompt injection marker detected: '{marker}'",
                    )
                    break

        result.log()
        return result

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _record_result(self, symbol: str, result: ValidationResult) -> None:
        if not result.is_valid:
            self._violation_counts[symbol] = self._violation_counts.get(symbol, 0) + 1

    def violation_summary(self) -> dict[str, int]:
        """Return per-symbol violation counts since startup."""
        return dict(self._violation_counts)


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_validator: MCPResponseValidator | None = None


def get_mcp_validator() -> MCPResponseValidator:
    """Return the singleton MCPResponseValidator."""
    global _validator
    if _validator is None:
        _validator = MCPResponseValidator()
    return _validator
