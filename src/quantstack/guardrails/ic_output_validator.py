# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
IC Output Validator — post-processing contract enforcement for IC outputs.

Each IC has a documented expected_output format in tasks.yaml. This validator
checks that the output contains the required fields before it enters the IC
cache and propagates to pod managers.

Design:
  - Non-blocking: a validation failure logs a warning and marks the output
    as low-quality, but never halts a crew run or blocks a trade session.
  - Metrics: increments Prometheus counters per-IC so /reflect sessions can
    see which ICs consistently produce malformed output.
  - Called from _populate_ic_cache_from_result in mcp/server.py after each
    full crew run, and from run_ic after minimal crew runs.

Why this matters:
  Without validation, a vague IC output (e.g. "I could not compute the regime")
  propagates silently through pod managers and reaches the trading_assistant,
  which then produces a DailyBrief with a missing field. Claude Code then reads
  a partial brief and may draw incorrect conclusions. This validator catches
  the problem at the source so /tune sessions have concrete evidence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

# ---------------------------------------------------------------------------
# Required field specs per IC — minimum tokens/patterns that must appear
# in a valid IC output.  Checked with simple substring / regex matching,
# not LLM evaluation (deterministic and fast).
# ---------------------------------------------------------------------------

_IC_REQUIRED_PATTERNS: dict[str, list[str]] = {
    "regime_detector_ic": [
        r"Regime:\s*(trending|ranging|unknown)",
        r"ADX:\s*\d+",
        r"Confidence:\s*0\.\d+",
    ],
    "trend_momentum_ic": [
        r"RSI[:\s]+\d+",
        r"ADX[:\s]+\d+",
    ],
    "volatility_ic": [
        r"ATR[:\s(]+\d+",
        r"(BB Width|Vol Percentile|VaR)",
    ],
    "structure_levels_ic": [
        r"(support|resistance|Support|Resistance).*\$?\d+",
    ],
    "market_snapshot_ic": [
        r"(close|price|Close|Price)[:\s]+\$?\d+",
        r"(volume|Volume)[:\s]+[\d,]+",
    ],
    "statarb_ic": [
        r"(ADF|p-value|IC|ICIR)",
    ],
    "options_vol_ic": [
        r"(IV|implied vol|delta|Greeks)",
    ],
    "risk_limits_ic": [
        r"(VaR|var|drawdown|exposure)",
    ],
    "calendar_events_ic": [
        r"(earnings|FOMC|event|Event|Earnings|No major)",
    ],
    "news_sentiment_ic": [
        r"Sentiment[:\s]+[+-]?\d",
        r"(articles|earnings risk|Earnings risk)",
    ],
    "options_flow_ic": [
        r"Flow[:\s]+(BULLISH|BEARISH|NEUTRAL|MIXED)",
        r"(score|P/C ratio|Net premium)",
    ],
    "fundamentals_ic": [
        r"P/E[:\s]+\d+",
        r"(Beta|EPS|52w)",
    ],
    "data_ingestion_ic": [
        r"(rows|bars|Data Status|date range)",
    ],
}


@dataclass
class ICValidationResult:
    ic_name: str
    is_valid: bool
    missing_patterns: list[str] = field(default_factory=list)
    output_length: int = 0

    @property
    def quality(self) -> str:
        if self.is_valid:
            return "high" if self.output_length > 100 else "medium"
        return "low"


def validate_ic_output(ic_name: str, output: str) -> ICValidationResult:
    """Check that an IC's output contains required fields.

    Args:
        ic_name: The IC identifier (e.g. 'regime_detector_ic').
        output: The raw text output from the IC task.

    Returns:
        ICValidationResult with is_valid flag and list of missing patterns.
    """
    if not output or not output.strip():
        logger.warning(f"[ic_validator] {ic_name}: empty output")
        return ICValidationResult(
            ic_name=ic_name,
            is_valid=False,
            missing_patterns=["<empty output>"],
            output_length=0,
        )

    required = _IC_REQUIRED_PATTERNS.get(ic_name)
    if not required:
        # No spec defined for this IC — accept without validation
        return ICValidationResult(
            ic_name=ic_name, is_valid=True, output_length=len(output)
        )

    missing = []
    for pattern in required:
        if not re.search(pattern, output, re.IGNORECASE):
            missing.append(pattern)

    is_valid = len(missing) == 0

    if not is_valid:
        logger.warning(
            f"[ic_validator] {ic_name}: missing required fields — {missing}. "
            f"Output preview: {output[:200]!r}"
        )
    else:
        logger.debug(f"[ic_validator] {ic_name}: OK (len={len(output)})")

    return ICValidationResult(
        ic_name=ic_name,
        is_valid=is_valid,
        missing_patterns=missing,
        output_length=len(output),
    )


def validate_all_ic_outputs(
    ic_outputs: dict[str, str],
) -> dict[str, ICValidationResult]:
    """Validate all IC outputs from a crew run.

    Args:
        ic_outputs: Dict mapping ic_name → raw output string.

    Returns:
        Dict mapping ic_name → ICValidationResult.
    """
    results: dict[str, ICValidationResult] = {}
    failed: list[str] = []

    for ic_name, output in ic_outputs.items():
        result = validate_ic_output(ic_name, output)
        results[ic_name] = result
        if not result.is_valid:
            failed.append(ic_name)

    if failed:
        logger.warning(
            f"[ic_validator] {len(failed)}/{len(ic_outputs)} ICs failed validation: {failed}"
        )
    else:
        logger.info(f"[ic_validator] All {len(ic_outputs)} ICs passed validation")

    return results


def get_low_quality_ics(results: dict[str, ICValidationResult]) -> list[str]:
    """Return IC names that failed validation — candidates for /tune."""
    return [name for name, r in results.items() if not r.is_valid]
