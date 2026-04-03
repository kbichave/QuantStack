"""Pre-trade slippage budget check.

Before submitting a trade, estimate expected slippage from historical data
and compare it to expected alpha. Rejects trades where execution costs
would consume too much of the expected edge.

Thresholds:
  < 1% of alpha  -> "ok"
  1-2% of alpha  -> "flagged" (proceeds with warning)
  > 2% of alpha  -> "rejected"
"""

from __future__ import annotations

from dataclasses import dataclass

from loguru import logger


_DEFAULT_GLOBAL_SLIPPAGE_BPS = 3.0


@dataclass
class SlippageBudgetResult:
    """Result of slippage budget check."""

    status: str  # "ok", "flagged", "rejected"
    expected_slippage_bps: float
    expected_alpha_bps: float
    slippage_as_pct_of_alpha: float
    reason: str
    source: str  # "historical" or "default"


def check_slippage_budget(
    expected_alpha_bps: float,
    historical_mean_shortfall_bps: float | None = None,
    global_default_bps: float = _DEFAULT_GLOBAL_SLIPPAGE_BPS,
) -> SlippageBudgetResult:
    """Check whether expected slippage would consume too much of expected alpha.

    Args:
        expected_alpha_bps: Expected alpha in basis points.
        historical_mean_shortfall_bps: Mean shortfall from historical TCA data.
            If None, falls back to global_default_bps.
        global_default_bps: Fallback slippage estimate when no history.

    Returns:
        SlippageBudgetResult with status, expected values, and reason.
    """
    if historical_mean_shortfall_bps is not None:
        slippage = historical_mean_shortfall_bps
        source = "historical"
    else:
        slippage = global_default_bps
        source = "default"

    if expected_alpha_bps <= 0:
        return SlippageBudgetResult(
            status="flagged",
            expected_slippage_bps=slippage,
            expected_alpha_bps=expected_alpha_bps,
            slippage_as_pct_of_alpha=0.0,
            reason="Expected alpha is zero or negative",
            source=source,
        )

    pct_of_alpha = slippage / expected_alpha_bps

    if pct_of_alpha > 0.02:
        status = "rejected"
        reason = (
            f"Slippage ({slippage:.1f}bps) is {pct_of_alpha:.1%} of expected alpha "
            f"({expected_alpha_bps:.1f}bps) — exceeds 2% threshold"
        )
    elif pct_of_alpha > 0.01:
        status = "flagged"
        reason = (
            f"Slippage ({slippage:.1f}bps) is {pct_of_alpha:.1%} of expected alpha "
            f"({expected_alpha_bps:.1f}bps) — between 1-2% threshold"
        )
    else:
        status = "ok"
        reason = (
            f"Slippage ({slippage:.1f}bps) is {pct_of_alpha:.1%} of expected alpha "
            f"({expected_alpha_bps:.1f}bps) — within budget"
        )

    return SlippageBudgetResult(
        status=status,
        expected_slippage_bps=slippage,
        expected_alpha_bps=expected_alpha_bps,
        slippage_as_pct_of_alpha=pct_of_alpha,
        reason=reason,
        source=source,
    )
