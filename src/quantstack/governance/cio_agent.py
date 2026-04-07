# Copyright 2024 QuantStack Contributors
# SPDX-License-Identifier: Apache-2.0

"""
CIO (Chief Investment Officer) agent — daily mandate generation.

Runs once per day before market open (typically ~09:00 ET via scheduler).
Produces a DailyMandate that governs all trading activity for the session.

Current implementation: returns a reasonable static mandate.
Production implementation: will use Sonnet LLM call with regime data,
portfolio state, macro indicators, and overnight news as context.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime, timezone

from quantstack.governance.mandate import DailyMandate, persist_mandate


async def generate_daily_mandate() -> DailyMandate:
    """Generate the daily mandate for today's trading session.

    # TODO(cio_agent, 2026-Q2): Replace with LLM-based mandate generation.
    # The production version will:
    #   1. Load current regime assessment from regime_snapshots table
    #   2. Load portfolio state and open positions
    #   3. Load macro stress score and overnight news sentiment
    #   4. Call Sonnet with structured output to produce the mandate
    #   5. Validate the mandate against safety constraints
    #   6. Persist and publish MANDATE_ISSUED event
    #
    # For now, returns a reasonable default that permits moderate trading
    # in liquid sectors with conservative position limits.

    Returns:
        DailyMandate with sensible defaults for a normal market day.
    """
    today = date.today().isoformat()

    mandate = DailyMandate(
        mandate_id=f"cio-{uuid.uuid4().hex[:8]}",
        date=today,
        regime_assessment="normal",
        allowed_sectors=["Technology", "Healthcare", "Finance", "Energy", "Consumer"],
        blocked_sectors=[],
        max_new_positions=5,
        max_daily_notional=50_000.0,
        strategy_directives={},  # empty = all strategies active
        risk_overrides={},
        focus_areas=["momentum", "mean_reversion"],
        reasoning=(
            "Placeholder CIO mandate — moderate risk appetite with standard "
            "sector allocation. Replace with LLM-generated mandate."
        ),
        created_at=datetime.now(timezone.utc),
    )

    persist_mandate(mandate)
    return mandate
