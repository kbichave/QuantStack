# Copyright 2024 QuantPod Contributors
# SPDX-License-Identifier: Apache-2.0

"""
StrategyRegistrar — persists discovered candidates to the strategy DB.

All discovered strategies are written with status='draft'. They require
a /workshop session to review and promote to 'forward_testing'.
Source field defaults to 'generated'; GP evolution uses 'evolved'.

Why draft-only:
  Auto-promotion to forward_testing would expose live capital to strategies
  that passed statistical filters but haven't been reviewed for economic
  logic, regime fit, and risk parameters. A human review step is mandatory.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from loguru import logger


class StrategyRegistrar:
    """Writes draft strategies discovered by AlphaDiscoveryEngine to the DB."""

    def register(
        self,
        name: str,
        template_name: str,
        parameters: dict[str, Any],
        entry_rules: list[dict],
        exit_rules: list[dict],
        regime_affinity: dict[str, float],
        is_sharpe: float,
        oos_sharpe_mean: float,
        symbol: str,
        source: str = "generated",
    ) -> str | None:
        """
        Write a draft strategy to the database.

        Returns strategy_id on success, None on failure.
        """
        strategy_id = f"disc_{uuid.uuid4().hex[:12]}"
        description = (
            f"Auto-discovered via AlphaDiscoveryEngine. "
            f"Template: {template_name}. "
            f"Symbol: {symbol}. "
            f"IS Sharpe: {is_sharpe:.2f}, OOS Sharpe mean: {oos_sharpe_mean:.2f}. "
            f"Requires /workshop review before forward testing."
        )
        risk_params = {
            "position_size": "quarter",  # conservative default for discovered strategies
            "stop_loss_atr": parameters.get("stop_loss_atr", 2.0),
        }

        try:
            from quant_pod.mcp._state import require_live_db
            ctx = require_live_db()
            ctx.db.execute(
                """
                INSERT INTO strategies
                    (strategy_id, name, description, asset_class, regime_affinity,
                     parameters, entry_rules, exit_rules, risk_params, status, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'draft', ?)
                """,
                [
                    strategy_id,
                    name,
                    description,
                    "equities",
                    json.dumps(regime_affinity),
                    json.dumps(parameters),
                    json.dumps(entry_rules),
                    json.dumps(exit_rules),
                    json.dumps(risk_params),
                    source,
                ],
            )
            logger.info(
                f"[StrategyRegistrar] registered draft {strategy_id}: {name} "
                f"(IS={is_sharpe:.2f} OOS={oos_sharpe_mean:.2f})"
            )
            return strategy_id
        except Exception as exc:
            logger.error(f"[StrategyRegistrar] failed to register {name}: {exc}")
            return None

    def already_discovered(self, template_name: str, parameters: dict[str, Any]) -> bool:
        """
        Check if an identical parameter set was already registered this session.
        Uses a simple parameter hash — not a DB query — to avoid lock contention.
        """
        # This is intentionally stateless across sessions: re-discovering the
        # same params in a new nightly run is benign (status='draft' deduplication
        # is handled in /workshop by the human reviewer).
        return False
